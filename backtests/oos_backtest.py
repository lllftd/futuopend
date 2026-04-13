"""
Out-of-sample backtest for the dual-view stack: L1a -> L1b -> L2 -> L3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.trainers.constants import (
    L1A_META_FILE,
    L1A_REGIME_COLS,
    L1B_META_FILE,
    L2_META_FILE,
    L3_META_FILE,
    MODEL_DIR,
)
from core.trainers.data_prep import ensure_breakout_features, ensure_structure_context_features
from core.trainers.layer1a_market import infer_l1a_market_encoder, load_l1a_market_encoder
from core.trainers.layer1b_descriptor import infer_l1b_market_descriptor, load_l1b_market_descriptor
from core.trainers.layer2_decision import infer_l2_trade_decision, load_l2_trade_decision
from core.trainers.layer3_exit import load_l3_exit_manager, load_l3_trajectory_encoder_for_infer
from core.trainers.l3_trajectory_hybrid import L3TrajRollingState, l3_single_trajectory_embedding
from core.trainers.tcn_constants import DEVICE as TORCH_DEVICE

DATA_DIR = _REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("OOS_RESULTS_DIR", str(_REPO_ROOT / "results")))
OOS_START = os.environ.get("OOS_START", "2025-01-01")
OOS_END = os.environ.get("OOS_END", "2026-01-01")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _prepare_symbol_df(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / f"{symbol}.csv")
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw[
        (raw["time_key"] >= pd.Timestamp(OOS_START) - pd.Timedelta(days=10))
        & (raw["time_key"] < pd.Timestamp(OOS_END))
    ].reset_index(drop=True)
    atr_1m = compute_atr(raw, length=14)
    df = add_pa_features(raw, atr_1m, timeframe="5min")
    df = ensure_breakout_features(df)
    df = ensure_structure_context_features(df)
    if "lbl_atr" not in df.columns:
        if "atr_5m" in df.columns:
            df["lbl_atr"] = pd.to_numeric(df["atr_5m"], errors="coerce").fillna(method="ffill").fillna(0.25)
        elif "atr_1m" in df.columns:
            df["lbl_atr"] = pd.to_numeric(df["atr_1m"], errors="coerce").fillna(method="ffill").fillna(0.25)
        else:
            df["lbl_atr"] = (df["high"] - df["low"]).ewm(span=14, min_periods=1).mean().clip(lower=1e-3)
    df["symbol"] = symbol
    return df[df["time_key"] >= pd.Timestamp(OOS_START)].reset_index(drop=True)


def _build_l3_feature_vector(
    idx: int,
    in_pos: int,
    hold: int,
    entry_idx: int,
    entry_price: float,
    entry_atr: float,
    df: pd.DataFrame,
    l1a_out: pd.DataFrame,
    l2_out: pd.DataFrame,
) -> np.ndarray:
    if in_pos == 1:
        live_mfe = max(0.0, (df["high"].iloc[idx] - entry_price) / entry_atr)
        live_mae = max(0.0, (entry_price - df["low"].iloc[idx]) / entry_atr)
        unreal = (df["close"].iloc[idx] - entry_price) / entry_atr
    else:
        live_mfe = max(0.0, (entry_price - df["low"].iloc[idx]) / entry_atr)
        live_mae = max(0.0, (df["high"].iloc[idx] - entry_price) / entry_atr)
        unreal = (entry_price - df["close"].iloc[idx]) / entry_atr
    live_edge = float(live_mfe - live_mae)
    entry_regime = l2_out.loc[entry_idx, [f"l2_entry_regime_{i}" for i in range(6)]].to_numpy(dtype=np.float32)
    current_regime = l1a_out.loc[idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
    p = np.clip(entry_regime, 1e-6, 1.0)
    q = np.clip(current_regime, 1e-6, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    regime_div = float(np.sum(p * (np.log(p) - np.log(q))))
    entry_vol = float(l2_out.loc[entry_idx, "l2_entry_vol"])
    current_vol = float(l1a_out.loc[idx, "l1a_vol_forecast"])
    vol_surprise = float(current_vol / max(entry_vol, 1e-3))
    log_h = float(np.log1p(hold))
    h_sq = float(hold * hold) / 100.0
    h_bkt = float(np.searchsorted(np.array([3, 8, 15, 30, 999], dtype=np.int64), int(hold), side="right"))
    return np.asarray(
        [
            float(l2_out.loc[entry_idx, "l2_decision_confidence"]),
            float(l2_out.loc[entry_idx, "l2_size"]),
            float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
            float(l2_out.loc[entry_idx, "l2_pred_mae"]),
            *entry_regime.tolist(),
            entry_vol,
            *current_regime.tolist(),
            current_vol,
            regime_div,
            vol_surprise,
            float(hold),
            float(unreal),
            float(live_mfe),
            float(live_mae),
            float(live_edge),
            float(in_pos),
            log_h,
            h_sq,
            h_bkt,
        ],
        dtype=np.float32,
    )


def _ensure_backtest_artifacts_exist() -> None:
    required = [
        L1A_META_FILE,
        L1B_META_FILE,
        L2_META_FILE,
        L3_META_FILE,
    ]
    missing = [name for name in required if not (Path(MODEL_DIR) / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing trained stack artifacts under {Path(MODEL_DIR).resolve()}: {missing_list}. "
            "Train the new stack first, e.g. `./scripts/run_train.sh layer1`."
        )


def run_single_symbol(
    symbol: str,
    l1a_model,
    l1a_meta: dict,
    l1b_models: dict,
    l1b_meta: dict,
    l2_models: dict,
    l2_meta: dict,
    l3_models: dict,
    l3_meta: dict,
    l3_traj_enc=None,
    l3_traj_cfg=None,
    torch_device=None,
) -> pd.DataFrame:
    print(f"\n[{symbol}] preparing features...")
    df = _prepare_symbol_df(symbol)
    if df.empty:
        return pd.DataFrame()
    l1a_out = infer_l1a_market_encoder(l1a_model, l1a_meta, df.copy())
    l1b_out = infer_l1b_market_descriptor(l1b_models, l1b_meta, df.copy())
    l2_out = infer_l2_trade_decision(l2_models, l2_meta, df.copy(), l1a_out, l1b_out)

    feature_cols = list(l3_meta["feature_cols"])
    static_l3_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    _sidx = {c: i for i, c in enumerate(static_l3_names)}
    exit_model = l3_models["exit"]
    value_model = l3_models["value"]
    dev = torch_device if torch_device is not None else TORCH_DEVICE
    hybrid = l3_traj_enc is not None and l3_traj_cfg is not None
    traj_buf: L3TrajRollingState | None = None
    max_hold = 30
    trades: list[dict[str, object]] = []

    in_pos = 0
    entry_idx = -1
    entry_price = 0.0
    entry_atr = 1e-3
    entry_time = None
    hold = 0

    for i in range(len(df) - 1):
        if in_pos == 0:
            cls = int(l2_out.loc[i, "l2_decision_class"])
            conf = float(l2_out.loc[i, "l2_decision_confidence"])
            size = float(l2_out.loc[i, "l2_size"])
            if cls in {0, 2} and conf >= 0.55 and size >= 0.12:
                in_pos = 1 if cls == 0 else -1
                entry_idx = i
                entry_price = float(df["open"].iloc[i + 1])
                entry_atr = max(float(df["lbl_atr"].iloc[i]), 1e-3)
                entry_time = df["time_key"].iloc[i + 1]
                hold = 0
                if hybrid:
                    ref = max(int(l3_traj_cfg.max_seq_len), max_hold)
                    traj_buf = L3TrajRollingState(
                        max_seq_len=int(l3_traj_cfg.max_seq_len),
                        max_seq_ref=ref,
                        seq_feat_dim=int(l3_traj_cfg.seq_feat_dim),
                    )
        else:
            hold += 1
            static = _build_l3_feature_vector(i, in_pos, hold, entry_idx, entry_price, entry_atr, df, l1a_out, l2_out).ravel()
            if hybrid and traj_buf is not None:
                close_prev = float(df["close"].iloc[i - 1])
                ts = np.datetime64(df["time_key"].iloc[i])
                traj_buf.append_step(
                    float(static[_sidx["l3_unreal_pnl_atr"]]),
                    int(hold),
                    ts,
                    close_prev,
                    float(df["close"].iloc[i]),
                    float(df["high"].iloc[i]),
                    float(df["low"].iloc[i]),
                    entry_atr,
                    float(static[_sidx["l3_vol_surprise"]]),
                    float(static[_sidx["l3_regime_divergence"]]),
                    float(static[_sidx["l3_live_mfe"]]),
                    float(static[_sidx["l3_live_mae"]]),
                )
                seq, sl = traj_buf.padded_sequence()
                emb = l3_single_trajectory_embedding(l3_traj_enc, seq, sl, dev)
                feat_vec = np.concatenate([static, emb], dtype=np.float32).reshape(1, -1)
            else:
                feat_vec = static.reshape(1, -1)
            if feat_vec.shape[1] != len(feature_cols):
                raise RuntimeError(f"L3 feature width mismatch: {feat_vec.shape[1]} vs {len(feature_cols)}")
            exit_prob = float(exit_model.predict(feat_vec)[0])
            value_left = float(value_model.predict(feat_vec)[0])
            flip = int(l2_out.loc[i, "l2_decision_class"])
            flip_against = (in_pos == 1 and flip == 2) or (in_pos == -1 and flip == 0)
            if exit_prob >= 0.55 and value_left <= 0.02 or flip_against or hold >= max_hold:
                exit_price = float(df["open"].iloc[i + 1])
                ret = (exit_price / entry_price - 1.0) * in_pos
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": df["time_key"].iloc[i + 1],
                        "direction": "LONG" if in_pos == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": ret,
                        "holding_bars": hold,
                        "exit_reason": "Policy_Exit" if exit_prob >= 0.55 and value_left <= 0.02 else ("Signal_Flip" if flip_against else "Max_Hold"),
                    }
                )
                in_pos = 0
                entry_idx = -1
                traj_buf = None
    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("  Running OOS Dual-View Pipeline (L1a -> L1b -> L2 -> L3)")
    print("=" * 70)
    _ensure_backtest_artifacts_exist()
    l1a_model, l1a_meta = load_l1a_market_encoder()
    l1b_models, l1b_meta = load_l1b_market_descriptor()
    l2_models, l2_meta = load_l2_trade_decision()
    _l2_da = l2_meta.get("direction_available", "direction" in l2_meta.get("model_files", {}))
    if not _l2_da and l2_meta.get("decision_mode") == "two_stage":
        print(
            "  [L2] NOTE: direction_available=False — no direction head; "
            "trade uses gate threshold only; long/short probs symmetric at d=0.5 (see l2_meta).",
            flush=True,
        )
    l3_models, l3_meta = load_l3_exit_manager()
    l3_enc, l3_tcfg = load_l3_trajectory_encoder_for_infer(l3_meta)
    if l3_enc is not None:
        l3_enc = l3_enc.to(TORCH_DEVICE)
    all_trades = []
    for sym in ["QQQ", "SPY"]:
        tr_df = run_single_symbol(
            sym,
            l1a_model,
            l1a_meta,
            l1b_models,
            l1b_meta,
            l2_models,
            l2_meta,
            l3_models,
            l3_meta,
            l3_traj_enc=l3_enc,
            l3_traj_cfg=l3_tcfg,
            torch_device=TORCH_DEVICE,
        )
        if not tr_df.empty:
            all_trades.append(tr_df)
            tr_df.to_csv(RESULTS_DIR / f"trades_{sym}.csv", index=False)
            print(f"[{sym}] generated {len(tr_df)} trades.")
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "trades_ALL.csv", index=False)
        print(f"\nTotal trades: {len(combined)}")
        print(f"Win Rate: {(combined['return'] > 0).mean():.2%}")
        print(f"Avg Return: {combined['return'].mean():.4%}")


if __name__ == "__main__":
    main()

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
    L1C_OUTPUT_CACHE_FILE,
    L2_META_FILE,
    L3_META_FILE,
    MODEL_DIR,
    PA_STATE_FEATURES,
)
from core.trainers.data_prep import ensure_breakout_features, ensure_structure_context_features
from core.trainers.l1a import infer_l1a_market_encoder, load_l1a_market_encoder
from core.trainers.l1b import infer_l1b_market_descriptor, load_l1b_market_descriptor
from core.trainers.l2 import infer_l2_trade_decision, load_l2_trade_decision
from core.trainers.stack_v2_common import load_output_cache
from core.trainers.l3 import (
    L3ExitInferenceState,
    L3TrajRollingState,
    l3_entry_policy_params,
    l3_entry_side_from_l2,
    l3_exit_decision_live,
    l3_exit_policy_params,
    l3_infer_cox_features,
    l3_load_cox_bundle,
    load_l3_exit_manager,
    load_l3_trajectory_encoder_for_infer,
    l3_single_trajectory_embedding,
)
from core.trainers.lgbm_utils import _live_trade_state_from_bar, _net_edge_atr_from_state
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
    peak_unreal: float,
    l3_aux: dict,
    l3_meta: dict,
    cox_bundle: dict | None,
) -> tuple[np.ndarray, float]:
    live_mfe, live_mae, unreal = _live_trade_state_from_bar(
        side=float(in_pos),
        entry_price=entry_price,
        atr=entry_atr,
        high_price=float(df["high"].iloc[idx]),
        low_price=float(df["low"].iloc[idx]),
        close_price=float(df["close"].iloc[idx]),
    )
    live_edge = float(_net_edge_atr_from_state(live_mfe, live_mae, hold))
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
    u = float(unreal)
    peak_in = float(peak_unreal)
    peak_unreal = max(peak_in, u)
    drawdown_from_peak = float(peak_unreal - u)
    j_vel = max(entry_idx + 1, idx - 3)
    vel3 = float((df["close"].iloc[idx] - df["close"].iloc[j_vel]) / max(entry_atr, 1e-6))
    j_rd = max(entry_idx + 1, idx - 3)
    q_past = np.clip(l1a_out.loc[j_rd, L1A_REGIME_COLS].to_numpy(dtype=np.float32), 1e-6, 1.0)
    q_past = q_past / q_past.sum()
    reg_div_past = float(np.sum(p * (np.log(p) - np.log(q_past))))
    mom_rd = float(regime_div - reg_div_past)
    if idx >= entry_idx + 2:
        vs0 = float(l1a_out.loc[idx, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs1 = float(l1a_out.loc[idx - 1, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs2 = float(l1a_out.loc[idx - 2, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs_acc = float(vs0 - 2.0 * vs1 + vs2)
    else:
        vs_acc = 0.0
    rids: list[int] = []
    for j in range(max(entry_idx + 1, idx - 2), idx + 1):
        rids.append(int(np.argmax(l1a_out.loc[j, L1A_REGIME_COLS].to_numpy(dtype=np.float32))))
    stab = float(np.mean(np.array(rids) == rids[-1])) if rids else 1.0
    pa_state = (
        df.loc[idx, PA_STATE_FEATURES].to_numpy(dtype=np.float32)
        if all(col in df.columns for col in PA_STATE_FEATURES)
        else np.zeros(len(PA_STATE_FEATURES), dtype=np.float32)
    )
    dec_conf_e = float(l2_out.loc[entry_idx, "l2_decision_confidence"])
    signal_conf_decay = float(l2_out.loc[idx, "l2_decision_confidence"]) - dec_conf_e
    cls_e = int(l2_out.loc[entry_idx, "l2_decision_class"])
    cls_c = int(l2_out.loc[idx, "l2_decision_class"])
    dir_e = 1.0 if cls_e == 0 else (-1.0 if cls_e == 2 else 0.0)
    dir_c = 1.0 if cls_c == 0 else (-1.0 if cls_c == 2 else 0.0)
    signal_direction_agree = float(dir_c == dir_e and dir_e != 0.0)
    rid_e = int(np.argmax(entry_regime.astype(np.float64)))
    regime_changed = float(int(np.argmax(current_regime.astype(np.float64)) != rid_e))
    if "l2_decision_neutral" in l2_out.columns:
        neut_e = float(l2_out.loc[entry_idx, "l2_decision_neutral"])
        neut_c = float(l2_out.loc[idx, "l2_decision_neutral"])
    else:
        neut_e = 1.0 if int(l2_out.loc[entry_idx, "l2_decision_class"]) == 1 else 0.25
        neut_c = 1.0 if int(l2_out.loc[idx, "l2_decision_class"]) == 1 else 0.25
    gate_e = float(1.0 - neut_e)
    gate_curr = float(1.0 - neut_c)
    gate_decay = float(gate_curr - gate_e)
    regime_probs_i = l2_out.loc[idx, [f"l2_entry_regime_{k}" for k in range(len(L1A_REGIME_COLS))]].to_numpy(
        dtype=np.float32
    )
    entry_vol_i = float(l2_out.loc[idx, "l2_entry_vol"])
    pa_row = df.loc[idx, PA_STATE_FEATURES] if all(c in df.columns for c in PA_STATE_FEATURES) else None
    min_c, min_sz, _, _ = l3_entry_policy_params(regime_probs_i, entry_vol_i, l3_meta, pa_state=pa_row)
    cls_i = int(l2_out.loc[idx, "l2_decision_class"])
    conf_i = float(l2_out.loc[idx, "l2_decision_confidence"])
    sz_i = float(l2_out.loc[idx, "l2_size"])
    would_enter_now = float(l3_entry_side_from_l2(cls_i, conf_i, sz_i, min_confidence=min_c, min_size=min_sz) != 0.0)
    if u >= peak_in - 1e-9:
        l3_aux["bars_since_peak"] = 0
    else:
        l3_aux["bars_since_peak"] = int(l3_aux.get("bars_since_peak", 0)) + 1
    bars_since_peak = float(l3_aux["bars_since_peak"])
    at_new_high = float(abs(u - peak_unreal) < 1e-9)
    if peak_unreal > 1e-6:
        regret_ratio = float(max(0.0, (peak_unreal - u) / peak_unreal))
    else:
        regret_ratio = 0.0
    regret_velocity = float(drawdown_from_peak / bars_since_peak) if bars_since_peak > 0.5 else 0.0

    w_fav = float(os.environ.get("L3_BAYES_LLR_FAV", "0.28"))
    w_adv = float(os.environ.get("L3_BAYES_LLR_ADV", "-0.35"))
    w_reg = float(os.environ.get("L3_BAYES_LLR_REGIME", "-0.45"))
    w_gate = float(os.environ.get("L3_BAYES_LLR_GATE", "-0.18"))
    w_gthr = float(os.environ.get("L3_BAYES_GATE_DECAY_THR", "-0.12"))
    prev_u = float(l3_aux.get("prev_unreal", 0.0))
    du = u - prev_u
    l3_aux["prev_unreal"] = u
    sgn = 1.0 if float(in_pos) > 0 else -1.0
    favorable = sgn * du > 0.0
    llr = w_fav if favorable else w_adv
    if regime_changed > 0.5:
        llr += w_reg
    if gate_decay < w_gthr:
        llr += w_gate
    lo = float(l3_aux["log_odds"]) + llr
    l3_aux["log_odds"] = lo
    trade_quality_bayes = float(1.0 / (1.0 + np.exp(-lo)))

    vals: dict[str, float] = {
        "l2_decision_confidence": float(l2_out.loc[entry_idx, "l2_decision_confidence"]),
        "l2_size": float(l2_out.loc[entry_idx, "l2_size"]),
        "l2_pred_mfe": float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
        "l2_pred_mae": float(l2_out.loc[entry_idx, "l2_pred_mae"]),
        **{f"l2_entry_regime_{i}": float(entry_regime[i]) for i in range(len(entry_regime))},
        "l2_entry_vol": float(entry_vol),
        **{c: float(current_regime[j]) for j, c in enumerate(L1A_REGIME_COLS)},
        "l1a_vol_forecast": float(current_vol),
        "l3_regime_divergence": float(regime_div),
        "l3_vol_surprise": float(vol_surprise),
        "l3_hold_bars": float(hold),
        "l3_unreal_pnl_atr": float(unreal),
        "l3_live_mfe": float(live_mfe),
        "l3_live_mae": float(live_mae),
        "l3_live_edge": float(live_edge),
        "l3_side": float(in_pos),
        "l3_log_hold_bars": float(log_h),
        "l3_hold_bars_sq": float(h_sq),
        "l3_hold_bucket": float(h_bkt),
        "l3_drawdown_from_peak_atr": float(drawdown_from_peak),
        "l3_price_velocity_3bar_atr": float(vel3),
        "l3_feature_momentum_regdiv_3bar": float(mom_rd),
        "l3_vol_surprise_accel": float(vs_acc),
        "l3_regime_stability_3bar": float(stab),
        **{PA_STATE_FEATURES[k]: float(pa_state[k]) for k in range(len(PA_STATE_FEATURES))},
        "l3_signal_conf_decay": float(signal_conf_decay),
        "l3_signal_direction_agree": float(signal_direction_agree),
        "l3_regime_changed": float(regime_changed),
        "l3_l2_gate_current": float(gate_curr),
        "l3_l2_gate_decay": float(gate_decay),
        "l3_would_enter_now": float(would_enter_now),
        "l3_regret_ratio": float(regret_ratio),
        "l3_bars_since_peak": float(bars_since_peak),
        "l3_at_new_high": float(at_new_high),
        "l3_regret_velocity": float(regret_velocity),
        "l3_trade_quality_bayes": float(trade_quality_bayes),
    }
    feature_cols = list(l3_meta["feature_cols"])
    static_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    cox_names = {"l3_cox_log_partial_hazard", "l3_cox_baseline_cumhaz_at_stop"}
    static_wo_cox = [c for c in static_names if c not in cox_names]
    feat_base = np.asarray([vals[c] for c in static_wo_cox], dtype=np.float32)
    cox_part = l3_infer_cox_features(cox_bundle, feat_base, static_wo_cox)
    feat = np.concatenate([feat_base, cox_part], dtype=np.float32)
    return feat, peak_unreal


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
    l1c_path = Path(MODEL_DIR) / L1C_OUTPUT_CACHE_FILE
    l1c_out = load_output_cache(L1C_OUTPUT_CACHE_FILE) if l1c_path.exists() else None
    l2_out = infer_l2_trade_decision(l2_models, l2_meta, df.copy(), l1a_out, l1b_out, l1c_out)

    feature_cols = list(l3_meta["feature_cols"])
    static_l3_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    _sidx = {c: i for i, c in enumerate(static_l3_names)}
    exit_model = l3_models["exit"]
    value_model = l3_models.get("value")
    if l3_meta.get("l3_value_mode") == "disabled" or l3_meta.get("l3_value_disabled"):
        value_model = None
    exit_calibrator = l3_models.get("exit_calibrator")
    dev = torch_device if torch_device is not None else TORCH_DEVICE
    hybrid = l3_traj_enc is not None and l3_traj_cfg is not None
    traj_buf: L3TrajRollingState | None = None
    max_hold = int(l3_meta.get("l3_target_horizon_bars", 30))
    trades: list[dict[str, object]] = []
    exit_infer_state = L3ExitInferenceState()
    peak_unreal_atr = float("-inf")
    cox_bundle = l3_load_cox_bundle(l3_meta)
    l3_aux_state: dict[str, float | int] = {}

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
            regime_probs = l2_out.loc[i, [f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32)
            entry_vol = float(l2_out.loc[i, "l2_entry_vol"])
            pa_state = df.loc[i, PA_STATE_FEATURES] if all(col in df.columns for col in PA_STATE_FEATURES) else None
            entry_min_conf, entry_min_size, max_hold, _ = l3_entry_policy_params(regime_probs, entry_vol, l3_meta, pa_state=pa_state)
            side = l3_entry_side_from_l2(
                cls,
                conf,
                size,
                min_confidence=entry_min_conf,
                min_size=entry_min_size,
            )
            if side != 0.0:
                in_pos = int(side)
                entry_idx = i
                entry_price = float(df["open"].iloc[i + 1])
                entry_atr = max(float(df["lbl_atr"].iloc[i]), 1e-3)
                entry_time = df["time_key"].iloc[i + 1]
                hold = 0
                peak_unreal_atr = float("-inf")
                exit_infer_state.reset()
                p0 = float(np.clip(conf, 0.05, 0.95))
                l3_aux_state.clear()
                l3_aux_state["log_odds"] = float(np.log(p0 / (1.0 - p0)))
                l3_aux_state["bars_since_peak"] = 0
                l3_aux_state["prev_unreal"] = 0.0
                if hybrid:
                    ref = max(int(l3_traj_cfg.max_seq_len), max_hold)
                    traj_buf = L3TrajRollingState(
                        max_seq_len=int(l3_traj_cfg.max_seq_len),
                        max_seq_ref=ref,
                        seq_feat_dim=int(l3_traj_cfg.seq_feat_dim),
                    )
        else:
            hold += 1
            static, peak_unreal_atr = _build_l3_feature_vector(
                i,
                in_pos,
                hold,
                entry_idx,
                entry_price,
                entry_atr,
                df,
                l1a_out,
                l2_out,
                peak_unreal_atr,
                l3_aux_state,
                l3_meta,
                cox_bundle,
            )
            static = static.ravel()
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
            exit_raw = float(exit_model.predict(feat_vec)[0])
            if exit_calibrator is not None:
                exit_prob = float(np.clip(exit_calibrator.predict(np.asarray([exit_raw], dtype=np.float64))[0], 0.0, 1.0))
            else:
                exit_prob = float(np.clip(exit_raw, 0.0, 1.0))
            value_left = 0.0 if value_model is None else float(value_model.predict(feat_vec)[0])
            flip = int(l2_out.loc[i, "l2_decision_class"])
            flip_against = (in_pos == 1 and flip == 2) or (in_pos == -1 and flip == 0)
            exit_state_probs = l1a_out.loc[i, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
            exit_state_vol = float(l1a_out.loc[i, "l1a_vol_forecast"])
            pa_state = df.loc[i, PA_STATE_FEATURES] if all(col in df.columns for col in PA_STATE_FEATURES) else None
            exit_prob_threshold, value_left_threshold, state_max_hold, _, value_policy_mode, value_tie_margin = l3_exit_policy_params(
                exit_state_probs,
                exit_state_vol,
                hold,
                l3_meta,
                pa_state=pa_state,
            )
            policy_exit, exit_infer_state = l3_exit_decision_live(
                exit_prob,
                value_left,
                exit_infer_state,
                hold,
                exit_prob_threshold=exit_prob_threshold,
                value_left_threshold=value_left_threshold,
                value_policy_mode=value_policy_mode,
                value_tie_margin=value_tie_margin,
                meta=l3_meta,
            )
            if policy_exit or flip_against or hold >= state_max_hold:
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
                        "exit_reason": "Policy_Exit" if policy_exit else ("Signal_Flip" if flip_against else "Max_Hold"),
                    }
                )
                in_pos = 0
                entry_idx = -1
                traj_buf = None
                peak_unreal_atr = float("-inf")
                l3_aux_state.clear()
                exit_infer_state.reset()
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

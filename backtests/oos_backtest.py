"""
Out-of-sample backtest: TCN → Layer 2a (regime) → Layer 2b (trade stack) → Layer 3 (execution sizer).

Run: ``python run_oos_backtest.py`` from repo root, or ``python -m backtests.oos_backtest``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features

from backtests.pa_pipeline_infer import (
    DATA_DIR,
    OOS_END,
    OOS_PRED_CHUNK,
    OOS_START,
    RESULTS_DIR,
    _chunked_booster_predict,
    _apply_cp_skip,
    _tq,
    load_layered_pa_pipeline,
    materialize_layer3_features_v2,
)
from backtests.train_lgbm_pa_state import (
    BO_FEAT_COLS,
    REGIME_NOW_PROB_COLS,
    TCN_REGIME_FUT_PROB_COLS,
    NUM_REGIME_CLASSES,
    _create_tcn_windows,
    compute_breakout_features,
)


def run_single_symbol(symbol: str, p: dict) -> pd.DataFrame:
    print(f"\n[ {symbol} ] Loading data...")
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw[
        (raw["time_key"] >= pd.Timestamp(OOS_START) - pd.Timedelta(days=30)) & (raw["time_key"] < pd.Timestamp(OOS_END))
    ].reset_index(drop=True)

    atr_1m = compute_atr(raw, length=14)
    df = add_pa_features(raw, atr_1m, timeframe="5min")

    bo_df = compute_breakout_features(df)
    for c in BO_FEAT_COLS:
        df[c] = bo_df[c].values

    tcn_feats = p["tcn_meta"]["feat_cols"]
    x_raw = df[tcn_feats].values.astype(np.float32)
    x_norm = np.nan_to_num((x_raw - p["tcn_meta"]["mean"]) / p["tcn_meta"]["std"], nan=0.0).astype(np.float32)
    windows, end_idx = _create_tcn_windows(x_norm, p["tcn_meta"]["seq_len"])

    n_bars = len(df)
    emb_dim = int(p["tcn_meta"].get("bottleneck_dim", p["tcn_meta"]["num_channels"][-1]))
    embeddings = np.full((n_bars, emb_dim), np.nan, dtype=np.float32)
    regime_probs_arr = np.full((n_bars, len(TCN_REGIME_FUT_PROB_COLS)), np.nan, dtype=np.float32)

    if len(windows) > 0:
        batch_size = max(8, int(os.environ.get("TCN_BATCH_SIZE", "4096")))
        device = p["device"]
        n_batches = (len(windows) + batch_size - 1) // batch_size
        with torch.inference_mode():
            batch_starts = range(0, len(windows), batch_size)
            for i in _tq(batch_starts, desc=f"TCN batches [{symbol}]", unit="batch", total=n_batches):
                xb = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).to(device)
                r_log, emb = p["tcn"].forward_with_embedding(xb)
                sl = slice(i, i + batch_size)
                regime_probs_arr[end_idx[sl]] = torch.softmax(r_log, dim=1).cpu().numpy()
                embeddings[end_idx[sl]] = emb.detach().cpu().numpy()

    embeddings = pd.DataFrame(embeddings).ffill().bfill().values.astype(np.float32)
    regime_probs_arr = pd.DataFrame(regime_probs_arr).ffill().bfill().values.astype(np.float32)

    for j in range(emb_dim):
        df[f"tcn_emb_{j}"] = embeddings[:, j]
    for j, col in enumerate(TCN_REGIME_FUT_PROB_COLS):
        df[col] = regime_probs_arr[:, j]
    eps = 1e-9
    df["tcn_regime_fut_entropy"] = -np.sum(regime_probs_arr * np.log(np.maximum(regime_probs_arr, eps)), axis=1)

    df = df[df["time_key"] >= pd.Timestamp(OOS_START)].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()

    print(f"[{symbol}] Layer 2a: Regime")
    pa_only = [
        c
        for c in p["tq_meta"]["pa_base_feat_cols"]
        + p["tq_meta"]["pa_hmm_feat_cols"]
        + p["tq_meta"]["pa_garch_feat_cols"]
        if c in df.columns
    ]
    l2a_x = df[pa_only].values.astype(np.float32)
    raw_regime = _chunked_booster_predict(
        p["l2a_model"], l2a_x, OOS_PRED_CHUNK, desc=f"L2a regime [{symbol}]",
    )
    cal_regime = np.column_stack([p["l2a_cals"][c].predict(raw_regime[:, c]) for c in range(6)])
    cal_regime /= np.maximum(cal_regime.sum(axis=1, keepdims=True), 1e-12)
    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        df[col] = cal_regime[:, j]
    df["regime_now_conf"] = cal_regime.max(axis=1)

    print(f"[{symbol}] Layer 2b: Trade Stack")
    l2b_feats = [c for c in p["tq_meta"]["feature_cols"] if c in df.columns]
    l2b_x = df[l2b_feats].fillna(0).values.astype(np.float32)
    rp = df[REGIME_NOW_PROB_COLS].values.astype(np.float32)
    
    # Step 1: Binary Trade Gate
    p_trade_raw = _chunked_booster_predict(p["l2b_s1"], l2b_x, OOS_PRED_CHUNK, desc=f"L2b step1 [{symbol}]")
    
    # Apply CP / TCN skip
    thr_cp = p["tq_meta"]["hierarchy_thresholds"].get("thr_cp", 0.0)
    tcn_prob = df["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in df.columns else None
    p_trade, _ = _apply_cp_skip(rp, p_trade_raw, thr_cp, tcn_prob)
    
    p_long = _chunked_booster_predict(p["l2b_s2"], l2b_x, OOS_PRED_CHUNK, desc=f"L2b step2 [{symbol}]")
    p_a = _chunked_booster_predict(p["l2b_s3"], l2b_x, OOS_PRED_CHUNK, desc=f"L2b step3 [{symbol}]")

    print(f"[{symbol}] Layer 3: Execution Sizer")
    df["tq_p_trade"] = p_trade
    df["tq_p_skip"] = 1.0 - p_trade
    df["tq_p_long"] = p_long
    df["tq_p_short"] = 1.0 - p_long
    df["tq_p_a"] = p_a
    df["tq_p_b"] = 1.0 - p_a
    materialize_layer3_features_v2(
        df,
        p,
        l2b_x,
        REGIME_NOW_PROB_COLS,
        OOS_PRED_CHUNK,
        desc=f"L3 L2b triplet [{symbol}]",
    )
    l3_feats = p["l3_meta"]["feature_cols"]
    l3_x = df[l3_feats].fillna(0).values.astype(np.float32)
    if int(p["l3_meta"].get("l3_schema", 1)) >= 2:
        pg = _chunked_booster_predict(
            p["l3_gate"], l3_x, OOS_PRED_CHUNK, desc=f"L3 gate [{symbol}]",
        )
        ps = _chunked_booster_predict(
            p["l3_size"], l3_x, OOS_PRED_CHUNK, desc=f"L3 size [{symbol}]",
        )
        exec_size = np.clip(pg * ps, -1.0, 1.0)
    else:
        exec_size = _chunked_booster_predict(
            p["l3_model"], l3_x, OOS_PRED_CHUNK, desc=f"L3 sizer [{symbol}]",
        )
        exec_size = np.clip(exec_size, -1.5, 1.5)

    df["exec_size"] = exec_size

    trades = []
    in_pos = 0
    entry_price = 0.0
    entry_time = None
    max_hold = 30

    n_sim = max(0, len(df) - 2)
    bar_range = range(1, len(df) - 1)
    for i in _tq(bar_range, desc=f"Simulate [{symbol}]", unit="bar", total=n_sim, mininterval=0.25):
        sz = exec_size[i]
        nxt_open = df["open"].iloc[i + 1]

        if in_pos == 0:
            if sz > 0.6:
                in_pos = 1
                entry_price = float(nxt_open)
                entry_time = df["time_key"].iloc[i + 1]
                hold = 0
            elif sz < -0.6:
                in_pos = -1
                entry_price = float(nxt_open)
                entry_time = df["time_key"].iloc[i + 1]
                hold = 0
        else:
            hold += 1
            exit_signal = False
            if in_pos == 1 and sz < -0.2:
                exit_signal = True
            if in_pos == -1 and sz > 0.2:
                exit_signal = True
            if hold >= max_hold:
                exit_signal = True

            if exit_signal:
                exit_price = float(nxt_open)
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
                    }
                )
                in_pos = 0

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("  Running FULL OOS Pipeline (TCN -> L2a -> L2b -> L3)")
    print("=" * 70)

    p = load_layered_pa_pipeline()
    all_trades = []
    for sym in _tq(["QQQ", "SPY"], desc="OOS symbols", unit="sym"):
        tr_df = run_single_symbol(sym, p)
        if not tr_df.empty:
            all_trades.append(tr_df)
            tr_df.to_csv(os.path.join(RESULTS_DIR, f"trades_{sym}.csv"), index=False)
            print(f"[{sym}] Generated {len(tr_df)} trades.")

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "trades_ALL.csv"), index=False)
        print(f"\nTotal trades: {len(combined)}")
        win_rate = (combined["return"] > 0).mean()
        print(f"Win Rate: {win_rate:.2%}")
        avg_ret = combined["return"].mean()
        print(f"Avg Return: {avg_ret:.4%}")

    print("\n[TCN embeddings]")
    print(
        "Layer 1 passes a learned bottleneck z (tcn_emb_0..K-1, Tanh-bounded) trained with the "
        "6-class future-regime head — see tcn_meta.pkl['bottleneck_dim'] (default K=8). "
        "PCA is no longer used."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Within L1b gate, split mean straddle PnL (ATR units) by L1a argmax regime.

Mirrors the experiment sketched in project notes: signal at bar *i*, entry at *i+1* open,
ATR from bar *i*, symmetric long+short legs with fixed SL, trailing activation, trailing
distance, and time exit.

Example::

  python scripts/diagnose_l1b_regime_straddle_ev.py \\
    --split cal --threshold 0.05 --max-rows 50000

Requires ``lgbm_models/prepared_lgbm_dataset.pkl`` (or ``--pickle``) with OHLC+ATR and
``l1b_edge_pred`` + ``l1a_regime_prob_*`` columns (post-merge stack frame).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.trainers.constants import (
    L1A_REGIME_COLS,
    MODEL_DIR,
    PREPARED_DATASET_CACHE_FILE,
    STATE_NAMES,
)
from core.trainers.stack_v2_common import build_stack_time_splits


def _load_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "df" in obj:
        return obj["df"]
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError(f"Expected pickle dict with 'df' or DataFrame, got {type(obj)}")


def _atr_series(df: pd.DataFrame) -> np.ndarray:
    if "lbl_atr" in df.columns:
        a = pd.to_numeric(df["lbl_atr"], errors="coerce")
    elif "atr" in df.columns:
        a = pd.to_numeric(df["atr"], errors="coerce")
    else:
        raise KeyError("Need lbl_atr or atr in dataframe.")
    return np.asarray(a.ffill().fillna(0.0), dtype=np.float64)


def _straddle_pnl_atr(
    i_sig: int,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    sym: np.ndarray,
    sl: float,
    ta: float,
    td: float,
    max_hold: int,
) -> float:
    """PnL in ATR units (long leg + short leg), signal at i_sig, entry at i_sig+1."""
    n = len(o)
    e = i_sig + 1
    if e >= n:
        return float("nan")
    if sym[i_sig] != sym[e]:
        return float("nan")
    last_t = min(e + max_hold, n) - 1
    if sym[e] != sym[last_t]:
        return float("nan")
    if np.any(sym[e : last_t + 1] != sym[e]):
        return float("nan")

    E = float(o[e])
    A = max(float(atr[i_sig]), 1e-9)

    def long_leg() -> float:
        peak = E
        armed = False
        for t in range(e, min(e + max_hold, n)):
            if sym[t] != sym[e]:
                return float("nan")
            bar_h, bar_l = float(h[t]), float(l[t])
            peak = max(peak, bar_h)
            if peak - E >= ta * A:
                armed = True
            if not armed:
                if bar_l <= E - sl * A:
                    return -sl
            else:
                trail = peak - td * A
                if bar_l <= trail:
                    return (trail - E) / A
        last = min(e + max_hold - 1, n - 1)
        return (float(c[last]) - E) / A

    def short_leg() -> float:
        trough = E
        armed = False
        for t in range(e, min(e + max_hold, n)):
            if sym[t] != sym[e]:
                return float("nan")
            bar_h, bar_l = float(h[t]), float(l[t])
            trough = min(trough, bar_l)
            if E - trough >= ta * A:
                armed = True
            if not armed:
                if bar_h >= E + sl * A:
                    return -sl
            else:
                trail = trough + td * A
                if bar_h >= trail:
                    return (E - trail) / A
        last = min(e + max_hold - 1, n - 1)
        return (E - float(c[last])) / A

    return long_leg() + short_leg()


def main() -> None:
    ap = argparse.ArgumentParser(description="L1b gate × L1a regime straddle mean EV (ATR units).")
    ap.add_argument(
        "--pickle",
        default=os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE),
        help="Prepared dataset pickle (df + feat_cols).",
    )
    ap.add_argument("--split", choices=("all", "train", "cal", "test", "l2_train", "l2_val"), default="cal")
    ap.add_argument("--threshold", type=float, default=float("nan"), help="l1b_edge_pred > threshold (if finite).")
    ap.add_argument(
        "--threshold-quantile",
        type=float,
        default=float("nan"),
        help="If set in (0,1), threshold = quantile of l1b_edge on split rows.",
    )
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--ta", type=float, default=1.0)
    ap.add_argument("--td", type=float, default=0.2)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--max-rows", type=int, default=0, help="Subsample rows (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = _load_df(args.pickle)
    need = {"open", "high", "low", "close", "time_key", "l1b_edge_pred", *L1A_REGIME_COLS}
    miss = [c for c in sorted(need) if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}")

    splits = build_stack_time_splits(df["time_key"])
    tk = pd.to_datetime(df["time_key"])
    if args.split == "all":
        split_mask = np.ones(len(df), dtype=bool)
    elif args.split == "train":
        split_mask = np.asarray(splits.train_mask)
    elif args.split == "cal":
        split_mask = np.asarray(splits.cal_mask)
    elif args.split == "test":
        split_mask = np.asarray(splits.test_mask)
    elif args.split == "l2_train":
        split_mask = np.asarray(splits.l2_train_mask)
    else:
        split_mask = np.asarray(splits.l2_val_mask)

    edge = pd.to_numeric(df["l1b_edge_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    thr = float(args.threshold)
    if np.isfinite(args.threshold_quantile) and 0.0 < args.threshold_quantile < 1.0:
        sub = edge[split_mask]
        thr = float(np.quantile(sub[np.isfinite(sub)], args.threshold_quantile)) if np.any(split_mask) else 0.0
    if not np.isfinite(thr):
        thr = float(np.quantile(edge[split_mask], 0.90)) if np.any(split_mask) else 0.05

    gate = split_mask & (edge > thr)

    R = df[list(L1A_REGIME_COLS)].to_numpy(dtype=np.float64)
    rid = np.argmax(R, axis=1).astype(np.int32)

    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=np.float64)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=np.float64)
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=np.float64)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr = _atr_series(df)
    if "symbol" in df.columns:
        sym = df["symbol"].to_numpy()
    else:
        sym = np.zeros(len(df), dtype=np.int32)

    idx = np.flatnonzero(np.asarray(gate, dtype=bool))
    if args.max_rows > 0 and idx.size > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(idx, size=args.max_rows, replace=False)

    print(
        f"pickle={args.pickle}\n"
        f"split={args.split}  n_split={int(split_mask.sum())}  gate(l1b>{thr:g}) n={idx.size}\n"
        f"straddle params: sl={args.sl} ta={args.ta} td={args.td} max_hold={args.max_hold}\n",
        flush=True,
    )

    rows: list[tuple[int, float, int]] = []
    for i in idx:
        pnl = _straddle_pnl_atr(
            int(i),
            o,
            h,
            l,
            c,
            atr,
            sym,
            args.sl,
            args.ta,
            args.td,
            int(args.max_hold),
        )
        if np.isfinite(pnl):
            rows.append((int(rid[i]), float(pnl), 1))

    if not rows:
        print("No valid straddle paths (check symbol continuity / horizon).", flush=True)
        return

    agg: dict[int, list[float]] = {}
    for r, p, _ in rows:
        agg.setdefault(r, []).append(p)

    print(f"{'regime':<14} {'n':>8} {'mean_EV':>12} {'contrib':>14}")
    print("-" * 52)
    total_n = 0
    total_sum = 0.0
    for k in sorted(agg.keys()):
        name = STATE_NAMES.get(k, f"id{k}")
        arr = np.asarray(agg[k], dtype=np.float64)
        n = arr.size
        ev = float(np.mean(arr))
        contrib = n * ev
        total_n += n
        total_sum += contrib
        print(f"{name:<14} {n:8d} {ev:12.4f} {contrib:14.0f}")
    print("-" * 52)
    overall = total_sum / max(total_n, 1)
    print(f"{'ALL':<14} {total_n:8d} {overall:12.4f} {total_sum:14.0f}")


if __name__ == "__main__":
    main()

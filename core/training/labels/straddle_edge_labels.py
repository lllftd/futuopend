"""
Forward-window straddle-edge targets for L1a (supervision only; features remain causal).

Label at bar t uses realized range over the next N bars vs a simple ATR-based cost proxy.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def straddle_edge_hold_bars() -> int:
    return max(1, int(os.environ.get("L1A_STRADDLE_EDGE_HOLD_BARS", "10")))


def straddle_edge_clip_abs() -> float:
    return float(np.clip(float(os.environ.get("L1A_STRADDLE_EDGE_CLIP_ABS", "0.1")), 1e-4, 1.0))


def compute_straddle_edge_labels(df: pd.DataFrame) -> pd.Series:
    """Returns values in [-1, 1]; NaN tail rows (no future window) -> 0 after fillna."""
    hold = straddle_edge_hold_bars()
    clip_abs = straddle_edge_clip_abs()
    out = pd.Series(0.0, index=df.index, dtype=np.float64)
    if df.empty:
        return out

    def _block(g: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        close = pd.to_numeric(g["close"], errors="coerce").clip(lower=1e-9)
        future_high = high.rolling(hold, min_periods=1).max().shift(-hold)
        future_low = low.rolling(hold, min_periods=1).min().shift(-hold)
        max_move = (future_high - future_low) / close
        if "lbl_atr" in g.columns:
            atr = pd.to_numeric(g["lbl_atr"], errors="coerce").ffill().bfill()
        else:
            atr = (high - low).rolling(20, min_periods=5).mean()
        atr_cost = (atr / close) * 2.0
        edge = max_move - atr_cost
        return (edge.clip(-clip_abs, clip_abs) / clip_abs).fillna(0.0).astype(np.float64)

    if "symbol" in df.columns:
        for _, g in df.groupby("symbol", sort=False):
            out.loc[g.index] = _block(g).to_numpy(dtype=np.float64, copy=False)
    else:
        out.iloc[:] = _block(df).to_numpy(dtype=np.float64, copy=False)
    return out

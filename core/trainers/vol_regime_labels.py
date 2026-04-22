"""
Causal (online-safe) 5-class volatility lifecycle labels for L1a (fixed contract; not optional).

No positive time shifts in the label definition: all inputs use data available at bar t.
Per-symbol rolling statistics only (caller passes single-symbol frames or full df grouped by symbol).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def vol_regime_lookback() -> int:
    return max(5, int(os.environ.get("L1A_VOL_REG_LOOKBACK", "20")))


def vol_regime_rv_window() -> int:
    return max(20, int(os.environ.get("L1A_VOL_REG_RV_WIN", "60")))


def _compute_block_labels(
    *,
    rv: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    rv_win: int,
) -> np.ndarray:
    rv_s = pd.Series(np.nan_to_num(rv, nan=np.nan), dtype=np.float64)
    rv_s = rv_s.ffill().bfill().fillna(0.0)
    rv_ma = rv_s.rolling(rv_win, min_periods=max(10, rv_win // 6)).mean()
    rv_std = rv_s.rolling(rv_win, min_periods=max(10, rv_win // 6)).std().replace(0.0, np.nan)
    rv_z = ((rv_s - rv_ma) / (rv_std + 1e-8)).to_numpy(dtype=np.float64)
    rv_z = np.nan_to_num(rv_z, nan=0.0, posinf=0.0, neginf=0.0)

    rv_arr = rv_s.to_numpy(dtype=np.float64)
    rv_lkb = np.roll(rv_arr, lookback)
    rv_lkb[:lookback] = np.nan
    rv_slope = (rv_arr - rv_lkb) / (float(lookback) * (np.abs(rv_lkb) + 1e-8))
    rv_slope = np.nan_to_num(rv_slope, nan=0.0, posinf=0.0, neginf=0.0)
    rv_accel = pd.Series(rv_slope).diff(5).to_numpy(dtype=np.float64)
    rv_accel = np.nan_to_num(rv_accel, nan=0.0)

    h = pd.Series(high, dtype=np.float64)
    l = pd.Series(low, dtype=np.float64)
    c = pd.Series(close, dtype=np.float64).clip(lower=1e-9)
    bb_w = (h.rolling(20, min_periods=5).max() - l.rolling(20, min_periods=5).min()) / c
    bb_w = bb_w.to_numpy(dtype=np.float64)
    bb_lkb = np.roll(bb_w, lookback)
    bb_lkb[:lookback] = np.nan
    bb_chg = (bb_w - bb_lkb) / (np.abs(bb_lkb) + 1e-8)
    bb_chg = np.nan_to_num(bb_chg, nan=0.0)

    z_lkb = np.roll(rv_z, lookback)
    z_lkb[:lookback] = np.nan

    slope_thr = float(os.environ.get("L1A_VOL_REG_SLOPE_THR", "0.02"))
    accel_thr = float(os.environ.get("L1A_VOL_REG_ACCEL_THR", "0.0"))
    trending_z = float(os.environ.get("L1A_VOL_REG_TRENDING_Z", "0.5"))
    compress_z = float(os.environ.get("L1A_VOL_REG_COMPRESS_Z", "-0.3"))
    bb_drop = float(os.environ.get("L1A_VOL_REG_BB_DROP", "-0.05"))
    mr_z = float(os.environ.get("L1A_VOL_REG_MR_Z", "0.5"))

    breakout = (
        (rv_slope > slope_thr)
        & (rv_accel > accel_thr)
        & (z_lkb < 0.0)
        & np.isfinite(z_lkb)
    )
    exhaust = (rv_z > 0.0) & (rv_slope < 0.0) & (rv_accel < 0.0)
    trending = (rv_z > trending_z) & (rv_slope >= 0.0)
    compress = (rv_z < compress_z) & (rv_slope < 0.0) & (bb_chg < bb_drop)
    mean_revert = (np.abs(rv_z) < mr_z) & (rv_slope < 0.0) & (z_lkb > mr_z) & np.isfinite(z_lkb)

    # Priority: breakout > exhaust > trending > compress > mean_revert > default mean_revert bucket
    return np.select(
        [breakout, exhaust, trending, compress, mean_revert],
        [1, 3, 2, 0, 4],
        default=4,
    ).astype(np.int64)


def compute_vol_regime_labels(df: pd.DataFrame) -> pd.Series:
    """
    Returns int64 Series aligned to df.index, classes 0..4:
      0 vol_compress, 1 vol_breakout, 2 vol_trending, 3 vol_exhaust, 4 vol_mean_revert
    """
    lookback = vol_regime_lookback()
    rv_win = vol_regime_rv_window()
    out = pd.Series(index=df.index, dtype=np.int64)
    if df.empty:
        return out

    if "symbol" in df.columns:
        for _, g in df.groupby("symbol", sort=False):
            idx = g.index
            if "pa_rv_gk_20" in g.columns:
                rv_g = pd.to_numeric(g["pa_rv_gk_20"], errors="coerce").to_numpy(dtype=np.float64)
            else:
                cg = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=np.float64)
                hg = pd.to_numeric(g["high"], errors="coerce").to_numpy(dtype=np.float64)
                lg = pd.to_numeric(g["low"], errors="coerce").to_numpy(dtype=np.float64)
                rv_g = np.nan_to_num((hg - lg) / np.clip(cg, 1e-9, np.inf), nan=0.0)
            h_g = pd.to_numeric(g["high"], errors="coerce").to_numpy(dtype=np.float64)
            l_g = pd.to_numeric(g["low"], errors="coerce").to_numpy(dtype=np.float64)
            c_g = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=np.float64)
            lab = _compute_block_labels(rv=rv_g, high=h_g, low=l_g, close=c_g, lookback=lookback, rv_win=rv_win)
            out.loc[idx] = lab
    else:
        hi = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        cl = pd.to_numeric(df["close"], errors="coerce")
        if "pa_rv_gk_20" in df.columns:
            rv = pd.to_numeric(df["pa_rv_gk_20"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            rng = (hi - lo).to_numpy(dtype=np.float64) / np.clip(cl.to_numpy(dtype=np.float64), 1e-9, np.inf)
            rv = np.nan_to_num(rng, nan=0.0, posinf=0.0, neginf=0.0)
        h = hi.to_numpy(dtype=np.float64)
        l = lo.to_numpy(dtype=np.float64)
        c = cl.to_numpy(dtype=np.float64)
        out.iloc[:] = _compute_block_labels(rv=rv, high=h, low=l, close=c, lookback=lookback, rv_win=rv_win)
    return out

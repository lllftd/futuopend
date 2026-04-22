from __future__ import annotations

import numpy as np
import pandas as pd


def _annualize_from_std(std_series: pd.Series) -> pd.Series:
    return std_series * np.sqrt(252.0 * 390.0)


def build_straddle_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "time_key",
) -> pd.DataFrame:
    """Build 1-minute straddle-oriented features from OHLCV.

    The input frame is expected to contain:
    `open`, `high`, `low`, `close`, `volume`, and a timestamp column.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    out[timestamp_col] = ts
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0)
    log_ret_1 = np.log(close / close.shift(1))

    for window in (5, 15, 30, 60, 120, 390):
        out[f"rv_{window}"] = _annualize_from_std(log_ret_1.rolling(window, min_periods=max(3, window // 3)).std())

    hl_var = (np.log(high / np.maximum(low, 1e-12))) ** 2 / (4.0 * np.log(2.0))
    for window in (15, 30, 60, 390):
        out[f"parkinson_vol_{window}"] = np.sqrt(
            hl_var.rolling(window, min_periods=max(5, window // 3)).mean().clip(lower=0.0)
        ) * np.sqrt(252.0 * 390.0)

    u = np.log(high / np.maximum(open_, 1e-12))
    d = np.log(np.maximum(low, 1e-12) / np.maximum(open_, 1e-12))
    c = np.log(close / np.maximum(open_, 1e-12))
    gk = 0.5 * (u - d) ** 2 - (2.0 * np.log(2.0) - 1.0) * c**2
    for window in (30, 60, 390):
        out[f"gk_vol_{window}"] = np.sqrt(
            gk.rolling(window, min_periods=max(5, window // 3)).mean().clip(lower=0.0)
        ) * np.sqrt(252.0 * 390.0)

    out["rv_acceleration"] = out["rv_30"] - out["rv_30"].shift(30)
    out["vol_of_vol"] = out["rv_30"].rolling(60, min_periods=10).std()

    for window in (5, 15, 30, 60):
        out[f"abs_return_{window}"] = (close / close.shift(window) - 1.0).abs()

    out["intraday_range"] = high.rolling(390, min_periods=30).max() - low.rolling(390, min_periods=30).min()
    out["range_vs_close"] = out["intraday_range"] / close.replace(0.0, np.nan)

    vol_mean = volume.rolling(390, min_periods=20).mean()
    vol_std = volume.rolling(390, min_periods=20).std()
    out["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    out["volume_spike"] = (out["volume_zscore"] > 2.0).astype(np.float32)

    net_move = (close - close.shift(30)).abs()
    total_move = (close - close.shift(1)).abs().rolling(30, min_periods=5).sum()
    out["efficiency_ratio"] = net_move / total_move.replace(0.0, np.nan)

    mins = ts.dt.hour * 60 + ts.dt.minute
    is_open = mins == (9 * 60 + 30)
    out["is_open"] = is_open.astype(np.float32)
    prev_close = close.shift(1)
    out["gap"] = np.where(is_open, (open_ - prev_close) / prev_close.replace(0.0, np.nan), 0.0)

    out["minute_of_day"] = mins.astype(np.float32)
    out["day_of_week"] = ts.dt.dayofweek.astype(np.float32)

    for col in out.columns:
        if col.startswith(("rv_", "parkinson_vol_", "gk_vol_", "abs_return_")) or col in {
            "rv_acceleration",
            "vol_of_vol",
            "intraday_range",
            "range_vs_close",
            "volume_zscore",
            "volume_spike",
            "efficiency_ratio",
            "gap",
            "minute_of_day",
            "day_of_week",
            "is_open",
        }:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

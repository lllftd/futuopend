"""
VIXY 1-min → straddle 波动特征（无方向）
对齐键：time_key（与主表一致）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Default: repo root data/ (train scripts cwd = project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIXY_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "VIXY.csv")


def load_vixy_1m(path: str | None = None) -> pd.DataFrame:
    """Read Futu-export VIXY 1-minute CSV."""
    p = path or VIXY_CSV_PATH
    df = pd.read_csv(p, parse_dates=["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    df = df.rename(
        columns={
            "close": "vixy_close",
            "high": "vixy_high",
            "low": "vixy_low",
            "volume": "vixy_volume",
        }
    )
    df = df.drop_duplicates(subset="time_key", keep="last")
    return df[["time_key", "vixy_close", "vixy_high", "vixy_low", "vixy_volume"]]


def build_vixy_straddle_features(vixy_df: pd.DataFrame) -> pd.DataFrame:
    """
    纯波动/水平/加速特征，不含方向。
    输入: load_vixy_1m 的返回
    输出: DataFrame，index 与输入一致，含 time_key + 25 列特征
    """
    c = vixy_df["vixy_close"].values.astype(float)
    h = vixy_df["vixy_high"].values.astype(float)
    l = vixy_df["vixy_low"].values.astype(float)
    v = vixy_df["vixy_volume"].values.astype(float)

    s = pd.Series(c, index=vixy_df.index)
    sh = pd.Series(h, index=vixy_df.index)
    sl = pd.Series(l, index=vixy_df.index)
    sv = pd.Series(v, index=vixy_df.index)
    ret = s.pct_change()

    feats = pd.DataFrame(index=vixy_df.index)
    feats["time_key"] = vixy_df["time_key"].values

    ma60 = s.rolling(60, min_periods=20).mean()
    ma390 = s.rolling(390, min_periods=60).mean()
    ma1950 = s.rolling(1950, min_periods=390).mean()
    std390 = s.rolling(390, min_periods=60).std()

    feats["vixy_level_ma60_ratio"] = s / (ma60 + 1e-8)
    feats["vixy_level_ma390_ratio"] = s / (ma390 + 1e-8)
    feats["vixy_level_ma1950_ratio"] = s / (ma1950 + 1e-8)
    feats["vixy_zscore_390"] = (s - ma390) / (std390 + 1e-8)
    feats["vixy_term_structure_slope"] = feats["vixy_level_ma60_ratio"] - feats["vixy_level_ma390_ratio"]

    for w in [5, 15, 30, 60]:
        feats[f"vixy_abs_ret_{w}"] = s.pct_change(w).abs()

    abs5 = feats["vixy_abs_ret_5"]
    abs15 = feats["vixy_abs_ret_15"]
    feats["vixy_accel_5"] = abs5 - abs5.shift(5)
    feats["vixy_accel_15"] = abs15 - abs15.shift(15)
    a5 = feats["vixy_accel_5"]
    a15 = feats["vixy_accel_15"]
    feats["vixy_accel_5_over_15"] = a5 / (a15.abs() + 1e-8)
    feats["vixy_accel_burst_minus_drift"] = a5 - a15
    feats["vixy_acceleration_2nd"] = a5 - a5.shift(5)

    feats["vixy_rvol_60"] = ret.rolling(60, min_periods=20).std() * np.sqrt(390)
    feats["vixy_rvol_390"] = ret.rolling(390, min_periods=60).std() * np.sqrt(390)
    feats["vixy_rvol_ratio"] = feats["vixy_rvol_60"] / (feats["vixy_rvol_390"] + 1e-8)
    implied_proxy = feats["vixy_abs_ret_30"] * np.sqrt(390.0 / 30.0)
    feats["vixy_realized_vs_implied_gap"] = feats["vixy_rvol_60"] - implied_proxy

    rh = sh.rolling(390, min_periods=60).max()
    rl = sl.rolling(390, min_periods=60).min()
    feats["vixy_intraday_range"] = (rh - rl) / (s + 1e-8)
    feats["vixy_intraday_rank"] = (s - rl) / (rh - rl + 1e-8)
    intraday_med = feats["vixy_intraday_range"].rolling(390, min_periods=60).median()
    feats["vixy_intraday_range_ratio"] = feats["vixy_intraday_range"] / (intraday_med + 1e-8)

    vol_ma = sv.rolling(390, min_periods=60).mean()
    feats["vixy_vol_surge"] = sv / (vol_ma + 1)
    feats["vixy_vol_surge_5"] = sv.rolling(5).mean() / (vol_ma + 1)

    feats["vixy_pct_rank_5d"] = s.rolling(1950, min_periods=390).rank(pct=True)
    z = feats["vixy_zscore_390"].fillna(0.0).to_numpy(dtype=float)
    regime_id = np.where(z > 1.0, 2, np.where(z < -1.0, 0, 1)).astype(np.int32)
    duration = np.ones(len(regime_id), dtype=np.float64)
    for i in range(1, len(regime_id)):
        duration[i] = duration[i - 1] + 1.0 if regime_id[i] == regime_id[i - 1] else 1.0
    feats["vixy_regime_duration"] = duration

    return feats


VIXY_FEATURE_COLS = [
    "vixy_level_ma60_ratio",
    "vixy_level_ma390_ratio",
    "vixy_level_ma1950_ratio",
    "vixy_zscore_390",
    "vixy_term_structure_slope",
    "vixy_abs_ret_5",
    "vixy_abs_ret_15",
    "vixy_abs_ret_30",
    "vixy_abs_ret_60",
    "vixy_accel_5",
    "vixy_accel_15",
    "vixy_accel_5_over_15",
    "vixy_accel_burst_minus_drift",
    "vixy_acceleration_2nd",
    "vixy_rvol_60",
    "vixy_rvol_390",
    "vixy_rvol_ratio",
    "vixy_realized_vs_implied_gap",
    "vixy_intraday_range",
    "vixy_intraday_rank",
    "vixy_intraday_range_ratio",
    "vixy_vol_surge",
    "vixy_vol_surge_5",
    "vixy_pct_rank_5d",
    "vixy_regime_duration",
]

_VIXY_CACHE: dict[str, pd.DataFrame] = {}


def get_vixy_feature_table(path: str | None = None) -> pd.DataFrame:
    """Return DataFrame indexed by time_key with VIXY feature columns; cached per path."""
    key = os.path.abspath(path or VIXY_CSV_PATH)
    if key not in _VIXY_CACHE:
        raw = load_vixy_1m(key)
        feats = build_vixy_straddle_features(raw)
        feats = feats.set_index("time_key")
        _VIXY_CACHE[key] = feats[VIXY_FEATURE_COLS]
        print(
            f"[vixy] built {len(_VIXY_CACHE[key])} rows × {len(VIXY_FEATURE_COLS)} features",
            flush=True,
        )
    return _VIXY_CACHE[key]


def attach_vixy_features_to_l2_merged(merged: pd.DataFrame, path: str | None = None) -> list[str]:
    """
    Align VIXY features to merged L2 frame rows by time_key (broadcast across symbols).
    Mutates merged in place. Returns list of column names attached (empty on skip/failure).
    """
    if merged.empty or "time_key" not in merged.columns:
        return []
    if all(c in merged.columns for c in VIXY_FEATURE_COLS):
        return list(VIXY_FEATURE_COLS)

    try:
        vixy_table = get_vixy_feature_table(path)
    except OSError as e:
        print(f"[L2] VIXY features skipped (file): {e}", flush=True)
        return []
    except Exception as e:
        print(f"[L2] VIXY features skipped: {e}", flush=True)
        return []

    tk = pd.to_datetime(merged["time_key"], errors="coerce")
    vixy_aligned = vixy_table.reindex(tk.values)
    vixy_aligned = vixy_aligned.ffill(limit=5)
    for c in VIXY_FEATURE_COLS:
        merged[c] = vixy_aligned[c].values
    return list(VIXY_FEATURE_COLS)

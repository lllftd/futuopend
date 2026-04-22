"""
Price Action (PA) — unified causal rules (single module).

Public API: :func:`add_pa_features`
(alias: ``add_all_pa_features``).
All features are computed with the causal / no–look-ahead pipeline.
"""


from __future__ import annotations

import os
import sys
from datetime import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.pa_numba_kernels import (
    causal_support_resistance_fast,
    fill_causal_stops_fast,
    mag_bar_fast,
    pullback_counting_fast,
)


# ---------------------------------------------------------------------------
# Internal 5-min resampler
# ---------------------------------------------------------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-min OHLCV to a higher timeframe (*rule*: \"5min\", \"15min\", \"60min\", ...)."""
    tmp = df.set_index("time_key").copy()
    ohlcv = tmp[["open", "high", "low", "close", "volume"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    return ohlcv.reset_index()


def _resample_ohlcv_htf_closed(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-min OHLCV to HTF bars with timestamp = **right edge** of each bucket
    (label=right, closed=left). A row at time T only includes data from [T-Δ, T),
    so it is *complete* at clock T and must not be used for intrabar 1m timestamps < T.
    """
    tmp = df.set_index("time_key").copy()
    ohlcv = tmp[["open", "high", "low", "close", "volume"]].resample(
        rule, label="right", closed="left"
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    ohlcv = ohlcv.dropna(subset=["open"])
    return ohlcv.reset_index()


def _drop_incomplete_htf_rows(tmp: pd.DataFrame, last_ts: pd.Timestamp) -> pd.DataFrame:
    """Remove the last HTF row if its period end (*time_key*) is after *last_ts* (still forming)."""
    if tmp.empty:
        return tmp
    right_edge = pd.to_datetime(tmp["time_key"].iloc[-1])
    if pd.to_datetime(last_ts) < right_edge:
        return tmp.iloc[:-1].copy()
    return tmp


def _map_htf_closed_to_1min(
    htf: pd.DataFrame,
    times_1m: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    """
    Causal HTF → 1m: each 1m bar at *t* gets the last **fully closed** HTF row with
    ``htf.time_key <= t`` (no lookahead into the in-progress HTF bucket).
    """
    left = pd.DataFrame({"time_key": pd.to_datetime(times_1m)})
    right = htf[["time_key"] + columns].drop_duplicates(subset=["time_key"]).sort_values("time_key").copy()
    lt = left["time_key"]
    rt = pd.to_datetime(right["time_key"])
    l_tz = getattr(lt.dt, "tz", None)
    r_tz = getattr(rt.dt, "tz", None)
    if l_tz is not None and r_tz is None:
        rt = rt.dt.tz_localize(l_tz, ambiguous="infer", nonexistent="shift_forward")
    elif l_tz is None and r_tz is not None:
        lt = lt.dt.tz_localize(r_tz, ambiguous="infer", nonexistent="shift_forward")
    elif l_tz is not None and r_tz is not None and l_tz != r_tz:
        rt = rt.dt.tz_convert(l_tz)
    left["time_key"] = lt
    right["time_key"] = rt
    merged = pd.merge_asof(
        left.sort_values("time_key"),
        right,
        on="time_key",
        direction="backward",
    )
    return merged[columns].reset_index(drop=True)


def _append_htf_regime_from_1m(result: pd.DataFrame, times_1m: pd.DatetimeIndex) -> None:
    """
    From the 1-min OHLCV in *result*, independently resample to HT **closed** bars
    (right-edge timestamps), compute regime, then merge onto 1-min rows without lookahead.

    Example: at 1m time 09:35, the in-progress 15m bucket ending 09:45 is **not** used;
    features come from the completed bucket ending 09:30.
    """
    last_ts = pd.to_datetime(times_1m.max())
    for rule, tag in (
        ("15min", "pa_htf_15min"),
        ("60min", "pa_htf_60min"),
    ):
        tmp = _resample_ohlcv_htf_closed(result, rule)
        tmp = _drop_incomplete_htf_rows(tmp, last_ts)
        col_state = f"{tag}_state"
        col_score = f"{tag}_trend_score"
        if tmp.empty or len(tmp) < 5:
            result[col_state] = pd.Series(pd.NA, index=result.index, dtype="string")
            result[col_score] = np.nan
            continue
        reg = _market_regime_5m(tmp)
        htf = pd.DataFrame(
            {
                "time_key": tmp["time_key"].values,
                col_state: reg["pa_env_state"].astype(str).to_numpy(),
                col_score: reg["pa_env_trend_score_ratio"].to_numpy(dtype=float),
            }
        )
        cols = [col_state, col_score]
        mapped = _map_htf_closed_to_1min(htf, times_1m, cols)
        result[col_state] = mapped[col_state].astype("string").values
        result[col_score] = mapped[col_score].to_numpy(dtype=float)
        
    # Add trend_alignment based on 15m and 60m trend_score
    if "pa_htf_15min_trend_score" in result.columns and "pa_htf_60min_trend_score" in result.columns:
        score_15m = result["pa_htf_15min_trend_score"].fillna(0).to_numpy(dtype=float)
        score_60m = result["pa_htf_60min_trend_score"].fillna(0).to_numpy(dtype=float)
        # +2 if both bullish, -2 if both bearish, 0 if mixed/neutral
        result["pa_htf_trend_alignment"] = np.sign(score_15m) + np.sign(score_60m)


def _prefix_pa_mtf_columns(df: pd.DataFrame, tf_tag: str) -> pd.DataFrame:
    """Rename ``pa_*`` → ``pa_{tf_tag}_*`` (e.g. tag ``5m`` → ``pa_5m_body_ratio``)."""
    rename: dict[str, str] = {}
    for c in df.columns:
        if c == "time_key":
            continue
        if c.startswith("pa_"):
            rename[c] = f"pa_{tf_tag}_{c[3:]}"
        else:
            rename[c] = f"pa_{tf_tag}_{c}"
    return df.rename(columns=rename)


def _normalize_pa_mtf_list(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        p = part.strip().lower()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _mtf_bar_direction_entropy(close: np.ndarray, window: int = 10) -> np.ndarray:
    """Binary entropy of up-close frequency over a causal rolling window (0 = one-way, 1 = balanced)."""
    n = len(close)
    if n == 0:
        return np.array([], dtype=float)
    diff = np.diff(close, prepend=close[0])
    directions = (diff > 0).astype(float)
    p = pd.Series(directions).rolling(window, min_periods=2).mean().to_numpy()
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-10, 1.0 - 1e-10)
    ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.clip(ent, 0.0, 1.0)


def _mtf_micro_1m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sparse 1m microstructure stack for multi-TF PA (``pa_1m_*`` only)."""
    out = pd.DataFrame(index=df.index)
    out["time_key"] = df["time_key"].values
    n = len(df)
    if n == 0:
        return out
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    rng = np.maximum(highs - lows, 1e-12)
    body = np.abs(closes - opens)
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    wick_ratio = (upper_wick + lower_wick) / rng
    out["pa_1m_wick_ratio"] = np.clip(wick_ratio, 0.0, 5.0)
    out["pa_1m_body_ratio"] = np.clip(body / rng, 0.0, 1.0)
    run = np.zeros(n, dtype=np.int32)
    sgn = np.sign(closes - opens)
    for i in range(1, n):
        if sgn[i] == sgn[i - 1] and sgn[i] != 0:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 0
    out["pa_1m_consec_dir_run"] = run.astype(float)
    prev_c = np.concatenate(([closes[0]], closes[:-1]))
    out["pa_1m_ret_1"] = (closes - prev_c) / np.maximum(prev_c, 1e-12)
    vol = df["volume"].to_numpy(dtype=float)
    vma = pd.Series(vol).rolling(20, min_periods=1).mean().to_numpy()
    out["pa_1m_vol_rel"] = vol / np.maximum(vma, 1e-9)
    out["pa_1m_dir_entropy_10"] = _mtf_bar_direction_entropy(closes, window=10)
    regime_1m = _market_regime_5m(df)
    vcl = _volume_climax_exhaustion_5m(df, regime_1m)
    for c in vcl.columns:
        if c != "time_key" and c.startswith("pa_"):
            out[f"pa_1m_{c[3:]}"] = vcl[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 1a. Opening Range (OR) — computed directly on 1-min bars
# ---------------------------------------------------------------------------

_OR_END_TIME = time(11, 0)  # first 90 min = 9:30 → 11:00 ET
_OR_START_TIME = time(9, 30)


# ---------------------------------------------------------------------------
# 1b. Bar Classification (5-min)
# ---------------------------------------------------------------------------

def _classify_bars_5m(bars: pd.DataFrame, atr_5m: pd.Series) -> pd.DataFrame:
    rng = (bars["high"] - bars["low"]).replace(0, np.nan)
    body = (bars["close"] - bars["open"]).abs()
    body_ratio = (body / rng).fillna(0)

    prev_close = bars["close"].shift(1)
    upper_wick = bars["high"] - bars[["open", "close"]].max(axis=1)
    lower_wick = bars[["open", "close"]].min(axis=1) - bars["low"]
    wick_ratio = ((upper_wick + lower_wick) / rng).fillna(1)

    is_trend_bar = (body_ratio > 0.6) & (rng > atr_5m * 0.5)
    is_range_bar = body_ratio < 0.35

    bull_body = bars["close"] > bars["open"]
    bear_body = bars["close"] < bars["open"]
    strong_bull = (
        bull_body
        & (bars["open"] >= prev_close.shift(0))
        & (bars["close"] > prev_close)
        & (wick_ratio < 0.5)
    )
    strong_bear = (
        bear_body
        & (bars["open"] <= prev_close.shift(0))
        & (bars["close"] < prev_close)
        & (wick_ratio < 0.5)
    )

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_body_ratio"] = body_ratio.values
    out["pa_is_trend_bar"] = is_trend_bar.values
    out["pa_is_range_bar"] = is_range_bar.values
    out["pa_strong_bull_signal"] = strong_bull.fillna(False).values
    out["pa_strong_bear_signal"] = strong_bear.fillna(False).values
    return out


# ---------------------------------------------------------------------------
# 1c. Pullback Counting  H1-H4 / L1-L4 (5-min)
# ---------------------------------------------------------------------------

def _pullback_counting_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """
    Count pullback attempts.
    In an up-trend (direction==1): after a bar with low < prev low (pullback),
    each subsequent bar whose high > prev high increments h_count.
    H2 setup fires on count == 2.
    """
    n = len(bars)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    h_count, l_count = pullback_counting_fast(highs, lows, direction)
    h_count = np.asarray(h_count, dtype=int)
    l_count = np.asarray(l_count, dtype=int)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_h_count"] = h_count
    out["pa_l_count"] = l_count
    out["pa_h_count_raw"] = h_count
    out["pa_l_count_raw"] = l_count
    out["pa_is_h1_setup"] = h_count == 1
    out["pa_is_h2_setup"] = h_count == 2
    out["pa_is_h3_setup"] = h_count == 3
    out["pa_is_h4_setup"] = h_count >= 4
    out["pa_is_l1_setup"] = l_count == 1
    out["pa_is_l2_setup"] = l_count == 2
    out["pa_is_l3_setup"] = l_count == 3
    out["pa_is_l4_setup"] = l_count >= 4
    out["pa_r09_short_setup"] = (direction == -1) & (h_count >= 4)
    out["pa_r09_long_setup"] = (direction == 1) & (l_count >= 4)
    return out


# ---------------------------------------------------------------------------
# 1d. MAG Bar (5-min)
# ---------------------------------------------------------------------------

def _mag_bar_5m(bars: pd.DataFrame, ema_20: pd.Series) -> pd.DataFrame:
    n = len(bars)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    ema = ema_20.to_numpy(dtype=float)
    is_mag = mag_bar_fast(highs, lows, ema)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_is_mag_bar"] = is_mag
    return out


# ---------------------------------------------------------------------------
# 1e. Inside Bars & ii Pattern (5-min)
# ---------------------------------------------------------------------------

def _inside_bars_5m(bars: pd.DataFrame) -> pd.DataFrame:
    prev_high = bars["high"].shift(1)
    prev_low = bars["low"].shift(1)
    # Explicit bool dtype avoids FutureWarning on object/array fillna downcasting (pandas 2.x).
    is_inside = (
        (bars["high"] < prev_high) & (bars["low"] > prev_low)
    ).fillna(False).astype(bool)
    prev_inside = is_inside.shift(1).astype("boolean").fillna(False).astype(bool)
    is_ii = is_inside & prev_inside

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_is_inside_bar"] = is_inside.values
    out["pa_is_ii_pattern"] = is_ii.values
    return out


# ---------------------------------------------------------------------------
# 1f. Market Regime Detection (80% Rule) — 5-min
# ---------------------------------------------------------------------------

def _market_regime_5m(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].astype(float)
    opn = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    rng = (high - low).replace(0, np.nan)
    atr1 = (high - low).fillna(0.0)
    atr20 = atr1.rolling(20, min_periods=5).mean()

    ema20 = _ema(close, 20)
    ema_slope = ema20.diff()
    ema_slope_abs_ma40 = ema_slope.abs().rolling(40, min_periods=10).mean().replace(0, np.nan)
    body = (close - opn).abs()
    body_ratio = (body / rng).fillna(0.0)
    upper_wick = high - pd.concat([opn, close], axis=1).max(axis=1)
    lower_wick = pd.concat([opn, close], axis=1).min(axis=1) - low
    wick_ratio = ((upper_wick + lower_wick) / rng).fillna(1.0)

    above_ema = close > ema20
    below_ema = close < ema20

    n = len(bars)
    above_run = np.zeros(n, dtype=int)
    below_run = np.zeros(n, dtype=int)
    same_color_run = np.zeros(n, dtype=int)
    body_overlap = np.zeros(n, dtype=float)
    cross = np.zeros(n, dtype=int)
    for i in range(n):
        above_run[i] = above_run[i - 1] + 1 if i > 0 and above_ema.iloc[i] else (1 if above_ema.iloc[i] else 0)
        below_run[i] = below_run[i - 1] + 1 if i > 0 and below_ema.iloc[i] else (1 if below_ema.iloc[i] else 0)
        dir_i = np.sign(close.iloc[i] - opn.iloc[i])
        if i == 0:
            same_color_run[i] = 1
        else:
            dir_prev = np.sign(close.iloc[i - 1] - opn.iloc[i - 1])
            same_color_run[i] = same_color_run[i - 1] + 1 if dir_i != 0 and dir_i == dir_prev else 1
            prev_low_body = min(opn.iloc[i - 1], close.iloc[i - 1])
            prev_high_body = max(opn.iloc[i - 1], close.iloc[i - 1])
            cur_low_body = min(opn.iloc[i], close.iloc[i])
            cur_high_body = max(opn.iloc[i], close.iloc[i])
            overlap = max(0.0, min(prev_high_body, cur_high_body) - max(prev_low_body, cur_low_body))
            prev_body = max(prev_high_body - prev_low_body, 1e-12)
            cur_body = max(cur_high_body - cur_low_body, 1e-12)
            body_overlap[i] = overlap / max(prev_body, cur_body)
            prev_side = np.sign(close.iloc[i - 1] - ema20.iloc[i - 1]) if np.isfinite(ema20.iloc[i - 1]) else 0
            cur_side = np.sign(close.iloc[i] - ema20.iloc[i]) if np.isfinite(ema20.iloc[i]) else 0
            cross[i] = 1 if prev_side != 0 and cur_side != 0 and prev_side != cur_side else 0

    hh_hl = (high.diff() > 0) & (low.diff() > 0)
    lh_ll = (high.diff() < 0) & (low.diff() < 0)
    avg_body20 = body.rolling(20, min_periods=5).mean()
    bull_close_extreme_q = close >= (high - 0.25 * (high - low))
    bear_close_extreme_q = close <= (low + 0.25 * (high - low))
    breakout_bull = (body_ratio >= 0.6) & (body > 1.5 * avg_body20) & (close > high.shift(1)) & bull_close_extreme_q
    breakout_bear = (body_ratio >= 0.6) & (body > 1.5 * avg_body20) & (close < low.shift(1)) & bear_close_extreme_q

    # Pullback depth/time metrics.
    leg_hi10 = high.rolling(10, min_periods=3).max()
    leg_lo10 = low.rolling(10, min_periods=3).min()
    leg_span10 = (leg_hi10 - leg_lo10).replace(0, np.nan)
    pullback_up = (leg_hi10 - close) / leg_span10
    pullback_down = (close - leg_lo10) / leg_span10
    counter_run = np.zeros(n, dtype=int)
    for i in range(1, n):
        if close.iloc[i] < opn.iloc[i]:
            counter_run[i] = counter_run[i - 1] + 1
        elif close.iloc[i] > opn.iloc[i]:
            counter_run[i] = 0

    # Trend majority conditions T1..T7 (long/short side).
    t1_long = (ema_slope > 0) & (pd.Series(above_ema, index=bars.index).rolling(20, min_periods=5).mean() >= 0.6)
    t1_short = (ema_slope < 0) & (pd.Series(below_ema, index=bars.index).rolling(20, min_periods=5).mean() >= 0.6)
    t2_long = pd.Series(above_run, index=bars.index) >= 5
    t2_short = pd.Series(below_run, index=bars.index) >= 5
    t3_long = (pd.Series(same_color_run, index=bars.index) >= 3) & (close > opn) & (body_ratio >= 0.6)
    t3_short = (pd.Series(same_color_run, index=bars.index) >= 3) & (close < opn) & (body_ratio >= 0.6)
    t4_long = pullback_up < 0.5
    t4_short = pullback_down < 0.5
    t5_long = pd.Series(counter_run, index=bars.index) <= 5
    t5_short = pd.Series(counter_run, index=bars.index) <= 5
    t6_long = hh_hl.rolling(5, min_periods=2).sum() >= 3
    t6_short = lh_ll.rolling(5, min_periods=2).sum() >= 3
    t7_long = breakout_bull.rolling(20, min_periods=5).max().fillna(False)
    t7_short = breakout_bear.rolling(20, min_periods=5).max().fillna(False)

    trend_long_count = (
        t1_long.astype(int)
        + t2_long.astype(int)
        + t3_long.astype(int)
        + t4_long.astype(int)
        + t5_long.astype(int)
        + t6_long.astype(int)
        + t7_long.astype(int)
    )
    trend_short_count = (
        t1_short.astype(int)
        + t2_short.astype(int)
        + t3_short.astype(int)
        + t4_short.astype(int)
        + t5_short.astype(int)
        + t6_short.astype(int)
        + t7_short.astype(int)
    )
    trend_count = np.maximum(trend_long_count.to_numpy(dtype=float), trend_short_count.to_numpy(dtype=float))
    trend_score_ratio = trend_count / 7.0
    trend_dir = np.where(trend_long_count > trend_short_count, 1, np.where(trend_short_count > trend_long_count, -1, 0))
    trend_dir_s = pd.Series(trend_dir, index=bars.index)

    # 20-bar inertia counter (reset on new trend extreme).
    bars_since_extreme = np.zeros(n, dtype=int)
    inertia_prob = np.full(n, 0.50, dtype=float)
    inertia_bucket = np.array(["neutral_50_50"] * n, dtype=object)
    last_dir = 0
    cur_count = 0
    last_extreme = np.nan
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    for i in range(n):
        d = int(trend_dir[i])
        if d == 1:
            if last_dir != 1:
                cur_count = 0
                last_extreme = highs[i]
            elif np.isfinite(last_extreme) and highs[i] >= last_extreme:
                cur_count = 0
                last_extreme = highs[i]
            else:
                cur_count += 1
        elif d == -1:
            if last_dir != -1:
                cur_count = 0
                last_extreme = lows[i]
            elif np.isfinite(last_extreme) and lows[i] <= last_extreme:
                cur_count = 0
                last_extreme = lows[i]
            else:
                cur_count += 1
        else:
            cur_count += 1
            last_extreme = np.nan
        bars_since_extreme[i] = cur_count
        last_dir = d
        if cur_count <= 5:
            inertia_prob[i] = 0.90
            inertia_bucket[i] = "bars_0_5"
        elif cur_count <= 10:
            inertia_prob[i] = 0.75
            inertia_bucket[i] = "bars_6_10"
        elif cur_count <= 15:
            inertia_prob[i] = 0.60
            inertia_bucket[i] = "bars_11_15"
        elif cur_count <= 19:
            inertia_prob[i] = 0.55
            inertia_bucket[i] = "bars_16_19"
        else:
            inertia_prob[i] = 0.50
            inertia_bucket[i] = "bars_ge_20"

    # Range conditions R1..R6
    prev_leg_h = (high.rolling(40, min_periods=10).max() - low.rolling(40, min_periods=10).min()).shift(20).replace(0, np.nan)
    range_h20 = (high.rolling(20, min_periods=5).max() - low.rolling(20, min_periods=5).min())
    r1 = range_h20 < (0.5 * prev_leg_h)
    r2 = ema_slope.abs() < (ema_slope_abs_ma40 / 3.0)
    r3 = pd.Series(cross, index=bars.index).rolling(10, min_periods=5).sum() >= 3
    failed_up = (high >= high.rolling(20, min_periods=5).max().shift(1)) & (close < high.rolling(20, min_periods=5).max().shift(1))
    failed_dn = (low <= low.rolling(20, min_periods=5).min().shift(1)) & (close > low.rolling(20, min_periods=5).min().shift(1))
    r4 = (failed_up | failed_dn).rolling(5, min_periods=3).sum() >= 4
    alt = (np.sign(close - opn).diff().abs() > 0).astype(float)
    r5 = (alt.rolling(10, min_periods=5).mean() >= 0.6) & (pd.Series(same_color_run, index=bars.index) < 3)
    r6 = (((upper_wick > body / 3.0) & (lower_wick > body / 3.0)).astype(float).rolling(10, min_periods=5).mean() >= 0.6)
    range_count = r1.astype(int) + r2.astype(int) + r3.astype(int) + r4.astype(int) + r5.astype(int) + r6.astype(int)
    range_score_ratio = range_count / 6.0

    # Tight channel / broad channel / TTR thresholds.
    pullback_len_ok = pd.Series(counter_run, index=bars.index).between(1, 3)
    pullback_amp_ok = (leg_span10 * pullback_up.fillna(0.0) < 2.0 * atr1) | (leg_span10 * pullback_down.fillna(0.0) < 2.0 * atr1)
    gap_unfilled = np.where(trend_dir == 1, low > opn.shift(1), np.where(trend_dir == -1, high < opn.shift(1), False))
    overlap_low = pd.Series(body_overlap, index=bars.index) < 0.3
    close_extreme = np.where(
        trend_dir == 1,
        close >= (low + (2.0 / 3.0) * (high - low)),
        np.where(trend_dir == -1, close <= (low + (1.0 / 3.0) * (high - low)), False),
    )
    close_extreme_ratio = pd.Series(close_extreme, index=bars.index).rolling(10, min_periods=5).mean() >= 0.7
    tight_channel_all = pullback_len_ok & pullback_amp_ok & pd.Series(gap_unfilled, index=bars.index).fillna(False) & overlap_low & close_extreme_ratio

    broad_1 = (pullback_up > 0.5) | (pullback_down > 0.5)
    broad_2 = pd.Series(same_color_run, index=bars.index).rolling(10, min_periods=5).mean() >= 5
    broad_3 = (hh_hl.rolling(10, min_periods=5).sum() >= 2) & (lh_ll.rolling(10, min_periods=5).sum() >= 2)
    broad_4 = ((low <= ema20) | (high >= ema20)).rolling(10, min_periods=5).mean() >= 0.6
    broad_majority = (broad_1.astype(int) + broad_2.astype(int) + broad_3.astype(int) + broad_4.astype(int)) >= 2

    ttr_1 = range_h20 < (3.0 * atr20)
    ttr_2 = pd.Series(body_overlap, index=bars.index).rolling(10, min_periods=5).mean() > 0.5
    ttr_3 = (((upper_wick > 0) & (lower_wick > 0)).astype(float).rolling(10, min_periods=5).mean() >= 0.6)
    ttr_4 = pd.Series(same_color_run, index=bars.index) <= 2
    ttr_5 = (range_score_ratio.rolling(5, min_periods=3).mean() >= 0.6)
    ttr_all = ttr_1 & ttr_2 & ttr_3 & ttr_4 & ttr_5

    # Scorecard (section 6).
    ema_dir_score = np.where(ema_slope > 0, 1, np.where(ema_slope < 0, -1, 0))
    price_vs_ema_score = np.where(above_run >= 5, 1, np.where(below_run >= 5, -1, 0))
    hh_hl_score = np.where(hh_hl.rolling(4, min_periods=2).sum() >= 3, 1, np.where(lh_ll.rolling(4, min_periods=2).sum() >= 3, -1, 0))
    breakout_score = np.where(breakout_bull, 1, np.where(breakout_bear, -1, 0))
    pb_depth_score = np.where((trend_dir_s == 1) & (pullback_up < 0.38), 1, np.where((trend_dir_s == -1) & (pullback_down < 0.38), -1, 0))
    pb_duration_score = np.where((trend_dir_s == 1) & (pd.Series(counter_run, index=bars.index) <= 3), 1, np.where((trend_dir_s == -1) & (pd.Series(counter_run, index=bars.index) <= 3), -1, 0))
    twenty_bar_score = np.where(bars_since_extreme < 10, trend_dir, 0)
    bar_quality_score = np.where((body_ratio >= 0.6) & (wick_ratio < 0.4) & (close > opn), 1, np.where((body_ratio >= 0.6) & (wick_ratio < 0.4) & (close < opn), -1, 0))
    total_score = ema_dir_score + price_vs_ema_score + hh_hl_score + breakout_score + pb_depth_score + pb_duration_score + twenty_bar_score + bar_quality_score

    # Final state mapping: scorecard bins follow Brooks-style quick template.
    state = np.array(["wide_tr"] * n, dtype=object)
    state = np.where(total_score >= 6, "tight_bull_channel", state)
    state = np.where((total_score >= 3) & (total_score <= 5), "wide_bull_channel", state)
    state = np.where((total_score >= -5) & (total_score <= -3), "wide_bear_channel", state)
    state = np.where(total_score <= -6, "tight_bear_channel", state)
    state = np.where((total_score >= -2) & (total_score <= 2), "wide_tr", state)

    # Tight channel must pass strict pullback/gap constraints.
    state = np.where((state == "tight_bull_channel") & (~tight_channel_all.to_numpy(dtype=bool)), "wide_bull_channel", state)
    state = np.where((state == "tight_bear_channel") & (~tight_channel_all.to_numpy(dtype=bool)), "wide_bear_channel", state)

    # EMA flat / repeated crossing + overlap compression => TTR.
    state = np.where((state == "wide_tr") & ttr_all.to_numpy(dtype=bool), "ttr", state)
    # 20-bar inertia exhausted -> neutral 50/50.
    state = np.where((bars_since_extreme >= 20) & (np.abs(total_score) <= 2), "neutral_50_50", state)

    order_type = np.array(["limit"] * n, dtype=object)
    entry_style = np.array(["range_thirds"] * n, dtype=object)
    stop_style = np.array(["range_plus_1atr"] * n, dtype=object)
    tp_style = np.array(["to_opposite_or_midline"] * n, dtype=object)
    position_size = np.array(["0.50_to_0.75x"] * n, dtype=object)
    tight = np.isin(state, ["tight_bull_channel", "tight_bear_channel"])
    wide_channel = np.isin(state, ["wide_bull_channel", "wide_bear_channel"])
    ttr = state == "ttr"
    neutral = state == "neutral_50_50"
    order_type[tight] = "stop"
    entry_style[tight] = "h1_h2_or_l1_l2_breakout"
    stop_style[tight] = "leg_extreme_or_ema_other_side"
    tp_style[tight] = "hold_until_trend_exhaustion"
    position_size[tight] = "1.00x"
    order_type[wide_channel] = "stop_plus_limit"
    entry_style[wide_channel] = "trend_pullback_or_boundary_reversal"
    stop_style[wide_channel] = "outside_recent_swing"
    tp_style[wide_channel] = "channel_opposite_or_mm_target"
    position_size[wide_channel] = "1.00x"
    order_type[ttr] = "none"
    entry_style[ttr] = "no_trade_wait_breakout_confirm"
    stop_style[ttr] = "n_a"
    tp_style[ttr] = "n_a"
    position_size[ttr] = "n_a"
    order_type[neutral] = "wait"
    entry_style[neutral] = "strong_breakout_plus_follow_through"
    stop_style[neutral] = "below_breakout_midpoint"
    tp_style[neutral] = "conservative_1r"
    position_size[neutral] = "0.50x"

    is_trend = np.isin(state, ["tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"])
    is_range = np.isin(state, ["wide_tr", "ttr", "neutral_50_50"])

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_env_trend_dir"] = trend_dir
    out["pa_env_bars_since_extreme"] = bars_since_extreme
    out["pa_env_trend_inertia"] = inertia_prob
    out["pa_env_inertia_bucket"] = inertia_bucket
    out["pa_env_trend_score_ratio"] = trend_score_ratio
    out["pa_env_range_score_ratio"] = range_score_ratio.to_numpy(dtype=float)
    out["pa_env_score_ema_dir"] = ema_dir_score
    out["pa_env_score_price_vs_ema"] = price_vs_ema_score
    out["pa_env_score_hh_hl"] = hh_hl_score
    out["pa_env_score_breakout_bar"] = breakout_score
    out["pa_env_score_pullback_depth"] = pb_depth_score
    out["pa_env_score_pullback_duration"] = pb_duration_score
    out["pa_env_score_twenty_bar"] = twenty_bar_score
    out["pa_env_score_bar_quality"] = bar_quality_score
    out["pa_env_score_total"] = total_score
    out["pa_env_state"] = state
    out["pa_env_order_type"] = order_type
    out["pa_env_entry_style"] = entry_style
    out["pa_env_stop_style"] = stop_style
    out["pa_env_take_profit_style"] = tp_style
    out["pa_env_position_size"] = position_size
    out["pa_regime_trend"] = is_trend
    out["pa_regime_range"] = is_range
    out["pa_reversal_likely_fail"] = is_trend
    out["pa_breakout_likely_fail"] = np.isin(state, ["wide_tr", "ttr"])
    out["pa_is_tight_channel"] = tight_channel_all.to_numpy(dtype=bool)
    out["pa_is_broad_channel"] = broad_majority.to_numpy(dtype=bool)
    out["pa_is_ttr"] = ttr_all.to_numpy(dtype=bool)
    return out


# ---------------------------------------------------------------------------
# 1g. 20-Bar Rule (5-min)
# ---------------------------------------------------------------------------

def _twenty_bar_rule_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    n = len(bars)
    bars_since_change = np.zeros(n, dtype=int)
    trend_weakened = np.zeros(n, dtype=bool)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)

    prev_dir = 0
    cur_count = 0
    last_extreme = np.nan
    for i in range(n):
        d = int(direction[i])
        if d == 1:
            if prev_dir != 1:
                cur_count = 0
                last_extreme = highs[i]
            elif np.isfinite(last_extreme) and highs[i] >= last_extreme:
                cur_count = 0
                last_extreme = highs[i]
            else:
                cur_count += 1
        elif d == -1:
            if prev_dir != -1:
                cur_count = 0
                last_extreme = lows[i]
            elif np.isfinite(last_extreme) and lows[i] <= last_extreme:
                cur_count = 0
                last_extreme = lows[i]
            else:
                cur_count += 1
        else:
            cur_count += 1
            last_extreme = np.nan

        bars_since_change[i] = cur_count
        trend_weakened[i] = cur_count >= 20
        prev_dir = d

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_bars_since_trend_change"] = bars_since_change
    out["pa_trend_weakened"] = trend_weakened
    return out


# ---------------------------------------------------------------------------
# 1h. Buying / Selling Pressure Score (5-min rolling window)
# ---------------------------------------------------------------------------

def _pressure_score_5m(bars: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    close = bars["close"].astype(float)
    opn = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    bull_bar = (close > opn).astype(float)
    bear_bar = (close < opn).astype(float)

    body = (close - opn).abs()
    rng = (high - low).replace(0, np.nan)

    bull_body = np.where(close > opn, body, 0.0)
    bear_body = np.where(close < opn, body, 0.0)
    bull_body_s = pd.Series(bull_body, index=bars.index)
    bear_body_s = pd.Series(bear_body, index=bars.index)

    close_pos = ((close - low) / rng).fillna(0.5)

    bull_count = bull_bar.rolling(window, min_periods=1).sum()
    bear_count = bear_bar.rolling(window, min_periods=1).sum()
    bar_score = (bull_count - bear_count) / window * 20

    bull_body_avg = bull_body_s.rolling(window, min_periods=1).mean()
    bear_body_avg = bear_body_s.rolling(window, min_periods=1).mean()
    total_body = (bull_body_avg + bear_body_avg).replace(0, np.nan)
    body_score = ((bull_body_avg - bear_body_avg) / total_body).fillna(0) * 20

    close_score = (close_pos.rolling(window, min_periods=1).mean() - 0.5) * 40

    # Micro channel: 3+ consecutive non-overlapping bars in same direction
    micro = np.zeros(len(bars), dtype=float)
    consec = 0
    for i in range(1, len(bars)):
        if close.iloc[i] > opn.iloc[i] and low.iloc[i] >= low.iloc[i - 1]:
            consec += 1
        elif close.iloc[i] < opn.iloc[i] and high.iloc[i] <= high.iloc[i - 1]:
            consec -= 1
        else:
            consec = 0
        if abs(consec) >= 3:
            micro[i] = np.sign(consec) * 20

    bull_pressure = (bar_score + body_score + close_score).clip(0, 100) + np.clip(micro, 0, 20)
    bear_pressure = (-bar_score - body_score - close_score).clip(0, 100) + np.clip(-micro, 0, 20)

    bull_pressure = bull_pressure.clip(0, 100)
    bear_pressure = bear_pressure.clip(0, 100)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_bull_pressure"] = bull_pressure.values
    out["pa_bear_pressure"] = bear_pressure.values
    out["pa_net_pressure"] = (bull_pressure - bear_pressure).values
    return out


# ---------------------------------------------------------------------------
# 1j. Gap Quantification (5-min)
# ---------------------------------------------------------------------------

def _gap_detection_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    n = len(bars)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    close_vals = bars["close"].to_numpy(dtype=float)

    is_exhaustion = np.zeros(n, dtype=bool)
    is_measured = np.zeros(n, dtype=bool)

    trend_bar_count = 0
    for i in range(2, n):
        if direction[i] == direction[i - 1] and direction[i] != 0:
            trend_bar_count += 1
        else:
            trend_bar_count = 0

        bull_gap = lows[i] > highs[i - 2]
        bear_gap = highs[i] < lows[i - 2]

        body = abs(close_vals[i] - bars["open"].iloc[i])
        rng = highs[i] - lows[i]
        is_big_body = body > rng * 0.6 if rng > 0 else False

        if (bull_gap or bear_gap) and trend_bar_count >= 20 and is_big_body:
            is_exhaustion[i] = True
        elif (bull_gap or bear_gap) and 5 <= trend_bar_count < 20 and is_big_body:
            is_measured[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_is_exhaustion_gap"] = is_exhaustion
    out["pa_is_measured_gap"] = is_measured
    return out


# ---------------------------------------------------------------------------
# EMA helper (used for MAG bar)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, min_periods=length, adjust=False).mean()


# ---------------------------------------------------------------------------
# Direction helper — derive from CE direction on 5-min data
# ---------------------------------------------------------------------------

def _derive_direction_5m(bars_5m: pd.DataFrame) -> np.ndarray:
    """Simple direction from close vs open and consecutive bars."""
    close = bars_5m["close"].to_numpy(dtype=float)
    ema20 = _ema(bars_5m["close"], 20).to_numpy(dtype=float)
    n = len(bars_5m)
    direction = np.zeros(n, dtype=np.int8)
    valid = np.isfinite(ema20) & np.isfinite(close)
    direction[valid & (close > ema20)] = 1
    direction[valid & (close < ema20)] = -1
    return direction.astype(int, copy=False)


# ── Bar-pattern supplement ──

def _signal_bar_scoring_5m(bars: pd.DataFrame) -> pd.DataFrame:
    """Score buy/sell signal bar quality (0–7) per B08–B20 style checks."""
    n = len(bars)
    opn = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    rng = np.maximum(high - low, 1e-12)
    body = np.abs(close - opn)
    body_ratio = body / rng

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low, 1)
    prev_low[0] = np.nan

    upper_wick = high - np.maximum(opn, close)
    lower_wick = np.minimum(opn, close) - low
    upper_wick_ratio = upper_wick / rng
    lower_wick_ratio = lower_wick / rng

    prev_body_hi = np.maximum(np.roll(opn, 1), np.roll(close, 1))
    prev_body_lo = np.minimum(np.roll(opn, 1), np.roll(close, 1))
    prev_body_hi[0] = np.nan
    prev_body_lo[0] = np.nan
    cur_body_hi = np.maximum(opn, close)
    cur_body_lo = np.minimum(opn, close)
    overlap_amt = np.maximum(
        0.0, np.minimum(prev_body_hi, cur_body_hi) - np.maximum(prev_body_lo, cur_body_lo)
    )
    overlap_ratio = overlap_amt / np.maximum(body, 1e-12)

    is_inside = (high < prev_high) & (low > prev_low)
    is_doji = body_ratio < 0.3

    buy_score = np.zeros(n, dtype=float)
    sell_score = np.zeros(n, dtype=float)

    for i in range(1, n):
        if np.isnan(prev_close[i]):
            continue
        bs = ss = 0.0
        if opn[i] <= prev_close[i] + 0.1 * rng[i]:
            bs += 1.0
        if close[i] > prev_close[i] + 0.3 * rng[i]:
            bs += 1.0
        if 0.2 <= lower_wick_ratio[i] <= 0.55:
            bs += 1.0
        if upper_wick_ratio[i] < 0.15:
            bs += 1.0
        if overlap_ratio[i] < 0.3:
            bs += 1.0
        if close[i] > prev_close[i] and close[i] > prev_high[i]:
            bs += 1.0
        if not is_doji[i] and not is_inside[i]:
            bs += 1.0

        if opn[i] >= prev_close[i] - 0.1 * rng[i]:
            ss += 1.0
        if close[i] < prev_close[i] - 0.3 * rng[i]:
            ss += 1.0
        if 0.2 <= upper_wick_ratio[i] <= 0.55:
            ss += 1.0
        if lower_wick_ratio[i] < 0.15:
            ss += 1.0
        if overlap_ratio[i] < 0.3:
            ss += 1.0
        if close[i] < prev_close[i] and close[i] < prev_low[i]:
            ss += 1.0
        if not is_doji[i] and not is_inside[i]:
            ss += 1.0

        buy_score[i] = bs
        sell_score[i] = ss

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_buy_signal_score"] = buy_score
    out["pa_sell_signal_score"] = sell_score
    out["pa_is_strong_buy_signal"] = buy_score >= 5
    out["pa_is_strong_sell_signal"] = sell_score >= 5
    return out


def _outside_engulfing_5m(bars: pd.DataFrame) -> pd.DataFrame:
    n = len(bars)
    opn = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan

    outside = (high > prev_high) & (low < prev_low)
    bull_eng = outside & (close > opn) & (close > prev_high)
    bear_eng = outside & (close < opn) & (close < prev_low)

    multi_bull = np.zeros(n, dtype=bool)
    multi_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if close[i] > close[i - 1] > close[i - 2] and low[i] > low[i - 1] and close[i] > opn[i]:
            multi_bull[i] = True
        if close[i] < close[i - 1] < close[i - 2] and high[i] < high[i - 1] and close[i] < opn[i]:
            multi_bear[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_is_outside_bar"] = np.asarray(outside, dtype=bool)
    out["pa_is_engulfing_bull"] = np.asarray(bull_eng, dtype=bool)
    out["pa_is_engulfing_bear"] = np.asarray(bear_eng, dtype=bool)
    out["pa_multi_bar_reversal_bull"] = multi_bull
    out["pa_multi_bar_reversal_bear"] = multi_bear
    return out


def _mtr_detection_5m(
    bars: pd.DataFrame,
    regime: pd.DataFrame,
    pressure: pd.DataFrame,
    double_tb: pd.DataFrame,
) -> pd.DataFrame:
    env = regime["pa_env_state"].astype(str).to_numpy()
    bear_tr = np.isin(env, ["tight_bear_channel", "wide_bear_channel"])
    bull_tr = np.isin(env, ["tight_bull_channel", "wide_bull_channel"])
    dt = double_tb.get("pa_double_top", pd.Series(False, index=bars.index)).to_numpy(dtype=bool)
    db = double_tb.get("pa_double_bottom", pd.Series(False, index=bars.index)).to_numpy(dtype=bool)
    netp = pressure["pa_net_pressure"].to_numpy(dtype=float)

    bull_ready = bull_tr & dt & (netp < 0)
    bear_ready = bear_tr & db & (netp > 0)
    score = bear_ready.astype(np.float32) * 2.0 - bull_ready.astype(np.float32) * 2.0
    score += (dt.astype(np.float32) - db.astype(np.float32)) * 0.5

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_mtr_bull_ready"] = bull_ready
    out["pa_mtr_bear_ready"] = bear_ready
    out["pa_mtr_score"] = score
    return out


def _climax_detection_5m(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    vol = bars["volume"].astype(float)
    rng = (high - low).replace(0, np.nan)
    body = (close - bars["open"].astype(float)).abs()
    body_ratio = (body / rng).fillna(0)
    avg_body = body.rolling(20, min_periods=5).mean()
    vol_ma = vol.rolling(20, min_periods=5).mean()

    buy_climax = (
        (close > bars["open"])
        & (body_ratio > 0.55)
        & (body > 1.8 * avg_body)
        & (close >= high - 0.15 * rng)
        & (vol > 1.5 * vol_ma)
    )
    sell_climax = (
        (close < bars["open"])
        & (body_ratio > 0.55)
        & (body > 1.8 * avg_body)
        & (close <= low + 0.15 * rng)
        & (vol > 1.5 * vol_ma)
    )

    consec = np.zeros(len(bars), dtype=int)
    for i in range(1, len(bars)):
        if buy_climax.iloc[i]:
            consec[i] = consec[i - 1] + 1 if buy_climax.iloc[i - 1] else 1
        elif sell_climax.iloc[i]:
            consec[i] = consec[i - 1] + 1 if sell_climax.iloc[i - 1] else 1
        else:
            consec[i] = 0

    vtop = (high.diff() > 0) & (high.diff(-1) > 0) & (body_ratio > 0.5)
    vbot = (low.diff() < 0) & (low.diff(-1) < 0) & (body_ratio > 0.5)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_buy_climax"] = buy_climax.fillna(False).values
    out["pa_sell_climax"] = sell_climax.fillna(False).values
    out["pa_consec_climax_count"] = consec
    out["pa_v_shape_top"] = vtop.fillna(False).values
    out["pa_v_shape_bottom"] = vbot.fillna(False).values
    return out


def _tr_measured_move_5m(bars: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Measured moves for Trading Ranges (TR)."""
    env = regime["pa_env_state"].astype(str).to_numpy()
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    n = len(bars)
    
    tr_mm_target_up = np.full(n, np.nan)
    tr_mm_target_down = np.full(n, np.nan)
    
    tr_mask = np.isin(env, ["wide_tr", "ttr", "neutral_50_50"])
    
    current_tr_high = np.nan
    current_tr_low = np.nan
    
    for i in range(n):
        if tr_mask[i]:
            if np.isnan(current_tr_high):
                current_tr_high = high.iloc[i]
                current_tr_low = low.iloc[i]
            else:
                current_tr_high = max(current_tr_high, high.iloc[i])
                current_tr_low = min(current_tr_low, low.iloc[i])
        else:
            if i > 0 and tr_mask[i - 1] and not np.isnan(current_tr_high):
                box_height = current_tr_high - current_tr_low
                if close.iloc[i] > current_tr_high:
                    tr_mm_target_up[i] = current_tr_high + box_height
                elif close.iloc[i] < current_tr_low:
                    tr_mm_target_down[i] = current_tr_low - box_height
            
            current_tr_high = np.nan
            current_tr_low = np.nan
            
        if i > 0 and not tr_mask[i]:
            if np.isnan(tr_mm_target_up[i]) and not np.isnan(tr_mm_target_up[i - 1]):
                if high.iloc[i] < tr_mm_target_up[i - 1]:
                    tr_mm_target_up[i] = tr_mm_target_up[i - 1]
            if np.isnan(tr_mm_target_down[i]) and not np.isnan(tr_mm_target_down[i - 1]):
                if low.iloc[i] > tr_mm_target_down[i - 1]:
                    tr_mm_target_down[i] = tr_mm_target_down[i - 1]
                    
    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_tr_mm_target_up"] = tr_mm_target_up
    out["pa_tr_mm_target_down"] = tr_mm_target_down
    return out


def _volume_climax_exhaustion_5m(bars: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Identify Volume Exhaustion/Climax based on extreme Z-Score after a trend."""
    vol = bars["volume"].astype(float)
    ages = regime["pa_env_bars_since_extreme"].to_numpy(dtype=int)
    env = regime["pa_env_state"].astype(str).to_numpy()
    
    vol_ma = vol.rolling(20, min_periods=5).mean()
    vol_std = vol.rolling(20, min_periods=5).std()
    vol_zscore = ((vol - vol_ma) / (vol_std + 1e-8)).fillna(0)
    
    is_trend = np.isin(env, ["tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"])
    vol_exhaustion = is_trend & (ages >= 20) & (vol_zscore > 3.0)
    
    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_vol_exhaustion_climax"] = vol_exhaustion.astype(bool).values
    out["pa_vol_zscore_20"] = vol_zscore.values
    return out


def _final_flag_5m(bars: pd.DataFrame, regime: pd.DataFrame, inside: pd.DataFrame) -> pd.DataFrame:
    env = regime["pa_env_state"].astype(str).to_numpy()
    bull_late = np.isin(env, ["tight_bull_channel", "wide_bull_channel"]) & (
        regime["pa_env_bars_since_extreme"].to_numpy(dtype=int) >= 15
    )
    bear_late = np.isin(env, ["tight_bear_channel", "wide_bear_channel"]) & (
        regime["pa_env_bars_since_extreme"].to_numpy(dtype=int) >= 15
    )
    tight_tr = regime["pa_is_ttr"].to_numpy(dtype=bool)
    ii = inside["pa_is_ii_pattern"].to_numpy(dtype=bool)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_final_flag_bull"] = bull_late & tight_tr & ii
    out["pa_final_flag_bear"] = bear_late & tight_tr & ii
    return out


def _momentum_accel_5m(bars: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    c = bars["close"].astype(float)
    o = bars["open"].astype(float)
    body = (c - o).abs()
    up = (body > body.shift(1)) & (c > o)
    down = (body > body.shift(1)) & (c < o)
    shrink_bull = (body < body.shift(1)) & (c > o)
    shrink_bear = (body < body.shift(1)) & (c < o)
    accel = (up.rolling(window, min_periods=2).sum() - down.rolling(window, min_periods=2).sum()).fillna(0)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_bull_body_growing"] = up.fillna(False).values
    out["pa_bear_body_growing"] = down.fillna(False).values
    out["pa_bull_body_shrinking"] = shrink_bull.fillna(False).values
    out["pa_bear_body_shrinking"] = shrink_bear.fillna(False).values
    out["pa_momentum_accel"] = accel.values
    return out


def _counter_trend_quality_5m(
    bars: pd.DataFrame, regime: pd.DataFrame, sig_scores: pd.DataFrame,
) -> pd.DataFrame:
    tr = regime["pa_regime_range"].to_numpy(dtype=bool)
    bull_sig = sig_scores["pa_buy_signal_score"].to_numpy(dtype=float)
    bear_sig = sig_scores["pa_sell_signal_score"].to_numpy(dtype=float)
    body = (bars["close"] - bars["open"]).abs() / (bars["high"] - bars["low"]).replace(0, np.nan)
    body = body.fillna(0).to_numpy(dtype=float)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_counter_trend_quality_long"] = np.where(tr, bull_sig * (0.5 + body), 0.0)
    out["pa_counter_trend_quality_short"] = np.where(tr, bear_sig * (0.5 + body), 0.0)
    return out


def _channel_zones_5m(bars: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float)
    leg_hi = high.rolling(30, min_periods=5).max()
    leg_lo = low.rolling(30, min_periods=5).min()
    span = (leg_hi - leg_lo).replace(0, np.nan)
    pos = ((close - leg_lo) / span).clip(0, 1).fillna(0.5)
    ch = regime["pa_is_tight_channel"] | regime["pa_is_broad_channel"]

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_channel_position"] = np.where(ch.to_numpy(), pos.to_numpy(), 0.5)
    out["pa_channel_50pct_zone"] = (pos - 0.5).abs() < 0.08
    out["pa_at_channel_buy_zone"] = ch.to_numpy(dtype=bool) & (pos.to_numpy(dtype=float) < 0.35)
    out["pa_at_channel_sell_zone"] = ch.to_numpy(dtype=bool) & (pos.to_numpy(dtype=float) > 0.65)
    return out


def _endless_pullback_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    n = len(bars)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    counter = np.zeros(n, dtype=int)
    for i in range(1, n):
        if bars["close"].iloc[i] < bars["open"].iloc[i]:
            counter[i] = counter[i - 1] + 1 if direction[i] == 1 else 0
        elif bars["close"].iloc[i] > bars["open"].iloc[i]:
            counter[i] = counter[i - 1] + 1 if direction[i] == -1 else 0
        else:
            counter[i] = 0
    endless = counter >= 20
    prob = np.where(counter >= 20, 0.5, np.clip(1.0 - counter / 25.0, 0.5, 0.95))

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_endless_pullback"] = endless
    out["pa_trend_resume_prob"] = prob
    return out


def _gap_enhanced_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    n = len(bars)

    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = prev_low[0] = np.nan

    low_prev_high = (low > prev_high) & np.isfinite(prev_high)
    high_prev_low = (high < prev_low) & np.isfinite(prev_low)

    island_top = np.zeros(n, dtype=bool)
    island_bot = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if low_prev_high[i] and close[i] < prev_high[i]:
            island_bot[i] = True
        if high_prev_low[i] and close[i] > prev_low[i]:
            island_top[i] = True

    neg_gap = high_prev_low

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_island_reversal_top"] = island_top
    out["pa_island_reversal_bottom"] = island_bot
    out["pa_negative_gap"] = neg_gap
    return out


def _day_session_phase(bars: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(bars["time_key"])
    bt = times.dt.time
    is_opening = (bt >= time(9, 30)) & (bt < time(11, 0))
    is_midday = (bt >= time(11, 0)) & (bt < time(14, 0))
    is_closing = bt >= time(14, 0)
    phase = np.where(is_opening, 0.0, np.where(is_midday, 0.5, 1.0))
    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_session_phase"] = phase
    out["pa_is_opening"] = is_opening.values
    out["pa_is_midday"] = is_midday.values
    out["pa_is_closing"] = is_closing.values
    return out


def _market_cycle_5m(regime: pd.DataFrame) -> pd.DataFrame:
    state = regime["pa_env_state"].astype(str).to_numpy()
    ages = regime["pa_env_bars_since_extreme"].to_numpy(dtype=int)

    phase_map = {
        "wide_tr": 0.15,
        "ttr": 0.35,
        "neutral_50_50": 0.5,
        "wide_bull_channel": 0.65,
        "wide_bear_channel": 0.65,
        "tight_bull_channel": 0.85,
        "tight_bear_channel": 0.85,
    }
    phase = np.array([phase_map.get(s, 0.4) for s in state], dtype=float)

    out = pd.DataFrame(index=regime.index)
    out["time_key"] = regime["time_key"].values
    out["pa_cycle_phase"] = phase
    out["pa_cycle_age"] = ages
    return out


def _volume_breakout_5m(bars: pd.DataFrame) -> pd.DataFrame:
    vol = bars["volume"].astype(float)
    vma = vol.rolling(20, min_periods=5).mean()
    rng = (bars["high"] - bars["low"]).astype(float)
    brng = rng.rolling(20, min_periods=5).mean()
    strong = rng > 1.2 * brng
    confirm = strong & (vol > 1.3 * vma)
    missing = strong & (vol < 0.9 * vma)
    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_breakout_volume_confirm"] = confirm.fillna(False).values
    out["pa_breakout_volume_missing"] = missing.fillna(False).values
    return out


def _prev_day_context(bars: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(bars["time_key"])
    dates = times.dt.date
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    opn = bars["open"].to_numpy(dtype=float)
    n = len(bars)

    daily: dict = {}
    for d, grp in bars.groupby(dates):
        idx = grp.index
        daily[d] = {
            "dir": np.sign(close[idx[-1]] - opn[idx[0]]),
            "range_pct": (float(np.max(high[idx]) - np.min(low[idx])) / max(close[idx[-1]], 1e-9)),
            "high": float(np.max(high[idx])),
            "low": float(np.min(low[idx])),
        }
    sorted_dates = sorted(daily.keys())
    prev_map: dict = {}
    for i in range(1, len(sorted_dates)):
        prev_map[sorted_dates[i]] = sorted_dates[i - 1]

    prev_dir = np.zeros(n)
    prev_rng = np.full(n, np.nan)
    prev_hi = np.full(n, np.nan)
    prev_lo = np.full(n, np.nan)
    fail_up = np.zeros(n, dtype=bool)
    fail_dn = np.zeros(n, dtype=bool)
    darr = dates.to_numpy()
    for i in range(n):
        d = darr[i]
        if d not in prev_map:
            continue
        p = daily[prev_map[d]]
        prev_dir[i] = p["dir"]
        prev_rng[i] = p["range_pct"]
        prev_hi[i] = p["high"]
        prev_lo[i] = p["low"]
        if np.isfinite(p["high"]) and high[i] > p["high"] and close[i] < p["high"]:
            fail_up[i] = True
        if np.isfinite(p["low"]) and low[i] < p["low"] and close[i] > p["low"]:
            fail_dn[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_prev_day_dir"] = prev_dir
    out["pa_prev_day_range_pct"] = prev_rng
    out["pa_prev_day_high"] = prev_hi
    out["pa_prev_day_low"] = prev_lo
    out["pa_prev_day_hl_fail_up"] = fail_up
    out["pa_prev_day_hl_fail_down"] = fail_dn
    return out


def _opening_patterns(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Gap-open flags and opening fade / reversal (causal).

    *pa_gap_open_flag*: True on every bar of days that opened with a gap vs prior session close.
    *pa_opening_reversal*: During the first *OPEN_WIN* bars after the open, before 10:00:
      gap-up fade: session low so far trades through prior close, current close below the open;
      gap-down fade: session high so far trades through prior close, current close above the open.
    """
    times = pd.to_datetime(bars["time_key"])
    dates = times.dt.date
    bt = times.dt.time
    close = bars["close"].to_numpy(dtype=float)
    opn = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(bars)
    darr = dates.to_numpy()

    last_close_by_day: dict = {}
    for d, grp in bars.groupby(dates):
        last_close_by_day[d] = float(close[grp.index[-1]])
    sorted_dates = sorted(last_close_by_day.keys())
    pdc_for_day: dict = {}
    for k in range(1, len(sorted_dates)):
        pdc_for_day[sorted_dates[k]] = last_close_by_day[sorted_dates[k - 1]]

    OPEN_WIN = 6
    CUTOFF = time(10, 0)

    rev = np.zeros(n, dtype=bool)
    gap_flag = np.zeros(n, dtype=bool)

    cur_d = None
    day_start_i = 0
    run_hi = run_lo = np.nan
    open0 = pdc = np.nan
    gap_up = gap_dn = False

    for i in range(n):
        d = darr[i]
        t = bt.iloc[i]

        if d != cur_d:
            cur_d = d
            day_start_i = i
            run_hi = float(high[i])
            run_lo = float(low[i])
            open0 = float(opn[i])
            pdc = float(pdc_for_day[d]) if d in pdc_for_day else np.nan
            gap_up = gap_dn = False
            if np.isfinite(pdc) and np.isfinite(open0):
                tol = 1e-4 * max(abs(pdc), 1.0)
                if open0 > pdc + tol:
                    gap_up = True
                elif open0 < pdc - tol:
                    gap_dn = True
        else:
            run_hi = max(run_hi, float(high[i]))
            run_lo = min(run_lo, float(low[i]))

        bars_since = i - day_start_i
        gap_today = gap_up or gap_dn
        if gap_today:
            gap_flag[i] = True

        if (
            gap_today
            and t < CUTOFF
            and 0 <= bars_since < OPEN_WIN
            and np.isfinite(pdc)
            and np.isfinite(open0)
        ):
            if gap_up and run_lo < pdc and close[i] < open0:
                rev[i] = True
            elif gap_dn and run_hi > pdc and close[i] > open0:
                rev[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_opening_reversal"] = rev
    out["pa_gap_open_flag"] = gap_flag
    return out


# ── Causal opening range … leading indicators ──

def _causal_opening_range(df: pd.DataFrame, daily_atr: pd.Series) -> pd.DataFrame:
    """
    OR high/low are RUNNING values: at bar t, OR_high = max(high[9:30..t]).
    Outputs compressed, normalized features instead of absolute prices.
    """
    times = pd.to_datetime(df["time_key"])
    dates = times.dt.date
    bar_time = times.dt.time

    dates_arr = dates.to_numpy()
    bar_time_arr = bar_time.to_numpy()

    n = len(df)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    or_range = np.full(n, np.nan)
    in_or = np.zeros(n, dtype=bool)

    cur_date = None
    running_high = np.nan
    running_low = np.nan

    for i in range(n):
        d = dates_arr[i]
        t = bar_time_arr[i]

        if d != cur_date:
            cur_date = d
            running_high = np.nan
            running_low = np.nan

        if _OR_START_TIME <= t < _OR_END_TIME:
            in_or[i] = True
            running_high = highs[i] if np.isnan(running_high) else max(running_high, highs[i])
            running_low = lows[i] if np.isnan(running_low) else min(running_low, lows[i])

        or_high[i] = running_high
        or_low[i] = running_low
        or_range[i] = running_high - running_low if np.isfinite(running_high) else np.nan

    vol_mean = pd.Series(volume).rolling(20, min_periods=1).mean().to_numpy()
    after_or = (~in_or) & (bar_time >= _OR_END_TIME).to_numpy()
    breakout_up = after_or & (close_arr > or_high) & np.isfinite(or_high)
    breakout_down = after_or & (close_arr < or_low) & np.isfinite(or_low)
    vol_breakout = (breakout_up | breakout_down) & (volume > vol_mean)

    atr_daily = daily_atr.groupby(dates).transform("last").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(atr_daily > 0, or_range / atr_daily, np.nan)

    result = pd.DataFrame(index=df.index)
    result["or_breakout_up"] = breakout_up.astype(np.int8)
    result["or_breakout_down"] = breakout_down.astype(np.int8)
    result["or_volume_breakout"] = vol_breakout.astype(np.int8)
    result["or_vs_atr_ratio"] = ratio
    return result


# ─────────────────────────────────────────────────────────────────
# 1. Causal Swing Detection (delayed confirmation)
# ─────────────────────────────────────────────────────────────────

def _causal_swings(bars: pd.DataFrame, confirm_len: int = 5):
    """
    Detect swing highs/lows with trailing-only confirmation.

    A swing high at bar j is confirmed at bar j+confirm_len when
    high[j] == max(high[j-confirm_len : j+1]) AND
    all(high[j+1:j+confirm_len+1] < high[j]).

    Returns two lists: [(confirm_bar, swing_bar, value), ...]
    At any bar i, only swings with confirm_bar <= i are available.
    """
    n = len(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    swing_highs: list[tuple[int, int, float]] = []
    swing_lows: list[tuple[int, int, float]] = []
    if n < 2 * confirm_len + 1:
        return swing_highs, swing_lows

    hw = sliding_window_view(high, confirm_len)
    lw = sliding_window_view(low, confirm_len)
    c = np.arange(confirm_len, n - confirm_len, dtype=np.int64)
    left_max = hw[c - confirm_len].max(axis=1)
    right_max = hw[c + 1].max(axis=1)
    left_min = lw[c - confirm_len].min(axis=1)
    right_min = lw[c + 1].min(axis=1)
    hc = high[c]
    lc = low[c]
    sh = hc > left_max
    sh &= hc > right_max
    sl = lc < left_min
    sl &= lc < right_min
    if np.any(sh):
        i_conf = c[sh] + confirm_len
        cand = c[sh]
        swing_highs = list(zip(i_conf.tolist(), cand.tolist(), hc[sh].tolist()))
    if np.any(sl):
        i_conf = c[sl] + confirm_len
        cand = c[sl]
        swing_lows = list(zip(i_conf.tolist(), cand.tolist(), lc[sl].tolist()))

    return swing_highs, swing_lows


def _coalesce_causal_swings(
    bars: pd.DataFrame,
    confirm_len: int,
    swing_highs: list[tuple[int, int, float]] | None,
    swing_lows: list[tuple[int, int, float]] | None,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    if swing_highs is None and swing_lows is None:
        return _causal_swings(bars, confirm_len)
    if swing_highs is None or swing_lows is None:
        raise ValueError("swing_highs and swing_lows must both be None or both provided")
    return swing_highs, swing_lows


# ─────────────────────────────────────────────────────────────────
# 2. Causal Stops (based on trailing-confirmed swings)
# ─────────────────────────────────────────────────────────────────

def _causal_pa_stops(
    bars: pd.DataFrame,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """PA stops using delayed-confirmation swing points. Rule X03, X04."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    sh_confirm = np.array([c for c, _, _ in swing_highs], dtype=int) if swing_highs else np.array([], dtype=int)
    sh_val = np.array([v for _, _, v in swing_highs], dtype=float) if swing_highs else np.array([], dtype=float)
    sl_confirm = np.array([c for c, _, _ in swing_lows], dtype=int) if swing_lows else np.array([], dtype=int)
    sl_val = np.array([v for _, _, v in swing_lows], dtype=float) if swing_lows else np.array([], dtype=float)

    stop_long, stop_short = fill_causal_stops_fast(sl_confirm, sl_val, sh_confirm, sh_val, n)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_stop_long"] = stop_long
    out["pa_stop_short"] = stop_short
    return out


# ─────────────────────────────────────────────────────────────────
# 3. Causal Support/Resistance (SR01-SR14)
# ─────────────────────────────────────────────────────────────────

def _causal_support_resistance(
    bars: pd.DataFrame,
    atr: pd.Series,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """S/R from trailing-confirmed swings + round numbers + 50% retrace."""
    n = len(bars)
    close = bars["close"].to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    nearest_resist, nearest_support, sr_position, retrace_50, at_50_retrace, round_num_dist = (
        causal_support_resistance_fast(close, atr_arr, swing_highs, swing_lows, n)
    )

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_nearest_resist"] = nearest_resist
    out["pa_nearest_support"] = nearest_support
    out["pa_sr_position"] = sr_position
    out["pa_50pct_retrace_level"] = retrace_50
    out["pa_at_50pct_retrace"] = at_50_retrace
    out["pa_round_number_dist"] = round_num_dist
    return out


# ─────────────────────────────────────────────────────────────────
# 4. Causal Measured Move (M15-M16)
# ─────────────────────────────────────────────────────────────────

def _causal_measured_move(
    bars: pd.DataFrame,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """Measured move targets from trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    mm_up = np.full(n, np.nan)
    mm_down = np.full(n, np.nan)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        if len(confirmed_sl) >= 2 and len(confirmed_sh) >= 1:
            sl1_idx, sl1_v = confirmed_sl[-2]
            sl2_idx, sl2_v = confirmed_sl[-1]
            sh_v = None
            for j in range(len(confirmed_sh) - 1, -1, -1):
                if confirmed_sh[j][0] > sl1_idx:
                    sh_v = confirmed_sh[j][1]
                    break
            if sh_v is not None and sl2_idx > sl1_idx:
                leg1 = sh_v - sl1_v
                if leg1 > 0:
                    mm_up[i] = sl2_v + leg1

        if len(confirmed_sh) >= 2 and len(confirmed_sl) >= 1:
            sh1_idx, sh1_v = confirmed_sh[-2]
            sh2_idx, sh2_v = confirmed_sh[-1]
            sl_v = None
            for j in range(len(confirmed_sl) - 1, -1, -1):
                if confirmed_sl[j][0] > sh1_idx:
                    sl_v = confirmed_sl[j][1]
                    break
            if sl_v is not None and sh2_idx > sh1_idx:
                leg1 = sh1_v - sl_v
                if leg1 > 0:
                    mm_down[i] = sh2_v - leg1

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_mm_target_up"] = mm_up
    out["pa_mm_target_down"] = mm_down
    return out


# ─────────────────────────────────────────────────────────────────
# 5. Causal Breakout Strength Score (replaces breakout_success)
#    Rules: K09-K21, K01-K08
# ─────────────────────────────────────────────────────────────────

def _causal_breakout_strength(bars: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """
    Score breakout quality at bar i using ONLY bar i and prior data.

    K09-K15 (strong breakout features):
      - Large body ratio (>0.6)
      - Body > 1.5× avg body
      - Close near high (bull) / low (bear)
      - Small wick ratio
      - Gap present (low > prev_high for bull)
      - Prior pressure accumulation
      - Breaks above recent range

    K16-K21 (weak breakout features):
      - Large wick
      - Small body
      - No volume
    """
    n = len(bars)
    opn = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    rng = np.maximum(high - low, 1e-12)
    body = np.abs(close - opn)
    body_ratio = body / rng
    avg_body20 = pd.Series(body).rolling(20, min_periods=5).mean().to_numpy(dtype=float)

    env_state = regime["pa_env_state"].astype(str).to_numpy()
    tr_state = np.isin(env_state, ["wide_tr", "ttr", "neutral_50_50"])

    bo_strength_up = np.zeros(n, dtype=float)
    bo_strength_down = np.zeros(n, dtype=float)
    is_bo_bar_up = np.zeros(n, dtype=bool)
    is_bo_bar_down = np.zeros(n, dtype=bool)

    for i in range(21, n):
        prev_hi = np.max(high[max(0, i - 20): i])
        prev_lo = np.min(low[max(0, i - 20): i])
        avg_b = avg_body20[i] if np.isfinite(avg_body20[i]) else 0.0

        # Bull breakout candidate
        if close[i] > prev_hi and close[i] > opn[i]:
            is_bo_bar_up[i] = True
            score = 0.0
            if body_ratio[i] > 0.6:
                score += 1.0
            if avg_b > 0 and body[i] > 1.5 * avg_b:
                score += 1.0
            if close[i] >= high[i] - 0.15 * rng[i]:
                score += 1.0
            upper_wick = high[i] - close[i]
            if upper_wick < 0.1 * rng[i]:
                score += 0.5
            if low[i] > prev_hi:
                score += 1.5
            elif low[i] > prev_hi - 0.1 * rng[i]:
                score += 0.5
            if close[i] > prev_hi + 0.5 * (prev_hi - prev_lo):
                score += 1.0
            bo_strength_up[i] = min(score, 6.0)

        # Bear breakout candidate
        if close[i] < prev_lo and close[i] < opn[i]:
            is_bo_bar_down[i] = True
            score = 0.0
            if body_ratio[i] > 0.6:
                score += 1.0
            if avg_b > 0 and body[i] > 1.5 * avg_b:
                score += 1.0
            if close[i] <= low[i] + 0.15 * rng[i]:
                score += 1.0
            lower_wick = close[i] - low[i]
            if lower_wick < 0.1 * rng[i]:
                score += 0.5
            if high[i] < prev_lo:
                score += 1.5
            elif high[i] < prev_lo + 0.1 * rng[i]:
                score += 0.5
            if close[i] < prev_lo - 0.5 * (prev_hi - prev_lo):
                score += 1.0
            bo_strength_down[i] = min(score, 6.0)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_bo_strength_up"] = bo_strength_up
    out["pa_bo_strength_down"] = bo_strength_down
    out["pa_is_bo_bar_up"] = is_bo_bar_up
    out["pa_is_bo_bar_down"] = is_bo_bar_down
    return out


# ─────────────────────────────────────────────────────────────────
# 6. Causal Trailing Momentum (replaces follow-through)
#    Uses bars i-3..i instead of i+1..i+3. Rules P01-P11.
# ─────────────────────────────────────────────────────────────────

def _causal_trailing_momentum(bars: pd.DataFrame) -> pd.DataFrame:
    """Score trailing momentum quality over last 3 bars."""
    n = len(bars)
    opn = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    rng = np.maximum(high - low, 1e-12)
    body = np.abs(close - opn)
    body_ratio = body / rng

    trail_bull = np.zeros(n, dtype=float)
    trail_bear = np.zeros(n, dtype=float)

    for i in range(3, n):
        score_bull = 0.0
        score_bear = 0.0
        for j in range(i - 2, i + 1):
            if close[j] > opn[j] and body_ratio[j] > 0.5:
                score_bull += 1.0
            if close[j] >= high[j] - 0.2 * rng[j]:
                score_bull += 0.5

            if close[j] < opn[j] and body_ratio[j] > 0.5:
                score_bear += 1.0
            if close[j] <= low[j] + 0.2 * rng[j]:
                score_bear += 0.5

        if high[i] > high[i - 1] > high[i - 2]:
            score_bull += 0.5
        if low[i] < low[i - 1] < low[i - 2]:
            score_bear += 0.5

        trail_bull[i] = min(score_bull, 5.0)
        trail_bear[i] = min(score_bear, 5.0)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_trailing_momentum_bull"] = trail_bull
    out["pa_trailing_momentum_bear"] = trail_bear
    return out


# ─────────────────────────────────────────────────────────────────
# 7. Causal Double Top / Bottom (D01-D04, R06-R08)
# ─────────────────────────────────────────────────────────────────

def _causal_double_top_bottom(
    bars: pd.DataFrame,
    direction: np.ndarray,
    atr: pd.Series,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """Double top/bottom using trailing-confirmed swings."""
    n = len(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    double_top = np.zeros(n, dtype=bool)
    double_bottom = np.zeros(n, dtype=bool)
    dt_is_flag = np.zeros(n, dtype=bool)
    db_is_flag = np.zeros(n, dtype=bool)
    dt_mm = np.full(n, np.nan)
    db_mm = np.full(n, np.nan)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0
    prev_dt_pair: tuple[int, int] | None = None
    cache_iv_low = np.nan
    prev_db_pair: tuple[int, int] | None = None
    cache_iv_high = np.nan

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        cur_atr = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else 1.0

        if len(confirmed_sh) >= 2:
            idx_a, val_a = confirmed_sh[-2]
            idx_b, val_b = confirmed_sh[-1]
            pair_t = (idx_a, idx_b)
            if pair_t != prev_dt_pair:
                cache_iv_low = float(np.min(low[idx_a : idx_b + 1]))
                prev_dt_pair = pair_t
            if abs(val_a - val_b) < 0.3 * cur_atr and (idx_b - idx_a) >= 5:
                double_top[i] = True
                dt_mm[i] = cache_iv_low - (max(val_a, val_b) - cache_iv_low)
                if direction[min(i, len(direction) - 1)] == -1:
                    dt_is_flag[i] = True

        if len(confirmed_sl) >= 2:
            idx_a, val_a = confirmed_sl[-2]
            idx_b, val_b = confirmed_sl[-1]
            pair_b = (idx_a, idx_b)
            if pair_b != prev_db_pair:
                cache_iv_high = float(np.max(high[idx_a : idx_b + 1]))
                prev_db_pair = pair_b
            if abs(val_a - val_b) < 0.3 * cur_atr and (idx_b - idx_a) >= 5:
                double_bottom[i] = True
                db_mm[i] = cache_iv_high + (cache_iv_high - min(val_a, val_b))
                if direction[min(i, len(direction) - 1)] == 1:
                    db_is_flag[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_double_top"] = double_top
    out["pa_double_bottom"] = double_bottom
    out["pa_dt_is_flag"] = dt_is_flag
    out["pa_db_is_flag"] = db_is_flag
    out["pa_dt_mm_target"] = dt_mm
    out["pa_db_mm_target"] = db_mm
    return out


# ─────────────────────────────────────────────────────────────────
# 8. Causal Wedge / Overshoot / Channel triggers
#    W01-W07, S22-S26
# ─────────────────────────────────────────────────────────────────

def _causal_advanced_triggers(
    bars: pd.DataFrame,
    regime: pd.DataFrame,
    mag: pd.DataFrame,
    confirm_len: int = 3,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """Wedge, overshoot, channel triggers — all causal."""
    n = len(bars)
    opn = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    rng = np.maximum(high - low, 1e-12)
    body = np.abs(close - opn)
    body_ratio = body / rng

    env_state = regime["pa_env_state"].astype(str).to_numpy()
    env_dir = regime["pa_env_trend_dir"].to_numpy(dtype=int)
    bars_since_extreme = regime["pa_env_bars_since_extreme"].to_numpy(dtype=int)

    mag_base = mag["pa_is_mag_bar"].to_numpy(dtype=bool)
    ema20 = _ema(bars["close"], 20).to_numpy(dtype=float)
    mag_bull = mag_base & (low > ema20) & (close > opn) & (body_ratio > 0.6)
    mag_bear = mag_base & (high < ema20) & (close < opn) & (body_ratio > 0.6)

    # Causal wedge: use trailing-confirmed swings
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)
    wedge_up = np.zeros(n, dtype=bool)
    wedge_down = np.zeros(n, dtype=bool)
    if swing_lows:
        sl_confirm = np.array([t[0] for t in swing_lows], dtype=np.int64)
        sl_bar = np.array([t[1] for t in swing_lows], dtype=np.int64)
        sl_val = np.array([t[2] for t in swing_lows], dtype=np.float64)
        sl_count = np.searchsorted(sl_confirm, np.arange(n, dtype=np.int64), side="right")
        sl_mask = sl_count >= 3
        if np.any(sl_mask):
            k2 = sl_count[sl_mask] - 1
            k1 = k2 - 1
            k0 = k1 - 1
            a_i = sl_bar[k0]
            b_i = sl_bar[k1]
            c_i = sl_bar[k2]
            a_v = sl_val[k0]
            b_v = sl_val[k1]
            c_v = sl_val[k2]
            leg1 = np.maximum(1, b_i - a_i)
            leg3 = np.maximum(1, c_i - b_i)
            wedge_up[sl_mask] = (leg3 <= 0.7 * leg1) & (c_v <= b_v) & (b_v <= a_v)

    if swing_highs:
        sh_confirm = np.array([t[0] for t in swing_highs], dtype=np.int64)
        sh_bar = np.array([t[1] for t in swing_highs], dtype=np.int64)
        sh_val = np.array([t[2] for t in swing_highs], dtype=np.float64)
        sh_count = np.searchsorted(sh_confirm, np.arange(n, dtype=np.int64), side="right")
        sh_mask = sh_count >= 3
        if np.any(sh_mask):
            k2 = sh_count[sh_mask] - 1
            k1 = k2 - 1
            k0 = k1 - 1
            a_i = sh_bar[k0]
            b_i = sh_bar[k1]
            c_i = sh_bar[k2]
            a_v = sh_val[k0]
            b_v = sh_val[k1]
            c_v = sh_val[k2]
            leg1 = np.maximum(1, b_i - a_i)
            leg3 = np.maximum(1, c_i - b_i)
            wedge_down[sh_mask] = (leg3 <= 0.7 * leg1) & (c_v >= b_v) & (b_v >= a_v)

    # Channel overshoot (already causal in original — uses trailing roll + pending)
    overshoot_up = np.zeros(n, dtype=bool)
    overshoot_down = np.zeros(n, dtype=bool)
    roll_hi = pd.Series(high).rolling(20, min_periods=5).max().shift(1).to_numpy(dtype=float)
    roll_lo = pd.Series(low).rolling(20, min_periods=5).min().shift(1).to_numpy(dtype=float)
    pending_up = []
    pending_down = []
    for i in range(n):
        if np.isfinite(roll_hi[i]) and close[i] > roll_hi[i]:
            pending_up.append((i, roll_hi[i]))
        if np.isfinite(roll_lo[i]) and close[i] < roll_lo[i]:
            pending_down.append((i, roll_lo[i]))
        pending_up = [(i0, lvl) for i0, lvl in pending_up if i - i0 <= 5]
        pending_down = [(i0, lvl) for i0, lvl in pending_down if i - i0 <= 5]
        if any(close[i] < lvl for _, lvl in pending_up):
            overshoot_up[i] = True
            pending_up = []
        if any(close[i] > lvl for _, lvl in pending_down):
            overshoot_down[i] = True
            pending_down = []

    channel_state = np.isin(env_state, [
        "tight_bull_channel", "wide_bull_channel",
        "tight_bear_channel", "wide_bear_channel",
    ])
    channel_reversal_ready = channel_state & (bars_since_extreme >= 20)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_mag20_bull"] = mag_bull
    out["pa_mag20_bear"] = mag_bear
    out["pa_wedge_third_push_up"] = wedge_up
    out["pa_wedge_third_push_down"] = wedge_down
    out["pa_channel_overshoot_revert_up"] = overshoot_up
    out["pa_channel_overshoot_revert_down"] = overshoot_down
    out["pa_channel_reversal_ready"] = channel_reversal_ready
    return out


# ─────────────────────────────────────────────────────────────────
# 9. Causal TBTL (V09-V10)
# ─────────────────────────────────────────────────────────────────

def _causal_tbtl(
    bars: pd.DataFrame,
    confirm_len: int = 3,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """TBTL pattern using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    tbtl_up = np.zeros(n, dtype=bool)
    tbtl_down = np.zeros(n, dtype=bool)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0
    sh_win_start = 0
    sl_win_start = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        while sh_win_start < len(confirmed_sh) and i - confirmed_sh[sh_win_start][0] > 12:
            sh_win_start += 1
        while sl_win_start < len(confirmed_sl) and i - confirmed_sl[sl_win_start][0] > 12:
            sl_win_start += 1
        recent_sh_len = len(confirmed_sh) - sh_win_start
        recent_sl_len = len(confirmed_sl) - sl_win_start

        if recent_sh_len >= 2 and recent_sl_len >= 1:
            sh1 = confirmed_sh[-2]
            sl1 = confirmed_sl[-1]
            sh2 = confirmed_sh[-1]
            if sh1[0] < sl1[0] < sh2[0]:
                span = sh2[0] - sh1[0]
                if span <= 12 and sh2[1] < sh1[1]:
                    tbtl_down[i] = True

        if recent_sl_len >= 2 and recent_sh_len >= 1:
            sl1 = confirmed_sl[-2]
            sh1 = confirmed_sh[-1]
            sl2 = confirmed_sl[-1]
            if sl1[0] < sh1[0] < sl2[0]:
                span = sl2[0] - sl1[0]
                if span <= 12 and sl2[1] > sl1[1]:
                    tbtl_up[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_tbtl_correction_up"] = tbtl_up
    out["pa_tbtl_correction_down"] = tbtl_down
    return out


# ─────────────────────────────────────────────────────────────────
# 10. Causal Triangle (T01-T04)
# ─────────────────────────────────────────────────────────────────

def _causal_triangle(
    bars: pd.DataFrame,
    confirm_len: int = 3,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """Triangle patterns using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    converging = np.zeros(n, dtype=bool)
    expanding = np.zeros(n, dtype=bool)
    breakout_bias = np.zeros(n, dtype=float)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0
    sh_win_start = 0
    sl_win_start = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        while sh_win_start < len(confirmed_sh) and i - confirmed_sh[sh_win_start][0] > 20:
            sh_win_start += 1
        while sl_win_start < len(confirmed_sl) and i - confirmed_sl[sl_win_start][0] > 20:
            sl_win_start += 1
        recent_sh_len = len(confirmed_sh) - sh_win_start
        recent_sl_len = len(confirmed_sl) - sl_win_start

        if recent_sh_len >= 2 and recent_sl_len >= 2:
            sh1, sh2 = confirmed_sh[-2], confirmed_sh[-1]
            sl1, sl2 = confirmed_sl[-2], confirmed_sl[-1]

            highs_falling = sh2[1] < sh1[1]
            lows_rising = sl2[1] > sl1[1]
            highs_rising = sh2[1] > sh1[1]
            lows_falling = sl2[1] < sl1[1]

            if highs_falling and lows_rising:
                converging[i] = True
                breakout_bias[i] = 0.0
            if highs_rising and lows_falling:
                expanding[i] = True
            if not highs_falling and not highs_rising and lows_rising:
                converging[i] = True
                breakout_bias[i] = 1.0
            if highs_falling and not lows_rising and not lows_falling:
                converging[i] = True
                breakout_bias[i] = -1.0

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_triangle_converging"] = converging
    out["pa_triangle_expanding"] = expanding
    out["pa_triangle_breakout_bias"] = breakout_bias
    return out


# ─────────────────────────────────────────────────────────────────
# 11. Causal Head & Shoulders (H01-H03, V16-V17)
# ─────────────────────────────────────────────────────────────────

def _causal_head_shoulders(
    bars: pd.DataFrame,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """H&S using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    hs_top = np.zeros(n, dtype=bool)
    hs_bottom = np.zeros(n, dtype=bool)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        if len(confirmed_sh) >= 3:
            a_i, a_v = confirmed_sh[-3]
            b_i, b_v = confirmed_sh[-2]
            c_i, c_v = confirmed_sh[-1]
            if b_v > a_v and b_v > c_v and (c_i - a_i) <= 40:
                shoulder_diff = abs(a_v - c_v) / max(b_v - min(a_v, c_v), 1e-12)
                if shoulder_diff < 0.5:
                    hs_top[i] = True

        if len(confirmed_sl) >= 3:
            a_i, a_v = confirmed_sl[-3]
            b_i, b_v = confirmed_sl[-2]
            c_i, c_v = confirmed_sl[-1]
            if b_v < a_v and b_v < c_v and (c_i - a_i) <= 40:
                shoulder_diff = abs(a_v - c_v) / max(max(a_v, c_v) - b_v, 1e-12)
                if shoulder_diff < 0.5:
                    hs_bottom[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_head_shoulders_top"] = hs_top
    out["pa_head_shoulders_bottom"] = hs_bottom
    return out


# ─────────────────────────────────────────────────────────────────
# 12. Causal Parabolic Wedge (W06)
# ─────────────────────────────────────────────────────────────────

def _causal_parabolic_wedge(
    bars: pd.DataFrame,
    confirm_len: int = 3,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """Parabolic wedge using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _coalesce_causal_swings(bars, confirm_len, swing_highs, swing_lows)

    parabolic_up = np.zeros(n, dtype=bool)
    parabolic_down = np.zeros(n, dtype=bool)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        if len(confirmed_sh) >= 3:
            a_i, a_v = confirmed_sh[-3]
            b_i, b_v = confirmed_sh[-2]
            c_i, c_v = confirmed_sh[-1]
            if c_v > b_v > a_v:
                bars_ab = max(1, b_i - a_i)
                bars_bc = max(1, c_i - b_i)
                slope_ab = (b_v - a_v) / bars_ab
                slope_bc = (c_v - b_v) / bars_bc
                if slope_bc > slope_ab * 1.3 and bars_bc < bars_ab:
                    parabolic_up[i] = True

        if len(confirmed_sl) >= 3:
            a_i, a_v = confirmed_sl[-3]
            b_i, b_v = confirmed_sl[-2]
            c_i, c_v = confirmed_sl[-1]
            if c_v < b_v < a_v:
                bars_ab = max(1, b_i - a_i)
                bars_bc = max(1, c_i - b_i)
                slope_ab = (a_v - b_v) / bars_ab
                slope_bc = (b_v - c_v) / bars_bc
                if slope_bc > slope_ab * 1.3 and bars_bc < bars_ab:
                    parabolic_down[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_parabolic_wedge_up"] = parabolic_up
    out["pa_parabolic_wedge_down"] = parabolic_down
    return out


# ─────────────────────────────────────────────────────────────────
# 14. Structure Features (HH/HL/LH/LL, swing range, break)
# ─────────────────────────────────────────────────────────────────

def _structure_features(
    bars_5m: pd.DataFrame,
    atr_5m: pd.Series,
    confirm_len: int = 5,
    *,
    swing_highs: list[tuple[int, int, float]] | None = None,
    swing_lows: list[tuple[int, int, float]] | None = None,
) -> pd.DataFrame:
    """
    Market structure features using causal swings.
    Tracks HH/HL/LH/LL sequences, trend structure score, swing range.
    """
    n = len(bars_5m)
    close = bars_5m["close"].to_numpy(dtype=float)
    atr_arr = atr_5m.to_numpy(dtype=float)
    swing_highs, swing_lows = _coalesce_causal_swings(bars_5m, confirm_len, swing_highs, swing_lows)

    hh_count = np.zeros(n, dtype=float)
    hl_count = np.zeros(n, dtype=float)
    lh_count = np.zeros(n, dtype=float)
    ll_count = np.zeros(n, dtype=float)
    swing_range_atr = np.full(n, np.nan)
    structure_score = np.zeros(n, dtype=float)  # +: bullish, -: bearish
    structure_break_up = np.zeros(n, dtype=bool)
    structure_break_down = np.zeros(n, dtype=bool)
    leg_count = np.zeros(n, dtype=float)

    confirmed_sh = []
    confirmed_sl = []
    sh_ptr = 0
    sl_ptr = 0
    sh_leg_start = 0
    sl_leg_start = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            _, swing_idx, val = swing_highs[sh_ptr]
            confirmed_sh.append((swing_idx, val))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            _, swing_idx, val = swing_lows[sl_ptr]
            confirmed_sl.append((swing_idx, val))
            sl_ptr += 1

        cur_atr = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else 1e-6

        # Count HH/HL/LH/LL in last 5 swings
        recent_sh = confirmed_sh[-5:] if len(confirmed_sh) >= 2 else []
        recent_sl = confirmed_sl[-5:] if len(confirmed_sl) >= 2 else []

        hh_c = hl_c = lh_c = ll_c = 0
        for k in range(1, len(recent_sh)):
            if recent_sh[k][1] > recent_sh[k - 1][1]:
                hh_c += 1
            else:
                lh_c += 1
        for k in range(1, len(recent_sl)):
            if recent_sl[k][1] > recent_sl[k - 1][1]:
                hl_c += 1
            else:
                ll_c += 1

        hh_count[i] = hh_c
        hl_count[i] = hl_c
        lh_count[i] = lh_c
        ll_count[i] = ll_c

        # Trend structure score: HH+HL positive, LH+LL negative
        structure_score[i] = (hh_c + hl_c) - (lh_c + ll_c)

        # Swing range
        if confirmed_sh and confirmed_sl:
            last_sh = confirmed_sh[-1][1]
            last_sl = confirmed_sl[-1][1]
            swing_range_atr[i] = abs(last_sh - last_sl) / cur_atr

        # Structure break: close breaks above last confirmed swing high (bull)
        # or below last confirmed swing low (bear)
        if confirmed_sh and close[i] > confirmed_sh[-1][1]:
            if len(confirmed_sh) >= 2 and confirmed_sh[-1][1] < confirmed_sh[-2][1]:
                structure_break_up[i] = True
        if confirmed_sl and close[i] < confirmed_sl[-1][1]:
            if len(confirmed_sl) >= 2 and confirmed_sl[-1][1] > confirmed_sl[-2][1]:
                structure_break_down[i] = True

        # Leg count: total confirmed swings in recent window.
        while sh_leg_start < len(confirmed_sh) and i - confirmed_sh[sh_leg_start][0] > 40:
            sh_leg_start += 1
        while sl_leg_start < len(confirmed_sl) and i - confirmed_sl[sl_leg_start][0] > 40:
            sl_leg_start += 1
        leg_count[i] = (len(confirmed_sh) - sh_leg_start) + (len(confirmed_sl) - sl_leg_start)

    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    out["pa_struct_hh_count"] = hh_count
    out["pa_struct_hl_count"] = hl_count
    out["pa_struct_lh_count"] = lh_count
    out["pa_struct_ll_count"] = ll_count
    out["pa_struct_score"] = structure_score
    out["pa_struct_swing_range_atr"] = swing_range_atr
    out["pa_struct_break_up"] = structure_break_up
    out["pa_struct_break_down"] = structure_break_down
    out["pa_struct_leg_count"] = leg_count
    return out


# ─────────────────────────────────────────────────────────────────
# 15. Moving Average Relationship Features
# ─────────────────────────────────────────────────────────────────

def _ma_relationship_features(bars_5m: pd.DataFrame, atr_5m: pd.Series) -> pd.DataFrame:
    """
    MA compression (squeeze). Other legacy MA features (slopes, distances, fan) 
    have been removed in favor of Kalman and PA Composite features.
    """
    n = len(bars_5m)
    close = bars_5m["close"].astype(float)
    atr_arr = atr_5m.to_numpy(dtype=float)
    safe_atr = np.where(atr_arr > 1e-12, atr_arr, 1e-12)

    ema8 = _ema(close, 8).to_numpy(dtype=float)
    ema20 = _ema(close, 20).to_numpy(dtype=float)
    ema50 = _ema(close, 50).to_numpy(dtype=float)

    # MA compression: std of [ema8, ema20, ema50] / ATR
    ma_compress = np.zeros(n, dtype=float)
    ma_stack = np.column_stack((ema8, ema20, ema50))
    finite = np.isfinite(ma_stack).all(axis=1)
    valid = finite & (safe_atr > 0)
    if np.any(valid):
        ma_compress[valid] = np.std(ma_stack[valid], axis=1) / safe_atr[valid]

    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    out["pa_ma_compress"] = ma_compress
    return out


# ─────────────────────────────────────────────────────────────────
# 16. Volume Features
# ─────────────────────────────────────────────────────────────────

def _volume_features(bars_5m: pd.DataFrame, atr_5m: pd.Series) -> pd.DataFrame:
    """
    Volume analysis focused on extreme relative levels (RVOL), volume trends, 
    and significant climactic events.
    """
    n = len(bars_5m)
    close = bars_5m["close"].to_numpy(dtype=float)
    opn = bars_5m["open"].to_numpy(dtype=float)
    high = bars_5m["high"].to_numpy(dtype=float)
    low = bars_5m["low"].to_numpy(dtype=float)
    vol = bars_5m["volume"].to_numpy(dtype=float)

    vol_s = pd.Series(vol)

    # Relative volume (RVOL): current vol / SMA20(vol)
    vol_sma20 = vol_s.rolling(20, min_periods=5).mean().to_numpy(dtype=float)
    safe_vol_sma20 = np.where(vol_sma20 > 0, vol_sma20, 1.0)
    rvol = vol / safe_vol_sma20

    # Volume SMA5 / SMA20 ratio (volume trend)
    vol_sma5 = vol_s.rolling(5, min_periods=2).mean().to_numpy(dtype=float)
    vol_trend = np.where(safe_vol_sma20 > 0, vol_sma5 / safe_vol_sma20, 1.0)

    # Volume climax: vol > 3× SMA20 AND large body
    body = np.abs(close - opn)
    rng = np.maximum(high - low, 1e-12)
    body_ratio = body / rng
    vol_climax = (rvol > 3.0) & (body_ratio > 0.5)

    # Net Absorption (Bullish vs Bearish long wick absorption on high volume)
    # +1 if bullish absorption (buying tail), -1 if bearish absorption (selling tail), 0 otherwise
    upper_wick = high - np.maximum(opn, close)
    lower_wick = np.minimum(opn, close) - low
    absorption_bull = (rvol > 1.5) & (lower_wick > body * 1.5) & (body_ratio < 0.4)
    absorption_bear = (rvol > 1.5) & (upper_wick > body * 1.5) & (body_ratio < 0.4)
    net_absorption = absorption_bull.astype(float) - absorption_bear.astype(float)

    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    out["pa_vol_rvol"] = rvol
    out["pa_vol_trend"] = vol_trend
    out["pa_vol_climax"] = vol_climax
    out["pa_vol_net_absorption"] = net_absorption
    return out


# ─────────────────────────────────────────────────────────────────
# 14. HMM-style + GARCH-style volatility features (causal)
# ─────────────────────────────────────────────────────────────────

def _hmm_garch_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight state-space/volatility features without external deps.

    **Causality / OOF:** This is **not** ``hmmlearn`` EM + Viterbi and **not** ``arch`` full-sample
    ``fit()``. Everything is **online**: EWMs and **forward** recursions only; each time *t* uses
    returns and state summaries available at *t* (or *t−1* for variance updates). There is no
    backward pass and no refit on future bars, so **expanding-window L1 OOF is not invalidated**
    by a separate global HMM/GARCH calibration step.

    HMM-style block:
      - Soft state probabilities from trend-strength + volatility context
      - State confidence / persistence / transition pressure

    HSMM-style block:
      - Duration-aware state smoothing / hazard / persistence features (forward loop only)

    GARCH-style block:
      - Recursive variance estimate: var_t = w + a*r_{t-1}^2 + b*var_{t-1}
      - Volatility ratio / z-score / shock / vol-of-vol

    EGARCH-style block:
      - Log-variance recursion with asymmetric shock response
      - Regime-aware leverage / standardized stress features
    """
    close = df_1m["close"].to_numpy(dtype=float)
    n = len(close)
    eps = 1e-10

    log_ret = np.zeros(n, dtype=float)
    if n > 1:
        prev = np.maximum(close[:-1], eps)
        cur = np.maximum(close[1:], eps)
        log_ret[1:] = np.log(cur / prev)

    ret_s = pd.Series(log_ret)
    trend_mu = ret_s.ewm(span=20, adjust=False, min_periods=5).mean().fillna(0.0).to_numpy(dtype=float)
    trend_sd = ret_s.ewm(span=20, adjust=False, min_periods=5).std().fillna(0.0).to_numpy(dtype=float)
    trend_sd = np.where(trend_sd > eps, trend_sd, eps)
    trend_strength = trend_mu / trend_sd

    vol_ewm = ret_s.ewm(span=30, adjust=False, min_periods=5).std().fillna(0.0).to_numpy(dtype=float)
    vol_ewm = np.where(vol_ewm > eps, vol_ewm, eps)
    # Replaced bfill() with fillna(current_vol) to prevent leaking bar 20's mean backwards to bars 0-19
    vol_baseline_s = pd.Series(vol_ewm).ewm(span=120, adjust=False, min_periods=20).mean()
    vol_baseline = vol_baseline_s.fillna(pd.Series(vol_ewm)).to_numpy(dtype=float)
    vol_baseline = np.where(vol_baseline > eps, vol_baseline, eps)
    vol_z = np.clip((vol_ewm / vol_baseline) - 1.0, -4.0, 4.0)

    # HMM-like latent-state scores -> probabilities via softmax.
    # We define 6 states based on trend direction and volatility regime:
    # 0: Low Volatility Bull
    # 1: High Volatility Bull
    # 2: Low Volatility Bear
    # 3: High Volatility Bear
    # 4: Low Volatility Range
    # 5: High Volatility Range

    trend_bull = 1.25 * trend_strength
    trend_bear = -1.25 * trend_strength
    trend_range = -0.90 * np.abs(trend_strength)

    vol_high = 0.5 * vol_z
    vol_low = -0.5 * vol_z

    score_0 = trend_bull + vol_low
    score_1 = trend_bull + vol_high
    score_2 = trend_bear + vol_low
    score_3 = trend_bear + vol_high
    # Weak-trend boost for range states: without it, ties at t≈0 softmax to argmax=0 (bull_conv)
    # and range_conv/range_div never appear in market_state — Layer 2a then collapses.
    trend_abs = np.abs(trend_strength)
    weak = np.clip(0.65 - trend_abs, 0.0, None)
    range_boost = 0.85 * weak
    score_4 = trend_range + vol_low + range_boost
    score_5 = trend_range + vol_high + range_boost

    scores = np.column_stack([score_0, score_1, score_2, score_3, score_4, score_5])
    scores = np.clip(scores, -12.0, 12.0)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), eps)

    hmm_state = probs.argmax(axis=1)
    hmm_conf = probs.max(axis=1)
    hmm_transition_pressure = 1.0 - hmm_conf

    hmm_persist = np.zeros(n, dtype=float)
    run = 0
    prev_state = int(hmm_state[0]) if n > 0 else 2
    for i in range(n):
        cur_state = int(hmm_state[i])
        if i == 0 or cur_state != prev_state:
            run = 0
        else:
            run += 1
        hmm_persist[i] = min(run, 40) / 40.0
        prev_state = cur_state

    # HSMM-style duration-aware state smoothing.
    hsmm_switch_hazard = np.zeros(n, dtype=float)
    hsmm_duration_norm = np.zeros(n, dtype=float)
    hsmm_remaining_duration = np.zeros(n, dtype=float)
    hsmm_duration_percentile = np.zeros(n, dtype=float)
    expected_duration = np.asarray([24.0, 14.0, 24.0, 14.0, 18.0, 10.0], dtype=float)
    hsmm_run = 0
    prev_hsmm_state = int(hmm_state[0]) if n > 0 else 4
    hsmm_state = np.zeros(n, dtype=int)
    for i in range(n):
        adj_scores = scores[i].copy()
        if i > 0:
            prev_target = expected_duration[prev_hsmm_state]
            prev_duration = min((hsmm_run + 1) / max(prev_target, 1.0), 2.0)
            continue_bonus = 0.45 * max(1.0 - prev_duration, 0.0)
            overstay_penalty = 0.35 * max(prev_duration - 1.0, 0.0)
            adj_scores[prev_hsmm_state] += continue_bonus - overstay_penalty
        adj_scores = np.clip(adj_scores, -12.0, 12.0)
        adj_scores -= np.max(adj_scores)
        adj_exp = np.exp(adj_scores)
        adj_prob = adj_exp / max(float(adj_exp.sum()), eps)
        cur_state = int(np.argmax(adj_prob))
        if i == 0 or cur_state != prev_hsmm_state:
            hsmm_run = 0
        else:
            hsmm_run += 1
        duration_bars = float(hsmm_run + 1)
        duration_target = expected_duration[cur_state]
        duration_norm = min(duration_bars / max(duration_target, 1.0), 2.0)
        remaining_duration = max(duration_target - duration_bars, 0.0) / max(duration_target, 1.0)
        duration_percentile = 1.0 - np.exp(-duration_bars / max(duration_target, 1.0))
        switch_hazard = np.clip((1.0 - float(adj_prob[cur_state])) + 0.55 * max(duration_norm - 1.0, 0.0), 0.0, 1.0)
        hsmm_state[i] = cur_state
        hsmm_duration_norm[i] = duration_norm
        hsmm_remaining_duration[i] = remaining_duration
        hsmm_duration_percentile[i] = np.clip(duration_percentile, 0.0, 1.0)
        hsmm_switch_hazard[i] = switch_hazard
        prev_hsmm_state = cur_state

    # GARCH(1,1)-style recursion (Pseudo-MSGARCH dynamics controlled by HMM state).
    var = np.zeros(n, dtype=float)
    # Calculate seed variance using strictly past/current data to prevent leakage from bar 60 to bar 0
    seed_var = float(log_ret[1] ** 2) if n > 1 else 1e-6
    seed_var = max(seed_var, 1e-6)
    var[0] = seed_var

    for i in range(1, n):
        prev_r2 = log_ret[i - 1] * log_ret[i - 1]
        
        # --- 动态切换参数 (Pseudo-MSGARCH) ---
        # 1, 3, 5 对应 High Volatility (Bull, Bear, Range)
        # 0, 2, 4 对应 Low Volatility  (Bull, Bear, Range)
        if hmm_state[i - 1] in (1, 3, 5):  
            # 高波动状态：市场情绪敏感，对最新冲击 (alpha) 赋予更高权重，历史记忆 (beta) 衰减更快
            alpha_dyn = 0.15
            beta_dyn = 0.80
        else:                        
            # 低波动状态：市场平稳，对最新冲击不敏感，依赖长记忆 (更平滑)
            alpha_dyn = 0.05
            beta_dyn = 0.94
            
        omega_dyn = max(seed_var * (1.0 - alpha_dyn - beta_dyn), 1e-10)
        var[i] = omega_dyn + alpha_dyn * prev_r2 + beta_dyn * var[i - 1]

    garch_vol = np.sqrt(np.maximum(var, eps))
    # Replaced bfill() with fillna(current_garch_vol) to prevent leaking bar 20's moving average backwards
    garch_vol_ma_s = pd.Series(garch_vol).ewm(span=120, adjust=False, min_periods=20).mean()
    garch_vol_ma = garch_vol_ma_s.fillna(pd.Series(garch_vol)).to_numpy(dtype=float)
    garch_vol_ma = np.where(garch_vol_ma > eps, garch_vol_ma, eps)

    garch_shock = (log_ret * log_ret) / np.maximum(var, eps) - 1.0
    garch_shock = np.clip(garch_shock, -3.0, 6.0)
    garch_vol_of_vol = np.zeros(n, dtype=float)
    if n > 1:
        garch_vol_of_vol[1:] = np.abs(np.diff(garch_vol)) / garch_vol_ma[1:]

    # EGARCH-style recursion with asymmetric response to downside shocks.
    egarch_log_var = np.zeros(n, dtype=float)
    egarch_leverage_effect = np.zeros(n, dtype=float)
    egarch_log_var[0] = np.log(max(seed_var, eps))
    expected_abs_z = np.sqrt(2.0 / np.pi)
    for i in range(1, n):
        prev_sigma = np.sqrt(max(np.exp(egarch_log_var[i - 1]), eps))
        z_prev = log_ret[i - 1] / max(prev_sigma, eps)
        prev_state = hsmm_state[i - 1]
        high_vol_state = prev_state in (1, 3, 5)
        omega = -0.18 if high_vol_state else -0.30
        alpha = 0.16 if high_vol_state else 0.10
        beta = 0.88 if high_vol_state else 0.92
        gamma = -0.10 if high_vol_state else -0.06
        asym_term = gamma * z_prev
        egarch_log_var[i] = omega + beta * egarch_log_var[i - 1] + alpha * (abs(z_prev) - expected_abs_z) + asym_term
        egarch_log_var[i] = np.clip(egarch_log_var[i], np.log(1e-10), np.log(5e-2))
        egarch_leverage_effect[i] = asym_term
    egarch_vol = np.sqrt(np.maximum(np.exp(egarch_log_var), eps))
    ret_neg_sq = np.square(np.minimum(log_ret, 0.0))
    ret_pos_sq = np.square(np.maximum(log_ret, 0.0))
    downside_vol = np.sqrt(
        pd.Series(ret_neg_sq).ewm(span=30, adjust=False, min_periods=5).mean().fillna(0.0).to_numpy(dtype=float) + eps
    )
    upside_vol = np.sqrt(
        pd.Series(ret_pos_sq).ewm(span=30, adjust=False, min_periods=5).mean().fillna(0.0).to_numpy(dtype=float) + eps
    )
    egarch_downside_vol_ratio = np.clip(downside_vol / np.maximum(upside_vol, eps), 0.0, 6.0)
    egarch_std_residual = np.clip(log_ret / np.maximum(egarch_vol, eps), -6.0, 6.0)
    egarch_vol_asymmetry = np.clip(egarch_std_residual * egarch_leverage_effect, -6.0, 6.0)

    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values

    # HMM-style features
    out["pa_hmm_state"] = hmm_state
    out["pa_hmm_transition_pressure"] = hmm_transition_pressure

    # HSMM-style features
    out["pa_hsmm_duration_norm"] = hsmm_duration_norm
    out["pa_hsmm_remaining_duration"] = hsmm_remaining_duration
    out["pa_hsmm_switch_hazard"] = hsmm_switch_hazard
    out["pa_hsmm_duration_percentile"] = hsmm_duration_percentile

    # GARCH-style features
    out["pa_garch_vol"] = garch_vol
    out["pa_garch_shock"] = garch_shock
    out["pa_garch_vol_of_vol"] = garch_vol_of_vol

    # EGARCH-style features
    out["pa_egarch_leverage_effect"] = np.clip(egarch_leverage_effect, -2.0, 2.0)
    out["pa_egarch_downside_vol_ratio"] = egarch_downside_vol_ratio
    out["pa_egarch_vol_asymmetry"] = egarch_vol_asymmetry
    out["pa_egarch_std_residual"] = egarch_std_residual
    return out


# ─────────────────────────────────────────────────────────────────
# 15. Advanced Math & Statistical Models (Kalman, Hurst, Entropy, Jump)
# ─────────────────────────────────────────────────────────────────

def _kalman_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    1D Steady-State Kalman Filter tracking core trend.
    Generates: pa_kalman_mean, pa_kalman_residual, pa_kalman_velocity
    """
    close = df_1m["close"].to_numpy(dtype=float)
    n = len(close)
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if n == 0: return out
    
    kf_mean = np.zeros(n, dtype=float)
    kf_var = np.zeros(n, dtype=float)
    
    Q = 1e-4  # process noise
    R = 1e-2  # measurement noise
    
    kf_mean[0] = close[0]
    kf_var[0] = 1.0
    
    for i in range(1, n):
        prior_mean = kf_mean[i-1]
        prior_var = kf_var[i-1] + Q
        K = prior_var / (prior_var + R)  # Kalman gain
        kf_mean[i] = prior_mean + K * (close[i] - prior_mean)
        kf_var[i] = (1 - K) * prior_var
        
    out["pa_kalman_mean"] = kf_mean
    out["pa_kalman_residual"] = close - kf_mean
    out["pa_kalman_velocity"] = np.diff(kf_mean, prepend=kf_mean[0])
    return out


def _hurst_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Ratio approximation of the Hurst Exponent (Fractal dimension).
    >0.6 = Trend persisting, <0.4 = Mean reverting.
    Uses multi-window (20 and 60) with EMA smoothing to reduce estimation noise.
    """
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out
    
    s_close = df_1m["close"]
    log_ret = np.log(s_close / s_close.shift(1).bfill())
    ret_2 = log_ret + log_ret.shift(1).fillna(0)
    
    # 20m = micro trend, 60m = 1h trend
    for w in [20, 60]:
        var_1 = log_ret.rolling(w, min_periods=1).var().fillna(1e-8)
        var_2 = ret_2.rolling(w, min_periods=1).var().fillna(1e-8)
        
        hurst = 0.5 * np.log(np.maximum(var_2, 1e-8) / np.maximum(var_1 * 2, 1e-8)) / np.log(2) + 0.5
        hurst = np.clip(hurst, 0.0, 1.0)
        
        # Smooth with EMA to reduce noise
        span = 5 if w == 20 else 10
        hurst_smooth = pd.Series(hurst).ewm(span=span, adjust=False).mean()
        
        out[f"pa_hurst_{w}"] = hurst_smooth.values
        
    return out


def _entropy_features(df_1m: pd.DataFrame, window: int = 75) -> pd.DataFrame:
    """
    Shannon Entropy of 5-state return distribution over a rolling window.
    States: Huge Up, Up, Flat, Down, Huge Down
    Measures disorder/indecision vs consensus in the market with higher information density.
    Normalized to [0, 1] range (max entropy = log2(5)).
    """
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out
    
    ret = df_1m["close"].pct_change().fillna(0)
    
    # Dynamic volatility threshold for state classification
    vol = ret.rolling(window * 4, min_periods=1).std().fillna(1e-5).clip(lower=1e-5)
    
    # 5-state distribution mapping
    state_huge_dn = (ret < -1.5 * vol).astype(float)
    state_dn = ((ret >= -1.5 * vol) & (ret < -0.25 * vol)).astype(float)
    state_fl = ((ret >= -0.25 * vol) & (ret <= 0.25 * vol)).astype(float)
    state_up = ((ret > 0.25 * vol) & (ret <= 1.5 * vol)).astype(float)
    state_huge_up = (ret > 1.5 * vol).astype(float)
    
    # Calculate probabilities within the window
    p_huge_dn = state_huge_dn.rolling(window, min_periods=1).mean().clip(1e-8, 1.0)
    p_dn = state_dn.rolling(window, min_periods=1).mean().clip(1e-8, 1.0)
    p_fl = state_fl.rolling(window, min_periods=1).mean().clip(1e-8, 1.0)
    p_up = state_up.rolling(window, min_periods=1).mean().clip(1e-8, 1.0)
    p_huge_up = state_huge_up.rolling(window, min_periods=1).mean().clip(1e-8, 1.0)
    
    # Shannon Entropy formula
    entropy = - (
        p_huge_dn * np.log2(p_huge_dn) +
        p_dn * np.log2(p_dn) +
        p_fl * np.log2(p_fl) +
        p_up * np.log2(p_up) +
        p_huge_up * np.log2(p_huge_up)
    )
    
    # Normalize to [0, 1]
    out[f"pa_entropy_{window}"] = (entropy / np.log2(5)).values
    return out


def _jump_diffusion_features(df_1m: pd.DataFrame, window: int = 100, tail_window: int = 500) -> pd.DataFrame:
    """
    Merton Jump-Diffusion style tail-risk detection.
    Isolates jumps exceeding a dynamic rolling 99th percentile threshold to track tail risk intensity.
    Adaptively protects against black swans without over-triggering in volatile markets.
    """
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out
    
    ret = df_1m["close"].pct_change().fillna(0)
    abs_ret = np.abs(ret)
    
    roll_mean = ret.rolling(window, min_periods=1).mean().fillna(0)
    roll_std = ret.rolling(window, min_periods=1).std().fillna(0).clip(lower=1e-8)
    z_score = ((ret - roll_mean) / roll_std).fillna(0)
    
    # Adaptive threshold: 99th percentile of recent absolute returns, floored at 3 standard deviations
    # to avoid false positives during dead-flat regimes.
    roll_99th = abs_ret.rolling(tail_window, min_periods=1).quantile(0.99).fillna(1e-8)
    dynamic_thr = np.maximum(roll_99th, roll_std * 3.0)
    
    # Jump magnitude: how much it exceeds the threshold (normalized)
    jumps = np.where(abs_ret > dynamic_thr, (abs_ret - dynamic_thr) / dynamic_thr, 0.0)
    jump_intensity = pd.Series(jumps).ewm(span=10, adjust=False).mean()
    
    out["pa_jump_tail_risk"] = jump_intensity.values
    out["pa_jump_z_score"] = z_score.values
    return out


# ─────────────────────────────────────────────────────────────────
# 17. Realized Volatility System Features
# ─────────────────────────────────────────────────────────────────

def _realized_volatility_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    c = df_1m["close"].astype(float)
    h = df_1m["high"].astype(float)
    l = df_1m["low"].astype(float)
    o = df_1m["open"].astype(float)

    # rv_cc 多尺度
    ret = c.pct_change().fillna(0)
    out["pa_rv_cc_10"] = ret.rolling(10, min_periods=2).std().fillna(0)
    out["pa_rv_cc_20"] = ret.rolling(20, min_periods=2).std().fillna(0)

    # rv_parkinson
    hl_ratio = np.log(h / l.replace(0, np.nan)).fillna(0)
    parkinson = np.sqrt((1.0 / (4.0 * np.log(2.0))) * hl_ratio**2)
    out["pa_rv_parkinson_20"] = parkinson.rolling(20, min_periods=2).mean().fillna(0)

    # rv_garman_klass
    log_hl = np.log(h / l.replace(0, np.nan)).fillna(0)
    log_co = np.log(c / o.replace(0, np.nan)).fillna(0)
    # clip to avoid negative inside sqrt due to floating point error
    gk_var = (0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2).clip(lower=0)
    gk = np.sqrt(gk_var).fillna(0)
    out["pa_rv_gk_20"] = gk.rolling(20, min_periods=2).mean().fillna(0)

    # vol_term_slope
    out["pa_vol_term_slope"] = (out["pa_rv_cc_10"] - out["pa_rv_cc_20"]).fillna(0)

    # vol_estimator_ratio (Garman-Klass vs Close-to-Close)
    # Encodes whether volatility is driven by intraday action (GK) or overnight/gap gaps (CC)
    out["pa_vol_estimator_ratio"] = (out["pa_rv_gk_20"] / (out["pa_rv_cc_20"] + 1e-8)).fillna(0)

    # vol_of_vol
    out["pa_vol_of_vol_20"] = out["pa_rv_cc_20"].rolling(20, min_periods=2).std().fillna(0)

    # vol_momentum
    out["pa_vol_momentum"] = out["pa_rv_cc_20"].diff(5).fillna(0)

    return out


def _straddle_focused_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Vol / range structure features for straddle-centric models (all causal on 1m bars).

    ``pa_iv_rv_spread`` uses a bar-data proxy (range-based vol vs close-to-close RV), not exchange IV.
    """
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0:
        return out

    c = df_1m["close"].astype(float)
    h = df_1m["high"].astype(float)
    l = df_1m["low"].astype(float)
    v = df_1m["volume"].astype(float)
    ret = c.pct_change().fillna(0)

    if "pa_rv_cc_20" in df_1m.columns:
        rv20 = pd.to_numeric(df_1m["pa_rv_cc_20"], errors="coerce").fillna(0.0)
    else:
        rv20 = ret.rolling(20, min_periods=2).std().fillna(0.0)
    if "pa_rv_cc_10" in df_1m.columns:
        rv10 = pd.to_numeric(df_1m["pa_rv_cc_10"], errors="coerce").fillna(0.0)
    else:
        rv10 = ret.rolling(10, min_periods=2).std().fillna(0.0)
    if "pa_rv_parkinson_20" in df_1m.columns:
        pk20 = pd.to_numeric(df_1m["pa_rv_parkinson_20"], errors="coerce").fillna(0.0)
    else:
        hl_ratio = np.log(h / l.replace(0, np.nan)).fillna(0.0)
        pk20 = np.sqrt((1.0 / (4.0 * np.log(2.0))) * hl_ratio**2).rolling(20, min_periods=2).mean().fillna(0.0)

    out["pa_iv_rv_spread"] = ((pk20 - rv20) / (rv20 + 1e-8)).clip(-5.0, 5.0).to_numpy(dtype=np.float64)

    rv_mu = rv20.rolling(120, min_periods=20).mean()
    rv_sd = rv20.rolling(120, min_periods=20).std().replace(0.0, np.nan)
    out["pa_rv_zscore"] = ((rv20 - rv_mu) / (rv_sd + 1e-8)).fillna(0.0).to_numpy(dtype=np.float64)

    out["pa_rv_acceleration"] = rv20.diff(5).diff(5).fillna(0.0).to_numpy(dtype=np.float64)

    tp = (h + l + c) / 3.0
    tp_std = tp.rolling(20, min_periods=5).std()
    tp_ma = tp.rolling(20, min_periods=5).mean()
    bb_w = (4.0 * tp_std / (tp_ma.replace(0, np.nan) + 1e-8)).fillna(0.0)
    out["pa_bb_width_pctile"] = (
        bb_w.rolling(120, min_periods=20)
        .apply(lambda x: float(np.mean(x <= x[-1])) if len(x) > 0 else 0.5, raw=True)
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy(dtype=np.float64)
    )

    bar_rng = (h - l).astype(float)
    atr5 = bar_rng.rolling(5, min_periods=1).mean()
    atr20 = bar_rng.rolling(20, min_periods=1).mean()
    out["pa_atr_ratio_5_20"] = (atr5 / (atr20 + 1e-8)).fillna(1.0).clip(0.0, 5.0).to_numpy(dtype=np.float64)

    rv60 = ret.rolling(60, min_periods=10).std().fillna(0.0)
    out["pa_rv_term_slope"] = (rv10 - rv60).to_numpy(dtype=np.float64)

    med = rv20.rolling(60, min_periods=10).median()
    sig_s = (rv20 > med).astype(np.int64)
    if len(sig_s) == 0:
        out["pa_vol_regime_duration"] = np.zeros(len(df_1m), dtype=np.float64)
    else:
        first = int(sig_s.iloc[0])
        ch = sig_s.ne(sig_s.shift(fill_value=first)).cumsum()
        dur = sig_s.groupby(ch).cumcount() + 1
        out["pa_vol_regime_duration"] = (dur.astype(np.float64) / 200.0).clip(0.0, 1.0).to_numpy()

    n_exp = max(3, int(os.environ.get("PA_REALIZED_VS_EXPECTED_BARS", "10")))
    past_rng = (h - l).rolling(n_exp, min_periods=1).sum()
    atr_n = bar_rng.rolling(n_exp, min_periods=1).mean() * float(n_exp)
    out["pa_realized_vs_expected"] = (past_rng / (atr_n + 1e-8)).fillna(0.0).clip(0.0, 10.0).to_numpy(dtype=np.float64)

    vm = v.rolling(60, min_periods=10).mean()
    vs = v.rolling(60, min_periods=10).std().replace(0.0, np.nan)
    out["pa_volume_zscore"] = ((v - vm) / (vs + 1e-8)).fillna(0.0).clip(-6.0, 6.0).to_numpy(dtype=np.float64)

    c_roll = ret.rolling(20, min_periods=5).corr(v).fillna(0.0)
    out["pa_volume_price_corr"] = c_roll.diff(5).fillna(0.0).clip(-2.0, 2.0).to_numpy(dtype=np.float64)

    return out


# ─────────────────────────────────────────────────────────────────
# 18. Intraday Time Structure Features
# ─────────────────────────────────────────────────────────────────

def _intraday_time_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    times = pd.to_datetime(df_1m["time_key"])
    dates = times.dt.date
    
    vol = df_1m["volume"].astype(float)
    c = df_1m["close"].astype(float)

    # session_progress [0, 1] (assuming 9:30 to 16:00 is 390 mins)
    # Better than absolute minutes_to_close across varying day lengths
    open_times = times.dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    mins_from_open = (times - open_times).dt.total_seconds() / 60.0
    out["pa_time_session_progress"] = (mins_from_open / 390.0).clip(0, 1).values

    # from_open_return
    first_open = df_1m["open"].groupby(dates).transform("first").astype(float)
    out["pa_time_from_open_return"] = ((c - first_open) / (first_open + 1e-8)).fillna(0).values

    # vol_time_deviation
    daily_vol_mean = vol.groupby(dates).transform(lambda x: x.expanding().mean())
    out["pa_time_vol_time_deviation"] = (vol - daily_vol_mean).fillna(0).values

    return out


# ─────────────────────────────────────────────────────────────────
# 19. Realized Higher Moments
# ─────────────────────────────────────────────────────────────────

def _realized_moments_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    ret = df_1m["close"].astype(float).pct_change().fillna(0)

    out["pa_realized_skew_20"] = ret.rolling(20, min_periods=5).skew().fillna(0)
    out["pa_realized_kurt_20"] = ret.rolling(20, min_periods=5).kurt().fillna(0)
    out["pa_skew_change_10"] = out["pa_realized_skew_20"].diff(10).fillna(0)

    return out


# ─────────────────────────────────────────────────────────────────
# 20. Volume Microstructure Features
# ─────────────────────────────────────────────────────────────────

def _volume_microstructure_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    c = df_1m["close"].astype(float)
    v = df_1m["volume"].astype(float)
    h = df_1m["high"].astype(float)
    l = df_1m["low"].astype(float)
    times = pd.to_datetime(df_1m["time_key"])
    dates = times.dt.date

    # vwap_deviation
    typ_price = (h + l + c) / 3.0
    cum_pv = (typ_price * v).groupby(dates).cumsum()
    cum_v = v.groupby(dates).cumsum()
    vwap = (cum_pv / (cum_v + 1e-8)).fillna(c)
    out["pa_vwap_deviation"] = ((c - vwap) / (vwap + 1e-8)).fillna(0).values

    # vol_price_diverge (Correlation between price change and volume)
    ret = c.pct_change().fillna(0)
    out["pa_vol_price_diverge"] = ret.rolling(20, min_periods=5).corr(v).fillna(0).values

    # volume_acceleration
    v_ma = v.rolling(10, min_periods=2).mean()
    out["pa_volume_acceleration"] = (v.diff() / (v_ma + 1e-8)).fillna(0).values

    # amihud_illiq
    out["pa_amihud_illiq_20"] = (ret.abs() / (v + 1e-8)).rolling(20, min_periods=2).mean().fillna(0).values

    # obv_slope
    direction = np.sign(ret)
    obv = (v * direction).cumsum()
    out["pa_obv_slope_10"] = obv.diff(10).fillna(0).values

    return out


# ─────────────────────────────────────────────────────────────────
# 21. Wavelet Decomposition Approximation
# ─────────────────────────────────────────────────────────────────

def _wavelet_approx_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    c = df_1m["close"].astype(float)

    # Use EWMA to approximate low-pass (trend) and high-pass (noise) filters
    trend_20 = c.ewm(span=20, adjust=False).mean()
    noise_20 = c - trend_20

    # trend_ratio (Signal-to-Noise Ratio proxy)
    trend_var = trend_20.diff().abs().rolling(20, min_periods=2).mean()
    noise_var = noise_20.abs().rolling(20, min_periods=2).mean()
    out["pa_trend_ratio"] = (trend_var / (noise_var + 1e-8)).fillna(0).values

    # noise_ratio
    out["pa_noise_ratio"] = (noise_var / (trend_var + noise_var + 1e-8)).fillna(0).values

    # trend_slope
    out["pa_trend_slope"] = trend_20.diff(5).fillna(0).values

    return out


# ─────────────────────────────────────────────────────────────────
# 22. Hawkes Self-Excitation
# ─────────────────────────────────────────────────────────────────

def _hawkes_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values
    if len(df_1m) == 0: return out

    ret = df_1m["close"].astype(float).pct_change().fillna(0)
    abs_ret = ret.abs()

    # Identify jumps (extreme returns)
    roll_std = abs_ret.rolling(20, min_periods=2).std().fillna(0.001).clip(lower=1e-5)
    jump_events = (abs_ret > 3 * roll_std).astype(float) * abs_ret

    # hawkes_intensity: exponentially decaying sum of jumps
    intensity = jump_events.ewm(alpha=0.1, adjust=False).mean()
    out["pa_hawkes_intensity"] = intensity.values

    # hawkes_clustering: rolling count of jump events
    out["pa_hawkes_clustering_20"] = (jump_events > 0).rolling(20, min_periods=1).sum().fillna(0).values

    return out


def _compressed_pa_features(bars_5m: pd.DataFrame, pullback_feats: pd.DataFrame, regime_feats: pd.DataFrame, pressure_feats: pd.DataFrame, struct_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Compress granular PA features into 5 core continuous scores.
    """
    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    if len(bars_5m) == 0: return out

    # 1. pa_reversal_composite: Combo of pressure divergence and structural breaks
    # net_pressure is theoretically [-100, 100], so net_pressure/100 is in [-1, 1].
    # Apply np.tanh to net_pressure / 50.0 to smoothly map and squash extreme values to [-1, 1].
    # struct_break is bool (0 or 1), we weight it to match the compressed pressure.
    net_pressure = pressure_feats["pa_net_pressure"].fillna(0).values
    norm_pressure = np.tanh(net_pressure / 50.0) 
    struct_break_up = struct_feats["pa_struct_break_up"].fillna(False).astype(float).values
    struct_break_dn = struct_feats["pa_struct_break_down"].fillna(False).astype(float).values
    
    # Weight: 60% pressure component, 40% structural break component
    out["pa_reversal_composite"] = np.clip(
        (norm_pressure * 0.6) + (struct_break_up - struct_break_dn) * 0.4, -1.0, 1.0
    )

    # 2. pa_continuation_composite: Combo of trend direction, inertia, and pullback completion
    trend_dir = regime_feats["pa_env_trend_dir"].fillna(0).values
    inertia = regime_feats["pa_env_trend_inertia"].fillna(0.5).values
    # PB stage gives confidence: H1/H2 (1/2) is good continuation, >=3 starts to fail
    h_count = pullback_feats["pa_h_count"].fillna(0).values
    l_count = pullback_feats["pa_l_count"].fillna(0).values
    
    cont_score = np.zeros(len(bars_5m), dtype=float)
    for i in range(len(bars_5m)):
        if trend_dir[i] > 0:
            pb_discount = max(0, 1.0 - h_count[i] * 0.25)
            cont_score[i] = inertia[i] * pb_discount
        elif trend_dir[i] < 0:
            pb_discount = max(0, 1.0 - l_count[i] * 0.25)
            cont_score[i] = -inertia[i] * pb_discount
            
    out["pa_continuation_composite"] = cont_score

    # 3. pa_structure_clarity: Meta-feature measuring if structure is clean or messy
    # Less legs (e.g. 1-2) = clean trend; More legs (e.g. 5+) = messy chop.
    # struct_score tracks HH/HL counts vs LH/LL counts. We take its absolute magnitude.
    leg_count = struct_feats["pa_struct_leg_count"].fillna(0).values
    struct_score = struct_feats["pa_struct_score"].fillna(0).values
    
    # Map leg count to a clarity multiplier: 0-2 legs -> 1.0 (clean), 5+ legs -> near 0 (messy)
    leg_clarity = np.clip(1.0 - (leg_count / 8.0), 0.1, 1.0)
    # Normalize struct_score (assume max typical score is around 10)
    norm_struct = np.clip(np.abs(struct_score) / 10.0, 0.0, 1.0)
    
    out["pa_structure_clarity"] = leg_clarity * norm_struct

    # 4. pa_pressure_score: Forward the normalized net pressure directly
    out["pa_pressure_score"] = np.clip(net_pressure / 100.0, -1.0, 1.0)

    # 5. pa_pullback_stage: -4 to 4 (L4 to H4 simplified)
    out["pa_pullback_stage"] = np.clip(h_count - l_count, -4.0, 4.0)

    return out


def _pa_merged_htf_from_bars(
    bars_htf: pd.DataFrame,
    atr_htf: pd.Series,
    *,
    variant: str = "full",
) -> tuple[pd.DataFrame, list[str]]:
    """
    PA feature merge on a single OHLCV bar series (any resolution; use closed HTF resample + map for causality).

    *variant* ``full`` = include HMM/GARCH block; ``structure`` = omit it (lighter higher-TF stack).

    The rule/MA/volume/compressed blocks do **not** consume ``pa_hmm_*`` / ``pa_garch_*`` columns,
    so omitting HMM/GARCH does not create dependent NaN columns on the 15m path.
    """
    if variant not in ("full", "structure"):
        raise ValueError(f"variant must be 'full' or 'structure', got {variant!r}")
    direction_htf = _derive_direction_5m(bars_htf)

    pullback = _pullback_counting_5m(bars_htf, direction_htf)
    regime_raw = _market_regime_5m(bars_htf)
    twenty = _twenty_bar_rule_5m(bars_htf, direction_htf)
    pressure = _pressure_score_5m(bars_htf)

    regime = regime_raw.copy()
    for c in twenty.columns:
        if c != "time_key":
            regime[c] = twenty[c].values

    swing_highs_5, swing_lows_5 = _causal_swings(bars_htf, 5)

    struct_feats = _structure_features(bars_htf, atr_htf, swing_highs=swing_highs_5, swing_lows=swing_lows_5)
    ma_feats = _ma_relationship_features(bars_htf, atr_htf)
    vol_feats = _volume_features(bars_htf, atr_htf)
    hmm_garch_feats = _hmm_garch_features(bars_htf)

    compressed_feats = _compressed_pa_features(
        bars_htf, pullback, regime, pressure, struct_feats
    )
    all_htf_parts: list[pd.DataFrame] = [
        compressed_feats,
        ma_feats,
        vol_feats,
    ]
    if variant == "full":
        all_htf_parts.append(hmm_garch_feats)

    feature_cols: list[str] = []
    for part in all_htf_parts:
        for c in part.columns:
            if c != "time_key" and c not in feature_cols:
                feature_cols.append(c)

    merged_parts = [bars_htf[["time_key"]]]
    for part in all_htf_parts:
        part_no_time = part.drop(columns=["time_key"], errors="ignore")
        merged_parts.append(part_no_time)

    merged_htf = pd.concat(merged_parts, axis=1)
    merged_htf = merged_htf.loc[:, ~merged_htf.columns.duplicated()]
    return merged_htf, feature_cols


def _log_mtf_pa_summary_and_nan_warnings(df: pd.DataFrame) -> None:
    """One-line multi-TF column counts; warn on ``pa_15m_*`` / ``pa_5m_*`` columns that are all-NaN."""
    if (os.environ.get("PA_MTF_LOG_STATS", "1") or "1").strip().lower() in {"0", "false", "no", "off"}:
        log_counts = False
    else:
        log_counts = True
    c1 = [c for c in df.columns if c.startswith("pa_1m_")]
    c5 = [c for c in df.columns if c.startswith("pa_5m_")]
    c15 = [c for c in df.columns if c.startswith("pa_15m_")]
    if not (c1 or c5 or c15):
        return
    if log_counts:
        print(
            f"  [PA MTF] column counts: pa_1m={len(c1)} pa_5m={len(c5)} pa_15m={len(c15)}",
            flush=True,
        )
    for group, label in ((c5, "pa_5m"), (c15, "pa_15m")):
        for c in group:
            s = df[c]
            if s.notna().sum() == 0:
                print(f"  [PA MTF] WARNING: {label} column {c!r} is all-NaN", flush=True)


def _add_pa_features_multi_tf(
    df: pd.DataFrame,
    atr_series: pd.Series,
    tf_list: list[str],
) -> pd.DataFrame:
    """Multi-timeframe PA: ``pa_1m_*``, ``pa_5m_*``, ``pa_15m_*`` mapped causally onto 1m rows."""
    allowed = {"1min", "5min", "15min"}
    for t in tf_list:
        if t not in allowed:
            raise ValueError(
                f"PA_TIMEFRAMES entry {t!r} unsupported (allowed: {sorted(allowed)})"
            )
    result = df.copy()
    times_1m = pd.to_datetime(result["time_key"])
    last_ts = times_1m.max()

    or_df = _causal_opening_range(result, atr_series)
    result = pd.concat([result.reset_index(drop=True), or_df.reset_index(drop=True)], axis=1)

    if "1min" in tf_list:
        micro = _mtf_micro_1m_features(result)
        micro_only = [c for c in micro.columns if c != "time_key"]
        result = pd.concat([result, micro[micro_only]], axis=1)
        rv_1m = _realized_volatility_features(result[["time_key", "open", "high", "low", "close", "volume"]])
        tmp_rv = pd.concat(
            [result.reset_index(drop=True), rv_1m.drop(columns=["time_key"], errors="ignore")],
            axis=1,
        )
        sd = _straddle_focused_features(tmp_rv)
        sd_only = [c for c in sd.columns if c != "time_key"]
        result = pd.concat([result, sd[sd_only]], axis=1)

    for tf in tf_list:
        if tf == "1min":
            continue
        ohlc = result[["time_key", "open", "high", "low", "close", "volume"]].copy()
        bars = _resample_ohlcv_htf_closed(ohlc, tf)
        bars = _drop_incomplete_htf_rows(bars, last_ts)
        if bars.empty or len(bars) < 5:
            continue
        rng = bars["high"] - bars["low"]
        atr_h = rng.ewm(span=14, min_periods=5).mean()
        variant = "full" if tf == "5min" else "structure"
        merged, feat_cols = _pa_merged_htf_from_bars(bars, atr_h, variant=variant)
        tag = "5m" if tf == "5min" else "15m"
        pref = _prefix_pa_mtf_columns(merged[["time_key"] + feat_cols], tag)
        map_cols = [c for c in pref.columns if c != "time_key"]
        mapped = _map_htf_closed_to_1min(pref, times_1m, map_cols)
        result = pd.concat([result, mapped.reset_index(drop=True)], axis=1)

    _append_htf_regime_from_1m(result, times_1m)
    _log_mtf_pa_summary_and_nan_warnings(result)
    return result


# ─────────────────────────────────────────────────────────────────
# ORCHESTRATOR: add_pa_features()
# ─────────────────────────────────────────────────────────────────



# ── Orchestrator ──

def add_pa_features(
    df: pd.DataFrame, atr_series: pd.Series, timeframe: str = "1min",
) -> pd.DataFrame:
    """
    Compute all PA features with ZERO look-ahead bias.

    Unified causal PA feature stack (single source of truth).
    Same interface: 1-min input DataFrame, returns DataFrame with pa_* columns.

    Env ``PA_TIMEFRAMES`` (e.g. ``1min,5min,15min``): multi-timeframe mode — produces
    ``pa_1m_*``, ``pa_5m_*``, ``pa_15m_*`` (5m full stack, 15m without HMM/GARCH) mapped
    with :func:`_map_htf_closed_to_1min`. In this mode *timeframe* is ignored.
    """
    _mtf_raw = (os.environ.get("PA_TIMEFRAMES") or "").strip()
    if _mtf_raw:
        return _add_pa_features_multi_tf(df, atr_series, _normalize_pa_mtf_list(_mtf_raw))

    # Same tqdm policy as core.trainers.lgbm_utils._tq (cannot import lgbm_utils: circular).
    _tqdm_file = getattr(sys, "__stderr__", None) or sys.stderr
    _pa_tqdm_on = (os.environ.get("PA_FEATURES_TQDM", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    _disable = os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}
    _force = os.environ.get("FORCE_TQDM", "").strip().lower() in {"1", "true", "yes"}
    _show_pa_bar = (
        _pa_tqdm_on
        and not _disable
        and (_tqdm_file.isatty() or _force)
    )
    try:
        from tqdm.auto import tqdm as _tqdm_pa  # type: ignore[import-not-found]
    except ImportError:
        _tqdm_pa = None  # type: ignore[assignment]
    pbar = None
    if _show_pa_bar and _tqdm_pa is not None:
        pbar = _tqdm_pa(
            total=8,
            desc="PA add_pa_features",
            unit="stage",
            leave=False,
            mininterval=0.25,
            file=_tqdm_file,
        )

    def _tick() -> None:
        if pbar is not None:
            pbar.update(1)

    try:
        result = df.copy()
        times_1m = pd.to_datetime(result["time_key"])

        # Opening Range (causal — running max/min)
        or_df = _causal_opening_range(result, atr_series)
        result = pd.concat([result.reset_index(drop=True), or_df.reset_index(drop=True)], axis=1)
        _tick()

        # Bar series used for PA rules: native 1m rows or resampled HTF (misnamed "5m" in legacy helpers).
        if timeframe == "1min":
            bars_pa = result.copy()
        else:
            ohlc = result[["time_key", "open", "high", "low", "close", "volume"]].copy()
            bars_pa = _resample_ohlcv_htf_closed(ohlc, str(timeframe))
            bars_pa = _drop_incomplete_htf_rows(bars_pa, times_1m.max())
        if bars_pa.empty:
            return result

        rng_pa = bars_pa["high"] - bars_pa["low"]
        atr_pa = rng_pa.ewm(span=14, min_periods=5).mean()

        direction_pa = _derive_direction_5m(bars_pa)
        _tick()

        pullback = _pullback_counting_5m(bars_pa, direction_pa)
        regime_raw = _market_regime_5m(bars_pa)
        twenty = _twenty_bar_rule_5m(bars_pa, direction_pa)
        pressure = _pressure_score_5m(bars_pa)
        _tick()

        regime = regime_raw.copy()
        for c in twenty.columns:
            if c != "time_key":
                regime[c] = twenty[c].values

        swing_highs_5, swing_lows_5 = _causal_swings(bars_pa, 5)
        _tick()

        # ── Feature groups merged into training frame ──
        struct_feats = _structure_features(
            bars_pa, atr_pa, swing_highs=swing_highs_5, swing_lows=swing_lows_5
        )
        ma_feats = _ma_relationship_features(bars_pa, atr_pa)
        vol_feats = _volume_features(bars_pa, atr_pa)
        hmm_garch_feats = _hmm_garch_features(bars_pa)
        _tick()

        # ── Compute 1m specific features directly on result ──
        realized_vol_feats = _realized_volatility_features(result)
        _tmp_rv = pd.concat(
            [result.reset_index(drop=True), realized_vol_feats.drop(columns=["time_key"], errors="ignore")],
            axis=1,
        )
        straddle_feats = _straddle_focused_features(_tmp_rv)
        intraday_time_feats = _intraday_time_features(result)
        realized_moments_feats = _realized_moments_features(result)
        vol_microstructure_feats = _volume_microstructure_features(result)
        _pa_light_1m = (os.environ.get("PA_LIGHT_1M_EXTRAS", "").strip().lower() in {"1", "true", "yes"})
        if _pa_light_1m:
            print(
                "  [PA] PA_LIGHT_1M_EXTRAS=1: skipping wavelet/hawkes/kalman/hurst/entropy/jump (faster prep)",
                flush=True,
            )
            _tm = pd.DataFrame({"time_key": result["time_key"].values}, index=result.index)
            wavelet_approx_feats = _tm.copy()
            hawkes_feats = _tm.copy()
            kalman_feats = _tm.copy()
            hurst_feats = _tm.copy()
            entropy_feats = _tm.copy()
            jump_feats = _tm.copy()
        else:
            wavelet_approx_feats = _wavelet_approx_features(result)
            hawkes_feats = _hawkes_features(result)
            kalman_feats = _kalman_features(result)
            hurst_feats = _hurst_features(result)
            entropy_feats = _entropy_features(result)
            jump_feats = _jump_diffusion_features(result)

        new_1m_parts = [
            realized_vol_feats.drop(columns=["time_key"], errors="ignore"),
            straddle_feats.drop(columns=["time_key"], errors="ignore"),
            intraday_time_feats.drop(columns=["time_key"], errors="ignore"),
            realized_moments_feats.drop(columns=["time_key"], errors="ignore"),
            vol_microstructure_feats.drop(columns=["time_key"], errors="ignore"),
            wavelet_approx_feats.drop(columns=["time_key"], errors="ignore"),
            hawkes_feats.drop(columns=["time_key"], errors="ignore"),
            kalman_feats.drop(columns=["time_key"], errors="ignore"),
            hurst_feats.drop(columns=["time_key"], errors="ignore"),
            entropy_feats.drop(columns=["time_key"], errors="ignore"),
            jump_feats.drop(columns=["time_key"], errors="ignore"),
        ]
        result = pd.concat([result] + new_1m_parts, axis=1)
        _tick()

        # ── Compress PA-bar features ──
        compressed_feats = _compressed_pa_features(
            bars_pa, pullback, regime, pressure, struct_feats
        )

        # ── Merge all PA-bar feature tables ──
        all_pa_parts = [
            compressed_feats,
            ma_feats, vol_feats, hmm_garch_feats,
        ]

        feature_cols = []
        for part in all_pa_parts:
            for c in part.columns:
                if c != "time_key" and c not in feature_cols:
                    feature_cols.append(c)

        merged_pa_parts = [bars_pa[["time_key"]]]
        for part in all_pa_parts:
            part_no_time = part.drop(columns=["time_key"], errors="ignore")
            merged_pa_parts.append(part_no_time)

        # Avoid DataFrame fragmentation warning by concatenating all new columns at once
        merged_pa = pd.concat(merged_pa_parts, axis=1)

        # Ensure no duplicate columns from potential overlaps in parts (other than time_key)
        merged_pa = merged_pa.loc[:, ~merged_pa.columns.duplicated()]

        # Map HTF → 1m (closed buckets only; no ffill from in-progress higher-TF bar).
        if timeframe == "1min":
            # Avoid DataFrame fragmentation warning by concatenating all new columns at once
            merged_pa_subset = merged_pa[feature_cols]
            result = pd.concat([result, merged_pa_subset], axis=1)
        else:
            mapped = _map_htf_closed_to_1min(merged_pa, times_1m, feature_cols)
            result = pd.concat([result, mapped.reset_index(drop=True)], axis=1)
        _tick()

        _append_htf_regime_from_1m(result, times_1m)
        _tick()

        return result
    finally:
        if pbar is not None:
            pbar.close()


def add_all_pa_features(
    df: pd.DataFrame, atr_series: pd.Series, timeframe: str = "1min",
) -> pd.DataFrame:
    """Backward-compatible alias — use :func:`add_pa_features`."""
    return add_pa_features(df, atr_series, timeframe=timeframe)

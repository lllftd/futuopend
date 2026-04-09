"""
Price Action (PA) — unified causal rules (single module).

Public API: :func:`add_pa_features`
(aliases: ``add_all_pa_features``, ``add_causal_pa_features``).
All features are computed with the causal / no–look-ahead pipeline.
"""


from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


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


def _resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 5-min bars, keyed by each 5-min period start."""
    return _resample_ohlcv(df, "5min")


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


def _map_5min_to_1min(
    features_5m: pd.DataFrame,
    original_index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    """Forward-fill 5-min features onto the 1-min index."""
    features_5m = features_5m.set_index("time_key")[columns]
    mapped = features_5m.reindex(original_index, method="ffill")
    return mapped.reset_index(drop=True)


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
    h_count = np.zeros(n, dtype=int)
    l_count = np.zeros(n, dtype=int)

    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)

    cur_h = 0
    cur_l = 0
    in_pullback_up = False
    in_pullback_down = False

    for i in range(1, n):
        if direction[i] == 1:
            if lows[i] < lows[i - 1]:
                in_pullback_up = True
                cur_h = 0
            if in_pullback_up and highs[i] > highs[i - 1]:
                cur_h += 1
            if cur_h >= 1 and highs[i] > highs[i - 1] and not (lows[i] < lows[i - 1]):
                pass
            h_count[i] = cur_h
            # if cur_h >= 4:
            #     in_pullback_up = False
            #     cur_h = 0
            cur_l = 0
            in_pullback_down = False
        elif direction[i] == -1:
            if highs[i] > highs[i - 1]:
                in_pullback_down = True
                cur_l = 0
            if in_pullback_down and lows[i] < lows[i - 1]:
                cur_l += 1
            l_count[i] = cur_l
            # if cur_l >= 4:
            #     in_pullback_down = False
            #     cur_l = 0
            cur_h = 0
            in_pullback_up = False
        else:
            cur_h = 0
            cur_l = 0
            in_pullback_up = False
            in_pullback_down = False

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

    above_count = 0
    below_count = 0
    is_mag = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(ema[i]):
            above_count = 0
            below_count = 0
            continue

        bar_above = lows[i] > ema[i]
        bar_below = highs[i] < ema[i]

        if bar_above:
            above_count += 1
            below_count = 0
        elif bar_below:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0

        if (bar_above and above_count >= 20) or (bar_below and below_count >= 20):
            if i > 0:
                prev_above = lows[i - 1] > ema[i - 1] if not np.isnan(ema[i - 1]) else False
                prev_below = highs[i - 1] < ema[i - 1] if not np.isnan(ema[i - 1]) else False
                if bar_above and above_count == 20:
                    is_mag[i] = True
                elif bar_below and below_count == 20:
                    is_mag[i] = True

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
    is_ii = is_inside & is_inside.shift(1).fillna(False).astype(bool)

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
# 1i. Measured Move (MM) Targets — on 5-min pivots
# ---------------------------------------------------------------------------

def _measured_move_5m(bars: pd.DataFrame, pivot_len: int = 5) -> pd.DataFrame:
    n = len(bars)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)

    mm_up = np.full(n, np.nan)
    mm_down = np.full(n, np.nan)

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    win = 2 * pivot_len + 1
    for i in range(pivot_len, n - pivot_len):
        window_h = highs[i - pivot_len: i + pivot_len + 1]
        window_l = lows[i - pivot_len: i + pivot_len + 1]
        if highs[i] == window_h.max():
            swing_highs.append((i, highs[i]))
        if lows[i] == window_l.min():
            swing_lows.append((i, lows[i]))

    swing_lows_arr = np.array(swing_lows) if swing_lows else np.empty((0, 2))
    swing_highs_arr = np.array(swing_highs) if swing_highs else np.empty((0, 2))

    for i in range(n):
        if len(swing_lows_arr) < 2 or len(swing_highs_arr) < 1:
            continue

        idx_l = np.searchsorted(swing_lows_arr[:, 0], i, side='right')
        idx_h = np.searchsorted(swing_highs_arr[:, 0], i, side='right')
        
        relevant_lows = swing_lows_arr[:idx_l]
        relevant_highs = swing_highs_arr[:idx_h]

        # upward MM target
        if len(relevant_lows) >= 2 and len(relevant_highs) >= 1:
            sl1 = relevant_lows[-2]
            sh = None
            for j in range(len(relevant_highs)-1, -1, -1):
                if relevant_highs[j, 0] > sl1[0]:
                    sh = relevant_highs[j]
                    break
            sl2 = relevant_lows[-1]
            if sh is not None and sl2[0] > sh[0]:
                leg1 = sh[1] - sl1[1]
                if leg1 > 0:
                    mm_up[i] = sl2[1] + leg1

        # downward MM target
        if len(relevant_highs) >= 2 and len(relevant_lows) >= 1:
            sh1 = relevant_highs[-2]
            sl = None
            for j in range(len(relevant_lows)-1, -1, -1):
                if relevant_lows[j, 0] > sh1[0]:
                    sl = relevant_lows[j]
                    break
            sh2 = relevant_highs[-1]
            if sl is not None and sh2[0] > sl[0]:
                leg1 = sh1[1] - sl[1]
                if leg1 > 0:
                    mm_down[i] = sh2[1] - leg1

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_mm_target_up"] = mm_up
    out["pa_mm_target_down"] = mm_down
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
    opn = bars_5m["open"].to_numpy(dtype=float)

    ema20 = _ema(bars_5m["close"], 20).to_numpy(dtype=float)
    n = len(bars_5m)
    direction = np.zeros(n, dtype=int)
    for i in range(n):
        if np.isnan(ema20[i]):
            continue
        if close[i] > ema20[i]:
            direction[i] = 1
        elif close[i] < ema20[i]:
            direction[i] = -1
    return direction


# ---------------------------------------------------------------------------
# PA stop helpers — recent swing HL / LH
# ---------------------------------------------------------------------------

def _pa_stops(bars_5m: pd.DataFrame, pivot_len: int = 5) -> pd.DataFrame:
    n = len(bars_5m)
    highs = bars_5m["high"].to_numpy(dtype=float)
    lows = bars_5m["low"].to_numpy(dtype=float)

    recent_hl = np.full(n, np.nan)
    recent_lh = np.full(n, np.nan)

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    for i in range(pivot_len, n - pivot_len):
        wh = highs[i - pivot_len: i + pivot_len + 1]
        wl = lows[i - pivot_len: i + pivot_len + 1]
        if highs[i] == wh.max():
            swing_highs.append((i, highs[i]))
        if lows[i] == wl.min():
            swing_lows.append((i, lows[i]))

    swing_high_idx = np.array([idx for idx, _ in swing_highs], dtype=int) if swing_highs else np.array([], dtype=int)
    swing_high_val = np.array([v for _, v in swing_highs], dtype=float) if swing_highs else np.array([], dtype=float)
    swing_low_idx = np.array([idx for idx, _ in swing_lows], dtype=int) if swing_lows else np.array([], dtype=int)
    swing_low_val = np.array([v for _, v in swing_lows], dtype=float) if swing_lows else np.array([], dtype=float)

    for i in range(n):
        # Recent higher-low for long stops
        if len(swing_low_idx) > 0:
            right_idx = np.searchsorted(swing_low_idx, i) - 1
            if right_idx >= 0 and swing_low_idx[right_idx] >= i - 40:
                recent_hl[i] = swing_low_val[right_idx]

        # Recent lower-high for short stops
        if len(swing_high_idx) > 0:
            right_idx = np.searchsorted(swing_high_idx, i) - 1
            if right_idx >= 0 and swing_high_idx[right_idx] >= i - 40:
                recent_lh[i] = swing_high_val[right_idx]

    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    out["pa_stop_long"] = recent_hl
    out["pa_stop_short"] = recent_lh
    return out


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

    # pa_or_position = (close - or_low) / or_range
    safe_range = np.where((np.isfinite(or_range)) & (or_range > 1e-8), or_range, 1e-8)
    or_position = np.where(np.isfinite(or_range), (close_arr - or_low) / safe_range, 0.5)

    result = pd.DataFrame(index=df.index)
    result["pa_or_position"] = or_position
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

    swing_highs = []  # (confirm_idx, swing_idx, value)
    swing_lows = []

    for i in range(confirm_len, n):
        candidate = i - confirm_len
        if candidate < confirm_len:
            continue

        left_window_h = high[candidate - confirm_len: candidate]
        right_window_h = high[candidate + 1: i + 1]
        if len(right_window_h) == confirm_len and high[candidate] > left_window_h.max() and high[candidate] > right_window_h.max():
            swing_highs.append((i, candidate, high[candidate]))

        left_window_l = low[candidate - confirm_len: candidate]
        right_window_l = low[candidate + 1: i + 1]
        if len(right_window_l) == confirm_len and low[candidate] < left_window_l.min() and low[candidate] < right_window_l.min():
            swing_lows.append((i, candidate, low[candidate]))

    return swing_highs, swing_lows


# ─────────────────────────────────────────────────────────────────
# 2. Causal Stops (based on trailing-confirmed swings)
# ─────────────────────────────────────────────────────────────────

def _causal_pa_stops(bars: pd.DataFrame, confirm_len: int = 5) -> pd.DataFrame:
    """PA stops using delayed-confirmation swing points. Rule X03, X04."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

    sh_confirm = np.array([c for c, _, _ in swing_highs], dtype=int) if swing_highs else np.array([], dtype=int)
    sh_val = np.array([v for _, _, v in swing_highs], dtype=float) if swing_highs else np.array([], dtype=float)
    sl_confirm = np.array([c for c, _, _ in swing_lows], dtype=int) if swing_lows else np.array([], dtype=int)
    sl_val = np.array([v for _, _, v in swing_lows], dtype=float) if swing_lows else np.array([], dtype=float)

    stop_long = np.full(n, np.nan)
    stop_short = np.full(n, np.nan)

    for i in range(n):
        idx_l = np.searchsorted(sl_confirm, i, side="right") - 1
        if idx_l >= 0:
            stop_long[i] = sl_val[idx_l]

        idx_h = np.searchsorted(sh_confirm, i, side="right") - 1
        if idx_h >= 0:
            stop_short[i] = sh_val[idx_h]

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_stop_long"] = stop_long
    out["pa_stop_short"] = stop_short
    return out


# ─────────────────────────────────────────────────────────────────
# 3. Causal Support/Resistance (SR01-SR14)
# ─────────────────────────────────────────────────────────────────

def _causal_support_resistance(bars: pd.DataFrame, atr_5m: pd.Series, confirm_len: int = 5) -> pd.DataFrame:
    """S/R from trailing-confirmed swings + round numbers + 50% retrace."""
    n = len(bars)
    close = bars["close"].to_numpy(dtype=float)
    atr_arr = atr_5m.to_numpy(dtype=float)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

    nearest_resist = np.full(n, np.nan)
    nearest_support = np.full(n, np.nan)
    sr_position = np.full(n, 0.5)
    retrace_50 = np.full(n, np.nan)
    at_50_retrace = np.zeros(n, dtype=bool)
    round_num_dist = np.full(n, np.nan)

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

        cur_atr = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else 1e-6

        if confirmed_sh:
            vals_above = [v for _, v in confirmed_sh if v > close[i]]
            if vals_above:
                nearest_resist[i] = min(vals_above)
            else:
                nearest_resist[i] = max(v for _, v in confirmed_sh)

        if confirmed_sl:
            vals_below = [v for _, v in confirmed_sl if v < close[i]]
            if vals_below:
                nearest_support[i] = max(vals_below)
            else:
                nearest_support[i] = min(v for _, v in confirmed_sl)

        sup, res = nearest_support[i], nearest_resist[i]
        if np.isfinite(sup) and np.isfinite(res) and res > sup:
            sr_position[i] = np.clip((close[i] - sup) / (res - sup), 0.0, 1.0)

        if confirmed_sh and confirmed_sl:
            last_sh_v = confirmed_sh[-1][1]
            last_sl_v = confirmed_sl[-1][1]
            retrace_50[i] = (last_sh_v + last_sl_v) / 2.0
            at_50_retrace[i] = abs(close[i] - retrace_50[i]) < 0.2 * cur_atr

        round_5 = round(close[i] / 5.0) * 5.0
        round_1 = round(close[i])
        nearest_round = round_5 if abs(close[i] - round_5) < abs(close[i] - round_1) else round_1
        round_num_dist[i] = abs(close[i] - nearest_round) / cur_atr

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

def _causal_measured_move(bars: pd.DataFrame, confirm_len: int = 5) -> pd.DataFrame:
    """Measured move targets from trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

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
    bars: pd.DataFrame, direction: np.ndarray, atr_5m: pd.Series, confirm_len: int = 5,
) -> pd.DataFrame:
    """Double top/bottom using trailing-confirmed swings."""
    n = len(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    atr_arr = atr_5m.to_numpy(dtype=float)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

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
            if abs(val_a - val_b) < 0.3 * cur_atr and (idx_b - idx_a) >= 5:
                double_top[i] = True
                intervening_low = np.min(low[idx_a:idx_b + 1])
                dt_mm[i] = intervening_low - (max(val_a, val_b) - intervening_low)
                if direction[min(i, len(direction) - 1)] == -1:
                    dt_is_flag[i] = True

        if len(confirmed_sl) >= 2:
            idx_a, val_a = confirmed_sl[-2]
            idx_b, val_b = confirmed_sl[-1]
            if abs(val_a - val_b) < 0.3 * cur_atr and (idx_b - idx_a) >= 5:
                double_bottom[i] = True
                intervening_high = np.max(high[idx_a:idx_b + 1])
                db_mm[i] = intervening_high + (intervening_high - min(val_a, val_b))
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
    bars: pd.DataFrame, regime: pd.DataFrame, mag: pd.DataFrame, confirm_len: int = 3,
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
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)
    wedge_up = np.zeros(n, dtype=bool)
    wedge_down = np.zeros(n, dtype=bool)

    confirmed_sh_list = []
    confirmed_sl_list = []
    sh_ptr = 0
    sl_ptr = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            confirmed_sh_list.append((swing_highs[sh_ptr][1], swing_highs[sh_ptr][2]))
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            confirmed_sl_list.append((swing_lows[sl_ptr][1], swing_lows[sl_ptr][2]))
            sl_ptr += 1

        if len(confirmed_sl_list) >= 3:
            a_i, a_v = confirmed_sl_list[-3]
            b_i, b_v = confirmed_sl_list[-2]
            c_i, c_v = confirmed_sl_list[-1]
            leg1 = max(1, b_i - a_i)
            leg3 = max(1, c_i - b_i)
            if leg3 <= 0.7 * leg1 and c_v <= b_v <= a_v:
                wedge_up[i] = True

        if len(confirmed_sh_list) >= 3:
            a_i, a_v = confirmed_sh_list[-3]
            b_i, b_v = confirmed_sh_list[-2]
            c_i, c_v = confirmed_sh_list[-1]
            leg1 = max(1, b_i - a_i)
            leg3 = max(1, c_i - b_i)
            if leg3 <= 0.7 * leg1 and c_v >= b_v >= a_v:
                wedge_down[i] = True

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

def _causal_tbtl(bars: pd.DataFrame, confirm_len: int = 3) -> pd.DataFrame:
    """TBTL pattern using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

    tbtl_up = np.zeros(n, dtype=bool)
    tbtl_down = np.zeros(n, dtype=bool)

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

        recent_sh = [(idx, v) for idx, v in confirmed_sh if i - idx <= 12]
        recent_sl = [(idx, v) for idx, v in confirmed_sl if i - idx <= 12]

        if len(recent_sh) >= 2 and len(recent_sl) >= 1:
            sh1 = recent_sh[-2]
            sl1 = recent_sl[-1]
            sh2 = recent_sh[-1]
            if sh1[0] < sl1[0] < sh2[0]:
                span = sh2[0] - sh1[0]
                if span <= 12 and sh2[1] < sh1[1]:
                    tbtl_down[i] = True

        if len(recent_sl) >= 2 and len(recent_sh) >= 1:
            sl1 = recent_sl[-2]
            sh1 = recent_sh[-1]
            sl2 = recent_sl[-1]
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

def _causal_triangle(bars: pd.DataFrame, confirm_len: int = 3) -> pd.DataFrame:
    """Triangle patterns using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

    converging = np.zeros(n, dtype=bool)
    expanding = np.zeros(n, dtype=bool)
    breakout_bias = np.zeros(n, dtype=float)

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

        recent_sh = [(idx, v) for idx, v in confirmed_sh if i - idx <= 20]
        recent_sl = [(idx, v) for idx, v in confirmed_sl if i - idx <= 20]

        if len(recent_sh) >= 2 and len(recent_sl) >= 2:
            sh1, sh2 = recent_sh[-2], recent_sh[-1]
            sl1, sl2 = recent_sl[-2], recent_sl[-1]

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

def _causal_head_shoulders(bars: pd.DataFrame, confirm_len: int = 5) -> pd.DataFrame:
    """H&S using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

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

def _causal_parabolic_wedge(bars: pd.DataFrame, confirm_len: int = 3) -> pd.DataFrame:
    """Parabolic wedge using trailing-confirmed swings."""
    n = len(bars)
    swing_highs, swing_lows = _causal_swings(bars, confirm_len)

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
# 13. Cross-Timeframe Features (15m / 30m / 1h)
# ─────────────────────────────────────────────────────────────────

def _resample_higher_tf(bars_5m: pd.DataFrame, period: str) -> pd.DataFrame:
    """Resample 5m bars to a higher timeframe (15min, 30min, 1h)."""
    df = bars_5m.set_index(pd.to_datetime(bars_5m["time_key"]))
    resampled = df[["open", "high", "low", "close", "volume"]].resample(period, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    resampled["time_key"] = resampled.index
    return resampled.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# 14. Structure Features (HH/HL/LH/LL, swing range, break)
# ─────────────────────────────────────────────────────────────────

def _structure_features(bars_5m: pd.DataFrame, atr_5m: pd.Series, confirm_len: int = 5) -> pd.DataFrame:
    """
    Market structure features using causal swings.
    Tracks HH/HL/LH/LL sequences, trend structure score, swing range.
    """
    n = len(bars_5m)
    close = bars_5m["close"].to_numpy(dtype=float)
    atr_arr = atr_5m.to_numpy(dtype=float)
    swing_highs, swing_lows = _causal_swings(bars_5m, confirm_len)

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

    prev_sh_val = np.nan
    prev_sl_val = np.nan

    for i in range(n):
        new_sh = False
        new_sl = False
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] <= i:
            _, swing_idx, val = swing_highs[sh_ptr]
            confirmed_sh.append((swing_idx, val))
            new_sh = True
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] <= i:
            _, swing_idx, val = swing_lows[sl_ptr]
            confirmed_sl.append((swing_idx, val))
            new_sl = True
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

        # Leg count: total confirmed swings in recent window
        leg_count[i] = len([s for s in confirmed_sh if i - s[0] <= 40]) + \
                        len([s for s in confirmed_sl if i - s[0] <= 40])

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
    for j in range(n):
        vals = [ema8[j], ema20[j], ema50[j]]
        if all(np.isfinite(v) for v in vals) and safe_atr[j] > 0:
            ma_compress[j] = np.std(vals) / safe_atr[j]

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

    HMM-style block:
      - Soft state probabilities from trend-strength + volatility context
      - State confidence / persistence / transition pressure

    GARCH-style block:
      - Recursive variance estimate: var_t = w + a*r_{t-1}^2 + b*var_{t-1}
      - Volatility ratio / z-score / shock / vol-of-vol
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
    garch_vol_ratio = garch_vol / garch_vol_ma
    garch_vol_z = np.clip(garch_vol_ratio - 1.0, -4.0, 4.0)

    garch_shock = (log_ret * log_ret) / np.maximum(var, eps) - 1.0
    garch_shock = np.clip(garch_shock, -3.0, 6.0)
    garch_vol_of_vol = np.zeros(n, dtype=float)
    if n > 1:
        garch_vol_of_vol[1:] = np.abs(np.diff(garch_vol)) / garch_vol_ma[1:]

    out = pd.DataFrame(index=df_1m.index)
    out["time_key"] = df_1m["time_key"].values

    # HMM-style features
    out["pa_hmm_state"] = hmm_state
    out["pa_hmm_transition_pressure"] = hmm_transition_pressure

    # GARCH-style features
    out["pa_garch_vol"] = garch_vol
    out["pa_garch_shock"] = garch_shock
    out["pa_garch_vol_of_vol"] = garch_vol_of_vol
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


# ─────────────────────────────────────────────────────────────────
# ORCHESTRATOR: add_pa_features()
# ─────────────────────────────────────────────────────────────────



# ── Orchestrator ──

def add_pa_features(
    df: pd.DataFrame, atr_series: pd.Series, timeframe: str = "5min",
) -> pd.DataFrame:
    """
    Compute all PA features with ZERO look-ahead bias.

    Unified causal PA feature stack (single source of truth).
    Same interface: 1-min input DataFrame, returns DataFrame with pa_* columns.
    """
    result = df.copy()
    times_1m = pd.to_datetime(result["time_key"])

    # Opening Range (causal — running max/min)
    or_df = _causal_opening_range(result, atr_series)
    result = pd.concat([result.reset_index(drop=True), or_df.reset_index(drop=True)], axis=1)

    # Resample to 5min
    if timeframe == "1min":
        bars_5m = result.copy()
    else:
        bars_5m = _resample_5min(result)
    if bars_5m.empty:
        return result

    # ATR on 5min
    rng_5m = bars_5m["high"] - bars_5m["low"]
    atr_5m = rng_5m.ewm(span=14, min_periods=5).mean()

    direction_5m = _derive_direction_5m(bars_5m)
    ema20_5m = _ema(bars_5m["close"], 20)

    # ── Clean features (imported from pa_rules.py) ──
    bar_class = _classify_bars_5m(bars_5m, atr_5m)
    pullback = _pullback_counting_5m(bars_5m, direction_5m)
    mag = _mag_bar_5m(bars_5m, ema20_5m)
    inside = _inside_bars_5m(bars_5m)
    regime_raw = _market_regime_5m(bars_5m)
    twenty = _twenty_bar_rule_5m(bars_5m, direction_5m)
    pressure = _pressure_score_5m(bars_5m)
    gaps = _gap_detection_5m(bars_5m, direction_5m)
    signal_scores = _signal_bar_scoring_5m(bars_5m)
    outside_eng = _outside_engulfing_5m(bars_5m)
    climax = _climax_detection_5m(bars_5m)
    momentum = _momentum_accel_5m(bars_5m)
    gap_enh = _gap_enhanced_5m(bars_5m, direction_5m)
    session = _day_session_phase(bars_5m)
    vol_bo = _volume_breakout_5m(bars_5m)
    prev_day = _prev_day_context(bars_5m)
    open_pat = _opening_patterns(bars_5m)

    # Build combined regime (regime_raw + twenty) for downstream consumers
    regime = regime_raw.copy()
    for c in twenty.columns:
        if c != "time_key":
            regime[c] = twenty[c].values

    final_flag = _final_flag_5m(bars_5m, regime, inside)
    channel_z = _channel_zones_5m(bars_5m, regime)
    endless_pb = _endless_pullback_5m(bars_5m, direction_5m)
    cycle = _market_cycle_5m(regime)
    tr_mm = _tr_measured_move_5m(bars_5m, regime)
    vol_climax = _volume_climax_exhaustion_5m(bars_5m, regime)

    # ── Causal replacements (new implementations) ──
    stops = _causal_pa_stops(bars_5m)
    sr = _causal_support_resistance(bars_5m, atr_5m)
    mm = _causal_measured_move(bars_5m)
    bo_strength = _causal_breakout_strength(bars_5m, regime)
    trail_mom = _causal_trailing_momentum(bars_5m)
    double_tb = _causal_double_top_bottom(bars_5m, direction_5m, atr_5m)
    adv_triggers = _causal_advanced_triggers(bars_5m, regime, mag)
    tbtl = _causal_tbtl(bars_5m)
    triangle = _causal_triangle(bars_5m)
    hs = _causal_head_shoulders(bars_5m)
    para_wedge = _causal_parabolic_wedge(bars_5m)

    # These depend on clean + causal features
    mtr = _mtr_detection_5m(bars_5m, regime, pressure, double_tb)
    ct_quality = _counter_trend_quality_5m(bars_5m, regime, signal_scores)

    # ── New feature groups (v2) ──
    struct_feats = _structure_features(bars_5m, atr_5m)
    ma_feats = _ma_relationship_features(bars_5m, atr_5m)
    vol_feats = _volume_features(bars_5m, atr_5m)
    hmm_garch_feats = _hmm_garch_features(bars_5m)
    
    # ── Compute 1m specific features directly on result ──
    realized_vol_feats = _realized_volatility_features(result)
    intraday_time_feats = _intraday_time_features(result)
    realized_moments_feats = _realized_moments_features(result)
    vol_microstructure_feats = _volume_microstructure_features(result)
    wavelet_approx_feats = _wavelet_approx_features(result)
    hawkes_feats = _hawkes_features(result)
    
    kalman_feats = _kalman_features(result)
    hurst_feats = _hurst_features(result)
    entropy_feats = _entropy_features(result)
    jump_feats = _jump_diffusion_features(result)
    
    new_1m_parts = [
        realized_vol_feats.drop(columns=["time_key"], errors="ignore"),
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

    # ── Compress 5m features ──
    compressed_feats = _compressed_pa_features(
        bars_5m, pullback, regime, pressure, struct_feats
    )

    # ── Merge all 5m features ──
    all_5m_parts = [
        compressed_feats,
        ma_feats, vol_feats, hmm_garch_feats,
    ]

    feature_cols = []
    for part in all_5m_parts:
        for c in part.columns:
            if c != "time_key" and c not in feature_cols:
                feature_cols.append(c)

    merged_5m_parts = [bars_5m[["time_key"]]]
    for part in all_5m_parts:
        part_no_time = part.drop(columns=["time_key"], errors="ignore")
        merged_5m_parts.append(part_no_time)
    
    # Avoid DataFrame fragmentation warning by concatenating all new columns at once
    merged_5m = pd.concat(merged_5m_parts, axis=1)
    
    # Ensure no duplicate columns from potential overlaps in parts (other than time_key)
    merged_5m = merged_5m.loc[:, ~merged_5m.columns.duplicated()]

    # Map 5m → 1m
    if timeframe == "1min":
        # Avoid DataFrame fragmentation warning by concatenating all new columns at once
        merged_5m_subset = merged_5m[feature_cols]
        result = pd.concat([result, merged_5m_subset], axis=1)
    else:
        mapped = _map_5min_to_1min(merged_5m, times_1m, feature_cols)
        # Avoid DataFrame fragmentation warning by concatenating all new columns at once
        mapped_subset = mapped[feature_cols]
        result = pd.concat([result, mapped_subset], axis=1)

    _append_htf_regime_from_1m(result, times_1m)

    return result


def add_all_pa_features(
    df: pd.DataFrame, atr_series: pd.Series, timeframe: str = "5min",
) -> pd.DataFrame:
    """Backward-compatible alias — use :func:`add_pa_features`."""
    return add_pa_features(df, atr_series, timeframe=timeframe)


add_causal_pa_features = add_pa_features

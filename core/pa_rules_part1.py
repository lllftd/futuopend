"""
Price Action (PA) rules module.

Computes all PA features on 1-minute bar data, using internal 5-min resampling
where bar-level patterns require it. Features are mapped back to the 1-min index
via forward-fill so the execution engine stays on 1-min granularity.
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
    right = htf[["time_key"] + columns].drop_duplicates(subset=["time_key"]).sort_values("time_key")
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
        ("5min", "pa_htf_5min"),
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


def compute_opening_range(df: pd.DataFrame, daily_atr: pd.Series) -> pd.DataFrame:
    """
    For each trading day compute the Opening Range (first 90 one-minute bars,
    equivalent to 18 five-minute bars).

    Returns DataFrame aligned to *df* with columns:
        or_high, or_low, or_range, or_period (bool – still inside OR window),
        or_breakout_up, or_breakout_down, or_volume_breakout, or_vs_atr_ratio, or_wide
    """
    times = pd.to_datetime(df["time_key"])
    dates = times.dt.date
    bar_time = times.dt.time

    in_or = (bar_time >= _OR_START_TIME) & (bar_time < _OR_END_TIME)

    or_high_map: dict = {}
    or_low_map: dict = {}
    for d, grp in df.groupby(dates):
        mask = in_or.loc[grp.index]
        or_bars = grp.loc[mask]
        if or_bars.empty:
            or_high_map[d] = np.nan
            or_low_map[d] = np.nan
        else:
            or_high_map[d] = float(or_bars["high"].max())
            or_low_map[d] = float(or_bars["low"].min())

    or_high = dates.map(or_high_map).astype(float)
    or_low = dates.map(or_low_map).astype(float)
    or_range = or_high - or_low

    vol_mean = df["volume"].rolling(20, min_periods=1).mean()

    after_or = ~in_or & (bar_time >= _OR_END_TIME)
    breakout_up = after_or & (df["close"].to_numpy() > or_high.to_numpy())
    breakout_down = after_or & (df["close"].to_numpy() < or_low.to_numpy())
    vol_breakout = (breakout_up | breakout_down) & (df["volume"] > vol_mean)

    atr_daily = daily_atr.groupby(dates).transform("last")
    ratio = or_range / atr_daily.replace(0, np.nan)

    result = pd.DataFrame(index=df.index)
    result["or_high"] = or_high.values
    result["or_low"] = or_low.values
    result["or_range"] = or_range.values
    result["or_period"] = in_or.values
    result["or_breakout_up"] = breakout_up.values
    result["or_breakout_down"] = breakout_down.values
    result["or_volume_breakout"] = vol_breakout.values
    result["or_vs_atr_ratio"] = ratio.values
    result["or_wide"] = (ratio > 0.5).values
    return result


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
    is_inside = (bars["high"] < prev_high) & (bars["low"] > prev_low)
    is_ii = is_inside & is_inside.shift(1).fillna(False)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_is_inside_bar"] = is_inside.fillna(False).values
    out["pa_is_ii_pattern"] = is_ii.fillna(False).values
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


# ---------------------------------------------------------------------------
# 1k. Advanced PA triggers (5-min)
# ---------------------------------------------------------------------------

def _advanced_pa_triggers_5m(
    bars: pd.DataFrame,
    regime: pd.DataFrame,
    mag: pd.DataFrame,
) -> pd.DataFrame:
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
    env_dir = regime["pa_env_trend_dir"].to_numpy(dtype=int)
    bars_since_extreme = regime["pa_env_bars_since_extreme"].to_numpy(dtype=int)

    mag_base = mag["pa_is_mag_bar"].to_numpy(dtype=bool)
    ema20 = _ema(bars["close"], 20).to_numpy(dtype=float)
    mag_bull = mag_base & (low > ema20) & (close > opn) & (body_ratio > 0.6)
    mag_bear = mag_base & (high < ema20) & (close < opn) & (body_ratio > 0.6)

    # Wedge third push with momentum decay (third leg bars < 70% first leg bars).
    wedge_rev_up = np.zeros(n, dtype=bool)
    wedge_rev_down = np.zeros(n, dtype=bool)
    swing_low_idx: list[int] = []
    swing_high_idx: list[int] = []
    pivot_len = 3
    for i in range(pivot_len, n - pivot_len):
        wl = low[i - pivot_len: i + pivot_len + 1]
        wh = high[i - pivot_len: i + pivot_len + 1]
        if low[i] == wl.min():
            swing_low_idx.append(i)
            if len(swing_low_idx) >= 3:
                a, b, c = swing_low_idx[-3], swing_low_idx[-2], swing_low_idx[-1]
                leg1_bars = max(1, b - a)
                leg3_bars = max(1, c - b)
                if leg3_bars <= 0.7 * leg1_bars and low[c] <= low[b] <= low[a]:
                    wedge_rev_up[c] = True
        if high[i] == wh.max():
            swing_high_idx.append(i)
            if len(swing_high_idx) >= 3:
                a, b, c = swing_high_idx[-3], swing_high_idx[-2], swing_high_idx[-1]
                leg1_bars = max(1, b - a)
                leg3_bars = max(1, c - b)
                if leg3_bars <= 0.7 * leg1_bars and high[c] >= high[b] >= high[a]:
                    wedge_rev_down[c] = True

    # Channel overshoot then back inside within 5 bars.
    overshoot_revert_up = np.zeros(n, dtype=bool)
    overshoot_revert_down = np.zeros(n, dtype=bool)
    pending_up: list[tuple[int, float]] = []
    pending_down: list[tuple[int, float]] = []
    roll_hi = pd.Series(high).rolling(20, min_periods=5).max().shift(1).to_numpy(dtype=float)
    roll_lo = pd.Series(low).rolling(20, min_periods=5).min().shift(1).to_numpy(dtype=float)
    for i in range(n):
        if np.isfinite(roll_hi[i]) and close[i] > roll_hi[i]:
            pending_up.append((i, roll_hi[i]))
        if np.isfinite(roll_lo[i]) and close[i] < roll_lo[i]:
            pending_down.append((i, roll_lo[i]))

        pending_up = [(idx0, lvl) for idx0, lvl in pending_up if i - idx0 <= 5]
        pending_down = [(idx0, lvl) for idx0, lvl in pending_down if i - idx0 <= 5]

        if any(close[i] < lvl for _, lvl in pending_up):
            overshoot_revert_up[i] = True
            pending_up = []
        if any(close[i] > lvl for _, lvl in pending_down):
            overshoot_revert_down[i] = True
            pending_down = []

    # Channel probability trigger: channel runs >=20 bars, prep reversal logic.
    channel_state = np.isin(env_state, ["tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"])
    channel_reversal_ready = channel_state & (bars_since_extreme >= 20)

    # TR -> trend breakout success / failure checks.
    tr_state = np.isin(env_state, ["wide_tr", "ttr", "neutral_50_50"])
    breakout_success_up = np.zeros(n, dtype=bool)
    breakout_success_down = np.zeros(n, dtype=bool)
    breakout_fail_up = np.zeros(n, dtype=bool)
    breakout_fail_down = np.zeros(n, dtype=bool)
    measured_gap_up = np.zeros(n, dtype=bool)
    measured_gap_down = np.zeros(n, dtype=bool)
    exhaustion_gap_up = np.zeros(n, dtype=bool)
    exhaustion_gap_down = np.zeros(n, dtype=bool)

    for i in range(21, n - 2):
        if not tr_state[i]:
            continue
        prev_hi = np.max(high[i - 20: i])
        prev_lo = np.min(low[i - 20: i])
        prev_span = max(prev_hi - prev_lo, 1e-12)

        strong_body = (
            body_ratio[i] > 0.6
            and body[i] > 1.5 * (avg_body20[i] if np.isfinite(avg_body20[i]) else body[i])
        )

        # Breakout up candidate
        if close[i] > prev_hi and strong_body and (close[i] >= high[i] - 0.25 * rng[i]):
            # Follow-through rule: bar+1 must not retrace >50% of breakout body.
            breakout_mid_body = opn[i] + 0.5 * (close[i] - opn[i])
            follow_ok = close[i + 1] >= breakout_mid_body
            gap = low[i] > prev_hi
            gap_hold3 = gap and np.min(low[i: min(n, i + 3)]) > prev_hi
            gap_fill_3_5 = gap and np.min(low[i + 3: min(n, i + 6)]) <= prev_hi
            leg = max(close[i] - prev_hi, 1e-12)
            pb_min = np.min(low[i + 1: min(n, i + 4)])
            depth = (close[i] - pb_min) / leg
            pb_depth_ok = depth < 0.382
            pb_len_ok = (min(n, i + 4) - (i + 1)) <= 3

            if follow_ok and gap_hold3 and pb_depth_ok and pb_len_ok:
                breakout_success_up[i] = True
                measured_gap_up[i] = True
            elif gap_fill_3_5:
                breakout_fail_up[i] = True
                exhaustion_gap_up[i] = True

        # Breakout down candidate
        if close[i] < prev_lo and strong_body and (close[i] <= low[i] + 0.25 * rng[i]):
            breakout_mid_body = opn[i] - 0.5 * (opn[i] - close[i])
            follow_ok = close[i + 1] <= breakout_mid_body
            gap = high[i] < prev_lo
            gap_hold3 = gap and np.max(high[i: min(n, i + 3)]) < prev_lo
            gap_fill_3_5 = gap and np.max(high[i + 3: min(n, i + 6)]) >= prev_lo
            leg = max(prev_lo - close[i], 1e-12)
            pb_max = np.max(high[i + 1: min(n, i + 4)])
            depth = (pb_max - close[i]) / leg
            pb_depth_ok = depth < 0.382
            pb_len_ok = (min(n, i + 4) - (i + 1)) <= 3

            if follow_ok and gap_hold3 and pb_depth_ok and pb_len_ok:
                breakout_success_down[i] = True
                measured_gap_down[i] = True
            elif gap_fill_3_5:
                breakout_fail_down[i] = True
                exhaustion_gap_down[i] = True

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_mag20_bull"] = mag_bull
    out["pa_mag20_bear"] = mag_bear
    out["pa_wedge_third_push_up"] = wedge_rev_up
    out["pa_wedge_third_push_down"] = wedge_rev_down
    out["pa_channel_overshoot_revert_up"] = overshoot_revert_up
    out["pa_channel_overshoot_revert_down"] = overshoot_revert_down
    out["pa_channel_reversal_ready"] = channel_reversal_ready
    out["pa_breakout_success_up"] = breakout_success_up
    out["pa_breakout_success_down"] = breakout_success_down
    out["pa_breakout_fail_up"] = breakout_fail_up
    out["pa_breakout_fail_down"] = breakout_fail_down
    out["pa_measured_gap_up"] = measured_gap_up
    out["pa_measured_gap_down"] = measured_gap_down
    out["pa_exhaustion_gap_up"] = exhaustion_gap_up
    out["pa_exhaustion_gap_down"] = exhaustion_gap_down
    return out

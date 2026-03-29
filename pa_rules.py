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

def _resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 5-min bars, keyed by each 5-min period start."""
    tmp = df.set_index("time_key").copy()
    ohlcv = tmp[["open", "high", "low", "close", "volume"]].resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    ohlcv = ohlcv.reset_index()
    return ohlcv


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
            if cur_h >= 4:
                in_pullback_up = False
                cur_h = 0
            cur_l = 0
            in_pullback_down = False
        elif direction[i] == -1:
            if highs[i] > highs[i - 1]:
                in_pullback_down = True
                cur_l = 0
            if in_pullback_down and lows[i] < lows[i - 1]:
                cur_l += 1
            l_count[i] = cur_l
            if cur_l >= 4:
                in_pullback_down = False
                cur_l = 0
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
    out["pa_is_h2_setup"] = h_count == 2
    out["pa_is_l2_setup"] = l_count == 2
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
    rng = (bars["high"] - bars["low"]).replace(0, np.nan)

    atr_short = rng.ewm(span=20, min_periods=10).mean()
    atr_long = rng.ewm(span=50, min_periods=20).mean()
    atr_ratio = (atr_short / atr_long.replace(0, np.nan)).fillna(1)

    body = (close - bars["open"].astype(float))
    direction_consistency = body.rolling(10, min_periods=5).apply(
        lambda x: abs(np.sum(np.sign(x))) / len(x), raw=True
    ).fillna(0)

    is_trend = (atr_ratio > 1.05) & (direction_consistency > 0.5)
    is_range = (atr_ratio < 0.95) | (direction_consistency < 0.3)

    out = pd.DataFrame(index=bars.index)
    out["time_key"] = bars["time_key"].values
    out["pa_regime_trend"] = is_trend.values
    out["pa_regime_range"] = is_range.values
    out["pa_reversal_likely_fail"] = is_trend.values
    out["pa_breakout_likely_fail"] = is_range.values
    return out


# ---------------------------------------------------------------------------
# 1g. 20-Bar Rule (5-min)
# ---------------------------------------------------------------------------

def _twenty_bar_rule_5m(bars: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    n = len(bars)
    bars_since_change = np.zeros(n, dtype=int)
    trend_weakened = np.zeros(n, dtype=bool)

    count = 0
    prev_dir = 0
    for i in range(n):
        d = direction[i]
        if d != prev_dir and d != 0:
            count = 0
            prev_dir = d
        else:
            count += 1
        bars_since_change[i] = count

    in_pullback = np.zeros(n, dtype=int)
    pb_count = 0
    for i in range(1, n):
        if direction[i] == direction[i - 1] and direction[i] != 0:
            close_val = bars["close"].iloc[i]
            open_val = bars["open"].iloc[i]
            if direction[i] == 1 and close_val < open_val:
                pb_count += 1
            elif direction[i] == -1 and close_val > open_val:
                pb_count += 1
            else:
                pb_count = 0
        else:
            pb_count = 0
        in_pullback[i] = pb_count
        if pb_count > 20:
            trend_weakened[i] = True

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

    # Leg1=Leg2: for upward MM, find last swing_low → swing_high → swing_low pattern
    for i in range(n):
        # upward MM target
        relevant_lows = [(idx, v) for idx, v in swing_lows if idx <= i]
        relevant_highs = [(idx, v) for idx, v in swing_highs if idx <= i]
        if len(relevant_lows) >= 2 and len(relevant_highs) >= 1:
            sl1 = relevant_lows[-2]
            sh = None
            for idx_h, v_h in reversed(relevant_highs):
                if idx_h > sl1[0]:
                    sh = (idx_h, v_h)
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
            for idx_l, v_l in reversed(relevant_lows):
                if idx_l > sh1[0]:
                    sl = (idx_l, v_l)
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

    for i in range(n):
        # Recent higher-low for long stops
        candidate_lows = [v for idx, v in swing_lows if idx < i and idx >= i - 40]
        if candidate_lows:
            recent_hl[i] = candidate_lows[-1]

        # Recent lower-high for short stops
        candidate_highs = [v for idx, v in swing_highs if idx < i and idx >= i - 40]
        if candidate_highs:
            recent_lh[i] = candidate_highs[-1]

    out = pd.DataFrame(index=bars_5m.index)
    out["time_key"] = bars_5m["time_key"].values
    out["pa_stop_long"] = recent_hl
    out["pa_stop_short"] = recent_lh
    return out


# ---------------------------------------------------------------------------
# 1k. Orchestrator: add_all_pa_features
# ---------------------------------------------------------------------------

def add_all_pa_features(df: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """
    Compute all PA features and merge them onto the 1-min DataFrame.

    Parameters
    ----------
    df : DataFrame with columns time_key, open, high, low, close, volume
    atr_series : pre-computed ATR on 1-min data (used for OR ratio)

    Returns
    -------
    DataFrame with all pa_* columns appended.
    """
    result = df.copy()
    times_1m = pd.to_datetime(result["time_key"])

    # --- Opening Range (directly on 1-min) ---
    or_df = compute_opening_range(result, atr_series)
    for col in or_df.columns:
        result[col] = or_df[col].values

    # --- Resample to 5-min for bar-level PA features ---
    bars_5m = _resample_5min(result)
    if bars_5m.empty:
        # Fallback: add empty PA columns
        pa_cols = [
            "pa_body_ratio", "pa_is_trend_bar", "pa_is_range_bar",
            "pa_strong_bull_signal", "pa_strong_bear_signal",
            "pa_h_count", "pa_l_count", "pa_is_h2_setup", "pa_is_l2_setup",
            "pa_is_mag_bar", "pa_is_inside_bar", "pa_is_ii_pattern",
            "pa_regime_trend", "pa_regime_range",
            "pa_reversal_likely_fail", "pa_breakout_likely_fail",
            "pa_bars_since_trend_change", "pa_trend_weakened",
            "pa_bull_pressure", "pa_bear_pressure", "pa_net_pressure",
            "pa_mm_target_up", "pa_mm_target_down",
            "pa_is_exhaustion_gap", "pa_is_measured_gap",
            "pa_stop_long", "pa_stop_short",
        ]
        for c in pa_cols:
            result[c] = np.nan
        return result

    # ATR on 5-min
    rng_5m = bars_5m["high"] - bars_5m["low"]
    atr_5m = rng_5m.ewm(span=14, min_periods=5).mean()

    # Direction on 5-min (EMA-based)
    direction_5m = _derive_direction_5m(bars_5m)

    # EMA-20 on 5-min close (for MAG bar)
    ema20_5m = _ema(bars_5m["close"], 20)

    # Compute each feature set on 5-min
    bar_class = _classify_bars_5m(bars_5m, atr_5m)
    pullback = _pullback_counting_5m(bars_5m, direction_5m)
    mag = _mag_bar_5m(bars_5m, ema20_5m)
    inside = _inside_bars_5m(bars_5m)
    regime = _market_regime_5m(bars_5m)
    twenty = _twenty_bar_rule_5m(bars_5m, direction_5m)
    pressure = _pressure_score_5m(bars_5m)
    mm = _measured_move_5m(bars_5m)
    gaps = _gap_detection_5m(bars_5m, direction_5m)
    stops = _pa_stops(bars_5m)

    # Map all 5-min features back to 1-min index
    feature_sets = [bar_class, pullback, mag, inside, regime, twenty, pressure, mm, gaps, stops]
    for fset in feature_sets:
        cols = [c for c in fset.columns if c != "time_key"]
        mapped = _map_5min_to_1min(fset, times_1m, cols)
        for c in cols:
            result[c] = mapped[c].values

    return result

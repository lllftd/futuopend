from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


PriceSource = Literal[
    "open",
    "high",
    "low",
    "close",
    "hl2",
    "hlc3",
    "ohlc4",
]


def get_price_source(df: pd.DataFrame, source: PriceSource = "close") -> pd.Series:
    source = source.lower()
    if source == "open":
        return df["open"]
    if source == "high":
        return df["high"]
    if source == "low":
        return df["low"]
    if source == "close":
        return df["close"]
    if source == "hl2":
        return (df["high"] + df["low"]) / 2.0
    if source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise ValueError(f"Unsupported source: {source}")


def linreg(series: pd.Series, length: int = 32, offset: int = 0) -> pd.Series:
    """
    TradingView `linreg(source, length, offset)` compatible rolling linear regression.
    """
    if length <= 1:
        raise ValueError("length must be greater than 1")
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0")

    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    variance = ((x - x_mean) ** 2).sum()
    target_x = length - 1 - offset

    def _calc(window: np.ndarray) -> float:
        y = window.astype(float)
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / variance
        intercept = y_mean - slope * x_mean
        return intercept + slope * target_x

    return series.rolling(window=length, min_periods=length).apply(_calc, raw=True)


def zlsma(
    df: pd.DataFrame,
    length: int = 32,
    offset: int = 0,
    source: PriceSource = "close",
) -> pd.Series:
    """
    Zero Lag LSMA translated from the supplied TradingView Pine Script.

    Pine formula:
        lsma = linreg(src, length, offset)
        lsma2 = linreg(lsma, length, offset)
        eq = lsma - lsma2
        zlsma = lsma + eq
    """
    src = get_price_source(df, source=source)
    lsma = linreg(src, length=length, offset=offset)
    lsma2 = linreg(lsma, length=length, offset=offset)
    eq = lsma - lsma2
    return lsma + eq


def add_zlsma(
    df: pd.DataFrame,
    length: int = 32,
    offset: int = 0,
    source: PriceSource = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    name = column_name or f"zlsma_{source}_{length}_{offset}"
    result[name] = zlsma(result, length=length, offset=offset, source=source)
    return result


def kama(
    df: pd.DataFrame,
    er_length: int = 10,
    fast_length: int = 2,
    slow_length: int = 30,
    source: PriceSource = "close",
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average compatible with TradingView's ta.kama().
    """
    if er_length <= 0:
        raise ValueError("er_length must be greater than 0")
    if fast_length <= 0 or slow_length <= 0:
        raise ValueError("fast_length and slow_length must be greater than 0")

    src = get_price_source(df, source=source).astype(float)
    change = src.diff(er_length).abs()
    volatility = src.diff().abs().rolling(er_length, min_periods=er_length).sum()
    efficiency_ratio = change / volatility.replace(0.0, np.nan)
    efficiency_ratio = efficiency_ratio.fillna(0.0)

    fast_sc = 2.0 / (fast_length + 1.0)
    slow_sc = 2.0 / (slow_length + 1.0)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    values = src.to_numpy(dtype=float)
    sc_values = smoothing_constant.to_numpy(dtype=float)
    kama_values = np.full(len(src), np.nan, dtype=float)

    if len(src) >= er_length:
        kama_values[er_length - 1] = values[er_length - 1]
        for idx in range(er_length, len(src)):
            previous = kama_values[idx - 1]
            if np.isnan(previous):
                previous = values[idx - 1]
            kama_values[idx] = previous + sc_values[idx] * (values[idx] - previous)

    return pd.Series(kama_values, index=df.index, name="kama")


def add_kama(
    df: pd.DataFrame,
    er_length: int = 10,
    fast_length: int = 2,
    slow_length: int = 30,
    source: PriceSource = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    name = column_name or f"kama_{source}_{er_length}_{fast_length}_{slow_length}"
    result[name] = kama(
        result,
        er_length=er_length,
        fast_length=fast_length,
        slow_length=slow_length,
        source=source,
    )
    return result


def _session_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["time_key"]).dt.date


def pseudo_cvd(
    df: pd.DataFrame,
    method: Literal["clv_volume", "clv_body_volume"] = "clv_body_volume",
) -> pd.DataFrame:
    if "volume" not in df.columns:
        raise ValueError("volume column is required to compute pseudo CVD")

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / bar_range).replace([np.inf, -np.inf], np.nan)
    clv = clv.fillna(0.0)
    body_ratio = ((df["close"] - df["open"]) / bar_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if method == "clv_volume":
        pressure = clv
    elif method == "clv_body_volume":
        pressure = 0.7 * clv + 0.3 * body_ratio
    else:
        raise ValueError(f"Unsupported pseudo CVD method: {method}")

    pressure = pressure.clip(-1.0, 1.0)
    delta_proxy = df["volume"].astype(float).fillna(0.0) * pressure
    session_cvd = delta_proxy.groupby(_session_dates(df)).cumsum()

    return pd.DataFrame(
        {
            "cvd_pressure": pressure,
            "cvd_delta_proxy": delta_proxy,
            "cvd_session": session_cvd,
        },
        index=df.index,
    )


def add_pseudo_cvd(
    df: pd.DataFrame,
    method: Literal["clv_volume", "clv_body_volume"] = "clv_body_volume",
) -> pd.DataFrame:
    result = df.copy()
    return result.join(pseudo_cvd(result, method=method))


def cvd_divergence_features(
    df: pd.DataFrame,
    cvd_column: str = "cvd_session",
    lookback: int = 20,
    slope_lookback: int = 3,
) -> pd.DataFrame:
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    if slope_lookback <= 0:
        raise ValueError("slope_lookback must be greater than 0")

    session_dates = _session_dates(df)
    close = df["close"].astype(float)
    cvd = df[cvd_column].astype(float)

    price_prev_high = close.groupby(session_dates).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback).max()
    )
    price_prev_low = close.groupby(session_dates).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback).min()
    )
    cvd_prev_high = cvd.groupby(session_dates).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback).max()
    )
    cvd_prev_low = cvd.groupby(session_dates).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=lookback).min()
    )
    cvd_slope = cvd.groupby(session_dates).transform(lambda s: s.diff(slope_lookback))

    bearish_classic_div = (close > price_prev_high) & (cvd <= cvd_prev_high)
    bullish_classic_div = (close < price_prev_low) & (cvd >= cvd_prev_low)

    return pd.DataFrame(
        {
            "cvd_price_prev_high": price_prev_high,
            "cvd_price_prev_low": price_prev_low,
            "cvd_prev_high": cvd_prev_high,
            "cvd_prev_low": cvd_prev_low,
            "cvd_slope": cvd_slope,
            "cvd_classic_long_ok": (~bearish_classic_div).fillna(True),
            "cvd_classic_short_ok": (~bullish_classic_div).fillna(True),
            "cvd_slope_long_ok": cvd_slope.ge(0.0).fillna(True),
            "cvd_slope_short_ok": cvd_slope.le(0.0).fillna(True),
        },
        index=df.index,
    )


def atr(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    if length <= 0:
        raise ValueError("length must be greater than 0")

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def chandelier_exit(
    df: pd.DataFrame,
    length: int = 22,
    multiplier: float = 3.0,
    use_close: bool = True,
) -> pd.DataFrame:
    """
    Chandelier Exit translated from the supplied TradingView Pine Script.
    """
    if length <= 0:
        raise ValueError("length must be greater than 0")
    if multiplier <= 0:
        raise ValueError("multiplier must be greater than 0")

    result = pd.DataFrame(index=df.index)
    atr_values = multiplier * atr(df, length=length)

    highest_source = df["close"] if use_close else df["high"]
    lowest_source = df["close"] if use_close else df["low"]

    raw_long_stop = highest_source.rolling(length, min_periods=length).max() - atr_values
    raw_short_stop = lowest_source.rolling(length, min_periods=length).min() + atr_values

    long_stop = np.full(len(df), np.nan, dtype=float)
    short_stop = np.full(len(df), np.nan, dtype=float)
    direction = np.full(len(df), np.nan, dtype=float)

    close_values = df["close"].to_numpy(dtype=float)
    raw_long_values = raw_long_stop.to_numpy(dtype=float)
    raw_short_values = raw_short_stop.to_numpy(dtype=float)

    current_dir = 1
    for idx in range(len(df)):
        long_value = raw_long_values[idx]
        short_value = raw_short_values[idx]

        if idx == 0:
            long_stop[idx] = long_value
            short_stop[idx] = short_value
            direction[idx] = current_dir
            continue

        prev_long_stop = long_stop[idx - 1]
        prev_short_stop = short_stop[idx - 1]
        prev_close_value = close_values[idx - 1]

        long_stop[idx] = long_value
        if not np.isnan(prev_long_stop) and not np.isnan(long_value) and prev_close_value > prev_long_stop:
            long_stop[idx] = max(long_value, prev_long_stop)

        short_stop[idx] = short_value
        if not np.isnan(prev_short_stop) and not np.isnan(short_value) and prev_close_value < prev_short_stop:
            short_stop[idx] = min(short_value, prev_short_stop)

        if not np.isnan(prev_short_stop) and close_values[idx] > prev_short_stop:
            current_dir = 1
        elif not np.isnan(prev_long_stop) and close_values[idx] < prev_long_stop:
            current_dir = -1

        direction[idx] = current_dir

    direction_series = pd.Series(direction, index=df.index)
    buy_signal = (direction_series == 1) & (direction_series.shift(1) == -1)
    sell_signal = (direction_series == -1) & (direction_series.shift(1) == 1)

    result["ce_atr"] = atr_values
    result["ce_long_stop"] = long_stop
    result["ce_short_stop"] = short_stop
    result["ce_dir"] = direction_series.astype("Int64")
    result["ce_buy_signal"] = buy_signal.fillna(False)
    result["ce_sell_signal"] = sell_signal.fillna(False)
    result["ce_state_long"] = (direction_series == 1).fillna(False)
    result["ce_state_short"] = (direction_series == -1).fillna(False)
    return result


def add_chandelier_exit(
    df: pd.DataFrame,
    length: int = 22,
    multiplier: float = 3.0,
    use_close: bool = True,
    prefix: str = "ce",
) -> pd.DataFrame:
    result = df.copy()
    ce = chandelier_exit(
        result,
        length=length,
        multiplier=multiplier,
        use_close=use_close,
    )
    rename_map = {column: f"{prefix}_{column.removeprefix('ce_')}" for column in ce.columns}
    ce = ce.rename(columns=rename_map)
    return result.join(ce)


if __name__ == "__main__":
    sample = pd.read_csv("data/QQQ.csv")
    sample["time_key"] = pd.to_datetime(sample["time_key"])
    sample = add_zlsma(sample, length=50, offset=0, source="hlc3", column_name="zlsma")
    sample = add_kama(
        sample,
        er_length=9,
        fast_length=2,
        slow_length=30,
        source="hlc3",
        column_name="kama",
    )
    sample = add_chandelier_exit(
        sample,
        length=1,
        multiplier=2.0,
        use_close=True,
    )
    print(
        sample[
            [
                "time_key",
                "close",
                "zlsma",
                "kama",
                "ce_long_stop",
                "ce_short_stop",
                "ce_dir",
                "ce_buy_signal",
                "ce_sell_signal",
            ]
        ].tail(10).to_string(index=False)
    )

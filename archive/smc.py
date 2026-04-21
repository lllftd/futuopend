from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BULLISH = 1
BEARISH = -1


@dataclass
class OrderBlock:
    top: float
    bottom: float
    bias: int
    created_index: int


@dataclass
class FairValueGap:
    top: float
    bottom: float
    bias: int
    created_index: int


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, length: int = 200) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def detect_pivots(df: pd.DataFrame, length: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 1:
        raise ValueError("length must be greater than 0")

    window = 2 * length + 1
    pivot_highs = (
        df["high"].rolling(window=window, center=True, min_periods=window).max().eq(df["high"])
    )
    pivot_lows = (
        df["low"].rolling(window=window, center=True, min_periods=window).min().eq(df["low"])
    )
    return pivot_highs.fillna(False).to_numpy(dtype=bool), pivot_lows.fillna(False).to_numpy(dtype=bool)


def _crossed_above(prev_close: float, close: float, level: float) -> bool:
    return not np.isnan(level) and prev_close <= level < close


def _crossed_below(prev_close: float, close: float, level: float) -> bool:
    return not np.isnan(level) and prev_close >= level > close


def _select_order_block(
    parsed_highs: np.ndarray,
    parsed_lows: np.ndarray,
    start_idx: int,
    end_idx: int,
    bias: int,
) -> OrderBlock:
    start = max(start_idx, 0)
    end = max(end_idx, start + 1)

    if bias == BULLISH:
        local_idx = int(np.argmin(parsed_lows[start:end]))
    else:
        local_idx = int(np.argmax(parsed_highs[start:end]))

    idx = start + local_idx
    return OrderBlock(
        top=float(parsed_highs[idx]),
        bottom=float(parsed_lows[idx]),
        bias=bias,
        created_index=end_idx - 1,
    )


def add_smc_signals(
    df: pd.DataFrame,
    swing_length: int = 50,
    internal_length: int = 5,
    atr_length: int = 200,
    order_block_filter: str = "atr",
    order_block_mitigation: str = "highlow",
    detect_order_blocks: bool = True,
    detect_fvg: bool = True,
    fvg_auto_threshold: bool = True,
) -> pd.DataFrame:
    """
    Backtest-oriented Smart Money Concepts core signals.

    This is a signal version of the LuxAlgo Pine script, focused on:
    - swing/internal BOS and CHoCH
    - order block creation and active ranges
    - bullish/bearish fair value gap formation and active ranges

    It intentionally does not reproduce TradingView drawing objects.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    n = len(result)
    if n == 0:
        return result

    swing_high_raw, swing_low_raw = detect_pivots(result, swing_length)
    internal_high_raw, internal_low_raw = detect_pivots(result, internal_length)

    volatility = atr(result, atr_length)
    if order_block_filter.lower() == "range":
        volatility = true_range(result).expanding(min_periods=1).mean()

    high_volatility_bar = (result["high"] - result["low"]) >= (2.0 * volatility.fillna(np.inf))
    parsed_high = np.where(high_volatility_bar, result["low"], result["high"]).astype(float)
    parsed_low = np.where(high_volatility_bar, result["high"], result["low"]).astype(float)

    if order_block_mitigation.lower() == "close":
        bearish_mitigation = result["close"].to_numpy(dtype=float)
        bullish_mitigation = result["close"].to_numpy(dtype=float)
    else:
        bearish_mitigation = result["high"].to_numpy(dtype=float)
        bullish_mitigation = result["low"].to_numpy(dtype=float)

    columns = [
        "swing_pivot_high",
        "swing_pivot_low",
        "internal_pivot_high",
        "internal_pivot_low",
        "swing_bullish_bos",
        "swing_bullish_choch",
        "swing_bearish_bos",
        "swing_bearish_choch",
        "internal_bullish_bos",
        "internal_bullish_choch",
        "internal_bearish_bos",
        "internal_bearish_choch",
        "bullish_order_block_created",
        "bearish_order_block_created",
        "bullish_order_block_top",
        "bullish_order_block_bottom",
        "bearish_order_block_top",
        "bearish_order_block_bottom",
        "active_bullish_ob_top",
        "active_bullish_ob_bottom",
        "active_bearish_ob_top",
        "active_bearish_ob_bottom",
        "bullish_fvg",
        "bearish_fvg",
        "bullish_fvg_top",
        "bullish_fvg_bottom",
        "bearish_fvg_top",
        "bearish_fvg_bottom",
        "active_bullish_fvg_top",
        "active_bullish_fvg_bottom",
        "active_bearish_fvg_top",
        "active_bearish_fvg_bottom",
    ]
    for column in columns:
        result[column] = False if column.endswith(("bos", "choch", "created", "fvg")) else np.nan

    close_values = result["close"].to_numpy(dtype=float)
    high_values = result["high"].to_numpy(dtype=float)
    low_values = result["low"].to_numpy(dtype=float)

    prev_close = np.roll(close_values, 1)
    prev_close[0] = close_values[0]

    swing_high_level = np.nan
    swing_low_level = np.nan
    internal_high_level = np.nan
    internal_low_level = np.nan

    swing_high_idx: int | None = None
    swing_low_idx: int | None = None
    internal_high_idx: int | None = None
    internal_low_idx: int | None = None

    swing_high_crossed = False
    swing_low_crossed = False
    internal_high_crossed = False
    internal_low_crossed = False

    swing_trend = 0
    internal_trend = 0

    bullish_order_blocks: list[OrderBlock] = []
    bearish_order_blocks: list[OrderBlock] = []
    bullish_fvgs: list[FairValueGap] = []
    bearish_fvgs: list[FairValueGap] = []

    bar_delta_percent = ((result["close"].shift(1) - result["open"].shift(1)) / result["open"].shift(1)).fillna(0.0)
    if fvg_auto_threshold:
        threshold = bar_delta_percent.abs().expanding(min_periods=1).mean() * 2.0
    else:
        threshold = pd.Series(0.0, index=result.index)

    for i in range(n):
        swing_candidate = i - swing_length
        if swing_candidate >= 0:
            if swing_low_raw[swing_candidate]:
                swing_low_level = low_values[swing_candidate]
                swing_low_idx = swing_candidate
                swing_low_crossed = False
                result.at[result.index[i], "swing_pivot_low"] = swing_low_level
            if swing_high_raw[swing_candidate]:
                swing_high_level = high_values[swing_candidate]
                swing_high_idx = swing_candidate
                swing_high_crossed = False
                result.at[result.index[i], "swing_pivot_high"] = swing_high_level

        internal_candidate = i - internal_length
        if internal_candidate >= 0:
            if internal_low_raw[internal_candidate]:
                internal_low_level = low_values[internal_candidate]
                internal_low_idx = internal_candidate
                internal_low_crossed = False
                result.at[result.index[i], "internal_pivot_low"] = internal_low_level
            if internal_high_raw[internal_candidate]:
                internal_high_level = high_values[internal_candidate]
                internal_high_idx = internal_candidate
                internal_high_crossed = False
                result.at[result.index[i], "internal_pivot_high"] = internal_high_level

        if _crossed_above(prev_close[i], close_values[i], swing_high_level) and not swing_high_crossed:
            column = "swing_bullish_choch" if swing_trend == BEARISH else "swing_bullish_bos"
            result.at[result.index[i], column] = True
            swing_high_crossed = True
            swing_trend = BULLISH
            if detect_order_blocks and swing_high_idx is not None and swing_high_idx < i:
                ob = _select_order_block(parsed_high, parsed_low, swing_high_idx, i + 1, BULLISH)
                bullish_order_blocks.insert(0, ob)
                result.at[result.index[i], "bullish_order_block_created"] = True
                result.at[result.index[i], "bullish_order_block_top"] = ob.top
                result.at[result.index[i], "bullish_order_block_bottom"] = ob.bottom

        if _crossed_below(prev_close[i], close_values[i], swing_low_level) and not swing_low_crossed:
            column = "swing_bearish_choch" if swing_trend == BULLISH else "swing_bearish_bos"
            result.at[result.index[i], column] = True
            swing_low_crossed = True
            swing_trend = BEARISH
            if detect_order_blocks and swing_low_idx is not None and swing_low_idx < i:
                ob = _select_order_block(parsed_high, parsed_low, swing_low_idx, i + 1, BEARISH)
                bearish_order_blocks.insert(0, ob)
                result.at[result.index[i], "bearish_order_block_created"] = True
                result.at[result.index[i], "bearish_order_block_top"] = ob.top
                result.at[result.index[i], "bearish_order_block_bottom"] = ob.bottom

        internal_bull_extra = not np.isnan(internal_high_level) and internal_high_level != swing_high_level
        if (
            internal_bull_extra
            and _crossed_above(prev_close[i], close_values[i], internal_high_level)
            and not internal_high_crossed
        ):
            column = "internal_bullish_choch" if internal_trend == BEARISH else "internal_bullish_bos"
            result.at[result.index[i], column] = True
            internal_high_crossed = True
            internal_trend = BULLISH

        internal_bear_extra = not np.isnan(internal_low_level) and internal_low_level != swing_low_level
        if (
            internal_bear_extra
            and _crossed_below(prev_close[i], close_values[i], internal_low_level)
            and not internal_low_crossed
        ):
            column = "internal_bearish_choch" if internal_trend == BULLISH else "internal_bearish_bos"
            result.at[result.index[i], column] = True
            internal_low_crossed = True
            internal_trend = BEARISH

        bullish_order_blocks = [
            ob for ob in bullish_order_blocks if bullish_mitigation[i] >= ob.bottom
        ]
        bearish_order_blocks = [
            ob for ob in bearish_order_blocks if bearish_mitigation[i] <= ob.top
        ]

        if bullish_order_blocks:
            result.at[result.index[i], "active_bullish_ob_top"] = bullish_order_blocks[0].top
            result.at[result.index[i], "active_bullish_ob_bottom"] = bullish_order_blocks[0].bottom
        if bearish_order_blocks:
            result.at[result.index[i], "active_bearish_ob_top"] = bearish_order_blocks[0].top
            result.at[result.index[i], "active_bearish_ob_bottom"] = bearish_order_blocks[0].bottom

        if detect_fvg and i >= 2:
            bullish_fvg = (
                low_values[i] > high_values[i - 2]
                and close_values[i - 1] > high_values[i - 2]
                and bar_delta_percent.iloc[i] > threshold.iloc[i]
            )
            bearish_fvg = (
                high_values[i] < low_values[i - 2]
                and close_values[i - 1] < low_values[i - 2]
                and -bar_delta_percent.iloc[i] > threshold.iloc[i]
            )

            if bullish_fvg:
                gap = FairValueGap(
                    top=float(low_values[i]),
                    bottom=float(high_values[i - 2]),
                    bias=BULLISH,
                    created_index=i,
                )
                bullish_fvgs.insert(0, gap)
                result.at[result.index[i], "bullish_fvg"] = True
                result.at[result.index[i], "bullish_fvg_top"] = gap.top
                result.at[result.index[i], "bullish_fvg_bottom"] = gap.bottom

            if bearish_fvg:
                gap = FairValueGap(
                    top=float(low_values[i - 2]),
                    bottom=float(high_values[i]),
                    bias=BEARISH,
                    created_index=i,
                )
                bearish_fvgs.insert(0, gap)
                result.at[result.index[i], "bearish_fvg"] = True
                result.at[result.index[i], "bearish_fvg_top"] = gap.top
                result.at[result.index[i], "bearish_fvg_bottom"] = gap.bottom

        bullish_fvgs = [gap for gap in bullish_fvgs if low_values[i] >= gap.bottom]
        bearish_fvgs = [gap for gap in bearish_fvgs if high_values[i] <= gap.top]

        if bullish_fvgs:
            result.at[result.index[i], "active_bullish_fvg_top"] = bullish_fvgs[0].top
            result.at[result.index[i], "active_bullish_fvg_bottom"] = bullish_fvgs[0].bottom
        if bearish_fvgs:
            result.at[result.index[i], "active_bearish_fvg_top"] = bearish_fvgs[0].top
            result.at[result.index[i], "active_bearish_fvg_bottom"] = bearish_fvgs[0].bottom

    return result


if __name__ == "__main__":
    sample = pd.read_csv("data/QQQ.csv")
    sample["time_key"] = pd.to_datetime(sample["time_key"])
    smc = add_smc_signals(sample)
    signal_columns = [
        "time_key",
        "close",
        "swing_bullish_bos",
        "swing_bullish_choch",
        "swing_bearish_bos",
        "swing_bearish_choch",
        "bullish_order_block_created",
        "bearish_order_block_created",
        "bullish_fvg",
        "bearish_fvg",
    ]
    triggered = smc.loc[
        smc[
            [
                "swing_bullish_bos",
                "swing_bullish_choch",
                "swing_bearish_bos",
                "swing_bearish_choch",
                "bullish_order_block_created",
                "bearish_order_block_created",
                "bullish_fvg",
                "bearish_fvg",
            ]
        ].any(axis=1),
        signal_columns,
    ]
    print(triggered.tail(10).to_string(index=False))

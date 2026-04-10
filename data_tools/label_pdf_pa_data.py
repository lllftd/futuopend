from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


STATE_LABELS = {
    0: "bull_conv",
    1: "bull_div",
    2: "bear_conv",
    3: "bear_div",
    4: "range_conv",
    5: "range_div",
}


@dataclass(frozen=True)
class LabelConfig:
    input_dir: Path
    symbols: tuple[str, ...]
    output_suffix: str
    pressure_window: int
    swing_left: int
    swing_right: int
    atr_period: int
    ma_period: int
    tr_lookback: int
    hold_bars: int
    atr_multiplier_tp: float
    atr_multiplier_sl: float
    warmup_bars: int


def parse_args() -> LabelConfig:
    parser = argparse.ArgumentParser(
        description="Label QQQ/SPY minute bars with the PDF PA auto-labeling rules."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing source CSV files such as QQQ.csv and SPY.csv.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["QQQ", "SPY"],
        help="Tickers to label. Files are read as {SYMBOL}.csv from --input-dir.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_labeled",
        help="Suffix appended before .csv when writing outputs.",
    )
    parser.add_argument("--pressure-window", type=int, default=10)
    parser.add_argument("--swing-left", type=int, default=5)
    parser.add_argument("--swing-right", type=int, default=3)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--ma-period", type=int, default=20)
    parser.add_argument("--tr-lookback", type=int, default=40)
    parser.add_argument("--hold-bars", type=int, default=30)
    parser.add_argument("--atr-multiplier-tp", type=float, default=2.0)
    parser.add_argument("--atr-multiplier-sl", type=float, default=1.0)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=50,
        help="Drop the first N bars after labeling to remove warmup noise.",
    )
    args = parser.parse_args()

    return LabelConfig(
        input_dir=Path(args.input_dir),
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        output_suffix=args.output_suffix,
        pressure_window=args.pressure_window,
        swing_left=args.swing_left,
        swing_right=args.swing_right,
        atr_period=args.atr_period,
        ma_period=args.ma_period,
        tr_lookback=args.tr_lookback,
        hold_bars=args.hold_bars,
        atr_multiplier_tp=args.atr_multiplier_tp,
        atr_multiplier_sl=args.atr_multiplier_sl,
        warmup_bars=args.warmup_bars,
    )


def compute_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["body"] = df["close"] - df["open"]
    df["body_abs"] = df["body"].abs()
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-10)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_ratio"] = df["body_abs"] / df["range"]
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"]

    df["is_trend_bar"] = (df["body_ratio"] > 0.6).astype(int)
    df["is_doji"] = (df["body_ratio"] < 0.3).astype(int)
    df["is_bull"] = (df["close"] > df["open"]).astype(int)
    df["is_bear"] = (df["close"] < df["open"]).astype(int)

    df["is_strong_bull_trend"] = (
        (df["is_bull"] == 1)
        & (df["body_ratio"] > 0.6)
        & (df["upper_wick_ratio"] < 0.2)
    ).astype(int)
    df["is_strong_bear_trend"] = (
        (df["is_bear"] == 1)
        & (df["body_ratio"] > 0.6)
        & (df["lower_wick_ratio"] < 0.2)
    ).astype(int)

    df["close_position"] = (df["close"] - df["low"]) / df["range"]
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
    df["higher_low"] = (df["low"] > df["low"].shift(1)).astype(int)
    df["lower_high"] = (df["high"] < df["high"].shift(1)).astype(int)

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["is_inside_bar"] = ((df["high"] <= prev_high) & (df["low"] >= prev_low)).astype(int)
    df["is_outside_bar"] = ((df["high"] > prev_high) & (df["low"] < prev_low)).astype(int)

    df["gap_up"] = (df["low"] > prev_high).astype(int)
    df["gap_down"] = (df["high"] < prev_low).astype(int)
    df["body_gap_up"] = (df["open"] > df["close"].shift(1)).astype(int)
    df["body_gap_down"] = (df["open"] < df["close"].shift(1)).astype(int)

    overlap_top = np.minimum(df["high"], prev_high)
    overlap_bottom = np.maximum(df["low"], prev_low)
    df["overlap"] = np.maximum(0.0, overlap_top - overlap_bottom) / df["range"]

    return df


def compute_pressure(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()

    bull_body = df["body_abs"].where(df["is_bull"] == 1, 0.0)
    bear_body = df["body_abs"].where(df["is_bear"] == 1, 0.0)

    bull_groups = (df["is_bull"] != df["is_bull"].shift()).cumsum()
    df["consec_bull"] = df["is_bull"].groupby(bull_groups).cumcount() + 1
    df.loc[df["is_bull"] == 0, "consec_bull"] = 0

    bear_groups = (df["is_bear"] != df["is_bear"].shift()).cumsum()
    df["consec_bear"] = df["is_bear"].groupby(bear_groups).cumcount() + 1
    df.loc[df["is_bear"] == 0, "consec_bear"] = 0

    df["bull_ratio"] = df["is_bull"].rolling(window, min_periods=1).mean()
    df["bear_ratio"] = df["is_bear"].rolling(window, min_periods=1).mean()
    df["bull_body_avg"] = bull_body.rolling(window, min_periods=1).mean()
    df["bear_body_avg"] = bear_body.rolling(window, min_periods=1).mean()
    df["body_dominance"] = df["bull_body_avg"] / (df["bear_body_avg"] + 1e-10)
    df["avg_close_pos"] = df["close_position"].rolling(window, min_periods=1).mean()
    df["avg_lower_wick"] = df["lower_wick_ratio"].rolling(window, min_periods=1).mean()
    df["avg_upper_wick"] = df["upper_wick_ratio"].rolling(window, min_periods=1).mean()

    half_window = max(2, window // 2)
    recent_bull = bull_body.rolling(half_window, min_periods=1).mean()
    earlier_bull = bull_body.shift(half_window).rolling(half_window, min_periods=1).mean()
    recent_bear = bear_body.rolling(half_window, min_periods=1).mean()
    earlier_bear = bear_body.shift(half_window).rolling(half_window, min_periods=1).mean()
    df["bull_body_growing"] = (recent_bull > earlier_bull).astype(int)
    df["bear_body_growing"] = (recent_bear > earlier_bear).astype(int)

    df["buy_pressure"] = (
        df["bull_ratio"] * 25
        + (df["body_dominance"].clip(0, 3) / 3) * 25
        + df["avg_close_pos"] * 25
        + df["avg_lower_wick"] * 25
    )
    df["sell_pressure"] = (
        df["bear_ratio"] * 25
        + (1 / (df["body_dominance"].clip(0.33, 3) + 1e-10)).clip(0, 1) * 25
        + (1 - df["avg_close_pos"]) * 25
        + df["avg_upper_wick"] * 25
    )

    return df


def find_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["swing_high"] = np.nan
    df["swing_low"] = np.nan

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    for i in range(left, len(df) - right):
        high_window = highs[i - left : i + right + 1]
        low_window = lows[i - left : i + right + 1]

        if highs[i] == np.max(high_window):
            df.iat[i, df.columns.get_loc("swing_high")] = highs[i]
        if lows[i] == np.min(low_window):
            df.iat[i, df.columns.get_loc("swing_low")] = lows[i]

    return df


def build_swing_structure(df: pd.DataFrame, right: int = 3) -> pd.DataFrame:
    df = df.copy()
    
    # 防未来函数：只有当波段极值点形成 `right` 根 K 线之后，我们才能确认并使用它。
    # 比如在 i 根识别到了高点，这个高点其实是 i-right 那根 K 线的，直到第 i 根我们才能确认。
    # 因此我们需要将识别到的极值向右平移 `right` 格，然后再 ffill 向前填充，防止未来信息穿越。
    
    shifted_high = df["swing_high"].shift(right)
    shifted_low = df["swing_low"].shift(right)
    
    df["last_swing_high"] = shifted_high.ffill()
    df["last_swing_low"] = shifted_low.ffill()

    sh = df.dropna(subset=["swing_high"])["swing_high"]
    sl = df.dropna(subset=["swing_low"])["swing_low"]

    swing_high_types: list[str] = []
    for i, value in enumerate(sh.to_numpy(dtype=float)):
        if i == 0:
            swing_high_types.append("NA")
        else:
            prev_value = sh.iloc[i - 1]
            swing_high_types.append("HH" if value > prev_value else "LH")
    sh_df = pd.DataFrame({"swing_high_type": swing_high_types}, index=sh.index)
    df = df.join(sh_df)
    df["swing_high_type"] = df["swing_high_type"].ffill().fillna("NA")

    swing_low_types: list[str] = []
    for i, value in enumerate(sl.to_numpy(dtype=float)):
        if i == 0:
            swing_low_types.append("NA")
        else:
            prev_value = sl.iloc[i - 1]
            swing_low_types.append("HL" if value > prev_value else "LL")
    sl_df = pd.DataFrame({"swing_low_type": swing_low_types}, index=sl.index)
    df = df.join(sl_df)
    df["swing_low_type"] = df["swing_low_type"].ffill().fillna("NA")

    return df


def label_market_state(
    df: pd.DataFrame,
    atr_period: int = 20,
    ma_period: int = 20,
) -> pd.DataFrame:
    """
    Labels market state using KMeans clustering on continuous features:
    - MA slope
    - ATR percentile
    - Bollinger Band width
    - Volume trend
    
    Replaces old heuristic rules (HH/HL/LH/LL swing comparisons).
    """
    df = df.copy()
    
    # 1. Compute continuous features for clustering
    # MA Slope (Direction)
    df["ma"] = df["close"].rolling(ma_period, min_periods=1).mean()
    df["ma_slope"] = (df["ma"] - df["ma"].shift(5)) / df["close"] * 1000
    df["ma_slope_smooth"] = df["ma_slope"].rolling(3, min_periods=1).mean()

    # ATR Percentile (Volatility)
    df["atr"] = df["range"].rolling(atr_period, min_periods=1).mean()
    df["atr_100"] = df["atr"].rolling(100, min_periods=1).mean()
    df["atr_percentile"] = df["atr"] / (df["atr_100"] + 1e-8)

    # Bollinger Band Width (Convergence/Divergence)
    std = df["close"].rolling(ma_period, min_periods=1).std()
    df["bb_width"] = (std * 2) / (df["ma"] + 1e-8)
    
    # Volume feature
    vol_ma = df["volume"].rolling(ma_period, min_periods=1).mean()
    df["vol_ratio"] = df["volume"] / (vol_ma + 1e-8)
    df["vol_ratio_smooth"] = df["vol_ratio"].rolling(5, min_periods=1).mean()
    
    # 2. Prepare data for clustering
    dir_cols = ["ma_slope_smooth"]
    vol_cols = ["atr_percentile", "bb_width", "vol_ratio_smooth"]
    
    df["dir_label"] = 2  # Default to Range
    df["vol_label"] = 0  # Default to Converging
    
    valid_mask = df[dir_cols + vol_cols].notna().all(axis=1)
    if not valid_mask.any():
        return df
        
    X_dir = df.loc[valid_mask, dir_cols].values
    X_vol = df.loc[valid_mask, vol_cols].values
    
    chunk_size = 10000
    dir_mapped = np.full(len(X_dir), 2)
    vol_mapped = np.full(len(X_vol), 0)
    
    # OPTIMIZATION: RobustScaler handles Fat Tails without hard 5%-95% truncation
    from sklearn.preprocessing import RobustScaler
    scaler_dir = RobustScaler()
    scaler_vol = RobustScaler()
    
    kmeans_dir = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_vol = KMeans(n_clusters=2, random_state=42, n_init=10)
    
    for i in range(0, len(X_dir), chunk_size):
        end_idx = min(i + chunk_size, len(X_dir))
        
        # --- DIRECTION STREAM ---
        Xd_chunk = X_dir[i:end_idx]
        fit_start = max(0, i - 100000)
        Xd_past = X_dir[fit_start:i] if i > 0 else Xd_chunk
        
        scaler_dir.fit(Xd_past)
        kmeans_dir.fit(scaler_dir.transform(Xd_past))
        
        dir_clusters = kmeans_dir.predict(scaler_dir.transform(Xd_chunk))
        d_centers = kmeans_dir.cluster_centers_[:, 0]
        
        bull_c = np.argmax(d_centers)
        bear_c = np.argmin(d_centers)
        range_c = [c for c in range(3) if c not in (bull_c, bear_c)][0]
        
        d_map = {bull_c: 0, bear_c: 1, range_c: 2}
        dir_mapped[i:end_idx] = [d_map[c] for c in dir_clusters]
        
        # --- VOLATILITY STREAM ---
        Xv_chunk = X_vol[i:end_idx]
        Xv_past = X_vol[fit_start:i] if i > 0 else Xv_chunk
        
        scaler_vol.fit(Xv_past)
        kmeans_vol.fit(scaler_vol.transform(Xv_past))
        
        vol_clusters = kmeans_vol.predict(scaler_vol.transform(Xv_chunk))
        # Use sum of scaled centers (BB width + ATR) to determine Divergence
        v_centers = kmeans_vol.cluster_centers_.sum(axis=1)
        div_c = np.argmax(v_centers)
        conv_c = np.argmin(v_centers)
        
        v_map = {conv_c: 0, div_c: 1}
        vol_mapped[i:end_idx] = [v_map[c] for c in vol_clusters]
        
    df.loc[valid_mask, "dir_label"] = dir_mapped
    df.loc[valid_mask, "vol_label"] = vol_mapped
    
    # Smooth ground truth (Mode of 3)
    df["dir_label"] = df["dir_label"].rolling(3, min_periods=1).apply(lambda x: pd.Series(x).mode()[0], raw=False).astype(int)
    df["vol_label"] = df["vol_label"].rolling(3, min_periods=1).apply(lambda x: pd.Series(x).mode()[0], raw=False).astype(int)

    # Map (dir_label, vol_label) directly to 6 classes
    # dir_label: 0=bull, 1=bear, 2=range
    # vol_label: 0=conv, 1=div
    # 0: bull_conv, 1: bull_div, 2: bear_conv, 3: bear_div, 4: range_conv, 5: range_div
    df["market_state"] = df["dir_label"] * 2 + df["vol_label"]
    df["market_state_name"] = df["market_state"].map(STATE_LABELS)

    df["above_ma"] = (df["close"] > df["ma"]).astype(int)
    return df


def label_trading_range(
    df: pd.DataFrame,
    lookback: int = 40,
) -> pd.DataFrame:
    df = df.copy()
    df["rolling_high"] = df["high"].rolling(lookback, min_periods=1).max()
    df["rolling_low"] = df["low"].rolling(lookback, min_periods=1).min()
    df["rolling_range"] = df["rolling_high"] - df["rolling_low"]
    df["tr_position"] = (df["close"] - df["rolling_low"]) / (df["rolling_range"] + 1e-10)

    df["direction_changes"] = 0
    for i in range(1, 11):
        df["direction_changes"] += (df["is_bull"] != df["is_bull"].shift(i)).astype(int)

    # We completely trust KMeans and no longer override states to pure trading range (2).
    # The 4 classes (0=bull, 1=bear, 3=converging, 4=diverging) are preserved here.

    return df


def label_pullback_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    h_counts = np.zeros(n, dtype=int)
    l_counts = np.zeros(n, dtype=int)

    states = df["market_state"].to_numpy(dtype=int)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    h_count = 0
    l_count = 0
    in_pullback_bull = False
    in_pullback_bear = False

    for i in range(1, n):
        state = states[i]
        if state in (0, 1):
            if lows[i] < lows[i - 1]:
                in_pullback_bull = True
            if in_pullback_bull and highs[i] > highs[i - 1]:
                h_count += 1
                in_pullback_bull = False
            h_counts[i] = h_count
            l_count = 0
            in_pullback_bear = False
        elif state in (2, 3):
            if highs[i] > highs[i - 1]:
                in_pullback_bear = True
            if in_pullback_bear and lows[i] < lows[i - 1]:
                l_count += 1
                in_pullback_bear = False
            l_counts[i] = l_count
            h_count = 0
            in_pullback_bull = False
        else:
            h_count = 0
            l_count = 0
            in_pullback_bull = False
            in_pullback_bear = False

    df["h_count"] = h_counts
    df["l_count"] = l_counts
    return df


def label_pullback_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    pullback_bars = np.zeros(n, dtype=int)

    states = df["market_state"].to_numpy(dtype=int)
    closes = df["close"].to_numpy(dtype=float)
    count = 0

    for i in range(1, n):
        state = states[i]
        if state in (0, 1):
            count = count + 1 if closes[i] < closes[i - 1] else 0
        elif state in (2, 3):
            count = count + 1 if closes[i] > closes[i - 1] else 0
        else:
            count = 0
        pullback_bars[i] = count

    df["pullback_bars"] = pullback_bars
    df["endless_pullback"] = (df["pullback_bars"] >= 20).astype(int)
    return df


def label_quality_breakouts(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    
    # 1. consecutive strong bars
    strong_bull_2 = ((df["is_strong_bull_trend"] == 1) & (df["is_strong_bull_trend"].shift(1) == 1)).astype(int)
    strong_bear_2 = ((df["is_strong_bear_trend"] == 1) & (df["is_strong_bear_trend"].shift(1) == 1)).astype(int)
    
    # 2. level break (breakout past 20-bar high/low)
    past_high_20 = df["high"].shift(1).rolling(lookback, min_periods=1).max()
    past_low_20 = df["low"].shift(1).rolling(lookback, min_periods=1).min()
    level_break_bull = (df["close"] > past_high_20).astype(int)
    level_break_bear = (df["close"] < past_low_20).astype(int)
    
    # 3. Volume confirmation
    avg_vol_20 = df["volume"].shift(1).rolling(lookback, min_periods=1).mean()
    vol_confirm = (df["volume"] > 1.5 * avg_vol_20).astype(int)
    
    # 4. Compression (ATR of last 10 bars < 0.7 * ATR of previous 20 bars)
    atr_10 = df["range"].rolling(10, min_periods=1).mean()
    atr_20_past = df["range"].shift(10).rolling(20, min_periods=1).mean()
    compression = (atr_10 < 0.7 * atr_20_past).astype(int)
    
    bull_score = strong_bull_2 + level_break_bull + vol_confirm + compression
    bear_score = strong_bear_2 + level_break_bear + vol_confirm + compression
    
    df["quality_bull_breakout"] = (bull_score >= 3).astype(int)
    df["quality_bear_breakout"] = (bear_score >= 3).astype(int)
    return df


def label_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    signal = np.zeros(n, dtype=int)
    signal_strength = np.zeros(n, dtype=int)
    signal_reason = [""] * n

    states = df["market_state"].to_numpy(dtype=int)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ma = df["ma"].to_numpy(dtype=float)
    tr_position = df["tr_position"].to_numpy(dtype=float)
    is_bull = df["is_bull"].to_numpy(dtype=int)
    is_bear = df["is_bear"].to_numpy(dtype=int)
    strong_bull = df["is_strong_bull_trend"].to_numpy(dtype=int)
    strong_bear = df["is_strong_bear_trend"].to_numpy(dtype=int)
    quality_bull = df["quality_bull_breakout"].to_numpy(dtype=int)
    quality_bear = df["quality_bear_breakout"].to_numpy(dtype=int)
    h_count = df["h_count"].to_numpy(dtype=int)
    l_count = df["l_count"].to_numpy(dtype=int)
    buy_pressure = df["buy_pressure"].to_numpy(dtype=float)
    sell_pressure = df["sell_pressure"].to_numpy(dtype=float)
    endless_pullback = df["endless_pullback"].to_numpy(dtype=int)

    for i in range(2, n):
        state = states[i]
        strength = 0
        reason: list[str] = []
        cur_signal = 0

        if (
            quality_bull[i] == 1
        ):
            cur_signal = 1
            strength = 90
            reason.append("E01_quality_bull_breakout")
        elif (
            state in (0, 1)
            and h_count[i] == 2
            and close[i] > open_[i]
            and close[i] > ma[i]
        ):
            cur_signal = 1
            strength = 70
            reason.append("E03_H2_bull_pullback")
        elif (
            state in (4, 5)
            and tr_position[i] < 0.3
            and is_bull[i] == 1
            and buy_pressure[i] > 50
        ):
            cur_signal = 1
            strength = 50
            reason.append("E13_TR_bottom_buy")
        elif (
            state in (0, 1)
            and h_count[i] == 1
            and close[i] > open_[i]
            and buy_pressure[i] > 60
        ):
            cur_signal = 1
            strength = 50
            reason.append("H1_bull_pullback")
        elif (
            quality_bear[i] == 1
        ):
            cur_signal = -1
            strength = 90
            reason.append("E01_quality_bear_breakout")
        elif (
            state in (2, 3)
            and l_count[i] == 2
            and close[i] < open_[i]
            and close[i] < ma[i]
        ):
            cur_signal = -1
            strength = 70
            reason.append("L2_bear_pullback")
        elif (
            state in (4, 5)
            and tr_position[i] > 0.7
            and is_bear[i] == 1
            and sell_pressure[i] > 50
        ):
            cur_signal = -1
            strength = 50
            reason.append("E13_TR_top_sell")

        if endless_pullback[i] == 1:
            cur_signal = 0
            strength = 0
            reason = ["E22_endless_pullback_skip"]

        signal[i] = cur_signal
        signal_strength[i] = strength
        signal_reason[i] = "|".join(reason)

    df["signal"] = signal
    df["signal_strength"] = signal_strength
    df["signal_reason"] = signal_reason
    return df


def label_outcomes(
    df: pd.DataFrame,
    hold_bars: int = 30,
    atr_multiplier_tp: float = 2.0,
    atr_multiplier_sl: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    outcome = np.zeros(n, dtype=int)
    max_favorable = np.zeros(n, dtype=float)
    max_adverse = np.zeros(n, dtype=float)
    exit_bar = np.zeros(n, dtype=int)
    reward_risk = np.zeros(n, dtype=float)

    signal = df["signal"].to_numpy(dtype=int)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = df["atr"].to_numpy(dtype=float)

    for i in range(n):
        if signal[i] == 0:
            continue

        if i + 1 < n:
            entry_price = open_[i + 1] # 使用下一根K线的开盘价进场
        else:
            continue # 最后一根K线无法进场
            
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        sl_distance = atr * atr_multiplier_sl
        tp_distance = atr * atr_multiplier_tp

        if signal[i] == 1:
            tp = entry_price + tp_distance
            sl = entry_price - sl_distance
        else:
            tp = entry_price - tp_distance
            sl = entry_price + sl_distance

        max_fav = 0.0
        max_adv = 0.0
        last_exit_bar = 0

        for j in range(i + 1, min(i + 1 + hold_bars, n)):
            last_exit_bar = j - i
            if signal[i] == 1:
                favorable = high[j] - entry_price
                adverse = entry_price - low[j]
                max_fav = max(max_fav, favorable)
                max_adv = max(max_adv, adverse)
                if low[j] <= sl:
                    outcome[i] = -1
                    exit_bar[i] = last_exit_bar
                    break
                if high[j] >= tp:
                    outcome[i] = 1
                    exit_bar[i] = last_exit_bar
                    break
            else:
                favorable = entry_price - low[j]
                adverse = high[j] - entry_price
                max_fav = max(max_fav, favorable)
                max_adv = max(max_adv, adverse)
                if high[j] >= sl:
                    outcome[i] = -1
                    exit_bar[i] = last_exit_bar
                    break
                if low[j] <= tp:
                    outcome[i] = 1
                    exit_bar[i] = last_exit_bar
                    break
        else:
            exit_bar[i] = last_exit_bar

        max_favorable[i] = max_fav
        max_adverse[i] = max_adv
        if sl_distance > 0:
            reward_risk[i] = max_fav / sl_distance

    df["outcome"] = outcome
    df["max_favorable"] = max_favorable
    df["max_adverse"] = max_adverse
    df["exit_bar"] = exit_bar
    df["reward_risk"] = reward_risk
    return df


def auto_label_pipeline(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    labeled = df.copy()
    labeled = compute_bar_features(labeled)
    labeled = compute_pressure(labeled, window=config.pressure_window)
    labeled = find_swing_points(labeled, left=config.swing_left, right=config.swing_right)
    labeled = build_swing_structure(labeled, right=config.swing_right)
    labeled = label_market_state(
        labeled,
        atr_period=config.atr_period,
        ma_period=config.ma_period,
    )
    labeled = label_trading_range(labeled, lookback=config.tr_lookback)
    labeled = label_pullback_count(labeled)
    labeled = label_pullback_bars(labeled)
    labeled = label_quality_breakouts(labeled)
    labeled = label_signals(labeled)
    labeled = label_outcomes(
        labeled,
        hold_bars=config.hold_bars,
        atr_multiplier_tp=config.atr_multiplier_tp,
        atr_multiplier_sl=config.atr_multiplier_sl,
    )
    if config.warmup_bars > 0:
        labeled = labeled.iloc[config.warmup_bars :].reset_index(drop=True)
    return labeled


def prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input file is missing required columns: {missing_list}")

    df = frame.copy()
    if "time_key" in df.columns:
        df["time_key"] = pd.to_datetime(df["time_key"])
        df = df.sort_values("time_key", kind="mergesort").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def label_symbol_csv(config: LabelConfig, symbol: str) -> tuple[Path, int, int]:
    input_path = config.input_dir / f"{symbol}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    original = pd.read_csv(input_path)
    prepared = prepare_input(original)
    labeled = auto_label_pipeline(prepared, config)

    output_path = config.input_dir / f"{symbol}{config.output_suffix}.csv"
    labeled_to_write = labeled.copy()
    if "time_key" in labeled_to_write.columns:
        labeled_to_write["time_key"] = labeled_to_write["time_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
    labeled_to_write.to_csv(output_path, index=False, encoding="utf-8")

    signal_count = int((labeled["signal"] != 0).sum())
    return output_path, len(labeled), signal_count


def main() -> int:
    config = parse_args()
    config.input_dir.mkdir(parents=True, exist_ok=True)

    try:
        for symbol in config.symbols:
            output_path, row_count, signal_count = label_symbol_csv(config, symbol)
            print(f"[{symbol}] wrote {row_count} labeled rows -> {output_path}")
            print(f"[{symbol}] non-zero signals: {signal_count}")
    except Exception as exc:
        print(f"Labeling failed: {exc}")
        return 1

    print("All requested symbols labeled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core.pa_rules import _hmm_garch_features


def _tq(it, **kwargs):
    """Progress bar; set DISABLE_TQDM=1 to disable (same convention as training scripts)."""
    if os.environ.get("DISABLE_TQDM", "").strip() in {"1", "true", "yes"}:
        return it
    return tqdm(it, **kwargs)

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
    decision_horizon_bars: int
    theta_start_bars: int
    theta_decay_bars: float
    adverse_penalty: float
    atr_multiplier_tp: float
    atr_multiplier_sl: float
    warmup_bars: int

def parse_args() -> LabelConfig:
    parser = argparse.ArgumentParser(description="Label v2 pipeline with HMM regime.")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    parser.add_argument("--output-suffix", default="_labeled_v2")
    parser.add_argument("--pressure-window", type=int, default=10)
    parser.add_argument("--swing-left", type=int, default=5)
    parser.add_argument("--swing-right", type=int, default=3)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--ma-period", type=int, default=20)
    parser.add_argument("--tr-lookback", type=int, default=40)
    parser.add_argument("--hold-bars", type=int, default=30)
    parser.add_argument("--decision-horizon-bars", type=int, default=20)
    parser.add_argument("--theta-start-bars", type=int, default=15)
    parser.add_argument("--theta-decay-bars", type=float, default=5.0)
    parser.add_argument("--adverse-penalty", type=float, default=1.0)
    parser.add_argument("--atr-multiplier-tp", type=float, default=2.0)
    parser.add_argument("--atr-multiplier-sl", type=float, default=1.0)
    parser.add_argument("--warmup-bars", type=int, default=50)
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
        decision_horizon_bars=args.decision_horizon_bars,
        theta_start_bars=args.theta_start_bars,
        theta_decay_bars=args.theta_decay_bars,
        adverse_penalty=args.adverse_penalty,
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

    df["is_bull"] = (df["close"] > df["open"]).astype(int)
    df["is_bear"] = (df["close"] < df["open"]).astype(int)
    df["is_strong_bull_trend"] = ((df["is_bull"] == 1) & (df["body_ratio"] > 0.6) & (df["upper_wick_ratio"] < 0.2)).astype(int)
    df["is_strong_bear_trend"] = ((df["is_bear"] == 1) & (df["body_ratio"] > 0.6) & (df["lower_wick_ratio"] < 0.2)).astype(int)
    df["close_position"] = (df["close"] - df["low"]) / df["range"]
    
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["is_inside_bar"] = ((df["high"] <= prev_high) & (df["low"] >= prev_low)).astype(int)
    
    df["ma"] = df["close"].rolling(20, min_periods=1).mean()
    df["atr"] = df["range"].rolling(20, min_periods=1).mean()
    return df

def compute_pressure(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    bull_body = df["body_abs"].where(df["is_bull"] == 1, 0.0)
    bear_body = df["body_abs"].where(df["is_bear"] == 1, 0.0)

    df["bull_ratio"] = df["is_bull"].rolling(window, min_periods=1).mean()
    df["bear_ratio"] = df["is_bear"].rolling(window, min_periods=1).mean()
    df["bull_body_avg"] = bull_body.rolling(window, min_periods=1).mean()
    df["bear_body_avg"] = bear_body.rolling(window, min_periods=1).mean()
    df["body_dominance"] = df["bull_body_avg"] / (df["bear_body_avg"] + 1e-10)
    df["avg_close_pos"] = df["close_position"].rolling(window, min_periods=1).mean()
    df["avg_lower_wick"] = df["lower_wick_ratio"].rolling(window, min_periods=1).mean()
    df["avg_upper_wick"] = df["upper_wick_ratio"].rolling(window, min_periods=1).mean()

    df["buy_pressure"] = (
        df["bull_ratio"] * 25 + (df["body_dominance"].clip(0, 3) / 3) * 25 +
        df["avg_close_pos"] * 25 + df["avg_lower_wick"] * 25
    )
    df["sell_pressure"] = (
        df["bear_ratio"] * 25 + (1 / (df["body_dominance"].clip(0.33, 3) + 1e-10)).clip(0, 1) * 25 +
        (1 - df["avg_close_pos"]) * 25 + df["avg_upper_wick"] * 25
    )
    return df

def label_hmm_regime(df: pd.DataFrame) -> pd.DataFrame:
    """① HMM regime (替代 KMeans)"""
    df = df.copy()
    hmm_df = _hmm_garch_features(df)
    
    # pa_hmm_state:
    # 0: Low Vol Bull -> 0: bull_conv
    # 1: High Vol Bull -> 1: bull_div
    # 2: Low Vol Bear -> 2: bear_conv
    # 3: High Vol Bear -> 3: bear_div
    # 4: Low Vol Range -> 4: range_conv
    # 5: High Vol Range -> 5: range_div
    
    df["market_state"] = hmm_df["pa_hmm_state"]
    df["market_state_name"] = df["market_state"].map(STATE_LABELS)
    return df

def label_trading_range(df: pd.DataFrame, lookback: int = 40) -> pd.DataFrame:
    df = df.copy()
    df["rolling_high"] = df["high"].rolling(lookback, min_periods=1).max()
    df["rolling_low"] = df["low"].rolling(lookback, min_periods=1).min()
    df["rolling_range"] = df["rolling_high"] - df["rolling_low"]
    df["tr_position"] = (df["close"] - df["rolling_low"]) / (df["rolling_range"] + 1e-10)
    return df

def label_pullback_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    h_counts = np.zeros(n, dtype=int)
    l_counts = np.zeros(n, dtype=int)

    states = df["market_state"].to_numpy(dtype=int)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    h_count = l_count = 0
    in_pullback_bull = in_pullback_bear = False

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
            h_count = l_count = 0
            in_pullback_bull = in_pullback_bear = False

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
    strong_bull_2 = ((df["is_strong_bull_trend"] == 1) & (df["is_strong_bull_trend"].shift(1) == 1)).astype(int)
    strong_bear_2 = ((df["is_strong_bear_trend"] == 1) & (df["is_strong_bear_trend"].shift(1) == 1)).astype(int)
    past_high_20 = df["high"].shift(1).rolling(lookback, min_periods=1).max()
    past_low_20 = df["low"].shift(1).rolling(lookback, min_periods=1).min()
    level_break_bull = (df["close"] > past_high_20).astype(int)
    level_break_bear = (df["close"] < past_low_20).astype(int)
    avg_vol_20 = df["volume"].shift(1).rolling(lookback, min_periods=1).mean()
    vol_confirm = (df["volume"] > 1.5 * avg_vol_20).astype(int)
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

        if quality_bull[i] == 1:
            cur_signal = 1
            strength = 90
            reason.append("E01_quality_bull_breakout")
        elif state in (0, 1) and h_count[i] == 2 and close[i] > open_[i] and close[i] > ma[i]:
            cur_signal = 1
            strength = 70
            reason.append("E03_H2_bull_pullback")
        elif state in (4, 5) and tr_position[i] < 0.3 and is_bull[i] == 1 and buy_pressure[i] > 50:
            cur_signal = 1
            strength = 50
            reason.append("E13_TR_bottom_buy")
        elif state in (0, 1) and h_count[i] == 1 and close[i] > open_[i] and buy_pressure[i] > 60:
            cur_signal = 1
            strength = 50
            reason.append("H1_bull_pullback")
        elif quality_bear[i] == 1:
            cur_signal = -1
            strength = 90
            reason.append("E01_quality_bear_breakout")
        elif state in (2, 3) and l_count[i] == 2 and close[i] < open_[i] and close[i] < ma[i]:
            cur_signal = -1
            strength = 70
            reason.append("L2_bear_pullback")
        elif state in (4, 5) and tr_position[i] > 0.7 and is_bear[i] == 1 and sell_pressure[i] > 50:
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
    decision_horizon_bars: int = 20,
    theta_start_bars: int = 15,
    theta_decay_bars: float = 5.0,
    adverse_penalty: float = 1.0,
    atr_multiplier_tp: float = 2.0,
    atr_multiplier_sl: float = 1.0,
) -> pd.DataFrame:
    """③ 前瞻审计 + 因果隔离: Only labels look into the future for outcome mapping."""
    df = df.copy()
    n = len(df)
    outcome = np.zeros(n, dtype=int)
    max_favorable = np.zeros(n, dtype=float)
    max_adverse = np.zeros(n, dtype=float)
    exit_bar = np.zeros(n, dtype=int)
    reward_risk = np.zeros(n, dtype=float)
    decision_mfe_atr = np.zeros(n, dtype=float)
    decision_mae_atr = np.zeros(n, dtype=float)
    decision_peak_bar = np.zeros(n, dtype=int)
    decision_theta_decay = np.ones(n, dtype=float)
    decision_net_edge_atr = np.zeros(n, dtype=float)
    optimal_tp_atr = np.zeros(n, dtype=float)
    optimal_sl_atr = np.zeros(n, dtype=float)
    optimal_exit_bar = np.zeros(n, dtype=int)
    optimal_net_edge_atr = np.zeros(n, dtype=float)

    signal = df["signal"].to_numpy(dtype=int)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = df["atr"].to_numpy(dtype=float)
    decision_horizon_bars = max(1, int(decision_horizon_bars))
    theta_start_bars = max(0, int(theta_start_bars))
    theta_decay_bars = max(float(theta_decay_bars), 1e-6)
    adverse_penalty = max(float(adverse_penalty), 0.0)

    bar_iter = _tq(range(n), desc="label_outcomes", unit="bar", leave=False, mininterval=0.5)
    for i in bar_iter:
        if signal[i] == 0:
            continue
        if i + 1 < n:
            entry_price = open_[i + 1] # Strict causal execution at next open
        else:
            continue
            
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        sl_distance = atr * atr_multiplier_sl
        tp_distance = atr * atr_multiplier_tp
        tp = entry_price + tp_distance if signal[i] == 1 else entry_price - tp_distance
        sl = entry_price - sl_distance if signal[i] == 1 else entry_price + sl_distance

        max_fav = max_adv = 0.0
        last_exit_bar = 0
        best_decision_fav = 0.0
        best_decision_adv = 0.0
        best_peak_bar = 0
        best_net_edge = float("-inf")
        best_tp = 0.0
        best_sl = 0.0
        best_bar = 0

        for j in range(i + 1, min(i + 1 + hold_bars, n)):
            last_exit_bar = j - i
            if signal[i] == 1:
                favorable = high[j] - entry_price
                adverse = entry_price - low[j]
                max_fav = max(max_fav, favorable)
                max_adv = max(max_adv, adverse)
                favorable_atr = favorable / atr
                adverse_atr = adverse / atr
                if last_exit_bar <= decision_horizon_bars:
                    if favorable_atr > best_decision_fav + 1e-12:
                        best_decision_fav = favorable_atr
                        best_peak_bar = last_exit_bar
                    best_decision_adv = max(best_decision_adv, adverse_atr)
                bar_decay = np.exp(-max(last_exit_bar - theta_start_bars, 0) / theta_decay_bars)
                net_edge = favorable_atr * bar_decay - adverse_penalty * adverse_atr
                if net_edge > best_net_edge + 1e-12:
                    best_net_edge = net_edge
                    best_tp = favorable_atr
                    best_sl = adverse_atr
                    best_bar = last_exit_bar
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
                favorable_atr = favorable / atr
                adverse_atr = adverse / atr
                if last_exit_bar <= decision_horizon_bars:
                    if favorable_atr > best_decision_fav + 1e-12:
                        best_decision_fav = favorable_atr
                        best_peak_bar = last_exit_bar
                    best_decision_adv = max(best_decision_adv, adverse_atr)
                bar_decay = np.exp(-max(last_exit_bar - theta_start_bars, 0) / theta_decay_bars)
                net_edge = favorable_atr * bar_decay - adverse_penalty * adverse_atr
                if net_edge > best_net_edge + 1e-12:
                    best_net_edge = net_edge
                    best_tp = favorable_atr
                    best_sl = adverse_atr
                    best_bar = last_exit_bar
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
        decision_mfe_atr[i] = best_decision_fav
        decision_mae_atr[i] = best_decision_adv
        decision_peak_bar[i] = best_peak_bar
        theta_decay = np.exp(-max(best_peak_bar - theta_start_bars, 0) / theta_decay_bars) if best_peak_bar > 0 else 1.0
        decision_theta_decay[i] = theta_decay
        decision_net_edge_atr[i] = best_decision_fav * theta_decay - adverse_penalty * best_decision_adv
        optimal_tp_atr[i] = best_tp
        optimal_sl_atr[i] = best_sl
        optimal_exit_bar[i] = best_bar
        optimal_net_edge_atr[i] = 0.0 if best_net_edge == float("-inf") else best_net_edge

    df["outcome"] = outcome
    df["max_favorable"] = max_favorable
    df["max_adverse"] = max_adverse
    df["exit_bar"] = exit_bar
    df["reward_risk"] = reward_risk
    df["decision_mfe_atr"] = decision_mfe_atr
    df["decision_mae_atr"] = decision_mae_atr
    df["decision_peak_bar"] = decision_peak_bar
    df["decision_theta_decay"] = decision_theta_decay
    df["decision_net_edge_atr"] = decision_net_edge_atr
    df["optimal_tp_atr"] = optimal_tp_atr
    df["optimal_sl_atr"] = optimal_sl_atr
    df["optimal_exit_bar"] = optimal_exit_bar
    df["optimal_net_edge_atr"] = optimal_net_edge_atr
    return df

def auto_label_pipeline(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    labeled = df.copy()
    steps = [
        ("bar_features", lambda d: compute_bar_features(d)),
        ("pressure", lambda d: compute_pressure(d, window=config.pressure_window)),
        ("hmm_regime", label_hmm_regime),
        ("trading_range", lambda d: label_trading_range(d, lookback=config.tr_lookback)),
        ("pullback_count", label_pullback_count),
        ("pullback_bars", label_pullback_bars),
        ("quality_breakouts", label_quality_breakouts),
        ("signals", label_signals),
        (
            "outcomes",
            lambda d: label_outcomes(
                d,
                hold_bars=config.hold_bars,
                decision_horizon_bars=config.decision_horizon_bars,
                theta_start_bars=config.theta_start_bars,
                theta_decay_bars=config.theta_decay_bars,
                adverse_penalty=config.adverse_penalty,
                atr_multiplier_tp=config.atr_multiplier_tp,
                atr_multiplier_sl=config.atr_multiplier_sl,
            ),
        ),
    ]
    pbar = _tq(steps, desc="label_v2 stages", unit="stage", total=len(steps))
    for name, fn in pbar:
        sp = getattr(pbar, "set_postfix_str", None)
        if callable(sp):
            sp(name, refresh=False)
        labeled = fn(labeled)
    if config.warmup_bars > 0:
        labeled = labeled.iloc[config.warmup_bars :].reset_index(drop=True)
    return labeled

def prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "time_key" in df.columns:
        df["time_key"] = pd.to_datetime(df["time_key"])
        df = df.sort_values("time_key").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df

def label_symbol_csv(config: LabelConfig, symbol: str) -> tuple[Path, int, int]:
    input_path = config.input_dir / f"{symbol}.csv"
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
        for symbol in _tq(config.symbols, desc="label_v2 symbols", unit="sym"):
            output_path, row_count, signal_count = label_symbol_csv(config, symbol)
            print(f"[{symbol}] wrote {row_count} labeled rows -> {output_path}")
            print(f"[{symbol}] non-zero signals: {signal_count}")
    except Exception as exc:
        print(f"Labeling failed: {exc}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from core.utils import MINUTES_PER_YEAR, calculate_max_drawdown
from core.indicators import add_chandelier_exit, add_pseudo_cvd, atr, cvd_divergence_features, kama, zlsma
from core.pa_rules import add_all_pa_features


RAW_ATR_LENGTH = 14


@dataclass(frozen=True)
class RuleParams:
    ce_length: int
    ce_multiplier: float
    trend_mode: str
    tp_atr_multiple: float
    sl_atr_multiple: float
    session_filter: str
    exit_model: str = "atr"
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    zlsma_length: int = 50
    zlsma_offset: int = 0
    kama_er_length: int = 9
    kama_fast_length: int = 2
    kama_slow_length: int = 30
    confirmation_mode: str = "next_bar_body"
    zlsma_slope_threshold: float = 0.0
    atr_percentile_lookback: int = 120
    atr_percentile_min: float = 0.0
    pseudo_cvd_method: str = "clv_body_volume"
    cvd_lookback: int = 20
    cvd_slope_lookback: int = 3
    cvd_classic_divergence: bool = False
    cvd_slope_divergence: bool = False
    time_stop_minutes: int = 30
    force_time_stop: bool = False
    time_progress_threshold: float = 0.5
    profit_lock_trigger_pct: float = 0.0015
    profit_lock_fraction: float = 0.5
    pa_or_filter: bool = False
    pa_or_wide_tp_scale: float = 0.7
    pa_require_signal_bar: bool = False
    pa_require_h2_l2: bool = False
    pa_pressure_min: float = 0.0
    pa_use_mm_target: bool = False
    pa_use_pa_stops: bool = False
    pa_mag_bar_exit: bool = False
    pa_exhaustion_gap_exit: bool = False
    pa_regime_filter: bool = False
    pa_20bar_neutral: bool = False
    pa_ii_breakout_entry: bool = False
    pa_position_sizing_mode: str = "fixed"
    pa_use_env_execution: bool = False
    pa_mtf_trend_filter: bool = False
    pa_mtf_min_rr: float = 0.0
    pa_use_advanced_pa_triggers: bool = False
    pa_env_ce_lookback_tight: int = 1
    pa_env_ce_lookback_wide: int = 2
    pa_env_ce_lookback_tr: int = 0
    pa_env_ce_lookback_neutral: int = 0
    pa_env_price_tol_tight: float = 0.0010
    pa_env_price_tol_wide: float = 0.0015
    pa_env_price_tol_tr: float = 0.0008
    pa_env_price_tol_neutral: float = 0.0006
    pa_env_slope_tol_tight: float = 0.00025
    pa_env_slope_tol_wide: float = 0.00020
    pa_env_slope_tol_tr: float = 0.00035
    pa_env_slope_tol_neutral: float = 0.00040
    pa_env_ce_only_gate: bool = False
    pa_env_relaxed_indicators: bool = False
    pa_trailing_stop_mode: bool = False
    pa_trailing_activation_atr: float = 1.0
    pa_trailing_lock_trend: float = 0.5
    pa_trailing_lock_range: float = 0.8
    pa_timeframe: str = "5min"

    @property
    def strategy_name(self) -> str:
        return (
            f"ce{self.ce_length}_x{self.ce_multiplier:g}_"
            f"{self.trend_mode}_tp{self.tp_atr_multiple:g}_"
            f"sl{self.sl_atr_multiple:g}_{self.session_filter}"
            f"_em{self.exit_model}_tpp{self.tp_pct:g}_slp{self.sl_pct:g}"
            f"_z{self.zlsma_length}_k{self.kama_er_length}-{self.kama_fast_length}-{self.kama_slow_length}"
            f"_cfm{self.confirmation_mode}_zs{self.zlsma_slope_threshold:g}"
            f"_aw{self.atr_percentile_lookback}_ap{self.atr_percentile_min:g}"
            f"_cvd{self.pseudo_cvd_method}_l{self.cvd_lookback}_s{self.cvd_slope_lookback}"
            f"_cd{int(self.cvd_classic_divergence)}_sd{int(self.cvd_slope_divergence)}"
            f"_ts{self.time_stop_minutes}_ft{int(self.force_time_stop)}_pg{self.time_progress_threshold:g}"
            f"_pl{self.profit_lock_trigger_pct:g}_pf{self.profit_lock_fraction:g}"
            f"_paOR{int(self.pa_or_filter)}_paSB{int(self.pa_require_signal_bar)}"
            f"_paHL{int(self.pa_require_h2_l2)}_paPR{self.pa_pressure_min:g}"
            f"_paMM{int(self.pa_use_mm_target)}_paPS{int(self.pa_use_pa_stops)}"
            f"_paMAG{int(self.pa_mag_bar_exit)}_paEG{int(self.pa_exhaustion_gap_exit)}"
            f"_paRF{int(self.pa_regime_filter)}_pa20{int(self.pa_20bar_neutral)}"
            f"_paII{int(self.pa_ii_breakout_entry)}_paSZ{self.pa_position_sizing_mode}"
            f"_paENV{int(self.pa_use_env_execution)}"
            f"_paMTF{int(self.pa_mtf_trend_filter)}_rr{self.pa_mtf_min_rr:g}"
            f"_paADV{int(self.pa_use_advanced_pa_triggers)}"
            f"_eLb{self.pa_env_ce_lookback_tight}-{self.pa_env_ce_lookback_wide}-{self.pa_env_ce_lookback_tr}-{self.pa_env_ce_lookback_neutral}"
            f"_ePt{self.pa_env_price_tol_tight:g}-{self.pa_env_price_tol_wide:g}-{self.pa_env_price_tol_tr:g}-{self.pa_env_price_tol_neutral:g}"
            f"_eSt{self.pa_env_slope_tol_tight:g}-{self.pa_env_slope_tol_wide:g}-{self.pa_env_slope_tol_tr:g}-{self.pa_env_slope_tol_neutral:g}"
            f"_eCE{int(self.pa_env_ce_only_gate)}_eRX{int(self.pa_env_relaxed_indicators)}"
        )


@dataclass
class OptimizationResult:
    summary: dict[str, object]
    equity_curve: pd.Series
    trade_log: pd.DataFrame


def build_base_features(
    df: pd.DataFrame,
    zlsma_length: int = 50,
    zlsma_offset: int = 0,
    kama_er_length: int = 9,
    kama_fast_length: int = 2,
    kama_slow_length: int = 30,
    atr_percentile_lookback: int = 120,
    pseudo_cvd_method: str = "clv_body_volume",
    cvd_lookback: int = 20,
    cvd_slope_lookback: int = 3,
    real_cvd_session: pd.Series | None = None,
    pa_timeframe: str = "5min",
) -> pd.DataFrame:
    result = df.copy()
    result["zlsma"] = zlsma(result, length=zlsma_length, offset=zlsma_offset, source="hlc3")
    result["kama"] = kama(
        result,
        er_length=kama_er_length,
        fast_length=kama_fast_length,
        slow_length=kama_slow_length,
        source="hlc3",
    )
    result["kama_slope"] = result["kama"].pct_change()
    result["zlsma_slope"] = result["zlsma"].pct_change()
    result["atr_raw"] = atr(result, length=RAW_ATR_LENGTH)
    result["atr_percentile"] = rolling_percentile_rank(result["atr_raw"], window=atr_percentile_lookback)
    if real_cvd_session is None:
        cvd_frame = add_pseudo_cvd(result, method=pseudo_cvd_method)[["cvd_pressure", "cvd_delta_proxy", "cvd_session"]]
    else:
        cvd_values = pd.to_numeric(real_cvd_session, errors="coerce").reindex(result.index)
        cvd_frame = pd.DataFrame(
            {
                "cvd_pressure": np.nan,
                "cvd_delta_proxy": np.nan,
                "cvd_session": cvd_values,
            },
            index=result.index,
        )
    result = result.join(cvd_frame)
    result = result.join(
        cvd_divergence_features(
            result,
            cvd_column="cvd_session",
            lookback=cvd_lookback,
            slope_lookback=cvd_slope_lookback,
        )
    )
    result = add_all_pa_features(result, result["atr_raw"], timeframe=pa_timeframe)
    return result


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be greater than 1")

    def _calc(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan
        current = valid[-1]
        return float(np.mean(valid <= current))

    return series.rolling(window=window, min_periods=window).apply(_calc, raw=True)


def apply_ce_features(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    return add_chandelier_exit(df, length=length, multiplier=multiplier, use_close=True, prefix="ce")


def build_exit_prices(entry_price: float, atr_now: float, direction: int, params: RuleParams) -> tuple[float, float]:
    if params.exit_model == "atr":
        tp_distance = params.tp_atr_multiple * atr_now
        sl_distance = params.sl_atr_multiple * atr_now
    elif params.exit_model == "pct":
        tp_distance = entry_price * params.tp_pct
        sl_distance = entry_price * params.sl_pct
    elif params.exit_model == "pct_tp_atr_sl":
        tp_distance = entry_price * params.tp_pct
        sl_distance = params.sl_atr_multiple * atr_now
    elif params.exit_model == "atr_tp_pct_sl":
        tp_distance = params.tp_atr_multiple * atr_now
        sl_distance = entry_price * params.sl_pct
    else:
        raise ValueError(f"Unsupported exit_model: {params.exit_model}")

    if direction == 1:
        stop_price = entry_price - sl_distance if sl_distance > 0 else np.nan
        take_profit = entry_price + tp_distance if tp_distance > 0 else np.nan
    else:
        stop_price = entry_price + sl_distance if sl_distance > 0 else np.nan
        take_profit = entry_price - tp_distance if tp_distance > 0 else np.nan

    return stop_price, take_profit


def apply_mtf_rr_floor(
    entry_price: float,
    stop_price: float,
    take_profit: float,
    direction: int,
    params: RuleParams,
) -> tuple[float, float]:
    """
    Enforce a minimum R:R floor (if configured) to avoid too-close TP.
    """
    rr_floor = float(params.pa_mtf_min_rr)
    if rr_floor <= 0:
        return stop_price, take_profit
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return stop_price, take_profit
    if direction == 1:
        min_tp = entry_price + rr_floor * risk
        take_profit = max(take_profit, min_tp)
    else:
        min_tp = entry_price - rr_floor * risk
        take_profit = min(take_profit, min_tp)
    return stop_price, take_profit


def build_entry_signals(df: pd.DataFrame, params: RuleParams) -> tuple[pd.Series, pd.Series]:
    def _ce_long_active_until_reverse(ce_buy: pd.Series, ce_sell: pd.Series) -> pd.Series:
        ce_buy_b = ce_buy.fillna(False).astype(bool).to_numpy()
        ce_sell_b = ce_sell.fillna(False).astype(bool).to_numpy()
        active = np.zeros(len(ce_buy_b), dtype=bool)
        long_active = False
        for i in range(len(ce_buy_b)):
            if ce_sell_b[i]:
                long_active = False
            if ce_buy_b[i]:
                long_active = True
            active[i] = long_active
        return pd.Series(active, index=ce_buy.index)

    def _ce_short_active_until_reverse(ce_sell: pd.Series, ce_buy: pd.Series) -> pd.Series:
        ce_sell_b = ce_sell.fillna(False).astype(bool).to_numpy()
        ce_buy_b = ce_buy.fillna(False).astype(bool).to_numpy()
        active = np.zeros(len(ce_sell_b), dtype=bool)
        short_active = False
        for i in range(len(ce_sell_b)):
            if ce_buy_b[i]:
                short_active = False
            if ce_sell_b[i]:
                short_active = True
            active[i] = short_active
        return pd.Series(active, index=ce_sell.index)

    long_condition = (df["close"] > df["zlsma"]) & (df["close"] > df["kama"])
    short_condition = (df["close"] < df["zlsma"]) & (df["close"] < df["kama"])

    if params.trend_mode == "stacked_trend":
        long_condition = long_condition & (df["zlsma"] > df["kama"])
        short_condition = short_condition & (df["zlsma"] < df["kama"])

    slope = df["zlsma_slope"].fillna(0.0)
    if params.zlsma_slope_threshold > 0:
        long_allowed = slope >= -params.zlsma_slope_threshold
        short_allowed = slope <= params.zlsma_slope_threshold
    else:
        long_allowed = pd.Series(True, index=df.index)
        short_allowed = pd.Series(True, index=df.index)

    if params.atr_percentile_min > 0:
        atr_allowed = df["atr_percentile"].ge(params.atr_percentile_min).fillna(False)
    else:
        atr_allowed = pd.Series(True, index=df.index)

    if params.pa_env_ce_only_gate:
        raw_long = df["ce_buy_signal"].fillna(False) & atr_allowed
        raw_short = df["ce_sell_signal"].fillna(False) & atr_allowed
    elif getattr(params, "pa_env_relaxed_indicators", False) and "pa_env_state" in df.columns:
        env_state = df["pa_env_state"].astype(str)
        is_trend = env_state.isin(["tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"])
        
        strict_long_cond = long_condition.fillna(False) & long_allowed
        strict_short_cond = short_condition.fillna(False) & short_allowed
        
        # Relaxed conditions for range/TTR: Price > ZLSMA, ignoring KAMA and slope
        relaxed_long_cond = (df["close"] > df["zlsma"]).fillna(False)
        relaxed_short_cond = (df["close"] < df["zlsma"]).fillna(False)
        
        env_long_cond = np.where(is_trend, strict_long_cond, relaxed_long_cond)
        env_short_cond = np.where(is_trend, strict_short_cond, relaxed_short_cond)
        
        raw_long = df["ce_buy_signal"].fillna(False) & pd.Series(env_long_cond, index=df.index) & atr_allowed
        raw_short = df["ce_sell_signal"].fillna(False) & pd.Series(env_short_cond, index=df.index) & atr_allowed
    else:
        raw_long = df["ce_buy_signal"].fillna(False) & long_condition.fillna(False) & long_allowed & atr_allowed
        raw_short = df["ce_sell_signal"].fillna(False) & short_condition.fillna(False) & short_allowed & atr_allowed

    kama_slope = df.get("kama_slope", df["kama"].pct_change()).fillna(0.0)
    kama_above_zlsma = df["kama"].gt(df["zlsma"]).fillna(False)
    kama_below_zlsma = df["kama"].lt(df["zlsma"]).fillna(False)
    ce_long_active = _ce_long_active_until_reverse(df["ce_buy_signal"], df["ce_sell_signal"])
    ce_short_active = _ce_short_active_until_reverse(df["ce_sell_signal"], df["ce_buy_signal"])
    if params.zlsma_slope_threshold > 0:
        kama_slope_long_ok = kama_slope.ge(params.zlsma_slope_threshold)
        kama_slope_short_ok = kama_slope.le(-params.zlsma_slope_threshold)
    else:
        kama_slope_long_ok = kama_slope.gt(0.0)
        kama_slope_short_ok = kama_slope.lt(0.0)
    if not params.pa_env_ce_only_gate:
        long_persist_condition = ce_long_active & kama_above_zlsma & kama_slope_long_ok & atr_allowed
        short_persist_condition = ce_short_active & kama_below_zlsma & kama_slope_short_ok & atr_allowed
        raw_long = raw_long | long_persist_condition
        raw_short = raw_short | short_persist_condition

    if params.cvd_classic_divergence:
        raw_long = raw_long & df["cvd_classic_long_ok"].fillna(True)
        raw_short = raw_short & df["cvd_classic_short_ok"].fillna(True)

    if params.cvd_slope_divergence:
        raw_long = raw_long & df["cvd_slope_long_ok"].fillna(True)
        raw_short = raw_short & df["cvd_slope_short_ok"].fillna(True)

    if params.pa_or_filter and "or_period" in df.columns:
        after_or = ~df["or_period"].fillna(True)
        raw_long = raw_long & after_or
        raw_short = raw_short & after_or

    if params.pa_require_signal_bar and "pa_strong_bull_signal" in df.columns:
        raw_long = raw_long & df["pa_strong_bull_signal"].fillna(False)
        raw_short = raw_short & df["pa_strong_bear_signal"].fillna(False)

    if params.pa_require_h2_l2 and "pa_is_h2_setup" in df.columns:
        raw_long = raw_long & df["pa_is_h2_setup"].fillna(False)
        raw_short = raw_short & df["pa_is_l2_setup"].fillna(False)

    if params.pa_pressure_min > 0 and "pa_net_pressure" in df.columns:
        raw_long = raw_long & df["pa_net_pressure"].ge(params.pa_pressure_min).fillna(False)
        raw_short = raw_short & df["pa_net_pressure"].le(-params.pa_pressure_min).fillna(False)

    if params.pa_regime_filter and "pa_regime_trend" in df.columns:
        ce_is_reversal = df["ce_buy_signal"].fillna(False) | df["ce_sell_signal"].fillna(False)
        reversal_suppress = ce_is_reversal & df["pa_regime_trend"].fillna(False)
        raw_long = raw_long & ~reversal_suppress
        raw_short = raw_short & ~reversal_suppress

    if params.pa_20bar_neutral and "pa_trend_weakened" in df.columns:
        raw_long = raw_long & ~df["pa_trend_weakened"].fillna(False)
        raw_short = raw_short & ~df["pa_trend_weakened"].fillna(False)

    if params.pa_ii_breakout_entry and "pa_is_ii_pattern" in df.columns:
        ii_long = df["pa_is_ii_pattern"].fillna(False) & (df["close"] > df["open"])
        ii_short = df["pa_is_ii_pattern"].fillna(False) & (df["close"] < df["open"])
        raw_long = raw_long | ii_long
        raw_short = raw_short | ii_short

    if params.pa_mtf_trend_filter and "pa_env_state" in df.columns and "pa_env_trend_dir" in df.columns:
        env_state = df["pa_env_state"].astype(str)
        env_dir = pd.to_numeric(df["pa_env_trend_dir"], errors="coerce").fillna(0).astype(int)
        bull_ok = env_state.isin(["tight_bull_channel", "wide_bull_channel"]) & env_dir.eq(1)
        bear_ok = env_state.isin(["tight_bear_channel", "wide_bear_channel"]) & env_dir.eq(-1)
        # 5m PA decides direction; 1m CE/ZLSMA/KAMA decides timing.
        raw_long = raw_long & bull_ok
        raw_short = raw_short & bear_ok

    if params.pa_use_advanced_pa_triggers:
        # Continuation triggers for trend phase + validated breakout triggers for TR->trend phase.
        trend_cont_long = (
            df.get("pa_is_h2_setup", pd.Series(False, index=df.index)).fillna(False)
            | df.get("pa_mag20_bull", pd.Series(False, index=df.index)).fillna(False)
            | df.get("pa_breakout_success_up", pd.Series(False, index=df.index)).fillna(False)
        )
        trend_cont_short = (
            df.get("pa_is_l2_setup", pd.Series(False, index=df.index)).fillna(False)
            | df.get("pa_mag20_bear", pd.Series(False, index=df.index)).fillna(False)
            | df.get("pa_breakout_success_down", pd.Series(False, index=df.index)).fillna(False)
        )
        raw_long = raw_long & trend_cont_long
        raw_short = raw_short & trend_cont_short

    same_day_prev = df["time_key"].dt.date.eq(df["time_key"].dt.date.shift(1)).fillna(False)
    if params.confirmation_mode == "none":
        long_entries = raw_long
        short_entries = raw_short
    elif params.confirmation_mode == "next_bar_body":
        confirm_up = (df["close"] > df["open"]).fillna(False)
        confirm_down = (df["close"] < df["open"]).fillna(False)
        long_entries = raw_long.shift(1, fill_value=False) & same_day_prev & confirm_up
        short_entries = raw_short.shift(1, fill_value=False) & same_day_prev & confirm_down
    elif params.confirmation_mode == "next_bar_no_reverse_close":
        confirm_up = (df["close"] >= df["open"]).fillna(False)
        confirm_down = (df["close"] <= df["open"]).fillna(False)
        long_entries = raw_long.shift(1, fill_value=False) & same_day_prev & confirm_up
        short_entries = raw_short.shift(1, fill_value=False) & same_day_prev & confirm_down
    else:
        raise ValueError(f"Unsupported confirmation_mode: {params.confirmation_mode}")
    return long_entries, short_entries


def session_entry_allowed(times: pd.Series, session_filter: str) -> pd.Series:
    bar_time = times.dt.time
    if session_filter == "all_day":
        return pd.Series(True, index=times.index)
    if session_filter == "first_hour":
        return pd.Series(
            [(t >= time(9, 35)) and (t <= time(10, 30)) for t in bar_time],
            index=times.index,
        )
    if session_filter == "morning":
        return pd.Series(
            [(t >= time(9, 35)) and (t <= time(11, 0)) for t in bar_time],
            index=times.index,
        )
    if session_filter == "before_1230":
        return pd.Series(
            [(t >= time(9, 30)) and (t <= time(12, 30)) for t in bar_time],
            index=times.index,
        )
    if session_filter == "trend_windows":
        return pd.Series(
            [
                ((t >= time(9, 35)) and (t <= time(11, 0)))
                or ((t >= time(14, 0)) and (t <= time(15, 30)))
                for t in bar_time
            ],
            index=times.index,
        )
    raise ValueError(f"Unsupported session_filter: {session_filter}")


def run_intraday_rule(
    df: pd.DataFrame,
    symbol: str,
    params: RuleParams,
) -> OptimizationResult:
    long_signal, short_signal = build_entry_signals(df, params)
    entry_allowed = session_entry_allowed(df["time_key"], params.session_filter)
    profit_lock_enabled = params.profit_lock_fraction > 0 and params.profit_lock_trigger_pct > 0

    session_date = df["time_key"].dt.date
    is_first_bar = session_date.ne(session_date.shift(1)).fillna(True)
    is_last_bar = session_date.ne(session_date.shift(-1)).fillna(True)

    equity = 1.0
    equity_values: list[float] = []
    trade_returns: list[float] = []
    holding_minutes: list[int] = []
    trade_records: list[dict[str, object]] = []
    partial_exit_flags: list[bool] = []
    exit_reasons: list[str] = []

    position = 0
    entry_price = np.nan
    entry_time = None
    stop_price = np.nan
    take_profit = np.nan
    profit_lock_price = np.nan
    remaining_weight = 1.0
    realized_trade_return = 0.0
    partial_exit_triggered = False
    partial_exit_time = None
    partial_exit_price = np.nan
    max_progress_to_target = 0.0

    open_prices = df["open"].to_numpy(dtype=float)
    high_prices = df["high"].to_numpy(dtype=float)
    low_prices = df["low"].to_numpy(dtype=float)
    close_prices = df["close"].to_numpy(dtype=float)
    atr_values = df["ce_atr"].to_numpy(dtype=float)
    zlsma_prices = df["zlsma"].to_numpy(dtype=float)
    kama_prices = df["kama"].to_numpy(dtype=float)
    zlsma_slopes = df["zlsma_slope"].fillna(0.0).to_numpy(dtype=float)
    kama_slopes = df.get("kama_slope", df["kama"].pct_change()).fillna(0.0).to_numpy(dtype=float)
    ce_buy_active = df["ce_buy_signal"].fillna(False).to_numpy(dtype=bool)
    ce_sell_active = df["ce_sell_signal"].fillna(False).to_numpy(dtype=bool)
    prev_high = np.roll(high_prices, 1)
    prev_low = np.roll(low_prices, 1)
    prev_close = np.roll(close_prices, 1)
    prev_high[0] = high_prices[0]
    prev_low[0] = low_prices[0]
    prev_close[0] = close_prices[0]

    _has_pa = "pa_is_mag_bar" in df.columns
    pa_mag = df["pa_is_mag_bar"].to_numpy(dtype=bool) if _has_pa else np.zeros(len(df), dtype=bool)
    pa_exh = df["pa_is_exhaustion_gap"].to_numpy(dtype=bool) if _has_pa else np.zeros(len(df), dtype=bool)
    pa_mm_up = df["pa_mm_target_up"].to_numpy(dtype=float) if _has_pa else np.full(len(df), np.nan)
    pa_mm_down = df["pa_mm_target_down"].to_numpy(dtype=float) if _has_pa else np.full(len(df), np.nan)
    pa_stop_long = df["pa_stop_long"].to_numpy(dtype=float) if _has_pa else np.full(len(df), np.nan)
    pa_stop_short = df["pa_stop_short"].to_numpy(dtype=float) if _has_pa else np.full(len(df), np.nan)
    pa_or_wide = df["or_wide"].to_numpy(dtype=bool) if "or_wide" in df.columns else np.zeros(len(df), dtype=bool)
    env_state = df["pa_env_state"].astype(str).to_numpy() if "pa_env_state" in df.columns else np.array([""] * len(df))
    env_dir = (
        pd.to_numeric(df["pa_env_trend_dir"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if "pa_env_trend_dir" in df.columns
        else np.zeros(len(df), dtype=int)
    )
    env_breakout = (
        df["pa_env_score_breakout_bar"].to_numpy(dtype=float)
        if "pa_env_score_breakout_bar" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    pa_breakout_success_up = (
        df["pa_breakout_success_up"].to_numpy(dtype=bool)
        if "pa_breakout_success_up" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_breakout_success_down = (
        df["pa_breakout_success_down"].to_numpy(dtype=bool)
        if "pa_breakout_success_down" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_wedge_up = (
        df["pa_wedge_third_push_up"].to_numpy(dtype=bool)
        if "pa_wedge_third_push_up" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_wedge_down = (
        df["pa_wedge_third_push_down"].to_numpy(dtype=bool)
        if "pa_wedge_third_push_down" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_overshoot_up = (
        df["pa_channel_overshoot_revert_up"].to_numpy(dtype=bool)
        if "pa_channel_overshoot_revert_up" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_overshoot_down = (
        df["pa_channel_overshoot_revert_down"].to_numpy(dtype=bool)
        if "pa_channel_overshoot_revert_down" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    pa_reversal_ready = (
        df["pa_channel_reversal_ready"].to_numpy(dtype=bool)
        if "pa_channel_reversal_ready" in df.columns
        else np.zeros(len(df), dtype=bool)
    )

    pa_or_wide_day_count = 0
    pa_mm_hit_count = 0
    pa_pa_stop_count = 0
    pa_env_skip_count = 0
    pa_env_neutral_entry_count = 0
    pa_trend_reversal_signal_count = 0
    pa_tr_breakout_attempt_count = 0
    trend_reversal_count_day = 0
    tr_breakout_count_day = 0

    for idx in range(len(df)):
        current_time = df.iloc[idx]["time_key"]

        if is_first_bar.iloc[idx]:
            position = 0
            entry_price = np.nan
            stop_price = np.nan
            take_profit = np.nan
            entry_time = None
            profit_lock_price = np.nan
            remaining_weight = 1.0
            realized_trade_return = 0.0
            partial_exit_triggered = False
            partial_exit_time = None
            partial_exit_price = np.nan
            max_progress_to_target = 0.0
            trend_reversal_count_day = 0
            tr_breakout_count_day = 0
            max_price_since_entry = np.nan
            min_price_since_entry = np.nan

        exit_reason = None
        exit_price = np.nan

        if position != 0:
            held_minutes = max(int((pd.Timestamp(current_time) - pd.Timestamp(entry_time)).total_seconds() / 60), 0)
            target_distance = abs(take_profit - entry_price)
            if target_distance > 0:
                if position == 1:
                    bar_progress = max((high_prices[idx] - entry_price) / target_distance, 0.0)
                else:
                    bar_progress = max((entry_price - low_prices[idx]) / target_distance, 0.0)
                max_progress_to_target = max(max_progress_to_target, float(bar_progress))

            # --- DYNAMIC TRAILING STOP LOGIC (Apply previous bar's MFE to current bar) ---
            if getattr(params, "pa_trailing_stop_mode", False) and not is_first_bar.iloc[idx]:
                atr_prev = atr_values[idx - 1] if np.isfinite(atr_values[idx - 1]) else 0.0
                activation_dist = getattr(params, "pa_trailing_activation_atr", 1.0) * atr_prev
                state_pos = env_state[idx - 1] if (idx - 1) < len(env_state) else ""
                is_trend_pos = state_pos in {"tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"}
                lock_ratio = getattr(params, "pa_trailing_lock_trend", 0.5) if is_trend_pos else getattr(params, "pa_trailing_lock_range", 0.8)

                if position == 1:
                    # We use the previous bar's high to update the trailing stop for the current bar
                    mfe = max_price_since_entry - entry_price if not np.isnan(max_price_since_entry) else 0.0
                    if mfe >= activation_dist and atr_prev > 0:
                        dynamic_stop = entry_price + mfe * lock_ratio
                        if np.isnan(stop_price) or dynamic_stop > stop_price:
                            stop_price = dynamic_stop
                else:
                    mfe = entry_price - min_price_since_entry if not np.isnan(min_price_since_entry) else 0.0
                    if mfe >= activation_dist and atr_prev > 0:
                        dynamic_stop = entry_price - mfe * lock_ratio
                        if np.isnan(stop_price) or dynamic_stop < stop_price:
                            stop_price = dynamic_stop

            if position == 1:
                if low_prices[idx] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                else:
                    if profit_lock_enabled and (not partial_exit_triggered) and high_prices[idx] >= profit_lock_price:
                        partial_exit_triggered = True
                        partial_exit_time = current_time
                        partial_exit_price = profit_lock_price
                        realized_trade_return += params.profit_lock_fraction * (profit_lock_price / entry_price - 1.0)
                        remaining_weight -= params.profit_lock_fraction
                    if high_prices[idx] >= take_profit:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif params.pa_mag_bar_exit and pa_mag[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "mag_bar_exit"
                    elif params.pa_exhaustion_gap_exit and pa_exh[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "exhaustion_gap_exit"
                    elif held_minutes >= params.time_stop_minutes and (
                        params.force_time_stop or max_progress_to_target < params.time_progress_threshold
                    ):
                        exit_price = close_prices[idx]
                        exit_reason = "time_stop"
                    elif is_last_bar.iloc[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "end_of_day"
            else:
                if high_prices[idx] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                else:
                    if profit_lock_enabled and (not partial_exit_triggered) and low_prices[idx] <= profit_lock_price:
                        partial_exit_triggered = True
                        partial_exit_time = current_time
                        partial_exit_price = profit_lock_price
                        realized_trade_return += params.profit_lock_fraction * position * (profit_lock_price / entry_price - 1.0)
                        remaining_weight -= params.profit_lock_fraction
                    if low_prices[idx] <= take_profit:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif params.pa_mag_bar_exit and pa_mag[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "mag_bar_exit"
                    elif params.pa_exhaustion_gap_exit and pa_exh[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "exhaustion_gap_exit"
                    elif held_minutes >= params.time_stop_minutes and (
                        params.force_time_stop or max_progress_to_target < params.time_progress_threshold
                    ):
                        exit_price = close_prices[idx]
                        exit_reason = "time_stop"
                    elif is_last_bar.iloc[idx]:
                        exit_price = close_prices[idx]
                        exit_reason = "end_of_day"

            if exit_reason is not None:
                final_leg_return = remaining_weight * position * (exit_price / entry_price - 1.0)
                trade_return = realized_trade_return + final_leg_return
                equity *= 1.0 + trade_return
                trade_returns.append(trade_return)
                holding_minutes.append(held_minutes)
                partial_exit_flags.append(partial_exit_triggered)
                exit_reasons.append(exit_reason)
                trade_records.append(
                    {
                        "symbol": symbol,
                        "strategy": params.strategy_name,
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "side": "long" if position == 1 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "trade_return": trade_return,
                        "holding_minutes": held_minutes,
                        "partial_exit_triggered": partial_exit_triggered,
                        "partial_exit_time": partial_exit_time,
                        "partial_exit_price": partial_exit_price,
                        "partial_exit_fraction": params.profit_lock_fraction if partial_exit_triggered else 0.0,
                        "max_progress_to_target": max_progress_to_target,
                        "exit_reason": exit_reason,
                    }
                )
                position = 0
                entry_price = np.nan
                stop_price = np.nan
                take_profit = np.nan
                entry_time = None
                profit_lock_price = np.nan
                remaining_weight = 1.0
                realized_trade_return = 0.0
                partial_exit_triggered = False
                partial_exit_time = None
                partial_exit_price = np.nan
                max_progress_to_target = 0.0

        if position == 0 and not is_last_bar.iloc[idx]:
            atr_now = atr_values[idx]
            if np.isfinite(atr_now) and atr_now > 0:
                state_now = env_state[idx] if idx < len(env_state) else ""
                allow_long = True
                allow_short = True
                size_mult = 1.0
                neutral_mode = False
                range_mode = False
                long_try = bool(long_signal.iloc[idx])
                short_try = bool(short_signal.iloc[idx])
                rr_floor_long = float(params.pa_mtf_min_rr)
                rr_floor_short = float(params.pa_mtf_min_rr)

                trend_bull = state_now in {"tight_bull_channel", "wide_bull_channel"}
                trend_bear = state_now in {"tight_bear_channel", "wide_bear_channel"}
                tr_state = state_now in {"wide_tr", "ttr", "neutral_50_50"}
                tradable_env = state_now in {
                    "tight_bull_channel",
                    "wide_bull_channel",
                    "tight_bear_channel",
                    "wide_bear_channel",
                    "wide_tr",
                    "neutral_50_50",
                }

                if params.pa_use_env_execution and state_now:
                    if state_now == "ttr":
                        allow_long = False
                        allow_short = False
                    elif state_now == "neutral_50_50":
                        neutral_mode = True
                        size_mult = 0.5
                    elif state_now == "wide_tr":
                        range_mode = True
                        size_mult = 0.6

                # 5m environment selects "where to trade", 1m bar action decides "when to execute".
                if params.pa_use_env_execution:
                    if not tradable_env:
                        long_try = False
                        short_try = False
                    else:
                        # PA state -> technical profile:
                        # each environment uses different CE/KAMA/ZLSMA strictness.
                        if state_now in {"tight_bull_channel", "tight_bear_channel"}:
                            ce_lookback = int(params.pa_env_ce_lookback_tight)
                            price_tol = float(params.pa_env_price_tol_tight)
                            slope_tol = float(params.pa_env_slope_tol_tight)
                        elif state_now in {"wide_bull_channel", "wide_bear_channel"}:
                            ce_lookback = int(params.pa_env_ce_lookback_wide)
                            price_tol = float(params.pa_env_price_tol_wide)
                            slope_tol = float(params.pa_env_slope_tol_wide)
                        elif state_now == "wide_tr":
                            ce_lookback = int(params.pa_env_ce_lookback_tr)
                            price_tol = float(params.pa_env_price_tol_tr)
                            slope_tol = float(params.pa_env_slope_tol_tr)
                        else:  # neutral_50_50
                            ce_lookback = int(params.pa_env_ce_lookback_neutral)
                            price_tol = float(params.pa_env_price_tol_neutral)
                            slope_tol = float(params.pa_env_slope_tol_neutral)

                        lookback_start = max(0, idx - ce_lookback)
                        ce_long_recent = bool(np.any(ce_buy_active[lookback_start: idx + 1]))
                        ce_short_recent = bool(np.any(ce_sell_active[lookback_start: idx + 1]))
                        core_long_relaxed = bool(
                            ce_long_recent
                            and close_prices[idx] >= zlsma_prices[idx] * (1.0 - price_tol)
                            and close_prices[idx] >= kama_prices[idx] * (1.0 - price_tol)
                            and zlsma_slopes[idx] >= -slope_tol
                            and kama_slopes[idx] >= -slope_tol
                        )
                        core_short_relaxed = bool(
                            ce_short_recent
                            and close_prices[idx] <= zlsma_prices[idx] * (1.0 + price_tol)
                            and close_prices[idx] <= kama_prices[idx] * (1.0 + price_tol)
                            and zlsma_slopes[idx] <= slope_tol
                            and kama_slopes[idx] <= slope_tol
                        )
                        long_core_ok = core_long_relaxed
                        short_core_ok = core_short_relaxed

                        rng_now = max(high_prices[idx] - low_prices[idx], 1e-12)
                        close_pos = (close_prices[idx] - low_prices[idx]) / rng_now
                        bull_reversal_1m = (
                            (close_prices[idx] > prev_close[idx])
                            and (close_prices[idx] > prev_high[idx])
                            and (close_pos >= 0.6)
                        )
                        bear_reversal_1m = (
                            (close_prices[idx] < prev_close[idx])
                            and (close_prices[idx] < prev_low[idx])
                            and (close_pos <= 0.4)
                        )
                        breakout_up_1m = close_prices[idx] > prev_high[idx]
                        breakout_down_1m = close_prices[idx] < prev_low[idx]
                        look = max(0, idx - 20)
                        range_high_1m = np.nanmax(high_prices[look: idx + 1]) if idx >= look else high_prices[idx]
                        range_low_1m = np.nanmin(low_prices[look: idx + 1]) if idx >= look else low_prices[idx]
                        range_mid_1m = 0.5 * (range_high_1m + range_low_1m)
                        range_third = (range_high_1m - range_low_1m) / 3.0
                        lower_third_cap = range_low_1m + range_third
                        upper_third_floor = range_high_1m - range_third
                        near_range_low = close_prices[idx] <= lower_third_cap
                        near_range_high = close_prices[idx] >= upper_third_floor

                        if trend_bull:
                            long_timing_ok = bool(breakout_up_1m or bull_reversal_1m)
                            long_try = bool(long_try and long_timing_ok and long_core_ok)
                            short_try = False
                        elif trend_bear:
                            short_timing_ok = bool(breakout_down_1m or bear_reversal_1m)
                            short_try = bool(short_try and short_timing_ok and short_core_ok)
                            long_try = False
                        elif range_mode or neutral_mode:
                            long_timing_ok = bool((near_range_low and bull_reversal_1m) or breakout_up_1m)
                            short_timing_ok = bool((near_range_high and bear_reversal_1m) or breakout_down_1m)
                            long_try = bool(long_try and long_timing_ok and long_core_ok)
                            short_try = bool(short_try and short_timing_ok and short_core_ok)

                if params.pa_use_advanced_pa_triggers:
                    # Trend state: first two reversal signals are ignored.
                    bearish_reversal_sig = bool(pa_wedge_down[idx] or pa_overshoot_up[idx] or (pa_reversal_ready[idx] and short_try))
                    bullish_reversal_sig = bool(pa_wedge_up[idx] or pa_overshoot_down[idx] or (pa_reversal_ready[idx] and long_try))
                    if trend_bull and bearish_reversal_sig:
                        trend_reversal_count_day += 1
                        pa_trend_reversal_signal_count += 1
                    if trend_bear and bullish_reversal_sig:
                        trend_reversal_count_day += 1
                        pa_trend_reversal_signal_count += 1

                    # In trend: reversal attempts need progressive confidence.
                    if trend_bull and short_try:
                        if trend_reversal_count_day <= 2:
                            short_try = False
                        elif trend_reversal_count_day == 3:
                            size_mult = min(size_mult, 0.5)
                            rr_floor_short = max(rr_floor_short, 2.0)
                    if trend_bear and long_try:
                        if trend_reversal_count_day <= 2:
                            long_try = False
                        elif trend_reversal_count_day == 3:
                            size_mult = min(size_mult, 0.5)
                            rr_floor_long = max(rr_floor_long, 2.0)

                    # TR state: first two breakout attempts are faded unless strong confirmation.
                    if tr_state and long_try:
                        pa_tr_breakout_attempt_count += 1
                        tr_breakout_count_day += 1
                        strong = bool(pa_breakout_success_up[idx])
                        if tr_breakout_count_day <= 2 and not strong:
                            long_try = False
                            short_try = True
                    if tr_state and short_try:
                        pa_tr_breakout_attempt_count += 1
                        tr_breakout_count_day += 1
                        strong = bool(pa_breakout_success_down[idx])
                        if tr_breakout_count_day <= 2 and not strong:
                            short_try = False
                            long_try = True

                if long_try:
                    entry_idx = idx + 1 if idx + 1 < len(df) else idx
                    if entry_allowed.iloc[entry_idx] and allow_long:
                        if neutral_mode:
                            neutral_ok = idx >= 1
                            # Neutral mode: only strong breakout + follow-through confirmation.
                            if neutral_ok:
                                neutral_ok = bool(pa_breakout_success_up[idx] and close_prices[idx] > open_prices[idx])
                            if not neutral_ok:
                                pa_env_skip_count += 1
                                allow_long = False
                    if entry_allowed.iloc[entry_idx] and allow_long:
                        position = 1
                        entry_price = open_prices[entry_idx] if entry_idx != idx else close_prices[idx]
                        stop_price, take_profit = build_exit_prices(entry_price, atr_now, 1, params)
                        stop_price, take_profit = apply_mtf_rr_floor(
                            entry_price, stop_price, take_profit, 1, params
                        )
                        if params.pa_use_env_execution:
                            # Anchor exits to 5m PA structure when available.
                            if np.isfinite(pa_stop_long[idx]) and pa_stop_long[idx] < entry_price:
                                stop_price = max(stop_price, pa_stop_long[idx])
                            if np.isfinite(pa_mm_up[idx]) and pa_mm_up[idx] > entry_price:
                                if state_now in {"tight_bull_channel", "wide_bull_channel"}:
                                    take_profit = max(take_profit, pa_mm_up[idx])
                                else:
                                    take_profit = min(take_profit, pa_mm_up[idx])
                        if rr_floor_long > 0:
                            risk = abs(entry_price - stop_price)
                            if risk > 0:
                                take_profit = max(take_profit, entry_price + rr_floor_long * risk)
                        if params.pa_use_mm_target and np.isfinite(pa_mm_up[idx]):
                            mm_tp = pa_mm_up[idx]
                            if mm_tp > entry_price:
                                take_profit = mm_tp
                                pa_mm_hit_count += 1
                        if params.pa_use_pa_stops and np.isfinite(pa_stop_long[idx]):
                            pa_sl = pa_stop_long[idx]
                            if pa_sl < entry_price:
                                stop_price = pa_sl
                                pa_pa_stop_count += 1
                        if params.pa_or_filter and pa_or_wide[idx] and params.pa_or_wide_tp_scale < 1.0:
                            tp_dist = abs(take_profit - entry_price)
                            take_profit = entry_price + tp_dist * params.pa_or_wide_tp_scale
                            pa_or_wide_day_count += 1
                        if params.pa_use_env_execution and state_now in {"wide_bull_channel", "wide_bear_channel"}:
                            if np.isfinite(pa_mm_up[idx]) and pa_mm_up[idx] > entry_price:
                                take_profit = pa_mm_up[idx]
                        if params.pa_use_env_execution and range_mode:
                            look = max(0, idx - 20)
                            range_high = np.nanmax(high_prices[look: idx + 1]) if idx >= look else high_prices[idx]
                            range_low = np.nanmin(low_prices[look: idx + 1]) if idx >= look else low_prices[idx]
                            range_mid = (range_high + range_low) / 2.0
                            stop_price = min(stop_price, range_low - atr_now)
                            # Wide TR: target opposite boundary or midline; choose closer conservative.
                            tp_mid = range_mid
                            tp_opp = range_high
                            take_profit = tp_mid if abs(tp_mid - entry_price) < abs(tp_opp - entry_price) else tp_opp
                        if params.pa_use_env_execution and neutral_mode:
                            # 50/50 mode: 1R conservative target.
                            risk = abs(entry_price - stop_price)
                            if risk > 0:
                                take_profit = entry_price + risk
                                pa_env_neutral_entry_count += 1
                        entry_time = df.iloc[entry_idx]["time_key"]
                        max_price_since_entry = entry_price
                        min_price_since_entry = np.nan
                        profit_lock_price = (
                            entry_price * (1.0 + params.profit_lock_trigger_pct) if profit_lock_enabled else np.nan
                        )
                        remaining_weight = 1.0
                        if params.pa_position_sizing_mode == "risk_based":
                            default_sl_dist = params.sl_atr_multiple * atr_now if params.sl_atr_multiple > 0 else atr_now
                            actual_sl_dist = abs(entry_price - stop_price)
                            if default_sl_dist > 0 and actual_sl_dist > 2.0 * default_sl_dist:
                                remaining_weight = 0.5
                        if params.pa_use_env_execution:
                            remaining_weight *= size_mult
                        realized_trade_return = 0.0
                        partial_exit_triggered = False
                        partial_exit_time = None
                        partial_exit_price = np.nan
                        max_progress_to_target = 0.0
                elif short_try:
                    entry_idx = idx + 1 if idx + 1 < len(df) else idx
                    if entry_allowed.iloc[entry_idx] and allow_short:
                        if neutral_mode:
                            neutral_ok = idx >= 1
                            if neutral_ok:
                                neutral_ok = bool(pa_breakout_success_down[idx] and close_prices[idx] < open_prices[idx])
                            if not neutral_ok:
                                pa_env_skip_count += 1
                                allow_short = False
                    if entry_allowed.iloc[entry_idx] and allow_short:
                        position = -1
                        entry_price = open_prices[entry_idx] if entry_idx != idx else close_prices[idx]
                        stop_price, take_profit = build_exit_prices(entry_price, atr_now, -1, params)
                        stop_price, take_profit = apply_mtf_rr_floor(
                            entry_price, stop_price, take_profit, -1, params
                        )
                        if params.pa_use_env_execution:
                            # Anchor exits to 5m PA structure when available.
                            if np.isfinite(pa_stop_short[idx]) and pa_stop_short[idx] > entry_price:
                                stop_price = min(stop_price, pa_stop_short[idx])
                            if np.isfinite(pa_mm_down[idx]) and pa_mm_down[idx] < entry_price:
                                if state_now in {"tight_bear_channel", "wide_bear_channel"}:
                                    take_profit = min(take_profit, pa_mm_down[idx])
                                else:
                                    take_profit = max(take_profit, pa_mm_down[idx])
                        if rr_floor_short > 0:
                            risk = abs(entry_price - stop_price)
                            if risk > 0:
                                take_profit = min(take_profit, entry_price - rr_floor_short * risk)
                        if params.pa_use_mm_target and np.isfinite(pa_mm_down[idx]):
                            mm_tp = pa_mm_down[idx]
                            if mm_tp < entry_price:
                                take_profit = mm_tp
                                pa_mm_hit_count += 1
                        if params.pa_use_pa_stops and np.isfinite(pa_stop_short[idx]):
                            pa_sl = pa_stop_short[idx]
                            if pa_sl > entry_price:
                                stop_price = pa_sl
                                pa_pa_stop_count += 1
                        if params.pa_or_filter and pa_or_wide[idx] and params.pa_or_wide_tp_scale < 1.0:
                            tp_dist = abs(take_profit - entry_price)
                            take_profit = entry_price - tp_dist * params.pa_or_wide_tp_scale
                            pa_or_wide_day_count += 1
                        if params.pa_use_env_execution and state_now in {"wide_bull_channel", "wide_bear_channel"}:
                            if np.isfinite(pa_mm_down[idx]) and pa_mm_down[idx] < entry_price:
                                take_profit = pa_mm_down[idx]
                        if params.pa_use_env_execution and range_mode:
                            look = max(0, idx - 20)
                            range_high = np.nanmax(high_prices[look: idx + 1]) if idx >= look else high_prices[idx]
                            range_low = np.nanmin(low_prices[look: idx + 1]) if idx >= look else low_prices[idx]
                            range_mid = (range_high + range_low) / 2.0
                            stop_price = max(stop_price, range_high + atr_now)
                            tp_mid = range_mid
                            tp_opp = range_low
                            take_profit = tp_mid if abs(tp_mid - entry_price) < abs(tp_opp - entry_price) else tp_opp
                        if params.pa_use_env_execution and neutral_mode:
                            risk = abs(entry_price - stop_price)
                            if risk > 0:
                                take_profit = entry_price - risk
                                pa_env_neutral_entry_count += 1
                        entry_time = df.iloc[entry_idx]["time_key"]
                        max_price_since_entry = np.nan
                        min_price_since_entry = entry_price
                        profit_lock_price = (
                            entry_price * (1.0 - params.profit_lock_trigger_pct) if profit_lock_enabled else np.nan
                        )
                        remaining_weight = 1.0
                        if params.pa_position_sizing_mode == "risk_based":
                            default_sl_dist = params.sl_atr_multiple * atr_now if params.sl_atr_multiple > 0 else atr_now
                            actual_sl_dist = abs(entry_price - stop_price)
                            if default_sl_dist > 0 and actual_sl_dist > 2.0 * default_sl_dist:
                                remaining_weight = 0.5
                        if params.pa_use_env_execution:
                            remaining_weight *= size_mult
                        realized_trade_return = 0.0
                        partial_exit_triggered = False
                        partial_exit_time = None
                        partial_exit_price = np.nan
                        max_progress_to_target = 0.0

        equity_values.append(equity)

    equity_curve = pd.Series(equity_values, index=df["time_key"], name="equity_curve")
    period_returns = equity_curve.pct_change().fillna(0.0)
    periods = len(period_returns)
    total_return = float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(period_returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * period_returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)
    avg_holding_minutes = float(np.mean(holding_minutes)) if holding_minutes else np.nan
    max_holding_minutes = float(np.max(holding_minutes)) if holding_minutes else np.nan
    partial_exit_rate = float(np.mean(partial_exit_flags)) if partial_exit_flags else np.nan
    time_stop_rate = float(np.mean(np.array(exit_reasons) == "time_stop")) if exit_reasons else np.nan
    take_profit_rate = float(np.mean(np.array(exit_reasons) == "take_profit")) if exit_reasons else np.nan
    stop_loss_rate = float(np.mean(np.array(exit_reasons) == "stop_loss")) if exit_reasons else np.nan
    end_of_day_rate = float(np.mean(np.array(exit_reasons) == "end_of_day")) if exit_reasons else np.nan

    summary = {
        "dataset": symbol,
        "strategy": params.strategy_name,
        "ce_length": params.ce_length,
        "ce_multiplier": params.ce_multiplier,
        "trend_mode": params.trend_mode,
        "tp_atr_multiple": params.tp_atr_multiple,
        "sl_atr_multiple": params.sl_atr_multiple,
        "session_filter": params.session_filter,
        "exit_model": params.exit_model,
        "tp_pct": params.tp_pct,
        "sl_pct": params.sl_pct,
        "zlsma_length": params.zlsma_length,
        "zlsma_offset": params.zlsma_offset,
        "kama_er_length": params.kama_er_length,
        "kama_fast_length": params.kama_fast_length,
        "kama_slow_length": params.kama_slow_length,
        "confirmation_mode": params.confirmation_mode,
        "zlsma_slope_threshold": params.zlsma_slope_threshold,
        "atr_percentile_lookback": params.atr_percentile_lookback,
        "atr_percentile_min": params.atr_percentile_min,
        "pseudo_cvd_method": params.pseudo_cvd_method,
        "cvd_lookback": params.cvd_lookback,
        "cvd_slope_lookback": params.cvd_slope_lookback,
        "cvd_classic_divergence": params.cvd_classic_divergence,
        "cvd_slope_divergence": params.cvd_slope_divergence,
        "time_stop_minutes": params.time_stop_minutes,
        "force_time_stop": params.force_time_stop,
        "time_progress_threshold": params.time_progress_threshold,
        "profit_lock_trigger_pct": params.profit_lock_trigger_pct,
        "profit_lock_fraction": params.profit_lock_fraction,
        "pa_or_filter": params.pa_or_filter,
        "pa_or_wide_tp_scale": params.pa_or_wide_tp_scale,
        "pa_require_signal_bar": params.pa_require_signal_bar,
        "pa_require_h2_l2": params.pa_require_h2_l2,
        "pa_pressure_min": params.pa_pressure_min,
        "pa_use_mm_target": params.pa_use_mm_target,
        "pa_use_pa_stops": params.pa_use_pa_stops,
        "pa_mag_bar_exit": params.pa_mag_bar_exit,
        "pa_exhaustion_gap_exit": params.pa_exhaustion_gap_exit,
        "pa_regime_filter": params.pa_regime_filter,
        "pa_20bar_neutral": params.pa_20bar_neutral,
        "pa_ii_breakout_entry": params.pa_ii_breakout_entry,
        "pa_position_sizing_mode": params.pa_position_sizing_mode,
        "pa_use_env_execution": params.pa_use_env_execution,
        "pa_mtf_trend_filter": params.pa_mtf_trend_filter,
        "pa_mtf_min_rr": params.pa_mtf_min_rr,
        "pa_use_advanced_pa_triggers": params.pa_use_advanced_pa_triggers,
        "pa_env_ce_lookback_tight": params.pa_env_ce_lookback_tight,
        "pa_env_ce_lookback_wide": params.pa_env_ce_lookback_wide,
        "pa_env_ce_lookback_tr": params.pa_env_ce_lookback_tr,
        "pa_env_ce_lookback_neutral": params.pa_env_ce_lookback_neutral,
        "pa_env_price_tol_tight": params.pa_env_price_tol_tight,
        "pa_env_price_tol_wide": params.pa_env_price_tol_wide,
        "pa_env_price_tol_tr": params.pa_env_price_tol_tr,
        "pa_env_price_tol_neutral": params.pa_env_price_tol_neutral,
        "pa_env_slope_tol_tight": params.pa_env_slope_tol_tight,
        "pa_env_slope_tol_wide": params.pa_env_slope_tol_wide,
        "pa_env_slope_tol_tr": params.pa_env_slope_tol_tr,
        "pa_env_slope_tol_neutral": params.pa_env_slope_tol_neutral,
        "pa_env_ce_only_gate": params.pa_env_ce_only_gate,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "max_drawdown": max_drawdown,
        "avg_holding_minutes": avg_holding_minutes,
        "max_holding_minutes": max_holding_minutes,
        "partial_exit_rate": partial_exit_rate,
        "time_stop_rate": time_stop_rate,
        "take_profit_rate": take_profit_rate,
        "stop_loss_rate": stop_loss_rate,
        "end_of_day_rate": end_of_day_rate,
        "mag_bar_exit_rate": float(np.mean(np.array(exit_reasons) == "mag_bar_exit")) if exit_reasons else np.nan,
        "exhaustion_gap_exit_rate": float(np.mean(np.array(exit_reasons) == "exhaustion_gap_exit")) if exit_reasons else np.nan,
        "pa_or_wide_day_count": pa_or_wide_day_count,
        "pa_mm_used_count": pa_mm_hit_count,
        "pa_pa_stop_used_count": pa_pa_stop_count,
        "pa_env_skip_count": pa_env_skip_count,
        "pa_env_neutral_entry_count": pa_env_neutral_entry_count,
        "pa_trend_reversal_signal_count": pa_trend_reversal_signal_count,
        "pa_tr_breakout_attempt_count": pa_tr_breakout_attempt_count,
    }
    return OptimizationResult(summary=summary, equity_curve=equity_curve, trade_log=pd.DataFrame(trade_records))

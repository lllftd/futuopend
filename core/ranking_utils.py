from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.optimize_ce_zlsma_kama_rule import RuleParams


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key in row.index:
        try:
            return float(row[key])
        except (ValueError, TypeError):
            return default
    return default


def _safe_int(row: pd.Series, key: str, default: int = 0) -> int:
    if key in row.index:
        try:
            return int(row[key])
        except (ValueError, TypeError):
            return default
    return default


def _safe_bool(row: pd.Series, key: str, default: bool = False) -> bool:
    if key not in row.index:
        return default
    value = row[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def _safe_str(row: pd.Series, key: str, default: str = "") -> str:
    if key in row.index:
        return str(row[key])
    return default


def ranking_row_to_rule_params(row: pd.Series) -> RuleParams:
    return RuleParams(
        ce_length=_safe_int(row, "ce_length", 1),
        ce_multiplier=_safe_float(row, "ce_multiplier", 2.0),
        trend_mode=_safe_str(row, "trend_mode", "price_above_both"),
        tp_atr_multiple=_safe_float(row, "tp_atr_multiple"),
        sl_atr_multiple=_safe_float(row, "sl_atr_multiple"),
        session_filter=_safe_str(row, "session_filter", "before_1230"),
        exit_model=_safe_str(row, "exit_model", "atr"),
        tp_pct=_safe_float(row, "tp_pct"),
        sl_pct=_safe_float(row, "sl_pct"),
        zlsma_length=_safe_int(row, "zlsma_length", 50),
        zlsma_offset=_safe_int(row, "zlsma_offset"),
        kama_er_length=_safe_int(row, "kama_er_length", 9),
        kama_fast_length=_safe_int(row, "kama_fast_length", 2),
        kama_slow_length=_safe_int(row, "kama_slow_length", 30),
        confirmation_mode=_safe_str(row, "confirmation_mode", "next_bar_body"),
        zlsma_slope_threshold=_safe_float(row, "zlsma_slope_threshold"),
        atr_percentile_lookback=_safe_int(row, "atr_percentile_lookback", 120),
        atr_percentile_min=_safe_float(row, "atr_percentile_min"),
        pseudo_cvd_method=_safe_str(row, "pseudo_cvd_method", "clv_body_volume"),
        cvd_lookback=_safe_int(row, "cvd_lookback", 20),
        cvd_slope_lookback=_safe_int(row, "cvd_slope_lookback", 3),
        cvd_classic_divergence=_safe_bool(row, "cvd_classic_divergence"),
        cvd_slope_divergence=_safe_bool(row, "cvd_slope_divergence"),
        time_stop_minutes=_safe_int(row, "time_stop_minutes", 30),
        force_time_stop=_safe_bool(row, "force_time_stop"),
        time_progress_threshold=_safe_float(row, "time_progress_threshold", 0.5),
        profit_lock_trigger_pct=_safe_float(row, "profit_lock_trigger_pct", 0.0015),
        profit_lock_fraction=_safe_float(row, "profit_lock_fraction", 0.5),
        pa_or_filter=_safe_bool(row, "pa_or_filter"),
        pa_or_wide_tp_scale=_safe_float(row, "pa_or_wide_tp_scale", 0.7),
        pa_require_signal_bar=_safe_bool(row, "pa_require_signal_bar"),
        pa_require_h2_l2=_safe_bool(row, "pa_require_h2_l2"),
        pa_pressure_min=_safe_float(row, "pa_pressure_min"),
        pa_use_mm_target=_safe_bool(row, "pa_use_mm_target"),
        pa_use_pa_stops=_safe_bool(row, "pa_use_pa_stops"),
        pa_mag_bar_exit=_safe_bool(row, "pa_mag_bar_exit"),
        pa_exhaustion_gap_exit=_safe_bool(row, "pa_exhaustion_gap_exit"),
        pa_regime_filter=_safe_bool(row, "pa_regime_filter"),
        pa_20bar_neutral=_safe_bool(row, "pa_20bar_neutral"),
        pa_ii_breakout_entry=_safe_bool(row, "pa_ii_breakout_entry"),
        pa_position_sizing_mode=_safe_str(row, "pa_position_sizing_mode", "fixed"),
    )


def load_best_rule_params(
    ranking_path: Path,
    symbols: Iterable[str],
    *,
    required_columns: Iterable[str] = (),
) -> dict[str, RuleParams]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Refined ranking file not found: {ranking_path}.")

    ranking = pd.read_csv(ranking_path)
    missing = {"dataset", *required_columns}.difference(ranking.columns)
    if missing:
        raise RuntimeError(f"Ranking file is missing required columns: {sorted(missing)}")

    params_by_symbol: dict[str, RuleParams] = {}
    for symbol in symbols:
        symbol_upper = symbol.upper()
        subset = ranking.loc[ranking["dataset"].astype(str).str.upper() == symbol_upper].copy()
        if subset.empty:
            raise RuntimeError(f"No ranking rows found for symbol {symbol_upper} in {ranking_path}.")
        subset = subset.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False])
        params_by_symbol[symbol_upper] = ranking_row_to_rule_params(subset.iloc[0])
    return params_by_symbol


def load_best_params_and_rows(
    ranking_path: Path,
    symbols: Iterable[str],
) -> dict[str, tuple[RuleParams, pd.Series]]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Refined ranking file not found: {ranking_path}.")

    ranking = pd.read_csv(ranking_path)
    if "dataset" not in ranking.columns:
        raise RuntimeError("Ranking file is missing required columns: ['dataset']")

    params_by_symbol: dict[str, tuple[RuleParams, pd.Series]] = {}
    for symbol in symbols:
        symbol_upper = symbol.upper()
        subset = ranking.loc[ranking["dataset"].astype(str).str.upper() == symbol_upper].copy()
        if subset.empty:
            raise RuntimeError(f"No rows for {symbol_upper} in {ranking_path}")
        subset = subset.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False])
        row = subset.iloc[0]
        params_by_symbol[symbol_upper] = (ranking_row_to_rule_params(row), row)
    return params_by_symbol

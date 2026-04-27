"""L3 defaults.

Only a small set of variables remain configurable via environment (operator-facing):

- Data: ``L3_OOT_START``, ``L3_HOLDOUT_START``, ``L3_OOF_FOLDS`` (in stack_v2_common),
  ``L3_VAL_TUNE_FRAC``, ``L3_STRICT_TRADE_LEVEL_SPLIT``
- Model: ``L3_MAX_HOLD_BARS``, ``L3_VALUE_MODE`` (train), ``L3_VALUE_TARGET_MODE``
- Search: ``L3_POLICY_UTILITY_TAIL_WEIGHT``, ``L3_POLICY_MIN_EXIT_RATE``,
  ``L3_POLICY_ADAPTIVE_GRID`` (exit probability calibration is isotonic-only)
- Inference shaping: ``L3_MIN_HOLD_BARS``, ``L3_EXIT_EARLY_PATIENCE_STRENGTH``,
  ``L3_EXIT_LATE_PRESSURE_STRENGTH``
- Straddle: ``L3_STRADDLE_SIM_MODE``, ``L3_TRADE_SEMANTICS``

All other former ``L3_*`` env hooks are frozen to the historical code defaults below.
"""

from __future__ import annotations

from core.training.unified.config.env import env_bool, env_float, env_int, env_str


def calib_min_rows() -> int:
    return 100


def val_split_min_rows_each() -> int:
    return 40


def max_hold_bars() -> int:
    return env_int("L3_MAX_HOLD_BARS", 30, lo=1, hi=5000)


def min_hold_bars() -> int:
    return env_int("L3_MIN_HOLD_BARS", 0, lo=0, hi=10_000)


def exit_early_patience_strength() -> float:
    return env_float("L3_EXIT_EARLY_PATIENCE_STRENGTH", 0.10, lo=0.0, hi=1.0)


def exit_late_pressure_strength() -> float:
    return env_float("L3_EXIT_LATE_PRESSURE_STRENGTH", 0.18, lo=0.0, hi=1.0)


def value_target_mode() -> str:
    raw = (env_str("L3_VALUE_TARGET_MODE", "remaining_value_atr") or "remaining_value_atr").strip().lower()
    if raw in {"regression", "reg"}:
        return "regression"
    if raw in {"trade_outcome", "trade_median", "trade_level"}:
        return "trade_outcome"
    if raw in {"remaining_value_atr", "rv_atr", "forward_atr", "remaining_atr"}:
        return "remaining_value_atr"
    if raw in {"remaining_value", "forward_pnl", "remaining_pnl", "hold_to_deadline"}:
        return "remaining_value"
    return "peak_cls"


def policy_min_exit_rate() -> float:
    return env_float("L3_POLICY_MIN_EXIT_RATE", 0.05, lo=0.0, hi=0.95)


def policy_utility_tail_weight() -> float:
    return env_float("L3_POLICY_UTILITY_TAIL_WEIGHT", 0.30, lo=0.0, hi=0.5)


def policy_adaptive_grid() -> bool:
    return env_bool("L3_POLICY_ADAPTIVE_GRID", True)


def trade_semantics_default() -> str:
    v = env_str("L3_TRADE_SEMANTICS", "underlying_atr_path").strip()
    return v if v else "underlying_atr_path"


def late_hold_max_hold_ratio() -> float:
    return 0.67


def late_hold_frac() -> float:
    return 0.75


def late_hold_min_bars() -> int:
    return 2


def late_hold_start_mode() -> str:
    return "target_horizon_frac"


def late_hold_ramp() -> bool:
    return False


def late_hold_ramp_eps_floor() -> float:
    return 0.25


def exit_loss_buffer_atr_fixed() -> float:
    return 0.08


def exit_loss_buffer_mode() -> str:
    return "fixed"


def exit_loss_buffer_data_q() -> float:
    return 75.0


def exit_loss_buffer_data_mult() -> float:
    return 1.0


def exit_prob_floor() -> float:
    return 0.05


def exit_prob_ceil() -> float:
    return 0.95


def hyst_enter_default() -> float:
    return 0.48


def hyst_leave_default() -> float:
    return 0.33


def exit_ema_alpha_default() -> float:
    return 0.5


def hyst_min_gap() -> float:
    return 0.05


def hyst_fallback_gap() -> float:
    return 0.20


def traj_mfe_scale_default() -> float:
    return 5.0


def traj_mae_scale_default() -> float:
    return 5.0


def sample_weight_clip_mode() -> str:
    return "quantile"


def sample_weight_clip_lo_str() -> str:
    return ""


def sample_weight_clip_hi_str() -> str:
    return ""

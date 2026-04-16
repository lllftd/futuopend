"""Central env-backed defaults for L3 (B-class). Call these at use-time so env overrides apply."""

from __future__ import annotations

from core.trainers.l3.l3_env import env_bool, env_float, env_int, env_str


def min_train_samples() -> int:
    return env_int("L3_MIN_TRAIN_SAMPLES", 10, lo=2, hi=10_000_000)


def calib_min_rows() -> int:
    return env_int("L3_CALIB_MIN_ROWS", 100, lo=10, hi=500_000)


def val_split_min_rows_each() -> int:
    return env_int("L3_VAL_SPLIT_MIN_ROWS_EACH", 40, lo=5, hi=500_000)


def max_hold_bars() -> int:
    return env_int("L3_MAX_HOLD_BARS", 30, lo=1, hi=5000)


def late_hold_max_hold_ratio() -> float:
    return env_float("L3_LATE_HOLD_MAX_HOLD_RATIO", 0.67, lo=0.1, hi=1.0)


def late_hold_frac() -> float:
    return env_float("L3_LATE_HOLD_FRAC", 0.75, lo=0.1, hi=1.0)


def late_hold_min_bars() -> int:
    return env_int("L3_LATE_HOLD_MIN_BARS", 2, lo=1, hi=5000)


def late_hold_start_mode() -> str:
    # Default matches legacy: ceil(target_horizon * LATE_HOLD_FRAC). Set to max_hold_ratio for data-relative late hold.
    return env_str("L3_LATE_HOLD_START_MODE", "target_horizon_frac").strip().lower()


def late_hold_ramp() -> bool:
    return env_bool("L3_LATE_HOLD_RAMP", False)


def late_hold_ramp_eps_floor() -> float:
    return env_float("L3_LATE_HOLD_RAMP_EPS_FLOOR", 0.25, lo=0.05, hi=1.0)


def exit_loss_buffer_atr_fixed() -> float:
    return env_float("L3_EXIT_LOSS_BUFFER_ATR", 0.08, lo=0.0, hi=10.0)


def exit_loss_buffer_mode() -> str:
    return env_str("L3_EXIT_LOSS_BUFFER_MODE", "fixed").strip().lower()


def exit_loss_buffer_data_q() -> float:
    return env_float("L3_EXIT_LOSS_BUFFER_DATA_Q", 75.0, lo=1.0, hi=99.0)


def exit_loss_buffer_data_mult() -> float:
    return env_float("L3_EXIT_LOSS_BUFFER_DATA_MULT", 1.0, lo=0.01, hi=10.0)


def policy_value_q_n() -> int:
    return env_int("L3_POLICY_VALUE_Q_N", 14, lo=5, hi=50)


def policy_value_grid_refine() -> bool:
    return env_bool("L3_POLICY_VALUE_GRID_REFINE", False)


def policy_value_refine_spread() -> float:
    return env_float("L3_POLICY_VALUE_REFINE_SPREAD", 0.08, lo=0.01, hi=0.5)


def policy_value_refine_n() -> int:
    return env_int("L3_POLICY_VALUE_REFINE_N", 5, lo=3, hi=15)


def exit_prob_floor() -> float:
    return env_float("L3_EXIT_PROB_FLOOR", 0.05, lo=0.001, hi=0.49)


def exit_prob_ceil() -> float:
    return env_float("L3_EXIT_PROB_CEIL", 0.95, lo=0.51, hi=0.999)


def hyst_enter_default() -> float:
    return env_float("L3_EXIT_HYST_ENTER", 0.55, lo=0.01, hi=0.99)


def hyst_leave_default() -> float:
    return env_float("L3_EXIT_HYST_LEAVE", 0.35, lo=0.01, hi=0.99)


def hyst_min_gap() -> float:
    return env_float("L3_EXIT_HYST_MIN_GAP", 0.05, lo=0.0, hi=0.5)


def hyst_fallback_gap() -> float:
    return env_float("L3_EXIT_HYST_FALLBACK_GAP", 0.20, lo=0.01, hi=0.9)


def traj_mfe_scale_default() -> float:
    return env_float("L3_TRAJ_MFE_SCALE_DEFAULT", 5.0, lo=0.01, hi=100.0)


def traj_mae_scale_default() -> float:
    return env_float("L3_TRAJ_MAE_SCALE_DEFAULT", 5.0, lo=0.01, hi=100.0)


def sample_weight_clip_mode() -> str:
    return env_str("L3_SAMPLE_WEIGHT_CLIP_MODE", "quantile")


def sample_weight_clip_lo_str() -> str:
    return env_str("L3_SAMPLE_WEIGHT_CLIP_LO", "")


def sample_weight_clip_hi_str() -> str:
    return env_str("L3_SAMPLE_WEIGHT_CLIP_HI", "")

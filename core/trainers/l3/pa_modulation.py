"""Single source of truth for PA state feature weights in L3 policy training.

PA_STATE_FEATURES (see core.trainers.constants) are continuous scores from price-action
state, not discrete regime classes (NUM_REGIME_CLASSES is separate).

Each "channel" below is a linear functional applied to named PA columns. Coefficients match
historical behavior in train.py before centralization; unused features are weight 0.
"""

from __future__ import annotations

import numpy as np

from core.trainers.constants import PA_STATE_FEATURES

# --- Entry policy search: bonus added to grid-search score (subset: no pullback, no always_in) ---
ENTRY_POLICY_SCORE_W: dict[str, float] = {
    "pa_state_trend_strength": 0.08,
    "pa_state_followthrough_quality": 0.04,
    "pa_state_range_risk": -0.10,
    "pa_state_breakout_failure_risk": -0.08,
    "pa_state_pullback_exhaustion": 0.0,
    "pa_state_always_in_bias": 0.0,
}

# --- Target horizon scaling: peak_bar *= clip(1 + dot, 0.55, 1.30) ---
HORIZON_SCALE_W: dict[str, float] = {
    "pa_state_trend_strength": 0.18,
    "pa_state_followthrough_quality": 0.10,
    "pa_state_range_risk": -0.22,
    "pa_state_breakout_failure_risk": -0.18,
    "pa_state_pullback_exhaustion": -0.10,
    "pa_state_always_in_bias": 0.0,
}
HORIZON_SCALE_CLIP: tuple[float, float] = (0.55, 1.30)

# --- Value target scaling: y *= clip(1 + dot, 0.75, 1.20) ---
VALUE_TARGET_SCALE_W: dict[str, float] = {
    "pa_state_trend_strength": 0.16,
    "pa_state_followthrough_quality": 0.08,
    "pa_state_range_risk": -0.18,
    "pa_state_breakout_failure_risk": -0.16,
    "pa_state_pullback_exhaustion": -0.08,
    "pa_state_always_in_bias": 0.0,
}
VALUE_TARGET_SCALE_CLIP: tuple[float, float] = (0.75, 1.20)

# --- Exit sample weights: w *= clip(1 + dot, 0.70, 1.80) ---
EXIT_TRADE_WEIGHT_W: dict[str, float] = {
    "pa_state_range_risk": 0.24,
    "pa_state_breakout_failure_risk": 0.18,
    "pa_state_pullback_exhaustion": 0.10,
    "pa_state_trend_strength": -0.10,
    "pa_state_followthrough_quality": 0.0,
    "pa_state_always_in_bias": 0.0,
}
EXIT_TRADE_WEIGHT_CLIP: tuple[float, float] = (0.70, 1.80)

# --- Exit label: epsilon ATR multiplier per step (PA-aware path) ---
EXIT_EPS_STEP_W: dict[str, float] = {
    "pa_state_range_risk": 0.25,
    "pa_state_breakout_failure_risk": 0.20,
    "pa_state_pullback_exhaustion": 0.10,
    "pa_state_trend_strength": -0.12,
    "pa_state_followthrough_quality": -0.08,
    "pa_state_always_in_bias": 0.0,
}
EXIT_EPS_STEP_CLIP: tuple[float, float] = (0.70, 1.60)

# --- Exit label: loss buffer multiplier ---
EXIT_LOSS_BUFFER_STEP_W: dict[str, float] = {
    "pa_state_range_risk": -0.18,
    "pa_state_breakout_failure_risk": -0.14,
    "pa_state_pullback_exhaustion": -0.08,
    "pa_state_trend_strength": 0.10,
    "pa_state_followthrough_quality": 0.0,
    "pa_state_always_in_bias": 0.0,
}
EXIT_LOSS_BUFFER_CLIP: tuple[float, float] = (0.55, 1.20)

# --- Exit label: live edge floor multiplier ---
EXIT_LIVE_EDGE_FLOOR_STEP_W: dict[str, float] = {
    "pa_state_range_risk": 0.20,
    "pa_state_breakout_failure_risk": 0.18,
    "pa_state_trend_strength": -0.12,
    "pa_state_followthrough_quality": 0.0,
    "pa_state_pullback_exhaustion": 0.0,
    "pa_state_always_in_bias": 0.0,
}
EXIT_LIVE_EDGE_FLOOR_CLIP: tuple[float, float] = (0.60, 1.50)

# --- Exit label: late_hold_start scale from entry-bar PA ---
EXIT_LATE_HOLD_ENTRY_W: dict[str, float] = {
    "pa_state_trend_strength": 0.20,
    "pa_state_followthrough_quality": 0.10,
    "pa_state_range_risk": -0.24,
    "pa_state_breakout_failure_risk": -0.20,
    "pa_state_pullback_exhaustion": -0.10,
    "pa_state_always_in_bias": 0.0,
}
EXIT_LATE_HOLD_ENTRY_CLIP: tuple[float, float] = (0.55, 1.25)


def _pa_linear(
    pa_state: dict[str, np.ndarray] | None,
    weights: dict[str, float],
    *,
    n: int,
) -> np.ndarray:
    if pa_state is None:
        return np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    for name, w in weights.items():
        if w == 0.0:
            continue
        acc += w * np.asarray(pa_state.get(name, np.zeros(n)), dtype=np.float64).ravel()[:n]
    if acc.size < n:
        acc = np.pad(acc, (0, n - acc.size))
    return acc[:n]


def pa_entry_policy_score_bonus_masked(pa_state: dict[str, np.ndarray] | None, entered: np.ndarray) -> float:
    """Bonus term for entry policy grid search: same as legacy mean of PA over ``entered`` rows."""
    if pa_state is None or not np.any(entered):
        return 0.0
    ent = np.asarray(entered, dtype=bool).ravel()
    s = 0.0
    for name, w in ENTRY_POLICY_SCORE_W.items():
        if w == 0.0:
            continue
        v = np.asarray(pa_state.get(name, np.zeros(len(ent))), dtype=np.float64).ravel()[ent]
        s += w * float(np.mean(v))
    return float(s)


def pa_horizon_scale(pa_state: dict[str, np.ndarray] | None, *, n: int) -> np.ndarray:
    lo, hi = HORIZON_SCALE_CLIP
    delta = _pa_linear(pa_state, HORIZON_SCALE_W, n=n)
    return np.clip(1.0 + delta, lo, hi).astype(np.float64)


def pa_value_target_scale(pa_state: dict[str, np.ndarray] | None, *, n: int) -> np.ndarray:
    lo, hi = VALUE_TARGET_SCALE_CLIP
    delta = _pa_linear(pa_state, VALUE_TARGET_SCALE_W, n=n)
    return np.clip(1.0 + delta, lo, hi).astype(np.float64)


def pa_exit_trade_weight_multiplier(pa_state: dict[str, np.ndarray] | None, *, n: int) -> np.ndarray:
    lo, hi = EXIT_TRADE_WEIGHT_CLIP
    delta = _pa_linear(pa_state, EXIT_TRADE_WEIGHT_W, n=n)
    return np.clip(1.0 + delta, lo, hi).astype(np.float64)


def pa_exit_eps_multiplier(
    step_range: np.ndarray,
    step_breakout: np.ndarray,
    step_pullback: np.ndarray,
    step_trend: np.ndarray,
    step_follow: np.ndarray,
) -> np.ndarray:
    """Per-step epsilon scale (broadcast PA step arrays)."""
    n = int(step_range.shape[0])
    lo, hi = EXIT_EPS_STEP_CLIP
    delta = (
        EXIT_EPS_STEP_W["pa_state_range_risk"] * step_range
        + EXIT_EPS_STEP_W["pa_state_breakout_failure_risk"] * step_breakout
        + EXIT_EPS_STEP_W["pa_state_pullback_exhaustion"] * step_pullback
        + EXIT_EPS_STEP_W["pa_state_trend_strength"] * step_trend
        + EXIT_EPS_STEP_W["pa_state_followthrough_quality"] * step_follow
    )
    return np.clip(1.0 + np.asarray(delta, dtype=np.float64), lo, hi).astype(np.float32)


def pa_exit_loss_buffer_multiplier(
    step_range: np.ndarray,
    step_breakout: np.ndarray,
    step_pullback: np.ndarray,
    step_trend: np.ndarray,
) -> np.ndarray:
    lo, hi = EXIT_LOSS_BUFFER_CLIP
    delta = (
        EXIT_LOSS_BUFFER_STEP_W["pa_state_range_risk"] * step_range
        + EXIT_LOSS_BUFFER_STEP_W["pa_state_breakout_failure_risk"] * step_breakout
        + EXIT_LOSS_BUFFER_STEP_W["pa_state_pullback_exhaustion"] * step_pullback
        + EXIT_LOSS_BUFFER_STEP_W["pa_state_trend_strength"] * step_trend
    )
    return np.clip(1.0 + np.asarray(delta, dtype=np.float64), lo, hi).astype(np.float32)


def pa_exit_live_edge_floor_multiplier(
    step_range: np.ndarray,
    step_breakout: np.ndarray,
    step_trend: np.ndarray,
) -> np.ndarray:
    lo, hi = EXIT_LIVE_EDGE_FLOOR_CLIP
    delta = (
        EXIT_LIVE_EDGE_FLOOR_STEP_W["pa_state_range_risk"] * step_range
        + EXIT_LIVE_EDGE_FLOOR_STEP_W["pa_state_breakout_failure_risk"] * step_breakout
        + EXIT_LIVE_EDGE_FLOOR_STEP_W["pa_state_trend_strength"] * step_trend
    )
    return np.clip(1.0 + np.asarray(delta, dtype=np.float64), lo, hi).astype(np.float32)


def pa_exit_late_hold_entry_scale(pa_state: dict[str, np.ndarray], entry_i: int) -> float:
    lo, hi = EXIT_LATE_HOLD_ENTRY_CLIP
    delta = 0.0
    for name in PA_STATE_FEATURES:
        w = EXIT_LATE_HOLD_ENTRY_W.get(name, 0.0)
        if w == 0.0:
            continue
        col = pa_state.get(name)
        if col is None:
            continue
        delta += w * float(np.asarray(col, dtype=np.float64).ravel()[entry_i])
    return float(np.clip(1.0 + delta, lo, hi))


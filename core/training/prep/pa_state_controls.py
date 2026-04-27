from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.training.common.constants import PA_STATE_FEATURES


PA_STATE_BUCKET_NEUTRAL = "neutral"
PA_STATE_BUCKET_TREND = "trend_confirmed"
PA_STATE_BUCKET_RANGE = "range_or_uncertain"
PA_STATE_BUCKET_BREAKOUT = "breakout_risky"
PA_STATE_BUCKET_PULLBACK = "pullback_late"


def _clip01(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _signed_clip(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)


def _numeric_from_frame(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default)).to_numpy(dtype=np.float64, copy=False)


def _numeric_from_mapping(values: Mapping[str, Any] | pd.Series | None, col: str, default: float = 0.0) -> float:
    if values is None:
        return float(default)
    raw = values.get(col, default) if hasattr(values, "get") else default
    try:
        return float(raw)
    except Exception:
        return float(default)


def ensure_pa_state_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PA_STATE_FEATURES if c not in df.columns]
    if not missing:
        return df

    setup_trend_long = _clip01(_numeric_from_frame(df, "pa_ctx_setup_trend_long"))
    setup_trend_short = _clip01(_numeric_from_frame(df, "pa_ctx_setup_trend_short"))
    setup_pullback_long = _clip01(_numeric_from_frame(df, "pa_ctx_setup_pullback_long"))
    setup_pullback_short = _clip01(_numeric_from_frame(df, "pa_ctx_setup_pullback_short"))
    setup_range_long = _clip01(_numeric_from_frame(df, "pa_ctx_setup_range_long"))
    setup_range_short = _clip01(_numeric_from_frame(df, "pa_ctx_setup_range_short"))
    setup_failed_breakout_long = _clip01(_numeric_from_frame(df, "pa_ctx_setup_failed_breakout_long"))
    setup_failed_breakout_short = _clip01(_numeric_from_frame(df, "pa_ctx_setup_failed_breakout_short"))
    setup_long = _clip01(_numeric_from_frame(df, "pa_ctx_setup_long"))
    setup_short = _clip01(_numeric_from_frame(df, "pa_ctx_setup_short"))
    follow_long = _clip01(_numeric_from_frame(df, "pa_ctx_follow_through_long"))
    follow_short = _clip01(_numeric_from_frame(df, "pa_ctx_follow_through_short"))
    range_pressure = _clip01(_numeric_from_frame(df, "pa_ctx_range_pressure"))
    structure_veto = _clip01(_numeric_from_frame(df, "pa_ctx_structure_veto"))
    premise_break_long = _clip01(_numeric_from_frame(df, "pa_ctx_premise_break_long"))
    premise_break_short = _clip01(_numeric_from_frame(df, "pa_ctx_premise_break_short"))

    trend_score = _clip01(
        np.where(
            "pa_env_trend_score_ratio" in df.columns,
            _numeric_from_frame(df, "pa_env_trend_score_ratio"),
            _numeric_from_frame(df, "pa_trend_ratio"),
        )
    )
    range_score = _clip01(
        np.where(
            "pa_env_range_score_ratio" in df.columns,
            _numeric_from_frame(df, "pa_env_range_score_ratio"),
            _numeric_from_frame(df, "pa_regime_range"),
        )
    )
    breakout_fail = _clip01(_numeric_from_frame(df, "pa_breakout_likely_fail"))
    trend_weakened = _clip01(_numeric_from_frame(df, "pa_trend_weakened"))
    pullback_stage_abs = _clip01(np.abs(_numeric_from_frame(df, "pa_pullback_stage")) / 4.0)

    trend_setup_max = np.maximum(setup_trend_long, setup_trend_short)
    pullback_setup_max = np.maximum(setup_pullback_long, setup_pullback_short)
    range_setup_max = np.maximum(setup_range_long, setup_range_short)
    failed_breakout_max = np.maximum(setup_failed_breakout_long, setup_failed_breakout_short)
    setup_max = np.maximum(setup_long, setup_short)
    follow_max = np.maximum(follow_long, follow_short)
    premise_break_max = np.maximum(premise_break_long, premise_break_short)

    long_bias = 0.58 * setup_trend_long + 0.42 * follow_long + 0.18 * setup_long
    short_bias = 0.58 * setup_trend_short + 0.42 * follow_short + 0.18 * setup_short
    always_in_bias = _signed_clip(np.tanh(2.4 * (long_bias - short_bias)))

    trend_strength = _clip01(
        0.30 * trend_setup_max
        + 0.20 * follow_max
        + 0.14 * setup_max
        + 0.12 * trend_score
        + 0.10 * np.abs(always_in_bias)
        + 0.08 * (1.0 - range_pressure)
        + 0.06 * (1.0 - structure_veto)
    )
    followthrough_quality = _clip01(
        0.46 * follow_max
        + 0.14 * trend_strength
        + 0.10 * np.abs(always_in_bias)
        + 0.08 * (1.0 - breakout_fail)
        - 0.14 * range_pressure
        - 0.08 * structure_veto
        - 0.06 * failed_breakout_max
    )
    range_risk = _clip01(
        0.38 * range_pressure
        + 0.20 * range_setup_max
        + 0.14 * structure_veto
        + 0.10 * failed_breakout_max
        + 0.10 * (1.0 - follow_max)
        + 0.08 * range_score
    )
    pullback_exhaustion = _clip01(
        0.34 * pullback_stage_abs
        + 0.24 * pullback_setup_max
        + 0.16 * trend_weakened
        + 0.12 * range_pressure
        + 0.08 * premise_break_max
        + 0.06 * failed_breakout_max
    )
    breakout_failure_risk = _clip01(
        0.40 * failed_breakout_max
        + 0.18 * breakout_fail
        + 0.14 * structure_veto
        + 0.12 * range_pressure
        + 0.08 * (1.0 - follow_max)
        + 0.08 * range_score
    )

    values = {
        "pa_state_trend_strength": trend_strength.astype(np.float32, copy=False),
        "pa_state_followthrough_quality": followthrough_quality.astype(np.float32, copy=False),
        "pa_state_range_risk": range_risk.astype(np.float32, copy=False),
        "pa_state_pullback_exhaustion": pullback_exhaustion.astype(np.float32, copy=False),
        "pa_state_breakout_failure_risk": breakout_failure_risk.astype(np.float32, copy=False),
        "pa_state_always_in_bias": always_in_bias.astype(np.float32, copy=False),
    }
    for col in missing:
        df[col] = values[col]
    return df


def pa_state_arrays_from_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ensure_pa_state_features(df)
    return {
        col: pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        for col in PA_STATE_FEATURES
    }


def pa_state_bucket_labels_from_arrays(
    values: Mapping[str, Any],
    *,
    length: int,
) -> np.ndarray:
    trend_strength = _clip01(np.asarray(values.get("pa_state_trend_strength", np.zeros(length)), dtype=np.float64).reshape(-1))
    followthrough_quality = _clip01(
        np.asarray(values.get("pa_state_followthrough_quality", np.zeros(length)), dtype=np.float64).reshape(-1)
    )
    range_risk = _clip01(np.asarray(values.get("pa_state_range_risk", np.zeros(length)), dtype=np.float64).reshape(-1))
    pullback_exhaustion = _clip01(
        np.asarray(values.get("pa_state_pullback_exhaustion", np.zeros(length)), dtype=np.float64).reshape(-1)
    )
    breakout_failure_risk = _clip01(
        np.asarray(values.get("pa_state_breakout_failure_risk", np.zeros(length)), dtype=np.float64).reshape(-1)
    )
    always_in_bias = _signed_clip(np.asarray(values.get("pa_state_always_in_bias", np.zeros(length)), dtype=np.float64).reshape(-1))

    out = np.full(length, PA_STATE_BUCKET_NEUTRAL, dtype=object)
    breakout_mask = breakout_failure_risk >= 0.60
    pullback_mask = (~breakout_mask) & (pullback_exhaustion >= 0.58) & (trend_strength <= 0.72)
    range_mask = (~breakout_mask) & (~pullback_mask) & (range_risk >= 0.57)
    trend_mask = (
        (~breakout_mask)
        & (~pullback_mask)
        & (~range_mask)
        & (trend_strength >= 0.56)
        & (followthrough_quality >= 0.52)
        & (np.abs(always_in_bias) >= 0.12)
    )
    out[breakout_mask] = PA_STATE_BUCKET_BREAKOUT
    out[pullback_mask] = PA_STATE_BUCKET_PULLBACK
    out[range_mask] = PA_STATE_BUCKET_RANGE
    out[trend_mask] = PA_STATE_BUCKET_TREND
    return out


def pa_state_bucket_labels_from_frame(df: pd.DataFrame) -> np.ndarray:
    arrays = pa_state_arrays_from_frame(df)
    return pa_state_bucket_labels_from_arrays(arrays, length=len(df))


def pa_state_bucket_label_from_mapping(values: Mapping[str, Any] | pd.Series | None) -> str:
    arrays = {
        col: np.asarray([_numeric_from_mapping(values, col)], dtype=np.float32)
        for col in PA_STATE_FEATURES
    }
    return str(pa_state_bucket_labels_from_arrays(arrays, length=1)[0])

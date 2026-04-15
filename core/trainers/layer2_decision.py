from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.trainers.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1A_MODEL_FILE,
    L1A_REGIME_COLS,
    L1B_META_FILE,
    L1C_MODEL_FILE,
    L2_DECISION_FILE,
    L2_DIRECTION_FILE,
    L2_ENTRY_REGIME_COLS,
    L2_GATE_FILE,
    L2_MAE_FILE,
    L2_META_FILE,
    L2_MFE_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
    L2_SIGNED_EDGE_FILE,
    L2_SIZE_FILE,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    PA_STATE_FEATURES,
    STATE_NAMES,
    TRAIN_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _mfe_mae_atr_arrays,
)
from core.trainers.pa_state_controls import (
    PA_STATE_BUCKET_TREND,
    ensure_pa_state_features,
    pa_state_arrays_from_frame,
    pa_state_bucket_labels_from_frame,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.trainers.stack_v2_common import build_stack_time_splits, l2_val_start_time, log_label_baseline, save_output_cache
from core.trainers.val_metrics_extra import (
    brier_multiclass,
    ece_multiclass_maxprob,
    pearson_corr,
    regression_degen_flag,
    tail_mae_truth_upper,
)


def _l2_no_direction_gate_bump_cap() -> tuple[float, float]:
    """When direction head is skipped, raise gate threshold by bump (capped) — see meta direction_available."""
    bump = float(os.environ.get("L2_NO_DIRECTION_GATE_BUMP", "0.05"))
    cap = float(os.environ.get("L2_NO_DIRECTION_GATE_CAP", "0.55"))
    return bump, cap


def _l2_early_stopping_rounds_from_env(key: str, fallback: int) -> int:
    """Parse early_stopping_rounds from env; unset/empty uses fallback. Min 1."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return max(1, int(fallback))
    return max(1, int(raw))


def _l2_apply_l1b_feature_dropout_train_only(
    X: np.ndarray,
    train_mask: np.ndarray,
    feature_cols: list[str],
    drop_prob: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Training-time copy of X: each train row independently drops all l1b_* columns with prob drop_prob."""
    if drop_prob <= 0.0:
        return X
    l1b_idx = [j for j, c in enumerate(feature_cols) if c.startswith("l1b_")]
    if not l1b_idx:
        return X
    out = np.array(X, dtype=np.float32, copy=True)
    rows = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    if rows.size == 0:
        return out
    drop = rng.random(rows.size) < float(drop_prob)
    drop_rows = rows[drop]
    if drop_rows.size:
        out[np.ix_(drop_rows, l1b_idx)] = 0.0
    return out


# Drop from L2 without retraining upstream layers: failed L1b heads / redundant pairs.
# Override with L2_SKIP_FEATURE_HARD_DROP=1; extend via L2_EXTRA_HARD_DROP=col1,col2
L2_FEATURE_HARD_DROP_DEFAULT = frozenset(
    {
    }
)


def _l2_select_features_for_training(
    X: np.ndarray,
    feature_cols: list[str],
    train_mask: np.ndarray,
    *,
    min_std: float,
    hard_drop: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Remove hard-listed columns, near-constants, and near-duplicate train features."""
    tm = np.asarray(train_mask, dtype=bool)
    if not tm.any():
        raise RuntimeError("L2: empty train_mask for feature selection.")
    xt = X[tm].astype(np.float64, copy=False)
    dropped: list[str] = []
    keep_idx: list[int] = []
    for j, name in enumerate(feature_cols):
        if name in hard_drop:
            dropped.append(f"{name}(hard_drop)")
            continue
        col = xt[:, j]
        sd = float(np.nanstd(col))
        if not np.isfinite(sd) or sd < min_std:
            dropped.append(f"{name}(std={sd:.2e})")
            continue
        keep_idx.append(j)

    corr_thr = float(os.environ.get("L2_MAX_PAIRWISE_CORR", "0.995"))
    if len(keep_idx) >= 2 and corr_thr < 0.999999:
        kept_after_corr: list[int] = []
        kept_cols: list[np.ndarray] = []
        for j in keep_idx:
            col = xt[:, j]
            drop_name = None
            for prev_j, prev_col in zip(kept_after_corr, kept_cols):
                corr = np.corrcoef(col, prev_col)[0, 1]
                if np.isfinite(corr) and abs(float(corr)) >= corr_thr:
                    drop_name = feature_cols[prev_j]
                    break
            if drop_name is not None:
                dropped.append(f"{feature_cols[j]}(corr~{drop_name})")
                continue
            kept_after_corr.append(j)
            kept_cols.append(col)
        keep_idx = kept_after_corr

    keep = [feature_cols[j] for j in keep_idx]
    return keep, dropped


L2_OUTPUT_COLS = [
    "l2_decision_class",
    "l2_decision_long",
    "l2_decision_neutral",
    "l2_decision_short",
    "l2_decision_confidence",
    "l2_size",
    "l2_pred_mfe",
    "l2_pred_mae",
    *L2_ENTRY_REGIME_COLS,
    "l2_entry_vol",
    "l2_expected_edge",
    "l2_rr_proxy",
]


@dataclass
class L2TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _l2_build_two_stage_labels(y_decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_decision, dtype=np.int64).ravel()
    y_gate = (y != 1).astype(np.int64)
    y_dir = np.full(len(y), -1, dtype=np.int64)
    y_dir[y == 0] = 1
    y_dir[y == 2] = 0
    return y_gate, y_dir


def _l2_compose_probs_from_gate_dir(gate_p: np.ndarray, dir_p: np.ndarray) -> np.ndarray:
    """Columns order: long, neutral, short. Sums to 1 row-wise."""
    g = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    d = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    p_long = g * d
    p_short = g * (1.0 - d)
    p_neu = 1.0 - g
    return np.column_stack([p_long, p_neu, p_short]).astype(np.float32)


def _l2_direction_margin_to_prob(direction_margin: np.ndarray, *, temperature: float) -> np.ndarray:
    margin = np.asarray(direction_margin, dtype=np.float64).ravel()
    temp = max(float(temperature), 1e-3)
    z = np.clip(margin / temp, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)


def _l2_recenter_direction_prob(dir_p: np.ndarray, *, center: float | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 1e-6, 1.0 - 1e-6)
    c = np.asarray(center, dtype=np.float64)
    if c.ndim == 0:
        c = np.full(len(p), float(c), dtype=np.float64)
    else:
        c = np.broadcast_to(c.reshape(-1), p.shape)
    c = np.clip(c, 1e-3, 1.0 - 1e-3)
    logits = np.log(p / (1.0 - p)) - np.log(c / (1.0 - c))
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))).astype(np.float64)


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -12.0, 12.0)))


def _l2_direction_sample_weights(y_dir: np.ndarray, fusion_frame: pd.DataFrame | None = None) -> np.ndarray:
    y = np.asarray(y_dir, dtype=np.int64).ravel()
    active = y >= 0
    w = np.ones(len(y), dtype=np.float32)
    if not active.any():
        return w
    y_act = y[active]
    n_short = max(int(np.sum(y_act == 0)), 1)
    n_long = max(int(np.sum(y_act == 1)), 1)
    total = n_short + n_long
    short_w = float(np.clip(total / (2.0 * n_short), 0.75, 3.0))
    long_w = float(np.clip(total / (2.0 * n_long), 0.75, 3.0))
    w[active & (y == 0)] = short_w
    w[active & (y == 1)] = long_w
    if fusion_frame is not None:
        trend = fusion_frame["pa_state_trend_strength"].to_numpy(dtype=np.float64, copy=False) if "pa_state_trend_strength" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        follow = fusion_frame["pa_state_followthrough_quality"].to_numpy(dtype=np.float64, copy=False) if "pa_state_followthrough_quality" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        range_risk = fusion_frame["pa_state_range_risk"].to_numpy(dtype=np.float64, copy=False) if "pa_state_range_risk" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        breakout = fusion_frame["pa_state_breakout_failure_risk"].to_numpy(dtype=np.float64, copy=False) if "pa_state_breakout_failure_risk" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        pullback = fusion_frame["pa_state_pullback_exhaustion"].to_numpy(dtype=np.float64, copy=False) if "pa_state_pullback_exhaustion" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        always_in = fusion_frame["pa_state_always_in_bias"].to_numpy(dtype=np.float64, copy=False) if "pa_state_always_in_bias" in fusion_frame.columns else np.zeros(len(y), dtype=np.float64)
        long_scale = np.clip(1.0 + 0.30 * np.maximum(always_in, 0.0) + 0.18 * trend + 0.12 * follow - 0.18 * range_risk - 0.15 * breakout - 0.08 * pullback, 0.45, 2.0)
        short_scale = np.clip(1.0 + 0.30 * np.maximum(-always_in, 0.0) + 0.18 * trend + 0.12 * follow - 0.18 * range_risk - 0.15 * breakout - 0.08 * pullback, 0.45, 2.0)
        w[active & (y == 1)] *= long_scale[active & (y == 1)].astype(np.float32)
        w[active & (y == 0)] *= short_scale[active & (y == 0)].astype(np.float32)
        uncertain = np.clip(1.0 + 0.25 * range_risk + 0.20 * breakout + 0.10 * pullback - 0.10 * trend, 0.60, 1.75)
        w[active] *= uncertain[active].astype(np.float32)
    return w


def _l2_signed_edge_target_prep(
    edge_raw: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    edge = np.asarray(edge_raw, dtype=np.float64).ravel()
    tm = np.asarray(train_mask, dtype=bool).ravel()
    finite = np.isfinite(edge) & tm
    abs_train = np.abs(edge[finite])
    if abs_train.size == 0:
        clip_abs = 1.0
    else:
        clip_q = float(np.clip(float(os.environ.get("L2_SIGNED_EDGE_CLIP_Q", "0.995")), 0.90, 0.9999))
        clip_abs = float(np.quantile(abs_train, clip_q))
    clip_abs = float(np.clip(clip_abs, 0.10, 5.0))
    target = np.clip(edge, -clip_abs, clip_abs).astype(np.float32)
    nz_train = abs_train[abs_train > 1e-4]
    if nz_train.size == 0:
        asinh_scale = 0.10
    else:
        asinh_scale = float(np.median(nz_train))
    asinh_scale = float(np.clip(asinh_scale, 0.02, clip_abs))
    transform = (os.environ.get("L2_SIGNED_EDGE_TARGET_TRANSFORM", "asinh").strip().lower() or "asinh")
    if transform not in {"none", "asinh"}:
        transform = "asinh"
    if transform == "asinh":
        target_fit = np.arcsinh(target.astype(np.float64) / asinh_scale).astype(np.float32)
    else:
        target_fit = target.astype(np.float32, copy=False)
    prep: dict[str, float | str] = {
        "clip_abs": clip_abs,
        "target_transform": transform,
        "asinh_scale": asinh_scale,
        "objective": "huber",
        "metric": "l1",
        "alpha": 0.90,
    }
    return target_fit, prep


def _l2_signed_edge_predict(model: lgb.Booster, X: np.ndarray, prep: dict[str, Any] | None) -> np.ndarray:
    cfg = dict(prep or {})
    clip_abs = float(cfg.get("clip_abs", 5.0))
    transform = str(cfg.get("target_transform", "none")).strip().lower() or "none"
    asinh_scale = float(cfg.get("asinh_scale", 0.10))
    pred = model.predict(X).astype(np.float64)
    if transform == "asinh":
        pred = np.sinh(pred) * max(asinh_scale, 1e-4)
    return np.clip(pred, -clip_abs, clip_abs).astype(np.float32)


def _l2_signed_edge_sample_weights(
    edge_raw: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    edge = np.asarray(edge_raw, dtype=np.float64).ravel()
    tm = np.asarray(train_mask, dtype=bool).ravel()
    abs_edge = np.abs(edge)
    weights = np.ones(len(edge), dtype=np.float32)
    train_abs = abs_edge[tm & np.isfinite(abs_edge)]
    if train_abs.size == 0:
        return weights, {"focus_floor": 0.0, "nonzero_share": 0.0}
    target_rate = _l2_target_trade_rate()
    focus_floor = float(np.quantile(train_abs, 1.0 - target_rate))
    focus_floor = float(
        max(
            focus_floor,
            _env_float_clipped("L2_SIGNED_EDGE_FOCUS_FLOOR_MIN", 0.02, lo=0.0, hi=2.0),
        )
    )
    nonzero_mask = abs_edge > 1e-4
    informative_mask = abs_edge >= focus_floor
    weights[~nonzero_mask] = float(_env_float_clipped("L2_SIGNED_EDGE_ZERO_WEIGHT", 0.35, lo=0.05, hi=2.0))
    weights[nonzero_mask] = float(_env_float_clipped("L2_SIGNED_EDGE_NONZERO_WEIGHT", 1.0, lo=0.1, hi=5.0))
    weights[informative_mask] *= float(_env_float_clipped("L2_SIGNED_EDGE_INFORMATIVE_MULT", 2.5, lo=1.0, hi=10.0))
    hi_q = float(np.clip(float(os.environ.get("L2_SIGNED_EDGE_TOP_Q", "0.98")), 0.80, 0.999))
    top_floor = float(np.quantile(train_abs, hi_q))
    weights[abs_edge >= top_floor] *= float(_env_float_clipped("L2_SIGNED_EDGE_TOP_MULT", 1.75, lo=1.0, hi=10.0))
    return np.clip(weights, 0.05, 20.0).astype(np.float32), {
        "focus_floor": float(focus_floor),
        "nonzero_share": float(np.mean(train_abs > 1e-4)),
    }


def _l2_rebalanced_binary_weights(y: np.ndarray, base_weights: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64).ravel()
    weights = np.ones(len(labels), dtype=np.float32) if base_weights is None else np.asarray(base_weights, dtype=np.float32).ravel().copy()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return np.clip(weights, 0.05, 20.0).astype(np.float32)
    total = n_pos + n_neg
    weights[pos] *= float(np.clip(total / (2.0 * n_pos), 0.5, 6.0))
    weights[neg] *= float(np.clip(total / (2.0 * n_neg), 0.5, 6.0))
    return np.clip(weights, 0.05, 20.0).astype(np.float32)


def _l2_triple_gate_targets(
    edge_raw: np.ndarray,
    train_mask: np.ndarray,
    frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    edge = np.asarray(edge_raw, dtype=np.float64).ravel()
    tm = np.asarray(train_mask, dtype=bool).ravel()
    finite_train = tm & np.isfinite(edge)
    abs_train = np.abs(edge[finite_train])
    target_rate = _l2_target_trade_rate()
    if abs_train.size == 0:
        trade_floor = _env_float_clipped("L2_TRADE_LABEL_FLOOR_MIN", 0.02, lo=0.0, hi=2.0)
    else:
        trade_floor = float(np.quantile(abs_train, 1.0 - target_rate))
        trade_floor = float(max(trade_floor, _env_float_clipped("L2_TRADE_LABEL_FLOOR_MIN", 0.02, lo=0.0, hi=2.0)))
    side_floor = float(
        max(
            trade_floor * _env_float_clipped("L2_SIDE_LABEL_FLOOR_MULT", 1.0, lo=0.25, hi=4.0),
            _env_float_clipped("L2_SIDE_LABEL_FLOOR_MIN", 0.02, lo=0.0, hi=2.0),
        )
    )
    abs_edge = np.abs(edge)
    trade_floor_row = np.full(len(edge), trade_floor, dtype=np.float32)
    side_floor_row = np.full(len(edge), side_floor, dtype=np.float32)
    trade_base = np.ones(len(edge), dtype=np.float32)
    long_state_mult = np.ones(len(edge), dtype=np.float32)
    short_state_mult = np.ones(len(edge), dtype=np.float32)
    if frame is not None:
        pa = pa_state_arrays_from_frame(frame)
        trend = np.asarray(pa["pa_state_trend_strength"], dtype=np.float64)
        follow = np.asarray(pa["pa_state_followthrough_quality"], dtype=np.float64)
        range_risk = np.asarray(pa["pa_state_range_risk"], dtype=np.float64)
        breakout = np.asarray(pa["pa_state_breakout_failure_risk"], dtype=np.float64)
        pullback = np.asarray(pa["pa_state_pullback_exhaustion"], dtype=np.float64)
        always_in = np.asarray(pa["pa_state_always_in_bias"], dtype=np.float64)
        structure_veto = (
            pd.to_numeric(frame["pa_ctx_structure_veto"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            if "pa_ctx_structure_veto" in frame.columns
            else np.zeros(len(edge), dtype=np.float64)
        )
        floor_scale = np.clip(
            1.0
            + 0.24 * range_risk
            + 0.18 * breakout
            + 0.10 * pullback
            + 0.08 * structure_veto
            - 0.16 * trend
            - 0.10 * follow,
            0.75,
            1.45,
        )
        side_scale = np.clip(
            1.0
            + 0.18 * range_risk
            + 0.18 * breakout
            + 0.12 * pullback
            + 0.06 * structure_veto
            - 0.12 * trend
            - 0.08 * follow,
            0.80,
            1.45,
        )
        trade_floor_row = (trade_floor * floor_scale).astype(np.float32)
        side_floor_row = (side_floor * side_scale).astype(np.float32)
        trade_base = np.clip(
            1.0
            + 0.20 * trend
            + 0.15 * follow
            + 0.10 * np.abs(always_in)
            - 0.18 * range_risk
            - 0.18 * breakout
            - 0.10 * pullback,
            0.45,
            1.65,
        ).astype(np.float32)
        long_state_mult = np.clip(
            1.0 + 0.32 * np.maximum(always_in, 0.0) + 0.16 * trend + 0.10 * follow - 0.16 * range_risk - 0.14 * breakout - 0.08 * pullback,
            0.45,
            1.90,
        ).astype(np.float32)
        short_state_mult = np.clip(
            1.0 + 0.32 * np.maximum(-always_in, 0.0) + 0.16 * trend + 0.10 * follow - 0.16 * range_risk - 0.14 * breakout - 0.08 * pullback,
            0.45,
            1.90,
        ).astype(np.float32)
    y_trade = (abs_edge >= trade_floor_row).astype(np.int64)
    y_long = (edge >= side_floor_row).astype(np.int64)
    y_short = (edge <= -side_floor_row).astype(np.int64)

    trade_w = _l2_rebalanced_binary_weights(y_trade, trade_base)

    neutral_w = _env_float_clipped("L2_SIDE_NEUTRAL_WEIGHT", 0.20, lo=0.05, hi=5.0)
    opposite_w = _env_float_clipped("L2_SIDE_OPPOSITE_WEIGHT", 0.55, lo=0.05, hi=5.0)
    same_side_w = _env_float_clipped("L2_SIDE_SAME_SIGN_WEIGHT", 1.0, lo=0.1, hi=10.0)
    informative_mult = _env_float_clipped("L2_SIDE_INFORMATIVE_MULT", 2.0, lo=1.0, hi=10.0)

    long_base = np.full(len(edge), neutral_w, dtype=np.float32)
    long_base[edge < 0.0] = opposite_w
    long_base[edge > 0.0] = same_side_w
    long_base[edge >= side_floor_row] *= informative_mult
    long_base *= long_state_mult

    short_base = np.full(len(edge), neutral_w, dtype=np.float32)
    short_base[edge > 0.0] = opposite_w
    short_base[edge < 0.0] = same_side_w
    short_base[edge <= -side_floor_row] *= informative_mult
    short_base *= short_state_mult

    long_w = _l2_rebalanced_binary_weights(y_long, long_base)
    short_w = _l2_rebalanced_binary_weights(y_short, short_base)

    stats = {
        "target_trade_rate": float(target_rate),
        "trade_floor": float(trade_floor),
        "side_floor": float(side_floor),
        "trade_floor_row_mean": float(np.mean(trade_floor_row)),
        "side_floor_row_mean": float(np.mean(side_floor_row)),
        "train_trade_rate": float(np.mean(y_trade[finite_train])) if finite_train.any() else 0.0,
        "train_long_rate": float(np.mean(y_long[finite_train])) if finite_train.any() else 0.0,
        "train_short_rate": float(np.mean(y_short[finite_train])) if finite_train.any() else 0.0,
    }
    return y_trade, y_long, y_short, trade_w, long_w, short_w, stats


def _l2_apply_triple_gate_side_policy(
    long_p: np.ndarray,
    short_p: np.ndarray,
    *,
    short_bias: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    long_raw = np.clip(np.asarray(long_p, dtype=np.float64).ravel(), 0.0, None)
    short_raw = np.clip(np.asarray(short_p, dtype=np.float64).ravel(), 0.0, None)
    bias = np.asarray(short_bias, dtype=np.float64)
    if bias.ndim == 0:
        bias = np.full(len(long_raw), float(bias), dtype=np.float64)
    else:
        bias = np.broadcast_to(bias.reshape(-1), long_raw.shape)
    bias = np.clip(bias, 0.25, 4.0)
    return long_raw, (short_raw * bias)


def _l2_compose_probs_from_triple_gate(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    *,
    short_bias: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trade = np.clip(np.asarray(trade_p, dtype=np.float64).ravel(), 0.0, 1.0)
    long_raw, short_raw = _l2_apply_triple_gate_side_policy(long_p, short_p, short_bias=short_bias)
    side_sum = long_raw + short_raw
    long_cond = np.divide(long_raw, side_sum, out=np.full_like(long_raw, 0.5), where=side_sum > 1e-6)
    short_cond = 1.0 - long_cond
    p_long = trade * long_cond
    p_short = trade * short_cond
    p_neu = np.clip(1.0 - trade, 0.0, 1.0)
    probs = np.column_stack([p_long, p_neu, p_short]).astype(np.float32)
    return long_cond.astype(np.float32), short_cond.astype(np.float32), probs


def _l2_hard_decision_from_triple_gate(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    *,
    trade_threshold: float | np.ndarray,
    short_bias: float | np.ndarray = 1.0,
) -> np.ndarray:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    long_raw, short_raw = _l2_apply_triple_gate_side_policy(long_p, short_p, short_bias=short_bias)
    thr = np.asarray(trade_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(trade), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), trade.shape)
    out = np.ones(len(trade), dtype=np.int64)
    active = trade >= thr
    out[active & (long_raw >= short_raw)] = 0
    out[active & (long_raw < short_raw)] = 2
    return out


def _l2_hard_decode_from_triple_gate(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    decision_probs: np.ndarray,
    *,
    trade_threshold: float | np.ndarray,
    short_bias: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    hard_cls = _l2_hard_decision_from_triple_gate(
        trade_p,
        long_p,
        short_p,
        trade_threshold=trade_threshold,
        short_bias=short_bias,
    )
    prob_mat = np.asarray(decision_probs, dtype=np.float32)
    confidence = prob_mat[np.arange(len(prob_mat)), hard_cls].astype(np.float32)
    return hard_cls.astype(np.int64), confidence


def _l2_expected_edge_from_triple_gate(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    size_pred: np.ndarray,
    mfe_pred: np.ndarray,
    mae_pred: np.ndarray,
    *,
    trade_threshold: float | np.ndarray,
    short_bias: float | np.ndarray = 1.0,
) -> np.ndarray:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    long_score, short_score = _l2_apply_triple_gate_side_policy(long_p, short_p, short_bias=short_bias)
    size = np.asarray(size_pred, dtype=np.float64).ravel()
    thr = np.asarray(trade_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(trade), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), trade.shape)
    active_strength = np.clip((trade - thr) / np.maximum(1.0 - thr, 1e-3), 0.0, 1.0)
    edge = active_strength * size * (long_score - short_score)
    return np.clip(edge, -5.0, 5.0).astype(np.float32)


def _l2_trade_score(gate_p: np.ndarray, signed_edge_pred: np.ndarray) -> np.ndarray:
    gate = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    edge = np.asarray(signed_edge_pred, dtype=np.float64).ravel()
    return (gate * np.abs(edge)).astype(np.float64)


def _l2_compose_probs_from_gate_edge(
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    *,
    trade_score_threshold: float | np.ndarray,
    trade_score_temperature: float | np.ndarray,
    edge_temperature: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    edge = np.asarray(signed_edge_pred, dtype=np.float64).ravel()
    thr = np.asarray(trade_score_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(edge), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), edge.shape)
    score_temp = np.asarray(trade_score_temperature, dtype=np.float64)
    if score_temp.ndim == 0:
        score_temp = np.full(len(edge), float(score_temp), dtype=np.float64)
    else:
        score_temp = np.broadcast_to(score_temp.reshape(-1), edge.shape)
    score_temp = np.maximum(score_temp, 1e-4)
    edge_temp = np.asarray(edge_temperature, dtype=np.float64)
    if edge_temp.ndim == 0:
        edge_temp = np.full(len(edge), float(edge_temp), dtype=np.float64)
    else:
        edge_temp = np.broadcast_to(edge_temp.reshape(-1), edge.shape)
    edge_temp = np.maximum(edge_temp, 1e-4)
    trade_score = _l2_trade_score(gate, edge)
    active_p = _sigmoid((trade_score - thr) / score_temp)
    dir_p = _sigmoid(edge / edge_temp)
    p_long = active_p * dir_p
    p_short = active_p * (1.0 - dir_p)
    p_neu = np.clip(1.0 - active_p, 0.0, 1.0)
    probs = np.column_stack([p_long, p_neu, p_short]).astype(np.float32)
    return trade_score.astype(np.float32), dir_p.astype(np.float32), probs


def _l2_hard_decision_from_gate_edge(
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    *,
    trade_score_threshold: float | np.ndarray,
) -> np.ndarray:
    edge = np.asarray(signed_edge_pred, dtype=np.float64).ravel()
    thr = np.asarray(trade_score_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(edge), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), edge.shape)
    trade = _l2_trade_score(gate_p, edge) >= thr
    out = np.ones(len(edge), dtype=np.int64)
    out[trade & (edge > 0.0)] = 0
    out[trade & (edge < 0.0)] = 2
    return out


def _l2_hard_decode_from_gate_edge(
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    decision_probs: np.ndarray,
    *,
    trade_score_threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hard_cls = _l2_hard_decision_from_gate_edge(
        gate_p,
        signed_edge_pred,
        trade_score_threshold=trade_score_threshold,
    )
    prob_mat = np.asarray(decision_probs, dtype=np.float32)
    confidence = prob_mat[np.arange(len(prob_mat)), hard_cls].astype(np.float32)
    return hard_cls.astype(np.int64), confidence


def _split_mask_for_tuning_and_report(
    time_key: pd.Series | np.ndarray,
    base_mask: np.ndarray,
    *,
    tune_frac: float,
    min_rows_each: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_mask, dtype=bool)
    idx = np.flatnonzero(base)
    tune_mask = np.zeros_like(base, dtype=bool)
    report_mask = np.zeros_like(base, dtype=bool)
    if idx.size == 0:
        return tune_mask, report_mask
    ts = np.asarray(pd.to_datetime(time_key))
    idx = idx[np.argsort(ts[idx])]
    if idx.size < 2 * min_rows_each:
        tune_mask[idx] = True
        report_mask[idx] = True
        return tune_mask, report_mask
    split = int(round(idx.size * float(np.clip(tune_frac, 0.2, 0.8))))
    split = max(min_rows_each, min(idx.size - min_rows_each, split))
    tune_mask[idx[:split]] = True
    report_mask[idx[split:]] = True
    return tune_mask, report_mask


def _fit_binary_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> IsotonicRegression | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return calib


def _apply_binary_calibrator(p: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)


@dataclass
class PlattCalibrator:
    slope: float
    intercept: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64).ravel()
        logits = self.slope * arr + self.intercept
        return _sigmoid(logits).astype(np.float64)

    def get_params_for_monitoring(self) -> dict[str, float]:
        return {
            "slope": float(self.slope),
            "intercept": float(self.intercept),
        }


def _fit_platt_calibrator(y_true: np.ndarray, score: np.ndarray) -> PlattCalibrator | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    x = np.asarray(score, dtype=np.float64).ravel()
    valid = np.isfinite(x)
    if int(np.sum(valid)) < 100:
        return None
    y = y[valid]
    x = x[valid]
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
    clf.fit(x.reshape(-1, 1), y)
    return PlattCalibrator(
        slope=float(clf.coef_[0, 0]),
        intercept=float(clf.intercept_[0]),
    )


def _apply_platt_calibrator(score: np.ndarray, calibrator: PlattCalibrator | None) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64).ravel()
    if calibrator is None:
        return np.clip(arr, 0.0, 1.0)
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)


def _l2_hard_decision_from_gate_dir(gate_p: np.ndarray, dir_p: np.ndarray, thr: float) -> np.ndarray:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    dir_p = np.asarray(dir_p, dtype=np.float64).ravel()
    thr_arr = np.asarray(thr, dtype=np.float64)
    if thr_arr.ndim == 0:
        thr_arr = np.full(len(gate_p), float(thr_arr), dtype=np.float64)
    else:
        thr_arr = np.broadcast_to(thr_arr.reshape(-1), gate_p.shape)
    out = np.ones(len(gate_p), dtype=np.int64)
    trade = gate_p >= thr_arr
    out[trade & (dir_p >= 0.5)] = 0
    out[trade & (dir_p < 0.5)] = 2
    return out


def _l2_predict_gate_dir_probs(
    gate_model: lgb.Booster,
    direction_model: lgb.Booster | None,
    X: np.ndarray,
    *,
    trade_threshold: float,
    direction_head_type: str = "probability",
    direction_temperature: float | np.ndarray = 0.35,
    direction_center: float | np.ndarray = 0.5,
    gate_calibrator: IsotonicRegression | None = None,
    gate_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_scores = gate_model.predict(X).astype(np.float64) if gate_raw is None else np.asarray(gate_raw, dtype=np.float64).ravel()
    gate_p = _apply_binary_calibrator(gate_scores, gate_calibrator)
    thr_arr = np.asarray(trade_threshold, dtype=np.float64)
    if thr_arr.ndim == 0:
        thr_arr = np.full(len(X), float(thr_arr), dtype=np.float64)
    else:
        thr_arr = np.broadcast_to(thr_arr.reshape(-1), (len(X),))
    dir_p = np.full(len(X), 0.5, dtype=np.float64)
    if direction_model is not None:
        temp_arr = np.asarray(direction_temperature, dtype=np.float64)
        if temp_arr.ndim == 0:
            temp_arr = np.full(len(X), float(temp_arr), dtype=np.float64)
        else:
            temp_arr = np.broadcast_to(temp_arr.reshape(-1), (len(X),))
        center_arr = np.asarray(direction_center, dtype=np.float64)
        if center_arr.ndim == 0:
            center_arr = np.full(len(X), float(center_arr), dtype=np.float64)
        else:
            center_arr = np.broadcast_to(center_arr.reshape(-1), (len(X),))
        m = gate_p >= thr_arr
        if m.any():
            raw = direction_model.predict(X[m]).astype(np.float64)
            if direction_head_type == "signed_edge_regression":
                z = np.clip(raw / np.maximum(temp_arr[m], 1e-3), -12.0, 12.0)
                dir_p[m] = _l2_recenter_direction_prob(
                    1.0 / (1.0 + np.exp(-z)),
                    center=center_arr[m],
                )
            else:
                dir_p[m] = _l2_recenter_direction_prob(raw, center=center_arr[m])
    decision_probs = _l2_compose_probs_from_gate_dir(gate_p, dir_p)
    return gate_p.astype(np.float32), dir_p.astype(np.float32), decision_probs


def _l2_hard_decode_outputs(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    decision_probs: np.ndarray,
    *,
    trade_threshold: float,
    confidence_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hard_cls = _l2_hard_decision_from_gate_dir(gate_p, dir_p, trade_threshold)
    gate_arr = np.asarray(gate_p, dtype=np.float32).ravel()
    prob_mat = np.asarray(decision_probs, dtype=np.float32)
    neutral_conf = np.clip(1.0 - gate_arr, 0.0, 1.0)
    chosen_prob = prob_mat[np.arange(len(prob_mat)), hard_cls]
    active_conf = np.maximum(chosen_prob, gate_arr)
    confidence = np.where(hard_cls == 1, neutral_conf, active_conf).astype(np.float32)
    if confidence_scale is not None:
        scale = np.asarray(confidence_scale, dtype=np.float32).ravel()
        if scale.size == 1:
            scale = np.full(len(confidence), float(scale[0]), dtype=np.float32)
        else:
            scale = np.broadcast_to(scale, confidence.shape)
        confidence = np.where(hard_cls == 1, confidence, np.clip(confidence * scale, 0.0, 1.0)).astype(np.float32)
    return hard_cls.astype(np.int64), confidence


def _search_l2_trade_threshold(
    gate_p: np.ndarray,
    *,
    target_trade_rate: float = 0.10,
    min_trade_rate: float | None = None,
    max_trade_rate: float | None = None,
) -> float:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    if gate_p.size == 0:
        return 0.35
    if min_trade_rate is None:
        min_trade_rate = float(os.environ.get("L2_TRADE_THR_SEARCH_MIN", "0.05"))
    if max_trade_rate is None:
        max_trade_rate = float(os.environ.get("L2_TRADE_THR_SEARCH_MAX", "0.12"))
    target = float(np.clip(target_trade_rate, min_trade_rate, max_trade_rate))
    thr = float(np.quantile(gate_p, 1.0 - target))
    realized = float(np.mean(gate_p >= thr))
    print("\n  [L2] trade_threshold search on live active probability", flush=True)
    for rate in sorted({min_trade_rate, target, max_trade_rate}):
        cand = float(np.quantile(gate_p, 1.0 - rate))
        cand_rate = float(np.mean(gate_p >= cand))
        mark = "  *" if abs(rate - target) < 1e-9 else ""
        print(f"    target_trade_rate={rate:.3f}  threshold={cand:.4f}  realized={cand_rate:.3f}{mark}", flush=True)
    print(f"  [L2] selected trade_threshold={thr:.4f}  target_trade_rate={target:.3f}  realized={realized:.3f}", flush=True)
    return thr


def _log_l2_signed_edge_val_diagnostics(
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    y_gate: np.ndarray,
    y_decision: np.ndarray,
    true_edge: np.ndarray,
    trade_score_threshold: float,
    trade_score_temperature: float,
    edge_temperature: float,
    *,
    label: str = "signed-edge gate",
) -> None:
    y_gate = np.asarray(y_gate, dtype=np.int64).ravel()
    y_decision = np.asarray(y_decision, dtype=np.int64).ravel()
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    pred_edge = np.asarray(signed_edge_pred, dtype=np.float64).ravel()
    true_edge = np.asarray(true_edge, dtype=np.float64).ravel()
    trade_score = _l2_trade_score(gate_p, pred_edge)
    print(f"\n  [L2] val — {label}", flush=True)
    try:
        auc_g = float(roc_auc_score(y_gate, gate_p))
    except ValueError:
        auc_g = float("nan")
    pred_gate = (trade_score >= float(trade_score_threshold)).astype(np.int32)
    print(
        f"    gate AUC={auc_g:.4f}  trade_score_threshold={float(trade_score_threshold):.4f}  "
        f"trade_score_temperature={float(trade_score_temperature):.4f}  edge_temperature={float(edge_temperature):.4f}",
        flush=True,
    )
    print(
        "    gate classification_report:\n"
        + classification_report(y_gate, pred_gate, target_names=["no_trade", "trade"], digits=4, zero_division=0),
        flush=True,
    )
    active = y_decision != 1
    corr_all = pearson_corr(pred_edge, true_edge)
    corr_active = pearson_corr(pred_edge[active], true_edge[active]) if active.any() else float("nan")
    sign_acc = float(np.mean(np.sign(pred_edge[active]) == np.sign(true_edge[active]))) if active.any() else float("nan")
    print(
        f"    signed_edge corr_all={corr_all:.4f}  corr_active={corr_active:.4f}  sign_acc_active={sign_acc:.4f}",
        flush=True,
    )
    print(
        f"    signed_edge pred pcts={np.round(np.percentile(pred_edge, [1, 5, 25, 50, 75, 95, 99]), 4).tolist()}",
        flush=True,
    )
    pred_hard = _l2_hard_decision_from_gate_edge(
        gate_p,
        pred_edge,
        trade_score_threshold=trade_score_threshold,
    )
    print(
        f"\n  [L2] val — hard signed-edge vs truth: pred_active={float(np.mean(pred_hard != 1)):.3f}  "
        f"true_active={float(np.mean(y_decision != 1)):.3f}",
        flush=True,
    )


def _log_l2_triple_gate_val_diagnostics(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    y_trade: np.ndarray,
    y_long: np.ndarray,
    y_short: np.ndarray,
    y_decision: np.ndarray,
    true_edge: np.ndarray,
    trade_threshold: float,
    short_bias: float = 1.0,
) -> None:
    y_trade = np.asarray(y_trade, dtype=np.int64).ravel()
    y_long = np.asarray(y_long, dtype=np.int64).ravel()
    y_short = np.asarray(y_short, dtype=np.int64).ravel()
    y_decision = np.asarray(y_decision, dtype=np.int64).ravel()
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    long_raw = np.asarray(long_p, dtype=np.float64).ravel()
    short_raw = np.asarray(short_p, dtype=np.float64).ravel()
    edge = np.asarray(true_edge, dtype=np.float64).ravel()
    print("\n  [L2] val — triple gate", flush=True)
    try:
        auc_trade = float(roc_auc_score(y_trade, trade))
    except ValueError:
        auc_trade = float("nan")
    pred_trade = (trade >= float(trade_threshold)).astype(np.int32)
    print(
        f"    trade_gate AUC={auc_trade:.4f}  trade_threshold={float(trade_threshold):.4f}  short_bias={float(short_bias):.3f}",
        flush=True,
    )
    print(
        "    trade_gate classification_report:\n"
        + classification_report(y_trade, pred_trade, target_names=["no_trade", "trade"], digits=4, zero_division=0),
        flush=True,
    )
    candidate = y_trade == 1
    pos_edge = np.clip(edge, 0.0, None)
    neg_edge = np.clip(-edge, 0.0, None)
    if candidate.any():
        corr_long = pearson_corr(long_raw[candidate], pos_edge[candidate])
        corr_short = pearson_corr(short_raw[candidate], neg_edge[candidate])
        pred_long = (long_raw[candidate] >= short_raw[candidate]).astype(np.int32)
        pred_short = (short_raw[candidate] > long_raw[candidate]).astype(np.int32)
        print(
            f"    long_score corr(candidate)={corr_long:.4f}  recall={float(np.mean(pred_long[y_long[candidate] == 1] == 1)) if np.any(y_long[candidate] == 1) else float('nan'):.4f}",
            flush=True,
        )
        print(
            f"    short_score corr(candidate)={corr_short:.4f}  recall={float(np.mean(pred_short[y_short[candidate] == 1] == 1)) if np.any(y_short[candidate] == 1) else float('nan'):.4f}",
            flush=True,
        )
    long_adj, short_adj = _l2_apply_triple_gate_side_policy(long_raw, short_raw, short_bias=short_bias)
    long_cond, short_cond, probs = _l2_compose_probs_from_triple_gate(trade, long_raw, short_raw, short_bias=short_bias)
    pred_hard, _ = _l2_hard_decode_from_triple_gate(
        trade,
        long_raw,
        short_raw,
        probs,
        trade_threshold=trade_threshold,
        short_bias=short_bias,
    )
    signed_side = long_adj - short_adj
    active = pred_hard != 1
    corr_all = pearson_corr(signed_side, edge)
    corr_active = pearson_corr(signed_side[active], edge[active]) if active.any() else float("nan")
    sign_acc = float(np.mean(np.sign(signed_side[active]) == np.sign(edge[active]))) if active.any() else float("nan")
    print(
        f"    side_signal corr_all={corr_all:.4f}  corr_active={corr_active:.4f}  sign_acc_active={sign_acc:.4f}",
        flush=True,
    )
    print(
        f"\n  [L2] val — hard triple-gate vs truth: pred_active={float(np.mean(pred_hard != 1)):.3f}  "
        f"true_active={float(np.mean(y_decision != 1)):.3f}",
        flush=True,
    )


def _log_l2_extended_val_metrics(
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    decision_probs: np.ndarray,
    hard_decision: np.ndarray | None,
    aux_active_mask: np.ndarray | None,
    y_size: np.ndarray,
    size_pred: np.ndarray,
    y_mfe: np.ndarray,
    mfe_pred: np.ndarray,
    y_mae: np.ndarray,
    mae_pred: np.ndarray,
    expected_edge_pred: np.ndarray | None = None,
    true_edge: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
) -> None:
    """Extra val diagnostics: multiclass Brier/ECE, lift, L1 entropy↔L2 conf, regression tails & degen."""
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return
    yv = y_decision[vm]
    Pv = np.asarray(decision_probs[vm], dtype=np.float64)
    Pv = np.clip(Pv, 1e-15, 1.0)
    Pv = Pv / Pv.sum(axis=1, keepdims=True)
    try:
        ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
    except ValueError:
        ll = float("nan")
    br = brier_multiclass(yv, Pv, 3)
    ece = ece_multiclass_maxprob(yv, Pv)
    if hard_decision is None:
        pred = np.argmax(Pv, axis=1)
    else:
        pred = np.asarray(hard_decision, dtype=np.int64)[vm]
    acc = float(accuracy_score(yv, pred))
    f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
    cm = confusion_matrix(yv, pred, labels=[0, 1, 2])
    conf = np.max(Pv, axis=1)
    f1w = float(f1_score(yv, pred, average="weighted", zero_division=0))
    print("\n  [L2] val — decision (extended)", flush=True)
    print(
        f"    log_loss={ll:.4f}  Brier={br:.4f}  ECE(max-prob)={ece:.4f}  acc={acc:.4f}  F1_macro={f1m:.4f}  F1_weighted={f1w:.4f}",
        flush=True,
    )
    print(f"    confusion [rows=true 0/1/2, cols=pred]:\n    {cm}", flush=True)
    print(
        "    classification report:\n"
        + classification_report(
            yv,
            pred,
            labels=[0, 1, 2],
            target_names=["long", "neutral", "short"],
            digits=4,
            zero_division=0,
        ),
        flush=True,
    )
    pct_rows = np.percentile(Pv, [5, 25, 50, 75, 95], axis=0)
    print(
        f"    P(long/neutral/short) pct [5,25,50,75,95] rows:\n"
        f"      long:    {pct_rows[:, 0]}\n"
        f"      neutral: {pct_rows[:, 1]}\n"
        f"      short:   {pct_rows[:, 2]}",
        flush=True,
    )
    try:
        dfq = pd.DataFrame({"conf": conf, "ok": (pred == yv).astype(np.float64)})
        dfq["bin"] = pd.qcut(dfq["conf"], 10, duplicates="drop")
        lift = dfq.groupby("bin", observed=True)["ok"].agg(["mean", "count"])
        print(f"    accuracy lift by confidence decile:\n{lift}", flush=True)
    except Exception as ex:
        print(f"    (skip lift table: {ex})", flush=True)
    for cls_idx, cls_name in ((0, "long"), (2, "short")):
        mask = yv == cls_idx
        if not mask.any():
            continue
        recall = float(np.mean(pred[mask] == cls_idx))
        pcts = np.percentile(Pv[mask, cls_idx], [5, 25, 50, 75, 95])
        print(
            f"    {cls_name}: n={int(mask.sum()):,}  recall={recall:.4f}  prob_pcts={np.round(pcts, 4).tolist()}",
            flush=True,
        )
    neutral_pred_rate = float(np.mean(pred == 1))
    long_pred_rate = float(np.mean(pred == 0))
    short_pred_rate = float(np.mean(pred == 2))
    if neutral_pred_rate > 0.95:
        print(f"    WARNING: still collapsing toward neutral ({100.0 * neutral_pred_rate:.1f}% predicted neutral)", flush=True)
    elif neutral_pred_rate > 0.85:
        print(f"    WARNING: borderline neutral-heavy predictions ({100.0 * neutral_pred_rate:.1f}%)", flush=True)
    else:
        print(f"    neutral prediction rate={100.0 * neutral_pred_rate:.1f}%", flush=True)
    print(
        f"    predicted class mix: long={100.0 * long_pred_rate:.1f}%  neutral={100.0 * neutral_pred_rate:.1f}%  "
        f"short={100.0 * short_pred_rate:.1f}%",
        flush=True,
    )
    active_pred = pred != 1
    if active_pred.any():
        print(
            f"    active-side mix: long_share={float(np.mean(pred[active_pred] == 0)):.3f}  "
            f"short_share={float(np.mean(pred[active_pred] == 2)):.3f}",
            flush=True,
        )
    R = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64)[vm]
    R = np.clip(R, 1e-12, 1.0)
    R = R / R.sum(axis=1, keepdims=True)
    ent = -np.sum(R * np.log(R), axis=1)
    c_l1_l2 = pearson_corr(ent, conf)
    print(f"    corr(L1 regime entropy, L2 max prob)={c_l1_l2:.4f}", flush=True)
    if expected_edge_pred is not None and true_edge is not None:
        e_pred = np.asarray(expected_edge_pred, dtype=np.float64).ravel()
        e_true = np.asarray(true_edge, dtype=np.float64).ravel()
        corr_all = pearson_corr(e_pred[vm], e_true[vm])
        active = vm & (y_decision != 1)
        corr_active = pearson_corr(e_pred[active], e_true[active]) if active.any() else float("nan")
        sign_acc = float(np.mean(np.sign(e_pred[active]) == np.sign(e_true[active]))) if active.any() else float("nan")
        print(
            f"    expected_edge: corr_all={corr_all:.4f}  corr_active={corr_active:.4f}  sign_acc_active={sign_acc:.4f}",
            flush=True,
        )
        rid_all = np.argmax(frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64), axis=1)
        tm = (
            np.asarray(test_mask, dtype=bool).ravel()
            if test_mask is not None and np.asarray(test_mask, dtype=bool).any()
            else None
        )
        if tm is not None:
            corr_test = pearson_corr(e_pred[tm], e_true[tm])
            act_t = tm & (y_decision != 1)
            corr_test_active = pearson_corr(e_pred[act_t], e_true[act_t]) if act_t.any() else float("nan")
            print(
                f"    expected_edge holdout: corr_all={corr_test:.4f}  corr_active={corr_test_active:.4f}",
                flush=True,
            )
        min_reg_n = 30
        for r in range(NUM_REGIME_CLASSES):
            mv = vm & (rid_all == r)
            n_v = int(np.sum(mv))
            n_t_precheck = int(np.sum(tm & (rid_all == r))) if tm is not None else 0
            if n_v == 0 and n_t_precheck == 0:
                continue
            parts: list[str] = [
                f"regime={r} ({STATE_NAMES.get(r, str(r))})",
            ]
            if n_v >= min_reg_n:
                c_r = pearson_corr(e_pred[mv], e_true[mv])
                act = mv & (y_decision != 1)
                c_a = pearson_corr(e_pred[act], e_true[act]) if act.any() else float("nan")
                wv = "  [WARN val n<50]" if n_v < 50 else ""
                parts.append(f"val n={n_v} corr={c_r:.4f} corr_active={c_a:.4f}{wv}")
            elif n_v > 0:
                parts.append(f"val n={n_v} (corr skipped: n<{min_reg_n})")
            elif tm is not None and n_t_precheck > 0:
                parts.append("val n=0")
            else:
                continue
            if tm is not None:
                mt = tm & (rid_all == r)
                n_t = int(np.sum(mt))
                if n_t >= min_reg_n:
                    c_ht = pearson_corr(e_pred[mt], e_true[mt])
                    act_h = mt & (y_decision != 1)
                    c_ha = pearson_corr(e_pred[act_h], e_true[act_h]) if act_h.any() else float("nan")
                    wt = "  [WARN holdout n<50]" if n_t < 50 else ""
                    parts.append(f"holdout n={n_t} corr={c_ht:.4f} corr_active={c_ha:.4f}{wt}")
                elif n_t > 0:
                    parts.append(f"holdout n={n_t} (corr skipped: n<{min_reg_n})")
            print(f"    expected_edge {' | '.join(parts)}", flush=True)

    if aux_active_mask is None:
        av = vm & (y_decision != 1)
    else:
        av = vm & np.asarray(aux_active_mask, dtype=bool).ravel()
    if av.sum() >= 5:
        yt = y_size[av].astype(np.float64)
        yp = size_pred[av].astype(np.float64)
        mae_s = float(mean_absolute_error(yt, yp))
        rmse_s = float(np.sqrt(mean_squared_error(yt, yp)))
        r2_s = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
        c_s = pearson_corr(yt, yp)
        std_s, degen_s = regression_degen_flag(yp)
        print("\n  [L2] val — size (active bars only)", flush=True)
        print(
            f"    MAE={mae_s:.4f}  RMSE={rmse_s:.4f}  R2={r2_s:.4f}  corr={c_s:.4f}  pred_std={std_s:.6f}  degen={degen_s}",
            flush=True,
        )
    else:
        print("\n  [L2] val — size: (skip: too few active val rows)", flush=True)

    for name, yt_a, yp_a in (
        ("mfe", y_mfe, mfe_pred),
        ("mae", y_mae, mae_pred),
    ):
        mask = av
        yt = yt_a[mask].astype(np.float64)
        yp = yp_a[mask].astype(np.float64)
        if yt.size < 5:
            print(f"\n  [L2] val — {name} head: (skip: too few active val rows)", flush=True)
            continue
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
        c = pearson_corr(yt, yp)
        tail = tail_mae_truth_upper(yt, yp, 90.0)
        print(f"\n  [L2] val — {name} head (active bars only)", flush=True)
        print(
            f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  corr={c:.4f}  tail_MAE(P90+)={tail:.4f}",
            flush=True,
        )


def _l2_l1b_importance_bucket(name: str) -> str:
    if name in {"l1b_edge_pred", "l1b_dq_pred"}:
        return "l1b_supervised_pred"
    if name.startswith("l1b_cluster_prob_"):
        return "l1b_unsup_cluster"
    if name.startswith("l1b_latent_") or name in {"l1b_novelty_score", "l1b_regime_change_score"}:
        return "l1b_unsup_latent_novelty"
    if name in {"l1b_sector_relative_strength", "l1b_correlation_regime", "l1b_market_breadth"}:
        return "l1b_context"
    if name.startswith("l1b_atom_"):
        return "l1b_atom_internal"
    if name.startswith("l1b_"):
        return "l1b_deterministic_semantic"
    return "non_l1b"


def _log_l2_l1b_gain_importance_by_group(model: lgb.Booster, feature_cols: list[str], label: str) -> None:
    """Round 0.5: aggregate LightGBM gain by L1b block (diagnostic for pruning dead features)."""
    if os.environ.get("L2_L1B_IMPORTANCE_LOG", "1").strip().lower() in {"0", "false", "no"}:
        return
    try:
        names = list(model.feature_name())
    except Exception:
        names = list(feature_cols)
    try:
        imp = model.feature_importance(importance_type="gain")
    except Exception:
        return
    imp = np.asarray(imp, dtype=np.float64).ravel()
    if imp.size != len(names):
        print(
            f"  [L2] l1b gain importance ({label}): skip (len mismatch imp={imp.size} names={len(names)})",
            flush=True,
        )
        return
    buckets: dict[str, float] = {}
    for n, g in zip(names, imp):
        b = _l2_l1b_importance_bucket(str(n))
        buckets[b] = float(buckets.get(b, 0.0) + float(g))
    total = float(sum(buckets.values()))
    if total <= 0.0:
        print(f"  [L2] l1b gain importance ({label}): all zero", flush=True)
        return
    order = sorted(buckets.keys(), key=lambda k: buckets[k], reverse=True)
    parts = [f"{k}={100.0 * buckets[k] / total:.1f}%" for k in order if buckets[k] > 0.0]
    print(f"  [L2] l1b gain share by block — {label}: " + "  ".join(parts), flush=True)
    l1b_only = sum(v for k, v in buckets.items() if k.startswith("l1b_"))
    if l1b_only > 0.0:
        det = buckets.get("l1b_deterministic_semantic", 0.0)
        print(
            f"    └─ L1b subtotal={100.0 * l1b_only / total:.1f}% of model gain  "
            f"(deterministic_semantic={100.0 * det / total:.1f}%)",
            flush=True,
        )


def _log_l2_l1b_ablation(
    X: np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    gate_model: lgb.Booster,
    signed_edge_model: lgb.Booster,
    *,
    signed_edge_prep: dict[str, Any] | None,
    trade_score_threshold: float,
    trade_score_temperature: float,
    edge_temperature: float,
    gate_calibrator: IsotonicRegression | None,
) -> None:
    l1b_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_")]
    if not l1b_idx:
        print("\n  [L2] l1b ablation: skip (no l1b_* columns after selection)", flush=True)
        return
    cluster_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_cluster_prob_")]
    latent_idx = [
        i
        for i, c in enumerate(feature_cols)
        if c.startswith("l1b_latent_") or c in {"l1b_novelty_score", "l1b_regime_change_score"}
    ]
    unsup_idx = sorted(set(cluster_idx + latent_idx))
    deterministic_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_") and i not in unsup_idx]
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return

    def _eval_probs(gate_p: np.ndarray, signed_edge_pred: np.ndarray, prob_mat: np.ndarray) -> tuple[float, float, float, float, float]:
        Pv = np.asarray(prob_mat[vm], dtype=np.float64)
        Pv = np.clip(Pv, 1e-15, 1.0)
        Pv = Pv / Pv.sum(axis=1, keepdims=True)
        yv = np.asarray(y_decision[vm], dtype=np.int64)
        thr_eval = np.asarray(trade_score_threshold, dtype=np.float32)
        if thr_eval.ndim == 0:
            thr_eval = np.full(int(np.sum(vm)), float(thr_eval), dtype=np.float32)
        else:
            thr_eval = np.asarray(thr_eval, dtype=np.float32).ravel()[vm]
        try:
            ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
        except ValueError:
            ll = float("nan")
        pred, _ = _l2_hard_decode_from_gate_edge(
            gate_p=np.asarray(gate_p, dtype=np.float32)[vm],
            signed_edge_pred=np.asarray(signed_edge_pred, dtype=np.float32)[vm],
            decision_probs=Pv,
            trade_score_threshold=thr_eval,
        )
        f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
        trade_rate = float(np.mean(pred != 1))
        long_rate = float(np.mean(pred == 0))
        short_rate = float(np.mean(pred == 2))
        return ll, f1m, trade_rate, long_rate, short_rate

    def _eval_variant(X_variant: np.ndarray) -> tuple[float, float, float, float, float]:
        gate_p = _apply_binary_calibrator(gate_model.predict(X_variant).astype(np.float64), gate_calibrator).astype(np.float32)
        signed_edge_pred = _l2_signed_edge_predict(signed_edge_model, X_variant, signed_edge_prep)
        _, _, probs = _l2_compose_probs_from_gate_edge(
            gate_p,
            signed_edge_pred,
            trade_score_threshold=trade_score_threshold,
            trade_score_temperature=trade_score_temperature,
            edge_temperature=edge_temperature,
        )
        return _eval_probs(gate_p, signed_edge_pred, probs)

    base_ll, base_f1, base_trade, base_long, base_short = _eval_variant(X)
    cluster_ll = cluster_f1 = cluster_trade = cluster_long = cluster_short = float("nan")
    if cluster_idx:
        X_no_cluster = np.array(X, copy=True)
        X_no_cluster[:, cluster_idx] = 0.0
        cluster_ll, cluster_f1, cluster_trade, cluster_long, cluster_short = _eval_variant(X_no_cluster)
    latent_ll = latent_f1 = latent_trade = latent_long = latent_short = float("nan")
    if latent_idx:
        X_no_latent = np.array(X, copy=True)
        X_no_latent[:, latent_idx] = 0.0
        latent_ll, latent_f1, latent_trade, latent_long, latent_short = _eval_variant(X_no_latent)
    deterministic_only_ll = deterministic_only_f1 = deterministic_only_trade = deterministic_only_long = deterministic_only_short = float("nan")
    if unsup_idx:
        X_deterministic_only = np.array(X, copy=True)
        X_deterministic_only[:, unsup_idx] = 0.0
        deterministic_only_ll, deterministic_only_f1, deterministic_only_trade, deterministic_only_long, deterministic_only_short = _eval_variant(X_deterministic_only)
    unsupervised_only_ll = unsupervised_only_f1 = unsupervised_only_trade = unsupervised_only_long = unsupervised_only_short = float("nan")
    if deterministic_idx:
        X_unsupervised_only = np.array(X, copy=True)
        X_unsupervised_only[:, deterministic_idx] = 0.0
        unsupervised_only_ll, unsupervised_only_f1, unsupervised_only_trade, unsupervised_only_long, unsupervised_only_short = _eval_variant(X_unsupervised_only)
    X_no_l1b = np.array(X, copy=True)
    X_no_l1b[:, l1b_idx] = 0.0
    abl_ll, abl_f1, abl_trade, abl_long, abl_short = _eval_variant(X_no_l1b)
    print("\n  [L2] l1b val ablation (zero l1b_* at inference)", flush=True)
    print(
        f"    baseline:      log_loss={base_ll:.4f}  F1_macro={base_f1:.4f}  trade_rate={base_trade:.3f}  "
        f"long_rate={base_long:.3f}  short_rate={base_short:.3f}",
        flush=True,
    )
    if cluster_idx:
        print(
            f"    no_cluster:    log_loss={cluster_ll:.4f}  F1_macro={cluster_f1:.4f}  trade_rate={cluster_trade:.3f}  "
            f"long_rate={cluster_long:.3f}  short_rate={cluster_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(no_cluster-base): log_loss={cluster_ll - base_ll:+.4f}  F1_macro={cluster_f1 - base_f1:+.4f}  "
            f"trade_rate={cluster_trade - base_trade:+.3f}  long_rate={cluster_long - base_long:+.3f}  "
            f"short_rate={cluster_short - base_short:+.3f}",
            flush=True,
        )
    if latent_idx:
        print(
            f"    no_latent:     log_loss={latent_ll:.4f}  F1_macro={latent_f1:.4f}  trade_rate={latent_trade:.3f}  "
            f"long_rate={latent_long:.3f}  short_rate={latent_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(no_lat-base):   log_loss={latent_ll - base_ll:+.4f}  F1_macro={latent_f1 - base_f1:+.4f}  "
            f"trade_rate={latent_trade - base_trade:+.3f}  long_rate={latent_long - base_long:+.3f}  "
            f"short_rate={latent_short - base_short:+.3f}",
            flush=True,
        )
    if unsup_idx:
        print(
            f"    deterministic_only: log_loss={deterministic_only_ll:.4f}  F1_macro={deterministic_only_f1:.4f}  trade_rate={deterministic_only_trade:.3f}  "
            f"long_rate={deterministic_only_long:.3f}  short_rate={deterministic_only_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(det-base):      log_loss={deterministic_only_ll - base_ll:+.4f}  F1_macro={deterministic_only_f1 - base_f1:+.4f}  "
            f"trade_rate={deterministic_only_trade - base_trade:+.3f}  long_rate={deterministic_only_long - base_long:+.3f}  "
            f"short_rate={deterministic_only_short - base_short:+.3f}",
            flush=True,
        )
    if deterministic_idx:
        print(
            f"    unsupervised_only: log_loss={unsupervised_only_ll:.4f}  F1_macro={unsupervised_only_f1:.4f}  trade_rate={unsupervised_only_trade:.3f}  "
            f"long_rate={unsupervised_only_long:.3f}  short_rate={unsupervised_only_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(unsup-base):    log_loss={unsupervised_only_ll - base_ll:+.4f}  F1_macro={unsupervised_only_f1 - base_f1:+.4f}  "
            f"trade_rate={unsupervised_only_trade - base_trade:+.3f}  long_rate={unsupervised_only_long - base_long:+.3f}  "
            f"short_rate={unsupervised_only_short - base_short:+.3f}",
            flush=True,
        )
    print(
        f"    without_l1b:   log_loss={abl_ll:.4f}  F1_macro={abl_f1:.4f}  trade_rate={abl_trade:.3f}  "
        f"long_rate={abl_long:.3f}  short_rate={abl_short:.3f}",
        flush=True,
    )
    print(
        f"    delta(no_l1b-base):  log_loss={abl_ll - base_ll:+.4f}  F1_macro={abl_f1 - base_f1:+.4f}  "
        f"trade_rate={abl_trade - base_trade:+.3f}  long_rate={abl_long - base_long:+.3f}  "
        f"short_rate={abl_short - base_short:+.3f}",
        flush=True,
    )


def _log_l2_l1b_ablation_triple_gate(
    X: np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    trade_model: lgb.Booster,
    long_model: lgb.Booster,
    short_model: lgb.Booster,
    *,
    trade_threshold: float,
    short_bias: float | np.ndarray = 1.0,
    gate_calibrators: dict[str, IsotonicRegression | None],
) -> None:
    l1b_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_")]
    if not l1b_idx:
        print("\n  [L2] l1b ablation: skip (no l1b_* columns after selection)", flush=True)
        return
    cluster_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_cluster_prob_")]
    latent_idx = [
        i
        for i, c in enumerate(feature_cols)
        if c.startswith("l1b_latent_") or c in {"l1b_novelty_score", "l1b_regime_change_score"}
    ]
    unsup_idx = sorted(set(cluster_idx + latent_idx))
    deterministic_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_") and i not in unsup_idx]
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return

    def _eval_variant(X_variant: np.ndarray) -> tuple[float, float, float, float, float]:
        trade_p = _apply_binary_calibrator(trade_model.predict(X_variant).astype(np.float64), gate_calibrators.get("trade")).astype(np.float32)
        long_p = _apply_binary_calibrator(long_model.predict(X_variant).astype(np.float64), gate_calibrators.get("long")).astype(np.float32)
        short_p = _apply_binary_calibrator(short_model.predict(X_variant).astype(np.float64), gate_calibrators.get("short")).astype(np.float32)
        _, _, probs = _l2_compose_probs_from_triple_gate(trade_p, long_p, short_p, short_bias=short_bias)
        Pv = np.asarray(probs[vm], dtype=np.float64)
        Pv = np.clip(Pv, 1e-15, 1.0)
        Pv = Pv / Pv.sum(axis=1, keepdims=True)
        yv = np.asarray(y_decision[vm], dtype=np.int64)
        thr_eval = np.asarray(trade_threshold, dtype=np.float32)
        if thr_eval.ndim == 0:
            thr_eval = np.full(int(np.sum(vm)), float(thr_eval), dtype=np.float32)
        else:
            thr_eval = np.asarray(thr_eval, dtype=np.float32).ravel()[vm]
        bias_eval = np.asarray(short_bias, dtype=np.float32)
        if bias_eval.ndim == 0:
            bias_eval = np.full(int(np.sum(vm)), float(bias_eval), dtype=np.float32)
        else:
            bias_eval = np.asarray(bias_eval, dtype=np.float32).ravel()[vm]
        try:
            ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
        except ValueError:
            ll = float("nan")
        pred, _ = _l2_hard_decode_from_triple_gate(
            trade_p=np.asarray(trade_p, dtype=np.float32)[vm],
            long_p=np.asarray(long_p, dtype=np.float32)[vm],
            short_p=np.asarray(short_p, dtype=np.float32)[vm],
            decision_probs=Pv,
            trade_threshold=thr_eval,
            short_bias=bias_eval,
        )
        f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
        trade_rate = float(np.mean(pred != 1))
        long_rate = float(np.mean(pred == 0))
        short_rate = float(np.mean(pred == 2))
        return ll, f1m, trade_rate, long_rate, short_rate

    base_ll, base_f1, base_trade, base_long, base_short = _eval_variant(X)
    cluster_ll = cluster_f1 = cluster_trade = cluster_long = cluster_short = float("nan")
    if cluster_idx:
        X_no_cluster = np.array(X, copy=True)
        X_no_cluster[:, cluster_idx] = 0.0
        cluster_ll, cluster_f1, cluster_trade, cluster_long, cluster_short = _eval_variant(X_no_cluster)
    latent_ll = latent_f1 = latent_trade = latent_long = latent_short = float("nan")
    if latent_idx:
        X_no_latent = np.array(X, copy=True)
        X_no_latent[:, latent_idx] = 0.0
        latent_ll, latent_f1, latent_trade, latent_long, latent_short = _eval_variant(X_no_latent)
    deterministic_only_ll = deterministic_only_f1 = deterministic_only_trade = deterministic_only_long = deterministic_only_short = float("nan")
    if unsup_idx:
        X_deterministic_only = np.array(X, copy=True)
        X_deterministic_only[:, unsup_idx] = 0.0
        deterministic_only_ll, deterministic_only_f1, deterministic_only_trade, deterministic_only_long, deterministic_only_short = _eval_variant(X_deterministic_only)
    unsupervised_only_ll = unsupervised_only_f1 = unsupervised_only_trade = unsupervised_only_long = unsupervised_only_short = float("nan")
    if deterministic_idx:
        X_unsupervised_only = np.array(X, copy=True)
        X_unsupervised_only[:, deterministic_idx] = 0.0
        unsupervised_only_ll, unsupervised_only_f1, unsupervised_only_trade, unsupervised_only_long, unsupervised_only_short = _eval_variant(X_unsupervised_only)
    X_no_l1b = np.array(X, copy=True)
    X_no_l1b[:, l1b_idx] = 0.0
    abl_ll, abl_f1, abl_trade, abl_long, abl_short = _eval_variant(X_no_l1b)
    print("\n  [L2] l1b val ablation (zero l1b_* at inference)", flush=True)
    print(
        f"    baseline:      log_loss={base_ll:.4f}  F1_macro={base_f1:.4f}  trade_rate={base_trade:.3f}  "
        f"long_rate={base_long:.3f}  short_rate={base_short:.3f}",
        flush=True,
    )
    if cluster_idx:
        print(
            f"    no_cluster:    log_loss={cluster_ll:.4f}  F1_macro={cluster_f1:.4f}  trade_rate={cluster_trade:.3f}  "
            f"long_rate={cluster_long:.3f}  short_rate={cluster_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(no_cluster-base): log_loss={cluster_ll - base_ll:+.4f}  F1_macro={cluster_f1 - base_f1:+.4f}  "
            f"trade_rate={cluster_trade - base_trade:+.3f}  long_rate={cluster_long - base_long:+.3f}  "
            f"short_rate={cluster_short - base_short:+.3f}",
            flush=True,
        )
    if latent_idx:
        print(
            f"    no_latent:     log_loss={latent_ll:.4f}  F1_macro={latent_f1:.4f}  trade_rate={latent_trade:.3f}  "
            f"long_rate={latent_long:.3f}  short_rate={latent_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(no_lat-base):   log_loss={latent_ll - base_ll:+.4f}  F1_macro={latent_f1 - base_f1:+.4f}  "
            f"trade_rate={latent_trade - base_trade:+.3f}  long_rate={latent_long - base_long:+.3f}  "
            f"short_rate={latent_short - base_short:+.3f}",
            flush=True,
        )
    if unsup_idx:
        print(
            f"    deterministic_only: log_loss={deterministic_only_ll:.4f}  F1_macro={deterministic_only_f1:.4f}  trade_rate={deterministic_only_trade:.3f}  "
            f"long_rate={deterministic_only_long:.3f}  short_rate={deterministic_only_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(det-base):      log_loss={deterministic_only_ll - base_ll:+.4f}  F1_macro={deterministic_only_f1 - base_f1:+.4f}  "
            f"trade_rate={deterministic_only_trade - base_trade:+.3f}  long_rate={deterministic_only_long - base_long:+.3f}  "
            f"short_rate={deterministic_only_short - base_short:+.3f}",
            flush=True,
        )
    if deterministic_idx:
        print(
            f"    unsupervised_only: log_loss={unsupervised_only_ll:.4f}  F1_macro={unsupervised_only_f1:.4f}  trade_rate={unsupervised_only_trade:.3f}  "
            f"long_rate={unsupervised_only_long:.3f}  short_rate={unsupervised_only_short:.3f}",
            flush=True,
        )
        print(
            f"    delta(unsup-base):    log_loss={unsupervised_only_ll - base_ll:+.4f}  F1_macro={unsupervised_only_f1 - base_f1:+.4f}  "
            f"trade_rate={unsupervised_only_trade - base_trade:+.3f}  long_rate={unsupervised_only_long - base_long:+.3f}  "
            f"short_rate={unsupervised_only_short - base_short:+.3f}",
            flush=True,
        )
    print(
        f"    without_l1b:   log_loss={abl_ll:.4f}  F1_macro={abl_f1:.4f}  trade_rate={abl_trade:.3f}  "
        f"long_rate={abl_long:.3f}  short_rate={abl_short:.3f}",
        flush=True,
    )
    print(
        f"    delta(no_l1b-base):  log_loss={abl_ll - base_ll:+.4f}  F1_macro={abl_f1 - base_f1:+.4f}  "
        f"trade_rate={abl_trade - base_trade:+.3f}  long_rate={abl_long - base_long:+.3f}  "
        f"short_rate={abl_short - base_short:+.3f}",
        flush=True,
    )


def _session_context(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["time_key"])
    out = pd.DataFrame(index=df.index)
    out["l2_session_progress"] = ((ts.dt.hour * 60 + ts.dt.minute) / (24.0 * 60.0)).astype(np.float32)
    out["l2_is_opening_hour"] = (ts.dt.hour <= 10).astype(np.float32)
    return out


def _l2_target_trade_rate() -> float:
    return float(np.clip(float(os.environ.get("L2_TARGET_TRADE_RATE", "0.08")), 0.05, 0.15))


def _l2_decision_edge_tau() -> float:
    return float(max(0.0, float(os.environ.get("STACK_DECISION_EDGE_TAU", "0.05"))))


def _env_float_candidates(key: str, default: list[float], *, lo: float, hi: float) -> list[float]:
    raw = os.environ.get(key, "").strip()
    vals = default
    if raw:
        parsed: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(float(part))
        if parsed:
            vals = parsed
    clipped = sorted({float(np.clip(v, lo, hi)) for v in vals})
    return clipped or [float(np.clip(default[0], lo, hi))]


def _env_float_clipped(key: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(key, "").strip()
    val = default if not raw else float(raw)
    return float(np.clip(val, lo, hi))


def _env_int_clipped(key: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(key, "").strip()
    val = default if not raw else int(raw)
    return int(np.clip(val, lo, hi))


def _l2_boost_rounds() -> int:
    default = 250 if FAST_TRAIN_MODE else 1200
    return _env_int_clipped("L2_BOOST_ROUNDS", default, lo=50, hi=5000)


def _l2_model_lgb_params(kind: str) -> dict[str, Any]:
    k = str(kind).strip().lower()
    prefix = {
        "gate": "L2_GATE",
        "direction": "L2_DIRECTION",
        "reg": "L2_REG",
    }[k]
    defaults: dict[str, Any] = {
        "gate": {
            "learning_rate": 0.01,
            "num_leaves": 15,
            "max_depth": 5,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_child_samples": 80,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": 42,
        },
        "direction": {
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 30,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": 44,
        },
        "reg": {
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 7,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 60,
            "lambda_l1": 0.05,
            "lambda_l2": 1.0,
            "seed": 43,
        },
    }[k]
    return {
        "learning_rate": _env_float_clipped(f"{prefix}_LEARNING_RATE", defaults["learning_rate"], lo=1e-4, hi=1.0),
        "num_leaves": _env_int_clipped(f"{prefix}_NUM_LEAVES", defaults["num_leaves"], lo=2, hi=1024),
        "max_depth": _env_int_clipped(f"{prefix}_MAX_DEPTH", defaults["max_depth"], lo=-1, hi=64),
        "feature_fraction": _env_float_clipped(f"{prefix}_FEATURE_FRACTION", defaults["feature_fraction"], lo=0.1, hi=1.0),
        "bagging_fraction": _env_float_clipped(f"{prefix}_BAGGING_FRACTION", defaults["bagging_fraction"], lo=0.1, hi=1.0),
        "bagging_freq": _env_int_clipped(f"{prefix}_BAGGING_FREQ", defaults["bagging_freq"], lo=0, hi=64),
        "min_child_samples": _env_int_clipped(f"{prefix}_MIN_CHILD_SAMPLES", defaults["min_child_samples"], lo=1, hi=10000),
        "lambda_l1": _env_float_clipped(f"{prefix}_LAMBDA_L1", defaults["lambda_l1"], lo=0.0, hi=100.0),
        "lambda_l2": _env_float_clipped(f"{prefix}_LAMBDA_L2", defaults["lambda_l2"], lo=0.0, hi=100.0),
        "seed": _env_int_clipped(f"{prefix}_SEED", defaults["seed"], lo=0, hi=2_147_483_647),
    }


def _policy_vol_quantiles(values: np.ndarray, *, fit_mask: np.ndarray | None = None, n_buckets: int = 3) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    if fit_mask is not None:
        mask &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[mask]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0 or n_buckets <= 1:
        return []
    qs = np.linspace(0.0, 1.0, int(n_buckets) + 1)[1:-1]
    return [float(np.quantile(finite, q)) for q in qs]


def _bucketize_by_quantiles(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not quantiles:
        return np.zeros(len(arr), dtype=np.int32)
    bins = np.asarray(sorted(float(x) for x in quantiles), dtype=np.float64)
    safe = np.nan_to_num(arr, nan=float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0)
    return np.searchsorted(bins, safe, side="right").astype(np.int32)


def _regime_ids_from_probs(regime_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(regime_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] == 0:
        return np.zeros(len(probs), dtype=np.int32)
    safe = np.nan_to_num(probs, nan=0.0)
    return np.argmax(safe, axis=1).astype(np.int32)


def _state_keys_from_regime_vol(regime_probs: np.ndarray, vol_values: np.ndarray, *, vol_quantiles: list[float]) -> np.ndarray:
    reg = _regime_ids_from_probs(regime_probs)
    vb = _bucketize_by_quantiles(vol_values, vol_quantiles)
    return np.asarray([f"r{int(r)}_v{int(v)}" for r, v in zip(reg, vb)], dtype=object)


def _l2_policy_state_keys(frame: pd.DataFrame, *, vol_quantiles: list[float]) -> np.ndarray:
    """Policy buckets: ``full`` = regime×vol×PA bucket (many states); ``coarse`` = 4 (HV/LV × trend/range)."""
    mode = os.environ.get("L2_POLICY_STATE_GRANULARITY", "coarse").strip().lower()
    if mode in {"full", "fine", "legacy"}:
        regime_probs = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
        vol_values = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
        base = _state_keys_from_regime_vol(regime_probs, vol_values, vol_quantiles=vol_quantiles)
        pa_bucket = pa_state_bucket_labels_from_frame(frame)
        return np.asarray([f"{key}_pa_{bucket}" for key, bucket in zip(base, pa_bucket)], dtype=object)
    vol_values = frame["l1a_vol_forecast"].to_numpy(dtype=np.float64, copy=False)
    vb = _bucketize_by_quantiles(vol_values, vol_quantiles)
    nvb = int(np.max(vb)) + 1 if vb.size else 1
    high_vol = vb >= max(0, nvb - 1)
    pa_bucket = pa_state_bucket_labels_from_frame(frame)
    is_trend = pa_bucket == PA_STATE_BUCKET_TREND
    out: list[str] = []
    for hv, tr in zip(high_vol, is_trend):
        out.append(f"{'HV' if hv else 'LV'}_{'TR' if tr else 'RN'}")
    return np.asarray(out, dtype=object)


def _l2_signed_edge_policy_arrays(
    meta: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vol_quantiles = [float(x) for x in (meta.get("policy_state_vol_quantiles") or [])]
    state_keys = _l2_policy_state_keys(frame, vol_quantiles=vol_quantiles)
    by_state = meta.get("conditional_policy_by_state") or {}
    n = len(frame)
    trade_thr = np.full(n, float(meta.get("trade_score_threshold", 0.10)), dtype=np.float32)
    trade_temp = np.full(n, float(meta.get("trade_score_temperature", 0.05)), dtype=np.float32)
    edge_temp = np.full(n, float(meta.get("edge_temperature", 0.10)), dtype=np.float32)
    for key, params in by_state.items():
        m = state_keys == key
        if not np.any(m):
            continue
        trade_thr[m] = float(params.get("trade_score_threshold", trade_thr[m][0]))
        trade_temp[m] = float(params.get("trade_score_temperature", trade_temp[m][0]))
        edge_temp[m] = float(params.get("edge_temperature", edge_temp[m][0]))
    return state_keys, trade_thr, trade_temp, edge_temp


def _l2_triple_gate_policy_arrays(
    meta: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vol_quantiles = [float(x) for x in (meta.get("policy_state_vol_quantiles") or [])]
    state_keys = _l2_policy_state_keys(frame, vol_quantiles=vol_quantiles)
    by_state = meta.get("conditional_policy_by_state") or {}
    n = len(frame)
    trade_thr = np.full(n, float(meta.get("trade_threshold", 0.50)), dtype=np.float32)
    short_bias = np.full(n, float(meta.get("short_bias", 1.0)), dtype=np.float32)
    for key, params in by_state.items():
        m = state_keys == key
        if not np.any(m):
            continue
        trade_thr[m] = float(params.get("trade_threshold", trade_thr[m][0]))
        short_bias[m] = float(params.get("short_bias", short_bias[m][0]))
    return state_keys, trade_thr, short_bias


def _l2_conditional_policy_arrays(
    meta: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_keys, trade_thr, trade_temp, _edge_temp = _l2_signed_edge_policy_arrays(meta, frame)
    n = len(frame)
    center = np.full(n, float(meta.get("direction_center", 0.5)), dtype=np.float32)
    reward = np.full(n, float(meta.get("expected_edge_reward_mult", 1.0)), dtype=np.float32)
    risk = np.full(n, float(meta.get("expected_edge_risk_mult", 1.0)), dtype=np.float32)
    return state_keys, trade_thr, trade_temp, center, reward, risk


def _conditional_tau_from_state(
    frame: pd.DataFrame,
    edge: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[float, list[float], dict[str, float], np.ndarray]:
    base_tau = _l2_decision_edge_tau()
    target_trade = _l2_target_trade_rate()
    tau_quantile = float(
        np.clip(
            1.0 - _env_float_clipped("L2_TAU_TARGET_TRADE_MULT", 2.0, lo=0.5, hi=4.0) * target_trade,
            _env_float_clipped("L2_TAU_QUANTILE_MIN", 0.50, lo=0.0, hi=0.99),
            _env_float_clipped("L2_TAU_QUANTILE_MAX", 0.88, lo=0.0, hi=0.99),
        )
    )
    vol_quantiles = _policy_vol_quantiles(frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False), fit_mask=train_mask)
    state_keys = _l2_policy_state_keys(frame, vol_quantiles=vol_quantiles)
    edge_abs = np.abs(np.asarray(edge, dtype=np.float64).ravel())
    train = np.asarray(train_mask, dtype=bool)
    finite_train = train & np.isfinite(edge_abs)
    global_tau = float(np.quantile(edge_abs[finite_train], tau_quantile)) if finite_train.any() else base_tau
    global_tau = float(max(base_tau * _env_float_clipped("L2_TAU_BASE_FLOOR_MULT", 0.5, lo=0.0, hi=5.0), global_tau))
    state_map: dict[str, float] = {}
    min_rows = max(80, int(os.environ.get("L2_CONDITIONAL_TAU_MIN_ROWS", "500")))
    for key in np.unique(state_keys[train]):
        m = finite_train & (state_keys == key)
        if int(np.sum(m)) < min_rows:
            continue
        state_tau = float(np.quantile(edge_abs[m], tau_quantile))
        state_map[str(key)] = float(
            np.clip(
                state_tau,
                base_tau * _env_float_clipped("L2_STATE_TAU_MIN_MULT", 0.5, lo=0.0, hi=5.0),
                max(
                    base_tau * _env_float_clipped("L2_STATE_TAU_MAX_BASE_MULT", 3.0, lo=0.1, hi=20.0),
                    global_tau * _env_float_clipped("L2_STATE_TAU_MAX_GLOBAL_MULT", 2.0, lo=0.1, hi=20.0),
                ),
            )
        )
    tau_row = np.full(len(frame), global_tau, dtype=np.float32)
    for key, val in state_map.items():
        tau_row[state_keys == key] = float(val)
    pa = pa_state_arrays_from_frame(frame)
    trend = np.asarray(pa["pa_state_trend_strength"], dtype=np.float64)
    follow = np.asarray(pa["pa_state_followthrough_quality"], dtype=np.float64)
    range_risk = np.asarray(pa["pa_state_range_risk"], dtype=np.float64)
    breakout = np.asarray(pa["pa_state_breakout_failure_risk"], dtype=np.float64)
    pullback = np.asarray(pa["pa_state_pullback_exhaustion"], dtype=np.float64)
    tau_scale = np.clip(
        1.0
        + 0.24 * range_risk
        + 0.18 * breakout
        + 0.12 * pullback
        - 0.18 * trend
        - 0.12 * follow,
        0.72,
        1.40,
    )
    tau_row = np.clip(
        np.asarray(tau_row, dtype=np.float64) * tau_scale,
        base_tau * _env_float_clipped("L2_STATE_TAU_MIN_MULT", 0.5, lo=0.0, hi=5.0),
        max(
            base_tau * _env_float_clipped("L2_STATE_TAU_MAX_BASE_MULT", 3.0, lo=0.1, hi=20.0),
            global_tau * _env_float_clipped("L2_STATE_TAU_MAX_GLOBAL_MULT", 2.0, lo=0.1, hi=20.0),
        ),
    ).astype(np.float32)
    for key in list(state_map):
        m = state_keys == key
        if np.any(m):
            state_map[key] = float(np.mean(tau_row[m]))
    print(
        f"  [L2] conditional decision tau: base={base_tau:.4f}  global={global_tau:.4f}  "
        f"states={len(state_map)}  vol_quantiles={np.round(vol_quantiles, 4).tolist()}  "
        f"tau_row_mean={float(np.mean(tau_row)):.4f}",
        flush=True,
    )
    return global_tau, vol_quantiles, state_map, tau_row


def _l2_expected_edge_proxy(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    size_pred: np.ndarray,
    mfe_pred: np.ndarray,
    mae_pred: np.ndarray,
    *,
    reward_mult: float,
    risk_mult: float,
    active_floor: float | np.ndarray = 0.0,
) -> np.ndarray:
    gate = np.asarray(gate_p, dtype=np.float64).ravel()
    direction = np.asarray(dir_p, dtype=np.float64).ravel()
    size = np.asarray(size_pred, dtype=np.float64).ravel()
    mfe = np.asarray(mfe_pred, dtype=np.float64).ravel()
    mae = np.asarray(mae_pred, dtype=np.float64).ravel()
    signed_dir = np.clip(2.0 * direction - 1.0, -1.0, 1.0)
    reward = np.asarray(reward_mult, dtype=np.float64)
    if reward.ndim == 0:
        reward = np.full(len(gate), float(reward), dtype=np.float64)
    else:
        reward = np.broadcast_to(reward.reshape(-1), gate.shape)
    risk = np.asarray(risk_mult, dtype=np.float64)
    if risk.ndim == 0:
        risk = np.full(len(gate), float(risk), dtype=np.float64)
    else:
        risk = np.broadcast_to(risk.reshape(-1), gate.shape)
    edge_floor = np.asarray(active_floor, dtype=np.float64)
    if edge_floor.ndim == 0:
        edge_floor = np.full(len(gate), float(edge_floor), dtype=np.float64)
    else:
        edge_floor = np.broadcast_to(edge_floor.reshape(-1), gate.shape)
    payoff = reward * mfe - risk * mae
    active_strength = np.clip((gate - edge_floor) / np.maximum(1.0 - edge_floor, 1e-3), 0.0, 1.0)
    edge = active_strength * size * signed_dir * payoff
    return np.clip(edge, -5.0, 5.0).astype(np.float32)


def _l2_search_policy_params(
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    true_edge: np.ndarray,
) -> dict[str, float]:
    gate = np.asarray(gate_p, dtype=np.float64).ravel()
    pred_edge = np.asarray(signed_edge_pred, dtype=np.float64).ravel()
    edge = np.asarray(true_edge, dtype=np.float64).ravel()
    valid = np.isfinite(gate) & np.isfinite(pred_edge) & np.isfinite(edge)
    if not valid.any():
        return {
            "trade_score_threshold": 0.10,
            "trade_score_temperature": 0.05,
            "target_trade_rate": _l2_target_trade_rate(),
            "edge_temperature": 0.10,
        }
    gate = gate[valid]
    pred_edge = pred_edge[valid]
    edge = edge[valid]
    target_rate = float(np.clip(_l2_target_trade_rate(), 0.04, 0.20))
    true_floor = float(np.quantile(np.abs(edge), 1.0 - target_rate)) if edge.size else 0.0
    true_floor = float(max(true_floor, _env_float_clipped("L2_POLICY_TRUE_EDGE_FLOOR_MIN", 0.02, lo=0.0, hi=2.0)))
    truth = np.ones(len(edge), dtype=np.int64)
    truth[edge >= true_floor] = 0
    truth[edge <= -true_floor] = 2
    true_active = truth != 1
    trade_score = _l2_trade_score(gate, pred_edge)
    thr = float(np.quantile(trade_score, 1.0 - target_rate))
    score_temp = float(np.median(np.abs(trade_score - thr))) if trade_score.size else 0.05
    score_temp = float(np.clip(max(score_temp, 1e-4), 0.02, 2.0))
    edge_temp = float(np.median(np.abs(pred_edge[true_active]))) if true_active.any() else float(np.median(np.abs(pred_edge)))
    edge_temp = float(np.clip(max(edge_temp, 1e-4), 0.02, 2.0))
    _, _, decision_probs = _l2_compose_probs_from_gate_edge(
        gate,
        pred_edge,
        trade_score_threshold=thr,
        trade_score_temperature=score_temp,
        edge_temperature=edge_temp,
    )
    pred, _ = _l2_hard_decode_from_gate_edge(
        gate,
        pred_edge,
        decision_probs,
        trade_score_threshold=thr,
    )
    trade_rate = float(np.mean(pred != 1))
    f1m = float(f1_score(truth, pred, average="macro", zero_division=0))
    active_pred = pred != 1
    true_long_share = float(np.mean(truth[true_active] == 0)) if true_active.any() else 0.5
    long_share = float(np.mean(pred[active_pred] == 0)) if active_pred.any() else true_long_share
    long_recall = float(np.mean(pred[truth == 0] == 0)) if np.any(truth == 0) else 0.0
    short_recall = float(np.mean(pred[truth == 2] == 2)) if np.any(truth == 2) else 0.0
    corr_all = pearson_corr(pred_edge, edge)
    corr_active = pearson_corr(pred_edge[active_pred], edge[active_pred]) if active_pred.any() else float("nan")
    sign_acc = float(np.mean(np.sign(pred_edge[active_pred]) == np.sign(edge[active_pred]))) if active_pred.any() else 0.0
    best = {
        "trade_score_threshold": float(thr),
        "trade_score_temperature": float(score_temp),
        "target_trade_rate": float(target_rate),
        "edge_temperature": float(edge_temp),
        "true_edge_floor": float(true_floor),
        "trade_rate": float(trade_rate),
        "long_share_active": float(long_share),
        "long_recall": float(long_recall),
        "short_recall": float(short_recall),
        "corr_all": float(np.nan_to_num(corr_all, nan=0.0)),
        "corr_active": float(np.nan_to_num(corr_active, nan=0.0)),
        "f1_macro": float(f1m),
        "sign_acc_active": float(sign_acc),
    }
    print("\n  [L2] signed-edge policy derivation on val_tune", flush=True)
    print(
        f"  [L2] selected policy: trade_thr={best['trade_score_threshold']:.4f}  trade_temp={best['trade_score_temperature']:.4f}  "
        f"edge_temp={best['edge_temperature']:.4f}  target_trade_rate={best['target_trade_rate']:.3f}  "
        f"realized_trade_rate={best['trade_rate']:.3f}  "
        f"corr_active={best['corr_active']:.4f}  F1_macro={best['f1_macro']:.4f}  "
        f"long_recall={best.get('long_recall', float('nan')):.4f}",
        flush=True,
    )
    return best


def _l2_search_triple_gate_policy_params(
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    true_edge: np.ndarray,
) -> dict[str, float]:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    long_raw = np.asarray(long_p, dtype=np.float64).ravel()
    short_raw = np.asarray(short_p, dtype=np.float64).ravel()
    edge = np.asarray(true_edge, dtype=np.float64).ravel()
    valid = np.isfinite(trade) & np.isfinite(long_raw) & np.isfinite(short_raw) & np.isfinite(edge)
    if not valid.any():
        return {
            "trade_threshold": 0.50,
            "short_bias": 1.0,
            "target_trade_rate": _l2_target_trade_rate(),
            "true_edge_floor": 0.02,
        }
    trade = trade[valid]
    long_raw = np.clip(long_raw[valid], 0.0, None)
    short_raw = np.clip(short_raw[valid], 0.0, None)
    edge = edge[valid]
    target_rate = float(np.clip(_l2_target_trade_rate(), 0.04, 0.20))
    true_floor = float(np.quantile(np.abs(edge), 1.0 - target_rate)) if edge.size else 0.0
    true_floor = float(max(true_floor, _env_float_clipped("L2_POLICY_TRUE_EDGE_FLOOR_MIN", 0.02, lo=0.0, hi=2.0)))
    truth = np.ones(len(edge), dtype=np.int64)
    truth[edge >= true_floor] = 0
    truth[edge <= -true_floor] = 2
    thr = float(np.quantile(trade, 1.0 - target_rate))
    true_active = truth != 1
    true_long_share = float(np.mean(truth[true_active] == 0)) if true_active.any() else 0.5
    bias_grid_raw = os.environ.get("L2_TRIPLE_GATE_SHORT_BIAS_GRID", "").strip()
    if bias_grid_raw:
        bias_candidates = np.asarray(
            [float(x.strip()) for x in bias_grid_raw.split(",") if x.strip()],
            dtype=np.float64,
        )
    else:
        bias_candidates = np.asarray([0.70, 0.85, 1.00, 1.15, 1.35, 1.60, 2.00, 2.60], dtype=np.float64)
    bias_candidates = np.unique(np.clip(bias_candidates, 0.25, 4.0))
    best_score = -np.inf
    best: dict[str, float] = {
        "trade_threshold": float(thr),
        "short_bias": 1.0,
        "target_trade_rate": float(target_rate),
        "true_edge_floor": float(true_floor),
        "trade_rate": 0.0,
        "f1_macro": 0.0,
        "long_recall": 0.0,
        "short_recall": 0.0,
        "corr_all": 0.0,
        "corr_active": 0.0,
        "sign_acc_active": 0.0,
        "long_share_active": true_long_share,
        "selection_score": -np.inf,
    }
    for short_bias in bias_candidates.tolist():
        _, _, probs = _l2_compose_probs_from_triple_gate(trade, long_raw, short_raw, short_bias=short_bias)
        pred, _ = _l2_hard_decode_from_triple_gate(
            trade,
            long_raw,
            short_raw,
            probs,
            trade_threshold=thr,
            short_bias=short_bias,
        )
        trade_rate = float(np.mean(pred != 1))
        active_pred = pred != 1
        f1m = float(f1_score(truth, pred, average="macro", zero_division=0))
        long_recall = float(np.mean(pred[truth == 0] == 0)) if np.any(truth == 0) else 0.0
        short_recall = float(np.mean(pred[truth == 2] == 2)) if np.any(truth == 2) else 0.0
        long_share = float(np.mean(pred[active_pred] == 0)) if active_pred.any() else true_long_share
        side_balance = 1.0 - abs(long_share - true_long_share)
        long_adj, short_adj = _l2_apply_triple_gate_side_policy(long_raw, short_raw, short_bias=short_bias)
        signed_side = long_adj - short_adj
        corr_all = pearson_corr(signed_side, edge)
        corr_active = pearson_corr(signed_side[active_pred], edge[active_pred]) if active_pred.any() else float("nan")
        sign_acc = float(np.mean(np.sign(signed_side[active_pred]) == np.sign(edge[active_pred]))) if active_pred.any() else float("nan")
        score = float(
            f1m
            + 0.08 * short_recall
            + 0.04 * max(side_balance, 0.0)
            + 0.02 * float(np.nan_to_num(corr_active, nan=0.0))
        )
        if score > best_score:
            best_score = score
            best = {
                "trade_threshold": float(thr),
                "short_bias": float(short_bias),
                "target_trade_rate": float(target_rate),
                "true_edge_floor": float(true_floor),
                "trade_rate": float(trade_rate),
                "f1_macro": float(f1m),
                "long_recall": float(long_recall),
                "short_recall": float(short_recall),
                "corr_all": float(np.nan_to_num(corr_all, nan=0.0)),
                "corr_active": float(np.nan_to_num(corr_active, nan=0.0)),
                "sign_acc_active": float(np.nan_to_num(sign_acc, nan=0.0)),
                "long_share_active": float(long_share),
                "selection_score": float(score),
            }
    print("\n  [L2] triple-gate policy derivation on val_tune", flush=True)
    print(
        f"  [L2] selected policy: trade_thr={best['trade_threshold']:.4f}  short_bias={best['short_bias']:.3f}  "
        f"target_trade_rate={best['target_trade_rate']:.3f}  realized_trade_rate={best['trade_rate']:.3f}  "
        f"corr_active={best['corr_active']:.4f}  F1_macro={best['f1_macro']:.4f}  "
        f"long_recall={best['long_recall']:.4f}  short_recall={best['short_recall']:.4f}",
        flush=True,
    )
    return best


def _l2_search_triple_gate_conditional_policy(
    state_keys: np.ndarray,
    trade_p: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    true_edge: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    global_policy = _l2_search_triple_gate_policy_params(trade_p, long_p, short_p, true_edge)
    keys = np.asarray(state_keys, dtype=object).ravel()
    valid = (
        np.isfinite(np.asarray(trade_p, dtype=np.float64).ravel())
        & np.isfinite(np.asarray(long_p, dtype=np.float64).ravel())
        & np.isfinite(np.asarray(short_p, dtype=np.float64).ravel())
    )
    keys = keys[valid]
    by_state: dict[str, dict[str, float]] = {}
    min_rows = max(80, int(os.environ.get("L2_POLICY_MIN_STATE_ROWS", "500")))
    shrink_rows = max(1.0, float(os.environ.get("L2_POLICY_STATE_SHRINK_ROWS", "2000")))
    min_state_corr = float(os.environ.get("L2_POLICY_STATE_MIN_CORR", "0.15"))
    unique_keys = sorted({str(k) for k in keys.tolist()})
    if unique_keys:
        print(
            f"  [L2] conditional policy search: candidate_states={len(unique_keys)}  min_rows={min_rows}  shrink_rows={shrink_rows:.0f}  min_corr={min_state_corr:.2f}",
            flush=True,
        )
    for key in unique_keys:
        m = keys == key
        n_rows = int(np.sum(m))
        if n_rows < min_rows:
            continue
        raw_policy = _l2_search_triple_gate_policy_params(
            np.asarray(trade_p, dtype=np.float64).ravel()[valid][m],
            np.asarray(long_p, dtype=np.float64).ravel()[valid][m],
            np.asarray(short_p, dtype=np.float64).ravel()[valid][m],
            np.asarray(true_edge, dtype=np.float64).ravel()[valid][m],
        )
        tp_v = np.asarray(trade_p, dtype=np.float64).ravel()[valid][m]
        lp_v = np.asarray(long_p, dtype=np.float64).ravel()[valid][m]
        sp_v = np.asarray(short_p, dtype=np.float64).ravel()[valid][m]
        te_v = np.asarray(true_edge, dtype=np.float64).ravel()[valid][m]
        sig = (lp_v - sp_v) * tp_v
        fin = np.isfinite(sig) & np.isfinite(te_v)
        corr_st = float(pearson_corr(sig[fin], te_v[fin])) if int(np.sum(fin)) > 40 else -1.0
        if corr_st < min_state_corr:
            print(
                f"  [L2] conditional policy: state={key!r} edge_corr_proxy={corr_st:.4f} < {min_state_corr:.2f} → global shrink only",
                flush=True,
            )
            by_state[key] = dict(global_policy)
            continue
        shrink = float(n_rows / (n_rows + shrink_rows))
        state_policy = dict(raw_policy)
        for param_name in ("trade_threshold", "true_edge_floor", "short_bias"):
            state_policy[param_name] = float(
                shrink * float(raw_policy[param_name]) + (1.0 - shrink) * float(global_policy[param_name])
            )
        by_state[key] = state_policy
    print(f"  [L2] conditional policy states learned={len(by_state)}", flush=True)
    return global_policy, by_state


def _l2_search_conditional_policy(
    state_keys: np.ndarray,
    gate_p: np.ndarray,
    signed_edge_pred: np.ndarray,
    true_edge: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    global_policy = _l2_search_policy_params(
        gate_p,
        signed_edge_pred,
        true_edge,
    )
    keys = np.asarray(state_keys, dtype=object).ravel()
    valid = np.isfinite(np.asarray(gate_p, dtype=np.float64).ravel()) & np.isfinite(np.asarray(signed_edge_pred, dtype=np.float64).ravel())
    keys = keys[valid]
    by_state: dict[str, dict[str, float]] = {}
    min_rows = max(80, int(os.environ.get("L2_POLICY_MIN_STATE_ROWS", "500")))
    shrink_rows = max(1.0, float(os.environ.get("L2_POLICY_STATE_SHRINK_ROWS", "2000")))
    min_state_corr = float(os.environ.get("L2_POLICY_STATE_MIN_CORR", "0.15"))
    unique_keys = sorted({str(k) for k in keys.tolist()})
    if unique_keys:
        print(
            f"  [L2] conditional policy search: candidate_states={len(unique_keys)}  min_rows={min_rows}  shrink_rows={shrink_rows:.0f}  min_corr={min_state_corr:.2f}",
            flush=True,
        )
    for key in unique_keys:
        m = keys == key
        n_rows = int(np.sum(m))
        if n_rows < min_rows:
            continue
        raw_policy = _l2_search_policy_params(
            np.asarray(gate_p, dtype=np.float64).ravel()[valid][m],
            np.asarray(signed_edge_pred, dtype=np.float64).ravel()[valid][m],
            np.asarray(true_edge, dtype=np.float64).ravel()[valid][m],
        )
        se_v = np.asarray(signed_edge_pred, dtype=np.float64).ravel()[valid][m]
        te_v = np.asarray(true_edge, dtype=np.float64).ravel()[valid][m]
        fin = np.isfinite(se_v) & np.isfinite(te_v)
        corr_st = float(pearson_corr(se_v[fin], te_v[fin])) if int(np.sum(fin)) > 40 else -1.0
        if corr_st < min_state_corr:
            print(
                f"  [L2] conditional policy: state={key!r} signed_edge_corr={corr_st:.4f} < {min_state_corr:.2f} → global shrink only",
                flush=True,
            )
            by_state[key] = dict(global_policy)
            continue
        shrink = float(n_rows / (n_rows + shrink_rows))
        state_policy = dict(raw_policy)
        for param_name in ("trade_score_threshold", "trade_score_temperature", "edge_temperature", "true_edge_floor"):
            state_policy[param_name] = float(
                shrink * float(raw_policy[param_name]) + (1.0 - shrink) * float(global_policy[param_name])
            )
        by_state[key] = state_policy
    print(f"  [L2] conditional policy states learned={len(by_state)}", flush=True)
    return global_policy, by_state


def _l2_fusion_feature_frame(
    frame: pd.DataFrame,
    trade_p: np.ndarray,
    triple_probs: np.ndarray,
    triple_expected_edge: np.ndarray,
    signed_probs: np.ndarray,
    signed_edge_pred: np.ndarray,
    size_pred: np.ndarray,
    mfe_pred: np.ndarray,
    mae_pred: np.ndarray,
) -> pd.DataFrame:
    tp = np.asarray(trade_p, dtype=np.float32).ravel()
    triple_mat = np.asarray(triple_probs, dtype=np.float32)
    signed_mat = np.asarray(signed_probs, dtype=np.float32)
    triple_edge = np.asarray(triple_expected_edge, dtype=np.float32).ravel()
    signed_edge = np.asarray(signed_edge_pred, dtype=np.float32).ravel()
    out = pd.DataFrame(index=frame.index)
    out["trade_p"] = tp
    out["triple_long_p"] = triple_mat[:, 0]
    out["triple_short_p"] = triple_mat[:, 2]
    out["triple_dir_margin"] = (triple_mat[:, 0] - triple_mat[:, 2]).astype(np.float32)
    out["triple_expected_edge"] = triple_edge
    out["signed_long_p"] = signed_mat[:, 0]
    out["signed_short_p"] = signed_mat[:, 2]
    out["signed_dir_margin"] = (signed_mat[:, 0] - signed_mat[:, 2]).astype(np.float32)
    out["signed_edge_pred"] = signed_edge
    out["edge_mean"] = (0.5 * (triple_edge + signed_edge)).astype(np.float32)
    out["edge_gap"] = (triple_edge - signed_edge).astype(np.float32)
    out["edge_abs_gap"] = (np.abs(triple_edge) - np.abs(signed_edge)).astype(np.float32)
    out["edge_sign_agree"] = (np.sign(triple_edge) == np.sign(signed_edge)).astype(np.float32)
    out["size_pred"] = np.asarray(size_pred, dtype=np.float32).ravel()
    out["mfe_pred"] = np.asarray(mfe_pred, dtype=np.float32).ravel()
    out["mae_pred"] = np.asarray(mae_pred, dtype=np.float32).ravel()
    out["l2_dq_derived_aux"] = (out["mfe_pred"] - out["mae_pred"]).astype(np.float32)
    out["rr_proxy_raw"] = out["mfe_pred"] / np.maximum(out["mae_pred"], 0.05)
    pa_cols = (
        *PA_STATE_FEATURES,
        "pa_ctx_structure_veto",
        "pa_ctx_premise_break_long",
        "pa_ctx_premise_break_short",
        "pa_ctx_range_pressure",
    )
    for col in pa_cols:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(np.float32)
    for col in (
        "l1a_regime_entropy",
        "l1a_regime_top2_gap",
        "l1a_transition_risk",
        "l1a_vol_trend",
        "l1a_time_in_regime",
        "l1a_state_persistence",
        "l1b_edge_pred",
        "l1b_dq_pred",
        "l1b_cluster_top1",
        "l1b_cluster_top2_gap",
        "l1b_novelty_score",
        "l1b_regime_change_score",
        "l1b_novelty_x_vol",
        "l1b_regime_change_x_entropy",
        "l1b_unsup_pressure",
        "l1a_vol_forecast",
        "l1c_direction_prob",
        "l1c_direction_score",
        "l1c_confidence",
        "l1c_direction_strength",
        "l1c_is_warm",
        "l1c_weighted_dir",
        "l1c_dir_x_vol",
        "l1c_dir_x_state_persistence",
        "l1c_strength_x_conf",
        "l2_dir_x_edge_opportunity",
        "l2_dir_conf_x_edge_mag",
        "l2_dq_x_edge",
        "l2_session_progress",
        "l2_is_opening_hour",
    ):
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(np.float32)
    return out.fillna(0.0).astype(np.float32)


def _l2_fit_edge_fusion_model(
    fusion_frame: pd.DataFrame,
    true_edge: np.ndarray,
    fit_mask: np.ndarray,
) -> dict[str, Any]:
    X_df = fusion_frame.copy()
    feature_cols = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64, copy=False)
    y = np.asarray(true_edge, dtype=np.float64).ravel()
    fit = np.asarray(fit_mask, dtype=bool).ravel() & np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if int(np.sum(fit)) < max(50, len(feature_cols) + 5):
        clip_abs = float(np.clip(np.nanquantile(np.abs(y[np.isfinite(y)]), 0.995) if np.isfinite(y).any() else 1.0, 0.10, 5.0))
        return {
            "feature_cols": feature_cols,
            "feature_mean": np.zeros(len(feature_cols), dtype=np.float32).tolist(),
            "feature_scale": np.ones(len(feature_cols), dtype=np.float32).tolist(),
            "coef": np.zeros(len(feature_cols), dtype=np.float32).tolist(),
            "intercept": 0.0,
            "alpha": 1.0,
            "clip_abs": clip_abs,
            "fit_rows": int(np.sum(fit)),
            "score": 0.0,
        }

    y_fit = y[fit]
    clip_abs = float(np.clip(np.quantile(np.abs(y_fit), 0.995), 0.10, 5.0))
    y_fit = np.clip(y_fit, -clip_abs, clip_abs)
    mu = X[fit].mean(axis=0)
    sigma = X[fit].std(axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
    Xn_fit = (X[fit] - mu) / sigma
    y_mean = float(np.mean(y_fit))
    y_ctr = y_fit - y_mean
    alpha_candidates = _env_float_candidates(
        "L2_FUSION_ALPHA_GRID", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0], lo=1e-4, hi=1e4
    )

    best_score = -np.inf
    best_coef = np.zeros(Xn_fit.shape[1], dtype=np.float64)
    best_alpha = float(alpha_candidates[0])
    # Discrete alpha grid + ridge normal eqs on the fit slice (not sklearn RidgeCV).
    for alpha in alpha_candidates:
        gram = Xn_fit.T @ Xn_fit
        rhs = Xn_fit.T @ y_ctr
        coef = np.linalg.pinv(gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)) @ rhs
        pred_fit = y_mean + Xn_fit @ coef
        corr = float(np.nan_to_num(pearson_corr(pred_fit, y_fit), nan=0.0))
        active = np.abs(y_fit) >= max(_env_float_clipped("L2_FUSION_ACTIVE_EDGE_FLOOR", 0.02, lo=0.0, hi=2.0), clip_abs * 0.05)
        corr_active = float(np.nan_to_num(pearson_corr(pred_fit[active], y_fit[active]), nan=0.0)) if active.any() else corr
        sign_acc = float(np.mean(np.sign(pred_fit[active]) == np.sign(y_fit[active]))) if active.any() else 0.0
        score = corr + 0.20 * corr_active + 0.05 * sign_acc
        if score > best_score:
            best_score = score
            best_coef = coef
            best_alpha = float(alpha)
    model = {
        "feature_cols": feature_cols,
        "feature_mean": mu.astype(np.float32).tolist(),
        "feature_scale": sigma.astype(np.float32).tolist(),
        "coef": best_coef.astype(np.float32).tolist(),
        "intercept": float(y_mean),
        "alpha": best_alpha,
        "clip_abs": clip_abs,
        "fit_rows": int(np.sum(fit)),
        "score": float(best_score),
    }
    top_idx = np.argsort(np.abs(best_coef))[::-1][:6]
    top_terms = ", ".join(f"{feature_cols[i]}={best_coef[i]:+.3f}" for i in top_idx if abs(best_coef[i]) > 1e-6)
    print(
        f"  [L2] fusion edge model: alpha={best_alpha:.4f}  fit_rows={int(np.sum(fit))}  score={best_score:.4f}  top={top_terms or 'all ~0'}",
        flush=True,
    )
    return model


def _l2_predict_edge_fusion_model(fusion_frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    cols = list(model.get("feature_cols", []))
    if not cols:
        return np.zeros(len(fusion_frame), dtype=np.float32)
    X = fusion_frame.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=np.float64, copy=False)
    mu = np.asarray(model.get("feature_mean", []), dtype=np.float64)
    sigma = np.asarray(model.get("feature_scale", []), dtype=np.float64)
    coef = np.asarray(model.get("coef", []), dtype=np.float64)
    intercept = float(model.get("intercept", 0.0))
    clip_abs = float(model.get("clip_abs", 5.0))
    if mu.size != X.shape[1] or sigma.size != X.shape[1] or coef.size != X.shape[1]:
        return np.zeros(len(fusion_frame), dtype=np.float32)
    Xn = (X - mu) / np.where(np.abs(sigma) > 1e-6, sigma, 1.0)
    pred = intercept + Xn @ coef
    return np.clip(pred, -clip_abs, clip_abs).astype(np.float32)


def _l2_hierarchical_direction_prob(
    frame: pd.DataFrame,
    triple_probs: np.ndarray,
    signed_edge_pred: np.ndarray,
    *,
    signed_edge_temperature: float | np.ndarray = 0.05,
) -> np.ndarray:
    probs = np.asarray(triple_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 3:
        return np.full(len(frame), 0.5, dtype=np.float32)
    triple_long = np.clip(probs[:, 0], 0.0, 1.0)
    triple_short = np.clip(probs[:, 2], 0.0, 1.0)
    triple_active = np.clip(triple_long + triple_short, 1e-6, 1.0)
    triple_dir = np.clip(triple_long / triple_active, 1e-4, 1.0 - 1e-4)

    temp = np.asarray(signed_edge_temperature, dtype=np.float64)
    if temp.ndim == 0:
        temp = np.full(len(frame), float(temp), dtype=np.float64)
    else:
        temp = np.broadcast_to(temp.reshape(-1), (len(frame),))
    signed_dir = np.clip(_l2_direction_margin_to_prob(signed_edge_pred, temperature=1.0), 1e-4, 1.0 - 1e-4)
    signed_dir = np.clip(
        _l2_direction_margin_to_prob(np.asarray(signed_edge_pred, dtype=np.float64) / np.maximum(temp, 1e-3), temperature=1.0),
        1e-4,
        1.0 - 1e-4,
    )

    regime_probs = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64, copy=False)
    regime_probs = np.nan_to_num(regime_probs, nan=0.0)
    regime_sum = np.maximum(regime_probs.sum(axis=1, keepdims=True), 1e-12)
    regime_probs = regime_probs / regime_sum
    bull_mass = regime_probs[:, 0] + regime_probs[:, 1]
    bear_mass = regime_probs[:, 2] + regime_probs[:, 3]
    range_mass = regime_probs[:, 4] + regime_probs[:, 5]
    trend_bias = np.clip(bull_mass - bear_mass, -1.0, 1.0)
    state_persistence = pd.to_numeric(frame.get("l1a_state_persistence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    trend_bias = trend_bias * np.clip(0.50 + 0.50 * state_persistence, 0.1, 1.0)
    trend_bias = np.clip(trend_bias, -1.0, 1.0)
    entropy = -np.sum(np.clip(regime_probs, 1e-12, 1.0) * np.log(np.clip(regime_probs, 1e-12, 1.0)), axis=1)
    entropy_norm = np.clip(entropy / np.log(max(regime_probs.shape[1], 2)), 0.0, 1.0)
    l1c_score = pd.to_numeric(frame.get("l1c_direction_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    l1c_conf = pd.to_numeric(frame.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    l1c_logit = np.log(np.clip((l1c_score + 1.0) * 0.5, 1e-4, 1.0 - 1e-4) / np.clip((1.0 - l1c_score) * 0.5, 1e-4, 1.0 - 1e-4))

    triple_logit = np.log(triple_dir / (1.0 - triple_dir))
    signed_logit = np.log(signed_dir / (1.0 - signed_dir))
    raw_logit = 0.75 * triple_logit + 0.25 * signed_logit + 1.20 * l1c_conf * l1c_logit + 0.55 * trend_bias
    uncertainty = np.clip(0.65 * range_mass + 0.35 * entropy_norm, 0.0, 1.0)
    shrunk_logit = raw_logit * np.clip(1.0 - 0.65 * uncertainty, 0.20, 1.0)
    dir_p = _sigmoid(shrunk_logit)
    return np.clip(dir_p, 1e-4, 1.0 - 1e-4).astype(np.float32)


def _l2_fit_direction_fusion_model(
    fusion_frame: pd.DataFrame,
    y_dir: np.ndarray,
    *,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
) -> dict[str, Any]:
    X_df = fusion_frame.copy()
    feature_cols = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64, copy=False)
    y = np.asarray(y_dir, dtype=np.int64).ravel()
    fit = np.asarray(fit_mask, dtype=bool).ravel() & (y >= 0) & np.all(np.isfinite(X), axis=1)
    tune = np.asarray(tune_mask, dtype=bool).ravel() & (y >= 0) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(fit)) < max(80, len(feature_cols) + 5) or int(np.sum(tune)) < 40:
        return {
            "feature_cols": feature_cols,
            "feature_mean": np.zeros(len(feature_cols), dtype=np.float32).tolist(),
            "feature_scale": np.ones(len(feature_cols), dtype=np.float32).tolist(),
            "coef": np.zeros(len(feature_cols), dtype=np.float32).tolist(),
            "intercept": 0.0,
            "alpha": 1.0,
            "temperature": 1.0,
            "center": 0.5,
            "fit_rows": int(np.sum(fit)),
            "tune_rows": int(np.sum(tune)),
            "score": 0.0,
        }
    mu = X[fit].mean(axis=0)
    sigma = X[fit].std(axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
    Xn_fit = (X[fit] - mu) / sigma
    Xn_tune = (X[tune] - mu) / sigma
    y_fit = np.where(y[fit] == 1, 1.0, -1.0)
    y_tune = np.asarray(y[tune], dtype=np.int64)
    w_fit = _l2_direction_sample_weights(y, fusion_frame=fusion_frame)[fit].astype(np.float64)

    alpha_candidates = _env_float_candidates(
        "L2_DIR_FUSION_ALPHA_GRID", [10.0, 20.0, 30.0, 40.0, 50.0, 80.0], lo=1e-4, hi=1e4
    )
    temp_candidates = _env_float_candidates("L2_DIR_FUSION_TEMP_GRID", [0.35, 0.5, 0.75, 1.0, 1.5], lo=0.05, hi=5.0)
    center_candidates = _env_float_candidates("L2_DIR_FUSION_CENTER_GRID", [0.40, 0.45, 0.50, 0.55, 0.60], lo=0.05, hi=0.95)

    Xw = Xn_fit * np.sqrt(w_fit)[:, None]
    yw = y_fit * np.sqrt(w_fit)
    best_score = -np.inf
    best_coef = np.zeros(Xn_fit.shape[1], dtype=np.float64)
    best_alpha = float(alpha_candidates[0])
    best_temp = 1.0
    best_center = 0.5
    best_intercept = 0.0
    for alpha in alpha_candidates:
        gram = Xw.T @ Xw
        rhs = Xw.T @ yw
        coef = np.linalg.pinv(gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)) @ rhs
        raw_tune = Xn_tune @ coef
        for temp in temp_candidates:
            base_p = _l2_direction_margin_to_prob(raw_tune, temperature=float(temp))
            for center in center_candidates:
                p = _l2_recenter_direction_prob(base_p, center=float(center))
                pred = (p >= 0.5).astype(np.int64)
                f1m = float(f1_score(y_tune, pred, average="macro", zero_division=0))
                long_recall = float(np.mean(pred[y_tune == 1] == 1)) if np.any(y_tune == 1) else 0.0
                short_recall = float(np.mean(pred[y_tune == 0] == 0)) if np.any(y_tune == 0) else 0.0
                score = f1m + 0.10 * short_recall + 0.03 * long_recall
                if score > best_score:
                    best_score = score
                    best_coef = coef
                    best_alpha = float(alpha)
                    best_temp = float(temp)
                    best_center = float(center)
                    best_intercept = 0.0
    model = {
        "feature_cols": feature_cols,
        "feature_mean": mu.astype(np.float32).tolist(),
        "feature_scale": sigma.astype(np.float32).tolist(),
        "coef": best_coef.astype(np.float32).tolist(),
        "intercept": float(best_intercept),
        "alpha": best_alpha,
        "temperature": best_temp,
        "center": best_center,
        "fit_rows": int(np.sum(fit)),
        "tune_rows": int(np.sum(tune)),
        "score": float(best_score),
    }
    top_idx = np.argsort(np.abs(best_coef))[::-1][:6]
    top_terms = ", ".join(f"{feature_cols[i]}={best_coef[i]:+.3f}" for i in top_idx if abs(best_coef[i]) > 1e-6)
    print(
        f"  [L2] direction fusion model: alpha={best_alpha:.4f}  temp={best_temp:.3f}  center={best_center:.3f}  "
        f"fit_rows={int(np.sum(fit))}  tune_rows={int(np.sum(tune))}  score={best_score:.4f}  top={top_terms or 'all ~0'}",
        flush=True,
    )
    return model


def _l2_predict_direction_fusion_model(
    fusion_frame: pd.DataFrame,
    model: dict[str, Any] | None,
    *,
    fallback_frame: pd.DataFrame,
    triple_probs: np.ndarray,
    signed_edge_pred: np.ndarray,
    signed_edge_temperature: float | np.ndarray = 0.05,
) -> np.ndarray:
    cfg = dict(model or {})
    cols = list(cfg.get("feature_cols", []))
    if not cols:
        return _l2_hierarchical_direction_prob(
            fallback_frame,
            triple_probs,
            signed_edge_pred,
            signed_edge_temperature=signed_edge_temperature,
        )
    X = fusion_frame.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=np.float64, copy=False)
    mu = np.asarray(cfg.get("feature_mean", []), dtype=np.float64)
    sigma = np.asarray(cfg.get("feature_scale", []), dtype=np.float64)
    coef = np.asarray(cfg.get("coef", []), dtype=np.float64)
    intercept = float(cfg.get("intercept", 0.0))
    temp = float(cfg.get("temperature", 1.0))
    center = float(cfg.get("center", 0.5))
    if mu.size != X.shape[1] or sigma.size != X.shape[1] or coef.size != X.shape[1]:
        return _l2_hierarchical_direction_prob(
            fallback_frame,
            triple_probs,
            signed_edge_pred,
            signed_edge_temperature=signed_edge_temperature,
        )
    Xn = (X - mu) / np.where(np.abs(sigma) > 1e-6, sigma, 1.0)
    raw = intercept + Xn @ coef
    base_p = _l2_direction_margin_to_prob(raw, temperature=max(temp, 1e-3))
    dir_p = _l2_recenter_direction_prob(base_p, center=center)
    return np.clip(dir_p, 1e-4, 1.0 - 1e-4).astype(np.float32)


def _l2_compose_live_probs_from_gate_dir_edge(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    fused_edge_pred: np.ndarray,
    *,
    trade_score_threshold: float | np.ndarray,
    trade_score_temperature: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    direction = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 1e-4, 1.0 - 1e-4)
    fused_edge = np.asarray(fused_edge_pred, dtype=np.float64).ravel()
    thr = np.asarray(trade_score_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(gate), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), gate.shape)
    temp = np.asarray(trade_score_temperature, dtype=np.float64)
    if temp.ndim == 0:
        temp = np.full(len(gate), float(temp), dtype=np.float64)
    else:
        temp = np.broadcast_to(temp.reshape(-1), gate.shape)
    temp = np.maximum(temp, 1e-4)
    trade_score = _l2_trade_score(gate, fused_edge)
    active_p = _sigmoid((trade_score - thr) / temp)
    decision_probs = _l2_compose_probs_from_gate_dir(active_p, direction)
    return trade_score.astype(np.float32), active_p.astype(np.float32), decision_probs


def _l2_hard_decode_prob_aligned_outputs(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    decision_probs: np.ndarray,
    *,
    trade_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    hard_cls = _l2_hard_decision_from_gate_dir(gate_p, dir_p, trade_threshold)
    prob_mat = np.asarray(decision_probs, dtype=np.float32)
    confidence = prob_mat[np.arange(len(prob_mat)), hard_cls].astype(np.float32)
    return hard_cls.astype(np.int64), confidence


def _quantile_rescale_01(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[fit]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _l2_positive_head_target_prep(
    y_raw: np.ndarray,
    *,
    head_name: str,
    clip_max: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    head = str(head_name).strip().upper()
    y = np.clip(np.asarray(y_raw, dtype=np.float32), 0.0, float(clip_max))
    transform = (
        os.environ.get(f"L2_{head}_TARGET_TRANSFORM", os.environ.get("L2_POSITIVE_HEAD_TARGET_TRANSFORM", "log1p"))
        .strip()
        .lower()
        or "log1p"
    )
    if transform not in {"none", "log1p"}:
        transform = "log1p"
    objective = (
        os.environ.get(f"L2_{head}_OBJECTIVE", os.environ.get("L2_POSITIVE_HEAD_OBJECTIVE", "huber"))
        .strip()
        .lower()
        or "huber"
    )
    if objective not in {"regression", "huber", "fair"}:
        objective = "huber"
    metric_default = "l1" if objective in {"huber", "fair"} else "l2"
    metric = (
        os.environ.get(f"L2_{head}_METRIC", os.environ.get("L2_POSITIVE_HEAD_METRIC", metric_default))
        .strip()
        .lower()
        or metric_default
    )
    y_fit = np.log1p(y).astype(np.float32) if transform == "log1p" else y.astype(np.float32, copy=False)
    prep: dict[str, float | str] = {
        "clip_max": float(clip_max),
        "target_transform": str(transform),
        "objective": str(objective),
        "metric": str(metric),
    }
    if objective == "huber":
        prep["alpha"] = _env_float_clipped(f"L2_{head}_HUBER_ALPHA", 0.90, lo=0.50, hi=0.99)
    elif objective == "fair":
        prep["fair_c"] = _env_float_clipped(f"L2_{head}_FAIR_C", 1.0, lo=0.10, hi=10.0)
    return y_fit, prep


def _l2_positive_head_lgb_params(base_params: dict[str, Any], prep: dict[str, float | str]) -> dict[str, Any]:
    params = {
        **base_params,
        "objective": str(prep.get("objective", "regression")),
        "metric": str(prep.get("metric", "l2")),
    }
    if params["objective"] == "huber":
        params["alpha"] = float(prep.get("alpha", 0.90))
    elif params["objective"] == "fair":
        params["fair_c"] = float(prep.get("fair_c", 1.0))
    return params


def _l2_positive_head_predict(model: lgb.Booster, X: np.ndarray, prep: dict[str, Any] | None, *, clip_max: float) -> np.ndarray:
    cfg = dict(prep or {})
    transform = str(cfg.get("target_transform", "none")).strip().lower() or "none"
    cap = float(cfg.get("clip_max", clip_max))
    pred = model.predict(X).astype(np.float64)
    if transform == "log1p":
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, cap)
    return pred.astype(np.float32)


def _residual_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_pa_state_features(df)
    out = pd.DataFrame(index=df.index)
    for col in [
        "pa_ctx_structure_veto",
        "pa_ctx_premise_break_long",
        "pa_ctx_premise_break_short",
        "pa_ctx_range_pressure",
        *PA_STATE_FEATURES,
        "bo_wick_imbalance",
        "bo_or_dist",
    ]:
        out[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    out = pd.concat([out, _session_context(df)], axis=1)
    return out


def _l2_state_input_cols(merged: pd.DataFrame) -> list[str]:
    cols = list(L1A_REGIME_COLS)
    cols.extend(
        [
            "l1a_transition_risk",
            "l1a_vol_forecast",
            "l1a_vol_trend",
            "l1a_time_in_regime",
            "l1a_state_persistence",
            "l1a_is_warm",
        ]
    )
    cols.extend([c for c in merged.columns if c.startswith("l1a_market_embed_")])
    return [c for c in cols if c in merged.columns]


def _l2_condition_input_cols(merged: pd.DataFrame) -> list[str]:
    allow = [
        "l1b_edge_pred",
        "l1b_dq_pred",
        "l1b_novelty_score",
        "l1b_regime_change_score",
        "l1b_breakout_quality",
        "l1b_mean_reversion_setup",
        "l1b_trend_strength",
        "l1b_range_reversal_setup",
        "l1b_failed_breakout_setup",
        "l1b_setup_alignment",
        "l1b_follow_through_score",
        "l1b_liquidity_score",
    ]
    allow.extend([c for c in merged.columns if c.startswith("l1b_cluster_prob_")])
    return [c for c in allow if c in merged.columns]


def _l2_direction_input_cols(merged: pd.DataFrame) -> list[str]:
    allow = [
        "l1c_direction_prob",
        "l1c_direction_score",
        "l1c_confidence",
        "l1c_direction_strength",
        "l1c_is_warm",
    ]
    return [c for c in allow if c in merged.columns]


def _l2_residual_input_cols(residual: pd.DataFrame) -> list[str]:
    return [c for c in residual.columns if c not in {"symbol", "time_key"}]


def _derived_l2_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=merged.index)
    n_m = len(merged)
    regime_cols = [c for c in L1A_REGIME_COLS if c in merged.columns]
    if regime_cols:
        regime_probs = merged[regime_cols].to_numpy(dtype=np.float64, copy=False)
        regime_probs = np.nan_to_num(regime_probs, nan=0.0)
        regime_sums = np.maximum(regime_probs.sum(axis=1, keepdims=True), 1e-12)
        regime_probs = regime_probs / regime_sums
        entropy = -np.sum(np.clip(regime_probs, 1e-12, 1.0) * np.log(np.clip(regime_probs, 1e-12, 1.0)), axis=1)
        sorted_probs = np.sort(regime_probs, axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else np.zeros(len(sorted_probs), dtype=np.float64)
        out["l1a_regime_entropy"] = entropy.astype(np.float32)
        out["l1a_regime_top2_gap"] = (top1 - top2).astype(np.float32)
    else:
        out["l1a_regime_entropy"] = np.zeros(len(merged), dtype=np.float32)
        out["l1a_regime_top2_gap"] = np.zeros(len(merged), dtype=np.float32)

    cluster_cols = [c for c in merged.columns if c.startswith("l1b_cluster_prob_")]
    if cluster_cols:
        cluster_probs = merged[cluster_cols].to_numpy(dtype=np.float64, copy=False)
        cluster_probs = np.nan_to_num(cluster_probs, nan=0.0)
        cluster_sorted = np.sort(cluster_probs, axis=1)
        cluster_top1 = cluster_sorted[:, -1]
        cluster_top2 = cluster_sorted[:, -2] if cluster_sorted.shape[1] > 1 else np.zeros(len(cluster_sorted), dtype=np.float64)
        out["l1b_cluster_top1"] = cluster_top1.astype(np.float32)
        out["l1b_cluster_top2_gap"] = (cluster_top1 - cluster_top2).astype(np.float32)
    else:
        out["l1b_cluster_top1"] = np.zeros(len(merged), dtype=np.float32)
        out["l1b_cluster_top2_gap"] = np.zeros(len(merged), dtype=np.float32)

    novelty = pd.to_numeric(merged.get("l1b_novelty_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    regime_change = pd.to_numeric(merged.get("l1b_regime_change_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    vol = pd.to_numeric(merged.get("l1a_vol_forecast", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    persistence = pd.to_numeric(merged.get("l1a_state_persistence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    out["l1b_novelty_x_vol"] = (novelty * vol).astype(np.float32)
    out["l1b_regime_change_x_entropy"] = (regime_change * out["l1a_regime_entropy"].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
    out["l1b_unsup_pressure"] = (out["l1b_cluster_top2_gap"].to_numpy(dtype=np.float32, copy=False) - novelty).astype(np.float32)
    if "l1c_direction_score" in merged.columns:
        l1c_d = pd.to_numeric(
            merged.get("l1c_direction_score", 0.0),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        l1c_c = pd.to_numeric(merged.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        l1c_s = pd.to_numeric(merged.get("l1c_direction_strength", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        out["l1c_weighted_dir"] = (l1c_d * l1c_c).astype(np.float32)
        out["l1c_dir_x_vol"] = (l1c_d * vol).astype(np.float32)
        out["l1c_dir_x_state_persistence"] = (l1c_d * persistence).astype(np.float32)
        out["l1c_strength_x_conf"] = (l1c_s * l1c_c).astype(np.float32)
    else:
        out["l1c_weighted_dir"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_dir_x_vol"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_dir_x_state_persistence"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_strength_x_conf"] = np.zeros(n_m, dtype=np.float32)

    l1b_edge_f = pd.to_numeric(merged.get("l1b_edge_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1b_dq_f = pd.to_numeric(merged.get("l1b_dq_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    vol_f = pd.to_numeric(merged.get("l1a_vol_forecast", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    out["l2_vol_adjusted_l1b_edge"] = (l1b_edge_f / np.maximum(vol_f, 1e-4)).astype(np.float32)
    l1c_dir_f = pd.to_numeric(merged.get("l1c_direction_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1c_conf_f = pd.to_numeric(merged.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    out["l2_dir_x_edge_opportunity"] = (l1c_dir_f * l1b_edge_f).astype(np.float32)
    out["l2_dir_conf_x_edge_mag"] = (l1c_conf_f * np.abs(l1b_edge_f)).astype(np.float32)
    out["l2_dq_x_edge"] = (l1b_dq_f * l1b_edge_f).astype(np.float32)

    out["l2_signal_strength_mean_abs"] = ((np.abs(l1c_dir_f) + np.abs(l1b_edge_f)) / 2.0).astype(np.float32)
    sig_stack = np.column_stack([l1c_dir_f.astype(np.float64), l1b_edge_f.astype(np.float64), l1b_dq_f.astype(np.float64)])
    out["l2_signal_spread_var"] = np.var(sig_stack, axis=1).astype(np.float32)

    if len(regime_cols) >= NUM_REGIME_CLASSES:
        rp = merged[regime_cols].to_numpy(dtype=np.float64, copy=False)
        rp = np.nan_to_num(rp, nan=0.0)
        rp = rp / np.maximum(rp.sum(axis=1, keepdims=True), 1e-12)
        bull_mass = (rp[:, 0] + rp[:, 1]).astype(np.float32)
        range_mass = (rp[:, 4] + rp[:, 5]).astype(np.float32)
        out["l2_bull_mass_x_l1c_dir"] = (bull_mass * l1c_dir_f).astype(np.float32)
        out["l2_range_mass_x_l1c_dir"] = (range_mass * l1c_dir_f).astype(np.float32)
    else:
        out["l2_bull_mass_x_l1c_dir"] = np.full(n_m, np.nan, dtype=np.float64)
        out["l2_range_mass_x_l1c_dir"] = np.full(n_m, np.nan, dtype=np.float64)
    return out


_L2_REGIME_INTERACTION_NAN_COLS = frozenset({"l2_bull_mass_x_l1c_dir", "l2_range_mass_x_l1c_dir"})


def _build_l2_frame(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
    l1c_outputs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    merged = (
        df[["symbol", "time_key"]]
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l1b_outputs, on=["symbol", "time_key"], how="left")
    )
    if l1c_outputs is not None:
        merged = merged.merge(l1c_outputs, on=["symbol", "time_key"], how="left")
    residual = _residual_feature_frame(df)
    derived = _derived_l2_feature_frame(merged)
    merged = pd.concat([merged.reset_index(drop=True), residual.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    group_map = {
        "state": _l2_state_input_cols(merged),
        "condition": _l2_condition_input_cols(merged),
        "direction": _l2_direction_input_cols(merged),
        "residual": _l2_residual_input_cols(residual),
        "derived": [c for c in derived.columns if c in merged.columns],
    }
    feature_cols = list(dict.fromkeys([c for cols in group_map.values() for c in cols if c not in {"symbol", "time_key"}]))
    for c in feature_cols:
        s = pd.to_numeric(merged[c], errors="coerce")
        if c in _L2_REGIME_INTERACTION_NAN_COLS:
            merged[c] = s.astype(np.float32)
        else:
            merged[c] = s.fillna(0.0).astype(np.float32)
    for group_name, cols in group_map.items():
        print(f"  [L2] input group {group_name}: {len(cols)} cols", flush=True)
    return merged, feature_cols


def train_l2_trade_decision(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
    l1c_outputs: pd.DataFrame | None = None,
) -> L2TrainingBundle:
    frame, feature_cols = _build_l2_frame(df, l1a_outputs, l1b_outputs, l1c_outputs)
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(df["time_key"])
    l2_val_start = l2_val_start_time()

    train_mask = splits.l2_train_mask
    val_mask = splits.l2_val_mask
    test_mask = splits.test_mask
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("L2: calibration split is empty for train/val.")
    tune_frac = float(os.environ.get("L2_TUNE_FRAC_WITHIN_VAL", "0.5"))
    val_tune_mask, val_report_mask = _split_mask_for_tuning_and_report(
        df["time_key"], val_mask, tune_frac=tune_frac, min_rows_each=50
    )
    if not val_tune_mask.any() or not val_report_mask.any():
        raise RuntimeError("L2: failed to create non-empty tuning/report masks inside l2_val.")

    ts_all = pd.to_datetime(df["time_key"])
    for split_name, sm in (
        ("train", train_mask),
        ("val", val_mask),
        ("holdout", test_mask),
    ):
        sm = np.asarray(sm, dtype=bool)
        if sm.any():
            print(
                f"  [L2] {split_name} period: {ts_all[sm].min()} .. {ts_all[sm].max()}  rows={int(sm.sum()):,}",
                flush=True,
            )
    print(
        f"  [L2] policy_state_granularity={os.environ.get('L2_POLICY_STATE_GRANULARITY', 'coarse')!r} "
        f"(set L2_POLICY_STATE_GRANULARITY=full for legacy regime×vol×PA keys)",
        flush=True,
    )

    min_std = float(os.environ.get("L2_MIN_FEATURE_STD", "1e-4"))
    skip_hard = os.environ.get("L2_SKIP_FEATURE_HARD_DROP", "").strip().lower() in {"1", "true", "yes"}
    hard_drop = frozenset() if skip_hard else set(L2_FEATURE_HARD_DROP_DEFAULT)
    if not skip_hard:
        _extra = os.environ.get("L2_EXTRA_HARD_DROP", "").strip()
        if _extra:
            hard_drop |= {s.strip() for s in _extra.split(",") if s.strip()}
    hard_drop = frozenset(hard_drop)
    feature_cols, l2_dropped_features = _l2_select_features_for_training(
        X, feature_cols, train_mask, min_std=min_std, hard_drop=hard_drop
    )
    if l2_dropped_features:
        print(
            f"  [L2] feature selection: dropped {len(l2_dropped_features)} cols "
            f"(hard_drop={not skip_hard}, min_train_std={min_std:g})",
            flush=True,
        )
        for line in l2_dropped_features[:35]:
            print(f"       {line}", flush=True)
        if len(l2_dropped_features) > 35:
            print(f"       ... {len(l2_dropped_features) - 35} more", flush=True)
    if not feature_cols:
        raise RuntimeError(
            "L2: all features removed by selection; set L2_MIN_FEATURE_STD lower or "
            "L2_SKIP_FEATURE_HARD_DROP=1."
        )
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)

    use_l1b_latent_feats = os.environ.get("L2_USE_L1B_LATENT", "0").strip().lower() in {"1", "true", "yes"}
    if not use_l1b_latent_feats:
        latent_cols = [c for c in feature_cols if c.startswith("l1b_latent_")]
        if latent_cols:
            feature_cols = [c for c in feature_cols if c not in set(latent_cols)]
            X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
            print(
                f"  [L2] excluding {len(latent_cols)} l1b_latent_* from L2 features "
                f"(set L2_USE_L1B_LATENT=1 to keep)",
                flush=True,
            )

    l1b_train_dropout_p = float(os.environ.get("L2_L1B_TRAIN_FEATURE_DROPOUT", "0.3"))
    l1b_do_seed = int(os.environ.get("L2_L1B_DROPOUT_SEED", "42"))
    X_train_fit = _l2_apply_l1b_feature_dropout_train_only(
        X, train_mask, feature_cols, l1b_train_dropout_p, rng=np.random.default_rng(l1b_do_seed)
    )
    if l1b_train_dropout_p > 0.0:
        n_l1b = sum(1 for c in feature_cols if c.startswith("l1b_"))
        print(
            f"  [L2] L1b train feature dropout: p={l1b_train_dropout_p} "
            f"({n_l1b} l1b_* cols; train rows only; val/report unchanged; seed={l1b_do_seed})",
            flush=True,
        )

    edge = _decision_edge_atr_array(df)
    mfe, mae = _mfe_mae_atr_arrays(df)
    tau_global, policy_vol_quantiles, decision_tau_by_state, tau_row = _conditional_tau_from_state(frame, edge, train_mask)
    y_decision = np.full(len(df), 1, dtype=np.int64)
    y_decision[edge > tau_row] = 0
    y_decision[edge < -tau_row] = 2
    _, y_dir_stage = _l2_build_two_stage_labels(y_decision)
    size_mae_floor = _env_float_clipped("L2_SIZE_MAE_FLOOR", 0.10, lo=1e-4, hi=10.0)
    size_rr_cap = _env_float_clipped("L2_SIZE_RR_CAP", 4.0, lo=0.1, hi=20.0)
    size_edge_cap = _env_float_clipped("L2_SIZE_EDGE_CAP", 1.5, lo=0.1, hi=20.0)
    size_mae_decay = _env_float_clipped("L2_SIZE_MAE_DECAY", 0.35, lo=0.0, hi=5.0)
    size_mae_clip = _env_float_clipped("L2_SIZE_MAE_CLIP", 4.0, lo=0.1, hi=20.0)
    rr = np.clip(mfe / np.maximum(mae, size_mae_floor), 0.0, size_rr_cap)
    size_raw = np.clip(np.abs(edge), 0.0, size_edge_cap) * (rr / (1.0 + rr)) * np.exp(
        -size_mae_decay * np.clip(mae, 0.0, size_mae_clip)
    )
    y_trade, y_long, y_short, trade_weights, long_weights, short_weights, triple_gate_label_stats = _l2_triple_gate_targets(
        edge,
        train_mask,
        frame=frame,
    )
    finite_train_edge = np.isfinite(edge) & train_mask
    abs_train_edge = np.abs(edge[finite_train_edge])
    side_clip = float(np.quantile(abs_train_edge, 0.995)) if abs_train_edge.size else 2.0
    side_clip = float(np.clip(side_clip, 0.10, 5.0))
    y_long_score_raw = np.clip(edge, 0.0, side_clip).astype(np.float32)
    y_short_score_raw = np.clip(-edge, 0.0, side_clip).astype(np.float32)
    y_long_score_fit, long_head_prep = _l2_positive_head_target_prep(y_long_score_raw, head_name="long", clip_max=side_clip)
    y_short_score_fit, short_head_prep = _l2_positive_head_target_prep(y_short_score_raw, head_name="short", clip_max=side_clip)
    y_signed_edge_fit, signed_edge_prep = _l2_signed_edge_target_prep(edge, train_mask)
    signed_edge_weights, signed_edge_label_stats = _l2_signed_edge_sample_weights(edge, train_mask)
    active_train = train_mask & (y_trade == 1)
    active_val = val_mask & (y_trade == 1)
    y_size = _quantile_rescale_01(size_raw, fit_mask=active_train)
    y_size[y_trade == 0] = 0.0
    y_mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    y_mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    y_mfe_fit, mfe_head_prep = _l2_positive_head_target_prep(y_mfe, head_name="mfe", clip_max=5.0)
    y_mae_fit, mae_head_prep = _l2_positive_head_target_prep(y_mae, head_name="mae", clip_max=4.0)

    log_layer_banner("[L2] Trade decision (LGBM)")
    log_time_key_split(
        "L2",
        df["time_key"],
        train_mask,
        val_mask,
        train_label=f"l2_train [{TRAIN_END}, {str(l2_val_start)})",
        val_label=f"l2_val [{str(l2_val_start)}, {CAL_END})",
        extra_note=(
            f"Strict time split inside cal window: train in [{TRAIN_END}, {str(l2_val_start)}), "
            f"val in [{str(l2_val_start)}, {CAL_END})."
        ),
    )
    log_time_key_split(
        "L2(threshold/calibration)",
        df["time_key"],
        val_tune_mask,
        val_report_mask,
        train_label="val_tune (threshold/calibration)",
        val_label="val_report (metrics)",
        extra_note="L2 thresholds and probability calibration are fit on val_tune; headline validation metrics are reported on val_report.",
    )
    log_numpy_x_stats("L2", X[train_mask], label="X[l2_train]")
    l1a_cols = [c for c in feature_cols if c.startswith("l1a_")]
    l1b_cols = [c for c in feature_cols if c.startswith("l1b_")]
    l1c_cols = [c for c in feature_cols if c.startswith("l1c_")]
    res_cols = [c for c in feature_cols if c not in l1a_cols and c not in l1b_cols and c not in l1c_cols]
    print(
        f"  [L2] feature_cols total={len(feature_cols)} (expect ~51+)  "
        f"l1a_*={len(l1a_cols)}  l1b_*={len(l1b_cols)}  l1c_*={len(l1c_cols)}  residual/other={len(res_cols)}",
        flush=True,
    )
    print(f"  [L2] residual columns (n={len(res_cols)}): {res_cols}", flush=True)
    print(
        f"  [L2] upstream artifact refs: L1a={artifact_path(L1A_MODEL_FILE)}  "
        f"L1b meta={artifact_path(L1B_META_FILE)}  L1c={artifact_path(L1C_MODEL_FILE)}",
        flush=True,
    )
    print(
        "  [L2] note: l1a_*/l1b_* come from supplied upstream outputs; preferred pipeline path uses frozen-artifact inference caches.",
        flush=True,
    )
    print(f"  [L2] will write: {artifact_path(L2_META_FILE)} | {artifact_path(L2_OUTPUT_CACHE_FILE)}", flush=True)
    log_label_baseline("l2_size", y_size[active_train], task="reg")
    log_label_baseline("l2_mfe", y_mfe[active_train], task="reg")
    log_label_baseline("l2_mae", y_mae[active_train], task="reg")
    print(
        f"  [L2] aux target prep: long(transform={long_head_prep['target_transform']}, objective={long_head_prep['objective']}, metric={long_head_prep['metric']})  "
        f"short(transform={short_head_prep['target_transform']}, objective={short_head_prep['objective']}, metric={short_head_prep['metric']})  "
        f"signed_edge(transform={signed_edge_prep['target_transform']}, objective={signed_edge_prep['objective']}, metric={signed_edge_prep['metric']})  "
        f"mfe(transform={mfe_head_prep['target_transform']}, objective={mfe_head_prep['objective']}, metric={mfe_head_prep['metric']})  "
        f"mae(transform={mae_head_prep['target_transform']}, objective={mae_head_prep['objective']}, metric={mae_head_prep['metric']})",
        flush=True,
    )
    print(
        f"  [L2] triple-gate labels: trade_floor={triple_gate_label_stats['trade_floor']:.4f}  "
        f"side_floor={triple_gate_label_stats['side_floor']:.4f}  "
        f"train_trade_rate={triple_gate_label_stats['train_trade_rate']:.3f}  "
        f"train_long_rate={triple_gate_label_stats['train_long_rate']:.3f}  "
        f"train_short_rate={triple_gate_label_stats['train_short_rate']:.3f}",
        flush=True,
    )

    rounds = _l2_boost_rounds()
    # Gate: default 120 — not tied to FAST_TRAIN_MODE (set L2_GATE_EARLY_STOPPING_ROUNDS to override).
    gate_es_rounds = _l2_early_stopping_rounds_from_env("L2_GATE_EARLY_STOPPING_ROUNDS", 120)
    aux_es_fallback = 40 if FAST_TRAIN_MODE else 120
    aux_es_base = _l2_early_stopping_rounds_from_env("L2_EARLY_STOPPING_ROUNDS", aux_es_fallback)
    direction_es_rounds = _l2_early_stopping_rounds_from_env("L2_DIRECTION_EARLY_STOPPING_ROUNDS", aux_es_base)
    side_es_rounds = _l2_early_stopping_rounds_from_env("L2_SIDE_EARLY_STOPPING_ROUNDS", direction_es_rounds)
    size_es_rounds = _l2_early_stopping_rounds_from_env("L2_SIZE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mfe_es_rounds = _l2_early_stopping_rounds_from_env("L2_MFE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mae_es_rounds = _l2_early_stopping_rounds_from_env("L2_MAE_EARLY_STOPPING_ROUNDS", aux_es_base)
    log_label_baseline("l2_trade_gate", y_trade[train_mask], task="cls")
    log_label_baseline("l2_long_score", y_long_score_raw[train_mask], task="reg")
    log_label_baseline("l2_short_score", y_short_score_raw[train_mask], task="reg")
    log_label_baseline("l2_signed_edge", edge[train_mask], task="reg")
    pr_trade = float(np.mean(y_trade[train_mask]))
    gate_cfg = _l2_model_lgb_params("gate")
    direction_cfg = _l2_model_lgb_params("direction")
    reg_cfg = _l2_model_lgb_params("reg")
    trade_gate_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": gate_cfg["learning_rate"],
        "num_leaves": gate_cfg["num_leaves"],
        "max_depth": gate_cfg["max_depth"],
        "feature_fraction": gate_cfg["feature_fraction"],
        "bagging_fraction": gate_cfg["bagging_fraction"],
        "bagging_freq": gate_cfg["bagging_freq"],
        "min_child_samples": gate_cfg["min_child_samples"],
        "lambda_l1": gate_cfg["lambda_l1"],
        "lambda_l2": gate_cfg["lambda_l2"],
        "verbosity": -1,
        "seed": gate_cfg["seed"],
        "n_jobs": _lgbm_n_jobs(),
        "is_unbalance": True,
    }
    side_reg_base_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": direction_cfg["learning_rate"],
        "num_leaves": direction_cfg["num_leaves"],
        "max_depth": direction_cfg["max_depth"],
        "feature_fraction": direction_cfg["feature_fraction"],
        "bagging_fraction": direction_cfg["bagging_fraction"],
        "bagging_freq": direction_cfg["bagging_freq"],
        "min_child_samples": direction_cfg["min_child_samples"],
        "lambda_l1": direction_cfg["lambda_l1"],
        "lambda_l2": direction_cfg["lambda_l2"],
        "verbosity": -1,
        "seed": direction_cfg["seed"],
        "n_jobs": _lgbm_n_jobs(),
    }
    long_score_params = _l2_positive_head_lgb_params(side_reg_base_params, long_head_prep)
    short_score_params = _l2_positive_head_lgb_params(side_reg_base_params, short_head_prep)
    signed_edge_params = {
        **side_reg_base_params,
        "objective": str(signed_edge_prep.get("objective", "huber")),
        "metric": str(signed_edge_prep.get("metric", "l1")),
        "alpha": float(signed_edge_prep.get("alpha", 0.90)),
    }
    print(
        f"  [L2] gate: pos_rate={pr_trade:.3f}  is_unbalance=True  lr={gate_cfg['learning_rate']}  "
        f"num_leaves={gate_cfg['num_leaves']}  max_depth={gate_cfg['max_depth']}  min_child_samples={gate_cfg['min_child_samples']}  "
        f"bagging={gate_cfg['bagging_fraction']}/{gate_cfg['bagging_freq']}  early_stopping_rounds={gate_es_rounds}  "
        f"early_stop_metric=auc (first)",
        flush=True,
    )
    print(
        f"  [L2] early_stopping_rounds: side_reg={side_es_rounds}  size={size_es_rounds}  "
        f"mfe={mfe_es_rounds}  mae={mae_es_rounds}  (aux base={aux_es_base}; "
        f"override via L2_EARLY_STOPPING_ROUNDS / L2_*_EARLY_STOPPING_ROUNDS)",
        flush=True,
    )
    reg_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": reg_cfg["learning_rate"],
        "num_leaves": reg_cfg["num_leaves"],
        "max_depth": reg_cfg["max_depth"],
        "feature_fraction": reg_cfg["feature_fraction"],
        "bagging_fraction": reg_cfg["bagging_fraction"],
        "bagging_freq": reg_cfg["bagging_freq"],
        "min_child_samples": reg_cfg["min_child_samples"],
        "lambda_l1": reg_cfg["lambda_l1"],
        "lambda_l2": reg_cfg["lambda_l2"],
        "verbosity": -1,
        "seed": reg_cfg["seed"],
        "n_jobs": _lgbm_n_jobs(),
    }
    mfe_params = _l2_positive_head_lgb_params(reg_params, mfe_head_prep)
    mae_params = _l2_positive_head_lgb_params(reg_params, mae_head_prep)
    gate_nt_w = float(os.environ.get("L2_GATE_NO_TRADE_WEIGHT", "1.5"))
    gate_w_train = np.ones(int(np.sum(train_mask)), dtype=np.float64)
    gate_w_val = np.ones(int(np.sum(val_mask)), dtype=np.float64)
    if gate_nt_w != 1.0:
        tr_idx = np.flatnonzero(train_mask)
        va_idx = np.flatnonzero(val_mask)
        gate_w_train[y_trade[tr_idx] == 0] = gate_nt_w
        gate_w_val[y_trade[va_idx] == 0] = gate_nt_w
        print(f"  [L2] gate sample_weight: no_trade rows ×{gate_nt_w:.3f} (L2_GATE_NO_TRADE_WEIGHT)", flush=True)
    dtrain_gate = lgb.Dataset(
        X_train_fit[train_mask],
        label=y_trade[train_mask],
        weight=gate_w_train,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    dval_gate = lgb.Dataset(
        X[val_mask],
        label=y_trade[val_mask],
        weight=gate_w_val,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    l2_outer = tqdm(
        total=7,
        desc="[L2] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    long_gate_model: lgb.Booster
    short_gate_model: lgb.Booster
    signed_edge_model: lgb.Booster
    try:
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(gate_es_rounds, rounds, "[L2] gate", first_metric_only=True)
        try:
            gate_model = lgb.train(trade_gate_params, dtrain_gate, num_boost_round=rounds, valid_sets=[dval_gate], callbacks=cbs)
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("gate", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(side_es_rounds, rounds, "[L2] long_gate")
        try:
            long_gate_model = lgb.train(
                long_score_params,
                lgb.Dataset(
                    X_train_fit[train_mask],
                    label=y_long_score_fit[train_mask],
                    weight=long_weights[train_mask],
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[
                    lgb.Dataset(
                        X[val_mask],
                        label=y_long_score_fit[val_mask],
                        weight=long_weights[val_mask],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    )
                ],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("long_gate", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(side_es_rounds, rounds, "[L2] short_gate")
        try:
            short_gate_model = lgb.train(
                short_score_params,
                lgb.Dataset(
                    X_train_fit[train_mask],
                    label=y_short_score_fit[train_mask],
                    weight=short_weights[train_mask],
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[
                    lgb.Dataset(
                        X[val_mask],
                        label=y_short_score_fit[val_mask],
                        weight=short_weights[val_mask],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    )
                ],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("short_gate", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(direction_es_rounds, rounds, "[L2] signed_edge")
        try:
            signed_edge_model = lgb.train(
                signed_edge_params,
                lgb.Dataset(
                    X_train_fit[train_mask],
                    label=y_signed_edge_fit[train_mask],
                    weight=signed_edge_weights[train_mask],
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[
                    lgb.Dataset(
                        X[val_mask],
                        label=y_signed_edge_fit[val_mask],
                        weight=signed_edge_weights[val_mask],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    )
                ],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("signed_edge", refresh=False)
        l2_outer.update(1)

        if int(active_train.sum()) < 100 or int(active_val.sum()) < 25:
            raise RuntimeError(
                "L2: too few active rows for size/MFE/MAE heads after strict time split. "
                f"active_train={int(active_train.sum())}, active_val={int(active_val.sum())}"
            )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(size_es_rounds, rounds, "[L2] size")
        try:
            size_model = lgb.train(
                reg_params,
                lgb.Dataset(
                    X_train_fit[active_train], label=y_size[active_train], feature_name=feature_cols, free_raw_data=False
                ),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_size[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("size", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(mfe_es_rounds, rounds, "[L2] mfe")
        try:
            mfe_model = lgb.train(
                mfe_params,
                lgb.Dataset(
                    X_train_fit[active_train], label=y_mfe_fit[active_train], feature_name=feature_cols, free_raw_data=False
                ),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_mfe_fit[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("mfe", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(mae_es_rounds, rounds, "[L2] mae")
        try:
            mae_model = lgb.train(
                mae_params,
                lgb.Dataset(
                    X_train_fit[active_train], label=y_mae_fit[active_train], feature_name=feature_cols, free_raw_data=False
                ),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_mae_fit[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("mae", refresh=False)
        l2_outer.update(1)
    finally:
        l2_outer.close()

    gate_raw_all = gate_model.predict(X).astype(np.float64)
    long_raw_all = long_gate_model.predict(X).astype(np.float64)
    short_raw_all = short_gate_model.predict(X).astype(np.float64)
    gate_calibrator = _fit_binary_calibrator(y_trade[val_tune_mask], gate_raw_all[val_tune_mask])
    gate_calibrators = {
        "trade": gate_calibrator,
        "long": None,
        "short": None,
    }
    trade_p_all = _apply_binary_calibrator(gate_raw_all, gate_calibrator).astype(np.float32)
    long_p_all = _l2_positive_head_predict(long_gate_model, X, long_head_prep, clip_max=side_clip)
    short_p_all = _l2_positive_head_predict(short_gate_model, X, short_head_prep, clip_max=side_clip)
    signed_edge_all = _l2_signed_edge_predict(signed_edge_model, X, signed_edge_prep)
    size_pred = np.clip(size_model.predict(X).astype(np.float32), 0.0, 1.0)
    mfe_pred = _l2_positive_head_predict(mfe_model, X, mfe_head_prep, clip_max=5.0)
    mae_pred = _l2_positive_head_predict(mae_model, X, mae_head_prep, clip_max=4.0)
    state_keys_all = _l2_policy_state_keys(frame, vol_quantiles=policy_vol_quantiles)
    triple_policy, triple_conditional_policy_by_state = _l2_search_triple_gate_conditional_policy(
        state_keys_all[val_tune_mask],
        trade_p_all[val_tune_mask],
        long_p_all[val_tune_mask],
        short_p_all[val_tune_mask],
        edge[val_tune_mask],
    )
    triple_trade_threshold = float(triple_policy["trade_threshold"])
    _, triple_trade_threshold_arr_all, triple_short_bias_arr_all = _l2_triple_gate_policy_arrays(
        {
            "trade_threshold": triple_trade_threshold,
            "short_bias": float(triple_policy.get("short_bias", 1.0)),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": triple_conditional_policy_by_state,
        },
        frame,
    )
    trade_p_report = trade_p_all[val_report_mask]
    long_p_report = long_p_all[val_report_mask]
    short_p_report = short_p_all[val_report_mask]
    _log_l2_triple_gate_val_diagnostics(
        trade_p_report,
        long_p_report,
        short_p_report,
        y_trade[val_report_mask],
        y_long[val_report_mask],
        y_short[val_report_mask],
        y_decision[val_report_mask],
        edge[val_report_mask],
        float(np.mean(triple_trade_threshold_arr_all[val_report_mask])),
        float(np.mean(triple_short_bias_arr_all[val_report_mask])),
    )
    _, _, triple_decision_probs = _l2_compose_probs_from_triple_gate(
        trade_p_all,
        long_p_all,
        short_p_all,
        short_bias=triple_short_bias_arr_all,
    )
    triple_expected_edge_all = _l2_expected_edge_from_triple_gate(
        trade_p_all,
        long_p_all,
        short_p_all,
        size_pred,
        mfe_pred,
        mae_pred,
        trade_threshold=triple_trade_threshold_arr_all,
        short_bias=triple_short_bias_arr_all,
    )
    signed_policy, signed_conditional_policy_by_state = _l2_search_conditional_policy(
        state_keys_all[val_tune_mask],
        trade_p_all[val_tune_mask],
        signed_edge_all[val_tune_mask],
        edge[val_tune_mask],
    )
    _, signed_trade_threshold_arr_all, signed_trade_temp_arr_all, signed_edge_temp_arr_all = _l2_signed_edge_policy_arrays(
        {
            "trade_score_threshold": float(signed_policy["trade_score_threshold"]),
            "trade_score_temperature": float(signed_policy["trade_score_temperature"]),
            "edge_temperature": float(signed_policy["edge_temperature"]),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": signed_conditional_policy_by_state,
        },
        frame,
    )
    _, _, signed_decision_probs = _l2_compose_probs_from_gate_edge(
        trade_p_all,
        signed_edge_all,
        trade_score_threshold=signed_trade_threshold_arr_all,
        trade_score_temperature=signed_trade_temp_arr_all,
        edge_temperature=signed_edge_temp_arr_all,
    )
    _log_l2_signed_edge_val_diagnostics(
        trade_p_all[val_report_mask],
        signed_edge_all[val_report_mask],
        y_trade[val_report_mask],
        y_decision[val_report_mask],
        edge[val_report_mask],
        float(np.mean(signed_trade_threshold_arr_all[val_report_mask])),
        float(np.mean(signed_trade_temp_arr_all[val_report_mask])),
        float(np.mean(signed_edge_temp_arr_all[val_report_mask])),
        label="signed-edge branch",
    )
    fusion_frame = _l2_fusion_feature_frame(
        frame,
        trade_p_all,
        triple_decision_probs,
        triple_expected_edge_all,
        signed_decision_probs,
        signed_edge_all,
        size_pred,
        mfe_pred,
        mae_pred,
    )
    fusion_model = _l2_fit_edge_fusion_model(fusion_frame, edge, val_tune_mask)
    fused_edge_all = _l2_predict_edge_fusion_model(fusion_frame, fusion_model)
    direction_fusion_model = _l2_fit_direction_fusion_model(
        fusion_frame,
        y_dir_stage,
        fit_mask=train_mask & (y_dir_stage >= 0),
        tune_mask=val_tune_mask & (y_dir_stage >= 0),
    )
    ensemble_policy, ensemble_conditional_policy_by_state = _l2_search_conditional_policy(
        state_keys_all[val_tune_mask],
        trade_p_all[val_tune_mask],
        fused_edge_all[val_tune_mask],
        edge[val_tune_mask],
    )
    _, ensemble_trade_threshold_arr_all, ensemble_trade_temp_arr_all, ensemble_edge_temp_arr_all = _l2_signed_edge_policy_arrays(
        {
            "trade_score_threshold": float(ensemble_policy["trade_score_threshold"]),
            "trade_score_temperature": float(ensemble_policy["trade_score_temperature"]),
            "edge_temperature": float(ensemble_policy["edge_temperature"]),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": ensemble_conditional_policy_by_state,
        },
        frame,
    )
    _log_l2_signed_edge_val_diagnostics(
        trade_p_all[val_report_mask],
        fused_edge_all[val_report_mask],
        y_trade[val_report_mask],
        y_decision[val_report_mask],
        edge[val_report_mask],
        float(np.mean(ensemble_trade_threshold_arr_all[val_report_mask])),
        float(np.mean(ensemble_trade_temp_arr_all[val_report_mask])),
        float(np.mean(ensemble_edge_temp_arr_all[val_report_mask])),
        label="ensemble expected-edge",
    )
    direction_prob_all = _l2_predict_direction_fusion_model(
        fusion_frame,
        direction_fusion_model,
        fallback_frame=frame,
        triple_probs=triple_decision_probs,
        signed_edge_pred=signed_edge_all,
        signed_edge_temperature=signed_edge_temp_arr_all,
    )
    _, live_trade_prob_raw_all, _ = _l2_compose_live_probs_from_gate_dir_edge(
        trade_p_all,
        direction_prob_all,
        fused_edge_all,
        trade_score_threshold=ensemble_trade_threshold_arr_all,
        trade_score_temperature=ensemble_trade_temp_arr_all,
    )
    live_trade_calibrator = _fit_platt_calibrator((y_decision[val_tune_mask] != 1).astype(np.int32), live_trade_prob_raw_all[val_tune_mask])
    live_trade_prob_all = _apply_platt_calibrator(live_trade_prob_raw_all, live_trade_calibrator).astype(np.float32)
    if live_trade_calibrator is not None:
        platt_params = live_trade_calibrator.get_params_for_monitoring()
        print(
            f"[L2] live trade Platt params: slope={platt_params['slope']:.6f} intercept={platt_params['intercept']:.6f}",
            flush=True,
        )
    live_trade_threshold = _search_l2_trade_threshold(
        live_trade_prob_all[val_tune_mask],
        target_trade_rate=_l2_target_trade_rate(),
    )
    decision_probs = _l2_compose_probs_from_gate_dir(live_trade_prob_all, direction_prob_all)
    expected_edge_all = fused_edge_all.astype(np.float32)
    hard_decision_class, decision_confidence = _l2_hard_decode_prob_aligned_outputs(
        gate_p=live_trade_prob_all,
        dir_p=direction_prob_all,
        decision_probs=decision_probs,
        trade_threshold=live_trade_threshold,
    )
    _log_l2_extended_val_metrics(
        frame,
        val_report_mask,
        y_decision,
        decision_probs,
        hard_decision_class,
        y_trade == 1,
        y_size,
        size_pred,
        y_mfe,
        mfe_pred,
        y_mae,
        mae_pred,
        expected_edge_pred=expected_edge_all,
        true_edge=edge,
        test_mask=test_mask,
    )
    _log_l2_l1b_gain_importance_by_group(gate_model, feature_cols, "trade_gate")
    _log_l2_l1b_gain_importance_by_group(signed_edge_model, feature_cols, "signed_edge")
    _log_l2_l1b_ablation_triple_gate(
        X,
        feature_cols,
        val_report_mask,
        y_decision,
        gate_model,
        long_gate_model,
        short_gate_model,
        trade_threshold=triple_trade_threshold_arr_all,
        short_bias=triple_short_bias_arr_all,
        gate_calibrators=gate_calibrators,
    )
    _log_l2_l1b_ablation(
        X,
        feature_cols,
        val_report_mask,
        y_decision,
        gate_model,
        signed_edge_model,
        signed_edge_prep=signed_edge_prep,
        trade_score_threshold=signed_trade_threshold_arr_all,
        trade_score_temperature=signed_trade_temp_arr_all,
        edge_temperature=signed_edge_temp_arr_all,
        gate_calibrator=gate_calibrator,
    )
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_decision_class"] = hard_decision_class
    outputs["l2_decision_long"] = decision_probs[:, 0]
    outputs["l2_decision_neutral"] = decision_probs[:, 1]
    outputs["l2_decision_short"] = decision_probs[:, 2]
    outputs["l2_decision_confidence"] = decision_confidence
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_expected_edge"] = expected_edge_all
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)

    os.makedirs(MODEL_DIR, exist_ok=True)
    gate_model.save_model(os.path.join(MODEL_DIR, L2_GATE_FILE))
    long_gate_model.save_model(os.path.join(MODEL_DIR, L2_DIRECTION_FILE))
    short_gate_model.save_model(os.path.join(MODEL_DIR, L2_DECISION_FILE))
    signed_edge_model.save_model(os.path.join(MODEL_DIR, L2_SIGNED_EDGE_FILE))
    size_model.save_model(os.path.join(MODEL_DIR, L2_SIZE_FILE))
    mfe_model.save_model(os.path.join(MODEL_DIR, L2_MFE_FILE))
    mae_model.save_model(os.path.join(MODEL_DIR, L2_MAE_FILE))
    model_files: dict[str, str] = {
        "trade_gate": L2_GATE_FILE,
        "long_gate": L2_DIRECTION_FILE,
        "short_gate": L2_DECISION_FILE,
        "signed_edge": L2_SIGNED_EDGE_FILE,
        "size": L2_SIZE_FILE,
        "mfe": L2_MFE_FILE,
        "mae": L2_MAE_FILE,
    }
    if gate_calibrator is not None:
        gate_calib_file = "l2_trade_gate_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, gate_calib_file), "wb") as f:
            pickle.dump(gate_calibrator, f)
        model_files["trade_gate_calibrator"] = gate_calib_file
    if live_trade_calibrator is not None:
        live_trade_calib_file = "l2_live_trade_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, live_trade_calib_file), "wb") as f:
            pickle.dump(live_trade_calibrator, f)
        model_files["live_trade_calibrator"] = live_trade_calib_file
    meta = {
        "schema_version": L2_SCHEMA_VERSION,
        "l2_policy_state_granularity": os.environ.get("L2_POLICY_STATE_GRANULARITY", "coarse"),
        "l2_gate_no_trade_weight": float(gate_nt_w),
        "l2_l1b_train_feature_dropout": float(l1b_train_dropout_p),
        "l2_l1b_dropout_seed": int(l1b_do_seed),
        "l2_use_l1b_latent_features": bool(use_l1b_latent_feats),
        "feature_cols": feature_cols,
        "feature_group_counts": {
            "state": len(l1a_cols),
            "condition": len(l1b_cols),
            "direction": len(l1c_cols),
            "residual_other": len(res_cols),
        },
        "output_cols": L2_OUTPUT_COLS,
        "decision_mode": "live_hierarchical_gate_edge",
        "decision_tau": tau_global,
        "decision_tau_global": tau_global,
        "decision_tau_by_state": decision_tau_by_state,
        "policy_state_vol_quantiles": policy_vol_quantiles,
        "conditional_policy_by_state": ensemble_conditional_policy_by_state,
        "trade_score_threshold": float(ensemble_policy["trade_score_threshold"]),
        "trade_score_temperature": float(ensemble_policy["trade_score_temperature"]),
        "live_trade_threshold": float(live_trade_threshold),
        "edge_temperature": float(ensemble_policy["edge_temperature"]),
        "direction_available": True,
        "direction_head_type": "hierarchical_trade_then_direction_with_l1c_primary",
        "confidence_semantics": "probability-aligned; confidence equals the chosen class probability under the live trade-score plus direction decode",
        "decision_class_semantics": "first derive live trade probability from Platt-calibrated trade score built from calibrated trade gate and fused expected edge; once active, choose long vs short from the trained direction fusion model with L1c as the primary sign-bearing input",
        "size_semantics": "risk_adjusted_position_fraction",
        "expected_edge_semantics": "ridge-style fusion of triple-gate expected edge and signed-edge branch prediction, evaluated through gate-edge policy under explicit state/condition/direction input grouping",
        "feature_contract_semantics": {
            "state": "L1a contributes regime, volatility, persistence, and embedding context only",
            "condition": "L1b contributes tradeability, quality, novelty, and cluster context only",
            "direction": "L1c is the only directional owner consumed by L2",
            "residual_other": "PA state and session context features remain as local execution context",
        },
        "pa_state_features": list(PA_STATE_FEATURES),
        "pa_policy_semantics": "PA state buckets expand conditional policy keys, active label geometry, and direction weighting inside the fixed live Layer2 path",
        "pa_internal_semantics": {
            "baked_into_live_path": True,
            "tau": "PA states tighten or relax active decision tau before class labels are built",
            "triple_gate": "PA-conditioned row floors and sample weights define which edge examples count as active or trustworthy",
            "direction": "PA-conditioned fusion features and sample weights tell the direction head when long/short evidence is reliable",
        },
        "live_policy_surface": {
            "active_signal": "trade_gate_probability_times_abs_fused_expected_edge",
            "active_probability": "platt-calibrated sigmoid((trade_score-threshold)/temperature)",
            "direction_signal": "trained_direction_fusion_probability",
            "decode_threshold": float(live_trade_threshold),
        },
        "live_trade_calibration": (
            {
                "type": "platt",
                **live_trade_calibrator.get_params_for_monitoring(),
            }
            if live_trade_calibrator is not None
            else {"type": "none"}
        ),
        "l2_aux_head_target_prep": {
            "long": long_head_prep,
            "short": short_head_prep,
            "signed_edge": signed_edge_prep,
            "mfe": mfe_head_prep,
            "mae": mae_head_prep,
        },
        "l2_triple_gate_label_stats": triple_gate_label_stats,
        "l2_signed_edge_label_stats": signed_edge_label_stats,
        "policy_search": ensemble_policy,
        "triple_gate_branch": {
            "decision_mode": "triple_gate_regression",
            "trade_threshold": triple_trade_threshold,
            "short_bias": float(triple_policy.get("short_bias", 1.0)),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": triple_conditional_policy_by_state,
            "policy_search": triple_policy,
        },
        "signed_edge_branch": {
            "decision_mode": "signed_edge_gate",
            "trade_score_threshold": float(signed_policy["trade_score_threshold"]),
            "trade_score_temperature": float(signed_policy["trade_score_temperature"]),
            "edge_temperature": float(signed_policy["edge_temperature"]),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": signed_conditional_policy_by_state,
            "policy_search": signed_policy,
        },
        "ensemble_fusion_model": fusion_model,
        "direction_fusion_model": direction_fusion_model,
        "ensemble_policy": {
            "decision_mode": "live_hierarchical_gate_edge",
            "trade_score_threshold": float(ensemble_policy["trade_score_threshold"]),
            "trade_score_temperature": float(ensemble_policy["trade_score_temperature"]),
            "edge_temperature": float(ensemble_policy["edge_temperature"]),
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "conditional_policy_by_state": ensemble_conditional_policy_by_state,
            "policy_search": ensemble_policy,
        },
        "auxiliary_feature_semantics": "frozen L1 features, derived regime/uncertainty context, and L1b unsupervised interactions feed parallel triple-gate and signed-edge experts before fusion",
        "model_files": model_files,
        "output_cache_file": L2_OUTPUT_CACHE_FILE,
        "target_trade_rate": _l2_target_trade_rate(),
        "l2_val_tune_frac": tune_frac,
        "l2_min_feature_std": min_std,
        "l2_feature_hard_drop_skipped": skip_hard,
        "l2_feature_selection_dropped": l2_dropped_features,
        "l2_train_boost_rounds": int(rounds),
        "l2_early_stopping_rounds": {
            "gate": int(gate_es_rounds),
            "long_gate": int(side_es_rounds),
            "short_gate": int(side_es_rounds),
            "signed_edge": int(direction_es_rounds),
            "size": int(size_es_rounds),
            "mfe": int(mfe_es_rounds),
            "mae": int(mae_es_rounds),
        },
        "l2_trade_gate_config": {
            "learning_rate": float(trade_gate_params["learning_rate"]),
            "num_leaves": int(trade_gate_params["num_leaves"]),
            "max_depth": int(trade_gate_params["max_depth"]),
            "min_child_samples": int(trade_gate_params["min_child_samples"]),
            "bagging_fraction": float(trade_gate_params["bagging_fraction"]),
            "bagging_freq": int(trade_gate_params["bagging_freq"]),
            "is_unbalance": True,
            "metric_eval_order": ["auc", "binary_logloss"],
            "early_stopping_on": "first_metric (auc)",
            "early_stopping_rounds": int(gate_es_rounds),
        },
        "l2_long_score_config": long_score_params,
        "l2_short_score_config": short_score_params,
        "l2_signed_edge_config": signed_edge_params,
        "l2_regression_config": reg_params,
        "l2_size_target_config": {
            "mae_floor": float(size_mae_floor),
            "rr_cap": float(size_rr_cap),
            "edge_cap": float(size_edge_cap),
            "mae_decay": float(size_mae_decay),
            "mae_clip": float(size_mae_clip),
        },
    }
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    if test_mask.any():
        test_edge = outputs.loc[test_mask, "l2_expected_edge"].to_numpy(dtype=np.float32)
        corr = np.corrcoef(test_edge, edge[test_mask])[0, 1] if int(test_mask.sum()) > 2 else float("nan")
        print(f"  [L2] test corr(expected_edge, decision_edge_atr)={corr:.4f}", flush=True)
    print(f"  [L2] meta saved  -> {os.path.join(MODEL_DIR, L2_META_FILE)}", flush=True)
    print(f"  [L2] cache saved -> {cache_path}", flush=True)
    bundle_models: dict[str, Any] = {
        "trade_gate": gate_model,
        "long_gate": long_gate_model,
        "short_gate": short_gate_model,
        "signed_edge": signed_edge_model,
        "size": size_model,
        "mfe": mfe_model,
        "mae": mae_model,
    }
    if gate_calibrator is not None:
        bundle_models["trade_gate_calibrator"] = gate_calibrator
    if live_trade_calibrator is not None:
        bundle_models["live_trade_calibrator"] = live_trade_calibrator
    return L2TrainingBundle(models=bundle_models, meta=meta, outputs=outputs)


def load_l2_trade_decision() -> tuple[dict[str, Any], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L2_SCHEMA_VERSION:
        raise RuntimeError(
            f"L2 schema mismatch: artifact has {meta.get('schema_version')} but code expects {L2_SCHEMA_VERSION}. "
            f"Retrain L2 so artifacts match schema {L2_SCHEMA_VERSION}."
        )
    model_files = meta.get("model_files", {})
    models: dict[str, Any] = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in model_files.items()
        if not name.endswith("_calibrator")
    }
    for key, fname in model_files.items():
        if not key.endswith("_calibrator"):
            continue
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[key] = pickle.load(f)
    return models, meta


def _l2_require_inference_features(frame: pd.DataFrame, target_cols: list[str]) -> np.ndarray:
    missing = [col for col in target_cols if col not in frame.columns]
    if missing:
        sample = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f" ... +{len(missing) - 12} more"
        raise RuntimeError(
            "L2 live inference feature contract violated: missing trained feature columns: "
            f"{sample}{suffix}."
        )
    return frame[target_cols].to_numpy(dtype=np.float32, copy=False)


def infer_l2_trade_decision(
    models: dict[str, Any],
    meta: dict[str, Any],
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
    l1c_outputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame, _feature_cols = _build_l2_frame(df, l1a_outputs, l1b_outputs, l1c_outputs)
    target_cols = list(meta["feature_cols"])
    X = _l2_require_inference_features(frame, target_cols)
    mode = meta.get("decision_mode", "multiclass")
    aux_prep = meta.get("l2_aux_head_target_prep", {})
    if mode != "live_hierarchical_gate_edge":
        raise RuntimeError(
            f"L2 live inference only supports decision_mode='live_hierarchical_gate_edge', got {mode!r}. "
            "Retrain L2 with the live-hardening code path."
        )
    triple_cfg = meta.get("triple_gate_branch") or meta
    signed_cfg = meta.get("signed_edge_branch") or meta
    ensemble_cfg = meta.get("ensemble_policy") or meta
    trade_m = models["trade_gate"]
    long_m = models["long_gate"]
    short_m = models["short_gate"]
    signed_edge_m = models["signed_edge"]
    trade_p = _apply_binary_calibrator(
        trade_m.predict(X).astype(np.float64),
        models.get("trade_gate_calibrator"),
    ).astype(np.float32)
    long_p = _l2_positive_head_predict(long_m, X, aux_prep.get("long"), clip_max=float((aux_prep.get("long") or {}).get("clip_max", 5.0)))
    short_p = _l2_positive_head_predict(short_m, X, aux_prep.get("short"), clip_max=float((aux_prep.get("short") or {}).get("clip_max", 5.0)))
    _, triple_thr_arr, short_bias_arr = _l2_triple_gate_policy_arrays(triple_cfg, frame)
    _, _, triple_decision_probs = _l2_compose_probs_from_triple_gate(trade_p, long_p, short_p, short_bias=short_bias_arr)
    size_pred = np.clip(models["size"].predict(X).astype(np.float32), 0.0, 1.0)
    mfe_pred = _l2_positive_head_predict(models["mfe"], X, aux_prep.get("mfe"), clip_max=5.0)
    mae_pred = _l2_positive_head_predict(models["mae"], X, aux_prep.get("mae"), clip_max=4.0)
    triple_expected_edge = _l2_expected_edge_from_triple_gate(
        trade_p,
        long_p,
        short_p,
        size_pred,
        mfe_pred,
        mae_pred,
        trade_threshold=triple_thr_arr,
        short_bias=short_bias_arr,
    )
    signed_edge_pred = _l2_signed_edge_predict(signed_edge_m, X, aux_prep.get("signed_edge"))
    _, signed_thr_arr, signed_trade_temp_arr, signed_edge_temp_arr = _l2_signed_edge_policy_arrays(signed_cfg, frame)
    _, _, signed_decision_probs = _l2_compose_probs_from_gate_edge(
        trade_p,
        signed_edge_pred,
        trade_score_threshold=signed_thr_arr,
        trade_score_temperature=signed_trade_temp_arr,
        edge_temperature=signed_edge_temp_arr,
    )
    fusion_frame = _l2_fusion_feature_frame(
        frame,
        trade_p,
        triple_decision_probs,
        triple_expected_edge,
        signed_decision_probs,
        signed_edge_pred,
        size_pred,
        mfe_pred,
        mae_pred,
    )
    fused_edge_pred = _l2_predict_edge_fusion_model(fusion_frame, meta.get("ensemble_fusion_model") or {})
    _, ensemble_trade_thr_arr, ensemble_trade_temp_arr, _ensemble_edge_temp_arr = _l2_signed_edge_policy_arrays(ensemble_cfg, frame)
    direction_prob = _l2_predict_direction_fusion_model(
        fusion_frame,
        meta.get("direction_fusion_model") or {},
        fallback_frame=frame,
        triple_probs=triple_decision_probs,
        signed_edge_pred=signed_edge_pred,
        signed_edge_temperature=signed_edge_temp_arr,
    )
    _, live_trade_prob_raw, _ = _l2_compose_live_probs_from_gate_dir_edge(
        trade_p,
        direction_prob,
        fused_edge_pred,
        trade_score_threshold=ensemble_trade_thr_arr,
        trade_score_temperature=ensemble_trade_temp_arr,
    )
    live_trade_prob = _apply_platt_calibrator(live_trade_prob_raw, models.get("live_trade_calibrator")).astype(np.float32)
    decision_probs = _l2_compose_probs_from_gate_dir(live_trade_prob, direction_prob)
    hard_decision_class, decision_confidence = _l2_hard_decode_prob_aligned_outputs(
        gate_p=live_trade_prob,
        dir_p=direction_prob,
        decision_probs=decision_probs,
        trade_threshold=float(meta.get("live_trade_threshold", 0.5)),
    )
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_decision_class"] = hard_decision_class
    outputs["l2_decision_long"] = decision_probs[:, 0]
    outputs["l2_decision_neutral"] = decision_probs[:, 1]
    outputs["l2_decision_short"] = decision_probs[:, 2]
    outputs["l2_decision_confidence"] = decision_confidence
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_expected_edge"] = fused_edge_pred.astype(np.float32)
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)
    return outputs

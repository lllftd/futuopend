from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
    L2_DIRECTION_FILE,
    L2_DIRECTION_CALIBRATOR_FILE,
    L2_ENTRY_REGIME_COLS,
    L2_GATE_FILE,
    L2_TRADE_GATE_CALIBRATOR_FILE,
    L2_MAE_FILE,
    L2_META_FILE,
    L2_MFE_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
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
from core.trainers.l2.calibration import (
    apply_binary_calibrator as _apply_binary_calibrator,
    fit_binary_calibrator as _fit_binary_calibrator,
)
from core.trainers.pa_state_controls import (
    PA_STATE_BUCKET_TREND,
    ensure_pa_state_features,
    pa_state_arrays_from_frame,
    pa_state_bucket_labels_from_frame,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.trainers.stack_v2_common import build_stack_time_splits, l2_val_start_time, log_label_baseline, save_output_cache
from core.trainers.threshold_registry import attach_threshold_registry, threshold_entry
from core.trainers.val_metrics_extra import (
    brier_binary,
    brier_multiclass,
    ece_binary,
    ece_multiclass_maxprob,
    pearson_corr,
    regression_degen_flag,
    tail_mae_truth_upper,
)


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


_L2_GATE_BLOCKED_SIGN_FEATURES = frozenset(
    {
        "l1c_direction_prob",
        "l1c_direction_score",
        "l1c_weighted_dir",
        "l1c_dir_x_vol",
        "l1c_dir_x_state_persistence",
        "l2_dir_x_edge_opportunity",
        "l2_bull_mass_x_l1c_dir",
        "l2_range_mass_x_l1c_dir",
        "l2_signal_spread_var",
        "l2_breakout_quality_x_l1c_dir",
    }
)


def _l2_project_gate_features(
    X: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    blocked_idx = [idx for idx, name in enumerate(feature_cols) if name in _L2_GATE_BLOCKED_SIGN_FEATURES]
    if not blocked_idx:
        return X, []
    out = np.array(X, dtype=np.float32, copy=True)
    out[:, blocked_idx] = 0.0
    return out, [feature_cols[idx] for idx in blocked_idx]


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
    gamma = float(np.clip(float(os.environ.get("L2_DIRECTION_CLASS_WEIGHT_GAMMA", "1.25")), 0.5, 2.5))
    short_w = float(np.clip((total / (2.0 * n_short)) ** gamma, 0.60, 4.0))
    long_w = float(np.clip((total / (2.0 * n_long)) ** gamma, 0.60, 4.0))
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


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pp = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    return np.log(pp) - np.log1p(-pp)


def _l2_direction_prior_corrected_prob(
    dir_p: np.ndarray,
    *,
    train_long_prior: float,
    target_long_prior: float,
) -> np.ndarray:
    p = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    tr = float(np.clip(train_long_prior, 1e-4, 1.0 - 1e-4))
    tgt = float(np.clip(target_long_prior, 1e-4, 1.0 - 1e-4))
    shift = float(_safe_logit(np.asarray([tgt]))[0] - _safe_logit(np.asarray([tr]))[0])
    z = _safe_logit(p) + shift
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _l2_formula_size_from_context(
    frame: pd.DataFrame,
    trade_p: np.ndarray,
    dir_p: np.ndarray,
    *,
    trade_threshold: float | np.ndarray,
    clip_max: float = 5.0,
) -> np.ndarray:
    def _frame_col_or_default(name: str, default: float) -> np.ndarray:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
        return np.full(len(frame), float(default), dtype=np.float64)

    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    thr = np.asarray(trade_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(trade), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), trade.shape)
    direction_margin = np.abs(2.0 * direction - 1.0)
    active_strength = np.clip((trade - thr) / np.maximum(1.0 - thr, 1e-3), 0.0, 1.0)
    l1c_strength = _frame_col_or_default("l1c_direction_strength", 0.0)
    l1c_conf = _frame_col_or_default("l1c_confidence", 0.0)
    vol = _frame_col_or_default("l1a_vol_forecast", 1.0)
    persistence = _frame_col_or_default("l1a_state_persistence", 0.0)
    strength_norm = np.clip(l1c_strength / max(float(clip_max), 1e-3), 0.0, 1.0)
    vol_damp = 1.0 / (1.0 + np.clip(vol, 0.0, 5.0))
    stability_boost = 0.60 + 0.40 * np.clip(persistence, 0.0, 1.0)
    size = (
        active_strength
        * (0.35 + 0.65 * direction_margin)
        * (0.25 + 0.75 * strength_norm)
        * (0.50 + 0.50 * np.clip(l1c_conf, 0.0, 1.0))
        * vol_damp
        * stability_boost
    )
    return np.clip(size, 0.0, 1.0).astype(np.float32)


def _l2_bucketized_size_from_signals(
    frame: pd.DataFrame,
    trade_p: np.ndarray,
    dir_p: np.ndarray,
    *,
    trade_threshold: float,
    fit_mask: np.ndarray | None = None,
    bins: list[float] | None = None,
    levels: list[float] | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    thr = float(np.clip(trade_threshold, 0.01, 0.99))
    active_strength = np.clip((trade - thr) / max(1.0 - thr, 1e-3), 0.0, 1.0)
    dir_conf = np.abs((2.0 * direction) - 1.0)
    l1c_conf = pd.to_numeric(frame.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    l1c_strength = pd.to_numeric(frame.get("l1c_direction_strength", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    strength_norm = _quantile_rescale_01(np.abs(l1c_strength), fit_mask=fit_mask, q_low=0.05, q_high=0.95).astype(np.float64)
    edge_proxy = pd.to_numeric(frame.get("l1b_edge_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    edge_norm = _quantile_rescale_01(np.abs(edge_proxy), fit_mask=fit_mask, q_low=0.05, q_high=0.95).astype(np.float64)
    score = 0.45 * active_strength + 0.30 * dir_conf + 0.15 * np.clip(l1c_conf, 0.0, 1.0) + 0.10 * edge_norm
    score = np.clip(score, 0.0, 1.0)
    if levels is None:
        levels = [0.20, 0.35, 0.50, 0.70, 0.90]
    lv = [float(np.clip(x, 0.0, 1.0)) for x in levels]
    fit = np.ones(len(score), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).ravel()
    active_fit = fit & (trade >= thr)
    if bins is None:
        if np.any(active_fit):
            quantiles = np.linspace(0.2, 0.8, len(lv) - 1)
            bs = np.quantile(score[active_fit], quantiles).astype(np.float64).tolist()
        else:
            bs = [0.25, 0.40, 0.55, 0.70][: max(0, len(lv) - 1)]
    else:
        bs = [float(x) for x in bins]
    out = np.full(len(score), lv[0], dtype=np.float64)
    idx = np.searchsorted(np.asarray(bs, dtype=np.float64), score, side="right")
    idx = np.clip(idx, 0, len(lv) - 1)
    out = np.asarray([lv[int(i)] for i in idx], dtype=np.float64)
    out[trade < thr] = 0.0
    return out.astype(np.float32), bs, lv


def _l2_expected_edge_from_gate_dir(
    trade_p: np.ndarray,
    dir_p: np.ndarray,
    size_pred: np.ndarray,
    *,
    trade_threshold: float | np.ndarray,
    direction_strength: pd.Series | np.ndarray | None = None,
    l1b_edge_proxy: pd.Series | np.ndarray | None = None,
    range_mass: pd.Series | np.ndarray | None = None,
    regime_entropy: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    size = np.asarray(size_pred, dtype=np.float64).ravel()
    thr = np.asarray(trade_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(trade), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), trade.shape)
    active_strength = np.clip((trade - thr) / np.maximum(1.0 - thr, 1e-3), 0.0, 1.0)
    dir_margin = (2.0 * direction) - 1.0
    if direction_strength is None:
        strength = np.abs(dir_margin)
    else:
        strength = np.asarray(direction_strength, dtype=np.float64).ravel()
        strength = np.clip(strength, 0.0, 5.0)
    if l1b_edge_proxy is None:
        edge_proxy_mag = np.abs(dir_margin)
    else:
        edge_proxy_mag = np.asarray(l1b_edge_proxy, dtype=np.float64).ravel()
        edge_proxy_mag = np.clip(edge_proxy_mag, 0.0, 5.0) / 5.0
    if range_mass is None:
        rm = np.zeros(len(trade), dtype=np.float64)
    else:
        rm = np.clip(np.asarray(range_mass, dtype=np.float64).ravel(), 0.0, 1.0)
    if regime_entropy is None:
        ent_norm = np.zeros(len(trade), dtype=np.float64)
    else:
        ent = np.clip(np.asarray(regime_entropy, dtype=np.float64).ravel(), 0.0, np.log(float(NUM_REGIME_CLASSES)))
        ent_norm = ent / max(np.log(float(NUM_REGIME_CLASSES)), 1e-6)
    range_lambda = float(np.clip(float(os.environ.get("L2_EXPECTED_EDGE_RANGE_LAMBDA", "0.45")), 0.0, 1.5))
    entropy_lambda = float(np.clip(float(os.environ.get("L2_EXPECTED_EDGE_ENTROPY_LAMBDA", "0.30")), 0.0, 1.5))
    regime_shrink = np.clip(1.0 - range_lambda * rm - entropy_lambda * ent_norm, 0.20, 1.0)
    edge = (
        active_strength
        * size
        * np.sign(dir_margin)
        * np.maximum(np.abs(dir_margin), 1e-3)
        * np.maximum(strength, 1e-3)
        * (0.35 + 0.65 * edge_proxy_mag)
        * regime_shrink
    )
    return np.clip(edge, -5.0, 5.0).astype(np.float32)


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


def _l2_hard_decision_from_gate_dir(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    thr: float,
    *,
    direction_abstain_margin: float = 0.0,
) -> np.ndarray:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    dir_p = np.asarray(dir_p, dtype=np.float64).ravel()
    thr_arr = np.asarray(thr, dtype=np.float64)
    if thr_arr.ndim == 0:
        thr_arr = np.full(len(gate_p), float(thr_arr), dtype=np.float64)
    else:
        thr_arr = np.broadcast_to(thr_arr.reshape(-1), gate_p.shape)
    out = np.ones(len(gate_p), dtype=np.int64)
    trade = gate_p >= thr_arr
    margin = float(np.clip(direction_abstain_margin, 0.0, 0.49))
    dir_conf = np.abs((2.0 * dir_p) - 1.0)
    abstain = dir_conf <= margin
    long_mask = trade & (~abstain) & (dir_p >= 0.5)
    short_mask = trade & (~abstain) & (dir_p < 0.5)
    out[long_mask] = 0
    out[short_mask] = 2
    return out


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


def _search_l2_direction_abstain_margin(
    trade_p: np.ndarray,
    direction_p: np.ndarray,
    *,
    trade_threshold: float,
    target_abstain_rate: float = 0.10,
) -> float:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.asarray(direction_p, dtype=np.float64).ravel()
    live = trade >= float(trade_threshold)
    if not np.any(live):
        return 0.0
    conf = np.abs((2.0 * np.clip(direction[live], 0.0, 1.0)) - 1.0)
    target = float(np.clip(target_abstain_rate, 0.0, 0.5))
    if target <= 1e-6:
        return 0.0
    margin = float(np.quantile(conf, target))
    realized = float(np.mean(conf <= margin))
    tie_ratio = 1.0 - (float(np.unique(conf).size) / float(max(conf.size, 1)))
    at_margin_share = float(np.mean(conf == margin))
    print("\n  [L2] direction abstain-margin search on active trades", flush=True)
    for rate in [0.05, target, min(0.25, max(target, 0.05) + 0.10)]:
        cand = float(np.quantile(conf, float(np.clip(rate, 0.0, 0.5))))
        cand_rate = float(np.mean(conf <= cand))
        mark = "  *" if abs(rate - target) < 1e-9 else ""
        print(f"    target_abstain_rate={rate:.3f}  margin={cand:.4f}  realized={cand_rate:.3f}{mark}", flush=True)
    print(
        f"    direction_conf stats: pcts={np.round(np.percentile(conf, [0, 5, 25, 50, 75, 95, 100]), 4).tolist()}  "
        f"tie_ratio={tie_ratio:.4f}  at_margin_share={at_margin_share:.4f}",
        flush=True,
    )
    print(
        f"  [L2] selected direction_abstain_margin={margin:.4f}  target_abstain_rate={target:.3f}  realized={realized:.3f}",
        flush=True,
    )
    return float(np.clip(margin, 0.0, 0.45))


def _l2_edge_diagnosis_branch(
    e_pred: np.ndarray,
    e_true: np.ndarray,
    y_decision: np.ndarray,
    mask: np.ndarray,
) -> str:
    sm = np.asarray(mask, dtype=bool)
    yp = np.asarray(e_pred, dtype=np.float64).ravel()
    yt = np.asarray(e_true, dtype=np.float64).ravel()
    yd = np.asarray(y_decision, dtype=np.int64).ravel()
    active = sm & (yd != 1)
    if not np.any(active):
        return "A2"
    sign_acc = float(np.mean(np.sign(yp[active]) == np.sign(yt[active])))
    corr_active = pearson_corr(yp[active], yt[active])
    if sign_acc < 0.35 and np.isfinite(corr_active) and corr_active <= 0.0:
        return "A1"
    if sign_acc < 0.45 and np.isfinite(corr_active) and corr_active < 0.05:
        return "A3"
    return "A2"


def _log_l2_two_stage_val_diagnostics(
    trade_p: np.ndarray,
    dir_p: np.ndarray,
    y_trade: np.ndarray,
    y_dir: np.ndarray,
    y_decision: np.ndarray,
    *,
    trade_threshold: float,
    direction_abstain_margin: float = 0.0,
    split_label: str = "val_report",
) -> None:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.asarray(dir_p, dtype=np.float64).ravel()
    y_trade = np.asarray(y_trade, dtype=np.int64).ravel()
    y_dir = np.asarray(y_dir, dtype=np.int64).ravel()
    y_decision = np.asarray(y_decision, dtype=np.int64).ravel()
    print(f"\n  [L2] {split_label} — two-stage gate + direction", flush=True)
    try:
        auc_trade = float(roc_auc_score(y_trade, trade))
    except ValueError:
        auc_trade = float("nan")
    brier_trade = brier_binary(y_trade.astype(np.float64), trade)
    ece_trade = ece_binary(y_trade.astype(np.float64), trade)
    pred_trade = (trade >= float(trade_threshold)).astype(np.int32)
    print(
        f"    trade_gate AUC={auc_trade:.4f}  Brier={brier_trade:.4f}  ECE={ece_trade:.4f}  "
        f"trade_threshold={float(trade_threshold):.4f}",
        flush=True,
    )
    print(
        "    trade_gate classification_report:\n"
        + classification_report(y_trade, pred_trade, target_names=["no_trade", "trade"], digits=4, zero_division=0),
        flush=True,
    )
    active = y_dir >= 0
    if active.any():
        try:
            auc_dir = float(roc_auc_score(y_dir[active], direction[active]))
        except ValueError:
            auc_dir = float("nan")
        pred_dir = (direction[active] >= 0.5).astype(np.int32)
        brier_dir = brier_binary(y_dir[active].astype(np.float64), direction[active])
        ece_dir = ece_binary(y_dir[active].astype(np.float64), direction[active])
        long_mask = y_dir[active] == 1
        short_mask = y_dir[active] == 0
        pred_long_mask = pred_dir == 1
        pred_short_mask = pred_dir == 0
        long_recall = float(np.mean(pred_dir[long_mask] == 1)) if long_mask.any() else float("nan")
        short_recall = float(np.mean(pred_dir[short_mask] == 0)) if short_mask.any() else float("nan")
        long_precision = float(np.mean(y_dir[active][pred_long_mask] == 1)) if pred_long_mask.any() else float("nan")
        short_precision = float(np.mean(y_dir[active][pred_short_mask] == 0)) if pred_short_mask.any() else float("nan")
        print(
            f"    direction AUC(active)={auc_dir:.4f}  Brier(active)={brier_dir:.4f}  ECE(active)={ece_dir:.4f}  "
            f"p(long) pcts(true-active)={np.round(np.percentile(direction[active], [5, 25, 50, 75, 95]), 4).tolist()}",
            flush=True,
        )
        print(
            "    direction classification_report (active rows):\n"
            + classification_report(y_dir[active], pred_dir, target_names=["short", "long"], digits=4, zero_division=0),
            flush=True,
        )
        print(
            f"    direction active mix: true_long={float(np.mean(long_mask)):.3f}  true_short={float(np.mean(short_mask)):.3f}  "
            f"pred_long={float(np.mean(pred_long_mask)):.3f}  pred_short={float(np.mean(pred_short_mask)):.3f}",
            flush=True,
        )
        print(
            f"    short-bias audit(active): long_recall={long_recall:.4f}  short_recall={short_recall:.4f}  "
            f"recall_delta(long-short)={long_recall - short_recall:+.4f}  "
            f"long_precision={long_precision:.4f}  short_precision={short_precision:.4f}  "
            f"precision_delta(long-short)={long_precision - short_precision:+.4f}",
            flush=True,
        )
    pred_hard = _l2_hard_decision_from_gate_dir(
        trade,
        direction,
        float(trade_threshold),
        direction_abstain_margin=float(direction_abstain_margin),
    )
    active_trade = trade >= float(trade_threshold)
    if np.any(active_trade):
        pred_dir = direction[active_trade]
        pred_conf = np.abs((2.0 * pred_dir) - 1.0)
        pred_tie_ratio = 1.0 - (float(np.unique(pred_conf).size) / float(max(pred_conf.size, 1)))
        margin = float(np.clip(direction_abstain_margin, 0.0, 0.49))
        pred_abstain_hit = float(np.mean(pred_conf <= margin))
        print(
            f"    direction p(long) pcts(pred-trade-active)={np.round(np.percentile(pred_dir, [5, 25, 50, 75, 95]), 4).tolist()}  "
            f"conf pcts(pred-trade-active)={np.round(np.percentile(pred_conf, [0, 5, 25, 50, 75, 95, 100]), 4).tolist()}  "
            f"conf_min={float(np.min(pred_conf)):.4f}  tie_ratio={pred_tie_ratio:.4f}  "
            f"abstain_hit(<=margin)={pred_abstain_hit:.4f}",
            flush=True,
        )
        active_pred = pred_hard[active_trade]
        abstain_rate = float(np.mean(active_pred == 1))
        directional = active_pred != 1
        if directional.any():
            long_share = float(np.mean(active_pred[directional] == 0))
            short_share = float(np.mean(active_pred[directional] == 2))
        else:
            long_share = float("nan")
            short_share = float("nan")
        print(
            f"    active trade mix: abstain_rate={abstain_rate:.4f}  long_share={long_share:.4f}  short_share={short_share:.4f}",
            flush=True,
        )
    print(
        f"\n  [L2] {split_label} — hard two-stage vs truth: pred_active={float(np.mean(pred_hard != 1)):.3f}  "
        f"true_active={float(np.mean(y_decision != 1)):.3f}  abstain_margin={float(direction_abstain_margin):.4f}",
        flush=True,
    )


def _log_l2_decision_split_metrics(
    frame: pd.DataFrame,
    split_mask: np.ndarray,
    y_decision: np.ndarray,
    decision_probs: np.ndarray,
    hard_decision: np.ndarray | None,
    *,
    split_label: str,
) -> None:
    sm = np.asarray(split_mask, dtype=bool)
    if not sm.any():
        return
    yv = y_decision[sm]
    Pv = np.asarray(decision_probs[sm], dtype=np.float64)
    Pv = np.clip(Pv, 1e-15, 1.0)
    Pv = Pv / Pv.sum(axis=1, keepdims=True)
    pred = np.argmax(Pv, axis=1) if hard_decision is None else np.asarray(hard_decision, dtype=np.int64)[sm]
    try:
        ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
    except ValueError:
        ll = float("nan")
    br = brier_multiclass(yv, Pv, 3)
    ece = ece_multiclass_maxprob(yv, Pv)
    acc = float(accuracy_score(yv, pred))
    f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
    cm = confusion_matrix(yv, pred, labels=[0, 1, 2])
    conf = np.max(Pv, axis=1)
    f1w = float(f1_score(yv, pred, average="weighted", zero_division=0))
    print(f"\n  [L2] {split_label} — decision (extended)", flush=True)
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
    class_stats: dict[int, dict[str, float]] = {}
    for cls_idx, cls_name in ((0, "long"), (2, "short")):
        mask = yv == cls_idx
        if not mask.any():
            continue
        recall = float(np.mean(pred[mask] == cls_idx))
        pred_mask = pred == cls_idx
        precision = float(np.mean(yv[pred_mask] == cls_idx)) if pred_mask.any() else float("nan")
        pcts = np.percentile(Pv[mask, cls_idx], [5, 25, 50, 75, 95])
        class_stats[cls_idx] = {"recall": recall, "precision": precision}
        print(
            f"    {cls_name}: n={int(mask.sum()):,}  recall={recall:.4f}  precision={precision:.4f}  "
            f"prob_pcts={np.round(pcts, 4).tolist()}",
            flush=True,
        )
    if 0 in class_stats and 2 in class_stats:
        print(
            f"    short-bias audit: recall_delta(long-short)={class_stats[0]['recall'] - class_stats[2]['recall']:+.4f}  "
            f"precision_delta(long-short)={class_stats[0]['precision'] - class_stats[2]['precision']:+.4f}",
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
    R = frame.loc[sm, L1A_REGIME_COLS].to_numpy(dtype=np.float64, copy=False)
    R = np.clip(R, 1e-12, 1.0)
    R = R / R.sum(axis=1, keepdims=True)
    ent = -np.sum(R * np.log(R), axis=1)
    c_l1_l2 = pearson_corr(ent, conf)
    print(f"    corr(L1 regime entropy, L2 max prob)={c_l1_l2:.4f}", flush=True)


def _log_l2_expected_edge_time_slices(
    frame: pd.DataFrame,
    split_mask: np.ndarray,
    e_pred: np.ndarray,
    e_true: np.ndarray,
    y_decision: np.ndarray,
    *,
    split_label: str,
) -> None:
    sm = np.asarray(split_mask, dtype=bool)
    if not sm.any() or "time_key" not in frame.columns:
        return
    ts = pd.to_datetime(frame.loc[sm, "time_key"])
    months = ts.dt.to_period("M").astype(str).to_numpy()
    yp = np.asarray(e_pred, dtype=np.float64).ravel()[sm]
    yt = np.asarray(e_true, dtype=np.float64).ravel()[sm]
    yd = np.asarray(y_decision, dtype=np.int64).ravel()[sm]
    print(f"    expected_edge {split_label} monthly stability:", flush=True)
    for month in pd.unique(months):
        mm = months == month
        n = int(np.sum(mm))
        if n == 0:
            continue
        corr_all = pearson_corr(yp[mm], yt[mm]) if n >= 10 else float("nan")
        act = mm & (yd != 1)
        corr_active = pearson_corr(yp[act], yt[act]) if int(np.sum(act)) >= 10 else float("nan")
        pred_active = float(np.mean(np.abs(yp[mm]) > 1e-6))
        print(
            f"      {month}: n={n:,}  pred_active≈{pred_active:.3f}  corr_all={corr_all:.4f}  corr_active={corr_active:.4f}",
            flush=True,
        )


def _log_l2_extended_val_metrics(
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    decision_probs: np.ndarray,
    hard_decision: np.ndarray | None,
    aux_active_mask: np.ndarray | None,
    y_size: np.ndarray | None,
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
    _log_l2_decision_split_metrics(frame, vm, y_decision, decision_probs, hard_decision, split_label="val_report")
    tm = (
        np.asarray(test_mask, dtype=bool).ravel()
        if test_mask is not None and np.asarray(test_mask, dtype=bool).any()
        else None
    )
    if tm is not None:
        _log_l2_decision_split_metrics(frame, tm, y_decision, decision_probs, hard_decision, split_label="holdout")
    if expected_edge_pred is not None and true_edge is not None:
        e_pred = np.asarray(expected_edge_pred, dtype=np.float64).ravel()
        e_true = np.asarray(true_edge, dtype=np.float64).ravel()
        corr_all = pearson_corr(e_pred[vm], e_true[vm])
        active_true = vm & (y_decision != 1)
        active_pred = (
            vm & (np.asarray(hard_decision, dtype=np.int64).ravel() != 1)
            if hard_decision is not None
            else vm & (np.abs(e_pred) > 1e-6)
        )
        corr_active = pearson_corr(e_pred[active_true], e_true[active_true]) if active_true.any() else float("nan")
        sign_acc_true = (
            float(np.mean(np.sign(e_pred[active_true]) == np.sign(e_true[active_true])))
            if active_true.any()
            else float("nan")
        )
        sign_acc_pred = (
            float(np.mean(np.sign(e_pred[active_pred]) == np.sign(e_true[active_pred])))
            if active_pred.any()
            else float("nan")
        )
        print(
            f"    expected_edge: corr_all={corr_all:.4f}  corr_active={corr_active:.4f}  "
            f"sign_acc_true_active={sign_acc_true:.4f}  sign_acc_pred_active={sign_acc_pred:.4f}  "
            f"pred_active={float(np.mean(active_pred[vm])):.3f}",
            flush=True,
        )
        pos = vm & (e_pred > 0.0)
        neg = vm & (e_pred < 0.0)
        mean_ret_pos = float(np.mean(e_true[pos])) if np.any(pos) else float("nan")
        mean_ret_neg = float(np.mean(e_true[neg])) if np.any(neg) else float("nan")
        print(
            f"    expected_edge sign buckets: mean_true_edge(edge>0)={mean_ret_pos:.6f}  "
            f"mean_true_edge(edge<0)={mean_ret_neg:.6f}",
            flush=True,
        )
        branch = "A2"
        if np.isfinite(sign_acc_true):
            if sign_acc_true < 0.35 and np.isfinite(corr_active) and corr_active <= 0.0:
                branch = "A1"
            elif sign_acc_true < 0.45 and np.isfinite(corr_active) and corr_active < 0.05:
                branch = "A3"
        print(
            f"    [L2][P0-A] diagnosis_branch={branch}  "
            "A1=likely_sign_flip  A2=formula_scaling_issue  A3=upstream_signal_quality_issue",
            flush=True,
        )
        rid_all = np.argmax(frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64), axis=1)
        if tm is not None:
            corr_test = pearson_corr(e_pred[tm], e_true[tm])
            act_t = tm & (y_decision != 1)
            corr_test_active = pearson_corr(e_pred[act_t], e_true[act_t]) if act_t.any() else float("nan")
            print(
                f"    expected_edge holdout: corr_all={corr_test:.4f}  corr_active={corr_test_active:.4f}",
                flush=True,
            )
            _log_l2_expected_edge_time_slices(frame, tm, e_pred, e_true, y_decision, split_label="holdout")
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
    if y_size is not None and av.sum() >= 5:
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
    elif y_size is None:
        print("\n  [L2] val — size: skipped supervised metrics (formula-derived, no direct target)", flush=True)
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


def _l2_dynamic_hard_drop_from_prev_model(feature_cols: list[str]) -> set[str]:
    path = os.path.join(MODEL_DIR, L2_GATE_FILE)
    if not os.path.exists(path):
        return set()
    try:
        booster = lgb.Booster(model_file=path)
    except Exception:
        return set()
    names = list(booster.feature_name() or [])
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64).ravel()
    if len(names) != len(gains) or len(names) == 0:
        return set()
    total = float(np.sum(np.maximum(gains, 0.0)))
    rel = np.maximum(gains, 0.0) / max(total, 1e-12)
    gain_map = {str(n): float(g) for n, g in zip(names, rel)}
    extra: set[str] = set()
    det_thr = float(np.clip(float(os.environ.get("L2_PRUNE_DETERMINISTIC_GAIN_THR", "0.005")), 0.0, 0.05))
    for n in feature_cols:
        if _l2_l1b_importance_bucket(n) == "l1b_deterministic_semantic" and gain_map.get(n, 1.0) < det_thr:
            extra.add(n)
    bottom_n = int(np.clip(int(os.environ.get("L2_PRUNE_BOTTOM_N", "0")), 0, 128))
    if bottom_n > 0:
        ranked = sorted([(n, gain_map.get(n, 0.0)) for n in feature_cols], key=lambda x: x[1])
        for n, _ in ranked[:bottom_n]:
            extra.add(n)
    if extra:
        print(f"  [L2] dynamic hard-drop from previous gate model: {len(extra)} cols", flush=True)
    return extra


def _l2_l1b_importance_bucket(name: str) -> str:
    if name in {
        "l1b_edge_pred",
        "l1b_dq_pred",
        "l2_vol_adjusted_l1b_edge",
        "l2_dir_x_edge_opportunity",
        "l2_dir_conf_x_edge_mag",
        "l2_dq_x_edge",
        "l2_signal_strength_mean_abs",
        "l2_signal_spread_var",
    }:
        return "l1b_supervised_pred"
    if name.startswith("l1b_cluster_prob_") or name in {"l1b_cluster_top1", "l1b_cluster_top2_gap"}:
        return "l1b_unsup_cluster"
    if name.startswith("l1b_latent_") or name in {
        "l1b_novelty_score",
        "l1b_regime_change_score",
        "l1b_novelty_x_vol",
        "l1b_regime_change_x_entropy",
        "l1b_unsup_pressure",
    }:
        return "l1b_unsup_descriptor"
    if name in {"l1b_sector_relative_strength", "l1b_correlation_regime", "l1b_market_breadth"}:
        return "l1b_context"
    if name.startswith("l1b_atom_"):
        return "l1b_atom_internal"
    if name == "l2_breakout_quality_x_l1c_dir":
        return "l1b_deterministic_semantic"
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


def _l2_zero_feature_buckets(
    X: np.ndarray,
    feature_cols: list[str],
    *,
    buckets: set[str] | None = None,
    all_l1b: bool = False,
) -> tuple[np.ndarray, list[str]]:
    if all_l1b:
        idx = [j for j, name in enumerate(feature_cols) if _l2_l1b_importance_bucket(str(name)) != "non_l1b"]
    else:
        tgt = set(buckets or set())
        idx = [j for j, name in enumerate(feature_cols) if _l2_l1b_importance_bucket(str(name)) in tgt]
    if not idx:
        return np.asarray(X, dtype=np.float32, copy=True), []
    out = np.asarray(X, dtype=np.float32, copy=True)
    out[:, idx] = 0.0
    return out, [feature_cols[j] for j in idx]


def _log_l2_l1b_masking_audit(
    X: np.ndarray,
    feature_cols: list[str],
    *,
    val_mask: np.ndarray,
    y_trade: np.ndarray,
    y_dir_stage: np.ndarray,
    y_decision: np.ndarray,
    trade_model: lgb.Booster,
    direction_model: lgb.Booster,
    gate_calibrator: Any | None,
    direction_calibrator: Any | None,
    trade_threshold: float,
    direction_abstain_margin: float = 0.0,
) -> None:
    if os.environ.get("L2_L1B_MASKING_AUDIT", "1").strip().lower() in {"0", "false", "no"}:
        return
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return
    audits = [
        ("baseline", None, False),
        ("without_l1b", None, True),
        ("no_l1b_supervised", {"l1b_supervised_pred"}, False),
        ("no_l1b_cluster", {"l1b_unsup_cluster"}, False),
        ("no_l1b_unsup_descriptor", {"l1b_unsup_descriptor"}, False),
        ("no_l1b_deterministic", {"l1b_deterministic_semantic"}, False),
    ]
    print("\n  [L2] val_report — L1b masking audit (post-fit, no retraining)", flush=True)
    base_log_loss = float("nan")
    base_f1 = float("nan")
    for label, buckets, all_l1b in audits:
        X_masked, zeroed = _l2_zero_feature_buckets(X, feature_cols, buckets=buckets, all_l1b=all_l1b)
        if label != "baseline" and not zeroed:
            print(f"    {label}: skipped (no matching features)", flush=True)
            continue
        X_gate, _ = _l2_project_gate_features(X_masked, feature_cols)
        trade_p = _apply_binary_calibrator(
            trade_model.predict(X_gate).astype(np.float64),
            gate_calibrator,
        ).astype(np.float32)
        direction_p = _apply_binary_calibrator(
            direction_model.predict(X_masked).astype(np.float64),
            direction_calibrator,
        ).astype(np.float32)
        probs = _l2_compose_probs_from_gate_dir(trade_p, direction_p).astype(np.float64)
        probs = np.clip(probs, 1e-15, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pred = _l2_hard_decision_from_gate_dir(
            trade_p,
            direction_p,
            float(trade_threshold),
            direction_abstain_margin=float(direction_abstain_margin),
        )
        try:
            gate_auc = float(roc_auc_score(y_trade[vm], trade_p[vm]))
        except ValueError:
            gate_auc = float("nan")
        active = vm & (y_dir_stage >= 0)
        try:
            dir_auc = float(roc_auc_score(y_dir_stage[active], direction_p[active])) if active.any() else float("nan")
        except ValueError:
            dir_auc = float("nan")
        try:
            ll = float(log_loss(y_decision[vm], probs[vm], labels=[0, 1, 2]))
        except ValueError:
            ll = float("nan")
        f1m = float(f1_score(y_decision[vm], pred[vm], average="macro", zero_division=0))
        pred_active = float(np.mean(pred[vm] != 1))
        if label == "baseline":
            base_log_loss = ll
            base_f1 = f1m
        delta_ll = ll - base_log_loss if np.isfinite(base_log_loss) else float("nan")
        delta_f1 = f1m - base_f1 if np.isfinite(base_f1) else float("nan")
        zeroed_desc = f"  zeroed={len(zeroed)}" if label != "baseline" else ""
        print(
            f"    {label}: gate_auc={gate_auc:.4f}  dir_auc={dir_auc:.4f}  log_loss={ll:.4f}  "
            f"Δlog_loss={delta_ll:+.4f}  F1_macro={f1m:.4f}  ΔF1={delta_f1:+.4f}  pred_active={pred_active:.3f}{zeroed_desc}",
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


def _l2_hard_decode_prob_aligned_outputs(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    decision_probs: np.ndarray,
    *,
    trade_threshold: float,
    direction_abstain_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    hard_cls = _l2_hard_decision_from_gate_dir(
        gate_p,
        dir_p,
        trade_threshold,
        direction_abstain_margin=direction_abstain_margin,
    )
    prob_mat = np.asarray(decision_probs, dtype=np.float32)
    confidence = prob_mat[np.arange(len(prob_mat)), hard_cls].astype(np.float32)
    if float(direction_abstain_margin) > 0.0:
        direction = np.asarray(dir_p, dtype=np.float64).ravel()
        dir_conf = np.abs((2.0 * np.clip(direction, 0.0, 1.0)) - 1.0).astype(np.float32)
        abstain = hard_cls == 1
        if np.any(abstain):
            confidence[abstain] = np.maximum(confidence[abstain], (1.0 - dir_conf[abstain]).astype(np.float32))
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
    ]
    allow.extend([c for c in merged.columns if c.startswith("l1b_cluster_prob_")])
    return [c for c in allow if c in merged.columns]


def _l2_direction_input_cols(merged: pd.DataFrame) -> list[str]:
    allow = [
        "l1c_direction_prob",
        "l1c_direction_score",
        "l1c_confidence",
        "l1c_direction_strength",
        "l1c_conf_zone",
        "l1c_is_warm",
    ]
    return [c for c in allow if c in merged.columns]


def _l2_residual_input_cols(residual: pd.DataFrame) -> list[str]:
    return [c for c in residual.columns if c not in {"symbol", "time_key"}]


def _legacy_l1a_direction_cols(merged: pd.DataFrame) -> list[str]:
    legacy_named = {"l1a_bull_convergence", "l1a_bear_convergence"}
    return [c for c in merged.columns if c.startswith("l1a_dir_") or c in legacy_named]


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
        l1c_p = pd.to_numeric(merged.get("l1c_direction_prob", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float32, copy=False)
        l1c_c = pd.to_numeric(merged.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        l1c_s = pd.to_numeric(merged.get("l1c_direction_strength", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        abst_margin = float(np.clip(float(os.environ.get("L1C_ABSTAIN_MARGIN", "0.10")), 0.02, 0.30))
        low_margin = float(np.clip(abst_margin * 2.0, abst_margin + 0.01, 0.45))
        d = np.abs(l1c_p - 0.5)
        conf_zone = np.where(d < abst_margin, 0.0, np.where(d < low_margin, 1.0, 2.0)).astype(np.float32)
        out["l1c_conf_zone"] = conf_zone
        out["l1c_weighted_dir"] = (l1c_d * l1c_c).astype(np.float32)
        out["l1c_dir_x_vol"] = (l1c_d * vol).astype(np.float32)
        out["l1c_dir_x_state_persistence"] = (l1c_d * persistence).astype(np.float32)
        out["l1c_strength_x_conf"] = (l1c_s * l1c_c).astype(np.float32)
    else:
        out["l1c_conf_zone"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_weighted_dir"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_dir_x_vol"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_dir_x_state_persistence"] = np.zeros(n_m, dtype=np.float32)
        out["l1c_strength_x_conf"] = np.zeros(n_m, dtype=np.float32)

    l1b_edge_f = pd.to_numeric(merged.get("l1b_edge_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1b_dq_f = pd.to_numeric(merged.get("l1b_dq_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1b_breakout_f = pd.to_numeric(merged.get("l1b_breakout_quality", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1b_tau_f = pd.to_numeric(merged.get("l1b_edge_candidate_tau", np.nan), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if np.isfinite(l1b_tau_f).any():
        tau_eff = np.where(np.isfinite(l1b_tau_f), l1b_tau_f, np.nanmedian(l1b_tau_f[np.isfinite(l1b_tau_f)]))
    else:
        tau_q = float(np.clip(float(os.environ.get("L2_L1B_TAU_PROXY_Q", "0.70")), 0.50, 0.95))
        tau_proxy = float(np.quantile(np.abs(l1b_edge_f).astype(np.float64), tau_q)) if len(l1b_edge_f) else 0.05
        tau_eff = np.full(len(l1b_edge_f), max(tau_proxy, 1e-3), dtype=np.float64)
    out["l2_l1b_edge_over_tau"] = (l1b_edge_f.astype(np.float64) / np.maximum(tau_eff, 1e-3)).astype(np.float32)
    vol_f = pd.to_numeric(merged.get("l1a_vol_forecast", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    out["l2_vol_adjusted_l1b_edge"] = (l1b_edge_f / np.maximum(vol_f, 1e-4)).astype(np.float32)
    l1c_dir_f = pd.to_numeric(merged.get("l1c_direction_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    l1c_conf_f = pd.to_numeric(merged.get("l1c_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    out["l2_dir_x_edge_opportunity"] = (l1c_dir_f * l1b_edge_f).astype(np.float32)
    out["l2_dir_conf_x_edge_mag"] = (l1c_conf_f * np.abs(l1b_edge_f)).astype(np.float32)
    out["l2_dq_x_edge"] = (l1b_dq_f * l1b_edge_f).astype(np.float32)
    out["l2_breakout_quality_x_l1c_dir"] = (l1b_breakout_f * l1c_dir_f).astype(np.float32)

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
    legacy_l1a_dir_cols = _legacy_l1a_direction_cols(merged)
    if legacy_l1a_dir_cols:
        merged = merged.drop(columns=legacy_l1a_dir_cols)
        print(
            f"  [L2] ignoring {len(legacy_l1a_dir_cols)} legacy L1a direction cols: {legacy_l1a_dir_cols}",
            flush=True,
        )
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
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L2] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
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
        hard_drop |= _l2_dynamic_hard_drop_from_prev_model(feature_cols)
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
    y_trade, y_dir_stage = _l2_build_two_stage_labels(y_decision)
    dir_weights = _l2_direction_sample_weights(y_dir_stage, fusion_frame=frame)
    active_train = train_mask & (y_dir_stage >= 0)
    active_val = val_mask & (y_dir_stage >= 0)
    y_mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    y_mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    y_mfe_fit, mfe_head_prep = _l2_positive_head_target_prep(y_mfe, head_name="mfe", clip_max=5.0)
    y_mae_fit, mae_head_prep = _l2_positive_head_target_prep(y_mae, head_name="mae", clip_max=4.0)
    two_stage_label_stats = {
        "train_trade_rate": float(np.mean(y_trade[train_mask])) if np.any(train_mask) else 0.0,
        "train_direction_long_rate": float(np.mean(y_dir_stage[active_train] == 1)) if np.any(active_train) else 0.0,
        "train_direction_short_rate": float(np.mean(y_dir_stage[active_train] == 0)) if np.any(active_train) else 0.0,
    }

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
    print(
        "  [L2] legacy-note: current live/train path is retained two-stage trade_gate + binary direction; "
        "old triple-gate findings are archived context only. Focus diagnostics on short-bias, holdout stability, and L1b contribution.",
        flush=True,
    )
    print(f"  [L2] will write: {artifact_path(L2_META_FILE)} | {artifact_path(L2_OUTPUT_CACHE_FILE)}", flush=True)
    log_label_baseline("l2_mfe", y_mfe[active_train], task="reg")
    log_label_baseline("l2_mae", y_mae[active_train], task="reg")
    print(
        f"  [L2] aux target prep: mfe(transform={mfe_head_prep['target_transform']}, objective={mfe_head_prep['objective']}, metric={mfe_head_prep['metric']})  "
        f"mae(transform={mae_head_prep['target_transform']}, objective={mae_head_prep['objective']}, metric={mae_head_prep['metric']})",
        flush=True,
    )
    print(
        f"  [L2] two-stage labels: train_trade_rate={two_stage_label_stats['train_trade_rate']:.3f}  "
        f"train_long_rate={two_stage_label_stats['train_direction_long_rate']:.3f}  "
        f"train_short_rate={two_stage_label_stats['train_direction_short_rate']:.3f}",
        flush=True,
    )

    rounds = _l2_boost_rounds()
    # Gate: default 120 — not tied to FAST_TRAIN_MODE (set L2_GATE_EARLY_STOPPING_ROUNDS to override).
    gate_es_rounds = _l2_early_stopping_rounds_from_env("L2_GATE_EARLY_STOPPING_ROUNDS", 120)
    aux_es_fallback = 40 if FAST_TRAIN_MODE else 120
    aux_es_base = _l2_early_stopping_rounds_from_env("L2_EARLY_STOPPING_ROUNDS", aux_es_fallback)
    direction_es_rounds = _l2_early_stopping_rounds_from_env("L2_DIRECTION_EARLY_STOPPING_ROUNDS", aux_es_base)
    side_es_rounds = _l2_early_stopping_rounds_from_env("L2_SIDE_EARLY_STOPPING_ROUNDS", direction_es_rounds)
    mfe_es_rounds = _l2_early_stopping_rounds_from_env("L2_MFE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mae_es_rounds = _l2_early_stopping_rounds_from_env("L2_MAE_EARLY_STOPPING_ROUNDS", aux_es_base)
    log_label_baseline("l2_trade_gate", y_trade[train_mask], task="cls")
    log_label_baseline("l2_direction", y_dir_stage[active_train], task="cls")
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
    direction_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
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
        "is_unbalance": True,
    }
    print(
        f"  [L2] gate: pos_rate={pr_trade:.3f}  is_unbalance=True  lr={gate_cfg['learning_rate']}  "
        f"num_leaves={gate_cfg['num_leaves']}  max_depth={gate_cfg['max_depth']}  min_child_samples={gate_cfg['min_child_samples']}  "
        f"bagging={gate_cfg['bagging_fraction']}/{gate_cfg['bagging_freq']}  early_stopping_rounds={gate_es_rounds}  "
        f"early_stop_metric=auc (first)",
        flush=True,
    )
    print(
        f"  [L2] direction: active_rows(train)={int(np.sum(active_train))}  is_unbalance=True  "
        f"lr={direction_cfg['learning_rate']}  num_leaves={direction_cfg['num_leaves']}  "
        f"max_depth={direction_cfg['max_depth']}  early_stopping_rounds={side_es_rounds}",
        flush=True,
    )
    print(
        f"  [L2] early_stopping_rounds: direction={side_es_rounds}  "
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
    X_gate_train_fit, gate_blocked_cols = _l2_project_gate_features(X_train_fit, feature_cols)
    X_gate_all, _ = _l2_project_gate_features(X, feature_cols)
    if gate_blocked_cols:
        print(
            f"  [L2] trade_gate blocks {len(gate_blocked_cols)} sign-bearing L1c/derived cols: {gate_blocked_cols}",
            flush=True,
        )
    dtrain_gate = lgb.Dataset(
        X_gate_train_fit[train_mask],
        label=y_trade[train_mask],
        weight=gate_w_train,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    dval_gate = lgb.Dataset(
        X_gate_all[val_mask],
        label=y_trade[val_mask],
        weight=gate_w_val,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    l2_outer = tqdm(
        total=4,
        desc="[L2] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    direction_model: lgb.Booster
    try:
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(gate_es_rounds, rounds, "[L2] gate", first_metric_only=True)
        try:
            gate_model = lgb.train(trade_gate_params, dtrain_gate, num_boost_round=rounds, valid_sets=[dval_gate], callbacks=cbs)
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("gate", refresh=False)
        l2_outer.update(1)

        if int(active_train.sum()) < 100 or int(active_val.sum()) < 25:
            raise RuntimeError(
                "L2: too few active rows for direction/MFE/MAE heads after strict time split. "
                f"active_train={int(active_train.sum())}, active_val={int(active_val.sum())}"
            )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(side_es_rounds, rounds, "[L2] direction")
        try:
            direction_model = lgb.train(
                direction_params,
                lgb.Dataset(
                    X_train_fit[active_train],
                    label=y_dir_stage[active_train],
                    weight=dir_weights[active_train],
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[
                    lgb.Dataset(
                        X[active_val],
                        label=y_dir_stage[active_val],
                        weight=dir_weights[active_val],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    )
                ],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("direction", refresh=False)
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

    gate_raw_all = gate_model.predict(X_gate_all).astype(np.float64)
    direction_raw_all = direction_model.predict(X).astype(np.float64)
    gate_calibrator = _fit_binary_calibrator(y_trade[val_tune_mask], gate_raw_all[val_tune_mask])
    trade_p_all = _apply_binary_calibrator(gate_raw_all, gate_calibrator).astype(np.float32)
    direction_tune_mask = val_tune_mask & (y_dir_stage >= 0)
    direction_calibrator = _fit_binary_calibrator(y_dir_stage[direction_tune_mask], direction_raw_all[direction_tune_mask])
    direction_p_cal = _apply_binary_calibrator(direction_raw_all, direction_calibrator).astype(np.float32)
    train_long_prior = float(np.mean(y_dir_stage[active_train] == 1)) if np.any(active_train) else 0.5
    target_long_prior = float(np.clip(float(os.environ.get("L2_DIRECTION_TARGET_LONG_PRIOR", "0.5")), 1e-4, 1.0 - 1e-4))
    direction_p_all = _l2_direction_prior_corrected_prob(
        direction_p_cal,
        train_long_prior=train_long_prior,
        target_long_prior=target_long_prior,
    )
    print(
        f"  [L2] direction prior correction: train_long_prior={train_long_prior:.4f}  "
        f"target_long_prior={target_long_prior:.4f}  mean_p_before={float(np.mean(direction_p_cal[active_val])):.4f}  "
        f"mean_p_after={float(np.mean(direction_p_all[active_val])):.4f}",
        flush=True,
    )
    mfe_pred = _l2_positive_head_predict(mfe_model, X, mfe_head_prep, clip_max=5.0)
    mae_pred = _l2_positive_head_predict(mae_model, X, mae_head_prep, clip_max=4.0)
    trade_threshold = _search_l2_trade_threshold(
        trade_p_all[val_tune_mask],
        target_trade_rate=_l2_target_trade_rate(),
    )
    direction_abstain_target = float(np.clip(float(os.environ.get("L2_DIRECTION_ABSTAIN_RATE", "0.10")), 0.0, 0.5))
    direction_abstain_margin = _search_l2_direction_abstain_margin(
        trade_p_all[val_tune_mask],
        direction_p_all[val_tune_mask],
        trade_threshold=float(trade_threshold),
        target_abstain_rate=direction_abstain_target,
    )
    _log_l2_two_stage_val_diagnostics(
        trade_p_all[val_report_mask],
        direction_p_all[val_report_mask],
        y_trade[val_report_mask],
        y_dir_stage[val_report_mask],
        y_decision[val_report_mask],
        trade_threshold=float(trade_threshold),
        direction_abstain_margin=float(direction_abstain_margin),
        split_label="val_report",
    )
    if test_mask.any():
        _log_l2_two_stage_val_diagnostics(
            trade_p_all[test_mask],
            direction_p_all[test_mask],
            y_trade[test_mask],
            y_dir_stage[test_mask],
            y_decision[test_mask],
            trade_threshold=float(trade_threshold),
            direction_abstain_margin=float(direction_abstain_margin),
            split_label="holdout",
        )
    decision_probs = _l2_compose_probs_from_gate_dir(trade_p_all, direction_p_all)
    size_raw = _l2_formula_size_from_context(
        frame,
        trade_p_all,
        direction_p_all,
        trade_threshold=float(trade_threshold),
    )
    size_min_std = float(np.clip(float(os.environ.get("L2_SIZE_MIN_ACTIVE_STD", "0.02")), 1e-4, 0.5))
    fit_active = val_tune_mask & (trade_p_all >= float(trade_threshold))
    std_raw = float(np.std(size_raw[fit_active])) if np.any(fit_active) else float(np.std(size_raw))
    size_mode = "formula"
    size_bins: list[float] = []
    size_levels: list[float] = []
    if std_raw < size_min_std:
        size_pred, size_bins, size_levels = _l2_bucketized_size_from_signals(
            frame,
            trade_p_all,
            direction_p_all,
            trade_threshold=float(trade_threshold),
            fit_mask=val_tune_mask,
        )
        size_mode = "bucketized_fallback"
    else:
        size_pred = size_raw
    print(
        f"  [L2] size rebuild: mode={size_mode}  active_std_raw={std_raw:.5f}  "
        f"min_active_std={size_min_std:.5f}  active_rows={int(np.sum(fit_active))}",
        flush=True,
    )
    expected_edge_all = _l2_expected_edge_from_gate_dir(
        trade_p_all,
        direction_p_all,
        size_pred,
        trade_threshold=float(trade_threshold),
        direction_strength=frame.get("l1c_direction_strength"),
        l1b_edge_proxy=frame.get("l1b_edge_pred"),
        range_mass=(
            pd.to_numeric(frame.get("l1a_regime_prob_range_conv", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(frame.get("l1a_regime_prob_range_div", 0.0), errors="coerce").fillna(0.0)
        ),
        regime_entropy=frame.get("l1a_regime_entropy"),
    )
    hard_decision_class, decision_confidence = _l2_hard_decode_prob_aligned_outputs(
        trade_p_all,
        direction_p_all,
        decision_probs,
        trade_threshold=float(trade_threshold),
        direction_abstain_margin=float(direction_abstain_margin),
    )
    _log_l2_extended_val_metrics(
        frame,
        val_report_mask,
        y_decision,
        decision_probs,
        hard_decision_class,
        y_trade == 1,
        None,
        size_pred,
        y_mfe,
        mfe_pred,
        y_mae,
        mae_pred,
        expected_edge_pred=expected_edge_all,
        true_edge=edge,
        test_mask=test_mask,
    )
    diag_branch = _l2_edge_diagnosis_branch(expected_edge_all, edge, y_decision, val_report_mask)
    fail_fast_a3 = os.environ.get("L2_FAIL_FAST_ON_A3", "0").strip().lower() in {"1", "true", "yes"}
    if diag_branch == "A3" and fail_fast_a3:
        raise RuntimeError(
            "L2 P0-A fail-fast triggered: diagnosis_branch=A3 (direction sign and edge formula look consistent, "
            "but expected_edge remains anti-informative). Continue P0-B/P1-A in parallel and open L1c quality investigation."
        )
    _log_l2_l1b_masking_audit(
        X,
        feature_cols,
        val_mask=val_report_mask,
        y_trade=y_trade,
        y_dir_stage=y_dir_stage,
        y_decision=y_decision,
        trade_model=gate_model,
        direction_model=direction_model,
        gate_calibrator=gate_calibrator,
        direction_calibrator=direction_calibrator,
        trade_threshold=float(trade_threshold),
        direction_abstain_margin=float(direction_abstain_margin),
    )
    _log_l2_l1b_gain_importance_by_group(gate_model, feature_cols, "trade_gate")
    _log_l2_l1b_gain_importance_by_group(direction_model, feature_cols, "direction")
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
    direction_model.save_model(os.path.join(MODEL_DIR, L2_DIRECTION_FILE))
    mfe_model.save_model(os.path.join(MODEL_DIR, L2_MFE_FILE))
    mae_model.save_model(os.path.join(MODEL_DIR, L2_MAE_FILE))
    model_files: dict[str, str] = {
        "trade_gate": L2_GATE_FILE,
        "direction": L2_DIRECTION_FILE,
        "mfe": L2_MFE_FILE,
        "mae": L2_MAE_FILE,
    }
    if gate_calibrator is not None:
        gate_calib_file = L2_TRADE_GATE_CALIBRATOR_FILE
        with open(os.path.join(MODEL_DIR, gate_calib_file), "wb") as f:
            pickle.dump(gate_calibrator, f)
        model_files["trade_gate_calibrator"] = gate_calib_file
    if direction_calibrator is not None:
        direction_calib_file = L2_DIRECTION_CALIBRATOR_FILE
        with open(os.path.join(MODEL_DIR, direction_calib_file), "wb") as f:
            pickle.dump(direction_calibrator, f)
        model_files["direction_calibrator"] = direction_calib_file
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
        "decision_mode": "two_stage_trade_then_direction",
        "decision_tau": tau_global,
        "decision_tau_global": tau_global,
        "decision_tau_by_state": decision_tau_by_state,
        "trade_threshold": float(trade_threshold),
        "direction_threshold": 0.5,
        "direction_abstain_margin": float(direction_abstain_margin),
        "direction_train_long_prior": float(train_long_prior),
        "direction_target_long_prior": float(target_long_prior),
        "direction_available": True,
        "direction_head_type": "binary_direction_classifier_conditioned_on_state_and_l1c",
        "confidence_semantics": "probability-aligned; confidence equals the chosen class probability under trade_gate plus binary direction decode",
        "decision_class_semantics": "first infer calibrated trade probability from the retained trade_gate, then choose long vs short from the calibrated binary direction head once the trade threshold is passed",
        "decision_abstain_semantics": "if trade gate passes but direction confidence |2p-1| is below direction_abstain_margin, abstain to neutral",
        "size_semantics": "formula_from_trade_probability_direction_strength_and_state_context",
        "expected_edge_semantics": "two-stage expected edge from trade probability, binary direction margin, formula size, L1c strength, and regime-aware shrinkage (range mass + entropy + L1b edge proxy)",
        "feature_contract_semantics": {
            "state": "L1a contributes regime, volatility, persistence, and embedding context only",
            "condition": (
                "L1b contributes tradeability, quality, novelty, and cluster context only; "
                "no fixed L1B tau constant is assumed (edge features are scale-robust and can consume dynamic tau)"
            ),
            "direction": "L1c is the only directional owner consumed by L2",
            "residual_other": "PA state and session context features remain as local execution context",
        },
        "pa_state_features": list(PA_STATE_FEATURES),
        "pa_policy_semantics": "PA state buckets expand conditional policy keys, active label geometry, and direction weighting inside the fixed live Layer2 path",
        "pa_internal_semantics": {
            "baked_into_live_path": True,
            "tau": "PA states tighten or relax active decision tau before class labels are built",
            "trade_gate": "PA-conditioned row floors define which edge examples count as active",
            "direction": "PA-conditioned context and explicit state/condition x L1c interactions tell the direction head when L1c long/short evidence is trustworthy",
        },
        "live_policy_surface": {
            "active_signal": "calibrated_trade_gate_probability",
            "active_probability": "isotonic-calibrated trade_gate output",
            "direction_signal": "calibrated binary direction probability",
            "decode_threshold": float(trade_threshold),
            "direction_abstain_margin": float(direction_abstain_margin),
        },
        "live_trade_calibration": {"type": "isotonic_trade_gate_only"},
        "direction_calibration": {
            "type": "isotonic_binary_direction" if direction_calibrator is not None else "none",
            "prior_correction": True,
            "train_long_prior": float(train_long_prior),
            "target_long_prior": float(target_long_prior),
            "abstain_rate_target": float(direction_abstain_target),
            "abstain_margin": float(direction_abstain_margin),
        },
        "l2_aux_head_target_prep": {
            "mfe": mfe_head_prep,
            "mae": mae_head_prep,
        },
        "l2_two_stage_label_stats": two_stage_label_stats,
        "policy_search": {
            "trade_threshold": float(trade_threshold),
            "target_trade_rate": float(_l2_target_trade_rate()),
            "direction_abstain_target": float(direction_abstain_target),
            "direction_abstain_margin": float(direction_abstain_margin),
        },
        "two_stage_policy": {
            "decision_mode": "two_stage_trade_then_direction",
            "trade_threshold": float(trade_threshold),
            "direction_threshold": 0.5,
            "direction_abstain_margin": float(direction_abstain_margin),
            "policy_search": {
                "trade_threshold": float(trade_threshold),
                "target_trade_rate": float(_l2_target_trade_rate()),
                "direction_abstain_target": float(direction_abstain_target),
                "direction_abstain_margin": float(direction_abstain_margin),
            },
        },
        "auxiliary_feature_semantics": "frozen L1 features, derived regime context, and explicit state/condition x L1c interactions feed one retained two-stage trade gate plus binary direction path; mfe/mae remain post-decision risk auxiliaries",
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
            "direction": int(side_es_rounds),
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
        "l2_direction_config": {
            key: value for key, value in direction_params.items() if key not in {"verbosity", "n_jobs"}
        },
        "l2_regression_config": reg_params,
        "l2_size_formula_config": {
            "trade_threshold": float(trade_threshold),
            "clip_max": 5.0,
            "inputs": ["trade_probability", "direction_margin", "l1c_direction_strength", "l1c_confidence", "l1a_vol_forecast", "l1a_state_persistence"],
            "mode": size_mode,
            "active_std_raw": float(std_raw),
            "min_active_std": float(size_min_std),
            "bucket_bins": [float(x) for x in size_bins],
            "bucket_levels": [float(x) for x in size_levels],
        },
        "rollback_criteria": {
            "p1a_abstain": {"rollback_if_gate_auc_drop_gt": 0.01},
            "p1c_l3_data_expansion": {"rollback_if_exit_auc_drop_gt": 0.02},
            "p2a_l1b_prune": {"rollback_if_sign_acc_drop_gt": 0.02},
        },
    }
    meta = attach_threshold_registry(
        meta,
        "l2",
        [
            threshold_entry(
                "trade_threshold",
                float(trade_threshold),
                category="adaptive_candidate",
                role="active trade gate cutoff",
                adaptive_hint="quantile by target trade rate",
            ),
            threshold_entry(
                "direction_abstain_margin",
                float(direction_abstain_margin),
                category="adaptive_candidate",
                role="direction abstain band",
                adaptive_hint="quantile over active direction confidence",
            ),
            threshold_entry(
                "L2_TARGET_TRADE_RATE",
                float(_l2_target_trade_rate()),
                category="adaptive_candidate",
                role="target active coverage",
            ),
            threshold_entry(
                "L2_SIZE_MIN_ACTIVE_STD",
                float(size_min_std),
                category="data_guardrail",
                role="size collapse detector",
            ),
        ],
    )
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
        "direction": direction_model,
        "mfe": mfe_model,
        "mae": mae_model,
    }
    if gate_calibrator is not None:
        bundle_models["trade_gate_calibrator"] = gate_calibrator
    if direction_calibrator is not None:
        bundle_models["direction_calibrator"] = direction_calibrator
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L2] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )
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
    mode = meta.get("decision_mode", "two_stage_trade_then_direction")
    aux_prep = meta.get("l2_aux_head_target_prep", {})
    if mode != "two_stage_trade_then_direction":
        raise RuntimeError(
            f"L2 live inference only supports decision_mode='two_stage_trade_then_direction', got {mode!r}. "
            "Retrain L2 with the retained two-stage Layer2 direction path."
        )
    two_stage_cfg = meta.get("two_stage_policy") or meta
    trade_m = models["trade_gate"]
    direction_m = models["direction"]
    X_gate, _ = _l2_project_gate_features(X, target_cols)
    trade_p = _apply_binary_calibrator(
        trade_m.predict(X_gate).astype(np.float64),
        models.get("trade_gate_calibrator"),
    ).astype(np.float32)
    direction_p = _apply_binary_calibrator(
        direction_m.predict(X).astype(np.float64),
        models.get("direction_calibrator"),
    ).astype(np.float32)
    direction_p = _l2_direction_prior_corrected_prob(
        direction_p,
        train_long_prior=float(meta.get("direction_train_long_prior", 0.5)),
        target_long_prior=float(meta.get("direction_target_long_prior", 0.5)),
    )
    trade_thr = float((two_stage_cfg or {}).get("trade_threshold", meta.get("trade_threshold", 0.5)))
    direction_abstain_margin = float(
        (two_stage_cfg or {}).get("direction_abstain_margin", meta.get("direction_abstain_margin", 0.0))
    )
    decision_probs = _l2_compose_probs_from_gate_dir(trade_p, direction_p)
    mfe_pred = _l2_positive_head_predict(models["mfe"], X, aux_prep.get("mfe"), clip_max=5.0)
    mae_pred = _l2_positive_head_predict(models["mae"], X, aux_prep.get("mae"), clip_max=4.0)
    size_cfg = meta.get("l2_size_formula_config") or {}
    size_raw = _l2_formula_size_from_context(
        frame,
        trade_p,
        direction_p,
        trade_threshold=trade_thr,
        clip_max=float(size_cfg.get("clip_max", 5.0)),
    )
    if str(size_cfg.get("mode", "formula")) == "bucketized_fallback":
        size_pred, _bins, _levels = _l2_bucketized_size_from_signals(
            frame,
            trade_p,
            direction_p,
            trade_threshold=trade_thr,
            fit_mask=None,
            bins=[float(x) for x in (size_cfg.get("bucket_bins") or [])],
            levels=[float(x) for x in (size_cfg.get("bucket_levels") or [])] or None,
        )
    else:
        size_pred = size_raw
    expected_edge = _l2_expected_edge_from_gate_dir(
        trade_p,
        direction_p,
        size_pred,
        trade_threshold=trade_thr,
        direction_strength=frame.get("l1c_direction_strength"),
        l1b_edge_proxy=frame.get("l1b_edge_pred"),
        range_mass=(
            pd.to_numeric(frame.get("l1a_regime_prob_range_conv", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(frame.get("l1a_regime_prob_range_div", 0.0), errors="coerce").fillna(0.0)
        ),
        regime_entropy=frame.get("l1a_regime_entropy"),
    )
    hard_decision_class, decision_confidence = _l2_hard_decode_prob_aligned_outputs(
        trade_p,
        direction_p,
        decision_probs,
        trade_threshold=trade_thr,
        direction_abstain_margin=direction_abstain_margin,
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
    outputs["l2_expected_edge"] = expected_edge.astype(np.float32)
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)
    return outputs

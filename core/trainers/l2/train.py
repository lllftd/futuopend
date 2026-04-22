from __future__ import annotations

import json
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
    L2_ENTRY_REGIME_COLS,
    L2_GATE_FILE,
    L2_TRADE_GATE_CALIBRATOR_FILE,
    L2_MAE_FILE,
    L2_META_FILE,
    L2_MFE_FILE,
    L2_RANGE10_FILE,
    L2_RANGE20_FILE,
    L2_RANGE_FILE,
    L2_TTP90_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
    L2_USE_VIXY,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    PA_STATE_FEATURES,
    RANGE_REGIME_INDICES,
    STATE_NAMES,
    TRAIN_END,
    VIXY_DATA_PATH,
)
from core.trainers.lgbm_utils import (
    _tqdm_stream,
    _decision_forward_range_atr_array,
    _decision_forward_range_h_atr_array,
    _decision_range_ttp90_norm_30_array,
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
from core.trainers.stack_v2_common import (
    build_stack_time_splits,
    l1_oof_mode_from_env,
    l2_oof_folds_from_env,
    l2_val_start_time,
    log_label_baseline,
    save_output_cache,
    split_mask_for_tuning_and_report,
    time_blocked_fold_masks,
)
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


def _l2_multi_horizon_enabled(df: pd.DataFrame) -> bool:
    mode = os.environ.get("L2_MULTI_HORIZON", "auto").strip().lower() or "auto"
    need = (
        "decision_forward_range_10_atr",
        "decision_forward_range_20_atr",
        "decision_forward_range_30_atr",
        "decision_range_ttp90_norm_30",
    )
    has_all = all(c in df.columns for c in need)
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode in {"1", "true", "yes", "on"} and not has_all:
        raise RuntimeError(
            "L2_MULTI_HORIZON=1 but prepared data lacks multi-horizon label columns. "
            "Re-run data_tools/label_v2.py and PREPARED_DATASET_CACHE_REBUILD=1."
        )
    return bool(has_all)


def _l2_mh_horizon_costs() -> tuple[float, float, float]:
    """Per-bar-window **hurdles** in ATR units for the trade_gate label (OR across horizons).

    Interpreting straddle economics (schematic): shorter holds face lower cumulative theta drag,
    so a smaller range can clear the bar; longer holds need a wider realized range to beat
    higher *effective* cost — so a **late** gamma burst that only shows up in the 20–30m
    envelope can still yield ``y_trade_gate=1`` without committing to a single fixed hold (e.g. 15m).
    Tune via ``L2_MH_COST_10`` / ``_20`` / ``_30``.
    """
    return (
        float(os.environ.get("L2_MH_COST_10", "2.0")),
        float(os.environ.get("L2_MH_COST_20", "3.0")),
        float(os.environ.get("L2_MH_COST_30", "3.8")),
    )


def _l2_mh_gate_sample_weights_full(
    y_trade: np.ndarray,
    r10: np.ndarray,
    r20: np.ndarray,
    r30: np.ndarray,
    c10: float,
    c20: float,
    c30: float,
) -> np.ndarray:
    """Time-aware gate weights: early breakout vs late vs chop (neg). Length = len(y_trade)."""
    n = len(y_trade)
    w = np.ones(n, dtype=np.float64)
    w_pos_early = float(os.environ.get("L2_MH_GATE_W_EARLY", "2.0"))
    w_pos_late = float(os.environ.get("L2_MH_GATE_W_LATE", "0.8"))
    w_neg = float(os.environ.get("L2_MH_GATE_W_NEG_CHOP", "3.0"))
    yt = np.asarray(y_trade, dtype=np.int64).ravel()
    r10v = np.asarray(r10, dtype=np.float64).ravel()
    r20v = np.asarray(r20, dtype=np.float64).ravel()
    r30v = np.asarray(r30, dtype=np.float64).ravel()
    finite = np.isfinite(r10v) & np.isfinite(r20v) & np.isfinite(r30v)
    early = (yt == 1) & finite & (r10v > c10)
    late = (yt == 1) & finite & ~early
    w[early] = w_pos_early
    w[late] = w_pos_late
    w[yt == 0] = w_neg
    return w


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


def _l2_project_gate_features(
    X: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    _ = feature_cols
    return X, []


# Drop from L2 without retraining upstream layers: failed L1b heads / redundant pairs.
# Override with L2_SKIP_FEATURE_HARD_DROP=1; extend via L2_EXTRA_HARD_DROP=col1,col2
L2_FEATURE_HARD_DROP_DEFAULT = frozenset(
    {
    }
)


L2_RANGE_HEAD_HARD_DROP_DEFAULT = frozenset(
    {
        "l1b_edge_pred",
        "l2_l1b_edge_over_tau",
        "l2_vol_adjusted_l1b_edge",
        "l2_edge_x_breakout",
        "l2_breakout_x_abs_edge",
        "l2_signal_strength_mean_abs",
        "l2_signal_spread_var",
    }
)


def _l2_range_head_hard_drop(feature_cols: list[str]) -> frozenset[str]:
    keep_raw = os.environ.get("L2_RANGE_HEAD_KEEP_L1B_SUPERVISED", "").strip().lower()
    if keep_raw in {"1", "true", "yes"}:
        base: set[str] = set()
    else:
        base = set(L2_RANGE_HEAD_HARD_DROP_DEFAULT)
    extra_raw = os.environ.get("L2_RANGE_HEAD_EXTRA_HARD_DROP", "").strip()
    if extra_raw:
        base |= {s.strip() for s in extra_raw.split(",") if s.strip()}
    return frozenset([c for c in feature_cols if c in base])


def _l2_feature_view_cols(feature_cols: list[str], *, hard_drop: frozenset[str]) -> list[str]:
    return [c for c in feature_cols if c not in set(hard_drop)]


def _l2_parse_protected_features() -> frozenset[str]:
    raw = (os.environ.get("L2_PROTECTED_FEATURES", "") or "").strip()
    if raw == "":
        return frozenset(
            {
                "l1b_edge_pred",
                "vixy_level_ma60_ratio",
                "pa_ctx_structure_veto",
            }
        )
    return frozenset({s.strip() for s in raw.split(",") if s.strip()})


def _l2_stability_threshold_for_feature(name: str) -> float:
    """Per-group stability gate for feature selection."""
    global_raw = (os.environ.get("L2_FEATURE_STABILITY_THRESHOLD", "") or "").strip()
    group_default_raw = (os.environ.get("L2_FEATURE_STABILITY_THRESHOLD_DEFAULT", "") or "").strip()
    l1a_raw = (os.environ.get("L2_FEATURE_STABILITY_THRESHOLD_L1A", "") or "").strip()
    vixy_raw = (os.environ.get("L2_FEATURE_STABILITY_THRESHOLD_VIXY", "") or "").strip()
    global_thr = float(np.clip(float(global_raw), 0.0, 1.0)) if global_raw else 0.50
    default_thr = float(np.clip(float(group_default_raw), 0.0, 1.0)) if group_default_raw else global_thr
    l1a_thr = float(np.clip(float(l1a_raw), 0.0, 1.0)) if l1a_raw else 0.30
    vixy_thr = float(np.clip(float(vixy_raw), 0.0, 1.0)) if vixy_raw else 0.25
    if name.startswith("l1a_"):
        return l1a_thr
    if name.startswith("vixy_"):
        return vixy_thr
    return default_thr


def _l2_parse_corr_prefix_thresholds() -> dict[str, float]:
    """Prefix keys longest-match first; ``default`` fallback."""
    raw = (os.environ.get("L2_CORR_PREFIX_THRESHOLDS", "") or "").strip()
    if not raw:
        return {
            "l1b_": 0.95,
            "vixy_": 0.90,
            "l2_": 0.85,
            "l1a_": 0.92,
            "pa_": 0.88,
            "default": 0.90,
        }
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    if "default" not in out:
        out["default"] = 0.90
    return out


def _l2_corr_threshold_for_feature_name(name: str, thr_map: dict[str, float]) -> float:
    best_thr: float | None = None
    best_len = -1
    for prefix, thr in thr_map.items():
        if prefix == "default":
            continue
        if name.startswith(prefix) and len(prefix) > best_len:
            best_len = len(prefix)
            best_thr = float(thr)
    if best_thr is not None:
        return float(np.clip(best_thr, 0.0, 0.999999))
    return float(np.clip(float(thr_map.get("default", 0.90)), 0.0, 0.999999))


def _l2_single_column_stability(col: np.ndarray) -> float:
    """Score in [0, 1]: higher = more stable across contiguous segments of train rows."""
    col = np.asarray(col, dtype=np.float64).ravel()
    n = col.size
    if n < 32:
        return 1.0
    n_w = max(4, min(16, n // 100))
    parts = np.array_split(col, n_w)
    seg_means = np.array([np.nanmean(p) for p in parts if len(p) > 0], dtype=np.float64)
    if seg_means.size < 2:
        return 1.0
    am = float(np.nanmean(np.abs(seg_means)) + 1e-12)
    cv = float(np.nanstd(seg_means) / am)
    return float(np.clip(1.0 / (1.0 + 5.0 * cv), 0.0, 1.0))


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

    improved = (os.environ.get("L2_IMPROVED_FEATURE_SELECT", "1") or "").strip().lower() not in {"0", "false", "no"}
    if not improved:
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

    protected = _l2_parse_protected_features()
    thr_map = _l2_parse_corr_prefix_thresholds()
    stab_by_j = {j: _l2_single_column_stability(xt[:, j]) for j in keep_idx}

    idx_after_stab: list[int] = []
    for j in keep_idx:
        name = feature_cols[j]
        if name in protected:
            idx_after_stab.append(j)
            continue
        stab_thr = _l2_stability_threshold_for_feature(name)
        if stab_by_j[j] >= stab_thr:
            idx_after_stab.append(j)
        else:
            dropped.append(f"{name}(unstable={stab_by_j[j]:.3f}<thr={stab_thr:.3f})")

    prot_first = [j for j in idx_after_stab if feature_cols[j] in protected]
    rest = [j for j in idx_after_stab if feature_cols[j] not in protected]
    corr_order = prot_first + rest

    if len(corr_order) < 2:
        keep = [feature_cols[j] for j in corr_order]
        return keep, dropped

    kept_after_corr: list[int] = []
    kept_cols: list[np.ndarray] = []
    for j in corr_order:
        name = feature_cols[j]
        col = xt[:, j]
        if name in protected:
            kept_after_corr.append(j)
            kept_cols.append(col)
            continue
        drop_name = None
        for prev_j, prev_col in zip(kept_after_corr, kept_cols):
            t = min(
                _l2_corr_threshold_for_feature_name(name, thr_map),
                _l2_corr_threshold_for_feature_name(feature_cols[prev_j], thr_map),
            )
            corr = np.corrcoef(col, prev_col)[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= t:
                drop_name = feature_cols[prev_j]
                break
        if drop_name is not None:
            dropped.append(f"{name}(corr~{drop_name})")
            continue
        kept_after_corr.append(j)
        kept_cols.append(col)

    keep = [feature_cols[j] for j in kept_after_corr]
    return keep, dropped


L2_OUTPUT_COLS = [
    "l2_straddle_on",
    "l2_gate_prob",
    "l2_decision_confidence",
    "l2_range_pred",
    "l2_range_pred_10",
    "l2_range_pred_20",
    "l2_pred_ttp90_norm",
    "l2_size",
    "l2_pred_mfe",
    "l2_pred_mae",
    *L2_ENTRY_REGIME_COLS,
    "l2_entry_vol",
    "l2_vol_regime",
    "l2_regime_size_mult",
    "l2_predicted_profit",
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


def _l2_bracket_atr_array(df: pd.DataFrame) -> np.ndarray:
    if "lbl_atr" in df.columns:
        atr = pd.to_numeric(df["lbl_atr"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    elif "atr_5m" in df.columns:
        atr = pd.to_numeric(df["atr_5m"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    elif "atr_1m" in df.columns:
        atr = pd.to_numeric(df["atr_1m"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    else:
        atr = np.full(len(df), 1.0, dtype=np.float64)
    atr = np.nan_to_num(atr, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(atr, 1e-4, np.inf).astype(np.float32)


def _l2_straddle_cost_atr() -> float:
    # decision_forward_range_atr = (H-bar path range in $) / (1-bar ATR at i) — often O(2–8) for H≈20.
    # Cost in the same units must be ~quantile of that range (~2.5–4), not a tiny spread like 0.3 ATR.
    return float(np.clip(float(os.environ.get("L2_STRADDLE_COST_ATR", "3.0")), 1e-6, 5.0))


def _l2_dynamic_straddle_cost_enabled() -> bool:
    return os.environ.get("L2_DYNAMIC_STRADDLE_COST", "").strip().lower() in {"1", "true", "yes", "on"}


def _l2_range_roll_cost_enabled() -> bool:
    return os.environ.get("L2_STRADDLE_RANGE_ROLL_COST", "").strip().lower() in {"1", "true", "yes", "on"}


def _l2_range_roll_cost_env_config() -> dict[str, float | int]:
    return {
        "lookback": max(100, int(os.environ.get("L2_RANGE_ROLL_LOOKBACK", "1950"))),
        "percentile": float(np.clip(float(os.environ.get("L2_RANGE_ROLL_PERCENTILE", "50")), 1.0, 99.0)),
        "floor": float(os.environ.get("L2_RANGE_ROLL_FLOOR", "2.0")),
        "cap": float(os.environ.get("L2_RANGE_ROLL_CAP", "8.0")),
    }


def _l2_range_roll_cost_array(
    df: pd.DataFrame,
    realized_range_atr: np.ndarray,
    cfg: dict[str, float | int] | None = None,
) -> np.ndarray:
    """
    Causal dynamic cost in same units as decision_forward_range_atr:
    cost[i] = quantile_p( R[i-L], ..., R[i-1] ) with R = past realized forward ranges (excludes R[i]).
    """
    cfg = cfg if cfg is not None else _l2_range_roll_cost_env_config()
    L = int(cfg["lookback"])
    p_pct = float(cfg["percentile"])
    floor = float(cfg["floor"])
    cap = float(cfg["cap"])
    r = np.asarray(realized_range_atr, dtype=np.float64).ravel()
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    minp = max(50, L // 4)
    p = p_pct / 100.0
    s = pd.Series(r)
    if "symbol" in df.columns:
        cost = s.groupby(df["symbol"].astype(str).values, sort=False).transform(
            lambda x: x.shift(1).rolling(L, min_periods=minp).quantile(p)
        )
    else:
        cost = s.shift(1).rolling(L, min_periods=minp).quantile(p)
    cost = np.asarray(cost, dtype=np.float64)
    finite_r = r[np.isfinite(r)]
    med = float(np.nanmedian(finite_r)) if finite_r.size else 3.0
    if not np.isfinite(med) or med <= 0:
        med = 3.0
    cost = np.where(np.isfinite(cost), cost, med)
    return np.clip(cost, floor, cap).astype(np.float64, copy=False)


def _l2_dynamic_straddle_cost_env_config() -> dict[str, float | int]:
    return {
        "lookback": max(20, int(os.environ.get("L2_DYNAMIC_COST_LOOKBACK", "390"))),
        "ref_quantile": float(
            np.clip(float(os.environ.get("L2_DYNAMIC_COST_REF_QUANTILE", "0.7")), 0.05, 0.95)
        ),
        "min_mult": float(np.clip(float(os.environ.get("L2_DYNAMIC_COST_MIN_MULT", "0.5")), 0.05, 2.0)),
        "max_mult": float(np.clip(float(os.environ.get("L2_DYNAMIC_COST_MAX_MULT", "3.0")), 0.5, 10.0)),
        "abs_min": float(np.clip(float(os.environ.get("L2_DYNAMIC_ABS_MIN", "0.05")), 1e-6, 5.0)),
        "abs_max": float(np.clip(float(os.environ.get("L2_DYNAMIC_ABS_MAX", "5.0")), 0.01, 20.0)),
    }


def _l2_dynamic_straddle_cost_array(
    df: pd.DataFrame,
    base: float,
    cfg: dict[str, float | int] | None = None,
) -> np.ndarray:
    """
    Per-bar straddle cost in ATR units: base * clip(atr / ref, min_mult, max_mult).
    ref = rolling ref_quantile of lbl_atr (causal), per symbol.
    """
    cfg = cfg if cfg is not None else _l2_dynamic_straddle_cost_env_config()
    w = int(cfg["lookback"])
    q = float(cfg["ref_quantile"])
    min_mult = float(cfg["min_mult"])
    max_mult = float(cfg["max_mult"])
    abs_min = float(cfg["abs_min"])
    abs_max = float(cfg["abs_max"])
    base = float(np.clip(base, 1e-6, 50.0))

    atr = _l2_bracket_atr_array(df).astype(np.float64, copy=False)
    if atr.size == 0:
        return np.zeros(0, dtype=np.float64)

    minp = max(10, w // 10)
    if "symbol" in df.columns:
        wdf = pd.DataFrame({"atr": atr, "sym": df["symbol"].astype(str).values})
        ref = wdf.groupby("sym", sort=False)["atr"].transform(
            lambda s: s.rolling(w, min_periods=minp).quantile(q)
        )
        ref = ref.to_numpy(dtype=np.float64, copy=False)
    else:
        ref = pd.Series(atr).rolling(w, min_periods=minp).quantile(q).to_numpy(dtype=np.float64, copy=False)

    med = float(np.nanmedian(atr))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    ref = np.where(np.isfinite(ref) & (ref > 1e-12), ref, np.nan)
    if "symbol" in df.columns:
        ref = (
            pd.Series(ref)
            .groupby(df["symbol"].astype(str).values)
            .transform(lambda s: s.ffill().bfill())
            .to_numpy(dtype=np.float64, copy=False)
        )
    else:
        ref = pd.Series(ref).ffill().bfill().to_numpy(dtype=np.float64, copy=False)
    ref = np.where(np.isfinite(ref) & (ref > 1e-12), ref, med)
    mult = np.clip(atr / ref, min_mult, max_mult)
    return np.clip(base * mult, abs_min, abs_max).astype(np.float64, copy=False)


def _l2_range_horizon_bars() -> int:
    return max(1, int(os.environ.get("L2_RANGE_HORIZON", "20")))


def _l2_straddle_expected_edge(
    gate_p: np.ndarray,
    range_pred: np.ndarray,
    *,
    cost_atr: float | np.ndarray,
) -> np.ndarray:
    g = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    r = np.maximum(np.asarray(range_pred, dtype=np.float64).ravel(), 0.0)
    c = np.asarray(cost_atr, dtype=np.float64).ravel()
    if c.size == 1:
        c = np.full_like(g, float(c[0]), dtype=np.float64)
    elif c.size != g.size:
        raise ValueError("cost_atr length must match gate_p or be scalar.")
    half_lost = float(np.clip(float(os.environ.get("L2_STRADDLE_INACTIVE_COST_FRAC", "0.5")), 0.0, 1.0))
    return (g * (r - c) + (1.0 - g) * (-c * half_lost)).astype(np.float32)


def _l2_straddle_predicted_profit(
    range_pred: np.ndarray,
    *,
    cost_atr: float | np.ndarray,
) -> np.ndarray:
    """Realized-vol vs implied (cost) in ATR units: max(range_pred,0) - cost. Primary straddle economics."""
    r = np.maximum(np.asarray(range_pred, dtype=np.float64).ravel(), 0.0)
    c = np.asarray(cost_atr, dtype=np.float64).ravel()
    if c.size == 1:
        c = np.full_like(r, float(c[0]), dtype=np.float64)
    elif c.size != r.size:
        raise ValueError("cost_atr length must match range_pred or be scalar.")
    return (r - c).astype(np.float32)


L2_STRADDLE_VOL_REGIME_NAMES: tuple[str, ...] = (
    "low_vol_stable",
    "low_vol_rising",
    "mid_vol",
    "high_vol_stable",
    "high_vol_falling",
)

_DEFAULT_L2_REGIME_STRADDLE_CONFIG: dict[str, dict[str, float]] = {
    "low_vol_stable": {"min_profit": 2.5, "size_mult": 0.5},
    "low_vol_rising": {"min_profit": 1.0, "size_mult": 1.5},
    "mid_vol": {"min_profit": 1.5, "size_mult": 1.0},
    "high_vol_stable": {"min_profit": 2.0, "size_mult": 0.8},
    "high_vol_falling": {"min_profit": 3.0, "size_mult": 0.3},
}


def _l2_regime_straddle_config_merged() -> dict[str, dict[str, float]]:
    cfg = {k: dict(v) for k, v in _DEFAULT_L2_REGIME_STRADDLE_CONFIG.items()}
    raw = os.environ.get("L2_REGIME_STRADDLE_CONFIG_JSON", "").strip()
    if raw:
        try:
            ovr = json.loads(raw)
            if isinstance(ovr, dict):
                for k, v in ovr.items():
                    if isinstance(v, dict) and k in cfg:
                        if "min_profit" in v:
                            cfg[k]["min_profit"] = float(v["min_profit"])
                        if "size_mult" in v:
                            cfg[k]["size_mult"] = float(v["size_mult"])
        except json.JSONDecodeError as e:
            raise RuntimeError(f"L2_REGIME_STRADDLE_CONFIG_JSON invalid JSON: {e}") from e
    return cfg


def _l2_regime_straddle_arrays() -> tuple[np.ndarray, np.ndarray]:
    cfg = _l2_regime_straddle_config_merged()
    mp = np.array([float(cfg[k]["min_profit"]) for k in L2_STRADDLE_VOL_REGIME_NAMES], dtype=np.float64)
    sm = np.array([float(cfg[k]["size_mult"]) for k in L2_STRADDLE_VOL_REGIME_NAMES], dtype=np.float64)
    return mp, sm


def _l2_fit_vol_quantiles_vol_forecast(vol: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vol, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if v.size != m.size or v.size == 0:
        return 0.33, 0.66
    vv = v[m & np.isfinite(v)]
    if vv.size < 30:
        return float(np.quantile(vv, 0.33)) if vv.size else 0.33, float(np.quantile(vv, 0.66)) if vv.size else 0.66
    return float(np.quantile(vv, 0.33)), float(np.quantile(vv, 0.66))


def _l2_vol_regime_ids(
    vol: np.ndarray,
    trend: np.ndarray,
    q33: float,
    q66: float,
    trend_eps: float,
) -> np.ndarray:
    """5 buckets from L1a vol level × vol trend (straddle-oriented)."""
    v = np.asarray(vol, dtype=np.float64).ravel()
    t = np.asarray(trend, dtype=np.float64).ravel()
    if t.size != v.size:
        raise ValueError("vol and trend length mismatch for vol regime")
    eps = float(max(float(trend_eps), 1e-12))
    n = len(v)
    ids = np.full(n, 2, dtype=np.int32)
    low = v < q33
    high = v > q66
    mid = ~low & ~high
    ids[low & (t > eps)] = 1
    ids[low & (t <= eps)] = 0
    ids[high & (t < -eps)] = 4
    ids[high & (t >= -eps)] = 3
    ids[mid] = 2
    return ids


def _l2_vol_regime_names_from_ids(ids: np.ndarray) -> np.ndarray:
    names = np.asarray(L2_STRADDLE_VOL_REGIME_NAMES, dtype=object)
    i = np.asarray(ids, dtype=np.int32).ravel().clip(0, len(L2_STRADDLE_VOL_REGIME_NAMES) - 1)
    return names[i]


def _l2_sortino_trade_search_score(
    gate_p: np.ndarray,
    pp: np.ndarray,
    thr: float | np.ndarray,
    min_edge: float,
    *,
    target_trade_rate: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (score, straddle_on, size_pred) for one gate threshold candidate."""
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    pp = np.asarray(pp, dtype=np.float64).ravel()
    thr_arr = np.asarray(thr, dtype=np.float64)
    if thr_arr.ndim == 0:
        thr_arr = np.full(gate_p.shape, float(np.clip(float(thr_arr), 0.0, 1.0)), dtype=np.float64)
    else:
        thr_arr = np.broadcast_to(np.clip(thr_arr.reshape(-1), 0.0, 1.0), gate_p.shape)
    gate_on = gate_p >= thr_arr
    straddle_on = gate_on & (pp > float(min_edge))
    size_pred = _l2_straddle_size(gate_p, np.maximum(pp, 0.0).astype(np.float32), straddle_on)
    profit_row = size_pred.astype(np.float64) * pp
    lamb = float(np.clip(float(os.environ.get("L2_SORTINO_LAMBDA", "0.25")), 0.0, 10.0))
    rate_pen = float(np.clip(float(os.environ.get("L2_SORTINO_TARGET_RATE_PENALTY", "0.35")), 0.0, 10.0))
    dd_pen = float(np.clip(float(os.environ.get("L2_SORTINO_DRAWDOWN_PENALTY", "0.10")), 0.0, 10.0))
    tail_pen = float(np.clip(float(os.environ.get("L2_SORTINO_TAIL_LOSS_PENALTY", "0.15")), 0.0, 10.0))
    pf_bonus = float(np.clip(float(os.environ.get("L2_SORTINO_PROFIT_FACTOR_BONUS", "0.05")), 0.0, 10.0))
    realized_trade_rate = float(np.mean(straddle_on))
    if lamb <= 0.0:
        score = float(np.mean(profit_row))
    else:
        if not np.any(straddle_on):
            score = -1e18
        else:
            pr = profit_row[straddle_on]
            mean_p = float(np.mean(pr))
            dn = pr[pr < 0]
            dstd = float(np.sqrt(np.mean(dn * dn))) if dn.size else 0.0
            eq = np.cumsum(pr)
            peak = np.maximum.accumulate(eq)
            drawdowns = peak - eq
            max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0
            p05 = float(np.quantile(pr, 0.05)) if pr.size else 0.0
            tail_loss = float(max(0.0, -p05))
            gross_pos = float(np.sum(pr[pr > 0.0]))
            gross_neg = float(np.sum(-pr[pr < 0.0]))
            profit_factor = gross_pos / max(gross_neg, 1e-9)
            pf_term = pf_bonus * np.clip(profit_factor, 0.0, 5.0)
            score = mean_p - lamb * dstd - dd_pen * max_dd - tail_pen * tail_loss + pf_term
    score -= rate_pen * abs(realized_trade_rate - float(target_trade_rate))
    return score, straddle_on, size_pred


def _l2_warn_trade_search_shape(records: list[dict[str, float]], *, chosen_trade_rate: float) -> None:
    if not records:
        return
    trade_rate = np.asarray([float(r["realized_trade_rate"]) for r in records], dtype=np.float64)
    score = np.asarray([float(r["score"]) for r in records], dtype=np.float64)
    if trade_rate.size < 8 or not np.all(np.isfinite(trade_rate)) or not np.all(np.isfinite(score)):
        return
    if np.std(trade_rate) < 1e-9 or np.std(score) < 1e-9:
        return
    corr = float(np.corrcoef(trade_rate, score)[0, 1])
    if np.isfinite(corr) and corr < -0.80 and chosen_trade_rate < 0.06:
        print(
            f"  [L2] WARNING: threshold objective appears over-conservative (corr(score,trade_rate)={corr:.3f}, "
            f"chosen_trade_rate={chosen_trade_rate:.3f} < 0.06). Consider lowering drawdown/tail penalties.",
            flush=True,
        )


def _l2_straddle_size(
    gate_p: np.ndarray,
    expected_edge: np.ndarray,
    straddle_on: np.ndarray,
) -> np.ndarray:
    g = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    ee = np.clip(np.asarray(expected_edge, dtype=np.float64).ravel(), 0.0, None)
    on = np.asarray(straddle_on, dtype=bool).ravel()
    raw = g * ee
    if not np.any(on):
        return np.zeros_like(raw, dtype=np.float32)
    std = float(np.std(raw[on]))
    den = std + 1e-6
    z = raw / den
    mx = float(np.max(z[on]))
    out = np.zeros_like(z, dtype=np.float64)
    if mx > 1e-12:
        out[on] = np.clip(z[on] / mx, 0.0, 1.0)
    return out.astype(np.float32)


def _l2_build_bracket_plan(
    df: pd.DataFrame,
    *,
    offset_atr: float,
    tp_atr: float,
    sl_atr: float,
    max_hold: int,
) -> dict[str, np.ndarray]:
    close = _l2_series_numeric_or_zero(df, "close").to_numpy(dtype=np.float64, copy=False)
    atr = _l2_bracket_atr_array(df).astype(np.float64)
    off = np.full(len(df), float(np.clip(offset_atr, 0.0, 5.0)), dtype=np.float32)
    tp = np.full(len(df), float(np.clip(tp_atr, 0.01, 20.0)), dtype=np.float32)
    sl = np.full(len(df), float(np.clip(sl_atr, 0.01, 20.0)), dtype=np.float32)
    mh = np.full(len(df), int(np.clip(max_hold, 1, 512)), dtype=np.int32)
    buy = (close + atr * off.astype(np.float64)).astype(np.float32)
    sell = (close - atr * off.astype(np.float64)).astype(np.float32)
    return {
        "l2_bracket_buy_trigger": buy,
        "l2_bracket_sell_trigger": sell,
        "l2_bracket_offset_atr": off,
        "l2_bracket_tp_atr": tp,
        "l2_bracket_sl_atr": sl,
        "l2_bracket_max_hold": mh,
    }


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
    l1b_edge = _frame_col_or_default("l1b_edge_pred", 0.0)
    l1b_bo = _frame_col_or_default("l1b_breakout_quality", 0.0)
    vol = _frame_col_or_default("l1a_vol_forecast", 1.0)
    persistence = _frame_col_or_default("l1a_state_persistence", 0.0)
    strength_norm = np.clip(np.abs(l1b_edge) / max(float(clip_max), 1e-3), 0.0, 1.0)
    vol_damp = 1.0 / (1.0 + np.clip(vol, 0.0, 5.0))
    stability_boost = 0.60 + 0.40 * np.clip(persistence, 0.0, 1.0)
    size = (
        active_strength
        * (0.35 + 0.65 * direction_margin)
        * (0.25 + 0.75 * strength_norm)
        * (0.50 + 0.50 * np.clip(l1b_bo, 0.0, 1.0))
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
    bo_proxy = _l2_series_numeric_or_zero(frame, "l1b_breakout_quality").to_numpy(dtype=np.float64, copy=False)
    edge_proxy = _l2_series_numeric_or_zero(frame, "l1b_edge_pred").to_numpy(dtype=np.float64, copy=False)
    edge_norm = _quantile_rescale_01(np.abs(edge_proxy), fit_mask=fit_mask, q_low=0.05, q_high=0.95).astype(np.float64)
    bo_norm = _quantile_rescale_01(np.abs(bo_proxy), fit_mask=fit_mask, q_low=0.05, q_high=0.95).astype(np.float64)
    score = 0.45 * active_strength + 0.30 * dir_conf + 0.15 * np.clip(bo_norm, 0.0, 1.0) + 0.10 * edge_norm
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


def _l2_train_exclude_regime_ids_from_env() -> list[int]:
    """L1a vol-regime argmax ids (0..4) withheld from LGBM fit. Default ``none`` (fit on all). Comma-separated ids."""
    raw = os.environ.get("L2_TRAIN_EXCLUDE_REGIME_IDS", "none").strip()
    if raw.lower() in ("none", "off", "-"):
        return []
    out: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if x.isdigit() or (x.startswith("-") and x[1:].isdigit()):
            out.append(int(x))
    return out


def _l2_apply_expected_edge_regime_blacklist(expected_edge: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Zero ``expected_edge`` where argmax L1a regime id is listed in ``L2_EXPECTED_EDGE_ZERO_REGIME_IDS``."""
    raw = os.environ.get("L2_EXPECTED_EDGE_ZERO_REGIME_IDS", "").strip()
    if not raw or raw.lower() in {"none", "off"}:
        return expected_edge
    ids: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if x.isdigit() or (x.startswith("-") and x[1:].isdigit()):
            ids.append(int(x))
    if not ids:
        return expected_edge
    rid = np.argmax(frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64), axis=1)
    out = np.asarray(expected_edge, dtype=np.float32).copy()
    sel = np.isin(rid, np.asarray(ids, dtype=np.int64))
    out[sel] = 0.0
    if os.environ.get("L2_LOG_REGIME_EDGE_ZERO", "1").strip().lower() not in {"0", "false", "no"}:
        print(
            f"  [L2] L2_EXPECTED_EDGE_ZERO_REGIME_IDS={sorted(set(ids))}: "
            f"zeroed expected_edge on {int(np.sum(sel))} / {len(out)} rows",
            flush=True,
        )
    return out


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
    direction_abstain_margin: float = 0.0,
) -> np.ndarray:
    trade = np.asarray(trade_p, dtype=np.float64).ravel()
    direction = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    size = np.asarray(size_pred, dtype=np.float64).ravel()
    if np.any(size < 0) and os.environ.get("L2_EXPECTED_EDGE_WARN_NEG_SIZE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        print(
            f"  [L2-expected-edge] warning: negative size_pred share={float(np.mean(size < 0)):.6f}",
            flush=True,
        )
    thr = np.asarray(trade_threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(len(trade), float(thr), dtype=np.float64)
    else:
        thr = np.broadcast_to(thr.reshape(-1), trade.shape)
    active_strength = np.clip((trade - thr) / np.maximum(1.0 - thr, 1e-3), 0.0, 1.0)
    dir_margin = (2.0 * direction) - 1.0
    apply_abstain = os.environ.get("L2_EXPECTED_EDGE_APPLY_DIRECTION_ABSTAIN", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    abstain_thr = float(np.clip(direction_abstain_margin, 0.0, 0.49)) if apply_abstain else 0.0
    margin_floor = float(os.environ.get("L2_EXPECTED_EDGE_DIR_MARGIN_FLOOR", "1e-3"))
    margin_floor = float(np.clip(margin_floor, 0.0, 1.0))
    dir_mag_for_edge = np.abs(dir_margin) if margin_floor <= 0.0 else np.maximum(np.abs(dir_margin), margin_floor)
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
        * dir_mag_for_edge
        * np.maximum(strength, 1e-3)
        * (0.35 + 0.65 * edge_proxy_mag)
        * regime_shrink
    )
    out = np.clip(edge, -5.0, 5.0).astype(np.float32)
    if abstain_thr > 0.0:
        trade_on = trade >= thr
        dir_conf = np.abs(dir_margin)
        zero = trade_on & (dir_conf <= abstain_thr)
        if np.any(zero):
            out = out.copy()
            out[zero] = 0.0
    if os.environ.get("L2_EDGE_SIGN_TRACE", "0").strip().lower() in {"1", "true", "yes"}:
        max_n = int(np.clip(float(os.environ.get("L2_EDGE_SIGN_TRACE_MAX_N", "200000")), 1000, 10_000_000))
        n_tot = int(out.size)
        idx = np.arange(n_tot)
        if n_tot > max_n:
            idx = np.random.default_rng(0).choice(n_tot, size=max_n, replace=False)
            note = f"subsample n={max_n} of {n_tot}"
        else:
            note = f"full n={n_tot}"
        o = np.asarray(out, dtype=np.float64).ravel()[idx]
        dm = np.asarray(dir_margin, dtype=np.float64).ravel()[idx]
        sz = np.asarray(size, dtype=np.float64).ravel()[idx]
        st = np.asarray(strength, dtype=np.float64).ravel()[idx]
        tr = np.asarray(trade, dtype=np.float64).ravel()[idx]
        th = np.asarray(thr, dtype=np.float64).ravel()[idx]
        dc = np.abs(dm)
        trade_on_s = tr >= th
        dir_active_s = np.ones(len(idx), dtype=bool) if not apply_abstain else dc > abstain_thr
        sign_m = trade_on_s & dir_active_s & (np.abs(dm) > 1e-12) & (np.abs(o) > 1e-12)
        agree = float(np.mean(np.sign(o[sign_m]) == np.sign(dm[sign_m]))) if np.any(sign_m) else float("nan")
        as_ = np.asarray(active_strength, dtype=np.float64).ravel()[idx]
        dme = np.asarray(dir_mag_for_edge, dtype=np.float64).ravel()[idx]
        epm = np.asarray(edge_proxy_mag, dtype=np.float64).ravel()[idx]
        rs = np.asarray(regime_shrink, dtype=np.float64).ravel()[idx]
        prod_mag = as_ * sz * dme * np.maximum(st, 1e-3) * (0.35 + 0.65 * epm) * rs
        pm = prod_mag[sign_m]
        n_neg_pm = int(np.sum(pm < 0)) if pm.size else 0
        pm_min = float(np.min(pm)) if pm.size else float("nan")
        l1b_agree = float("nan")
        if l1b_edge_proxy is not None and np.any(sign_m):
            l1b_raw = np.asarray(l1b_edge_proxy, dtype=np.float64).ravel()[idx]
            l1b_agree = float(np.mean(np.sign(dm[sign_m] * l1b_raw[sign_m]) == np.sign(dm[sign_m])))
        print(
            f"\n  [L2-edge-sign-trace] {note}: sign(out)==sign(dir_margin) "
            f"on trade_on&dir_active&|out|>0&|dm|>0: {agree:.6f} (n={int(np.sum(sign_m))})",
            flush=True,
        )
        print(
            f"    size<0: {float(np.mean(sz < 0)):.6f}  strength<0: {float(np.mean(st < 0)):.6f}  "
            f"prod_mag (nonneg magnitude path): min={pm_min:.6f}  count(pm<0)={n_neg_pm}  "
            f"(do not compare sign(prod_mag) to sign(dm): prod_mag excludes sign(dir_margin))",
            flush=True,
        )
        if l1b_edge_proxy is not None:
            print(
                f"    sign(dm*l1b_raw)==sign(dm) same mask: {l1b_agree:.6f} (formula uses clip(|l1b|)/5; raw for audit)",
                flush=True,
            )

    return out


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


def _search_l2_trade_threshold_economic(
    gate_p: np.ndarray,
    predicted_profit: np.ndarray,
    *,
    target_trade_rate: float = 0.10,
    min_trade_rate: float | None = None,
    max_trade_rate: float | None = None,
    min_edge: float = 0.0,
    state_keys: np.ndarray | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    pp = np.asarray(predicted_profit, dtype=np.float64).ravel()
    if pp.size != gate_p.size:
        raise ValueError("predicted_profit must align with gate_p")
    if gate_p.size == 0:
        return 0.35, {
            "objective": "l2_trade_search",
            "best_score": float("nan"),
            "target_trade_rate": float(target_trade_rate),
            "min_trade_rate": float(target_trade_rate),
            "max_trade_rate": float(target_trade_rate),
            "realized_trade_rate": 0.0,
            "realized_gate_rate": 0.0,
            "candidate_count": 0,
            "constraint_mode": "fallback_empty",
        }
    if min_trade_rate is None or max_trade_rate is None:
        band_lo, band_hi = _l2_trade_rate_search_min_max()
        if min_trade_rate is None:
            min_trade_rate = band_lo
        if max_trade_rate is None:
            max_trade_rate = band_hi
    target = float(np.clip(target_trade_rate, min_trade_rate, max_trade_rate))
    grid_n = int(np.clip(int(os.environ.get("L2_TRADE_THR_SEARCH_GRID", "101")), 21, 401))
    q_grid = np.linspace(0.01, 0.99, grid_n)
    cand_from_q = np.quantile(gate_p, q_grid)
    cand_from_rates = [
        float(np.quantile(gate_p, 1.0 - rate))
        for rate in sorted({float(min_trade_rate), float(target), float(max_trade_rate)})
    ]
    candidates = np.unique(np.clip(np.concatenate([cand_from_q, np.asarray(cand_from_rates, dtype=np.float64)]), 0.0, 1.0))

    best_any: dict[str, float] | None = None
    best_in_band: dict[str, float] | None = None
    all_records: list[dict[str, float]] = []
    lamb_s = float(np.clip(float(os.environ.get("L2_SORTINO_LAMBDA", "0.25")), 0.0, 10.0))
    rate_pen_s = float(np.clip(float(os.environ.get("L2_SORTINO_TARGET_RATE_PENALTY", "0.35")), 0.0, 10.0))
    dd_pen_s = float(np.clip(float(os.environ.get("L2_SORTINO_DRAWDOWN_PENALTY", "0.10")), 0.0, 10.0))
    tail_pen_s = float(np.clip(float(os.environ.get("L2_SORTINO_TAIL_LOSS_PENALTY", "0.15")), 0.0, 10.0))
    pf_bonus_s = float(np.clip(float(os.environ.get("L2_SORTINO_PROFIT_FACTOR_BONUS", "0.05")), 0.0, 10.0))
    obj_lbl = "mean(size*predicted_profit)" if lamb_s <= 0.0 else "mean(active profit) - λ*downside_rms(losses)"
    if dd_pen_s > 0.0:
        obj_lbl = f"{obj_lbl} - dd_pen*max_drawdown"
    if tail_pen_s > 0.0:
        obj_lbl = f"{obj_lbl} - tail_pen*P5(loss)"
    if pf_bonus_s > 0.0:
        obj_lbl = f"{obj_lbl} + pf_bonus*profit_factor"
    if rate_pen_s > 0.0:
        obj_lbl = f"{obj_lbl} - rate_pen*|trade_rate-target|"
    print(
        f"\n  [L2] trade_threshold search (range-first gate + pp); objective={obj_lbl}  "
        f"λ={lamb_s:.3f}  rate_pen={rate_pen_s:.3f}",
        flush=True,
    )
    for thr in candidates:
        score, straddle_on, _ = _l2_sortino_trade_search_score(
            gate_p, pp, float(thr), min_edge, target_trade_rate=target
        )
        gate_on = gate_p >= float(thr)
        realized_trade_rate = float(np.mean(straddle_on))
        realized_gate_rate = float(np.mean(gate_on))
        record = {
            "threshold": float(thr),
            "score": score,
            "realized_trade_rate": realized_trade_rate,
            "realized_gate_rate": realized_gate_rate,
            "active_count": float(np.sum(straddle_on)),
        }
        all_records.append(record)
        if best_any is None or score > float(best_any["score"]) + 1e-12 or (
            abs(score - float(best_any["score"])) <= 1e-12
            and abs(realized_trade_rate - target) < abs(float(best_any["realized_trade_rate"]) - target)
        ):
            best_any = record
        if min_trade_rate <= realized_trade_rate <= max_trade_rate:
            if best_in_band is None or score > float(best_in_band["score"]) + 1e-12 or (
                abs(score - float(best_in_band["score"])) <= 1e-12
                and abs(realized_trade_rate - target) < abs(float(best_in_band["realized_trade_rate"]) - target)
            ):
                best_in_band = record

    dynamic_threshold_map: dict[str, float] = {}
    use_dynamic = os.environ.get("L2_DYNAMIC_THRESHOLD_REGIME", "1").strip().lower() not in {"0", "false", "no"}
    min_rows = int(np.clip(int(os.environ.get("L2_DYNAMIC_THRESHOLD_MIN_ROWS", "500")), 20, 5000))
    max_offset = float(np.clip(float(os.environ.get("L2_DYNAMIC_THRESHOLD_MAX_OFFSET", "0.010")), 0.001, 0.050))
    raw_steps = os.environ.get("L2_DYNAMIC_THRESHOLD_STEPS", "-0.010,-0.006,-0.003,0.0,0.003,0.006,0.010").strip()
    if use_dynamic and state_keys is not None and len(state_keys) == len(gate_p):
        parsed_steps: list[float] = []
        for s in raw_steps.split(","):
            ss = s.strip()
            if not ss:
                continue
            try:
                parsed_steps.append(float(ss))
            except ValueError:
                continue
        steps = sorted({float(np.clip(s, -max_offset, max_offset)) for s in parsed_steps} | {0.0})
        chosen_base = best_in_band if best_in_band is not None else best_any
        assert chosen_base is not None
        thr_row = np.full(len(gate_p), float(chosen_base["threshold"]), dtype=np.float64)
        keys = np.asarray(state_keys, dtype=object)
        for key in np.unique(keys):
            m = keys == key
            if int(np.sum(m)) < min_rows:
                continue
            local_best_step = 0.0
            local_best_score = -1e18
            for step in steps:
                trial_thr = np.clip(thr_row[m] + step, 0.0, 1.0)
                trial_full = thr_row.copy()
                trial_full[m] = trial_thr
                trial_score, _, _ = _l2_sortino_trade_search_score(
                    gate_p,
                    pp,
                    trial_full,
                    min_edge,
                    target_trade_rate=target,
                )
                if trial_score > local_best_score:
                    local_best_score = float(trial_score)
                    local_best_step = float(step)
            if abs(local_best_step) > 1e-9:
                thr_row[m] = np.clip(thr_row[m] + local_best_step, 0.0, 1.0)
            dynamic_threshold_map[str(key)] = float(local_best_step)
        score_dyn, on_dyn, _ = _l2_sortino_trade_search_score(
            gate_p,
            pp,
            thr_row,
            min_edge,
            target_trade_rate=target,
        )
        rec_dyn = {
            "threshold": float(chosen_base["threshold"]),
            "score": float(score_dyn),
            "realized_trade_rate": float(np.mean(on_dyn)),
            "realized_gate_rate": float(np.mean(gate_p >= thr_row)),
            "active_count": float(np.sum(on_dyn)),
        }
        in_band_dyn = min_trade_rate <= rec_dyn["realized_trade_rate"] <= max_trade_rate
        if in_band_dyn and (best_in_band is None or rec_dyn["score"] > float(best_in_band["score"]) + 1e-12):
            best_in_band = rec_dyn
        elif best_in_band is None and best_any is not None and rec_dyn["score"] > float(best_any["score"]) + 1e-12:
            best_any = rec_dyn

    chosen = best_in_band if best_in_band is not None else best_any
    assert chosen is not None
    hard_floor = float(np.clip(float(os.environ.get("L2_TRADE_RATE_HARD_FLOOR", "0.06")), 0.01, 0.50))
    if float(chosen["realized_trade_rate"]) < hard_floor:
        eligible = [r for r in all_records if float(r["realized_trade_rate"]) >= hard_floor]
        if eligible:
            chosen = max(
                eligible,
                key=lambda r: (float(r["score"]), -abs(float(r["realized_trade_rate"]) - target)),
            )
            print(
                f"  [L2] trade-rate hard floor applied: floor={hard_floor:.3f} "
                f"selected_threshold={float(chosen['threshold']):.4f}",
                flush=True,
            )
    _l2_warn_trade_search_shape(all_records, chosen_trade_rate=float(chosen["realized_trade_rate"]))
    print(
        f"    target_trade_rate={target:.3f}  band=[{float(min_trade_rate):.3f}, {float(max_trade_rate):.3f}]  "
        f"candidates={len(candidates)}",
        flush=True,
    )
    print(
        f"  [L2] selected trade_threshold={float(chosen['threshold']):.4f}  "
        f"score={float(chosen['score']):.6f}  realized_trade_rate={float(chosen['realized_trade_rate']):.3f}  "
        f"gate_rate={float(chosen['realized_gate_rate']):.3f}  active_n={int(chosen['active_count'])}",
        flush=True,
    )
    info: dict[str, float | int | str] = {
        "objective": obj_lbl,
        "sortino_lambda": float(lamb_s),
        "sortino_target_rate_penalty": float(rate_pen_s),
        "sortino_drawdown_penalty": float(dd_pen_s),
        "sortino_tail_penalty": float(tail_pen_s),
        "sortino_profit_factor_bonus": float(pf_bonus_s),
        "best_score": float(chosen["score"]),
        "target_trade_rate": float(target),
        "min_trade_rate": float(min_trade_rate),
        "max_trade_rate": float(max_trade_rate),
        "realized_trade_rate": float(chosen["realized_trade_rate"]),
        "realized_gate_rate": float(chosen["realized_gate_rate"]),
        "candidate_count": int(len(candidates)),
        "trade_rate_hard_floor": float(hard_floor),
        "dynamic_threshold_states": int(len(dynamic_threshold_map)),
        "dynamic_threshold_steps": raw_steps,
        "dynamic_threshold_max_offset": float(max_offset),
        "dynamic_threshold_map": json.dumps(dynamic_threshold_map, sort_keys=True),
        "constraint_mode": "in_band" if best_in_band is not None else "fallback_best_score",
        "l2_trade_rate_band_env": (os.environ.get("L2_TRADE_RATE_BAND", "") or "").strip(),
    }
    return float(chosen["threshold"]), info


def _l2_threshold_row_from_state_map(
    base_threshold: float,
    state_keys: np.ndarray | None,
    dynamic_threshold_map_json: str | None,
) -> np.ndarray | None:
    if state_keys is None:
        return None
    keys = np.asarray(state_keys, dtype=object).ravel()
    if keys.size == 0:
        return None
    out = np.full(keys.shape, float(np.clip(base_threshold, 0.0, 1.0)), dtype=np.float64)
    raw = (dynamic_threshold_map_json or "").strip()
    if not raw:
        return out
    try:
        mp = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(mp, dict):
        return out
    for k, step in mp.items():
        try:
            s = float(step)
        except (TypeError, ValueError):
            continue
        m = keys == str(k)
        if np.any(m):
            out[m] = np.clip(out[m] + s, 0.0, 1.0)
    return out


def _l2_edge_diagnosis_branch(
    e_pred: np.ndarray,
    e_true: np.ndarray,
    y_decision: np.ndarray,
    mask: np.ndarray,
    y_trade: np.ndarray | None = None,
) -> str:
    sm = np.asarray(mask, dtype=bool)
    yp = np.asarray(e_pred, dtype=np.float64).ravel()
    yt = np.asarray(e_true, dtype=np.float64).ravel()
    yd = np.asarray(y_decision, dtype=np.int64).ravel()
    if y_trade is not None:
        active = sm & (np.asarray(y_trade, dtype=np.int64).ravel() == 1)
    else:
        active = sm & (yd != 1)
    if not np.any(active):
        return "A2"
    nz = active & (np.abs(yp) > 1e-6)
    if int(np.sum(nz)) >= 30:
        sign_acc = float(np.mean(np.sign(yp[nz]) == np.sign(yt[nz])))
        corr_active = pearson_corr(yp[nz], yt[nz])
    else:
        sign_acc = float(np.mean(np.sign(yp[active]) == np.sign(yt[active])))
        corr_active = pearson_corr(yp[active], yt[active])
    if sign_acc < 0.35 and np.isfinite(corr_active) and corr_active <= 0.0:
        return "A1"
    if sign_acc < 0.45 and np.isfinite(corr_active) and corr_active < 0.05:
        return "A3"
    return "A2"


def _log_l2_gate_bar_edge_audit(
    *,
    val_report_mask: np.ndarray,
    y_decision: np.ndarray,
    y_trade: np.ndarray,
    hard_decision: np.ndarray,
    trade_p: np.ndarray,
    trade_threshold: float,
    direction_p: np.ndarray,
    true_edge: np.ndarray,
) -> None:
    """P(te>0) / mean(te) when gate fires and direction matches truth vs not; gate-off shadow direction."""
    if os.environ.get("L2_GATE_EDGE_AUDIT", "1").strip().lower() in {"0", "false", "no"}:
        return
    vm = np.asarray(val_report_mask, dtype=bool)
    yd = np.asarray(y_decision, dtype=np.int64).ravel()
    yt = np.asarray(y_trade, dtype=np.int64).ravel()
    hd = np.asarray(hard_decision, dtype=np.int64).ravel()
    tp = np.asarray(trade_p, dtype=np.float64).ravel()
    dp = np.clip(np.asarray(direction_p, dtype=np.float64).ravel(), 0.0, 1.0)
    te = np.asarray(true_edge, dtype=np.float64).ravel()
    thr = float(trade_threshold)
    gate_on = tp >= thr
    pred_dir_active = hd != 1
    dir_ok = ((hd == 0) & (yd == 0)) | ((hd == 2) & (yd == 2))
    true_long = yd == 0
    true_short = yd == 2
    shadow_ok = ((dp > 0.5) & true_long) | ((dp < 0.5) & true_short)

    def _stat(label: str, m: np.ndarray) -> None:
        m = np.asarray(m, dtype=bool)
        n = int(np.sum(m))
        if n == 0:
            print(f"    {label}: n=0", flush=True)
            return
        sub = te[m]
        print(
            f"    {label}: n={n}  P(te>0)={float(np.mean(sub > 0)):.4f}  mean(te)={float(np.mean(sub)):.4f}",
            flush=True,
        )

    print("\n  [L2] gate × direction × true_edge (val_report; decision_net_edge_atr)", flush=True)
    _stat("gate_on & pred_dir_active & dir_correct", vm & gate_on & pred_dir_active & dir_ok)
    _stat("gate_on & pred_dir_active & dir_wrong", vm & gate_on & pred_dir_active & ~dir_ok)
    _stat("gate_off & y_trade=1 & shadow_dir_p matches y_decision", vm & ~gate_on & (yt == 1) & shadow_ok)
    _stat("gate_off & y_trade=1 & ~shadow_ok", vm & ~gate_on & (yt == 1) & ~shadow_ok)


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


def _l2_flip_would_help(sign_acc: float) -> str:
    if not np.isfinite(sign_acc):
        return "unclear"
    if sign_acc < 0.40:
        return "yes"
    if sign_acc > 0.60:
        return "no"
    return "unclear"


def _log_l2_direction_diagnostics(
    frame: pd.DataFrame,
    val_report_mask: np.ndarray,
    *,
    dir_p: np.ndarray,
    trade_p: np.ndarray,
    y_trade: np.ndarray,
    y_decision: np.ndarray,
    hard_decision: np.ndarray,
    expected_edge: np.ndarray | None,
    true_edge: np.ndarray | None,
    direction_abstain_margin: float,
    train_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Direction head diagnostics on val_report ∩ {y_trade==1}. Env L2_DIRECTION_DIAG=0 disables."""
    if os.environ.get("L2_DIRECTION_DIAG", "1").strip().lower() in {"0", "false", "no"}:
        return {}

    n_min = int(np.clip(float(os.environ.get("L2_DIRECTION_DIAG_N_MIN", "30")), 5.0, 1_000_000.0))
    n_min_slice = int(np.clip(float(os.environ.get("L2_DIRECTION_DIAG_N_MIN_SLICE", "15")), 3.0, 1_000_000.0))

    vm = np.asarray(val_report_mask, dtype=bool)
    y_tr = np.asarray(y_trade, dtype=np.int64).ravel()
    yd = np.asarray(y_decision, dtype=np.int64).ravel()
    dp = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    tp = np.asarray(trade_p, dtype=np.float64).ravel()
    hd = np.asarray(hard_decision, dtype=np.int64).ravel()

    base = vm & (y_tr == 1)
    n_ta = int(np.sum(base))
    out: dict[str, Any] = {"n_true_active_val_report": n_ta}
    if n_ta < n_min:
        print(
            f"\n  [L2-dir-diag] skipped: true_active n={n_ta} < L2_DIRECTION_DIAG_N_MIN={n_min}",
            flush=True,
        )
        return out

    margin = float(np.clip(direction_abstain_margin, 0.0, 0.49))
    dir_conf = np.abs(2.0 * dp - 1.0)
    abst_share = float(np.mean(dir_conf[base] <= margin)) if n_ta else float("nan")

    print("\n  [L2-dir-diag] === Block A: Sign confusion on true_active (val_report) ===", flush=True)
    print(
        f"    n_true_active={n_ta}  mean(trade_p|subset)={float(np.mean(tp[base])):.4f}  "
        f"abstain_zone_share(|2p-1|<={margin:.4f})={abst_share:.4f}",
        flush=True,
    )
    a = int(np.sum((hd == 0) & (yd == 0) & base))
    b = int(np.sum((hd == 0) & (yd == 2) & base))
    c = int(np.sum((hd == 0) & (yd == 1) & base))
    d = int(np.sum((hd == 2) & (yd == 0) & base))
    e = int(np.sum((hd == 2) & (yd == 2) & base))
    f = int(np.sum((hd == 2) & (yd == 1) & base))
    g = int(np.sum((hd == 1) & (yd == 0) & base))
    h = int(np.sum((hd == 1) & (yd == 2) & base))
    i_ = int(np.sum((hd == 1) & (yd == 1) & base))
    print("    pred\\true     long    short    (flat)", flush=True)
    print(f"    long          {a:<7d} {b:<7d} {c:<7d}", flush=True)
    print(f"    short         {d:<7d} {e:<7d} {f:<7d}", flush=True)
    print(f"    (abstain)     {g:<7d} {h:<7d} {i_:<7d}", flush=True)
    denom = a + b + d + e
    sign_acc_overall = float((a + e) / denom) if denom > 0 else float("nan")
    denom_l = a + d
    denom_s = b + e
    sign_acc_long = float(a / denom_l) if denom_l > 0 else float("nan")
    sign_acc_short = float(e / denom_s) if denom_s > 0 else float("nan")
    asym = (
        float(sign_acc_long - sign_acc_short)
        if np.isfinite(sign_acc_long) and np.isfinite(sign_acc_short)
        else float("nan")
    )
    print(
        f"    sign_acc_overall={sign_acc_overall:.4f}  sign_acc_long={sign_acc_long:.4f}  "
        f"sign_acc_short={sign_acc_short:.4f}  asymmetry={asym:+.4f}",
        flush=True,
    )
    out["block_a"] = {
        "confusion": {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h, "i": i_},
        "sign_acc_overall": sign_acc_overall,
        "sign_acc_long": sign_acc_long,
        "sign_acc_short": sign_acc_short,
        "asymmetry": asym,
    }

    if expected_edge is not None and true_edge is not None:
        ee = np.asarray(expected_edge, dtype=np.float64).ravel()
        te = np.asarray(true_edge, dtype=np.float64).ravel()
        cov_ee = float(np.mean(np.abs(ee[base]) > 1e-6)) if n_ta else float("nan")
        edge_sign_acc = float(np.mean(np.sign(ee[base]) == np.sign(te[base])))
        m_ee_nz = base & (np.abs(ee) > 1e-6)
        edge_te_nz = (
            float(np.mean(np.sign(ee[m_ee_nz]) == np.sign(te[m_ee_nz]))) if np.any(m_ee_nz) else float("nan")
        )
        print(f"    expected_edge |ee|>1e-6 coverage on true_active: {cov_ee:.4f}", flush=True)
        print(
            f"    sign(ee)==sign(te) true_active (includes ee==0 abstain; sign(0) vs sign(te)): {edge_sign_acc:.4f}",
            flush=True,
        )
        print(f"    sign(ee)==sign(te) true_active where |ee|>0 only: {edge_te_nz:.4f}", flush=True)
        out["block_a"]["expected_edge_abs_coverage_true_active"] = cov_ee
        out["block_a"]["edge_sign_acc_vs_true_edge_raw"] = edge_sign_acc
        out["block_a"]["edge_sign_acc_vs_true_edge_ee_nonzero"] = edge_te_nz
        dir_margin_all = 2.0 * dp - 1.0
        m_nz = base & (np.abs(ee) > 1e-6)
        edge_vs_dm = (
            float(np.mean(np.sign(ee[m_nz]) == np.sign(dir_margin_all[m_nz]))) if np.any(m_nz) else float("nan")
        )
        true_dir_sign = np.where(yd == 0, 1.0, np.where(yd == 2, -1.0, 0.0))
        m_disc_nz = base & np.isin(yd, (0, 2)) & (np.abs(ee) > 1e-6)
        edge_vs_true_dir = (
            float(np.mean(np.sign(ee[m_disc_nz]) == true_dir_sign[m_disc_nz])) if np.any(m_disc_nz) else float("nan")
        )
        hard_dir_sign = np.where(hd == 0, 1.0, np.where(hd == 2, -1.0, 0.0))
        m_hard_nz = base & np.isin(hd, (0, 2)) & (np.abs(ee) > 1e-6)
        edge_vs_hard = (
            float(np.mean(np.sign(ee[m_hard_nz]) == hard_dir_sign[m_hard_nz])) if np.any(m_hard_nz) else float("nan")
        )
        print(
            f"    cross_check sign(expected_edge)==sign(2*dir_p-1) [|ee|>0 on true_active]: {edge_vs_dm:.4f}",
            flush=True,
        )
        print(
            f"    cross_check sign(expected_edge)==sign(true direction class) [|ee|>0, y_decision long/short]: "
            f"{edge_vs_true_dir:.4f}  (align with discrete label; compare to sign_acc_overall)",
            flush=True,
        )
        print(
            f"    cross_check sign(expected_edge)==sign(hard decision class) [|ee|>0]: {edge_vs_hard:.4f}",
            flush=True,
        )
        out["block_a"]["edge_sign_acc_vs_dir_margin_nz"] = edge_vs_dm
        out["block_a"]["edge_sign_acc_vs_true_direction_nz"] = edge_vs_true_dir
        out["block_a"]["edge_sign_acc_vs_hard_decision_nz"] = edge_vs_hard

    m_long = base & (yd == 0)
    m_short = base & (yd == 2)
    m_flat = base & (yd == 1)
    print("\n  [L2-dir-diag] === Block B: dir_p distribution by true direction ===", flush=True)
    block_b: dict[str, Any] = {}
    for label, m, name in (
        ("long", m_long, "true=long"),
        ("short", m_short, "true=short"),
        ("flat", m_flat, "true=flat"),
    ):
        if not np.any(m):
            print(f"    {name}  (n=0):  —", flush=True)
            block_b[label] = {"n": 0}
            continue
        xv = dp[m]
        pcts = np.percentile(xv, [5, 25, 50, 75, 95]).tolist()
        print(
            f"    {name}  (n={int(np.sum(m))}):  mean(dir_p)={float(np.mean(xv)):.4f}  std={float(np.std(xv)):.4f}  "
            f"median={float(np.median(xv)):.4f}",
            flush=True,
        )
        print(f"                        pcts[5/25/50/75/95] = {np.round(pcts, 4).tolist()}", flush=True)
        block_b[label] = {
            "n": int(np.sum(m)),
            "mean": float(np.mean(xv)),
            "std": float(np.std(xv)),
            "median": float(np.median(xv)),
            "pcts_5_95": [float(x) for x in pcts],
        }
    sep = float("nan")
    if np.any(m_long) and np.any(m_short):
        sep = float(np.mean(dp[m_long]) - np.mean(dp[m_short]))
    if sep < -0.01:
        act = "negative (reversed vs ideal)"
    elif sep > 0.01:
        act = "positive"
    else:
        act = "near-zero"
    print(
        f"    separation: mean(dir_p|long) - mean(dir_p|short) = {sep:.4f}  "
        f"(expected positive; actual: {act})",
        flush=True,
    )
    block_b["separation"] = sep
    out["block_b"] = block_b

    rid = np.argmax(frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64), axis=1)
    print("\n  [L2-dir-diag] === Block C: Sign accuracy by L1a regime (true_active) ===", flush=True)
    print(
        "    regime  n    sign_acc   mean(dp|L)  mean(dp|S)  separation  flip_would_help",
        flush=True,
    )
    block_c: list[dict[str, Any]] = []
    for r in range(NUM_REGIME_CLASSES):
        m = base & (rid == r)
        n_r = int(np.sum(m))
        if n_r == 0:
            continue
        tiny = n_r < n_min_slice
        ml = m & (yd == 0)
        ms_ = m & (yd == 2)
        denom_r = int(np.sum((hd != 1) & m))
        correct = int(np.sum(((hd == 0) & (yd == 0)) & m) + np.sum(((hd == 2) & (yd == 2)) & m))
        s_acc = float(correct / denom_r) if denom_r > 0 else float("nan")
        mnl = float(np.mean(dp[ml])) if np.any(ml) else float("nan")
        mns = float(np.mean(dp[ms_])) if np.any(ms_) else float("nan")
        sep_r = float(mnl - mns) if np.isfinite(mnl) and np.isfinite(mns) else float("nan")
        fw = _l2_flip_would_help(s_acc)
        twarn = "  [tiny]" if tiny else ""
        print(
            f"    {r:<7d} {n_r:<4d} {s_acc:.4f}     {mnl:.4f}      {mns:.4f}      {sep_r:+.4f}      {fw}{twarn}",
            flush=True,
        )
        block_c.append(
            {
                "regime": r,
                "name": STATE_NAMES.get(r, str(r)),
                "n": n_r,
                "sign_acc": s_acc,
                "mean_dir_p_long": mnl,
                "mean_dir_p_short": mns,
                "separation": sep_r,
                "flip_would_help": fw,
                "tiny_slice": tiny,
            }
        )
    out["block_c"] = block_c

    print("\n  [L2-dir-diag] === Block D: Sign accuracy by time (true_active, monthly) ===", flush=True)
    print("    month      n    sign_acc   separation   flip_would_help", flush=True)
    idx_all = np.flatnonzero(base)
    ts_sub = pd.to_datetime(frame.iloc[idx_all]["time_key"])
    months_sub = ts_sub.dt.to_period("M").astype(str).to_numpy()
    block_d: list[dict[str, Any]] = []
    for month in pd.unique(months_sub):
        pick = months_sub == month
        idx_m = idx_all[pick]
        mrows = np.zeros(len(yd), dtype=bool)
        mrows[idx_m] = True
        n_m = int(idx_m.size)
        if n_m == 0:
            continue
        ml = mrows & (yd == 0)
        ms_ = mrows & (yd == 2)
        denom_m = int(np.sum((hd != 1) & mrows))
        correct = int(np.sum(((hd == 0) & (yd == 0)) & mrows) + np.sum(((hd == 2) & (yd == 2)) & mrows))
        s_acc = float(correct / denom_m) if denom_m > 0 else float("nan")
        mnl = float(np.mean(dp[ml])) if np.any(ml) else float("nan")
        mns = float(np.mean(dp[ms_])) if np.any(ms_) else float("nan")
        sep_m = float(mnl - mns) if np.isfinite(mnl) and np.isfinite(mns) else float("nan")
        fw = _l2_flip_would_help(s_acc)
        tiny = n_m < n_min_slice
        twarn = "  [tiny]" if tiny else ""
        print(
            f"    {str(month):<10} {n_m:<4d} {s_acc:.4f}     {sep_m:+.4f}       {fw}{twarn}",
            flush=True,
        )
        block_d.append(
            {
                "month": str(month),
                "n": n_m,
                "sign_acc": s_acc,
                "separation": sep_m,
                "flip_would_help": fw,
                "tiny_slice": tiny,
            }
        )
    out["block_d"] = block_d

    print("\n  [L2-dir-diag] === Block E: Direction calibration (true_active, binned dir_p) ===", flush=True)
    edges = np.linspace(0.0, 1.0, 11)
    block_e_rows: list[dict[str, Any]] = []
    print("    bin(dir_p)       n     frac_true_long   ideal    gap", flush=True)
    for j in range(10):
        lo, hi = float(edges[j]), float(edges[j + 1])
        if j < 9:
            mbin = base & (dp >= lo) & (dp < hi)
        else:
            mbin = base & (dp >= lo) & (dp <= hi)
        n_b = int(np.sum(mbin))
        if n_b == 0:
            br = f"[{lo:.2f}, {hi:.2f})" if j < 9 else f"[{lo:.2f}, {hi:.2f}]"
            print(f"    {br:<16} 0     —", flush=True)
            block_e_rows.append({"lo": lo, "hi": hi, "n": 0})
            continue
        frac_tl = float(np.mean(yd[mbin] == 0))
        ideal = 0.5 * (lo + hi)
        gap = frac_tl - ideal
        br = f"[{lo:.2f}, {hi:.2f})" if j < 9 else f"[{lo:.2f}, {hi:.2f}]"
        print(f"    {br:<16} {n_b:<5d} {frac_tl:.4f}           {ideal:.2f}     {gap:+.4f}", flush=True)
        block_e_rows.append(
            {"lo": lo, "hi": hi, "n": n_b, "frac_true_long": frac_tl, "ideal": ideal, "gap": gap}
        )

    slope = float("nan")
    intercept = float("nan")
    sub_longshort = base & np.isin(yd, (0, 2))
    if int(np.sum(sub_longshort)) >= 20 and len(np.unique(yd[sub_longshort])) >= 2:
        Xl = dp[sub_longshort].reshape(-1, 1)
        yl = (yd[sub_longshort] == 0).astype(np.int32)
        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(Xl, yl)
            slope = float(clf.coef_[0, 0])
            intercept = float(clf.intercept_[0])
        except ValueError:
            pass
    print(
        f"    logistic(dir_p -> P(true long)): slope={slope:.4f}  intercept={intercept:.4f}  "
        f"(strong negative slope often indicates sign flip vs calibrated prob)",
        flush=True,
    )
    out["block_e"] = {"bins": block_e_rows, "logistic_slope": slope, "logistic_intercept": intercept}

    if train_mask is not None:
        tm = np.asarray(train_mask, dtype=bool)
        base_tr = tm & (y_tr == 1)
        n_tr = int(np.sum(base_tr))
        if n_tr >= n_min:
            a_t = int(np.sum((hd == 0) & (yd == 0) & base_tr))
            b_t = int(np.sum((hd == 0) & (yd == 2) & base_tr))
            d_t = int(np.sum((hd == 2) & (yd == 0) & base_tr))
            e_t = int(np.sum((hd == 2) & (yd == 2) & base_tr))
            denom_tr = a_t + b_t + d_t + e_t
            sign_acc_train = float((a_t + e_t) / denom_tr) if denom_tr > 0 else float("nan")
            print(
                f"\n  [L2-dir-diag] train true_active: n={n_tr}  sign_acc_overall={sign_acc_train:.4f}  "
                f"(val_report {sign_acc_overall:.4f})",
                flush=True,
            )
            out["train_true_active"] = {"n": n_tr, "sign_acc_overall": sign_acc_train}
        else:
            print(
                f"\n  [L2-dir-diag] train true_active n={n_tr} < {n_min} (skip train sign_acc)",
                flush=True,
            )

    return out


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
        pred_nonzero = vm & (np.abs(e_pred) > 1e-6)
        active_pred_nonzero = active_pred & pred_nonzero
        corr_active = pearson_corr(e_pred[active_true], e_true[active_true]) if active_true.any() else float("nan")
        corr_pred_active_nonzero = (
            pearson_corr(e_pred[active_pred_nonzero], e_true[active_pred_nonzero]) if active_pred_nonzero.any() else float("nan")
        )
        sign_acc_true = (
            float(np.mean(np.sign(e_pred[active_true]) == np.sign(e_true[active_true])))
            if active_true.any()
            else float("nan")
        )
        active_true_nz = active_true & (np.abs(e_pred) > 1e-6)
        sign_acc_true_nz = (
            float(np.mean(np.sign(e_pred[active_true_nz]) == np.sign(e_true[active_true_nz])))
            if active_true_nz.any()
            else float("nan")
        )
        corr_true_nz = (
            pearson_corr(e_pred[active_true_nz], e_true[active_true_nz]) if int(np.sum(active_true_nz)) >= 10 else float("nan")
        )
        cov_true_nz = float(np.mean(np.abs(e_pred[active_true]) > 1e-6)) if active_true.any() else float("nan")
        sign_acc_pred = (
            float(np.mean(np.sign(e_pred[active_pred]) == np.sign(e_true[active_pred])))
            if active_pred.any()
            else float("nan")
        )
        sign_acc_pred_nonzero = (
            float(np.mean(np.sign(e_pred[active_pred_nonzero]) == np.sign(e_true[active_pred_nonzero])))
            if active_pred_nonzero.any()
            else float("nan")
        )
        print(
            f"    expected_edge: corr_all={corr_all:.4f}  corr_active(yd≠neutral)={corr_active:.4f}  "
            f"corr(yd≠neutral & |ee|>0)={corr_true_nz:.4f}  "
            f"sign_acc_raw={sign_acc_true:.4f}  sign_acc_|ee|>0={sign_acc_true_nz:.4f}  "
            f"cov_|ee|>0={cov_true_nz:.3f}  (y_trade==1 matches yd≠neutral here)",
            flush=True,
        )
        print(
            f"    expected_edge (pred-active & |ee|>0): corr={corr_pred_active_nonzero:.4f}  "
            f"sign_acc={sign_acc_pred_nonzero:.4f}  "
            f"coverage_of_val_report={float(np.mean(active_pred_nonzero[vm])):.3f}  "
            f"active_pred_nonzero_n={int(np.sum(active_pred_nonzero))}",
            flush=True,
        )
        print(
            f"    expected_edge pred_active (any ee): sign_acc={sign_acc_pred:.4f}  pred_active_share={float(np.mean(active_pred[vm])):.3f}",
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
        acc_br = sign_acc_true_nz if int(np.sum(active_true_nz)) >= 30 else sign_acc_true
        corr_br = corr_true_nz if int(np.sum(active_true_nz)) >= 30 else corr_active
        if np.isfinite(acc_br):
            if acc_br < 0.35 and np.isfinite(corr_br) and corr_br <= 0.0:
                branch = "A1"
            elif acc_br < 0.45 and np.isfinite(corr_br) and corr_br < 0.05:
                branch = "A3"
        print(
            f"    [L2][P0-A] diagnosis_branch={branch}  (uses sign/corr on yd≠neutral & |ee|>0 when n≥30)  "
            "A1=likely_sign_flip  A2=formula_scaling_issue  A3=upstream_signal_quality_issue",
            flush=True,
        )
        if np.isfinite(sign_acc_pred_nonzero) and np.isfinite(sign_acc_true_nz):
            print(
                f"    [L2][P0-A] dual-view: sign_yd_dir_|ee|>0={sign_acc_true_nz:.4f}  "
                f"sign_pred_active_|ee|>0={sign_acc_pred_nonzero:.4f}",
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
        "l2_vol_adjusted_l1b_edge",
        "l2_edge_x_breakout",
        "l2_breakout_x_abs_edge",
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
    if name in {"l2_bull_mass_x_abs_edge", "l2_range_mass_x_abs_edge"}:
        return "l1b_deterministic_semantic"
    if name.startswith("l1b_"):
        return "l1b_deterministic_semantic"
    return "non_l1b"


def _l2_unsup_descriptor_feature_names(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if _l2_l1b_importance_bucket(c) == "l1b_unsup_descriptor"]


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
        return np.array(X, dtype=np.float32, copy=True), []
    out = np.array(X, dtype=np.float32, copy=True)
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
    direction_raw_source: np.ndarray,
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
            np.asarray(direction_raw_source, dtype=np.float64),
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
    return float(np.clip(float(os.environ.get("L2_TARGET_TRADE_RATE", "0.10")), 0.05, 0.15))


def _l2_trade_rate_search_min_max() -> tuple[float, float]:
    """Realized straddle trade-rate band for gate threshold search.

    ``L2_TRADE_RATE_BAND=lo,hi`` overrides ``L2_TRADE_THR_SEARCH_MIN`` / ``MAX`` when set.
    Defaults widen the feasible band (higher signal frequency vs legacy 5–12%).
    """
    raw = (os.environ.get("L2_TRADE_RATE_BAND", "") or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 2:
            lo = float(parts[0])
            hi = float(parts[1])
            if lo > hi:
                lo, hi = hi, lo
            return float(np.clip(lo, 0.01, 0.50)), float(np.clip(hi, 0.01, 0.50))
    lo = float(os.environ.get("L2_TRADE_THR_SEARCH_MIN", "0.07"))
    hi = float(os.environ.get("L2_TRADE_THR_SEARCH_MAX", "0.15"))
    return float(np.clip(lo, 0.01, 0.50)), float(np.clip(hi, 0.01, 0.50))


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
            "max_depth": 6,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "lambda_l1": 0.10,
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


def _l2_mfe_sample_weights(y_mfe_fit: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    y = np.asarray(y_mfe_fit, dtype=np.float64).ravel()
    mask = np.asarray(fit_mask, dtype=bool).ravel()
    w = np.ones_like(y, dtype=np.float64)
    if y.size == 0 or not np.any(mask):
        return w.astype(np.float32)
    fit_vals = y[mask & np.isfinite(y)]
    if fit_vals.size == 0:
        return w.astype(np.float32)
    hard_thr_raw = os.environ.get("L2_MFE_UNDEREST_PENALTY_THRESHOLD", "").strip()
    hard_thr = float(hard_thr_raw) if hard_thr_raw else None
    pos_eps = float(np.clip(float(os.environ.get("L2_MFE_POS_EPS", "1e-8")), 0.0, 0.5))
    q = float(np.clip(float(os.environ.get("L2_MFE_UNDEREST_PENALTY_Q", "0.50")), 0.0, 0.98))
    conditional = os.environ.get("L2_MFE_QUANTILE_CONDITIONAL", "1").strip().lower() in {"1", "true", "yes"}
    pos_mask = mask & np.isfinite(y) & (y > pos_eps)
    pos_vals = y[pos_mask]
    if conditional and pos_vals.size >= 16:
        src = pos_vals
    else:
        src = fit_vals
    if hard_thr is not None:
        high = float(max(hard_thr, pos_eps))
    else:
        high = float(np.quantile(src, q))
    extra = float(np.clip(float(os.environ.get("L2_MFE_UNDEREST_PENALTY_MULT", "2.0")), 1.0, 5.0))
    positive_only = os.environ.get("L2_MFE_BOOST_ON_POSITIVE_ONLY", "1").strip().lower() in {"1", "true", "yes"}
    base_mask = pos_mask if positive_only else (mask & np.isfinite(y))
    boosted_mask = base_mask & (y > max(high, pos_eps))
    w[boosted_mask] = extra
    return w.astype(np.float32)


def _l2_log_mfe_weighting_diagnostics(y_mfe_fit: np.ndarray, weights: np.ndarray, *, active_mask: np.ndarray) -> None:
    y = np.asarray(y_mfe_fit, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    m = np.asarray(active_mask, dtype=bool).ravel() & np.isfinite(y) & np.isfinite(w)
    if not np.any(m):
        return
    yv = y[m]
    wv = w[m]
    boosted = wv > 1.000001
    boosted_share = float(np.mean(boosted))
    q = float(np.clip(float(os.environ.get("L2_MFE_UNDEREST_PENALTY_Q", "0.50")), 0.0, 0.98))
    pos_eps = float(np.clip(float(os.environ.get("L2_MFE_POS_EPS", "1e-8")), 0.0, 0.5))
    pos_share = float(np.mean(yv > pos_eps))
    boosted_pos_share = float(np.mean(boosted[yv > pos_eps])) if np.any(yv > pos_eps) else 0.0
    print(
        f"  [L2] MFE asymmetric weighting: boosted_share={boosted_share:.3f} "
        f"boosted_share_pos={boosted_pos_share:.3f}  q={q:.2f}  pos_share={pos_share:.3f} "
        f"mult={float(np.clip(float(os.environ.get('L2_MFE_UNDEREST_PENALTY_MULT', '2.0')), 1.0, 5.0)):.2f}",
        flush=True,
    )
    if boosted_pos_share < 0.20 or boosted_pos_share > 0.80:
        print(
            "  [L2] WARNING: MFE boosted_share_pos outside preferred [0.20, 0.80]. "
            "Tune L2_MFE_UNDEREST_PENALTY_Q.",
            flush=True,
        )
    p = np.quantile(yv, [0.25, 0.75])
    low = yv <= float(p[0])
    high = yv >= float(p[1])
    if np.any(low) and np.any(high):
        print(
            f"  [L2] MFE weight buckets: low_q25_mean_w={float(np.mean(wv[low])):.3f}  "
            f"high_q75_mean_w={float(np.mean(wv[high])):.3f}",
            flush=True,
        )


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
            "l1a_is_warm",
        ]
    )
    cols.extend([c for c in merged.columns if c.startswith("l1a_market_embed_")])
    return [c for c in cols if c in merged.columns]


def _l2_condition_input_cols(merged: pd.DataFrame) -> list[str]:
    allow = [
        "l1b_edge_pred",
        "l1b_novelty_score",
        "l1b_regime_change_score",
        "l1b_breakout_quality",
        "l1b_edge_confidence",
        "l1b_edge_trend",
        "l1b_breakout_strength",
        "l1b_cluster_diversity",
    ]
    allow.extend([c for c in merged.columns if c.startswith("l1b_cluster_prob_")])
    return [c for c in allow if c in merged.columns]


def _l2_l1b_condition_extras(merged: pd.DataFrame) -> pd.DataFrame:
    """L1b-side scalars built in L2 (not from L1b cache): confidence, trend, breakout z, cluster diversity."""
    out = pd.DataFrame(index=merged.index)
    n = len(merged)
    edge_ser = pd.to_numeric(merged.get("l1b_edge_pred", pd.Series(0.0, index=merged.index)), errors="coerce").fillna(0.0)
    edge = edge_ser.to_numpy(dtype=np.float64)
    out["l1b_edge_confidence"] = np.abs(np.tanh(edge)).astype(np.float32)

    if "symbol" in merged.columns:
        out["l1b_edge_trend"] = (
            merged.groupby("symbol", sort=False)["l1b_edge_pred"]
            .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).diff(5))
            .fillna(0.0)
            .astype(np.float32)
        )
    else:
        out["l1b_edge_trend"] = edge_ser.diff(5).fillna(0.0).astype(np.float32)

    def _bq_z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        m = s.rolling(390, min_periods=20).mean()
        sd = s.rolling(390, min_periods=20).std() + 1e-6
        return ((s - m) / sd).fillna(0.0)

    if "symbol" in merged.columns and "l1b_breakout_quality" in merged.columns:
        out["l1b_breakout_strength"] = (
            merged.groupby("symbol", sort=False)["l1b_breakout_quality"].transform(_bq_z).astype(np.float32)
        )
    else:
        out["l1b_breakout_strength"] = np.zeros(n, dtype=np.float32)

    cluster_cols = [c for c in merged.columns if c.startswith("l1b_cluster_prob_")]
    if cluster_cols:
        cp = merged[cluster_cols].to_numpy(dtype=np.float64, copy=False)
        cp = np.nan_to_num(cp, nan=0.0)
        cp = cp / np.maximum(cp.sum(axis=1, keepdims=True), 1e-12)
        k = cp.shape[1]
        h = -np.sum(np.clip(cp, 1e-12, 1.0) * np.log(np.clip(cp, 1e-12, 1.0)), axis=1)
        div = h / max(float(np.log(max(k, 2))), 1e-12)
        out["l1b_cluster_diversity"] = np.clip(div, 0.0, 1.0).astype(np.float32)
    else:
        out["l1b_cluster_diversity"] = np.zeros(n, dtype=np.float32)
    return out


def _l2_direction_input_cols(merged: pd.DataFrame) -> list[str]:
    """Direction-pressure proxies for straddle asymmetry (magnitude only, no directional label leak)."""
    allow = [
        "l2_directional_pressure",
        "l2_skew_imbalance",
        "l2_order_flow_asymmetry",
        "l2_recent_momentum_abs",
        "l2_direction_vol_pressure",
    ]
    return [c for c in allow if c in merged.columns]


def _l2_residual_input_cols(residual: pd.DataFrame) -> list[str]:
    return [c for c in residual.columns if c not in {"symbol", "time_key"}]


def _legacy_l1a_direction_cols(merged: pd.DataFrame) -> list[str]:
    legacy_named = {"l1a_bull_convergence", "l1a_bear_convergence"}
    return [c for c in merged.columns if c.startswith("l1a_dir_") or c in legacy_named]


def _l2_np_float32_col(merged: pd.DataFrame, name: str) -> np.ndarray:
    """Numeric column as float32 vector; zeros if missing (``DataFrame.get`` default would be a scalar)."""
    if name not in merged.columns:
        return np.zeros(len(merged), dtype=np.float32)
    return pd.to_numeric(merged[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def _l2_series_numeric_or_zero(frame: pd.DataFrame, name: str) -> pd.Series:
    """Like ``frame[name]`` numeric coerced; all zeros if column absent (avoids ``.get(..., 0.0)`` scalar bug)."""
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def _merge_l2_upstream_outputs(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        df[["symbol", "time_key"]]
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l1b_outputs, on=["symbol", "time_key"], how="left")
    )
    legacy_l1a_dir_cols = _legacy_l1a_direction_cols(merged)
    if legacy_l1a_dir_cols:
        merged = merged.drop(columns=legacy_l1a_dir_cols)
        print(
            f"  [L2] ignoring {len(legacy_l1a_dir_cols)} legacy L1a direction cols: {legacy_l1a_dir_cols}",
            flush=True,
        )
    return merged


def _l2_fit_derived_feature_stats(
    merged: pd.DataFrame,
    fit_mask: np.ndarray | None,
) -> dict[str, float]:
    fit = np.asarray(fit_mask, dtype=bool).ravel() if fit_mask is not None else np.ones(len(merged), dtype=bool)
    l1b_edge_f = _l2_np_float32_col(merged, "l1b_edge_pred").astype(np.float64)
    fit_vals = np.abs(l1b_edge_f[fit & np.isfinite(l1b_edge_f)])
    if fit_vals.size == 0:
        fit_vals = np.abs(l1b_edge_f[np.isfinite(l1b_edge_f)])
    tau_q = float(np.clip(float(os.environ.get("L2_L1B_TAU_PROXY_Q", "0.70")), 0.50, 0.95))
    tau_proxy = float(np.quantile(fit_vals, tau_q)) if fit_vals.size else 0.05
    tau_proxy = float(max(tau_proxy, 1e-3))
    return {
        "l1b_edge_abs_tau_proxy": tau_proxy,
        "l1b_edge_abs_tau_proxy_q": tau_q,
        "fit_row_count": int(np.sum(fit)),
    }


def _derived_l2_feature_frame(
    merged: pd.DataFrame,
    *,
    derived_feature_stats: dict[str, Any] | None = None,
    price_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    novelty = _l2_np_float32_col(merged, "l1b_novelty_score")
    regime_change = _l2_np_float32_col(merged, "l1b_regime_change_score")
    vol = _l2_np_float32_col(merged, "l1a_vol_forecast")
    out["l1b_novelty_x_vol"] = (novelty * vol).astype(np.float32)
    out["l1b_regime_change_x_entropy"] = (regime_change * out["l1a_regime_entropy"].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
    out["l1b_unsup_pressure"] = (out["l1b_cluster_top2_gap"].to_numpy(dtype=np.float32, copy=False) - novelty).astype(np.float32)
    l1b_edge_f = _l2_np_float32_col(merged, "l1b_edge_pred")
    l1b_breakout_f = _l2_np_float32_col(merged, "l1b_breakout_quality")
    stats = dict(derived_feature_stats or {})
    tau_proxy = float(stats.get("l1b_edge_abs_tau_proxy", 0.05))
    tau_eff = np.full(len(l1b_edge_f), max(tau_proxy, 1e-3), dtype=np.float64)
    out["l2_l1b_edge_over_tau"] = (l1b_edge_f.astype(np.float64) / np.maximum(tau_eff, 1e-3)).astype(np.float32)
    vol_f = _l2_np_float32_col(merged, "l1a_vol_forecast")
    out["l2_vol_adjusted_l1b_edge"] = (l1b_edge_f / np.maximum(vol_f, 1e-4)).astype(np.float32)
    abs_edge = np.abs(l1b_edge_f.astype(np.float64))
    out["l2_edge_x_breakout"] = (l1b_edge_f * l1b_breakout_f).astype(np.float32)
    out["l2_breakout_x_abs_edge"] = (l1b_breakout_f.astype(np.float64) * abs_edge).astype(np.float32)
    out["l2_signal_strength_mean_abs"] = abs_edge.astype(np.float32)
    sig_stack = np.column_stack([l1b_edge_f.astype(np.float64), l1b_breakout_f.astype(np.float64)])
    out["l2_signal_spread_var"] = np.var(sig_stack, axis=1).astype(np.float32)

    if len(regime_cols) >= NUM_REGIME_CLASSES:
        rp = merged[regime_cols].to_numpy(dtype=np.float64, copy=False)
        rp = np.nan_to_num(rp, nan=0.0)
        rp = rp / np.maximum(rp.sum(axis=1, keepdims=True), 1e-12)
        # Vol lifecycle (5): range-like mass = compress + mean-revert; remainder = breakout/trending/exhaust.
        range_mass = rp[:, RANGE_REGIME_INDICES].sum(axis=1).astype(np.float32)
        mask = np.ones(rp.shape[1], dtype=bool)
        mask[np.asarray(RANGE_REGIME_INDICES, dtype=int)] = False
        bull_mass = rp[:, mask].sum(axis=1).astype(np.float32)
        out["l2_bull_mass_x_abs_edge"] = (bull_mass.astype(np.float64) * abs_edge).astype(np.float32)
        out["l2_range_mass_x_abs_edge"] = (range_mass.astype(np.float64) * abs_edge).astype(np.float32)
    else:
        out["l2_bull_mass_x_abs_edge"] = np.zeros(n_m, dtype=np.float32)
        out["l2_range_mass_x_abs_edge"] = np.zeros(n_m, dtype=np.float32)

    vol_rf = _l2_np_float32_col(merged, "l1a_vol_forecast")
    vol_tr = _l2_np_float32_col(merged, "l1a_vol_trend")
    edge_rf = _l2_np_float32_col(merged, "l1b_edge_pred")
    den_v = np.maximum(vol_rf.astype(np.float64), 1e-4) * (1.0 + np.abs(vol_tr.astype(np.float64)))
    out["volatility_regime_adjusted"] = (edge_rf.astype(np.float64) / den_v).astype(np.float32)
    out["l2_directional_pressure"] = np.abs(edge_rf.astype(np.float64)).astype(np.float32)
    skew_signed = os.environ.get("L2_DIRECTION_SKEW_SIGNED", "0").strip().lower() in {"1", "true", "yes"}
    out["l2_skew_imbalance"] = (
        vol_tr.astype(np.float64) if skew_signed else np.abs(vol_tr.astype(np.float64))
    ).astype(np.float32)
    if price_df is not None and len(price_df) == len(merged) and "pa_state_always_in_bias" in price_df.columns:
        of_raw = pd.to_numeric(price_df["pa_state_always_in_bias"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        # Fallback for offline/ad-hoc callers that already carry PA state on merged.
        of_raw = _l2_np_float32_col(merged, "pa_state_always_in_bias").astype(np.float64)
    out["l2_order_flow_asymmetry"] = np.abs(of_raw).astype(np.float32)
    out["l2_direction_vol_pressure"] = (
        np.abs(edge_rf.astype(np.float64)) * (1.0 + np.clip(np.abs(vol_tr.astype(np.float64)), 0.0, 3.0))
    ).astype(np.float32)
    if price_df is not None and len(price_df) == len(merged) and "close" in price_df.columns:
        close = pd.to_numeric(price_df["close"], errors="coerce").fillna(0.0)
        if "symbol" in price_df.columns:
            mom = close.groupby(price_df["symbol"].astype(str), sort=False).transform(lambda s: s.pct_change(5))
        else:
            mom = close.pct_change(5)
        out["l2_recent_momentum_abs"] = np.abs(mom).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    else:
        out["l2_recent_momentum_abs"] = np.zeros(n_m, dtype=np.float32)

    if price_df is not None and len(price_df) == len(merged):
        pa_stack: list[np.ndarray] = []
        for c in PA_STATE_FEATURES:
            if c in price_df.columns:
                pa_stack.append(pd.to_numeric(price_df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        if pa_stack:
            pa_mean = np.mean(np.stack(pa_stack, axis=0), axis=0)
            ent = out["l1a_regime_entropy"].to_numpy(dtype=np.float64)
            out["pa_l1a_regime_interaction"] = (pa_mean * ent).astype(np.float32)
        else:
            out["pa_l1a_regime_interaction"] = np.zeros(n_m, dtype=np.float32)
    else:
        out["pa_l1a_regime_interaction"] = np.zeros(n_m, dtype=np.float32)
    return out


def _l2_attach_vixy_l1b_interaction(merged: pd.DataFrame, *, vixy_enabled: bool) -> None:
    """VIXY level × L1b edge (after VIXY columns are on merged). Zeros when VIXY off or missing."""
    n = len(merged)
    edge = pd.to_numeric(merged.get("l1b_edge_pred"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if vixy_enabled and "vixy_zscore_390" in merged.columns:
        vz = pd.to_numeric(merged["vixy_zscore_390"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        merged["vixy_l1b_interaction"] = (vz * edge).astype(np.float32)
    else:
        merged["vixy_l1b_interaction"] = np.zeros(n, dtype=np.float32)


def _l2_derived_group_column_names(derived: pd.DataFrame, merged: pd.DataFrame) -> list[str]:
    dir_like = {"l2_skew_imbalance", "l2_order_flow_asymmetry", "l2_recent_momentum_abs"}
    names = [
        c
        for c in derived.columns
        if c in merged.columns and not c.startswith("l2_direction") and c not in dir_like
    ]
    if "vixy_l1b_interaction" in merged.columns:
        names.append("vixy_l1b_interaction")
    return list(dict.fromkeys(names))


_L2_REGIME_INTERACTION_NAN_COLS: frozenset[str] = frozenset()

_L2_PRE_FILTER_PROTECT: frozenset[str] = frozenset({"symbol", "time_key"})


def _l2_pre_filter_features(merged: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """After feature merge: drop constant/degenerate columns and optional business hard-drops (before fillna).

    Disable with ``L2_PRE_FILTER_FEATURES=0``. Extra hard drops: ``L2_PRE_FILTER_HARD_DROP=col1,col2`` (replaces
    default ``l1a_is_warm,bo_or_dist`` when set non-empty — use empty string for constant-only).
    """
    if (os.environ.get("L2_PRE_FILTER_FEATURES", "1") or "").strip().lower() in {"0", "false", "no"}:
        return merged, feature_cols
    raw_hd = (os.environ.get("L2_PRE_FILTER_HARD_DROP", "l1a_is_warm,bo_or_dist") or "").strip()
    if raw_hd:
        hard_drop = {x.strip() for x in raw_hd.split(",") if x.strip()}
    else:
        hard_drop = set()
    drop: set[str] = set(hard_drop) & set(merged.columns)
    for c in merged.columns:
        if c in _L2_PRE_FILTER_PROTECT:
            continue
        try:
            nu = int(merged[c].nunique(dropna=False))
        except (TypeError, ValueError):
            continue
        if nu <= 1:
            drop.add(c)
    drop = {c for c in drop if c in merged.columns}
    if not drop:
        return merged, feature_cols
    merged = merged.drop(columns=list(drop), errors="ignore")
    feature_cols = [c for c in feature_cols if c not in drop]
    preview = sorted(drop)
    tail = f" ... +{len(preview) - 24}" if len(preview) > 24 else ""
    print(
        f"  [L2] pre_filter: dropped {len(drop)} cols (nunique<=1 or hard) — {preview[:24]}{tail}",
        flush=True,
    )
    print(f"  [L2] feature_cols after pre_filter: {len(feature_cols)}", flush=True)
    return merged, feature_cols


def _build_l2_frame(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
    *,
    fit_mask: np.ndarray | None = None,
    derived_feature_stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    merged = _merge_l2_upstream_outputs(df, l1a_outputs, l1b_outputs)
    merged = pd.concat(
        [merged.reset_index(drop=True), _l2_l1b_condition_extras(merged).reset_index(drop=True)],
        axis=1,
    )
    stats = (
        dict(derived_feature_stats)
        if derived_feature_stats is not None
        else _l2_fit_derived_feature_stats(merged, fit_mask)
    )
    residual = _residual_feature_frame(df)
    derived = _derived_l2_feature_frame(merged, derived_feature_stats=stats, price_df=df)
    merged = pd.concat([merged.reset_index(drop=True), residual.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    vixy_cols: list[str] = []
    if L2_USE_VIXY:
        from core.features.vixy_features import attach_vixy_features_to_l2_merged

        vixy_cols = attach_vixy_features_to_l2_merged(merged, path=VIXY_DATA_PATH)
        if vixy_cols:
            cov = float(pd.to_numeric(merged[vixy_cols[0]], errors="coerce").notna().mean())
            print(f"[L2] VIXY: {len(vixy_cols)} features, coverage={cov:.1%}", flush=True)
    _l2_attach_vixy_l1b_interaction(merged, vixy_enabled=L2_USE_VIXY)
    group_map = {
        "state": _l2_state_input_cols(merged),
        "condition": _l2_condition_input_cols(merged),
        "direction": _l2_direction_input_cols(merged),
        "residual": _l2_residual_input_cols(residual),
        "derived": _l2_derived_group_column_names(derived, merged),
    }
    if vixy_cols:
        group_map["vixy"] = vixy_cols
    feature_cols = list(dict.fromkeys([c for cols in group_map.values() for c in cols if c not in {"symbol", "time_key"}]))
    merged, feature_cols = _l2_pre_filter_features(merged, feature_cols)
    for c in feature_cols:
        s = pd.to_numeric(merged[c], errors="coerce")
        if c in _L2_REGIME_INTERACTION_NAN_COLS:
            merged[c] = s.astype(np.float32)
        else:
            merged[c] = s.fillna(0.0).astype(np.float32)
    for group_name, cols in group_map.items():
        print(f"  [L2] input group {group_name}: {len(cols)} cols", flush=True)
    return merged, feature_cols, stats


def train_l2_trade_decision(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
) -> L2TrainingBundle:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L2] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    prep_bar = tqdm(
        total=8,
        desc="[L2] prep",
        unit="step",
        leave=True,
        file=_tqdm_stream(),
        disable=not _lgb_round_tqdm_enabled(),
    )
    prep_step = 0

    def _prep_tick(label: str) -> None:
        nonlocal prep_step
        if prep_bar.total is not None and prep_step < int(prep_bar.total):
            prep_bar.update(1)
            prep_step += 1
        prep_bar.set_postfix_str(label, refresh=False)

    splits = build_stack_time_splits(df["time_key"])
    l2_val_start = l2_val_start_time()
    n_oof = l2_oof_folds_from_env()
    if l1_oof_mode_from_env() == "expanding" and n_oof >= 2:
        print(
            "  [L2] L1_OOF_MODE=expanding: using strict l2_train/l2_val calendar (L2_OOF_FOLDS ignored; "
            "L1 caches are stitched OOF on cal).",
            flush=True,
        )
        n_oof = 1

    if n_oof >= 2:
        train_mask = np.asarray(splits.cal_mask, dtype=bool)
        val_mask = np.asarray(splits.cal_mask, dtype=bool)
        print(
            f"  [L2] blocked time OOF: L2_OOF_FOLDS={n_oof} on full cal window [{TRAIN_END}, {CAL_END}) "
            f"(set L2_OOF_FOLDS=1 for legacy split at L2_VAL_START={l2_val_start})",
            flush=True,
        )
    else:
        train_mask = splits.l2_train_mask
        val_mask = splits.l2_val_mask
    test_mask = splits.test_mask
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("L2: calibration split is empty for train/val.")
    tune_frac = float(os.environ.get("L2_TUNE_FRAC_WITHIN_VAL", "0.5"))
    val_tune_mask, val_report_mask = split_mask_for_tuning_and_report(
        df["time_key"], val_mask, tune_frac=tune_frac, min_rows_each=50
    )
    if not val_tune_mask.any() or not val_report_mask.any():
        raise RuntimeError("L2: failed to create non-empty tuning/report masks inside l2_val.")
    _prep_tick("time_splits")

    base_merged = _merge_l2_upstream_outputs(df, l1a_outputs, l1b_outputs)
    _prep_tick("merge_upstream")

    rid_l2 = np.argmax(base_merged[L1A_REGIME_COLS].to_numpy(dtype=np.float64), axis=1)
    train_excl_ids = _l2_train_exclude_regime_ids_from_env()
    fit_train_mask = np.asarray(train_mask, dtype=bool).copy()
    if train_excl_ids:
        ex = np.isin(rid_l2, np.asarray(train_excl_ids, dtype=np.int64))
        n_drop = int(np.sum(train_mask & ex))
        fit_train_mask &= ~ex
        print(
            f"  [L2] LGBM fit excludes L1a argmax regime id(s) {sorted(set(train_excl_ids))}: "
            f"withheld {n_drop} / {int(train_mask.sum())} l2_train rows "
            f"(L2_TRAIN_EXCLUDE_REGIME_IDS; default=none)",
            flush=True,
        )
    if not fit_train_mask.any():
        raise RuntimeError(
            "L2: fit_train_mask empty after L2_TRAIN_EXCLUDE_REGIME_IDS; disable or narrow exclusion."
        )
    _prep_tick("fit_mask")

    frame, feature_cols, derived_feature_stats = _build_l2_frame(
        df,
        l1a_outputs,
        l1b_outputs,
        fit_mask=fit_train_mask,
    )
    _prep_tick("build_l2_frame")
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)

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
        drop_unsup = os.environ.get("L2_DROP_L1B_UNSUP_DESCRIPTOR", "1").strip().lower() not in {"0", "false", "no"}
        if drop_unsup:
            hard_drop |= set(_l2_unsup_descriptor_feature_names(feature_cols))
        hard_drop |= _l2_dynamic_hard_drop_from_prev_model(feature_cols)
    hard_drop = frozenset(hard_drop)
    feature_cols, l2_dropped_features = _l2_select_features_for_training(
        X, feature_cols, fit_train_mask, min_std=min_std, hard_drop=hard_drop
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
    _prep_tick("feature_select")

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

    range_head_drop = _l2_range_head_hard_drop(feature_cols)
    range_feature_cols = _l2_feature_view_cols(feature_cols, hard_drop=range_head_drop)
    if not range_feature_cols:
        raise RuntimeError("L2 range head feature contract removed all columns; adjust L2_RANGE_HEAD_* settings.")
    if range_head_drop:
        print(
            f"  [L2] range head feature contract: dropped {len(range_head_drop)} cols from range heads",
            flush=True,
        )
        for name in sorted(range_head_drop):
            print(f"       {name}", flush=True)
    X_range = frame[range_feature_cols].to_numpy(dtype=np.float32, copy=False)

    l1b_train_dropout_p = float(os.environ.get("L2_L1B_TRAIN_FEATURE_DROPOUT", "0.1"))
    l1b_do_seed = int(os.environ.get("L2_L1B_DROPOUT_SEED", "42"))
    X_train_fit = _l2_apply_l1b_feature_dropout_train_only(
        X, fit_train_mask, feature_cols, l1b_train_dropout_p, rng=np.random.default_rng(l1b_do_seed)
    )
    X_range_train_fit = _l2_apply_l1b_feature_dropout_train_only(
        X_range, fit_train_mask, range_feature_cols, l1b_train_dropout_p, rng=np.random.default_rng(l1b_do_seed)
    )
    if l1b_train_dropout_p > 0.0:
        n_l1b = sum(1 for c in feature_cols if c.startswith("l1b_"))
        print(
            f"  [L2] L1b train feature dropout: p={l1b_train_dropout_p} "
            f"({n_l1b} l1b_* cols; train rows only; val/report unchanged; seed={l1b_do_seed})",
            flush=True,
        )

    mh_on = _l2_multi_horizon_enabled(df)
    if mh_on:
        r10a = _decision_forward_range_h_atr_array(df, bars=10)
        r20a = _decision_forward_range_h_atr_array(df, bars=20)
        r30a = _decision_forward_range_h_atr_array(df, bars=30)
        rr_for_cost = r30a
    else:
        r10a = r20a = r30a = None
        rr_for_cost = _decision_forward_range_atr_array(df)

    cost_atr_base = _l2_straddle_cost_atr()
    range_roll_cfg = _l2_range_roll_cost_env_config() if _l2_range_roll_cost_enabled() else None
    if range_roll_cfg is not None and _l2_dynamic_straddle_cost_enabled():
        print(
            "  [L2] L2_STRADDLE_RANGE_ROLL_COST=1 overrides L2_DYNAMIC_STRADDLE_COST for labels/policy cost.",
            flush=True,
        )
    if range_roll_cfg is not None:
        cost_atr_row = _l2_range_roll_cost_array(df, rr_for_cost, cfg=range_roll_cfg)
        dyn_cfg = None
        straddle_cost_mode = "range_roll_quantile"
    elif _l2_dynamic_straddle_cost_enabled():
        dyn_cfg = _l2_dynamic_straddle_cost_env_config()
        cost_atr_row = _l2_dynamic_straddle_cost_array(df, cost_atr_base, cfg=dyn_cfg)
        straddle_cost_mode = "dynamic_atr_quantile"
    else:
        dyn_cfg = None
        cost_atr_row = np.full(len(df), float(cost_atr_base), dtype=np.float64)
        straddle_cost_mode = "fixed"

    range_horizon_meta = _l2_range_horizon_bars()
    c10 = c20 = c30 = float("nan")
    if mh_on:
        c10, c20, c30 = _l2_mh_horizon_costs()
        rng_nan = ~(np.isfinite(r10a) & np.isfinite(r20a) & np.isfinite(r30a))
        rr10 = np.where(rng_nan, 0.0, r10a)
        rr20 = np.where(rng_nan, 0.0, r20a)
        rr30 = np.where(rng_nan, 0.0, r30a)
        y_trade = ((rr10 > c10) | (rr20 > c20) | (rr30 > c30)).astype(np.int64)
        rr_raw = r30a
        realized_range_atr = np.where(rng_nan, 0.0, r30a)
        y_range = realized_range_atr.astype(np.float32)
        y_r10 = np.clip(rr10, 0.0, 15.0).astype(np.float32)
        y_r20 = np.clip(rr20, 0.0, 15.0).astype(np.float32)
        y_ttp = np.clip(_decision_range_ttp90_norm_30_array(df), 0.0, 1.0).astype(np.float32)
        print(
            f"  [L2] L2_MULTI_HORIZON=1: trade_gate = OR_k( forward_range_k > cost_k ) with rising cost_k "
            f"(theta proxy): 10m={c10:.3f}, 20m={c20:.3f}, 30m={c30:.3f} ATR — late-window gamma can still win; "
            f"range head = 30m; ttp90_norm = timing of range build-up.",
            flush=True,
        )
    else:
        rr_raw = rr_for_cost
        rng_nan = ~np.isfinite(rr_raw)
        realized_range_atr = np.where(rng_nan, 0.0, rr_raw)
        y_trade = (realized_range_atr > cost_atr_row).astype(np.int64)
        y_range = realized_range_atr.astype(np.float32)
        y_r10 = y_r20 = y_ttp = None

    mfe, mae = _mfe_mae_atr_arrays(df)
    active_train = fit_train_mask & (y_trade == 1)
    active_val = val_mask & (y_trade == 1)
    y_mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    y_mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    y_mfe_fit, mfe_head_prep = _l2_positive_head_target_prep(y_mfe, head_name="mfe", clip_max=5.0)
    y_mae_fit, mae_head_prep = _l2_positive_head_target_prep(y_mae, head_name="mae", clip_max=4.0)
    y_range_fit, range_head_prep = _l2_positive_head_target_prep(y_range, head_name="range", clip_max=10.0)
    mfe_w_all = _l2_mfe_sample_weights(y_mfe_fit, y_trade == 1).astype(np.float64, copy=False)
    _l2_log_mfe_weighting_diagnostics(y_mfe_fit, mfe_w_all, active_mask=(y_trade == 1))
    y_r10_fit = y_r20_fit = y_ttp_fit = None
    r10_head_prep = r20_head_prep = ttp_head_prep = None
    if mh_on and y_r10 is not None and y_r20 is not None and y_ttp is not None:
        y_r10_fit, r10_head_prep = _l2_positive_head_target_prep(y_r10, head_name="range10", clip_max=15.0)
        y_r20_fit, r20_head_prep = _l2_positive_head_target_prep(y_r20, head_name="range20", clip_max=15.0)
        y_ttp_fit, ttp_head_prep = _l2_positive_head_target_prep(y_ttp, head_name="ttp90", clip_max=1.0)
    straddle_label_stats = {
        "train_trade_rate": float(np.mean(y_trade[fit_train_mask])) if np.any(fit_train_mask) else 0.0,
        "train_mean_range_atr": float(np.mean(realized_range_atr[fit_train_mask])) if np.any(fit_train_mask) else 0.0,
        "straddle_cost_atr": float(cost_atr_base),
        "straddle_cost_mode": straddle_cost_mode,
        "train_mean_cost_atr": float(np.mean(cost_atr_row[fit_train_mask])) if np.any(fit_train_mask) else float(cost_atr_base),
        "range_horizon_bars": int(range_horizon_meta),
        "multi_horizon": bool(mh_on),
        "multi_horizon_label_rule": (
            "trade_gate = OR[(range_10>cost_10),(range_20>cost_20),(range_30>cost_30)]; "
            "primary range target uses 30m range; auxiliary heads predict range_10/range_20/ttp90"
            if mh_on
            else "trade_gate = (forward_range_horizon > cost_row); primary range target uses configured horizon"
        ),
        "mh_costs_atr": {"10": c10, "20": c20, "30": c30} if mh_on else None,
    }
    _prep_tick("labels_targets")

    log_layer_banner("[L2] Trade decision (LGBM)")
    if n_oof >= 2:
        log_time_key_split(
            "L2",
            df["time_key"],
            train_mask,
            val_mask,
            train_label=f"l2_cal(full) [{TRAIN_END}, {CAL_END})",
            val_label=f"l2_cal(full) [{TRAIN_END}, {CAL_END})",
            extra_note=f"OOF: {n_oof} contiguous time folds; val masks match cal for tune/report slicing only.",
        )
    else:
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
    log_numpy_x_stats("L2", X[fit_train_mask], label="X[l2_train_fit]")
    l1a_cols = [c for c in feature_cols if c.startswith("l1a_")]
    l1b_cols = [c for c in feature_cols if c.startswith("l1b_")]
    dir_cols = [c for c in feature_cols if c.startswith("l2_direction") or c in {"l2_skew_imbalance", "l2_recent_momentum_abs", "l2_order_flow_asymmetry"}]
    res_cols = [c for c in feature_cols if c not in l1a_cols and c not in l1b_cols and c not in dir_cols]
    print(
        f"  [L2] feature_cols total={len(feature_cols)} (expect ~51+)  "
        f"l1a_*={len(l1a_cols)}  l1b_*={len(l1b_cols)}  residual/other={len(res_cols)}",
        flush=True,
    )
    print(f"  [L2] residual columns (n={len(res_cols)}): {res_cols}", flush=True)
    print(
        f"  [L2] upstream artifact refs: L1a={artifact_path(L1A_MODEL_FILE)}  "
        f"L1b meta={artifact_path(L1B_META_FILE)}",
        flush=True,
    )
    print(
        "  [L2] note: l1a_*/l1b_* come from supplied upstream outputs; preferred pipeline path uses frozen-artifact inference caches.",
        flush=True,
    )
    print(
        "  [L2] mode=straddle: trade_gate on forward_range>cost, range head, mfe/mae on y_trade-active rows.",
        flush=True,
    )
    print(f"  [L2] will write: {artifact_path(L2_META_FILE)} | {artifact_path(L2_OUTPUT_CACHE_FILE)}", flush=True)
    log_label_baseline("l2_mfe", y_mfe[active_train], task="reg")
    log_label_baseline("l2_mae", y_mae[active_train], task="reg")
    log_label_baseline("l2_range", y_range[fit_train_mask], task="reg")
    print(
        f"  [L2] aux target prep: mfe(transform={mfe_head_prep['target_transform']}, objective={mfe_head_prep['objective']}, metric={mfe_head_prep['metric']})  "
        f"mae(transform={mae_head_prep['target_transform']}, objective={mae_head_prep['objective']}, metric={mae_head_prep['metric']})  "
        f"range(transform={range_head_prep['target_transform']}, objective={range_head_prep['objective']}, metric={range_head_prep['metric']})",
        flush=True,
    )
    print(
        f"  [L2] straddle labels: train_trade_rate={straddle_label_stats['train_trade_rate']:.3f}  "
        f"train_mean_range_atr={straddle_label_stats['train_mean_range_atr']:.4f}  "
        f"cost_base={straddle_label_stats['straddle_cost_atr']:.4f}  "
        f"mode={straddle_label_stats['straddle_cost_mode']}  "
        f"train_mean_cost={straddle_label_stats['train_mean_cost_atr']:.4f}  "
        f"horizon_bars={straddle_label_stats['range_horizon_bars']}",
        flush=True,
    )
    _prep_tick("pretrain_logs")

    rounds = _l2_boost_rounds()
    # Gate: default 120 — not tied to FAST_TRAIN_MODE (set L2_GATE_EARLY_STOPPING_ROUNDS to override).
    gate_es_rounds = _l2_early_stopping_rounds_from_env("L2_GATE_EARLY_STOPPING_ROUNDS", 120)
    aux_es_fallback = 40 if FAST_TRAIN_MODE else 120
    aux_es_base = _l2_early_stopping_rounds_from_env("L2_EARLY_STOPPING_ROUNDS", aux_es_fallback)
    range_es_rounds = _l2_early_stopping_rounds_from_env("L2_RANGE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mfe_es_rounds = _l2_early_stopping_rounds_from_env("L2_MFE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mae_es_rounds = _l2_early_stopping_rounds_from_env("L2_MAE_EARLY_STOPPING_ROUNDS", aux_es_base)
    log_label_baseline("l2_trade_gate", y_trade[fit_train_mask], task="cls")
    pr_trade = float(np.mean(y_trade[fit_train_mask]))
    if pr_trade >= 0.98 or pr_trade <= 0.02:
        print(
            f"  [L2] WARNING: gate label almost single-class (pos_rate={pr_trade:.3f}). "
            f"Raise/lower L2_STRADDLE_COST_ATR (same units as decision_forward_range_atr) or enable "
            f"L2_DYNAMIC_STRADDLE_COST=1 so y_trade mixes.",
            flush=True,
        )
    gate_cfg = _l2_model_lgb_params("gate")
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
    print(
        f"  [L2] gate: pos_rate={pr_trade:.3f}  is_unbalance=True  lr={gate_cfg['learning_rate']}  "
        f"num_leaves={gate_cfg['num_leaves']}  max_depth={gate_cfg['max_depth']}  min_child_samples={gate_cfg['min_child_samples']}  "
        f"bagging={gate_cfg['bagging_fraction']}/{gate_cfg['bagging_freq']}  early_stopping_rounds={gate_es_rounds}  "
        f"early_stop_metric=auc (first)",
        flush=True,
    )
    print(
        f"  [L2] early_stopping_rounds: range={range_es_rounds}  mfe={mfe_es_rounds}  mae={mae_es_rounds}  "
        f"(aux base={aux_es_base}; override via L2_EARLY_STOPPING_ROUNDS / L2_*_EARLY_STOPPING_ROUNDS)",
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
    range_params = _l2_positive_head_lgb_params(reg_params, range_head_prep)
    mfe_params = _l2_positive_head_lgb_params(reg_params, mfe_head_prep)
    mae_params = _l2_positive_head_lgb_params(reg_params, mae_head_prep)
    gate_nt_w = float(os.environ.get("L2_GATE_NO_TRADE_WEIGHT", "1.5"))
    tr_idx = np.flatnonzero(fit_train_mask)
    va_idx = np.flatnonzero(val_mask)
    if mh_on and r10a is not None:
        mh_full = _l2_mh_gate_sample_weights_full(y_trade, r10a, r20a, r30a, c10, c20, c30)
        gate_w_train = mh_full[tr_idx].astype(np.float64, copy=False)
        gate_w_train[y_trade[tr_idx] == 0] *= gate_nt_w
        gate_w_val = mh_full[va_idx].astype(np.float64, copy=False)
        gate_w_val[y_trade[va_idx] == 0] *= gate_nt_w
        print(
            f"  [L2] gate sample_weight: MH time-weights (early/late/chop) × L2_GATE_NO_TRADE_WEIGHT={gate_nt_w:.3f} on negatives",
            flush=True,
        )
    else:
        gate_w_train = np.ones(int(np.sum(fit_train_mask)), dtype=np.float64)
        gate_w_val = np.ones(int(np.sum(val_mask)), dtype=np.float64)
        if gate_nt_w != 1.0:
            gate_w_train[y_trade[tr_idx] == 0] = gate_nt_w
            gate_w_val[y_trade[va_idx] == 0] = gate_nt_w
            print(f"  [L2] gate sample_weight: no_trade rows ×{gate_nt_w:.3f} (L2_GATE_NO_TRADE_WEIGHT)", flush=True)
    X_gate_train_fit, gate_blocked_cols = _l2_project_gate_features(X_train_fit, feature_cols)
    X_gate_all, _ = _l2_project_gate_features(X, feature_cols)
    if gate_blocked_cols:
        print(
            f"  [L2] trade_gate projection dropped {len(gate_blocked_cols)} cols: {gate_blocked_cols}",
            flush=True,
        )
    _prep_tick("init_heads")
    prep_bar.close()
    gate_model: lgb.Booster
    range_model: lgb.Booster
    mfe_model: lgb.Booster
    mae_model: lgb.Booster
    n_samples = len(df)
    gate_oof = np.full(n_samples, np.nan, dtype=np.float64)
    mfe_oof = np.full(n_samples, np.nan, dtype=np.float64)
    mae_oof = np.full(n_samples, np.nan, dtype=np.float64)

    if n_oof >= 2:
        fold_masks = time_blocked_fold_masks(df["time_key"], fit_train_mask, n_oof, context="L2 OOF")
        best_gate: list[int] = []
        best_range: list[int] = []
        best_mfe: list[int] = []
        best_mae: list[int] = []
        l2_outer = tqdm(
            total=n_oof * 4,
            desc="[L2] OOF models",
            unit="fit",
            leave=True,
            file=_tqdm_stream(),
            disable=not _lgb_round_tqdm_enabled(),
        )
        try:
            for fk, (tr_m, va_m) in enumerate(fold_masks):
                fit_tr = fit_train_mask & tr_m
                fit_va = fit_train_mask & va_m
                active_tr = fit_tr & (y_trade == 1)
                active_va = fit_va & (y_trade == 1)
                if (
                    int(fit_tr.sum()) < 200
                    or int(fit_va.sum()) < 30
                    or int(active_tr.sum()) < 100
                    or int(active_va.sum()) < 25
                ):
                    raise RuntimeError(
                        "L2 OOF: fold too small for range/MFE/MAE. "
                        f"fold={fk} fit_tr={int(fit_tr.sum())} fit_va={int(fit_va.sum())} "
                        f"active_tr={int(active_tr.sum())} active_va={int(active_va.sum())}"
                    )
                tr_ix = np.flatnonzero(fit_tr)
                va_ix = np.flatnonzero(fit_va)
                if mh_on and r10a is not None:
                    w_tr = _l2_mh_gate_sample_weights_full(y_trade, r10a, r20a, r30a, c10, c20, c30)[tr_ix].astype(
                        np.float64, copy=True
                    )
                    w_tr[y_trade[tr_ix] == 0] *= gate_nt_w
                    w_va = _l2_mh_gate_sample_weights_full(y_trade, r10a, r20a, r30a, c10, c20, c30)[va_ix].astype(
                        np.float64, copy=True
                    )
                    w_va[y_trade[va_ix] == 0] *= gate_nt_w
                else:
                    w_tr = np.ones(int(fit_tr.sum()), dtype=np.float64)
                    w_tr[y_trade[tr_ix] == 0] = gate_nt_w
                    w_va = np.ones(int(fit_va.sum()), dtype=np.float64)
                    w_va[y_trade[va_ix] == 0] = gate_nt_w
                dtr_g = lgb.Dataset(
                    X_gate_train_fit[fit_tr],
                    label=y_trade[fit_tr],
                    weight=w_tr,
                    feature_name=feature_cols,
                    free_raw_data=False,
                )
                dva_g = lgb.Dataset(
                    X_gate_all[fit_va],
                    label=y_trade[fit_va],
                    weight=w_va,
                    feature_name=feature_cols,
                    free_raw_data=False,
                )
                cbs, cl = _lgb_train_callbacks_with_round_tqdm(
                    gate_es_rounds, rounds, f"[L2] gate oof {fk + 1}/{n_oof}", first_metric_only=True
                )
                try:
                    gm_fold = lgb.train(
                        trade_gate_params, dtr_g, num_boost_round=rounds, valid_sets=[dva_g], callbacks=cbs
                    )
                finally:
                    for fn in cl:
                        fn()
                bi_g = int(gm_fold.best_iteration) if gm_fold.best_iteration is not None else rounds
                best_gate.append(max(1, bi_g))
                gate_oof[fit_va] = gm_fold.predict(X_gate_all[fit_va]).astype(np.float64)
                l2_outer.update(1)

                cbs, cl = _lgb_train_callbacks_with_round_tqdm(range_es_rounds, rounds, f"[L2] range oof {fk + 1}/{n_oof}")
                try:
                    rm_fold = lgb.train(
                        range_params,
                        lgb.Dataset(
                            X_range_train_fit[fit_tr],
                            label=y_range_fit[fit_tr],
                            feature_name=range_feature_cols,
                            free_raw_data=False,
                        ),
                        num_boost_round=rounds,
                        valid_sets=[
                            lgb.Dataset(
                                X_range[fit_va], label=y_range_fit[fit_va], feature_name=range_feature_cols, free_raw_data=False
                            )
                        ],
                        callbacks=cbs,
                    )
                finally:
                    for fn in cl:
                        fn()
                best_range.append(max(1, int(rm_fold.best_iteration) if rm_fold.best_iteration is not None else rounds))
                l2_outer.update(1)

                cbs, cl = _lgb_train_callbacks_with_round_tqdm(mfe_es_rounds, rounds, f"[L2] mfe oof {fk + 1}/{n_oof}")
                try:
                    mf_fold = lgb.train(
                        mfe_params,
                        lgb.Dataset(
                            X_train_fit[active_tr],
                            label=y_mfe_fit[active_tr],
                            weight=mfe_w_all[active_tr],
                            feature_name=feature_cols,
                            free_raw_data=False,
                        ),
                        num_boost_round=rounds,
                        valid_sets=[
                            lgb.Dataset(
                                X[active_va],
                                label=y_mfe_fit[active_va],
                                weight=mfe_w_all[active_va],
                                feature_name=feature_cols,
                                free_raw_data=False,
                            )
                        ],
                        callbacks=cbs,
                    )
                finally:
                    for fn in cl:
                        fn()
                best_mfe.append(max(1, int(mf_fold.best_iteration) if mf_fold.best_iteration is not None else rounds))
                mfe_oof[fit_va] = mf_fold.predict(X[fit_va]).astype(np.float64)
                l2_outer.update(1)

                cbs, cl = _lgb_train_callbacks_with_round_tqdm(mae_es_rounds, rounds, f"[L2] mae oof {fk + 1}/{n_oof}")
                try:
                    ma_fold = lgb.train(
                        mae_params,
                        lgb.Dataset(
                            X_train_fit[active_tr],
                            label=y_mae_fit[active_tr],
                            feature_name=feature_cols,
                            free_raw_data=False,
                        ),
                        num_boost_round=rounds,
                        valid_sets=[
                            lgb.Dataset(
                                X[active_va], label=y_mae_fit[active_va], feature_name=feature_cols, free_raw_data=False
                            )
                        ],
                        callbacks=cbs,
                    )
                finally:
                    for fn in cl:
                        fn()
                best_mae.append(max(1, int(ma_fold.best_iteration) if ma_fold.best_iteration is not None else rounds))
                mae_oof[fit_va] = ma_fold.predict(X[fit_va]).astype(np.float64)
                l2_outer.update(1)
        finally:
            l2_outer.close()

        nr_gate = int(np.clip(np.median(best_gate), 10, rounds))
        nr_range = int(np.clip(np.median(best_range), 10, rounds))
        nr_mfe = int(np.clip(np.median(best_mfe), 10, rounds))
        nr_mae = int(np.clip(np.median(best_mae), 10, rounds))
        print(
            f"  [L2] OOF median best_iteration → gate={nr_gate} range={nr_range} mfe={nr_mfe} mae={nr_mae} (cap={rounds})",
            flush=True,
        )
        gate_model = lgb.train(
            trade_gate_params,
            lgb.Dataset(
                X_gate_train_fit[fit_train_mask],
                label=y_trade[fit_train_mask],
                weight=gate_w_train,
                feature_name=feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nr_gate,
        )
        range_model = lgb.train(
            range_params,
            lgb.Dataset(
                X_range_train_fit[fit_train_mask],
                label=y_range_fit[fit_train_mask],
                feature_name=range_feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nr_range,
        )
        mfe_model = lgb.train(
            mfe_params,
            lgb.Dataset(
                X_train_fit[active_train],
                label=y_mfe_fit[active_train],
                weight=mfe_w_all[active_train],
                feature_name=feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nr_mfe,
        )
        mae_model = lgb.train(
            mae_params,
            lgb.Dataset(
                X_train_fit[active_train],
                label=y_mae_fit[active_train],
                feature_name=feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nr_mae,
        )
    else:
        dtrain_gate = lgb.Dataset(
            X_gate_train_fit[fit_train_mask],
            label=y_trade[fit_train_mask],
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
            file=_tqdm_stream(),
            disable=not _lgb_round_tqdm_enabled(),
        )
        try:
            cbs, cl = _lgb_train_callbacks_with_round_tqdm(gate_es_rounds, rounds, "[L2] gate", first_metric_only=True)
            try:
                gate_model = lgb.train(
                    trade_gate_params, dtrain_gate, num_boost_round=rounds, valid_sets=[dval_gate], callbacks=cbs
                )
            finally:
                for fn in cl:
                    fn()
            l2_outer.set_postfix_str("gate", refresh=False)
            l2_outer.update(1)

            cbs, cl = _lgb_train_callbacks_with_round_tqdm(range_es_rounds, rounds, "[L2] range")
            try:
                range_model = lgb.train(
                    range_params,
                    lgb.Dataset(
                        X_range_train_fit[fit_train_mask],
                        label=y_range_fit[fit_train_mask],
                        feature_name=range_feature_cols,
                        free_raw_data=False,
                    ),
                    num_boost_round=rounds,
                    valid_sets=[
                        lgb.Dataset(
                            X_range[val_mask], label=y_range_fit[val_mask], feature_name=range_feature_cols, free_raw_data=False
                        )
                    ],
                    callbacks=cbs,
                )
            finally:
                for fn in cl:
                    fn()
            l2_outer.set_postfix_str("range", refresh=False)
            l2_outer.update(1)

            if int(active_train.sum()) < 100 or int(active_val.sum()) < 25:
                raise RuntimeError(
                    "L2: too few active rows for MFE/MAE heads after strict time split. "
                    f"active_train={int(active_train.sum())}, active_val={int(active_val.sum())}"
                )
            cbs, cl = _lgb_train_callbacks_with_round_tqdm(mfe_es_rounds, rounds, "[L2] mfe")
            try:
                mfe_model = lgb.train(
                    mfe_params,
                    lgb.Dataset(
                        X_train_fit[active_train],
                        label=y_mfe_fit[active_train],
                        weight=mfe_w_all[active_train],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    ),
                    num_boost_round=rounds,
                    valid_sets=[
                        lgb.Dataset(
                            X[active_val],
                            label=y_mfe_fit[active_val],
                            weight=mfe_w_all[active_val],
                            feature_name=feature_cols,
                            free_raw_data=False,
                        )
                    ],
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
                    valid_sets=[
                        lgb.Dataset(X[active_val], label=y_mae_fit[active_val], feature_name=feature_cols, free_raw_data=False)
                    ],
                    callbacks=cbs,
                )
            finally:
                for fn in cl:
                    fn()
            l2_outer.set_postfix_str("mae", refresh=False)
            l2_outer.update(1)
        finally:
            l2_outer.close()

    range10_model = None
    range20_model = None
    ttp_model = None
    range10_pred = np.zeros(n_samples, dtype=np.float32)
    range20_pred = np.zeros(n_samples, dtype=np.float32)
    ttp90_pred = np.zeros(n_samples, dtype=np.float32)
    if mh_on and y_r10_fit is not None and r10_head_prep is not None:
        if n_oof >= 2:
            nrt = nr_range
        else:
            bi = range_model.best_iteration
            nrt = max(10, int(bi) if bi is not None else rounds)
        r10_params = _l2_positive_head_lgb_params(reg_params, r10_head_prep)
        r20_params = _l2_positive_head_lgb_params(reg_params, r20_head_prep)
        ttp_params = _l2_positive_head_lgb_params(reg_params, ttp_head_prep)
        print(f"  [L2] multi-horizon aux regressors: range_10 / range_20 / ttp90 (rounds={nrt})", flush=True)
        range10_model = lgb.train(
            r10_params,
            lgb.Dataset(
                X_range_train_fit[fit_train_mask],
                label=y_r10_fit[fit_train_mask],
                feature_name=range_feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nrt,
        )
        range20_model = lgb.train(
            r20_params,
            lgb.Dataset(
                X_range_train_fit[fit_train_mask],
                label=y_r20_fit[fit_train_mask],
                feature_name=range_feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nrt,
        )
        ttp_model = lgb.train(
            ttp_params,
            lgb.Dataset(
                X_range_train_fit[fit_train_mask],
                label=y_ttp_fit[fit_train_mask],
                feature_name=range_feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=nrt,
        )
        range10_pred = _l2_positive_head_predict(range10_model, X_range, r10_head_prep, clip_max=15.0)
        range20_pred = _l2_positive_head_predict(range20_model, X_range, r20_head_prep, clip_max=15.0)
        ttp90_pred = _l2_positive_head_predict(ttp_model, X_range, ttp_head_prep, clip_max=1.0)

    gate_raw_all = gate_model.predict(X_gate_all).astype(np.float64)
    if n_oof >= 2:
        if not bool(np.all(np.isfinite(gate_oof[fit_train_mask]))):
            raise RuntimeError("L2 OOF: incomplete gate OOF predictions on fit_train_mask.")
        gate_calibrator = _fit_binary_calibrator(y_trade[fit_train_mask], gate_oof[fit_train_mask])
    else:
        gate_calibrator = _fit_binary_calibrator(y_trade[val_tune_mask], gate_raw_all[val_tune_mask])
    trade_p_all = _apply_binary_calibrator(gate_raw_all, gate_calibrator).astype(np.float32)
    range_pred = _l2_positive_head_predict(range_model, X_range, range_head_prep, clip_max=10.0)
    mfe_pred = _l2_positive_head_predict(mfe_model, X, mfe_head_prep, clip_max=5.0)
    mae_pred = _l2_positive_head_predict(mae_model, X, mae_head_prep, clip_max=4.0)
    min_edge = float(np.clip(float(os.environ.get("L2_MIN_STRADDLE_EDGE", "0.0")), -2.0, 2.0))
    predicted_profit_all = _l2_straddle_predicted_profit(range_pred, cost_atr=cost_atr_row)
    predicted_profit_all = _l2_apply_expected_edge_regime_blacklist(
        np.asarray(predicted_profit_all, dtype=np.float32), frame
    ).astype(np.float32)
    trend_eps = float(np.clip(float(os.environ.get("L2_REGIME_TREND_EPS", "0.02")), 1e-6, 1.0))
    vol_all = frame["l1a_vol_forecast"].to_numpy(dtype=np.float64)
    trend_all = (
        frame["l1a_vol_trend"].to_numpy(dtype=np.float64)
        if "l1a_vol_trend" in frame.columns
        else np.zeros(len(frame), dtype=np.float64)
    )
    dyn_q33, dyn_q66 = _l2_fit_vol_quantiles_vol_forecast(vol_all, val_tune_mask)
    dyn_reg_ids = _l2_vol_regime_ids(vol_all, trend_all, dyn_q33, dyn_q66, trend_eps)
    dyn_state_keys_all = _l2_vol_regime_names_from_ids(dyn_reg_ids)
    trade_threshold, policy_search_meta = _search_l2_trade_threshold_economic(
        trade_p_all[val_tune_mask],
        predicted_profit_all[val_tune_mask],
        target_trade_rate=_l2_target_trade_rate(),
        min_edge=min_edge,
        state_keys=dyn_state_keys_all[val_tune_mask],
    )
    trade_threshold_row = _l2_threshold_row_from_state_map(
        trade_threshold,
        dyn_state_keys_all,
        str(policy_search_meta.get("dynamic_threshold_map", "")),
    )
    threshold_for_decision: float | np.ndarray = (
        trade_threshold_row.astype(np.float64, copy=False) if trade_threshold_row is not None else float(trade_threshold)
    )
    expected_edge_all = _l2_straddle_expected_edge(trade_p_all, range_pred, cost_atr=cost_atr_row)
    expected_edge_all = _l2_apply_expected_edge_regime_blacklist(
        np.asarray(expected_edge_all, dtype=np.float32), frame
    ).astype(np.float32)
    regime_adjust = os.environ.get("L2_REGIME_STRADDLE_ADJUST", "").strip().lower() in {"1", "true", "yes"}
    vol_q33: float | None = None
    vol_q66: float | None = None
    reg_ids_all: np.ndarray | None = None
    regime_cfg_merged: dict[str, dict[str, float]] | None = None
    size_mult_row = np.ones(len(frame), dtype=np.float32)
    vol_regime_names = np.full(len(frame), "", dtype=object)
    if regime_adjust:
        regime_cfg_merged = _l2_regime_straddle_config_merged()
        vol_q33, vol_q66 = _l2_fit_vol_quantiles_vol_forecast(vol_all, val_tune_mask)
        reg_ids_all = _l2_vol_regime_ids(vol_all, trend_all, vol_q33, vol_q66, trend_eps)
        min_mp, mult_sm = _l2_regime_straddle_arrays()
        min_edge_row = np.maximum(float(min_edge), min_mp[reg_ids_all])
        straddle_on_mask = (trade_p_all >= threshold_for_decision) & (
            np.asarray(predicted_profit_all, dtype=np.float64) > min_edge_row
        )
        base_size = _l2_straddle_size(
            trade_p_all,
            np.maximum(np.asarray(predicted_profit_all, dtype=np.float32), 0.0),
            straddle_on_mask,
        )
        size_mult_row = mult_sm[reg_ids_all].astype(np.float32)
        size_pred = (base_size * size_mult_row).astype(np.float32)
        vol_regime_names = _l2_vol_regime_names_from_ids(reg_ids_all)
    else:
        straddle_on_mask = (trade_p_all >= threshold_for_decision) & (predicted_profit_all > min_edge)
        size_pred = _l2_straddle_size(
            trade_p_all,
            np.maximum(np.asarray(predicted_profit_all, dtype=np.float32), 0.0),
            straddle_on_mask,
        )
    decision_confidence = trade_p_all.astype(np.float32)
    if regime_adjust:
        print(
            f"  [L2] L2_REGIME_STRADDLE_ADJUST=1  val_tune vol q33={float(vol_q33):.6f} q66={float(vol_q66):.6f}  "
            f"trend_eps={trend_eps:.4f}",
            flush=True,
        )

    def _print_straddle_split(split_name: str, sm: np.ndarray) -> None:
        sm = np.asarray(sm, dtype=bool)
        if not sm.any():
            return
        try:
            auc_t = float(roc_auc_score(y_trade[sm], trade_p_all[sm]))
        except ValueError:
            auc_t = float("nan")
        b_t = brier_binary(y_trade[sm].astype(np.float64), trade_p_all[sm].astype(np.float64))
        e_t = ece_binary(y_trade[sm].astype(np.float64), trade_p_all[sm].astype(np.float64))
        yt = y_range[sm].astype(np.float64)
        yp = np.asarray(range_pred[sm], dtype=np.float64)
        mae_r = float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")
        print(
            f"\n  [L2] {split_name} — straddle  gate AUC={auc_t:.4f}  Brier={b_t:.4f}  ECE={e_t:.4f}  range_MAE={mae_r:.4f}",
            flush=True,
        )

    _print_straddle_split("val_report", val_report_mask)
    if test_mask.any():
        _print_straddle_split("holdout", test_mask)

    av = np.asarray(val_report_mask, dtype=bool) & (y_trade == 1)
    if av.sum() >= 5:
        for name, yt_a, yp_a in (("mfe", y_mfe, mfe_pred), ("mae", y_mae, mae_pred)):
            yt = yt_a[av].astype(np.float64)
            yp = yp_a[av].astype(np.float64)
            mae_h = float(mean_absolute_error(yt, yp))
            print(f"  [L2] val_report — {name} head (y_trade active): MAE={mae_h:.4f}", flush=True)

    _log_l2_l1b_gain_importance_by_group(gate_model, feature_cols, "trade_gate")
    _log_l2_l1b_gain_importance_by_group(range_model, feature_cols, "range")
    if range10_model is not None:
        _log_l2_l1b_gain_importance_by_group(range10_model, feature_cols, "range_10")
    if range20_model is not None:
        _log_l2_l1b_gain_importance_by_group(range20_model, feature_cols, "range_20")
    if ttp_model is not None:
        _log_l2_l1b_gain_importance_by_group(ttp_model, feature_cols, "ttp90")
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_straddle_on"] = straddle_on_mask.astype(np.int32)
    outputs["l2_gate_prob"] = trade_p_all.astype(np.float32)
    outputs["l2_decision_confidence"] = decision_confidence
    outputs["l2_range_pred"] = np.asarray(range_pred, dtype=np.float32)
    outputs["l2_range_pred_10"] = range10_pred.astype(np.float32)
    outputs["l2_range_pred_20"] = range20_pred.astype(np.float32)
    outputs["l2_pred_ttp90_norm"] = ttp90_pred.astype(np.float32)
    outputs["l2_predicted_profit"] = np.asarray(predicted_profit_all, dtype=np.float32)
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_vol_regime"] = vol_regime_names.astype(str)
    outputs["l2_regime_size_mult"] = size_mult_row.astype(np.float32)
    outputs["l2_expected_edge"] = expected_edge_all
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)

    os.makedirs(MODEL_DIR, exist_ok=True)
    gate_model.save_model(os.path.join(MODEL_DIR, L2_GATE_FILE))
    range_model.save_model(os.path.join(MODEL_DIR, L2_RANGE_FILE))
    mfe_model.save_model(os.path.join(MODEL_DIR, L2_MFE_FILE))
    mae_model.save_model(os.path.join(MODEL_DIR, L2_MAE_FILE))
    model_files: dict[str, str] = {
        "trade_gate": L2_GATE_FILE,
        "range": L2_RANGE_FILE,
        "mfe": L2_MFE_FILE,
        "mae": L2_MAE_FILE,
    }
    if range10_model is not None:
        range10_model.save_model(os.path.join(MODEL_DIR, L2_RANGE10_FILE))
        model_files["range_10"] = L2_RANGE10_FILE
    if range20_model is not None:
        range20_model.save_model(os.path.join(MODEL_DIR, L2_RANGE20_FILE))
        model_files["range_20"] = L2_RANGE20_FILE
    if ttp_model is not None:
        ttp_model.save_model(os.path.join(MODEL_DIR, L2_TTP90_FILE))
        model_files["ttp90"] = L2_TTP90_FILE
    if gate_calibrator is not None:
        gate_calib_file = L2_TRADE_GATE_CALIBRATOR_FILE
        with open(os.path.join(MODEL_DIR, gate_calib_file), "wb") as f:
            pickle.dump(gate_calibrator, f)
        model_files["trade_gate_calibrator"] = gate_calib_file
    meta = {
        "schema_version": L2_SCHEMA_VERSION,
        "mode": "straddle",
        "l2_oof_folds": int(n_oof) if n_oof >= 2 else 0,
        "l2_policy_state_granularity": os.environ.get("L2_POLICY_STATE_GRANULARITY", "coarse"),
        "l2_gate_no_trade_weight": float(gate_nt_w),
        "l2_l1b_train_feature_dropout": float(l1b_train_dropout_p),
        "l2_l1b_dropout_seed": int(l1b_do_seed),
        "l2_use_l1b_latent_features": bool(use_l1b_latent_feats),
        "feature_cols": feature_cols,
        "range_feature_cols": list(range_feature_cols),
        "range_feature_hard_drop": sorted(range_head_drop),
        "l2_derived_feature_stats": derived_feature_stats,
        "feature_group_counts": {
            "state": len(l1a_cols),
            "condition": len(l1b_cols),
            "direction": len(dir_cols),
            "residual_other": len(res_cols),
        },
        "output_cols": L2_OUTPUT_COLS,
        "decision_mode": "straddle",
        "straddle_cost_atr": float(cost_atr_base),
        "straddle_cost_mode": straddle_label_stats["straddle_cost_mode"],
        "straddle_dynamic_cost": (dyn_cfg if dyn_cfg is not None else None),
        "straddle_range_roll_cost": (range_roll_cfg if range_roll_cfg is not None else None),
        "min_straddle_edge": float(min_edge),
        "range_horizon_bars": int(range_horizon_meta),
        "trade_threshold": float(trade_threshold),
        "trade_threshold_is_dynamic": bool(trade_threshold_row is not None),
        "l2_regime_straddle_adjust": bool(regime_adjust),
        "l2_vol_regime_quantiles": (
            {"q33": float(vol_q33), "q66": float(vol_q66)} if regime_adjust and vol_q33 is not None and vol_q66 is not None else None
        ),
        "l2_regime_straddle_config": (regime_cfg_merged if regime_adjust else None),
        "l2_regime_trend_eps": float(trend_eps),
        "confidence_semantics": "l2_decision_confidence and l2_gate_prob are calibrated P(trade); straddle_on uses them as a low bar with range-first economics",
        "straddle_decision_semantics": (
            "range-first: gate>=trade_threshold(row-adaptive when dynamic map exists) AND predicted_profit > max(L2_MIN_STRADDLE_EDGE, regime min_profit) when L2_REGIME_STRADDLE_ADJUST=1; else global min edge"
            if regime_adjust
            else "range-first: (predicted_range - cost) > min_edge AND gate_prob >= trade_threshold(row-adaptive when enabled); see l2_predicted_profit"
        ),
        "expected_edge_semantics": "diagnostic mixture g*(r-c)+(1-g)*(-c*frac); not used for straddle_on (see straddle_decision_semantics)",
        "size_semantics": (
            "base straddle size (gate * norm max(predicted_profit,0)) times l2_regime_size_mult when L2_REGIME_STRADDLE_ADJUST=1"
            if regime_adjust
            else "normalized gate * positive max(predicted_profit,0) on straddle_on rows"
        ),
        "expected_edge_zero_regime_ids_env": os.environ.get("L2_EXPECTED_EDGE_ZERO_REGIME_IDS", ""),
        "l2_train_exclude_regime_ids": _l2_train_exclude_regime_ids_from_env(),
        "l2_train_exclude_regime_ids_env": os.environ.get("L2_TRAIN_EXCLUDE_REGIME_IDS", "none"),
        "pa_state_features": list(PA_STATE_FEATURES),
        "l2_aux_head_target_prep": (
            {
                "range": range_head_prep,
                "mfe": mfe_head_prep,
                "mae": mae_head_prep,
                **(
                    {
                        "range_10": r10_head_prep,
                        "range_20": r20_head_prep,
                        "ttp90": ttp_head_prep,
                    }
                    if mh_on and r10_head_prep is not None
                    else {}
                ),
            }
        ),
        "straddle_label_stats": straddle_label_stats,
        "policy_search": {
            "trade_threshold": float(trade_threshold),
            **policy_search_meta,
        },
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
            "range": int(range_es_rounds),
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
        "l2_regression_config": reg_params,
        "rollback_criteria": {
            "p1a_abstain": {"rollback_if_gate_auc_drop_gt": 0.01},
            "p1c_l3_data_expansion": {"rollback_if_exit_auc_drop_gt": 0.02},
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
                role="low gate bar (calibrated P) alongside range profit",
                adaptive_hint="economic search on val_tune with trade-rate constraints (range-first objective)",
            ),
            threshold_entry(
                "L2_MIN_STRADDLE_EDGE",
                float(min_edge),
                category="adaptive_candidate",
                role="minimum predicted_profit (range - cost) to set straddle_on",
            ),
            threshold_entry(
                "L2_TARGET_TRADE_RATE",
                float(_l2_target_trade_rate()),
                category="adaptive_candidate",
                role="target straddle trade-rate constraint",
            ),
        ],
    )
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    if test_mask.any():
        te_rng = y_range[test_mask].astype(np.float64)
        test_pp = outputs.loc[test_mask, "l2_predicted_profit"].to_numpy(dtype=np.float32)
        corr_pp = np.corrcoef(test_pp.astype(np.float64), te_rng)[0, 1] if int(test_mask.sum()) > 2 else float("nan")
        test_edge = outputs.loc[test_mask, "l2_expected_edge"].to_numpy(dtype=np.float32)
        corr_ee = np.corrcoef(test_edge.astype(np.float64), te_rng)[0, 1] if int(test_mask.sum()) > 2 else float("nan")
        print(
            f"  [L2] test corr(l2_predicted_profit, y_range_truth)={corr_pp:.4f}  "
            f"(mixture l2_expected_edge diag)={corr_ee:.4f}",
            flush=True,
        )
    print(f"  [L2] meta saved  -> {os.path.join(MODEL_DIR, L2_META_FILE)}", flush=True)
    print(f"  [L2] cache saved -> {cache_path}", flush=True)
    bundle_models: dict[str, Any] = {
        "trade_gate": gate_model,
        "range": range_model,
        "mfe": mfe_model,
        "mae": mae_model,
    }
    if range10_model is not None:
        bundle_models["range_10"] = range10_model
    if range20_model is not None:
        bundle_models["range_20"] = range20_model
    if ttp_model is not None:
        bundle_models["ttp90"] = ttp_model
    if gate_calibrator is not None:
        bundle_models["trade_gate_calibrator"] = gate_calibrator
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
) -> pd.DataFrame:
    frame, _feature_cols, _derived_stats = _build_l2_frame(
        df,
        l1a_outputs,
        l1b_outputs,
        derived_feature_stats=dict(meta.get("l2_derived_feature_stats") or {}),
    )
    target_cols = list(meta["feature_cols"])
    range_target_cols = list(meta.get("range_feature_cols") or target_cols)
    X = _l2_require_inference_features(frame, target_cols)
    X_range = _l2_require_inference_features(frame, range_target_cols)
    mode = meta.get("decision_mode", "straddle")
    aux_prep = meta.get("l2_aux_head_target_prep", {})
    if mode not in {"straddle"}:
        raise RuntimeError(
            f"L2 live inference only supports decision_mode='straddle', got {mode!r}. Retrain L2."
        )
    trade_m = models["trade_gate"]
    range_m = models["range"]
    X_gate, _ = _l2_project_gate_features(X, target_cols)
    trade_p = _apply_binary_calibrator(
        trade_m.predict(X_gate).astype(np.float64),
        models.get("trade_gate_calibrator"),
    ).astype(np.float32)
    range_pred = _l2_positive_head_predict(range_m, X_range, aux_prep.get("range"), clip_max=10.0)
    mfe_pred = _l2_positive_head_predict(models["mfe"], X, aux_prep.get("mfe"), clip_max=5.0)
    mae_pred = _l2_positive_head_predict(models["mae"], X, aux_prep.get("mae"), clip_max=4.0)
    n_inf = len(df)
    z0 = np.zeros(n_inf, dtype=np.float32)
    if models.get("range_10") is not None and aux_prep.get("range_10") is not None:
        range10_pred = _l2_positive_head_predict(models["range_10"], X_range, aux_prep["range_10"], clip_max=15.0)
    else:
        range10_pred = z0
    if models.get("range_20") is not None and aux_prep.get("range_20") is not None:
        range20_pred = _l2_positive_head_predict(models["range_20"], X_range, aux_prep["range_20"], clip_max=15.0)
    else:
        range20_pred = z0
    if models.get("ttp90") is not None and aux_prep.get("ttp90") is not None:
        ttp90_pred = _l2_positive_head_predict(models["ttp90"], X_range, aux_prep["ttp90"], clip_max=1.0)
    else:
        ttp90_pred = z0
    trade_thr = float(meta.get("trade_threshold", 0.5))
    cost_base = float(meta.get("straddle_cost_atr", _l2_straddle_cost_atr()))
    scm = str(meta.get("straddle_cost_mode", "fixed")).strip().lower()
    if scm == "range_roll_quantile":
        rrc = meta.get("straddle_range_roll_cost")
        if not isinstance(rrc, dict):
            rrc = _l2_range_roll_cost_env_config()
        rr = _decision_forward_range_atr_array(df)
        cost_row = _l2_range_roll_cost_array(df, rr, cfg=rrc)
    elif scm == "dynamic_atr_quantile":
        dcfg = meta.get("straddle_dynamic_cost")
        if not isinstance(dcfg, dict):
            dcfg = _l2_dynamic_straddle_cost_env_config()
        cost_row = _l2_dynamic_straddle_cost_array(df, cost_base, cfg=dcfg)
    else:
        cost_row = np.full(len(df), cost_base, dtype=np.float64)
    min_edge = float(meta.get("min_straddle_edge", float(os.environ.get("L2_MIN_STRADDLE_EDGE", "0.0"))))
    predicted_profit = _l2_straddle_predicted_profit(range_pred, cost_atr=cost_row)
    predicted_profit = _l2_apply_expected_edge_regime_blacklist(
        np.asarray(predicted_profit, dtype=np.float32), frame
    ).astype(np.float32)
    expected_edge = _l2_straddle_expected_edge(trade_p, range_pred, cost_atr=cost_row)
    expected_edge = _l2_apply_expected_edge_regime_blacklist(
        np.asarray(expected_edge, dtype=np.float32), frame
    ).astype(np.float32)
    trend_eps_m = float(
        np.clip(float(meta.get("l2_regime_trend_eps", os.environ.get("L2_REGIME_TREND_EPS", "0.02"))), 1e-6, 1.0)
    )
    regime_adjust_m = bool(meta.get("l2_regime_straddle_adjust", False))
    vol_q = meta.get("l2_vol_regime_quantiles") or {}
    q33_m = float(vol_q.get("q33", 0.33)) if isinstance(vol_q, dict) else 0.33
    q66_m = float(vol_q.get("q66", 0.66)) if isinstance(vol_q, dict) else 0.66
    vol_all_i = frame["l1a_vol_forecast"].to_numpy(dtype=np.float64)
    trend_all_i = (
        frame["l1a_vol_trend"].to_numpy(dtype=np.float64)
        if "l1a_vol_trend" in frame.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    reg_ids_i = _l2_vol_regime_ids(vol_all_i, trend_all_i, q33_m, q66_m, trend_eps_m)
    dyn_state_keys = _l2_vol_regime_names_from_ids(reg_ids_i)
    policy_search = meta.get("policy_search") if isinstance(meta.get("policy_search"), dict) else {}
    trade_thr_row = _l2_threshold_row_from_state_map(
        trade_thr,
        dyn_state_keys,
        str(policy_search.get("dynamic_threshold_map", "")) if isinstance(policy_search, dict) else "",
    )
    trade_thr_eval: float | np.ndarray = trade_thr_row if trade_thr_row is not None else trade_thr
    size_mult_i = np.ones(len(df), dtype=np.float32)
    vol_regime_i = np.full(len(df), "", dtype=object)
    if regime_adjust_m:
        cfg_inf = meta.get("l2_regime_straddle_config")
        if not isinstance(cfg_inf, dict):
            cfg_inf = _l2_regime_straddle_config_merged()
        min_mp_i = np.array(
            [float(cfg_inf[k]["min_profit"]) for k in L2_STRADDLE_VOL_REGIME_NAMES],
            dtype=np.float64,
        )
        mult_sm_i = np.array(
            [float(cfg_inf[k]["size_mult"]) for k in L2_STRADDLE_VOL_REGIME_NAMES],
            dtype=np.float64,
        )
        min_edge_row = np.maximum(float(min_edge), min_mp_i[reg_ids_i])
        straddle_on = (trade_p >= trade_thr_eval) & (np.asarray(predicted_profit, dtype=np.float64) > min_edge_row)
        base_sz = _l2_straddle_size(
            trade_p,
            np.maximum(np.asarray(predicted_profit, dtype=np.float32), 0.0),
            straddle_on,
        )
        size_mult_i = mult_sm_i[reg_ids_i].astype(np.float32)
        size_pred = (base_sz * size_mult_i).astype(np.float32)
        vol_regime_i = _l2_vol_regime_names_from_ids(reg_ids_i)
    else:
        straddle_on = (trade_p >= trade_thr_eval) & (predicted_profit > min_edge)
        size_pred = _l2_straddle_size(
            trade_p,
            np.maximum(np.asarray(predicted_profit, dtype=np.float32), 0.0),
            straddle_on,
        )
    decision_confidence = trade_p.astype(np.float32)
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_straddle_on"] = straddle_on.astype(np.int32)
    outputs["l2_gate_prob"] = trade_p.astype(np.float32)
    outputs["l2_decision_confidence"] = decision_confidence
    outputs["l2_range_pred"] = np.asarray(range_pred, dtype=np.float32)
    outputs["l2_range_pred_10"] = np.asarray(range10_pred, dtype=np.float32)
    outputs["l2_range_pred_20"] = np.asarray(range20_pred, dtype=np.float32)
    outputs["l2_pred_ttp90_norm"] = np.asarray(ttp90_pred, dtype=np.float32)
    outputs["l2_predicted_profit"] = np.asarray(predicted_profit, dtype=np.float32)
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_vol_regime"] = vol_regime_i.astype(str)
    outputs["l2_regime_size_mult"] = size_mult_i
    outputs["l2_expected_edge"] = expected_edge.astype(np.float32)
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)
    return outputs

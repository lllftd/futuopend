from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.training.common.constants import (
    CAL_END,
    L1A_META_FILE,
    L1A_REGIME_COLS,
    L1A_SCHEMA_VERSION,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
    L2_UNIFIED_MODEL_FILE,
    L2_USE_VIXY,
    L3_EXIT_FILE,
    L3_META_FILE,
    L3_SCHEMA_VERSION,
    L3_VOL_EXIT_DEFAULTS,
    L3_VALUE_EXTRA_ALLOWED,
    L3_VALUE_FEATURE_BLACKLIST,
    L3_VALUE_FILE,
    MODEL_DIR,
    PA_STATE_FEATURES,
    TEST_END,
    VIXY_DATA_PATH,
)

from core.training.common.lgbm_utils import (
    _tqdm_stream,
    _decision_edge_atr_array,
    _l3_policy_dataset_tqdm_enabled,
    _l3_policy_tqdm_file,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _net_edge_atr_from_state,
)
from core.training.unified.policy_params import derive_policy_params
from core.training.logging.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.training.common.stack_v2_common import (
    l2_l3_entry_decision_class_from_merged,
    l3_oof_folds_from_env,
    log_label_baseline,
    time_blocked_fold_masks,
)
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry
from core.training.common.val_metrics_extra import (
    brier_binary,
    directional_accuracy_regression,
    ece_binary,
    flip_rate_sorted,
    pearson_corr,
    regression_degen_flag,
)
from core.training.unified.pa_modulation import (
    pa_entry_policy_score_bonus_masked,
    pa_exit_eps_multiplier,
    pa_exit_late_hold_entry_scale,
    pa_exit_live_edge_floor_multiplier,
    pa_exit_loss_buffer_multiplier,
    pa_exit_trade_weight_multiplier,
    pa_horizon_scale,
    pa_value_target_scale,
)
from core.training.unified.trajectory import (
    L3TrajectoryConfig,
    l3_traj_step_features_straddle,
    l3_traj_step_features,
)
from core.training.unified.features import build_straddle_features
from core.training.unified.simulation.iv_models import build_base_iv_series
from core.training.unified.simulation.iv_scenarios import dte_grid_days, generate_iv_scenarios, scenario_count
from core.training.unified.simulation.straddle_simulator import StraddleSimulator
from core.training.prep.pa_state_controls import (
    ensure_pa_state_features,
    pa_state_arrays_from_frame,
    pa_state_bucket_label_from_mapping,
    pa_state_bucket_labels_from_arrays,
    pa_state_bucket_labels_from_frame,
)
from core.training.unified.config import defaults as L3DEF
from core.training.unified.config.constants import ISOTONIC_MIN_UNIQUE


@dataclass
class L3TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class L3ExitInferenceState:
    """Per-trade holder kept for call-site compatibility; live exit is prob threshold only."""

    ema_prob: float | None = None
    ema_prev_raw: float | None = None
    latch_exit: bool = False
    hyst_diag_steps: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.ema_prob = None
        self.ema_prev_raw = None
        self.latch_exit = False
        self.hyst_diag_steps.clear()


def l3_exit_decision_live(
    exit_prob_calibrated: float,
    state: L3ExitInferenceState,
    *,
    exit_prob_threshold: float,
) -> tuple[bool, L3ExitInferenceState]:
    """Exit from calibrated exit probability vs searched threshold (prob_only; value head does not gate)."""
    p_raw = float(np.clip(exit_prob_calibrated, 0.0, 1.0))
    ex = bool(p_raw >= float(exit_prob_threshold))
    return ex, state


def _l3_read_meta_schema_version(meta_file: str) -> str:
    path = os.path.join(MODEL_DIR, meta_file)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        return str(m.get("schema_version", ""))
    except Exception:
        return ""


def _l3_upstream_schema_fingerprint() -> dict[str, str]:
    return {
        "l1a_schema_meta": _l3_read_meta_schema_version(L1A_META_FILE) or L1A_SCHEMA_VERSION,
        "l2_schema_meta": _l3_read_meta_schema_version(L2_META_FILE) or L2_SCHEMA_VERSION,
    }


def _l3_extra_merged_feature_columns() -> list[str]:
    return []


def _l3_straddle_feature_cols_in_meta(meta: Mapping[str, Any] | None) -> bool:
    if meta is None:
        return False
    cols = meta.get("feature_cols")
    if not isinstance(cols, (list, tuple)):
        return False
    return "l3_straddle_pnl_pct" in cols


def _l3_straddle_sim_mode_enabled(meta: Mapping[str, Any] | None = None) -> bool:
    """OOS can force straddle sim with OOS_L3_STRADDLE_SIM=1 if meta feature_cols list straddle columns."""
    if os.environ.get("OOS_L3_STRADDLE_SIM", "").strip().lower() in {"1", "true", "yes"} and _l3_straddle_feature_cols_in_meta(
        meta
    ):
        return True
    if meta is not None and "l3_trade_semantics" in meta:
        return str(meta.get("l3_trade_semantics", "")).strip().lower() == "straddle_bs_sim"
    return os.environ.get("L3_STRADDLE_SIM_MODE", "").strip().lower() in {"1", "true", "yes"}


def _l3_straddle_feature_columns() -> list[str]:
    return [
        "rv_5",
        "rv_15",
        "rv_30",
        "rv_60",
        "rv_120",
        "rv_390",
        "parkinson_vol_15",
        "parkinson_vol_30",
        "parkinson_vol_60",
        "parkinson_vol_390",
        "gk_vol_30",
        "gk_vol_60",
        "gk_vol_390",
        "rv_acceleration",
        "vol_of_vol",
        "abs_return_5",
        "abs_return_15",
        "abs_return_30",
        "abs_return_60",
        "intraday_range",
        "range_vs_close",
        "volume_zscore",
        "volume_spike",
        "efficiency_ratio",
        "gap",
        "minute_of_day",
        "day_of_week",
        "l3_base_iv",
    ]

L3_MOMENTUM_LEADING_FEATURE_NAMES: tuple[str, ...] = (
    "l3_unreal_pnl_vel_3bar",
    "l3_unreal_pnl_accel_3bar",
    "l3_drawdown_vel_3bar",
    "l3_live_mae_vel_3bar",
    "l3_bars_since_mfe_high",
    "l3_mfe_stale_flag",
    "l3_pnl_vs_mfe_ratio",
    "l3_vol_surprise_ratio_accel",
    "l3_edge_consumed_pct",
    "l3_time_consumed_pct",
    "l3_edge_per_remaining_bar",
    "l3_path_smoothness",
    "l3_adverse_run_length",
    "l3_favorable_momentum_dying",
)


def _l3_policy_matrix_column_names(extra_merged: list[str]) -> list[str]:
    cols = [
        "l2_straddle_on",
        "l2_range_pred",
        "l2_gate_prob",
        "l2_decision_confidence",
        "l2_size",
        "l2_pred_mfe",
        "l2_pred_mae",
        *(
            [
                "l2_predicted_profit",
                "l3_l2_vol_regime_id",
                "l2_regime_size_mult",
            ]
            if _l3_straddle_sim_mode_enabled()
            else []
        ),
        *[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))],
        "l2_entry_vol",
        *L1A_REGIME_COLS,
        "l1a_vol_forecast",
        "l3_regime_divergence",
        "l3_vol_surprise",
        "l3_hold_bars",
        "l3_unreal_pnl_atr",
        "l3_live_mfe",
        "l3_live_mae",
        "l3_live_edge",
        "l3_side",
        "l3_log_hold_bars",
        "l3_hold_bars_sq",
        "l3_hold_bucket",
        "l3_drawdown_from_peak_atr",
        "l3_unreal_pnl_frac",
        "l3_drawdown_from_peak_frac",
        "l3_ret_last_3_frac",
        "l3_ret_last_5_frac",
        "l3_volatility_in_trade_frac",
        "l3_trend_slope_frac",
        *L3_MOMENTUM_LEADING_FEATURE_NAMES,
        "l3_price_velocity_3bar_atr",
        "l3_feature_momentum_regdiv_3bar",
        "l3_vol_surprise_accel",
        "l3_regime_stability_3bar",
        *PA_STATE_FEATURES,
        *extra_merged,
        "l3_signal_conf_decay",
        "l3_signal_direction_agree",
        "l3_regime_changed",
        "l3_l2_gate_current",
        "l3_l2_gate_decay",
        "l3_would_enter_now",
        "l3_regret_ratio",
        "l3_bars_since_peak",
        "l3_at_new_high",
        "l3_regret_velocity",
        "l3_trade_quality_bayes",
        "l3_range_realization_ratio",
        "l3_theta_burn_fraction",
        "l3_vixy_change_since_entry",
        "l3_range_expansion_speed",
    ]
    if (not _l3_straddle_sim_mode_enabled()) and _l3_exit_hold_interaction_features_enabled():
        cols.extend(
            [
                "l3_hold_bars_x_unreal_pnl_atr",
                "l3_hold_bars_x_price_velocity_3bar_atr",
            ]
        )
    if _l3_straddle_sim_mode_enabled():
        cols.extend(
            [
                "l3_straddle_value_rel",
                "l3_straddle_pnl_pct",
                "l3_straddle_theta",
                "l3_straddle_vega",
                "l3_straddle_gamma",
                "l3_straddle_iv",
                "l3_straddle_entry_iv",
                "l3_straddle_t_remaining",
                "l3_underlying_abs_move",
                "l3_underlying_gap_abs",
                "l3_theta_burn_rate",
                "l3_iv_rv_spread",
                "l3_remaining_dte_ratio",
                "l3_vixy_max_since_entry",
                "l3_vixy_rel_entry",
                "l3_roll_pnl_vol_5",
                "l3_pnl_path_curvature",
                *_l3_straddle_feature_columns(),
            ]
        )
    return cols


def _l3_validate_hysteresis_env() -> None:
    enter = float(L3DEF.hyst_enter_default())
    leave = float(L3DEF.hyst_leave_default())
    gap_min = float(L3DEF.hyst_min_gap())
    if leave >= enter:
        raise ValueError(
            f"L3 hysteresis invalid: L3_EXIT_HYST_LEAVE ({leave}) must be < L3_EXIT_HYST_ENTER ({enter})"
        )
    if enter - leave < gap_min:
        raise ValueError(
            f"L3 hysteresis gap too small: enter-leave={enter - leave:.4g} < L3_EXIT_HYST_MIN_GAP ({gap_min})"
        )


def _l3_finalize_exit_sample_weights(w: np.ndarray, *, stats_out: dict[str, Any] | None = None) -> np.ndarray:
    w64 = np.asarray(w, dtype=np.float64).ravel()
    pre = {
        "mean": float(np.mean(w64)),
        "std": float(np.std(w64)),
        "min": float(np.min(w64)),
        "max": float(np.max(w64)),
        "q01": float(np.quantile(w64, 0.01)),
        "q99": float(np.quantile(w64, 0.99)),
    }
    mode = (L3DEF.sample_weight_clip_mode() or "quantile").strip().lower()
    lo_e = L3DEF.sample_weight_clip_lo_str().strip()
    hi_e = L3DEF.sample_weight_clip_hi_str().strip()
    if lo_e and hi_e:
        lo, hi = float(lo_e), float(hi_e)
    elif mode in {"none", "off"}:
        lo, hi = float(np.min(w64)), float(np.max(w64))
    else:
        lo, hi = pre["q01"], pre["q99"]
    clipped = np.clip(w64, lo, hi)
    clipped = clipped / max(float(np.mean(clipped)), 1e-8)
    post = {
        "mean": float(np.mean(clipped)),
        "std": float(np.std(clipped)),
        "min": float(np.min(clipped)),
        "max": float(np.max(clipped)),
        "q01": float(np.quantile(clipped, 0.01)),
        "q99": float(np.quantile(clipped, 0.99)),
    }
    if stats_out is not None:
        stats_out.clear()
        stats_out.update({"pre_clip": pre, "post_clip": post, "clip_lo": lo, "clip_hi": hi, "clip_mode": mode})
    return clipped.astype(np.float32)


def _l3_oot_train_val_masks_by_trade(
    t_state: np.ndarray | pd.Series,
    rows_entry: np.ndarray,
    oot_mask: np.ndarray,
    *,
    train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Split OOT policy rows into train/val by whole trade (entry signal row id).

    Avoids putting prefixes of the same position in train and later bars in val, which
    would leak trajectory structure into GRU/LGBM validation.
    """
    n = int(rows_entry.shape[0])
    ts = np.asarray(pd.to_datetime(t_state))
    oot = np.asarray(oot_mask, dtype=bool)
    oot_idx = np.flatnonzero(oot)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    if oot_idx.size == 0:
        return train_mask, val_mask

    unique_e = np.unique(rows_entry[oot_idx])
    trade_t0: list[tuple[np.datetime64, int]] = []
    for e in unique_e:
        e = int(e)
        m = (rows_entry == e) & oot
        if not np.any(m):
            continue
        ix = np.flatnonzero(m)
        tmin = np.min(ts[ix])
        trade_t0.append((tmin, e))
    trade_t0.sort(key=lambda x: x[0])
    n_tr = len(trade_t0)
    if n_tr == 0:
        return train_mask, val_mask

    if n_tr >= 2:
        split = max(1, min(n_tr - 1, int(round(n_tr * train_frac))))
        train_entries = {e for _, e in trade_t0[:split]}
        val_entries = {e for _, e in trade_t0[split:]}
        for idx in oot_idx:
            e = int(rows_entry[idx])
            if e in train_entries:
                train_mask[idx] = True
            elif e in val_entries:
                val_mask[idx] = True
    else:
        strict = os.environ.get("L3_STRICT_TRADE_LEVEL_SPLIT", "1").strip().lower() in {"1", "true", "yes"}
        if strict:
            raise RuntimeError(
                "L3: only one distinct trade in OOT window; strict trade-level split forbids intra-trade train/val leakage. "
                "Expand OOT window (e.g. set L3_OOT_START earlier) before training."
            )
        print(
            "  [L3] WARNING: only one distinct entry in OOT window; val split is intra-trade "
            "(prefix leakage possible for GRU). Prefer more OOT trades or longer calendar span.",
            flush=True,
        )
        order = oot_idx[np.argsort(ts[oot_idx])]
        split = max(1, min(int(order.size) - 1, int(round(float(order.size) * train_frac))))
        train_mask[order[:split]] = True
        val_mask[order[split:]] = True

    if not val_mask.any() and oot_idx.size:
        val_mask[oot_idx[-1:]] = True
        train_mask[oot_idx[-1:]] = False
    train_entries = set(np.unique(rows_entry[np.asarray(train_mask & oot, dtype=bool)]).tolist())
    val_entries = set(np.unique(rows_entry[np.asarray(val_mask & oot, dtype=bool)]).tolist())
    overlap = train_entries & val_entries
    if overlap:
        raise RuntimeError(
            f"L3 trade-level split violation: {len(overlap)} trade_id(s) appear in both train and val (e.g. {sorted(list(overlap))[:5]})."
        )
    return train_mask, val_mask


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0)
    q = np.clip(q, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1).astype(np.float32)


def _split_l3_val_for_calibration(
    t_state: pd.Series | np.ndarray,
    val_mask: np.ndarray,
    *,
    tune_frac: float,
    min_rows_each: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    min_rows_each = int(L3DEF.val_split_min_rows_each() if min_rows_each is None else min_rows_each)
    base = np.asarray(val_mask, dtype=bool)
    idx = np.flatnonzero(base)
    tune_mask = np.zeros_like(base, dtype=bool)
    report_mask = np.zeros_like(base, dtype=bool)
    if idx.size == 0:
        return tune_mask, report_mask
    ts = np.asarray(pd.to_datetime(t_state))
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


def _l3_exit_hold_interaction_features_enabled() -> bool:
    """Hold×path interactions for exit (train + OOS must match)."""
    return True


def _l3_exit_scale_pos_weight_from_train(exit_pos_rate: float) -> float | None:
    """LightGBM ``scale_pos_weight`` when exit labels are very sparse or very dense (auto rule)."""
    p = float(exit_pos_rate)
    if not (0.0 < p < 1.0):
        return None
    if 0.02 <= p <= 0.20:
        return None
    return float((1.0 - p) / p)


def _fit_l3_exit_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> Any:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    n = y.size
    if n < 10 or len(np.unique(y)) < 2:
        print(
            f"  [CALIB] exit: skipped (n={n} or single class); using raw scores",
            flush=True,
        )
        return None
    min_cal = int(L3DEF.calib_min_rows())
    n_unique_p = int(len(np.unique(p)))
    if n < min_cal or n_unique_p < int(ISOTONIC_MIN_UNIQUE):
        print(
            f"  [CALIB] exit: isotonic skipped "
            f"(need n>={min_cal} and unique_p>={int(ISOTONIC_MIN_UNIQUE)}; got n={n}, unique_p={n_unique_p}); raw scores",
            flush=True,
        )
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    out: Any = ("isotonic", calib)
    print("  [CALIB] isotonic fitted", flush=True)
    p_cal = _apply_l3_exit_calibrator(p, out)
    label_mean = float(np.mean(y.astype(np.float64)))
    print(
        f"  [CALIB CHECK] before: mean={p.mean():.4f} std={p.std():.4f}  "
        f"after: mean={p_cal.mean():.4f} std={p_cal.std():.4f}  n={n}  label_mean={label_mean:.4f}",
        flush=True,
    )
    return out


def _apply_l3_exit_calibrator(p: np.ndarray, calibrator: Any) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    if isinstance(calibrator, tuple) and len(calibrator) == 2:
        tag, obj = calibrator[0], calibrator[1]
        if tag == "isotonic" and isinstance(obj, IsotonicRegression):
            return np.clip(np.asarray(obj.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)
    return arr


def _l3_log_exit_lgbm_train_start(label: str, X_tr: np.ndarray, feature_cols: list[str]) -> None:
    """Log proof that the exit head calls ``lgb.train`` (not loading a cached booster from disk)."""
    Xe = np.asarray(X_tr, dtype=np.float32, order="C")
    nr, nf = int(Xe.shape[0]), int(Xe.shape[1])
    n_expect = len(feature_cols)
    if nf != n_expect:
        print(
            f"  [L3 EXIT] ACTUALLY TRAINING exit LightGBM — {label}: SHAPE MISMATCH  X.shape[1]={nf}  len(feature_cols)={n_expect}",
            flush=True,
        )
    fc = [str(c) for c in feature_cols]
    tail5 = fc[-5:] if len(fc) >= 5 else fc
    last3_mean = float("nan")
    r0_tail3_abs = float("nan")
    if nr > 0 and nf > 0:
        try:
            c0 = max(0, nf - 3)
            last3_mean = float(np.mean(Xe[:, c0:].astype(np.float64)))
        except Exception:
            pass
        try:
            r0_tail3_abs = float(np.sum(np.abs(Xe[0, -min(3, nf) :].astype(np.float64))))
        except Exception:
            pass
    print(
        f"  [L3 EXIT] ACTUALLY TRAINING exit LightGBM — {label}: n_train_rows={nr:,}  n_features={nf}  "
        f"feature_tail={tail5!r}  train_mean_last3cols={last3_mean:.6g}  abs_sum_row0_last3cols={r0_tail3_abs:.6g}",
        flush=True,
    )


def _l3_entry_regime_ids_from_policy_X(X: np.ndarray, feature_cols: list[str]) -> np.ndarray | None:
    idxs: list[int] = []
    for i in range(len(L1A_REGIME_COLS)):
        c = f"l2_entry_regime_{i}"
        if c not in feature_cols:
            return None
        idxs.append(feature_cols.index(c))
    sub = np.asarray(X[:, idxs], dtype=np.float64)
    return np.argmax(sub, axis=1).astype(np.int32, copy=False)


def _l3_compute_adaptive_hold_sample_weights(
    y_exit: np.ndarray,
    hold_bars: np.ndarray,
    *,
    entry_regime_id: np.ndarray | None,
    unreal_pnl_atr: np.ndarray | None,
    regime_div: np.ndarray | None,
    max_hold_bars: int,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Per-row class weights: exit -> w_exit_base; hold -> w_hold_base * factors (clamped).

    ``w_hold_base`` defaults to 1.0 so neutral holds match exit (1.0); regime/time/pnl factors then
    move exhaust below 1.0 and breakout above 1.0. Base 1.5 made almost all holds > exit.
    Override: ``L3_ADAPTIVE_HOLD_W_BASE``.
    """
    cfg: dict[str, Any] = {
        "w_hold_base": 1.0,
        "w_exit_base": 1.0,
        "regime_mult": {
            0: 1.0,
            1: 1.3,
            2: 1.0,
            3: 0.55,
            4: 0.80,
        },
        "time_decay_start": 1.15,
        "time_decay_end": 0.65,
        "pnl_breakpoints": [-1.5, -0.5, 0.0],
        "pnl_multipliers": [0.50, 0.70, 0.85, 1.0],
        "div_threshold": 0.5,
        "div_low_mult": 1.0,
        "div_high_mult": 0.70,
        "w_hold_min": 0.4,
        "w_hold_max": 2.5,
    }
    if config:
        cfg = {**cfg, **{k: v for k, v in config.items() if k != "regime_mult"}}
        rm = config.get("regime_mult")
        if isinstance(rm, dict):
            base_rm = dict(cfg["regime_mult"])
            for rk, rv in rm.items():
                base_rm[int(rk)] = float(rv)
            cfg["regime_mult"] = base_rm

    y = np.asarray(y_exit, dtype=np.int32).ravel()
    n = int(y.size)
    weights = np.ones(n, dtype=np.float64)
    is_hold = y == 0
    is_exit = y == 1
    weights[is_exit] = float(cfg["w_exit_base"])
    hold_idx = np.flatnonzero(is_hold)
    if hold_idx.size == 0:
        return weights

    hold_w = np.full(hold_idx.size, float(cfg["w_hold_base"]), dtype=np.float64)
    regime_mult: dict[int, float] = cfg["regime_mult"]
    if entry_regime_id is not None:
        regimes = np.asarray(entry_regime_id, dtype=np.int64).ravel()[hold_idx]
        f_regime = np.array([float(regime_mult.get(int(r), 1.0)) for r in regimes], dtype=np.float64)
        hold_w *= f_regime

    hb = np.asarray(hold_bars, dtype=np.float64).ravel()[hold_idx]
    max_b = max(int(max_hold_bars), 2)
    t_frac = np.clip(hb / float(max(max_b - 1, 1)), 0.0, 1.0)
    f_time = float(cfg["time_decay_start"]) + (float(cfg["time_decay_end"]) - float(cfg["time_decay_start"])) * t_frac
    hold_w *= f_time

    if unreal_pnl_atr is not None:
        pnl = np.asarray(unreal_pnl_atr, dtype=np.float64).ravel()[hold_idx]
        bps = cfg["pnl_breakpoints"]
        mults = cfg["pnl_multipliers"]
        f_pnl = np.full(hold_idx.size, float(mults[-1]), dtype=np.float64)
        for i, bp in enumerate(bps):
            f_pnl = np.where(pnl < float(bp), float(mults[i]), f_pnl)
        hold_w *= f_pnl

    if regime_div is not None:
        div = np.asarray(regime_div, dtype=np.float64).ravel()[hold_idx]
        f_div = np.where(div > float(cfg["div_threshold"]), float(cfg["div_high_mult"]), float(cfg["div_low_mult"]))
        hold_w *= f_div

    hold_w = np.clip(hold_w, float(cfg["w_hold_min"]), float(cfg["w_hold_max"]))
    weights[hold_idx] = hold_w
    return weights


def _l3_exit_weight_bar_shape_enabled() -> bool:
    return False


def _l3_exit_weight_regret_enabled() -> bool:
    return False


def _l3_exit_oracle_path_scalars(
    rows_entry: np.ndarray,
    hold_bars: np.ndarray,
    unreal_atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row episode oracle quantities (training-only): unreal-at-peak hold, trade length, suffix min/max unreal."""
    ent = np.asarray(rows_entry, dtype=np.int64).ravel()
    h = np.asarray(hold_bars, dtype=np.float64).ravel()
    u = np.asarray(unreal_atr, dtype=np.float64).ravel()
    n = int(ent.size)
    peak_h = np.zeros(n, dtype=np.float64)
    tmax = np.zeros(n, dtype=np.float64)
    min_fw = np.zeros(n, dtype=np.float64)
    max_fw = np.zeros(n, dtype=np.float64)
    for e in np.unique(ent):
        idx = np.flatnonzero(ent == e)
        if idx.size == 0:
            continue
        hh = h[idx]
        order = idx[np.argsort(hh, kind="mergesort")]
        ho = h[order]
        uo = u[order]
        uo_f = np.where(np.isfinite(uo), uo, 0.0)
        peak_h[order] = float(ho[int(np.argmax(uo_f))])
        tmax[order] = float(np.max(ho))
        min_fw[order] = np.minimum.accumulate(uo_f[::-1])[::-1]
        max_fw[order] = np.maximum.accumulate(uo_f[::-1])[::-1]
    return peak_h, tmax, min_fw, max_fw


def _l3_exit_bar_shape_multipliers(
    hold_bars: np.ndarray,
    peak_hold: np.ndarray,
    trade_max_hold: np.ndarray,
) -> np.ndarray:
    peak_near = 3.0
    peak_win = 5
    early_lt = 3
    early_m = 0.3
    late_lt = 3
    late_m = 0.5
    h = np.asarray(hold_bars, dtype=np.float64).ravel()
    m = np.ones(h.size, dtype=np.float64)
    if early_lt > 0 and early_m != 1.0:
        m = np.where(h < float(early_lt), m * early_m, m)
    tm = np.asarray(trade_max_hold, dtype=np.float64).ravel()
    if late_lt > 0 and late_m != 1.0:
        m = np.where((tm - h) < float(late_lt), m * late_m, m)
    ph = np.asarray(peak_hold, dtype=np.float64).ravel()
    if peak_win > 0 and peak_near != 1.0:
        m = np.where(np.abs(h - ph) <= float(peak_win), m * peak_near, m)
    return m


def _l3_exit_regret_cost_multipliers(
    y_exit: np.ndarray,
    unreal_atr: np.ndarray,
    min_fwd: np.ndarray,
    max_fwd: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    u = np.asarray(unreal_atr, dtype=np.float64).ravel()
    lo_r = 1.0
    hi_r = 10.0
    lo_c = 0.5
    hi_c = 2.0
    opp_cap = 3.0
    mn = np.asarray(min_fwd, dtype=np.float64).ravel()
    mx = np.asarray(max_fwd, dtype=np.float64).ravel()
    regret_raw = np.maximum(u - mn, 0.0)
    w_exit = np.clip(1.0 + regret_raw, lo_r, hi_r)
    opp = np.maximum(mx - u, 0.0)
    frac = np.minimum(opp, opp_cap) / max(opp_cap, 1e-6)
    w_hold = np.clip(0.5 + frac * (hi_c - 0.5), lo_c, hi_c)
    return np.where(y == 1, w_exit, w_hold).astype(np.float64)


def _log_l3_adaptive_hold_weights(
    cls_w: np.ndarray,
    y_exit: np.ndarray,
    entry_regime_id: np.ndarray | None,
) -> None:
    w = np.asarray(cls_w, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    hold_mask = y == 0
    exit_mask = y == 1
    print("  [L3] adaptive hold weights (class multiplier, pre trade-norm):", flush=True)
    if hold_mask.any():
        wh = w[hold_mask]
        print(
            f"    hold samples: n={int(hold_mask.sum())}  w_mean={float(np.mean(wh)):.3f}  "
            f"w_std={float(np.std(wh)):.3f}  w_min={float(np.min(wh)):.3f}  w_max={float(np.max(wh)):.3f}",
            flush=True,
        )
    else:
        print("    hold samples: n=0", flush=True)
    if exit_mask.any():
        print(
            f"    exit samples: n={int(exit_mask.sum())}  w={float(np.mean(w[exit_mask])):.3f} (fixed)",
            flush=True,
        )
    else:
        print("    exit samples: n=0", flush=True)
    if entry_regime_id is not None and hold_mask.any():
        er = np.asarray(entry_regime_id, dtype=np.int64).ravel()
        for r in sorted(np.unique(er[hold_mask]).tolist()):
            mask = hold_mask & (er == int(r))
            if mask.any():
                print(
                    f"    regime={int(r)}: n_hold={int(mask.sum())}  w_mean={float(np.mean(w[mask])):.3f}",
                    flush=True,
                )
    if hold_mask.any() and exit_mask.any():
        eff_hold = float(np.sum(w[hold_mask]))
        eff_exit = float(np.sum(w[exit_mask]))
        raw_ratio = float(np.sum(hold_mask) / max(float(np.sum(exit_mask)), 1.0))
        print(
            f"    effective ratio (hold/exit): {eff_hold / max(eff_exit, 1e-9):.2f}:1  "
            f"(was {raw_ratio:.2f}:1 unweighted)",
            flush=True,
        )


def _l3_prepare_value_targets(
    y_value: np.ndarray,
    train_mask: np.ndarray,
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    vtm = _l3_value_target_mode()
    if vtm in ("peak_cls", "trade_outcome", "remaining_value"):
        y = np.clip(np.round(np.asarray(y_value, dtype=np.float64)), 0.0, 1.0).astype(np.float32)
        return y, {
            "value_target_mode": vtm,
            "clip_enabled": False,
            "clip_lo_q": 0.0,
            "clip_hi_q": 1.0,
            "clip_lo": 0.0,
            "clip_hi": 1.0,
            "train_clipped_frac": 0.0,
            "target_transform": "none",
            "objective": "binary",
            "metric": "auc",
        }
    if vtm == "remaining_value_atr":
        y = np.asarray(y_value, dtype=np.float32).copy()
        train = np.asarray(train_mask, dtype=bool).ravel()
        finite_train = train & np.isfinite(y)
        clip_enabled = False
        q_lo = 0.01
        q_hi = 0.99
        abs_cap = 0.0
        clip_lo = float("nan")
        clip_hi = float("nan")
        clipped_frac = 0.0
        if clip_enabled and finite_train.any():
            clip_lo = float(np.quantile(y[finite_train], q_lo))
            clip_hi = float(np.quantile(y[finite_train], q_hi))
            if abs_cap > 0.0:
                clip_lo = max(clip_lo, -abs_cap)
                clip_hi = min(clip_hi, abs_cap)
            if clip_hi < clip_lo:
                clip_lo, clip_hi = clip_hi, clip_lo
            below = y < clip_lo
            above = y > clip_hi
            clipped_frac = float(np.mean((below | above)[finite_train])) if finite_train.any() else 0.0
            y[below] = clip_lo
            y[above] = clip_hi
        transform = "none"
        obj = "regression"
        metric = "mae"
        stats: dict[str, float | str | bool] = {
            "value_target_mode": "remaining_value_atr",
            "clip_enabled": bool(clip_enabled),
            "clip_lo_q": float(q_lo),
            "clip_hi_q": float(q_hi),
            "clip_abs_cap": float(abs_cap),
            "clip_lo": float(clip_lo),
            "clip_hi": float(clip_hi),
            "train_clipped_frac": float(clipped_frac),
            "target_transform": str(transform),
            "objective": str(obj),
            "metric": str(metric),
        }
        if obj == "huber":
            stats["huber_alpha"] = 0.90
        elif obj == "fair":
            stats["fair_c"] = 1.0
        return y, stats
    y = np.asarray(y_value, dtype=np.float32).copy()
    train = np.asarray(train_mask, dtype=bool).ravel()
    finite_train = train & np.isfinite(y)
    if pa_state is not None and _l3_pa_targets_enabled():
        scale = pa_value_target_scale(pa_state, n=len(y))
        y = (np.asarray(y, dtype=np.float64) * scale).astype(np.float32)
    clip_enabled = True
    q_lo = 0.01
    q_hi = 0.99
    abs_cap = 0.0
    clip_lo = float("nan")
    clip_hi = float("nan")
    clipped_frac = 0.0
    if clip_enabled and finite_train.any():
        clip_lo = float(np.quantile(y[finite_train], q_lo))
        clip_hi = float(np.quantile(y[finite_train], q_hi))
        if abs_cap > 0.0:
            clip_lo = max(clip_lo, -abs_cap)
            clip_hi = min(clip_hi, abs_cap)
        if clip_hi < clip_lo:
            clip_lo, clip_hi = clip_hi, clip_lo
        below = y < clip_lo
        above = y > clip_hi
        clipped_frac = float(np.mean((below | above)[finite_train])) if finite_train.any() else 0.0
        y[below] = clip_lo
        y[above] = clip_hi
    transform = "signed_log1p"
    if transform == "signed_log1p":
        y = (np.sign(y.astype(np.float64)) * np.log1p(np.abs(y.astype(np.float64)))).astype(np.float32)
    objective = "huber"
    metric_default = "l1" if objective in {"huber", "fair"} else "l2"
    metric = metric_default
    stats: dict[str, float | str | bool] = {
        "value_target_mode": "regression",
        "clip_enabled": bool(clip_enabled),
        "clip_lo_q": float(q_lo),
        "clip_hi_q": float(q_hi),
        "clip_abs_cap": float(abs_cap),
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "train_clipped_frac": float(clipped_frac),
        "target_transform": str(transform),
        "objective": str(objective),
        "metric": str(metric),
    }
    if pa_state is not None and _l3_pa_targets_enabled():
        stats["pa_target_scaling"] = True
    if objective == "huber":
        stats["huber_alpha"] = 0.90
    elif objective == "fair":
        stats["fair_c"] = 1.0
    return y, stats


def _l3_inverse_value_target_transform(pred: np.ndarray, prep: dict[str, Any] | None = None) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float64).ravel()
    cfg = dict(prep or {})
    transform = str(cfg.get("target_transform", "none")).strip().lower() or "none"
    if transform == "signed_log1p":
        arr = np.sign(arr) * np.expm1(np.abs(arr))
    return arr


def _l3_value_lgb_params(exit_params: dict[str, Any], *, seed: int, prep: dict[str, float | str | bool]) -> dict[str, Any]:
    if str(prep.get("objective")) == "binary":
        p = {**exit_params, "objective": "binary", "metric": "auc", "seed": int(seed)}
        return p
    obj = str(prep.get("objective", "regression"))
    met = str(prep.get("metric", "l2"))
    if obj == "regression" and met == "mae":
        params = {**exit_params, "objective": "regression", "metric": "mae", "seed": int(seed)}
    else:
        params = {**exit_params, "objective": obj, "metric": met, "seed": int(seed)}
    if obj == "huber":
        params["alpha"] = float(prep.get("huber_alpha", 0.90))
    elif obj == "fair":
        params["fair_c"] = float(prep.get("fair_c", 1.0))
    return params


def _l3_value_hurdle_epsilon(y_value: np.ndarray, train_mask: np.ndarray) -> float:
    y = np.asarray(y_value, dtype=np.float64).ravel()
    tr = np.asarray(train_mask, dtype=bool).ravel() & np.isfinite(y)
    if not tr.any():
        return 0.05
    abs_y = np.abs(y[tr])
    pos = abs_y[abs_y > 0]
    if pos.size == 0:
        return 0.05
    adaptive = float(np.percentile(pos, 10.0))
    adaptive = max(adaptive, 0.05)
    hi = 0.35
    eps = float(np.clip(adaptive, 0.05, hi))
    print(
        f"  [HURDLE] adaptive |value| p10 (train, >0) → eps={eps:.4f}  (L3_VALUE_HURDLE_EPS_MAX={hi:.2f} cap)",
        flush=True,
    )
    return eps


def _l3_hurdle_nonzero_weights(y_nonzero: np.ndarray) -> np.ndarray:
    y = np.asarray(y_nonzero, dtype=np.int32).ravel()
    w = np.ones(len(y), dtype=np.float32)
    if y.size == 0:
        return w
    n0 = max(int(np.sum(y == 0)), 1)
    n1 = max(int(np.sum(y == 1)), 1)
    total = n0 + n1
    w0 = float(np.clip(np.sqrt(total / (2.0 * n0)), 0.5, 3.0))
    w1 = float(np.clip(np.sqrt(total / (2.0 * n1)), 0.5, 3.0))
    w[y == 0] = w0
    w[y == 1] = w1
    return w


def _l3_value_predict_hurdle(
    X: np.ndarray,
    value_reg_model: lgb.Booster | None,
    value_nonzero_model: lgb.Booster | None,
    *,
    prob_power: float = 1.0,
    prep: dict[str, Any] | None = None,
) -> np.ndarray:
    if value_reg_model is None:
        return np.zeros(len(X), dtype=np.float64)
    vtm = str(prep.get("value_target_mode", "")).strip().lower() if prep is not None else ""
    if prep is not None and vtm in ("peak_cls", "trade_outcome", "remaining_value"):
        return np.clip(value_reg_model.predict(X).astype(np.float64), 0.0, 1.0)
    mu_model = value_reg_model.predict(X).astype(np.float64)
    mu = _l3_inverse_value_target_transform(mu_model, prep)
    if value_nonzero_model is None:
        return mu
    p_nz = np.clip(value_nonzero_model.predict(X).astype(np.float64), 0.0, 1.0)
    pw = float(np.clip(prob_power, 0.5, 2.0))
    return mu * np.power(p_nz, pw)


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


def _hold_bucket_ids(hold_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(hold_values, dtype=np.float64).ravel()
    bins = np.asarray([3.0, 8.0, 15.0], dtype=np.float64)
    safe = np.nan_to_num(arr, nan=0.0)
    return np.searchsorted(bins, safe, side="right").astype(np.int32)


def _regime_ids_from_probs(regime_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(regime_probs, dtype=np.float64)
    safe = np.nan_to_num(probs, nan=0.0)
    return np.argmax(safe, axis=1).astype(np.int32) if safe.ndim == 2 and safe.shape[1] else np.zeros(len(safe), dtype=np.int32)


def _l3_pa_policy_enabled() -> bool:
    return True


def _l3_pa_targets_enabled() -> bool:
    return True


def _l3_pa_dict_from_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if not _l3_pa_targets_enabled():
        return {col: np.zeros(len(df), dtype=np.float32) for col in PA_STATE_FEATURES}
    return pa_state_arrays_from_frame(df)


def _l3_pa_dict_from_matrix(X: np.ndarray, feature_cols: list[str]) -> dict[str, np.ndarray]:
    if not _l3_pa_targets_enabled():
        return {col: np.zeros(len(X), dtype=np.float32) for col in PA_STATE_FEATURES}
    out: dict[str, np.ndarray] = {}
    for col in PA_STATE_FEATURES:
        if col in feature_cols:
            out[col] = np.asarray(X[:, feature_cols.index(col)], dtype=np.float32)
        else:
            out[col] = np.zeros(len(X), dtype=np.float32)
    return out


def _append_pa_bucket_to_state_keys(base_keys: np.ndarray, pa_bucket: np.ndarray | None) -> np.ndarray:
    if not _l3_pa_policy_enabled() or pa_bucket is None:
        return np.asarray(base_keys, dtype=object)
    bucket = np.asarray(pa_bucket, dtype=object).ravel()
    if bucket.size != len(base_keys):
        return np.asarray(base_keys, dtype=object)
    return np.asarray([f"{key}_pa_{pb}" for key, pb in zip(base_keys, bucket)], dtype=object)


def _state_keys_from_regime_vol(regime_probs: np.ndarray, vol_values: np.ndarray, *, vol_quantiles: list[float]) -> np.ndarray:
    reg = _regime_ids_from_probs(regime_probs)
    vb = _bucketize_by_quantiles(vol_values, vol_quantiles)
    return np.asarray([f"r{int(r)}_v{int(v)}" for r, v in zip(reg, vb)], dtype=object)


def _exit_coarse_bucket(regime_probs: np.ndarray) -> np.ndarray:
    """3 buckets: 0=bull (0–1), 1=range/vol (4–5), 2=bear/adverse (2–3)."""
    rid = _regime_ids_from_probs(regime_probs)
    sid = np.full(len(rid), 2, dtype=np.int32)
    sid[rid <= 1] = 0
    sid[rid >= 4] = 1
    return sid


def _l3_exit_state_granularity_mode() -> str:
    return "coarse"


def _l3_exit_state_pnl_deep_atr() -> float:
    return 1.0


def _l3_exit_state_regime_div_enabled() -> bool:
    return False


def _l3_exit_state_pnl_with_pa_enabled() -> bool:
    return False


def _l3_median_max_hold_per_entry(rows_entry: np.ndarray, hold_bars: np.ndarray) -> float:
    ent = np.asarray(rows_entry, dtype=np.int64).ravel()
    hold = np.asarray(hold_bars, dtype=np.float64).ravel()
    if ent.size == 0 or hold.size == 0 or ent.shape[0] != hold.shape[0]:
        return 1.0
    emax = int(np.max(ent)) if ent.size else 0
    if emax < 0:
        return 1.0
    max_h = np.full(emax + 1, -np.inf, dtype=np.float64)
    np.maximum.at(max_h, ent, hold)
    active = max_h[np.isfinite(max_h) & (max_h >= 0.0)]
    if active.size == 0:
        return 1.0
    return float(max(1.0, np.median(active)))


def _l3_exit_pnl_phase_calibration_from_slice(
    X: np.ndarray,
    feature_cols: list[str],
    rows_entry: np.ndarray,
) -> dict[str, Any]:
    hold_v = np.asarray(X[:, feature_cols.index("l3_hold_bars")], dtype=np.float64).ravel()
    med_ep = float(_l3_median_max_hold_per_entry(rows_entry, hold_v))
    out: dict[str, Any] = {
        "median_episode_hold": med_ep,
        "pnl_deep_atr": float(_l3_exit_state_pnl_deep_atr()),
        "regime_div_cut": None,
    }
    if _l3_exit_state_regime_div_enabled() and "l3_regime_divergence" in feature_cols:
        rd = np.asarray(X[:, feature_cols.index("l3_regime_divergence")], dtype=np.float64).ravel()
        finite = rd[np.isfinite(rd)]
        if finite.size:
            q = 0.65
            out["regime_div_cut"] = float(np.quantile(finite, q))
    return out


def _l3_exit_pnl_phase_state_keys_array(
    X: np.ndarray,
    feature_cols: list[str],
    rows_entry: np.ndarray,
    cal: Mapping[str, Any],
) -> np.ndarray:
    unreal = np.asarray(X[:, feature_cols.index("l3_unreal_pnl_atr")], dtype=np.float64).ravel()
    hold_v = np.asarray(X[:, feature_cols.index("l3_hold_bars")], dtype=np.float64).ravel()
    n = int(X.shape[0])
    if unreal.size != n or hold_v.size != n:
        raise ValueError("pnl_phase state keys: length mismatch")
    med_ep = float(max(1.0, float(cal["median_episode_hold"])))
    deep = float(cal["pnl_deep_atr"])
    half = 0.5 * med_ep
    u = np.nan_to_num(unreal, nan=0.0, posinf=0.0, neginf=0.0)
    prof = u >= 0.0
    au = np.abs(u)
    h = np.nan_to_num(hold_v, nan=0.0, posinf=0.0, neginf=0.0)
    code = np.full(n, 3, dtype=np.int32)
    code = np.where((~prof) & (au < deep), 2, code)
    code = np.where(prof & (h >= half), 1, code)
    code = np.where(prof & (h < half), 0, code)
    labels = np.asarray(["pnl_pf", "pnl_ps", "pnl_um", "pnl_ud"], dtype=object)
    base = labels[code]
    rd_cut = cal.get("regime_div_cut")
    if rd_cut is not None and "l3_regime_divergence" in feature_cols:
        rd = np.asarray(X[:, feature_cols.index("l3_regime_divergence")], dtype=np.float64).ravel()
        hi = np.isfinite(rd) & (rd >= float(rd_cut))
        out = np.empty(n, dtype=object)
        b_list = base.tolist()
        h_list = hi.tolist()
        for i in range(n):
            out[i] = f"{b_list[i]}_rd" if h_list[i] else b_list[i]
        return out
    return np.asarray(base.tolist(), dtype=object)


def _exit_state_keys_from_regime_vol_hold(
    regime_probs: np.ndarray,
    vol_values: np.ndarray,
    hold_values: np.ndarray,
    *,
    vol_quantiles: list[float],
) -> np.ndarray:
    mode = _l3_exit_state_granularity_mode()
    hb = _hold_bucket_ids(hold_values)
    if mode in {"full", "legacy", "fine"}:
        base = _state_keys_from_regime_vol(regime_probs, vol_values, vol_quantiles=vol_quantiles)
        return np.asarray([f"{b}_h{int(h)}" for b, h in zip(base, hb)], dtype=object)
    sid = _exit_coarse_bucket(regime_probs)
    return np.asarray([f"ex{int(s)}_h{int(h)}" for s, h in zip(sid, hb)], dtype=object)


def _l3_entry_policy_defaults() -> tuple[float, float]:
    return 0.0, 0.05


def _l3_trust_l2_entry_enabled() -> bool:
    """If True, skip entry-policy grid search: L2 is the sole entry layer; L3 only learns exit."""
    return False


def _l3_entry_policy_trust_l2_fixed() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    mc = 0.0
    ms = 0.0
    global_policy: dict[str, float] = {
        "min_confidence": mc,
        "min_size": ms,
        "score": float("nan"),
        "trade_rate": float("nan"),
        "precision_active": float("nan"),
        "correct_side": float("nan"),
        "avg_abs_edge": float("nan"),
    }
    return global_policy, {}


def _l3_lookup_policy_map(
    state_keys: np.ndarray,
    mapping: dict[str, dict[str, float]] | None,
    *,
    defaults: dict[str, float],
) -> dict[str, np.ndarray]:
    keys = np.asarray(state_keys, dtype=object).ravel()
    out = {name: np.full(len(keys), float(value), dtype=np.float32) for name, value in defaults.items()}
    for key, params in (mapping or {}).items():
        m = keys == key
        if not np.any(m):
            continue
        for name in out:
            if name in params:
                out[name][m] = float(params[name])
    return out


def _l3_exit_epsilon_atr() -> float:
    return 0.03


def _resolve_l3_policy_loss_buffer_atr(merged: pd.DataFrame, oot_mask: np.ndarray) -> tuple[float, dict[str, Any]]:
    mode = L3DEF.exit_loss_buffer_mode()
    if mode in {"data", "adaptive", "quantile", "oot_mae"}:
        pm = pd.to_numeric(merged.get("l2_pred_mae"), errors="coerce").to_numpy(dtype=np.float64)
        oot = np.asarray(oot_mask, dtype=bool)
        finite = oot & np.isfinite(pm)
        if not finite.any():
            base = float(L3DEF.exit_loss_buffer_atr_fixed())
            return base, {"mode": "data_fallback_fixed", "reason": "no_oot_l2_pred_mae", "resolved_atr": base}
        q = float(L3DEF.exit_loss_buffer_data_q())
        raw = float(np.percentile(pm[finite], q))
        base = float(max(0.0, raw * float(L3DEF.exit_loss_buffer_data_mult())))
        return base, {
            "mode": "data",
            "percentile_q": q,
            "multiplier": float(L3DEF.exit_loss_buffer_data_mult()),
            "raw_percentile_mae_atr": raw,
            "resolved_atr": base,
        }
    base = float(L3DEF.exit_loss_buffer_atr_fixed())
    return base, {"mode": "fixed", "resolved_atr": base}


def _l3_exit_live_edge_floor() -> float:
    return 0.02


def _l3_continuation_score(future_gain_left: np.ndarray, live_edge: np.ndarray) -> np.ndarray:
    future = np.asarray(future_gain_left, dtype=np.float64).ravel()
    edge = np.asarray(live_edge, dtype=np.float64).ravel()
    edge_mult = 0.20
    return (future + edge_mult * np.maximum(edge, 0.0)).astype(np.float32)


def _l3_target_horizon_bars(max_hold: int) -> int:
    return max(1, min(int(max_hold), 30))


def _l3_policy_exit_label_mode() -> str:
    raw = str(L3_VOL_EXIT_DEFAULTS.get("label_mode", "straddle_vol")).strip().lower()
    if raw in {"straddle_vol", "v2", "fwd_return", "legacy"}:
        return raw
    return "straddle_vol"


def _l3_exit_fwd_cost_atr() -> float:
    return 0.15


def _l3_exit_fwd_path_bad_atr() -> float:
    return 2.0


def _l3_exit_fwd_path_weak_mult() -> float:
    """With path stress, also exit when ``fwd_return_atr < mult * cost_atr`` (default 0.5 × cost)."""
    return 0.5


def _l3_compute_exit_label_fwd_return(
    close_seg: np.ndarray,
    *,
    side: float,
    atr_seg: np.ndarray,
    cost_atr: float,
    path_bad_atr: float,
    path_weak_mult: float,
) -> np.ndarray:
    """Forward-return exit labels with adaptive H, terminal vs cost, and path drawdown stress (directional)."""
    prices = np.asarray(close_seg, dtype=np.float64).ravel()
    atrs = np.maximum(np.asarray(atr_seg, dtype=np.float64).ravel(), 1e-9)
    n = int(prices.size)
    sgn = 1.0 if float(side) > 0.0 else -1.0
    out = np.ones(n, dtype=np.int32)
    c = float(cost_atr)
    pb = float(path_bad_atr)
    wm = float(path_weak_mult)
    for j in range(n):
        remaining = n - 1 - j
        if remaining <= 0:
            out[j] = 1
            continue
        H = int(min(remaining, max(5, remaining // 2)))
        fut_j = j + H
        cur = float(prices[j])
        fut = float(prices[fut_j])
        atr = float(atrs[j])
        fwd_return_atr = sgn * (fut - cur) / atr
        sl = prices[j + 1 : fut_j + 1]
        if sl.size == 0:
            worst_drawdown = 0.0
        elif sgn > 0:
            worst_drawdown = float(cur - np.min(sl)) / atr
        else:
            worst_drawdown = float(np.max(sl) - cur) / atr
        terminal_bad = fwd_return_atr < -c
        path_bad = worst_drawdown > pb
        weak = fwd_return_atr < wm * c
        out[j] = 1 if (terminal_bad or (weak and path_bad)) else 0
    return out


def _l3_value_target_mode() -> str:
    """Value head: ``peak_cls``, ``trade_outcome``, ``remaining_value``, ``remaining_value_atr``, or ``regression``."""
    return L3DEF.value_target_mode()


def _l3_remaining_value_labels_from_unreal(unreal_seg: np.ndarray) -> np.ndarray:
    """Binary y: 1 iff ``pnl_at_deadline - pnl_at_bar >= 0`` (ATR units; same as ``future_gain_left`` sign)."""
    u = np.asarray(unreal_seg, dtype=np.float64).ravel()
    if u.size == 0:
        return np.zeros(0, dtype=np.float32)
    t = float(u[-1])
    return (t - u >= 0.0).astype(np.float32)


def _l3_remaining_value_atr_from_unreal(unreal_seg: np.ndarray, atr_seg: np.ndarray) -> np.ndarray:
    """Regression y: (terminal unreal - current) / ATR per bar (signed; not clipped)."""
    u = np.asarray(unreal_seg, dtype=np.float64).ravel()
    if u.size == 0:
        return np.zeros(0, dtype=np.float32)
    a = np.asarray(atr_seg, dtype=np.float64).ravel()
    if a.size == 1 and u.size > 1:
        a = np.full_like(u, float(a[0]), dtype=np.float64)
    elif a.size != u.size:
        a = np.resize(a, u.shape)
    t = float(u[-1])
    denom = np.maximum(a, 1e-9)
    return ((t - u) / denom).astype(np.float32)


def _l3_value_upside_threshold_atr() -> float:
    return 0.5


def _l3_value_median_fit_mask(
    oot_mask: np.ndarray,
    train_mask: np.ndarray,
    gain_train_mask: np.ndarray,
    n_l3_oof: int,
) -> np.ndarray:
    """
    Boolean mask of **rows** used only to *estimate* the median of terminal trade PnL.
    When L3_OOF_FOLDS>=2, ``train_mask`` covers the full OOT pool, so we use ``gain_train_mask``
    (chronological head of OOT) to avoid val-tail trades influencing the threshold.
    Otherwise (trade-level 70/30), use ``train_mask`` only.
    """
    oot = np.asarray(oot_mask, dtype=bool)
    if n_l3_oof >= 2:
        return oot & np.asarray(gain_train_mask, dtype=bool)
    return np.asarray(train_mask, dtype=bool) & oot


def _l3_apply_trade_median_value_labels_train_frozen(
    X: np.ndarray,
    rows_entry: np.ndarray,
    feature_cols: list[str],
    median_fit_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Trade-level y: 1 iff terminal ``l3_unreal_pnl_atr`` (last episode bar) >= **median of terminals
    from trades that appear in ``median_fit_mask`` only**; frozen rule applied to *all* rows
    (val/holdout) so the threshold is not fit on held-out final outcomes.
    """
    uix = feature_cols.index("l3_unreal_pnl_atr")
    re = np.asarray(rows_entry, dtype=np.int64).ravel()
    u = np.asarray(X[:, uix], dtype=np.float64)
    mf = np.asarray(median_fit_mask, dtype=bool).ravel()
    n = int(re.size)
    if n == 0 or not mf.any():
        raise RuntimeError("L3 trade_outcome: empty dataset or empty median_fit_mask; cannot fit median threshold.")
    fit_trade_ids = np.unique(re[mf])
    if fit_trade_ids.size < 1:
        raise RuntimeError("L3 trade_outcome: no trade ids in median_fit_mask.")
    train_finals: list[float] = []
    for e in fit_trade_ids.tolist():
        m = re == e
        train_finals.append(float(u[m][-1]))
    med = float(np.median(np.asarray(train_finals, dtype=np.float64)))
    uniq, inv = np.unique(re, return_inverse=True)
    finals = np.empty(len(uniq), dtype=np.float64)
    for k, e in enumerate(uniq.tolist()):
        m = re == e
        finals[k] = float(u[m][-1])
    good = finals >= med
    y = good.astype(np.int32)[inv].astype(np.float32)
    out_meta = {
        "l3_value_trade_median_atr": med,
        "l3_value_trade_median_n_fit": int(fit_trade_ids.size),
        "l3_value_trade_n": int(len(uniq)),
        "l3_value_trade_pos": int(good.sum()),
    }
    pos_fit = int(np.sum(np.asarray(train_finals, dtype=np.float64) >= med))
    print(
        f"  [L3] trade_outcome: median (train-fit only) final unreal={med:.4f} ATR  "
        f"fit_trades={fit_trade_ids.size}  pos_among_fit={pos_fit}/{fit_trade_ids.size}  "
        f"all_trades_pos={out_meta['l3_value_trade_pos']}/{len(uniq)}  row_pos_rate={float(np.mean(y)):.3f}  n_rows={n}",
        flush=True,
    )
    return y, out_meta


def _l3_peak_upside_value_labels(
    *,
    close_seg: np.ndarray,
    high_seg: np.ndarray,
    low_seg: np.ndarray,
    side: float,
    play_straddle: bool,
    atr_scale: float,
    upside_threshold_atr: float,
) -> np.ndarray:
    """Binary value target: 1 iff max favorable excursion from *current* close over remaining path is ≥ threshold (ATR)."""
    n = int(len(close_seg))
    out = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return out
    c = close_seg.astype(np.float64, copy=False)
    hi = high_seg.astype(np.float64, copy=False)
    lo = low_seg.astype(np.float64, copy=False)
    atr = max(float(atr_scale), 1e-9)
    thr = float(upside_threshold_atr)
    rmax = np.maximum.accumulate(hi[::-1])[::-1]
    rmin = np.minimum.accumulate(lo[::-1])[::-1]
    best_hi = rmax[1:]
    best_lo = rmin[1:]
    if play_straddle:
        mx = np.maximum((best_hi - c[:-1]) / atr, (c[:-1] - best_lo) / atr)
    elif float(side) > 0.0:
        mx = (best_hi - c[:-1]) / atr
    else:
        mx = (c[:-1] - best_lo) / atr
    out[:-1] = (mx >= thr).astype(np.float32)
    return out


def _l3_policy_label_lookahead_bars() -> int:
    return 5


def _l3_policy_min_continuation_frac() -> float:
    return 0.0003


def _l3_compute_exit_label_v2(
    bar_returns_frac: np.ndarray,
    *,
    lookahead: int,
    min_continuation_frac: float,
) -> np.ndarray:
    """1=exit, 0=hold using forward max MTM (fractional) over the next ``lookahead`` bars."""
    x = np.asarray(bar_returns_frac, dtype=np.float64).ravel()
    n = int(x.shape[0])
    lk = max(1, int(lookahead))
    eps = float(max(0.0, min_continuation_frac))
    out = np.ones(n, dtype=np.int32)
    for j in range(n):
        end_idx = min(j + 1 + lk, n)
        fut = x[j + 1 : end_idx]
        if fut.size == 0:
            out[j] = 1
            continue
        upside = float(np.max(fut) - x[j])
        out[j] = 0 if upside >= eps else 1
    return out


def _l3_short_vol_range_explosion_mult() -> float:
    return float(np.clip(float(L3_VOL_EXIT_DEFAULTS.get("short_vol_range_explosion_mult", 1.50)), 1.0, 4.0))


def _l3_long_vol_theta_frac() -> float:
    return float(np.clip(float(L3_VOL_EXIT_DEFAULTS.get("long_vol_theta_frac", 0.08)), 0.0, 1.0))


def _l3_compute_exit_label_straddle_vol(
    range_frac: np.ndarray,
    *,
    lookahead: int,
    vol_side: float,
    implied_range_frac: float,
) -> np.ndarray:
    """Vol-only exit label: long_vol checks residual expansion; short_vol checks range explosion."""
    x = np.asarray(range_frac, dtype=np.float64).ravel()
    n = int(x.shape[0])
    out = np.ones(n, dtype=np.int32)
    lk = max(1, int(lookahead))
    implied = max(float(implied_range_frac), 1e-6)
    theta_frac = _l3_long_vol_theta_frac()
    explode_mult = _l3_short_vol_range_explosion_mult()
    for j in range(n):
        end_idx = min(j + 1 + lk, n)
        fut = x[j + 1 : end_idx]
        if fut.size == 0:
            out[j] = 1
            continue
        fut_max = float(np.max(fut))
        if float(vol_side) >= 0.0:
            future_add = max(fut_max - float(x[j]), 0.0)
            # Same units as ``x`` / ``implied`` (fractional path); scale theta by implied range magnitude.
            theta_proxy = float(theta_frac) * implied * np.sqrt((j + 1) / max(n, 1))
            out[j] = 1 if future_add < theta_proxy else 0
        else:
            out[j] = 1 if fut_max >= implied * explode_mult else 0
    return out


def _l3_compute_value_label_v2(bar_returns_frac: np.ndarray, *, lookahead: int) -> np.ndarray:
    x = np.asarray(bar_returns_frac, dtype=np.float64).ravel()
    n = int(x.shape[0])
    lk = max(1, int(lookahead))
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        end_idx = min(j + 1 + lk, n)
        fut = x[j + 1 : end_idx]
        if fut.size == 0:
            out[j] = 0.0
        else:
            out[j] = max(float(np.max(fut) - x[j]), 0.0)
    return out.astype(np.float32, copy=False)


def _l3_search_entry_policy(
    state_keys: np.ndarray,
    decision_class: np.ndarray,
    decision_confidence: np.ndarray,
    size: np.ndarray,
    edge_atr: np.ndarray,
    tau_edge: float,
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    cls = np.asarray(decision_class, dtype=np.int64).ravel()
    conf = np.asarray(decision_confidence, dtype=np.float64).ravel()
    sz = np.asarray(size, dtype=np.float64).ravel()
    edge = np.asarray(edge_atr, dtype=np.float64).ravel()
    keys = np.asarray(state_keys, dtype=object).ravel()
    truth_dir = np.full(len(edge), 1, dtype=np.int64)
    truth_dir[edge > tau_edge] = 0
    truth_dir[edge < -tau_edge] = 2
    truth_active = truth_dir != 1
    conf_default, size_default = _l3_entry_policy_defaults()
    _qc = float(np.quantile(conf[np.isfinite(conf)], 0.35)) if np.isfinite(conf).any() else conf_default
    conf_candidates = sorted({float(np.clip(v, 0.0, 1.0)) for v in [0.0, _qc, 0.35, 0.55]}) or [0.0]
    _qs = float(np.quantile(sz[np.isfinite(sz)], 0.35)) if np.isfinite(sz).any() else size_default
    size_candidates = sorted({float(np.clip(v, 0.0, 1.0)) for v in [0.0, _qs, 0.05, 0.12]}) or [0.0]
    def _search(mask: np.ndarray) -> dict[str, float]:
        best: dict[str, float] | None = None
        best_score = -1e18
        active_rate_target = float(np.mean(truth_active[mask])) if np.any(mask) else 0.0
        for min_conf in conf_candidates:
            for min_sz in size_candidates:
                entered = mask & (sz >= min_sz) & (conf >= min_conf) & np.isin(cls, [0, 2])
                if not np.any(entered):
                    continue
                correct_side = float(np.mean(cls[entered] == truth_dir[entered]))
                precision = float(np.mean(truth_active[entered]))
                avg_abs_edge = float(np.mean(np.abs(edge[entered])))
                trade_rate = float(np.mean(entered[mask])) if np.any(mask) else 0.0
                pa_bonus = 0.0
                if pa_state is not None and np.any(entered):
                    pa_bonus = pa_entry_policy_score_bonus_masked(pa_state, entered)
                score = 0.55 * precision + 0.30 * correct_side + 0.20 * avg_abs_edge - 0.20 * abs(trade_rate - active_rate_target) + pa_bonus
                if score > best_score:
                    best_score = score
                    best = {
                        "min_confidence": float(min_conf),
                        "min_size": float(min_sz),
                        "score": float(score),
                        "trade_rate": float(trade_rate),
                        "precision_active": float(precision),
                        "correct_side": float(correct_side),
                        "avg_abs_edge": float(avg_abs_edge),
                    }
        if best is None:
            best = {"min_confidence": conf_default, "min_size": size_default, "score": float("nan")}
        return best
    global_policy = _search(np.isfinite(conf) & np.isfinite(sz) & np.isfinite(edge))
    min_rows = max(80, 140)
    by_state: dict[str, dict[str, float]] = {}
    for key in sorted({str(k) for k in keys.tolist()}):
        m = (keys == key) & np.isfinite(conf) & np.isfinite(sz) & np.isfinite(edge)
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = _search(m)
    print(
        f"  [L3] entry policy: global(min_conf={global_policy['min_confidence']:.3f}, min_size={global_policy['min_size']:.3f})  "
        f"states={len(by_state)}",
        flush=True,
    )
    return global_policy, by_state


def _l3_target_horizon_by_state(
    state_keys: np.ndarray,
    peak_bar: np.ndarray,
    *,
    max_hold: int,
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[int, dict[str, int]]:
    base = _l3_target_horizon_bars(max_hold)
    keys = np.asarray(state_keys, dtype=object).ravel()
    peak = np.asarray(peak_bar, dtype=np.float64).ravel()
    use_peak_median = False
    uniform_h = int(np.clip(base, 1, int(max_hold)))
    by_state: dict[str, int] = {}
    min_rows = max(80, 120)
    if not use_peak_median:
        for key in sorted({str(k) for k in keys.tolist()}):
            m = keys == key
            if int(np.sum(m)) < min_rows:
                continue
            by_state[key] = uniform_h
        print(
            f"  [L3] target horizon: uniform={uniform_h}  states={len(by_state)}  "
            f"(L3_TARGET_HORIZON_USE_PEAK_MEDIAN=0)",
            flush=True,
        )
        return uniform_h, by_state
    if pa_state is not None and _l3_pa_targets_enabled():
        horizon_scale = pa_horizon_scale(pa_state, n=len(peak))
        peak = peak * horizon_scale
    finite = np.isfinite(peak)
    global_h = int(np.clip(np.nanmedian(peak[finite]) if finite.any() else base, 2, max_hold))
    for key in sorted({str(k) for k in keys.tolist()}):
        m = (keys == key) & finite
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = int(np.clip(np.nanmedian(peak[m]), 2, max_hold))
    print(f"  [L3] target horizon: global={global_h}  states={len(by_state)}  (peak-median mode)", flush=True)
    return global_h, by_state


def _l3_gain_select_required_names(feature_cols: list[str]) -> set[str]:
    """Columns that must stay in LGBM matrices for indexing or inference."""
    return {"l3_hold_bars", "l1a_vol_forecast", *L1A_REGIME_COLS}


def _l3_maybe_prune_features_by_exit_gain(
    *,
    X_lgb: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    static_cols: list[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    y_exit: np.ndarray,
    rows_entry: np.ndarray,
    pa_state_all: dict[str, np.ndarray],
    exit_params: dict[str, Any],
    es_rounds: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Train a short exit probe; keep features with >= min share of total gain (default 1%); cap/floor count."""
    meta: dict[str, Any] = {"enabled": True, "pruned": False}
    disabled = False
    if disabled:
        meta["enabled"] = False
        return X_lgb, X, feature_cols, static_cols, meta

    min_gain = float(np.clip(0.01, 1e-6, 0.5))
    max_n = 25
    min_n = 20
    probe_rounds = 120

    fc = list(feature_cols)
    n = len(fc)
    if n <= min_n:
        meta["skip_reason"] = "already_small"
        return X_lgb, X, fc, static_cols, meta

    ih = fc.index("l3_hold_bars")
    hold_tr = X_lgb[np.asarray(train_mask, dtype=bool), ih].astype(np.float64)
    tm_arr = np.asarray(train_mask, dtype=bool)
    w = _l3_trade_normalized_exit_weights(
        rows_entry[tm_arr],
        hold_tr,
        y_exit[tm_arr],
        pa_state={k: v[tm_arr] for k, v in pa_state_all.items()},
        X_policy=np.asarray(X_lgb[tm_arr], dtype=np.float32, order="C"),
        feature_cols=fc,
        max_hold_bars=int(L3DEF.max_hold_bars()),
    )
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, probe_rounds, "[L3] gain-probe")
    try:
        probe = lgb.train(
            dict(exit_params),
            lgb.Dataset(
                X_lgb[np.asarray(train_mask, dtype=bool)],
                label=y_exit[np.asarray(train_mask, dtype=bool)],
                weight=w.astype(np.float32),
                feature_name=fc,
                free_raw_data=False,
            ),
            num_boost_round=probe_rounds,
            valid_sets=[
                lgb.Dataset(
                    X_lgb[np.asarray(val_mask, dtype=bool)],
                    label=y_exit[np.asarray(val_mask, dtype=bool)],
                    feature_name=fc,
                    free_raw_data=False,
                )
            ],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()

    gains = np.asarray(probe.feature_importance(importance_type="gain"), dtype=np.float64)
    gsum = float(np.sum(gains))
    if not np.isfinite(gsum) or gsum <= 0:
        meta["skip_reason"] = "zero_gain"
        return X_lgb, X, fc, static_cols, meta

    frac = gains / gsum
    req_names = _l3_gain_select_required_names(fc)
    always_idx: set[int] = set()
    for name in req_names:
        if name in fc:
            always_idx.add(fc.index(name))
    if len(always_idx) > max_n:
        max_n = len(always_idx)
        meta["max_n_bumped_to_required"] = int(max_n)

    optional_idx = [i for i in range(n) if i not in always_idx]
    selected: set[int] = set(always_idx)
    for i in optional_idx:
        if float(frac[i]) >= min_gain:
            selected.add(i)

    if len(selected) > max_n:
        opt_sorted = sorted((i for i in selected if i not in always_idx), key=lambda i: float(frac[i]))
        for i in opt_sorted:
            if len(selected) <= max_n:
                break
            selected.remove(i)

    if len(selected) < min_n:
        rest = [i for i in range(n) if i not in selected]
        rest.sort(key=lambda i: float(frac[i]), reverse=True)
        for i in rest:
            selected.add(i)
            if len(selected) >= min_n:
                break

    sel_idx = sorted(selected)
    if sel_idx == list(range(n)):
        meta["pruned"] = False
        return X_lgb, X, fc, static_cols, meta

    static_new = [fc[i] for i in sel_idx if not fc[i].startswith("l3_traj_emb_")]
    try:
        X_new = X[:, [static_cols.index(fc[i]) for i in sel_idx if not fc[i].startswith("l3_traj_emb_")]].astype(
            np.float32, copy=False
        )
    except ValueError as ex:
        meta["skip_reason"] = f"static_index:{ex}"
        return X_lgb, X, fc, static_cols, meta

    X_lgb_new = X_new
    feature_out = static_new

    dropped = [fc[j] for j in range(n) if j not in set(sel_idx)]
    meta.update(
        {
            "pruned": True,
            "n_before": n,
            "n_after": len(feature_out),
            "min_gain_frac": min_gain,
            "max_n": max_n,
            "min_n": min_n,
            "probe_rounds": probe_rounds,
            "dropped": dropped,
            "kept": list(feature_out),
        }
    )
    print(
        f"  [L3] gain feature select: {n}->{len(feature_out)} cols "
        f"(gain>={min_gain:.2%} of total; cap={max_n} floor={min_n}; probe_rounds={probe_rounds})",
        flush=True,
    )
    return X_lgb_new, X_new, feature_out, static_new, meta


def _l3_exit_policy_row_state_keys(
    X: np.ndarray,
    feature_cols: list[str],
    *,
    vol_quantiles: list[float],
    rows_entry: np.ndarray | None = None,
    pnl_phase_cal: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if _l3_exit_state_granularity_mode() == "pnl_phase":
        if rows_entry is None or pnl_phase_cal is None:
            raise ValueError("pnl_phase exit states require rows_entry and pnl_phase_cal")
        base = _l3_exit_pnl_phase_state_keys_array(X, feature_cols, rows_entry, pnl_phase_cal)
        pa_bucket = None
        if _l3_pa_policy_enabled() and _l3_exit_state_pnl_with_pa_enabled() and all(col in feature_cols for col in PA_STATE_FEATURES):
            pa_bucket = pa_state_bucket_labels_from_arrays(
                {col: np.asarray(X[:, feature_cols.index(col)], dtype=np.float32) for col in PA_STATE_FEATURES},
                length=len(X),
            )
        return _append_pa_bucket_to_state_keys(base, pa_bucket)
    reg_cols = [feature_cols.index(c) for c in L1A_REGIME_COLS]
    vol_idx = feature_cols.index("l1a_vol_forecast")
    hold_idx = feature_cols.index("l3_hold_bars")
    regime_probs = np.asarray(X[:, reg_cols], dtype=np.float32)
    vol_values = np.asarray(X[:, vol_idx], dtype=np.float32)
    hold_values = np.asarray(X[:, hold_idx], dtype=np.float32)
    base = _exit_state_keys_from_regime_vol_hold(regime_probs, vol_values, hold_values, vol_quantiles=vol_quantiles)
    pa_bucket = None
    if _l3_pa_policy_enabled() and all(col in feature_cols for col in PA_STATE_FEATURES):
        pa_bucket = pa_state_bucket_labels_from_arrays(
            {col: np.asarray(X[:, feature_cols.index(col)], dtype=np.float32) for col in PA_STATE_FEATURES},
            length=len(X),
        )
    return _append_pa_bucket_to_state_keys(base, pa_bucket)


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != v.size:
        return float(np.mean(v))
    w = np.clip(w, 0.0, np.inf)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / s)


def _l3_trade_uplift_vs_deadline(
    pred: np.ndarray,
    rows_entry: np.ndarray,
    unreal_atr: np.ndarray,
    hold_bars: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per trade: MTM at first simulated exit bar minus MTM at episode end (deadline), in ATR units.

    If every bar in the episode has ``pred==0`` (no modeled exit), the simulated exit bar is the
    last bar — same as the deadline — so ``uplift == 0`` (finite, included in aggregates; not NaN).
    """
    pred = np.asarray(pred, dtype=np.int32).ravel()
    ent = np.asarray(rows_entry, dtype=np.int64).ravel()
    u = np.asarray(unreal_atr, dtype=np.float64).ravel()
    h = np.asarray(hold_bars, dtype=np.int64).ravel()
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
    n = int(pred.shape[0])
    if n == 0 or ent.shape[0] != n or u.shape[0] != n or h.shape[0] != n:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if sw is not None and sw.shape[0] != n:
        sw = None
    deltas: list[float] = []
    weights: list[float] = []
    for e in np.unique(ent):
        m = ent == e
        if not np.any(m):
            continue
        hh = h[m]
        order = np.argsort(hh, kind="mergesort")
        u_e = u[m][order]
        p_e = pred[m][order]
        unreal_dead = float(u_e[-1])
        ex = np.flatnonzero(p_e == 1)
        j = int(ex[0]) if ex.size else len(u_e) - 1
        unreal_strat = float(u_e[j])
        deltas.append(unreal_strat - unreal_dead)
        if sw is None:
            weights.append(float(np.sum(m)))
        else:
            weights.append(float(np.sum(sw[m])))
    if not deltas:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.asarray(deltas, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _l3_pred_exit_tune_from_policies(
    prob: np.ndarray,
    hold: np.ndarray,
    state_keys: np.ndarray,
    global_pol: Mapping[str, float],
    by_state: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Match ``_l3_search_exit_policy`` thresholding (early/late split; prob-only)."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    hold = np.asarray(hold, dtype=np.float64).ravel()
    keys = np.asarray(state_keys, dtype=object).ravel()
    n = int(prob.shape[0])
    pred = np.zeros(n, dtype=np.int32)
    for key in sorted({str(k) for k in keys.tolist()}):
        m = keys == key
        if not np.any(m):
            continue
        pol = {**dict(global_pol), **dict(by_state.get(str(key), {}))}
        p_early = float(pol.get("exit_prob_threshold_early", pol.get("exit_prob_threshold", 0.55)))
        p_late = float(pol.get("exit_prob_threshold_late", pol.get("exit_prob_threshold", 0.55)))
        early_bar = float(pol.get("early_hold_split_bar", 3))
        thr = np.where(hold[m] < early_bar, p_early, p_late)
        pb = prob[m]
        pred[m] = (pb >= thr).astype(np.int32)
    return pred


def _l3_log_tune_trade_uplift_report(
    *,
    prob_cal: np.ndarray,
    value_pred: np.ndarray,
    y_value_row: np.ndarray,
    hold: np.ndarray,
    rows_entry: np.ndarray,
    unreal: np.ndarray,
    state_keys: np.ndarray,
    tune_weights: np.ndarray,
    exit_policy: Mapping[str, float],
    exit_by_state: Mapping[str, Mapping[str, float]],
) -> None:
    """Post policy-search log: trade uplift vs deadline, exit rate, per-state hints."""
    print("\n  [L3] tune — trade uplift vs deadline (selected policy, diagnostic)", flush=True)
    pred = _l3_pred_exit_tune_from_policies(
        prob_cal,
        hold,
        state_keys,
        exit_policy,
        exit_by_state,
    )
    exit_rate = float(np.mean(pred))
    deltas, tw = _l3_trade_uplift_vs_deadline(
        pred,
        rows_entry,
        unreal,
        hold.astype(np.int64, copy=False),
        tune_weights,
    )
    ent = np.asarray(rows_entry, dtype=np.int64).ravel()
    h = np.asarray(hold, dtype=np.int64).ravel()
    pr = np.asarray(pred, dtype=np.int32).ravel()
    sk = np.asarray(state_keys, dtype=object).ravel()
    n_trades = 0
    n_never_exit = 0
    max_abs_uplift_never_exit = 0.0
    for e in np.unique(ent):
        m = ent == e
        order = np.argsort(h[m], kind="mergesort")
        p_e = pr[m][order]
        u_e = unreal[m][order].astype(np.float64)
        n_trades += 1
        if not np.any(p_e == 1):
            n_never_exit += 1
            dead = float(u_e[-1])
            ex = np.flatnonzero(p_e == 1)
            j = int(ex[0]) if ex.size else len(u_e) - 1
            max_abs_uplift_never_exit = max(max_abs_uplift_never_exit, abs(float(u_e[j] - dead)))
    n_nan = int(np.sum(~np.isfinite(deltas))) if deltas.size else 0
    if deltas.size:
        mu = float(_weighted_mean(deltas, tw))
        p10 = float(_weighted_quantile(deltas, 0.10, tw))
        pos_w = float(np.sum(tw[deltas > 0])) if tw.size == deltas.size else 0.0
        tot_w = float(np.sum(tw)) if tw.size else 0.0
        frac_w_pos = float(pos_w / tot_w) if tot_w > 0 else float("nan")
    else:
        mu = float("nan")
        p10 = float("nan")
        frac_w_pos = float("nan")
    print(
        f"    bar_exit_rate={exit_rate:.4f}  (deploy sanity: often ~0.03–0.15; extreme ~0.001 suggests thresholds too high)",
        flush=True,
    )
    print(
        f"    n_trades={n_trades:,}  weighted_mean_uplift_ATR={mu:.6f}  weighted_p10_uplift_ATR={p10:.6f}  "
        f"weighted_frac_uplift>0={frac_w_pos:.4f}",
        flush=True,
    )
    print(
        f"    trades_never_exit_under_policy={n_never_exit:,}  "
        f"max|strat-dead|_on_those={max_abs_uplift_never_exit:.2e}  (expect 0; same as deadline path)",
        flush=True,
    )
    if n_nan:
        print(f"    [L3][warn] non-finite trade uplifts: {n_nan}", flush=True)

    sum_w = defaultdict(float)
    sum_wd = defaultdict(float)
    n_t = defaultdict(int)
    for e in np.unique(ent):
        m = ent == e
        ix = np.where(m)[0]
        i0 = int(ix[np.argmin(h[ix])])
        st = str(sk[i0])
        order = np.argsort(h[m], kind="mergesort")
        u_e = unreal[m][order].astype(np.float64)
        p_e = pr[m][order]
        dead = float(u_e[-1])
        ex = np.flatnonzero(p_e == 1)
        j = int(ex[0]) if ex.size else len(u_e) - 1
        du = float(u_e[j] - dead)
        wt = float(np.sum(tune_weights[m])) if tune_weights.size == pr.size else float(np.sum(m))
        sum_w[st] += wt
        sum_wd[st] += wt * du
        n_t[st] += 1
    if sum_w:
        means = {s: float(sum_wd[s] / max(sum_w[s], 1e-12)) for s in sum_w}
        sorted_states = sorted(means.items(), key=lambda x: x[1])
        neg = [s for s, v in means.items() if v < 0]
        med_st = float(np.median(np.asarray(list(means.values()), dtype=np.float64))) if means else 0.0
        hi = [s for s, v in means.items() if med_st > 0 and v > 2.0 * med_st]
        print(
            f"    by_state (first-bar key): n_states={len(means)}  "
            f"states_with_mean_uplift<0={len(neg)}  → prefer higher p_exit threshold (fewer exits) there",
            flush=True,
        )
        print(
            f"    states_with_mean_uplift>>median (heuristic)={len(hi)}  → exit timing useful; review thresholds",
            flush=True,
        )
        lo3 = sorted_states[: min(3, len(sorted_states))]
        hi3 = sorted_states[-min(3, len(sorted_states)) :]
        print(f"    lowest mean_uplift states (sample): {[(a, f'{b:.5f}') for a, b in lo3]}", flush=True)
        print(f"    highest mean_uplift states (sample): {[(a, f'{b:.5f}') for a, b in hi3]}", flush=True)


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    qq = float(np.clip(q, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(v, qq))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != v.size:
        return float(np.quantile(v, qq))
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(np.sum(ok)) == 0:
        return float(np.quantile(v[np.isfinite(v)], qq)) if np.isfinite(v).any() else float("nan")
    v_ok = v[ok]
    w_ok = w[ok]
    order = np.argsort(v_ok, kind="mergesort")
    v_s = v_ok[order]
    w_s = w_ok[order]
    cdf = np.cumsum(w_s)
    if cdf[-1] <= 0:
        return float(np.quantile(v_ok, qq))
    cdf = cdf / cdf[-1]
    idx = int(np.searchsorted(cdf, qq, side="left"))
    idx = min(max(idx, 0), len(v_s) - 1)
    return float(v_s[idx])


def _l3_economic_uplift_score(
    pred: np.ndarray,
    value_hold: np.ndarray,
    *,
    tail_weight: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Single objective used by both search and evaluation."""
    p = np.asarray(pred, dtype=np.int32).ravel()
    v = np.asarray(value_hold, dtype=np.float64).ravel()
    if p.size != v.size or p.size == 0:
        return {
            "uplift_mean": float("nan"),
            "uplift_p10": float("nan"),
            "score": float("nan"),
        }
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
    if sw is not None and sw.size != p.size:
        sw = None
    realized = np.where(p == 1, -v, v).astype(np.float64, copy=False)
    uplift_mean = float(_weighted_mean(realized, sw) - _weighted_mean(v, sw))
    uplift_p10 = float(_weighted_quantile(realized, 0.10, sw) - _weighted_quantile(v, 0.10, sw))
    score = float(uplift_mean + float(tail_weight) * uplift_p10)
    return {
        "uplift_mean": uplift_mean,
        "uplift_p10": uplift_p10,
        "score": score,
    }


def _estimate_half_life_ar1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    x_lag = x[:-1]
    y = x[1:]
    if np.std(x_lag) <= 1e-8 or np.std(y) <= 1e-8:
        return float("nan")
    rho = float(np.corrcoef(x_lag, y)[0, 1])
    if not np.isfinite(rho) or rho <= 0.0 or abs(rho) < 1e-6 or abs(rho) >= 0.999:
        return float("nan")
    return float(-np.log(2.0) / np.log(abs(rho)))


def _l3_exp_decay_weights(
    tune_idx: np.ndarray,
    t_state: pd.Series,
    signal: np.ndarray,
    *,
    min_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(tune_idx.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64), {"mode": "empty"}
    if n < min_rows:
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": f"rows<{min_rows}", "half_life_rows": float("nan")}
    ts = pd.to_datetime(np.asarray(t_state)[tune_idx], errors="coerce")
    sig = np.asarray(signal, dtype=np.float64).ravel()
    sig = sig[tune_idx]
    valid = np.isfinite(sig) & pd.notna(ts)
    if int(np.sum(valid)) < max(10, min_rows // 4):
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": "insufficient_valid_signal", "half_life_rows": float("nan")}
    day = pd.to_datetime(ts[valid]).floor("D")
    day_signal = pd.Series(sig[valid]).groupby(day).mean().to_numpy(dtype=np.float64)
    hl_days = _estimate_half_life_ar1(day_signal)
    if not np.isfinite(hl_days) or hl_days <= 0:
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": "unstable_half_life", "half_life_rows": float("nan")}
    unique_days = max(1, int(len(np.unique(day))))
    rows_per_day = max(1.0, float(n) / float(unique_days))
    hl_rows = float(np.clip(hl_days * rows_per_day, 1.0, max(1.0, n * 1.5)))
    lambda_ = float(np.log(2.0) / max(hl_rows, 1e-6))
    age = np.arange(n, dtype=np.float64)[::-1]
    w = np.exp(-lambda_ * age)
    w = w / max(float(np.sum(w)), 1e-12)
    return w, {"mode": "exp_decay", "half_life_days": float(hl_days), "half_life_rows": float(hl_rows), "lambda": lambda_}


def _l3_merge_exit_policy_state_overrides(by_state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return by_state


def _l3_policy_adaptive_grid_enabled() -> bool:
    """Percentile-based p_exit / value grids on val_tune (``L3_POLICY_ADAPTIVE_GRID``)."""
    return L3DEF.policy_adaptive_grid()


def _l3_policy_min_exit_rate() -> float:
    """Search-space floor: skip candidates whose implied pred exit_rate is below this (``L3_POLICY_MIN_EXIT_RATE``)."""
    return float(L3DEF.policy_min_exit_rate())


def _l3_policy_min_exit_rate_enforce() -> bool:
    # In pure-economic objective mode, keep this as a hard safety guard.
    return True


def _l3_policy_max_exit_rate() -> float:
    """Hard safety cap to avoid degenerate all-exit solutions."""
    return 0.60


def _l3_adaptive_percentile_list() -> list[int]:
    return [2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98]


def _l3_build_percentile_threshold_grid(
    arr: np.ndarray,
    qs: list[int],
    clip_lo: float | None,
    clip_hi: float | None,
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        mid = 0.5 * (float(clip_lo) + float(clip_hi)) if clip_lo is not None and clip_hi is not None else 0.0
        return np.asarray([round(mid, 4)], dtype=np.float64)
    pts = [round(float(np.percentile(a, float(q))), 4) for q in qs]
    u = np.unique(np.asarray(pts, dtype=np.float64))
    if clip_lo is not None and clip_hi is not None:
        u = np.unique(np.clip(u, float(clip_lo), float(clip_hi)))
    return u


def _l3_search_exit_policy(
    exit_prob: np.ndarray,
    value_pred: np.ndarray,
    y_exit: np.ndarray,
    *,
    y_value_true: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    hold_bars: np.ndarray | None = None,
    rows_entry: np.ndarray | None = None,
    unreal_atr: np.ndarray | None = None,
    report_guardrail: Mapping[str, float] | None = None,
    value_target_mode: str | None = None,
    early_delta_candidates: list[float] | None = None,
    record_prob_stats: bool = False,
) -> dict[str, float]:
    prob = np.asarray(exit_prob, dtype=np.float64).ravel()
    value = np.asarray(value_pred, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    valid = np.isfinite(prob) & np.isfinite(value)
    vtm = str(value_target_mode or _l3_value_target_mode())
    if not valid.any():
        return {
            "exit_prob_threshold": 0.55,
            "exit_prob_threshold_early": 0.55,
            "exit_prob_threshold_late": 0.55,
            "early_prob_threshold_delta": 0.0,
            "early_hold_split_bar": 3,
            "score": float("nan"),
        }
    prob = prob[valid]
    value = value[valid]
    y = y[valid]
    value_true = None if y_value_true is None else np.asarray(y_value_true, dtype=np.float64).ravel()[valid]
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()[valid]
        sw = np.clip(sw, 0.0, np.inf)
    else:
        sw = None
    if value_true is not None and value_true.size and vtm not in ("peak_cls", "trade_outcome", "remaining_value"):
        bias = float(np.median(value_true - value))
        value_adj = value + bias
    else:
        value_adj = value
    value_pred_std = float(np.std(value))
    value_target_std = float(np.std(value_true)) if value_true is not None and value_true.size else float(np.std(value_adj))
    policy_params = derive_policy_params(
        y.astype(np.float64),
        value_true if value_true is not None and value_true.size else value_adj,
        value_pred_std=value_pred_std,
        value_target_std=value_target_std,
    )

    p_clip_lo = float(L3DEF.exit_prob_floor())
    p_clip_hi = float(L3DEF.exit_prob_ceil())
    adaptive_grid = _l3_policy_adaptive_grid_enabled()
    pct_qs = _l3_adaptive_percentile_list()
    if adaptive_grid:
        prob_candidates = _l3_build_percentile_threshold_grid(prob, pct_qs, p_clip_lo, p_clip_hi)
    else:
        q_lo = 0.45
        q_hi = 0.97
        n_q = 19
        q_prob = np.linspace(q_lo, q_hi, n_q)
        prob_candidates = np.unique(np.clip(np.quantile(prob, q_prob), p_clip_lo, p_clip_hi))
    # Unified economic objective mode with hard anti-degeneracy safety bounds.
    target_exit_rate = float("nan")
    hold_recall_floor = 0.0
    exit_rate_cap = float(_l3_policy_max_exit_rate())
    report_exit_hint = float("nan")
    report_hold_floor_hint = float("nan")
    if report_guardrail is not None:
        report_exit_hint = float(report_guardrail.get("exit_rate_hint", float("nan")))
        report_hold_floor_hint = float(report_guardrail.get("hold_recall_floor_hint", float("nan")))
    hold_arr = np.zeros_like(prob) if hold_bars is None else np.asarray(hold_bars, dtype=np.float64).ravel()[valid]
    hold_i = np.asarray(hold_arr, dtype=np.int64).ravel()
    rows_e: np.ndarray | None
    unreal_e: np.ndarray | None
    if rows_entry is not None and unreal_atr is not None:
        rows_e = np.asarray(rows_entry, dtype=np.int64).ravel()[valid]
        unreal_e = np.asarray(unreal_atr, dtype=np.float64).ravel()[valid]
    else:
        rows_e = None
        unreal_e = None
    if not (
        rows_e is not None
        and unreal_e is not None
        and hold_bars is not None
        and int(rows_e.shape[0]) == int(prob.shape[0])
    ):
        raise RuntimeError(
            "L3 exit policy search requires trade-aligned rows_entry, unreal_atr, and l3_hold_bars "
            "(per-bar economic uplift objective was removed)."
        )
    early_split_enabled = True
    early_hold_bar = 3
    early_delta_candidates_eff: list[float]
    if not early_split_enabled:
        early_delta_candidates_eff = [0.0]
    elif early_delta_candidates is not None:
        early_delta_candidates_eff = [float(x) for x in early_delta_candidates]
    else:
        early_delta_candidates_eff = [-0.08, -0.04, 0.0, 0.04]
    utility_tail_weight = float(L3DEF.policy_utility_tail_weight())
    min_exit_rate = float(_l3_policy_min_exit_rate())
    min_exit_enforce = bool(_l3_policy_min_exit_rate_enforce())
    best: dict[str, float] | None = None
    best_relaxed: dict[str, float] | None = None
    best_under_cap: dict[str, float] | None = None
    best_under_floor: dict[str, float] | None = None
    n_total = 0
    n_under_cap = 0
    n_under_floor = 0
    n_passing = 0
    n_skipped_min_exit = 0
    print("  [L3] exit policy utility: trade_level_uplift (deadline MTM vs first exit bar)", flush=True)
    if adaptive_grid:
        print(
            f"  [L3] adaptive policy grid: percentiles={pct_qs}  "
            f"|p|={len(prob_candidates)}  "
            f"min_exit_rate>={min_exit_rate:.4f} (enforce={min_exit_enforce})  "
            f"max_exit_rate<={exit_rate_cap:.4f}",
            flush=True,
        )

    def _metrics_from_pred(pred0: np.ndarray) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        bool,
    ]:
        pred = np.asarray(pred0, dtype=np.int32).ravel()
        f1 = float(f1_score(y, pred, zero_division=0))
        acc = float(accuracy_score(y, pred))
        exit_rate = float(_weighted_mean(pred.astype(np.float64), sw))
        hold_mask = y == 0
        pred_hold = pred == 0
        hold_recall = (
            float(_weighted_mean((pred[hold_mask] == 0).astype(np.float64), None if sw is None else sw[hold_mask]))
            if hold_mask.any()
            else 0.0
        )
        hold_precision = (
            float(_weighted_mean((y[pred_hold] == 0).astype(np.float64), None if sw is None else sw[pred_hold]))
            if pred_hold.any()
            else 0.0
        )
        exit_recall = (
            float(_weighted_mean((pred[y == 1] == 1).astype(np.float64), None if sw is None else sw[y == 1]))
            if np.any(y == 1)
            else 0.0
        )
        miss_exit = float("nan")
        utility_mean = float("nan")
        utility_p10 = float("nan")
        utility_score = float("nan")
        hold_recall_contrib = 0.0
        exit_rate_penalty_contrib = 0.0
        skip_hr_gate = True
        deltas, tw = _l3_trade_uplift_vs_deadline(pred, rows_e, unreal_e, hold_i, sw)
        if deltas.size:
            utility_mean = float(_weighted_mean(deltas, tw))
            utility_p10 = float(_weighted_quantile(deltas, 0.10, tw))
            utility_score = float(utility_mean + utility_tail_weight * utility_p10)
        score = float(utility_score)
        return (
            f1,
            acc,
            exit_rate,
            hold_recall,
            hold_precision,
            exit_recall,
            miss_exit,
            utility_mean,
            utility_p10,
            utility_score,
            hold_recall_contrib,
            exit_rate_penalty_contrib,
            score,
            skip_hr_gate,
        )

    prob_iter = prob_candidates.tolist()
    for prob_thr in prob_iter:
        for early_delta in early_delta_candidates_eff:
            n_total += 1
            prob_thr_late = float(prob_thr)
            prob_thr_early = float(np.clip(prob_thr + early_delta, p_clip_lo, p_clip_hi))
            thr_vec = np.where(hold_arr < float(early_hold_bar), prob_thr_early, prob_thr_late)
            pred = (prob >= thr_vec).astype(np.int32)
            exit_rate_pred = float(np.mean(pred.astype(np.float64)))
            if min_exit_enforce and exit_rate_pred + 1e-15 < min_exit_rate:
                n_skipped_min_exit += 1
                continue
            (
                f1,
                acc,
                exit_rate,
                hold_recall,
                hold_precision,
                exit_recall,
                miss_exit,
                utility_mean,
                utility_p10,
                utility_score,
                hold_recall_contrib,
                exit_rate_penalty_contrib,
                score,
                skip_hr_gate,
            ) = _metrics_from_pred(pred)
            cand = {
                "exit_prob_threshold": float(prob_thr),
                "exit_prob_threshold_early": float(prob_thr_early),
                "exit_prob_threshold_late": float(prob_thr_late),
                "early_prob_threshold_delta": float(early_delta),
                "early_hold_split_bar": int(early_hold_bar),
                "score": float(score),
                "utility_mean": float(utility_mean),
                "utility_p10": float(utility_p10),
                "f1": float(f1),
                "acc": float(acc),
                "exit_rate": float(exit_rate),
                "target_exit_rate": float(target_exit_rate),
                "max_exit_rate": float(exit_rate_cap),
                "min_hold_recall": float(hold_recall_floor),
                "hold_recall": float(hold_recall),
                "hold_precision": float(hold_precision),
                "exit_recall": float(exit_recall),
                "hold_recall_contrib": float(hold_recall_contrib),
                "exit_rate_penalty_contrib": float(exit_rate_penalty_contrib),
            }
            if best_relaxed is None or float(cand["score"]) > float(best_relaxed["score"]):
                best_relaxed = cand
            if exit_rate <= exit_rate_cap:
                n_under_cap += 1
                if best_under_cap is None or float(cand["score"]) > float(best_under_cap["score"]):
                    best_under_cap = cand
            if skip_hr_gate or hold_recall >= hold_recall_floor:
                n_under_floor += 1
                if best_under_floor is None or float(cand["score"]) > float(best_under_floor["score"]):
                    best_under_floor = cand
            if not skip_hr_gate and hold_recall < hold_recall_floor:
                continue
            if exit_rate > exit_rate_cap:
                continue
            n_passing += 1
            if best is None or float(cand["score"]) > float(best["score"]):
                best = cand
    if min_exit_enforce and best is None and best_relaxed is None and int(prob.size) > 0:
        p_med = float(np.median(prob))
        for early_delta in [0.0]:
            prob_thr_late = p_med
            prob_thr_early = float(np.clip(p_med + float(early_delta), p_clip_lo, p_clip_hi))
            thr_vec = np.where(hold_arr < float(early_hold_bar), prob_thr_early, prob_thr_late)
            pred_fb = (prob >= thr_vec).astype(np.int32)
            (
                f1,
                acc,
                exit_rate,
                hold_recall,
                hold_precision,
                exit_recall,
                miss_exit,
                utility_mean,
                utility_p10,
                utility_score,
                hold_recall_contrib,
                exit_rate_penalty_contrib,
                score,
                skip_hr_gate,
            ) = _metrics_from_pred(pred_fb)
            fb = {
                "exit_prob_threshold": float(p_med),
                "exit_prob_threshold_early": float(prob_thr_early),
                "exit_prob_threshold_late": float(p_med),
                "early_prob_threshold_delta": float(early_delta),
                "early_hold_split_bar": int(early_hold_bar),
                "score": float(score),
                "utility_mean": float(utility_mean),
                "utility_p10": float(utility_p10),
                "f1": float(f1),
                "acc": float(acc),
                "exit_rate": float(exit_rate),
                "target_exit_rate": float(target_exit_rate),
                "max_exit_rate": float(exit_rate_cap),
                "min_hold_recall": float(hold_recall_floor),
                "hold_recall": float(hold_recall),
                "hold_precision": float(hold_precision),
                "exit_recall": float(exit_recall),
                "hold_recall_contrib": float(hold_recall_contrib),
                "exit_rate_penalty_contrib": float(exit_rate_penalty_contrib),
            }
            best = dict(fb)
            best_relaxed = dict(fb)
        print(
            f"  [L3][warn] min_exit_rate: no policy candidate met floor={min_exit_rate:.4f}; "
            f"fallback median p_exit={p_med:.4f}",
            flush=True,
        )
    if best is None:
        fallback_reason = "best_relaxed"
        fallback = best_relaxed
        if best_under_cap is not None:
            fallback_reason = "exit_rate_cap"
            fallback = best_under_cap
        elif best_under_floor is not None:
            fallback_reason = "hold_recall_floor"
            fallback = best_under_floor
        print(
            f"  [L3][warn] EXIT POLICY FALLBACK: no candidate met guardrails "
            f"(hold_recall>={hold_recall_floor:.4f}, exit_rate<={exit_rate_cap:.4f}); "
            f"fallback={fallback_reason}.",
            flush=True,
        )
        best = dict(fallback) if fallback is not None else {
            "exit_prob_threshold": 0.55,
            "exit_prob_threshold_early": 0.55,
            "exit_prob_threshold_late": 0.55,
            "early_prob_threshold_delta": 0.0,
            "early_hold_split_bar": int(early_hold_bar),
            "score": float("nan"),
            "utility_mean": float("nan"),
            "utility_p10": float("nan"),
            "f1": float("nan"),
            "acc": float("nan"),
            "exit_rate": float("nan"),
            "target_exit_rate": float(target_exit_rate),
            "max_exit_rate": float(exit_rate_cap),
            "min_hold_recall": float(hold_recall_floor),
            "hold_recall": float("nan"),
            "hold_precision": float("nan"),
            "exit_recall": float("nan"),
            "hold_recall_contrib": float("nan"),
            "exit_rate_penalty_contrib": float("nan"),
        }
    print("\n  [L3] exit policy derivation on val_tune", flush=True)
    print(
        f"  [L3] derived policy params: objective=economic_uplift(mean + {utility_tail_weight:.2f}*p10)  "
        f"hold_recall_floor={hold_recall_floor:.4f}  target_exit_rate={target_exit_rate:.4f}  "
        f"exit_rate_cap={exit_rate_cap:.4f}  report_exit_hint={report_exit_hint:.4f}  "
        f"report_hold_floor_hint={report_hold_floor_hint:.4f}  split_early_hold={early_split_enabled}({early_hold_bar})  "
        f"diag_hold_rate={policy_params.diag_hold_rate:.4f}  diag_opp_cost={policy_params.diag_opp_cost:.4f}  "
        f"diag_save_benefit={policy_params.diag_save_benefit:.4f}  diag_value_head_degen={policy_params.diag_value_head_degen}",
        flush=True,
    )
    if best_relaxed is not None:
        print(
            f"  [L3] unconstrained best: p_exit>={best_relaxed['exit_prob_threshold']:.4f}  "
            f"exit_rate={best_relaxed.get('exit_rate', float('nan')):.3f}  "
            f"hold_recall={best_relaxed.get('hold_recall', float('nan')):.4f}  score={best_relaxed.get('score', float('nan')):.4f}",
            flush=True,
        )
    print(
        f"  [L3] selected exit policy (prob_only): p_exit>={best['exit_prob_threshold']:.4f}  "
        f"utility_mean={best.get('utility_mean', float('nan')):.4f}  "
        f"utility_p10={best.get('utility_p10', float('nan')):.4f}  F1={best.get('f1', float('nan')):.4f}  acc={best.get('acc', float('nan')):.4f}  "
        f"exit_rate={best.get('exit_rate', float('nan')):.3f}  hold_recall={best.get('hold_recall', float('nan')):.4f}  "
        f"hold_precision={best.get('hold_precision', float('nan')):.4f}  "
        f"p_exit_early={best.get('exit_prob_threshold_early', float('nan')):.4f}  "
        f"p_exit_late={best.get('exit_prob_threshold_late', float('nan')):.4f}  "
        f"early_delta={best.get('early_prob_threshold_delta', float('nan')):+.4f}  "
        f"exit_rate_penalty_contrib={best.get('exit_rate_penalty_contrib', float('nan')):.4f}  "
        f"hold_recall_contrib={best.get('hold_recall_contrib', float('nan')):.4f}  "
        f"candidates_passing_floor={n_under_floor}/{n_total}  "
        f"candidates_under_cap={n_under_cap}/{n_total}  "
        f"candidates_guarded={n_passing}/{n_total}  "
        f"skipped_min_exit={n_skipped_min_exit}",
        flush=True,
    )
    out = dict(best)
    if record_prob_stats:
        out["prob_search_stats"] = {
            "mean": float(np.mean(prob)),
            "std": float(np.std(prob)),
            "p25": float(np.percentile(prob, 25)),
            "p50": float(np.percentile(prob, 50)),
            "p75": float(np.percentile(prob, 75)),
            "grid_used_p": [float(x) for x in np.asarray(prob_candidates, dtype=np.float64).ravel()],
            "grid_used_v": [],
            "min_exit_rate_constraint": float(min_exit_rate),
            "adaptive_grid": bool(adaptive_grid),
            "percentiles": [int(x) for x in pct_qs],
            "n_skipped_min_exit": int(n_skipped_min_exit),
            "value_adj_mean": float(np.mean(value_adj)) if value_adj.size else float("nan"),
            "value_adj_std": float(np.std(value_adj)) if value_adj.size else float("nan"),
        }
    return out


def _l3_trade_normalized_exit_weights(
    rows_entry: np.ndarray,
    hold_bars: np.ndarray,
    y_exit: np.ndarray,
    pa_state: dict[str, np.ndarray] | None = None,
    *,
    weight_stats: dict[str, Any] | None = None,
    X_policy: np.ndarray,
    feature_cols: list[str],
    max_hold_bars: int | None = None,
) -> np.ndarray:
    entry = np.asarray(rows_entry, dtype=np.int64).ravel()
    hold = np.asarray(hold_bars, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    if len(entry) == 0:
        return np.empty(0, dtype=np.float32)
    xp = np.asarray(X_policy)
    if int(xp.shape[0]) != int(y.size):
        raise ValueError(
            f"L3 exit weights: X_policy rows ({int(xp.shape[0])}) must match y_exit ({int(y.size)})."
        )
    uniq, inv, counts = np.unique(entry, return_inverse=True, return_counts=True)
    del uniq
    trade_norm = 1.0 / np.maximum(counts[inv].astype(np.float64), 1.0)
    hb = _hold_bucket_ids(hold)
    hb_counts = np.bincount(hb.astype(np.int32), minlength=4).astype(np.float64)
    hb_counts = np.maximum(hb_counts, 1.0)
    hold_w = np.asarray([float(np.clip(np.sqrt(np.mean(hb_counts) / hb_counts[int(h)]), 0.75, 1.5)) for h in hb], dtype=np.float64)
    entry_regime_id = _l3_entry_regime_ids_from_policy_X(xp, feature_cols)
    unreal_atr = (
        np.asarray(xp[:, feature_cols.index("l3_unreal_pnl_atr")], dtype=np.float64)
        if "l3_unreal_pnl_atr" in feature_cols
        else None
    )
    regime_div = (
        np.asarray(xp[:, feature_cols.index("l3_regime_divergence")], dtype=np.float64)
        if "l3_regime_divergence" in feature_cols
        else None
    )
    mh = int(L3DEF.max_hold_bars() if max_hold_bars is None else max_hold_bars)
    cls_w = _l3_compute_adaptive_hold_sample_weights(
        y,
        hold,
        entry_regime_id=entry_regime_id,
        unreal_pnl_atr=unreal_atr,
        regime_div=regime_div,
        max_hold_bars=mh,
    )
    if weight_stats is not None:
        _log_l3_adaptive_hold_weights(cls_w, y, entry_regime_id)
        hm = y == 0
        em = y == 1
        weight_stats["adaptive_hold_weights"] = {
            "hold_w_mean": float(np.mean(cls_w[hm])) if hm.any() else float("nan"),
            "exit_w_mean": float(np.mean(cls_w[em])) if em.any() else float("nan"),
        }
    w = trade_norm * hold_w * cls_w
    need_path = _l3_exit_weight_bar_shape_enabled() or _l3_exit_weight_regret_enabled()
    if need_path:
        if unreal_atr is None:
            if weight_stats is not None:
                weight_stats["exit_path_reweight_skipped"] = "missing l3_unreal_pnl_atr"
        else:
            uvec = unreal_atr.ravel()
            peak_h, tmax_h, min_fw, max_fw = _l3_exit_oracle_path_scalars(entry, hold, uvec)
            if _l3_exit_weight_bar_shape_enabled():
                bsm = _l3_exit_bar_shape_multipliers(hold, peak_h, tmax_h)
                w = w * bsm
                if weight_stats is not None:
                    weight_stats["bar_shape_weight_mult_mean"] = float(np.mean(bsm))
            if _l3_exit_weight_regret_enabled():
                rcm = _l3_exit_regret_cost_multipliers(y, uvec, min_fw, max_fw)
                w = w * rcm
                if weight_stats is not None:
                    weight_stats["regret_cost_weight_mult_mean"] = float(np.mean(rcm))
                    weight_stats["regret_mult_y1_mean"] = float(np.mean(rcm[y == 1])) if np.any(y == 1) else float("nan")
                    weight_stats["regret_mult_y0_mean"] = float(np.mean(rcm[y == 0])) if np.any(y == 0) else float("nan")
    if pa_state is not None and _l3_pa_targets_enabled():
        pa_w = pa_exit_trade_weight_multiplier(pa_state, n=len(w))
        w *= pa_w
    w = w / max(float(np.mean(w)), 1e-8)
    return _l3_finalize_exit_sample_weights(w, stats_out=weight_stats)


def _l3_boost_rounds() -> int:
    return 250


def _l3_exit_boost_rounds() -> int:
    return 300


def _l3_early_stopping_rounds() -> int:
    return 80


def _l3_lgb_params(prefix: str, *, seed_default: int) -> dict[str, Any]:
    if prefix in {"L3_EXIT", "L3_STATIC_EXIT"}:
        return {
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "lambda_l1": 0.2,
            "lambda_l2": 2.0,
            "seed": int(seed_default),
        }
    return {
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": int(seed_default),
    }


def _l3_value_atr_lgb_hyperparams(*, seed_default: int = 72) -> dict[str, Any]:
    """Tighter LGBM hyperparameters for ``remaining_value_atr``."""
    return {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 200,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": int(seed_default),
    }


def _l3_value_feature_indices(feature_cols: list[str]) -> tuple[list[int], list[str]]:
    """Column indices/names: ``feature_cols`` minus blacklist, union ``L3_VALUE_EXTRA_ALLOWED`` present in X."""
    bl = set(L3_VALUE_FEATURE_BLACKLIST)
    idx_set: set[int] = {i for i, c in enumerate(feature_cols) if c not in bl}
    names = {feature_cols[i] for i in idx_set}
    for c in L3_VALUE_EXTRA_ALLOWED:
        if c in names:
            continue
        if c in feature_cols:
            idx_set.add(int(feature_cols.index(c)))
            names.add(c)
    idx = sorted(idx_set)
    if not idx:
        raise RuntimeError(
            "L3: value-head feature blacklist removed all columns; check L3_VALUE_FEATURE_BLACKLIST and gain-select."
        )
    return idx, [feature_cols[i] for i in idx]


def _l3_print_value_head_top_gain_features(value_model: lgb.Booster) -> None:
    imp = value_model.feature_importance(importance_type="gain")
    feat_names = value_model.feature_name()
    top20 = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:20]
    print("  [L3] value head top-20 features (gain):", flush=True)
    for name, score in top20:
        print(f"    {name}: {score:.1f}", flush=True)


def l3_entry_side_from_l2(decision_class: int, decision_confidence: float, size: float, *, min_confidence: float, min_size: float) -> float:
    if float(size) < float(min_size) or float(decision_confidence) < float(min_confidence):
        return 0.0
    if int(decision_class) == 0:
        return 1.0
    if int(decision_class) == 2:
        return -1.0
    return 0.0


def l3_cox_covariate_names_from_features(feature_cols: list[str]) -> list[str]:
    _ = feature_cols
    return []


def l3_infer_cox_features(
    cox_bundle: Mapping[str, Any] | None,
    X_base: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    _ = (cox_bundle, X_base, feature_cols)
    return np.zeros(2, dtype=np.float32)


def l3_load_cox_bundle(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    _ = meta
    return None


def _l3_bayes_weights() -> dict[str, float]:
    return {
        "fav": 0.28,
        "adv": -0.35,
        "regime": -0.45,
        "gate": -0.18,
        "gate_thr": -0.12,
    }


def _l3_intra_episode_frac_derivatives(bar_ret_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling path stats on fractional mark-to-market series (matches _build_l3_policy_dataset loop)."""
    n_steps = int(len(bar_ret_frac))
    br = bar_ret_frac.astype(np.float64, copy=False)
    ret_last_3 = np.zeros(n_steps, dtype=np.float32)
    ret_last_5 = np.zeros(n_steps, dtype=np.float32)
    for j in range(n_steps):
        j3 = j - 3 if j >= 3 else 0
        j5 = j - 5 if j >= 5 else 0
        ret_last_3[j] = np.float32(float(br[j] - br[j3]))
        ret_last_5[j] = np.float32(float(br[j] - br[j5]))
    d1 = np.empty(n_steps, dtype=np.float64)
    d1[0] = br[0]
    if n_steps > 1:
        d1[1:] = np.diff(br)
    vol_trade = np.zeros(n_steps, dtype=np.float32)
    for j in range(n_steps):
        segd = d1[: j + 1]
        vol_trade[j] = np.float32(float(np.std(segd))) if segd.size > 1 else np.float32(0.0)
    slope_fb = np.zeros(n_steps, dtype=np.float32)
    t_idx = np.arange(n_steps, dtype=np.float64)
    for j in range(n_steps):
        if j == 0:
            slope_fb[j] = np.float32(0.0)
            continue
        tt = t_idx[: j + 1]
        yy = br[: j + 1]
        xm = float(np.mean(tt))
        ym = float(np.mean(yy))
        num = float(np.sum((tt - xm) * (yy - ym)))
        den = float(np.sum((tt - xm) ** 2))
        slope_fb[j] = np.float32(num / den) if den > 1e-18 else np.float32(0.0)
    return ret_last_3, ret_last_5, vol_trade, slope_fb


def _l3_episode_momentum_leading_block(
    n_steps: int,
    unreal_seg: np.ndarray,
    drawdown_from_peak: np.ndarray,
    live_mfe_seg: np.ndarray,
    live_mae_seg: np.ndarray,
    vol_surprise_seg: np.ndarray,
    bar_ret_frac: np.ndarray,
    holds: np.ndarray,
    target_horizon: int,
    pred_mfe: float,
) -> np.ndarray:
    """PnL/DD/MAE velocities, vol-ratio acceleration, edge/time consumption, path shape (causal)."""
    u = unreal_seg.astype(np.float64, copy=False)
    dd = drawdown_from_peak.astype(np.float64, copy=False)
    mfe = live_mfe_seg.astype(np.float64, copy=False)
    mae = live_mae_seg.astype(np.float64, copy=False)
    vs = vol_surprise_seg.astype(np.float64, copy=False)
    br = bar_ret_frac.astype(np.float64, copy=False)
    h = holds.astype(np.float64, copy=False)
    th = max(1, int(target_horizon))
    pm = float(pred_mfe)

    unreal_vel = np.zeros(n_steps, dtype=np.float64)
    unreal_accel = np.zeros(n_steps, dtype=np.float64)
    dd_vel = np.zeros(n_steps, dtype=np.float64)
    mae_vel = np.zeros(n_steps, dtype=np.float64)
    bars_since_mfe = np.zeros(n_steps, dtype=np.float64)
    mfe_stale = np.zeros(n_steps, dtype=np.float64)
    pnl_vs_mfe = np.zeros(n_steps, dtype=np.float64)
    vol_ratio_acc = np.zeros(n_steps, dtype=np.float64)
    edge_cons = np.zeros(n_steps, dtype=np.float64)
    time_cons = np.zeros(n_steps, dtype=np.float64)
    edge_per_rem = np.zeros(n_steps, dtype=np.float64)
    path_smooth = np.zeros(n_steps, dtype=np.float64)
    adv_run = np.zeros(n_steps, dtype=np.float64)
    mfe_die = np.zeros(n_steps, dtype=np.float64)

    running_mfe_peak = float(mfe[0])
    mfe_peak_j = 0
    run_adv = 0
    stale_thr = max(1.0, float(th) / 3.0)

    for j in range(n_steps):
        j3 = j - 3 if j >= 3 else 0
        unreal_vel[j] = (u[j] - u[j3]) / 3.0
        j3b = j - 3
        if j3b >= 3:
            vel_at_j3 = (u[j3b] - u[j3b - 3]) / 3.0
            unreal_accel[j] = unreal_vel[j] - vel_at_j3
        else:
            unreal_accel[j] = 0.0

        dd_vel[j] = (dd[j] - dd[j3]) / 3.0
        mae_vel[j] = (mae[j] - mae[j3]) / 3.0

        if mfe[j] >= running_mfe_peak - 1e-9:
            running_mfe_peak = float(mfe[j])
            mfe_peak_j = j
        bars_since_mfe[j] = float(j - mfe_peak_j)
        mfe_stale[j] = 1.0 if bars_since_mfe[j] > stale_thr else 0.0
        pnl_vs_mfe[j] = float(u[j]) / (max(float(mfe[j]), 0.0) + 1e-6)

        if j >= 2:
            d0 = vs[j] - vs[j - 1]
            d1 = vs[j - 1] - vs[j - 2]
            vol_ratio_acc[j] = float(d0 / (abs(d1) + 1e-6))
        elif j == 1:
            vol_ratio_acc[j] = float(vs[j] - vs[j - 1])
        else:
            vol_ratio_acc[j] = 0.0

        edge_cons[j] = float(u[j]) / (abs(pm) + 1e-6)
        time_cons[j] = float(h[j]) / float(th)
        rem_b = float(max(1.0, th - h[j]))
        rem_e = float(max(0.0, pm) - max(float(u[j]), 0.0))
        edge_per_rem[j] = rem_e / rem_b

        seg = br[: j + 1]
        if seg.size > 1:
            sm = float(np.mean(seg))
            ss = float(np.std(seg))
            path_smooth[j] = float(ss / (abs(sm) + 1e-8))
        else:
            path_smooth[j] = 0.0

        if j == 0:
            run_adv = 0
        elif u[j] < u[j - 1] - 1e-9:
            run_adv += 1
        else:
            run_adv = 0
        adv_run[j] = float(run_adv)

        j5 = max(0, j - 5)
        mfe_die[j] = 1.0 if float(mfe[j] - mfe[j5]) < 0.0 else 0.0

    return np.column_stack(
        [
            unreal_vel,
            unreal_accel,
            dd_vel,
            mae_vel,
            bars_since_mfe,
            mfe_stale,
            pnl_vs_mfe,
            vol_ratio_acc,
            edge_cons,
            time_cons,
            edge_per_rem,
            path_smooth,
            adv_run,
            mfe_die,
        ]
    ).astype(np.float32, copy=False)


def _l3_episode_aux_feature_block(
    *,
    n_steps: int,
    idx_arr: np.ndarray,
    entry_i: int,
    side: float,
    decision_class: np.ndarray,
    decision_conf: np.ndarray,
    size: np.ndarray,
    neutral: np.ndarray,
    entry_regime_row: np.ndarray,
    current_regime: np.ndarray,
    unreal_seg: np.ndarray,
    min_conf_arr: np.ndarray,
    min_size_arr: np.ndarray,
    drawdown_from_peak: np.ndarray,
) -> np.ndarray:
    """Entry decay, counterfactual regret, and Bayesian trade-quality path (one row per hold step)."""
    w = _l3_bayes_weights()
    dec_e = float(decision_conf[entry_i])
    gate_e = float(1.0 - neutral[entry_i])
    rid_e = int(np.argmax(entry_regime_row.astype(np.float64)))
    sgn_side = 1.0 if side > 0 else -1.0

    signal_conf_decay = np.empty(n_steps, dtype=np.float32)
    direction_agree = np.empty(n_steps, dtype=np.float32)
    regime_changed = np.empty(n_steps, dtype=np.float32)
    gate_curr = np.empty(n_steps, dtype=np.float32)
    gate_decay = np.empty(n_steps, dtype=np.float32)
    would_enter = np.empty(n_steps, dtype=np.float32)
    regret_ratio = np.empty(n_steps, dtype=np.float32)
    bars_since_peak = np.empty(n_steps, dtype=np.float32)
    at_new_high = np.empty(n_steps, dtype=np.float32)
    regret_velocity = np.empty(n_steps, dtype=np.float32)
    quality_bayes = np.empty(n_steps, dtype=np.float32)

    p0 = float(np.clip(dec_e, 0.05, 0.95))
    log_odds = float(np.log(p0 / (1.0 - p0)))

    running_peak = float(unreal_seg[0])
    peak_idx = 0
    peak_cum = np.maximum.accumulate(unreal_seg.astype(np.float64))

    for j in range(n_steps):
        k = int(idx_arr[j])
        signal_conf_decay[j] = np.float32(decision_conf[k] - dec_e)
        curr_side = 1.0 if int(decision_class[k]) == 0 else (-1.0 if int(decision_class[k]) == 2 else 0.0)
        direction_agree[j] = np.float32(1.0 if curr_side == sgn_side else 0.0)
        regime_changed[j] = np.float32(float(int(np.argmax(current_regime[k].astype(np.float64))) != rid_e))
        gc = float(1.0 - neutral[k])
        gate_curr[j] = np.float32(gc)
        gate_decay[j] = np.float32(gc - gate_e)
        would_enter[j] = np.float32(
            1.0
            if l3_entry_side_from_l2(
                int(decision_class[k]),
                float(decision_conf[k]),
                float(size[k]),
                min_confidence=float(min_conf_arr[k]),
                min_size=float(min_size_arr[k]),
            )
            != 0.0
            else 0.0
        )

        u = float(unreal_seg[j])
        if u >= running_peak - 1e-9:
            running_peak = u
            peak_idx = j
        bars_since_peak[j] = np.float32(j - peak_idx)
        at_new_high[j] = np.float32(1.0 if abs(u - running_peak) < 1e-9 else 0.0)

        pk = float(peak_cum[j])
        if pk > 1e-6:
            regret_ratio[j] = np.float32(max(0.0, (pk - u) / pk))
        else:
            regret_ratio[j] = np.float32(0.0)

        dd = float(drawdown_from_peak[j])
        bsp = float(bars_since_peak[j])
        regret_velocity[j] = np.float32(dd / bsp) if bsp > 0.5 else np.float32(0.0)

        du = float(unreal_seg[j] - unreal_seg[j - 1]) if j > 0 else float(unreal_seg[0])
        favorable = sgn_side * du > 0.0
        llr = w["fav"] if favorable else w["adv"]
        if float(regime_changed[j]) > 0.5:
            llr += w["regime"]
        if float(gate_decay[j]) < w["gate_thr"]:
            llr += w["gate"]
        log_odds = float(log_odds + llr)
        quality_bayes[j] = np.float32(1.0 / (1.0 + np.exp(-log_odds)))

    return np.column_stack(
        [
            signal_conf_decay,
            direction_agree,
            regime_changed,
            gate_curr,
            gate_decay,
            would_enter,
            regret_ratio,
            bars_since_peak,
            at_new_high,
            regret_velocity,
            quality_bayes,
        ]
    ).astype(np.float32, copy=False)


def l3_meta_max_hold_bars(meta: Mapping[str, Any]) -> int:
    """Episode cap consistent with training ``_build_l3_policy_dataset(..., max_hold=...)``.

    Prefer ``l3_max_hold_bars`` (written at train time). Else ``l3_target_horizon_bars``.
    Else ``L3_MAX_HOLD_BARS`` / :func:`l3_defaults.max_hold_bars` default (30).
    """
    mh = meta.get("l3_max_hold_bars")
    if mh is not None:
        return max(1, int(mh))
    th = meta.get("l3_target_horizon_bars")
    if th is not None:
        return max(1, int(th))
    return max(1, int(L3DEF.max_hold_bars()))


def l3_policy_state_key(
    regime_probs: np.ndarray,
    vol_value: float,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
) -> str:
    vol_quantiles = [float(x) for x in (meta.get("policy_state_vol_quantiles") or [])]
    keys = _state_keys_from_regime_vol(np.asarray(regime_probs, dtype=np.float32).reshape(1, -1), np.asarray([vol_value], dtype=np.float32), vol_quantiles=vol_quantiles)
    base = str(keys[0]) if len(keys) else "r0_v0"
    if not _l3_pa_policy_enabled():
        return base
    return str(_append_pa_bucket_to_state_keys(np.asarray([base], dtype=object), np.asarray([pa_state_bucket_label_from_mapping(pa_state)], dtype=object))[0])


def l3_entry_policy_params(
    regime_probs: np.ndarray,
    vol_value: float,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
) -> tuple[float, float, int, str]:
    state_key = l3_policy_state_key(regime_probs, vol_value, meta, pa_state=pa_state)
    params = (meta.get("l3_entry_policy_by_state") or {}).get(state_key, {})
    min_conf = float(params.get("min_confidence", meta.get("l3_entry_min_confidence", _l3_entry_policy_defaults()[0])))
    min_size = float(params.get("min_size", meta.get("l3_entry_min_size", _l3_entry_policy_defaults()[1])))
    cap = l3_meta_max_hold_bars(meta)
    hold_map = meta.get("l3_target_horizon_bars_by_state") or {}
    raw = int(hold_map.get(state_key, meta.get("l3_target_horizon_bars", cap)))
    max_hold = int(min(max(1, raw), cap))
    return min_conf, min_size, max_hold, state_key


def l3_exit_policy_params(
    regime_probs: np.ndarray,
    vol_value: float,
    hold_bars: int,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
    unreal_pnl_atr: float | None = None,
    regime_divergence: float | None = None,
) -> tuple[float, int, str]:
    pnl_cfg = meta.get("l3_exit_state_pnl_phase")
    if isinstance(pnl_cfg, dict) and pnl_cfg.get("enabled"):
        med = float(max(1.0, float(pnl_cfg.get("median_episode_hold", 1.0))))
        deep = float(pnl_cfg.get("pnl_deep_atr", _l3_exit_state_pnl_deep_atr()))
        half = 0.5 * med
        u = float(unreal_pnl_atr if unreal_pnl_atr is not None else 0.0)
        h = float(hold_bars)
        if not np.isfinite(u):
            u = 0.0
        if not np.isfinite(h):
            h = 0.0
        prof = u >= 0.0
        if prof:
            base_key = "pnl_ps" if h >= half else "pnl_pf"
        else:
            base_key = "pnl_ud" if abs(u) >= deep else "pnl_um"
        rd_cut = pnl_cfg.get("regime_div_cut")
        if rd_cut is not None and regime_divergence is not None and np.isfinite(regime_divergence):
            if float(regime_divergence) >= float(rd_cut):
                base_key = f"{base_key}_rd"
        state_key = base_key
        if _l3_pa_policy_enabled() and _l3_exit_state_pnl_with_pa_enabled():
            state_key = str(
                _append_pa_bucket_to_state_keys(
                    np.asarray([state_key], dtype=object),
                    np.asarray([pa_state_bucket_label_from_mapping(pa_state)], dtype=object),
                )[0]
            )
    else:
        base_key = str(
            _exit_state_keys_from_regime_vol_hold(
                np.asarray(regime_probs, dtype=np.float32).reshape(1, -1),
                np.asarray([vol_value], dtype=np.float32),
                np.asarray([hold_bars], dtype=np.float32),
                vol_quantiles=[float(x) for x in (meta.get("policy_state_vol_quantiles") or [])],
            )[0]
        )
        state_key = (
            base_key
            if not _l3_pa_policy_enabled()
            else str(
                _append_pa_bucket_to_state_keys(
                    np.asarray([base_key], dtype=object),
                    np.asarray([pa_state_bucket_label_from_mapping(pa_state)], dtype=object),
                )[0]
            )
        )
    params = (meta.get("l3_exit_policy_by_state") or {}).get(state_key, {})
    prob_thr = float(params.get("exit_prob_threshold", meta.get("l3_exit_prob_threshold", 0.55)))
    split_bar = int(params.get("early_hold_split_bar", meta.get("l3_policy_early_hold_split_bar", 3)))
    prob_thr_early = float(params.get("exit_prob_threshold_early", meta.get("l3_exit_prob_threshold_early", prob_thr)))
    prob_thr_late = float(params.get("exit_prob_threshold_late", meta.get("l3_exit_prob_threshold_late", prob_thr)))
    prob_thr = prob_thr_early if int(hold_bars) < split_bar else prob_thr_late
    cap = l3_meta_max_hold_bars(meta)
    hold_map = meta.get("l3_target_horizon_bars_by_state") or {}
    raw = int(hold_map.get(state_key, meta.get("l3_target_horizon_bars", cap)))
    max_hold = int(min(max(1, raw), cap))
    return prob_thr, max_hold, state_key


def _l3_straddle_risk_free_rate() -> float:
    return 0.04


def _l3_straddle_max_hold_minutes(default_max_hold: int) -> int:
    return max(int(default_max_hold), 390)


def _l3_straddle_iv_path_mode() -> str:
    """Default ``garch_plus_scenarios`` = multi-IV paths (slow L3 straddle dataset build).
    Set ``L3_STRADDLE_IV_PATH_MODE=deterministic`` (or ``proxy`` / ``garch``) for a **single** flat IV
    path per entry — much faster, weaker scenario diversity.
    """
    v = (os.environ.get("L3_STRADDLE_IV_PATH_MODE") or "garch_plus_scenarios").strip()
    return v if v else "garch_plus_scenarios"


def _l3_regime_name_to_id(name: str) -> float:
    order = {
        "low_vol_stable": 0.0,
        "low_vol_rising": 1.0,
        "mid_vol": 2.0,
        "high_vol_stable": 3.0,
        "high_vol_falling": 4.0,
    }
    return float(order.get(str(name), -1.0))


def _l3_attach_vixy_for_straddle_if_needed(merged: pd.DataFrame) -> pd.DataFrame:
    """Align VIXY ratio features for path-aware straddle columns (reuse L2 helper)."""
    if not L2_USE_VIXY or "vixy_level_ma60_ratio" in merged.columns:
        return merged
    try:
        from core.features.vixy_features import attach_vixy_features_to_l2_merged

        attach_vixy_features_to_l2_merged(merged, path=VIXY_DATA_PATH)
    except Exception as ex:
        print(f"  [L3] VIXY attach skipped in straddle prep: {ex}", flush=True)
    return merged


def _l3_straddle_value_target_mode() -> str:
    return "future_gain"


def _l3_compute_straddle_value_y(
    future_gain: np.ndarray,
    trade_df: pd.DataFrame,
    rv_seg: np.ndarray,
    *,
    mode: str | None = None,
) -> np.ndarray:
    """Value head label for straddle rows: future_gain (default), Sharpe-like, theta-adjusted, binary, or 3-class."""
    fg = np.asarray(future_gain, dtype=np.float64).ravel()
    m = (mode or _l3_straddle_value_target_mode()).strip().lower()
    if m in {"", "future_gain", "default"}:
        return fg.astype(np.float32)
    rv = np.maximum(np.asarray(rv_seg, dtype=np.float64).ravel(), 1e-4)
    if m in {"sharpe_proxy", "sharpe"}:
        return np.clip(fg / rv, -50.0, 50.0).astype(np.float32)
    if m in {"theta_net", "theta_adj"}:
        th = np.abs(trade_df["theta"].to_numpy(dtype=np.float64))
        sc = 1.0
        return (fg - sc * th).astype(np.float32)
    if m in {"binary_hurdle", "hurdle_binary"}:
        thr = 0.02
        return (fg > thr).astype(np.float32)
    if m in {"ternary_q", "quantile_3", "three_class"}:
        raw = "0.015,0.045".split(",")
        lo = float(raw[0]) if len(raw) > 0 else 0.015
        hi = float(raw[1]) if len(raw) > 1 else 0.045
        y = np.where(fg < lo, 0.0, np.where(fg < hi, 1.0, 2.0))
        return y.astype(np.float32)
    return fg.astype(np.float32)


def _l3_prepare_straddle_merged_frame(merged: pd.DataFrame) -> pd.DataFrame:
    out = build_straddle_features(merged, timestamp_col="time_key")
    out["l3_base_iv"] = build_base_iv_series(out, timestamp_col="time_key", close_col="close")
    if "l2_vol_regime" in out.columns:
        out["l3_l2_vol_regime_id"] = out["l2_vol_regime"].astype(str).map(_l3_regime_name_to_id).fillna(-1.0)
    else:
        out["l3_l2_vol_regime_id"] = -1.0
    if "l2_regime_size_mult" not in out.columns:
        out["l2_regime_size_mult"] = 1.0
    if "l2_predicted_profit" not in out.columns:
        out["l2_predicted_profit"] = 0.0
    out = _l3_attach_vixy_for_straddle_if_needed(out)
    return out


def _l3_build_straddle_policy_dataset(
    merged: pd.DataFrame,
    *,
    max_hold: int,
    traj_cfg: L3TrajectoryConfig | None,
    build_traj: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    merged = _l3_prepare_straddle_merged_frame(merged)
    extra_merged = _l3_extra_merged_feature_columns()
    for c in extra_merged:
        if c not in merged.columns:
            raise ValueError(f"L3_MERGED_EXTRA_FEATURE_COLUMNS: missing column {c!r} in merged frame")
    feature_cols = _l3_policy_matrix_column_names(extra_merged)
    print(f"  [L3] straddle value target mode={_l3_straddle_value_target_mode()!r}", flush=True)
    symbols = merged["symbol"].to_numpy()
    times = pd.to_datetime(merged["time_key"]).to_numpy()
    close_px = pd.to_numeric(merged["close"], errors="coerce").to_numpy(dtype=np.float64)
    high_px = pd.to_numeric(merged["high"], errors="coerce").to_numpy(dtype=np.float64)
    low_px = pd.to_numeric(merged["low"], errors="coerce").to_numpy(dtype=np.float64)
    current_regime = merged[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    entry_regime = merged[[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32, copy=False)
    current_vol = merged["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    entry_vol = merged["l2_entry_vol"].to_numpy(dtype=np.float32, copy=False)
    straddle_on_arr = pd.to_numeric(merged.get("l2_straddle_on", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    decision_class = l2_l3_entry_decision_class_from_merged(merged)
    decision_conf = merged["l2_decision_confidence"].to_numpy(dtype=np.float32, copy=False)
    size = merged["l2_size"].to_numpy(dtype=np.float32, copy=False)
    pred_mfe = merged["l2_pred_mfe"].to_numpy(dtype=np.float32, copy=False)
    pred_mae = merged["l2_pred_mae"].to_numpy(dtype=np.float32, copy=False)
    safe_atr = np.where(
        pd.to_numeric(merged["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3,
        merged["lbl_atr"].to_numpy(dtype=np.float64),
        1e-3,
    )
    pa_state = _l3_pa_dict_from_frame(merged)
    oot_mask = (times >= np.datetime64(CAL_END)) & (times < np.datetime64(TEST_END))
    policy_vol_quantiles = _policy_vol_quantiles(current_vol, fit_mask=oot_mask)
    state_keys_all = _append_pa_bucket_to_state_keys(
        _state_keys_from_regime_vol(current_regime, current_vol, vol_quantiles=policy_vol_quantiles),
        pa_state_bucket_labels_from_frame(merged) if _l3_pa_policy_enabled() else None,
    )
    target_horizon_global, target_horizon_by_state = _l3_target_horizon_by_state(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        np.full(int(max(1, np.sum(oot_mask) if oot_mask.any() else len(merged))), float(np.median(dte_grid_days())) * 390.0, dtype=np.float32),
        max_hold=_l3_straddle_max_hold_minutes(max_hold),
        pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
    )
    if _l3_trust_l2_entry_enabled():
        entry_policy_global, entry_policy_by_state = _l3_entry_policy_trust_l2_fixed()
        print(
            f"  [L3] entry policy: TRUST_L2_ENTRY=1 (straddle sim)  "
            f"min_conf={entry_policy_global['min_confidence']:.4f}  min_size={entry_policy_global['min_size']:.4f}  states=0",
            flush=True,
        )
    else:
        edge_atr = _decision_edge_atr_array(merged).astype(np.float64)
        tau_edge = float(max(0.0, float(os.environ.get("STACK_DECISION_EDGE_TAU", "0.05"))))
        entry_policy_global, entry_policy_by_state = _l3_search_entry_policy(
            state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
            decision_class[oot_mask] if oot_mask.any() else decision_class,
            decision_conf[oot_mask] if oot_mask.any() else decision_conf,
            size[oot_mask] if oot_mask.any() else size,
            edge_atr[oot_mask] if oot_mask.any() else edge_atr,
            tau_edge,
            pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
        )
    entry_policy_arrays = _l3_lookup_policy_map(
        state_keys_all,
        entry_policy_by_state,
        defaults={
            "min_confidence": float(entry_policy_global["min_confidence"]),
            "min_size": float(entry_policy_global["min_size"]),
        },
    )
    policy_loss_buffer_base, policy_loss_buffer_meta = _resolve_l3_policy_loss_buffer_atr(merged, oot_mask)
    base_iv = pd.to_numeric(merged["l3_base_iv"], errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
    traj_mfe_scale = max(float(np.nanquantile(np.maximum(pred_mfe.astype(np.float64), 0.0), 0.99)), 1.0)
    traj_mae_scale = max(float(np.nanquantile(np.maximum(pred_mae.astype(np.float64), 0.0), 0.99)), 1.0)
    _t_base = traj_cfg or L3TrajectoryConfig()
    _t_cfg_eff = replace(_t_base, mfe_norm_scale=float(traj_mfe_scale), mae_norm_scale=float(traj_mae_scale))
    _t_max = _t_cfg_eff.max_seq_len
    _t_ref = max(_t_max, int(_l3_straddle_max_hold_minutes(max_hold)))
    simulator = StraddleSimulator(risk_free_rate=_l3_straddle_risk_free_rate())
    run_end = np.empty(len(merged), dtype=np.int32)
    run_start = 0
    for idx in range(1, len(merged) + 1):
        if idx == len(merged) or symbols[idx] != symbols[run_start]:
            run_end[run_start:idx] = idx
            run_start = idx
    rows_x_blocks: list[np.ndarray] = []
    rows_exit_blocks: list[np.ndarray] = []
    rows_value_blocks: list[np.ndarray] = []
    rows_time_blocks: list[np.ndarray] = []
    rows_merged_idx_blocks: list[np.ndarray] = []
    rows_entry_blocks: list[np.ndarray] = []
    rows_from_model_blocks: list[np.ndarray] = []
    rows_traj: list[np.ndarray] = []
    rows_traj_len: list[int] = []
    n_policy_signals_model = 0
    n_policy_signals_truth = 0
    max_hold_minutes = _l3_straddle_max_hold_minutes(max_hold)
    dtes = dte_grid_days()
    _n_merge = len(merged)
    row_it: Iterable[int] = range(_n_merge)
    if _l3_policy_dataset_tqdm_enabled():
        _l3p_tf = _l3_policy_tqdm_file()
        row_it = tqdm(
            row_it,
            desc="[L3] straddle dataset",
            total=_n_merge,
            unit="bar",
            leave=bool(_l3p_tf.isatty()) if hasattr(_l3p_tf, "isatty") else False,
            file=_l3p_tf,
            mininterval=1.0,
        )
    else:
        print(
            f"  [L3] straddle policy dataset: scanning {_n_merge:,} merged bars (tqdm off: set DISABLE_TQDM=1) ...",
            file=sys.stderr,
            flush=True,
        )
    for i in row_it:
        if i + 2 >= len(merged) or run_end[i] <= i + 1:
            continue
        min_confidence = float(entry_policy_arrays["min_confidence"][i])
        min_size = float(entry_policy_arrays["min_size"][i])
        model_side = l3_entry_side_from_l2(
            int(decision_class[i]),
            float(decision_conf[i]),
            float(size[i]),
            min_confidence=min_confidence,
            min_size=min_size,
        )
        if model_side == 0.0:
            continue
        n_policy_signals_model += 1
        available = int(run_end[i] - (i + 1))
        if available <= 2:
            continue
        rng_i = np.random.default_rng(int(i) + 17)
        for dte in dtes:
            horizon = min(available, max_hold_minutes, int(dte * 390))
            if horizon <= 2:
                continue
            if _l3_straddle_iv_path_mode() in {"garch", "proxy", "deterministic"}:
                scenario_map = {"proxy_0": np.full(horizon, float(base_iv[i]), dtype=np.float32)}
            else:
                scenario_map = generate_iv_scenarios(float(base_iv[i]), horizon, rng=rng_i, n_scenarios=scenario_count())
            for _scenario_name, iv_path in scenario_map.items():
                trade_df = simulator.simulate_trade(
                    merged,
                    entry_idx=i,
                    dte_days=dte,
                    entry_iv=float(base_iv[i]),
                    iv_path=iv_path,
                    max_minutes=horizon,
                    timestamp_col="time_key",
                    base_iv_col="l3_base_iv",
                )
                if trade_df.empty:
                    continue
                n_steps = len(trade_df)
                idx_arr = np.arange(i + 1, i + 1 + n_steps, dtype=np.int32)
                holds = trade_df["minute"].to_numpy(dtype=np.float32)
                pnl_pct = trade_df["pnl_pct"].to_numpy(dtype=np.float32)
                straddle_value_rel = (trade_df["straddle_value"].to_numpy(dtype=np.float32) / max(float(trade_df["entry_value"].iloc[0]), 1e-6)).astype(np.float32)
                live_mfe_seg = np.maximum.accumulate(np.maximum(pnl_pct, 0.0)).astype(np.float32)
                peak_pnl = np.maximum.accumulate(pnl_pct.astype(np.float64))
                drawdown_from_peak = (peak_pnl - pnl_pct.astype(np.float64)).astype(np.float32)
                live_mae_seg = np.maximum.accumulate(np.maximum(-pnl_pct, 0.0)).astype(np.float32)
                future_best = np.maximum.accumulate(pnl_pct[::-1])[::-1].astype(np.float32)
                future_gain_left = (future_best - pnl_pct).astype(np.float32)
                live_edge_seg = (live_mfe_seg - live_mae_seg).astype(np.float32)
                regime_div_seg = _kl_divergence(np.repeat(entry_regime[i : i + 1], n_steps, axis=0), current_regime[idx_arr]).astype(np.float32, copy=False)
                safe_entry_vol = max(float(entry_vol[i]), 1e-3)
                vol_surprise_seg = (current_vol[idx_arr] / safe_entry_vol).astype(np.float32, copy=False)
                log_h_seg = np.log1p(holds).astype(np.float32, copy=False)
                h_sq_seg = ((holds * holds) / 100.0).astype(np.float32, copy=False)
                h_bkt_seg = np.searchsorted(np.array([3, 8, 15, 30, 999], dtype=np.int64), holds.astype(np.int64), side="right").astype(np.float32)
                und_path = trade_df["underlying"].to_numpy(dtype=np.float64)
                und0 = float(und_path[0])
                ep_u = max(abs(und0), 1e-9)
                bar_ret_frac = ((und_path - und0) / ep_u).astype(np.float32)
                peak_frac_s = np.maximum.accumulate(bar_ret_frac.astype(np.float64))
                drawdown_frac = (peak_frac_s - bar_ret_frac.astype(np.float64)).astype(np.float32)
                ret_last_3, ret_last_5, vol_trade, slope_fb = _l3_intra_episode_frac_derivatives(bar_ret_frac)
                target_hz = max(1, int(horizon))
                mom_lead = _l3_episode_momentum_leading_block(
                    n_steps,
                    pnl_pct,
                    drawdown_from_peak,
                    live_mfe_seg,
                    live_mae_seg,
                    vol_surprise_seg,
                    bar_ret_frac,
                    holds,
                    target_hz,
                    float(pred_mfe[i]),
                )
                close_seg = close_px[idx_arr].astype(np.float64, copy=False)
                atr_d = max(float(safe_atr[i]), 1e-6)
                vel3 = np.zeros(n_steps, dtype=np.float32)
                for j in range(n_steps):
                    j0 = max(0, j - 3)
                    vel3[j] = float((close_seg[j] - close_seg[j0]) / atr_d)
                reg_div_d = regime_div_seg.astype(np.float64, copy=False)
                mom_rd = np.zeros(n_steps, dtype=np.float32)
                for j in range(n_steps):
                    j0 = max(0, j - 3)
                    mom_rd[j] = float(reg_div_d[j] - reg_div_d[j0])
                vs = vol_surprise_seg.astype(np.float64, copy=False)
                vs_acc = np.zeros(n_steps, dtype=np.float32)
                for j in range(n_steps):
                    if j >= 2:
                        vs_acc[j] = float(vs[j] - 2.0 * vs[j - 1] + vs[j - 2])
                cr = current_regime[idx_arr].astype(np.float64, copy=False)
                rid = np.argmax(cr, axis=1).astype(np.int32, copy=False)
                stab = np.zeros(n_steps, dtype=np.float32)
                for j in range(n_steps):
                    lo = max(0, j - 2)
                    stab[j] = float(np.mean(rid[lo : j + 1] == rid[j]))
                if "l2_gate_prob" in merged.columns:
                    gp = np.clip(pd.to_numeric(merged["l2_gate_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)
                    neutral = (1.0 - gp).astype(np.float64)
                else:
                    neutral = np.where(decision_class == 1, 1.0, 0.25).astype(np.float64)
                aux_block = _l3_episode_aux_feature_block(
                    n_steps=n_steps,
                    idx_arr=idx_arr,
                    entry_i=i,
                    side=float(model_side),
                    decision_class=decision_class,
                    decision_conf=decision_conf,
                    size=size,
                    neutral=neutral,
                    entry_regime_row=entry_regime[i],
                    current_regime=current_regime,
                    unreal_seg=pnl_pct,
                    min_conf_arr=entry_policy_arrays["min_confidence"],
                    min_size_arr=entry_policy_arrays["min_size"],
                    drawdown_from_peak=drawdown_from_peak,
                )
                rng_i_row = float(merged["l2_range_pred"].iloc[i]) if "l2_range_pred" in merged.columns else 0.0
                gp_i = float(merged["l2_gate_prob"].iloc[i]) if "l2_gate_prob" in merged.columns else float(decision_conf[i])
                dyn_scalar_parts: list[np.ndarray] = [
                    np.full(n_steps, float(straddle_on_arr[i]), dtype=np.float32),
                    np.full(n_steps, rng_i_row, dtype=np.float32),
                    np.full(n_steps, gp_i, dtype=np.float32),
                    np.full(n_steps, decision_conf[i], dtype=np.float32),
                    np.full(n_steps, size[i], dtype=np.float32),
                    np.full(n_steps, pred_mfe[i], dtype=np.float32),
                    np.full(n_steps, pred_mae[i], dtype=np.float32),
                    np.full(n_steps, float(pd.to_numeric(merged.get("l2_predicted_profit", 0.0), errors="coerce").iloc[i]) if "l2_predicted_profit" in merged.columns else 0.0, dtype=np.float32),
                    np.full(n_steps, float(merged["l3_l2_vol_regime_id"].iloc[i]), dtype=np.float32),
                    np.full(n_steps, float(merged["l2_regime_size_mult"].iloc[i]), dtype=np.float32),
                ]
                straddle_dyn_parts: list[np.ndarray] = [
                    straddle_value_rel,
                    pnl_pct,
                    trade_df["theta"].to_numpy(dtype=np.float32),
                    trade_df["vega"].to_numpy(dtype=np.float32),
                    trade_df["gamma"].to_numpy(dtype=np.float32),
                    trade_df["iv"].to_numpy(dtype=np.float32),
                    np.asarray(trade_df["entry_iv"].to_numpy(dtype=np.float32), dtype=np.float32),
                    trade_df["T_remaining"].to_numpy(dtype=np.float32),
                    trade_df["underlying_abs_move"].to_numpy(dtype=np.float32),
                    trade_df["underlying_gap_abs"].to_numpy(dtype=np.float32),
                ]
                if "rv_60" in merged.columns:
                    rv_seg = merged.loc[idx_arr, "rv_60"].to_numpy(dtype=np.float64)
                else:
                    rv_seg = np.full(n_steps, 0.25, dtype=np.float64)
                theta_arr = trade_df["theta"].to_numpy(dtype=np.float64)
                Trem = trade_df["T_remaining"].to_numpy(dtype=np.float64)
                T0 = max(float(trade_df["T_remaining"].iloc[0]), 1e-8)
                rem_rat = (Trem / T0).astype(np.float32)
                theta_burn = (np.abs(theta_arr) / np.maximum(Trem, 1e-8)).astype(np.float32)
                iv_arr = trade_df["iv"].to_numpy(dtype=np.float64)
                iv_rv_spread = (iv_arr - rv_seg).astype(np.float32)
                if "vixy_level_ma60_ratio" in merged.columns:
                    vtrack = pd.to_numeric(merged["vixy_level_ma60_ratio"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
                    v0 = float(vtrack[i]) if np.isfinite(vtrack[i]) else 1.0
                    seg = vtrack[idx_arr]
                    vixy_run = (np.maximum.accumulate(seg) / max(v0, 1e-6)).astype(np.float32)
                    vixy_rel = (seg / max(v0, 1e-6)).astype(np.float32)
                    vixy_delta = (seg - v0).astype(np.float32)
                else:
                    vixy_run = np.ones(n_steps, dtype=np.float32)
                    vixy_rel = np.ones(n_steps, dtype=np.float32)
                    vixy_delta = np.zeros(n_steps, dtype=np.float32)
                pnlp = pnl_pct.astype(np.float64)
                roll_vol = np.zeros(n_steps, dtype=np.float32)
                for jj in range(n_steps):
                    lo = max(0, jj - 4)
                    if jj - lo >= 1:
                        roll_vol[jj] = float(np.std(pnlp[lo : jj + 1]))
                curv = np.zeros(n_steps, dtype=np.float32)
                if n_steps >= 3:
                    curv[2:] = np.abs(np.diff(pnlp, n=2)).astype(np.float32)
                implied_entry_atr = float(merged["l2_implied_proxy_range"].iloc[i]) if "l2_implied_proxy_range" in merged.columns else max(float(rng_i_row), 0.1)
                implied_entry_frac = float(max(implied_entry_atr * max(float(safe_atr[i]), 1e-6) / max(abs(und0), 1e-9), 1e-6))
                range_realization_ratio = (np.abs(bar_ret_frac.astype(np.float64)) / implied_entry_frac).astype(np.float32)
                theta_burn_fraction = np.sqrt(np.asarray(holds, dtype=np.float64) / max(float(horizon), 1.0)).astype(np.float32)
                range_expansion_speed = (
                    np.abs(bar_ret_frac.astype(np.float64)) / np.maximum(np.asarray(holds, dtype=np.float64), 1.0)
                ).astype(np.float32)
                straddle_extra_parts = [
                    theta_burn,
                    iv_rv_spread,
                    rem_rat,
                    vixy_run,
                    vixy_rel,
                    roll_vol,
                    curv,
                ]
                engineered_parts = [merged.loc[idx_arr, c].to_numpy(dtype=np.float32, copy=False) for c in _l3_straddle_feature_columns()]
                extra_parts = [merged.loc[idx_arr, _xc].to_numpy(dtype=np.float32, copy=False) for _xc in extra_merged]
                stack_parts: list[np.ndarray] = dyn_scalar_parts
                stack_parts.extend(
                    [
                        np.repeat(entry_regime[i : i + 1], n_steps, axis=0),
                        np.full(n_steps, entry_vol[i], dtype=np.float32),
                        current_regime[idx_arr],
                        current_vol[idx_arr].astype(np.float32, copy=False),
                        regime_div_seg,
                        vol_surprise_seg,
                        holds,
                        pnl_pct.astype(np.float32, copy=False),
                        live_mfe_seg,
                        live_mae_seg,
                        live_edge_seg,
                        np.full(n_steps, float(model_side), dtype=np.float32),
                        log_h_seg,
                        h_sq_seg,
                        h_bkt_seg,
                        drawdown_from_peak,
                        bar_ret_frac,
                        drawdown_frac,
                        ret_last_3,
                        ret_last_5,
                        vol_trade,
                        slope_fb,
                        mom_lead,
                        vel3,
                        mom_rd,
                        vs_acc,
                        stab,
                        merged.loc[idx_arr, PA_STATE_FEATURES].to_numpy(dtype=np.float32, copy=False),
                        *extra_parts,
                        aux_block,
                        range_realization_ratio,
                        theta_burn_fraction,
                        vixy_delta,
                        range_expansion_speed,
                        *straddle_dyn_parts,
                        *straddle_extra_parts,
                        *engineered_parts,
                    ]
                )
                feat_block = np.column_stack(stack_parts).astype(np.float32, copy=False)
                step_trend = np.asarray(pa_state["pa_state_trend_strength"][idx_arr], dtype=np.float32)
                step_follow = np.asarray(pa_state["pa_state_followthrough_quality"][idx_arr], dtype=np.float32)
                step_range = np.asarray(pa_state["pa_state_range_risk"][idx_arr], dtype=np.float32)
                step_breakout = np.asarray(pa_state["pa_state_breakout_failure_risk"][idx_arr], dtype=np.float32)
                step_pullback = np.asarray(pa_state["pa_state_pullback_exhaustion"][idx_arr], dtype=np.float32)
                if _l3_pa_targets_enabled():
                    eps_arr = (_l3_exit_epsilon_atr() * pa_exit_eps_multiplier(step_range, step_breakout, step_pullback, step_trend, step_follow)).astype(np.float32)
                    loss_buffer_arr = (policy_loss_buffer_base * pa_exit_loss_buffer_multiplier(step_range, step_breakout, step_pullback, step_trend)).astype(np.float32)
                else:
                    eps_arr = np.full(n_steps, _l3_exit_epsilon_atr(), dtype=np.float32)
                    loss_buffer_arr = np.full(n_steps, policy_loss_buffer_base, dtype=np.float32)
                last_step = np.arange(n_steps) == (n_steps - 1)
                if float(model_side) < 0.0:
                    short_mult = _l3_short_vol_range_explosion_mult()
                    range_explosion = np.abs(bar_ret_frac.astype(np.float64)) >= (implied_entry_frac * short_mult)
                    exit_block = (last_step | range_explosion).astype(np.int32, copy=False)
                else:
                    exit_block = (last_step | (future_gain_left <= eps_arr) | (drawdown_from_peak >= loss_buffer_arr)).astype(np.int32, copy=False)
                rows_x_blocks.append(feat_block)
                rows_exit_blocks.append(exit_block)
                vtm = _l3_value_target_mode()
                if vtm == "trade_outcome":
                    rows_value_blocks.append(np.zeros(n_steps, dtype=np.float32))
                elif vtm == "remaining_value":
                    rows_value_blocks.append(_l3_remaining_value_labels_from_unreal(pnl_pct))
                elif vtm == "remaining_value_atr":
                    a = np.full(n_steps, max(float(safe_atr[i]), 1e-9), dtype=np.float64)
                    rows_value_blocks.append(_l3_remaining_value_atr_from_unreal(pnl_pct, a))
                elif vtm == "peak_cls":
                    c_s = close_px[idx_arr]
                    h_s = high_px[idx_arr]
                    l_s = low_px[idx_arr]
                    v_b = _l3_peak_upside_value_labels(
                        close_seg=c_s,
                        high_seg=h_s,
                        low_seg=l_s,
                        side=float(model_side),
                        play_straddle=True,
                        atr_scale=float(safe_atr[i]),
                        upside_threshold_atr=_l3_value_upside_threshold_atr(),
                    )
                    rows_value_blocks.append(v_b)
                else:
                    rows_value_blocks.append(
                        _l3_compute_straddle_value_y(future_gain_left, trade_df, rv_seg, mode=_l3_straddle_value_target_mode())
                    )
                rows_time_blocks.append(times[idx_arr])
                rows_merged_idx_blocks.append(idx_arr.astype(np.int64, copy=False))
                rows_entry_blocks.append(np.full(n_steps, int(i), dtype=np.int64))
                rows_from_model_blocks.append(np.full(n_steps, 1, dtype=np.int32))
                if build_traj:
                    traj_hist = np.zeros((_t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32)
                    traj_len_cur = 0
                    peak_unreal = -1e9
                    prev_unreal = 0.0
                    price_rel = trade_df["underlying"].to_numpy(dtype=np.float32) / max(float(close_px[i]), 1e-6)
                    for local_idx in range(n_steps):
                        peak_unreal = max(peak_unreal, float(pnl_pct[local_idx]))
                        tvec = l3_traj_step_features_straddle(
                            float(pnl_pct[local_idx]),
                            prev_unreal,
                            peak_unreal,
                            int(holds[local_idx]),
                            np.datetime64(trade_df["timestamp"].iloc[local_idx]),
                            float(price_rel[max(local_idx - 1, 0)]),
                            float(price_rel[local_idx]),
                            float(trade_df["underlying_abs_move"].iloc[local_idx]),
                            float(trade_df["iv"].iloc[local_idx]),
                            float(vol_surprise_seg[local_idx]),
                            float(regime_div_seg[local_idx]),
                            float(trade_df["vega"].iloc[local_idx]),
                            float(abs(trade_df["theta"].iloc[local_idx])),
                            max_seq_ref=_t_ref,
                            mfe_scale=_t_cfg_eff.mfe_norm_scale,
                            mae_scale=_t_cfg_eff.mae_norm_scale,
                        )
                        prev_unreal = float(pnl_pct[local_idx])
                        if traj_len_cur < _t_max:
                            traj_hist[traj_len_cur] = tvec
                            traj_len_cur += 1
                        else:
                            traj_hist[:-1] = traj_hist[1:]
                            traj_hist[-1] = tvec
                        rows_traj.append(traj_hist.copy())
                        rows_traj_len.append(traj_len_cur)
    if not rows_x_blocks:
        print("  [L3] straddle policy dataset empty: no valid L2 straddle entries / simulated paths.", flush=True)
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            feature_cols,
            np.empty(0, dtype=np.int64),
            np.empty((0, _t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            {
                "policy_state_vol_quantiles": policy_vol_quantiles,
                "l3_entry_policy": entry_policy_global,
                "l3_entry_policy_by_state": entry_policy_by_state,
                "l3_trust_l2_entry": bool(_l3_trust_l2_entry_enabled()),
                "l3_target_horizon_bars": target_horizon_global,
                "l3_target_horizon_bars_by_state": target_horizon_by_state,
                "l3_traj_cfg_dict": asdict(_t_cfg_eff),
                "l3_policy_loss_buffer": dict(policy_loss_buffer_meta),
                "pa_target_semantics": "straddle simulator mode with PA-aware hold/exit scaling",
                "l3_trade_semantics": "straddle_bs_sim",
                "l3_iv_model": _l3_straddle_iv_path_mode(),
                "l3_straddle_dte_grid": dtes,
                "l3_straddle_scenario_count": scenario_count(),
                "l3_straddle_value_target": _l3_straddle_value_target_mode(),
                "policy_rows_merged_idx": np.empty(0, dtype=np.int64),
            },
        )
    print(
        f"  [L3] straddle policy dataset: entry signals model={n_policy_signals_model:,}  "
        f"policy_rows={sum(int(x.shape[0]) for x in rows_x_blocks):,}  dtes={dtes}  "
        f"scenario_count={scenario_count()}  target_horizon_global={target_horizon_global}",
        flush=True,
    )
    return (
        np.concatenate(rows_x_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_exit_blocks, axis=0).astype(np.int32, copy=False),
        np.concatenate(rows_value_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_time_blocks, axis=0),
        feature_cols,
        np.concatenate(rows_entry_blocks, axis=0).astype(np.int64, copy=False),
        (
            np.stack(rows_traj, axis=0).astype(np.float32, copy=False)
            if build_traj and rows_traj
            else np.empty((0, _t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32)
        ),
        (np.asarray(rows_traj_len, dtype=np.int32) if build_traj and rows_traj_len else np.empty(0, dtype=np.int32)),
        np.concatenate(rows_from_model_blocks, axis=0).astype(np.int32, copy=False),
        {
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "l3_entry_policy": entry_policy_global,
            "l3_entry_policy_by_state": entry_policy_by_state,
            "l3_trust_l2_entry": bool(_l3_trust_l2_entry_enabled()),
            "l3_target_horizon_bars": target_horizon_global,
            "l3_target_horizon_bars_by_state": target_horizon_by_state,
            "l3_traj_cfg_dict": asdict(_t_cfg_eff),
            "l3_policy_loss_buffer": dict(policy_loss_buffer_meta),
            "l3_trade_semantics": "straddle_bs_sim",
            "l3_iv_model": _l3_straddle_iv_path_mode(),
            "l3_straddle_dte_grid": dtes,
            "l3_straddle_scenario_count": scenario_count(),
            "l3_straddle_max_hold_minutes": max_hold_minutes,
            "l3_straddle_value_target": _l3_straddle_value_target_mode(),
            "policy_rows_merged_idx": np.concatenate(rows_merged_idx_blocks, axis=0).astype(np.int64, copy=False),
        },
    )


def _build_l3_policy_dataset(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
    *,
    max_hold: int = 30,
    exit_epsilon_atr: float | None = None,
    traj_cfg: L3TrajectoryConfig | None = None,
    build_traj: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    merged = (
        df.reset_index(drop=True)
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l2_outputs, on=["symbol", "time_key"], how="left")
    )
    if _l3_straddle_sim_mode_enabled():
        return _l3_build_straddle_policy_dataset(
            merged,
            max_hold=max_hold,
            traj_cfg=traj_cfg,
            build_traj=build_traj,
        )
    merged = ensure_pa_state_features(merged)
    pa_state = _l3_pa_dict_from_frame(merged)
    extra_merged = _l3_extra_merged_feature_columns()
    for c in extra_merged:
        if c not in merged.columns:
            raise ValueError(f"L3_MERGED_EXTRA_FEATURE_COLUMNS: missing column {c!r} in merged frame")
    feature_cols = _l3_policy_matrix_column_names(extra_merged)
    safe_atr = np.where(pd.to_numeric(merged["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, merged["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
    open_px = merged["open"].to_numpy(dtype=np.float64)
    high_px = merged["high"].to_numpy(dtype=np.float64)
    low_px = merged["low"].to_numpy(dtype=np.float64)
    close_px = merged["close"].to_numpy(dtype=np.float64)
    symbols = merged["symbol"].to_numpy()
    times = pd.to_datetime(merged["time_key"]).to_numpy()
    current_regime = merged[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    entry_regime = merged[[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32, copy=False)
    current_vol = merged["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    entry_vol = merged["l2_entry_vol"].to_numpy(dtype=np.float32, copy=False)
    straddle_on_arr = pd.to_numeric(merged.get("l2_straddle_on", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    decision_class = l2_l3_entry_decision_class_from_merged(merged)
    decision_conf = merged["l2_decision_confidence"].to_numpy(dtype=np.float32, copy=False)
    size = merged["l2_size"].to_numpy(dtype=np.float32, copy=False)
    edge_atr = _decision_edge_atr_array(merged).astype(np.float64)
    tau_edge = float(max(0.0, float(os.environ.get("STACK_DECISION_EDGE_TAU", "0.05"))))
    exit_epsilon_atr = _l3_exit_epsilon_atr() if exit_epsilon_atr is None else float(max(0.0, exit_epsilon_atr))
    pred_mfe = merged["l2_pred_mfe"].to_numpy(dtype=np.float32, copy=False)
    pred_mae = merged["l2_pred_mae"].to_numpy(dtype=np.float32, copy=False)
    _pred_mfe_q = pd.to_numeric(merged["l2_pred_mfe"], errors="coerce").to_numpy(dtype=np.float64)
    _pred_mae_q = pd.to_numeric(merged["l2_pred_mae"], errors="coerce").to_numpy(dtype=np.float64)
    _fin_pred = np.isfinite(_pred_mfe_q) & np.isfinite(_pred_mae_q)
    traj_mfe_scale = max(
        float(np.quantile(_pred_mfe_q[_fin_pred], 0.99)) if _fin_pred.any() else float(L3DEF.traj_mfe_scale_default()),
        1.0,
    )
    traj_mae_scale = max(
        float(np.quantile(_pred_mae_q[_fin_pred], 0.99)) if _fin_pred.any() else float(L3DEF.traj_mae_scale_default()),
        1.0,
    )
    _t_base = traj_cfg or L3TrajectoryConfig()
    _t_cfg_eff = replace(_t_base, mfe_norm_scale=float(traj_mfe_scale), mae_norm_scale=float(traj_mae_scale))
    _t_max = _t_cfg_eff.max_seq_len
    _t_ref = max(_t_max, int(max_hold))
    oot_mask = (times >= np.datetime64(CAL_END)) & (times < np.datetime64(TEST_END))
    policy_loss_buffer_base, policy_loss_buffer_meta = _resolve_l3_policy_loss_buffer_atr(merged, oot_mask)
    policy_vol_quantiles = _policy_vol_quantiles(current_vol, fit_mask=oot_mask)
    state_keys_all = _append_pa_bucket_to_state_keys(
        _state_keys_from_regime_vol(current_regime, current_vol, vol_quantiles=policy_vol_quantiles),
        pa_state_bucket_labels_from_frame(merged) if _l3_pa_policy_enabled() else None,
    )
    if "decision_peak_bar" in merged.columns:
        peak_bar = pd.to_numeric(merged["decision_peak_bar"], errors="coerce").fillna(_l3_target_horizon_bars(max_hold)).to_numpy(dtype=np.float32)
    else:
        peak_bar = np.full(len(merged), _l3_target_horizon_bars(max_hold), dtype=np.float32)
    target_horizon_global, target_horizon_by_state = _l3_target_horizon_by_state(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        peak_bar[oot_mask] if oot_mask.any() else peak_bar,
        max_hold=max_hold,
        pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
    )
    if _l3_trust_l2_entry_enabled():
        entry_policy_global, entry_policy_by_state = _l3_entry_policy_trust_l2_fixed()
        print(
            f"  [L3] entry policy: TRUST_L2_ENTRY=1 (skip grid; L2 opens, L3 exit-only)  "
            f"min_conf={entry_policy_global['min_confidence']:.4f}  min_size={entry_policy_global['min_size']:.4f}  "
            f"states=0",
            flush=True,
        )
    else:
        entry_policy_global, entry_policy_by_state = _l3_search_entry_policy(
            state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
            decision_class[oot_mask] if oot_mask.any() else decision_class,
            decision_conf[oot_mask] if oot_mask.any() else decision_conf,
            size[oot_mask] if oot_mask.any() else size,
            edge_atr[oot_mask] if oot_mask.any() else edge_atr,
            tau_edge,
            pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
        )
    entry_policy_arrays = _l3_lookup_policy_map(
        state_keys_all,
        entry_policy_by_state,
        defaults={
            "min_confidence": float(entry_policy_global["min_confidence"]),
            "min_size": float(entry_policy_global["min_size"]),
        },
    )
    if "l2_gate_prob" in merged.columns:
        gp = np.clip(pd.to_numeric(merged["l2_gate_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)
        neutral = (1.0 - gp).astype(np.float64)
    elif "l2_decision_neutral" in merged.columns:
        neutral = pd.to_numeric(merged["l2_decision_neutral"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    else:
        neutral = np.where(decision_class == 1, 1.0, 0.25).astype(np.float64)
    horizon_arr = np.full(len(merged), int(target_horizon_global), dtype=np.int32)
    for key, horizon in target_horizon_by_state.items():
        horizon_arr[state_keys_all == key] = int(horizon)

    run_end = np.empty(len(merged), dtype=np.int32)
    run_start = 0
    for idx in range(1, len(merged) + 1):
        if idx == len(merged) or symbols[idx] != symbols[run_start]:
            run_end[run_start:idx] = idx
            run_start = idx

    rows_x_blocks: list[np.ndarray] = []
    rows_exit_blocks: list[np.ndarray] = []
    rows_value_blocks: list[np.ndarray] = []
    rows_time_blocks: list[np.ndarray] = []
    rows_merged_idx_blocks: list[np.ndarray] = []
    rows_entry_blocks: list[np.ndarray] = []
    rows_from_model_blocks: list[np.ndarray] = []
    rows_traj: list[np.ndarray] = []
    rows_traj_len: list[int] = []
    n_policy_signals_model = 0
    n_policy_signals_truth = 0
    allow_truth_fallback = False
    _hold_bin_edges = np.array([3, 8, 15, 30, 999], dtype=np.int64)
    _n_merge2 = len(merged)
    row_it2: Iterable[int] = range(_n_merge2)
    if _l3_policy_dataset_tqdm_enabled():
        _l3p_tf2 = _l3_policy_tqdm_file()
        row_it2 = tqdm(
            row_it2,
            desc="[L3] policy dataset",
            total=_n_merge2,
            unit="bar",
            leave=bool(_l3p_tf2.isatty()) if hasattr(_l3p_tf2, "isatty") else False,
            file=_l3p_tf2,
            mininterval=1.0,
        )
    else:
        print(
            f"  [L3] policy dataset: scanning {_n_merge2:,} merged bars (tqdm off: set DISABLE_TQDM=1) ...",
            file=sys.stderr,
            flush=True,
        )
    for i in row_it2:
        if i + 1 >= len(merged) or run_end[i] <= i + 1:
            continue
        sz = float(size[i])
        min_confidence = float(entry_policy_arrays["min_confidence"][i])
        min_size = float(entry_policy_arrays["min_size"][i])
        model_side = l3_entry_side_from_l2(
            int(decision_class[i]),
            float(decision_conf[i]),
            sz,
            min_confidence=min_confidence,
            min_size=min_size,
        )
        model_active = model_side != 0.0
        ed = float(edge_atr[i])
        truth_dir = 1
        if ed > tau_edge:
            truth_dir = 0
        elif ed < -tau_edge:
            truth_dir = 2
        truth_active = truth_dir != 1
        if model_active:
            side = float(model_side)
            n_policy_signals_model += 1
            from_model = 1
        elif truth_active and allow_truth_fallback:
            # Optional legacy fallback when L2 is too sparse; disabled by default to match deployment.
            side = 1.0 if truth_dir == 0 else -1.0
            n_policy_signals_truth += 1
            from_model = 0
        else:
            continue
        entry_price = float(open_px[i + 1])
        atr = float(safe_atr[i])
        # Default 1: avoid inflating a short meta horizon (e.g. 2) to L3_POLICY_MIN_HORIZON_BARS (legacy 5).
        min_horizon = 1
        target_horizon = int(max(min_horizon, min(int(max_hold), int(horizon_arr[i]))))
        end = min(run_end[i], i + target_horizon + 1, i + max_hold + 1)
        n_steps = int(end - (i + 1))
        if n_steps <= 0:
            continue
        idx_arr = np.arange(i + 1, end, dtype=np.int32)
        holds = np.arange(1, n_steps + 1, dtype=np.float32)
        high_seg = high_px[idx_arr]
        low_seg = low_px[idx_arr]
        close_seg = close_px[idx_arr]
        safe_entry_vol = max(float(entry_vol[i]), 1e-3)
        play_straddle = straddle_on_arr[i] > 0.5
        implied_i = float(merged["l2_implied_proxy_range"].iloc[i]) if "l2_implied_proxy_range" in merged.columns else max(float(merged["l2_range_pred"].iloc[i]) if "l2_range_pred" in merged.columns else 1.0, 0.1)
        if play_straddle:
            # Symmetric path: reward larger of up/down excursion; simplified straddle mark-to-market on unreal.
            up_leg = np.maximum(0.0, (high_seg - entry_price) / atr)
            dn_leg = np.maximum(0.0, (entry_price - low_seg) / atr)
            move_atr = np.maximum(up_leg, dn_leg)
            if side >= 0.0:
                fav_seg = move_atr
                adv_seg = np.minimum(up_leg, dn_leg)
                unreal_seg = np.abs(close_seg - entry_price) / atr
            else:
                unreal_seg = (implied_i - move_atr).astype(np.float64, copy=False)
                fav_seg = np.maximum(unreal_seg, 0.0)
                adv_seg = np.maximum(-unreal_seg, 0.0)
        elif side > 0.0:
            fav_seg = np.maximum(0.0, (high_seg - entry_price) / atr)
            adv_seg = np.maximum(0.0, (entry_price - low_seg) / atr)
            unreal_seg = (close_seg - entry_price) / atr
        else:
            fav_seg = np.maximum(0.0, (entry_price - low_seg) / atr)
            adv_seg = np.maximum(0.0, (high_seg - entry_price) / atr)
            unreal_seg = (entry_price - close_seg) / atr
        live_mfe_seg = np.maximum.accumulate(fav_seg.astype(np.float32, copy=False))
        live_mae_seg = np.maximum.accumulate(adv_seg.astype(np.float32, copy=False))
        live_edge_seg = _net_edge_atr_from_state(live_mfe_seg, live_mae_seg, holds).astype(np.float32, copy=False)
        regime_div_seg = _kl_divergence(np.repeat(entry_regime[i : i + 1], n_steps, axis=0), current_regime[idx_arr]).astype(np.float32, copy=False)
        vol_surprise_seg = (current_vol[idx_arr] / safe_entry_vol).astype(np.float32, copy=False)
        log_h_seg = np.log1p(holds).astype(np.float32, copy=False)
        h_sq_seg = ((holds * holds) / 100.0).astype(np.float32, copy=False)
        h_bkt_seg = np.searchsorted(_hold_bin_edges, holds.astype(np.int64), side="right").astype(np.float32, copy=False)
        # Causal: peak at step j is max(unreal[0:j]); no look-ahead within the episode.
        peak_unreal_cum = np.maximum.accumulate(unreal_seg.astype(np.float64))
        drawdown_from_peak = (peak_unreal_cum - unreal_seg.astype(np.float64)).astype(np.float32, copy=False)
        ep_f = max(float(entry_price), 1e-9)
        if play_straddle:
            bar_ret_frac = (np.abs(close_seg - entry_price) / ep_f).astype(np.float32, copy=False)
        else:
            bar_ret_frac = np.where(
                side > 0.0,
                (close_seg - entry_price) / ep_f,
                (entry_price - close_seg) / ep_f,
            ).astype(np.float32, copy=False)
        peak_frac = np.maximum.accumulate(bar_ret_frac.astype(np.float64))
        drawdown_frac = (peak_frac - bar_ret_frac.astype(np.float64)).astype(np.float32, copy=False)
        implied_frac = float(max(implied_i * atr / max(entry_price, 1e-9), 1e-6))
        range_realization_ratio = (bar_ret_frac.astype(np.float64) / implied_frac).astype(np.float32, copy=False)
        theta_burn_fraction = np.sqrt(holds.astype(np.float64) / max(float(target_horizon), 1.0)).astype(np.float32, copy=False)
        vixy_change_since_entry = np.zeros(n_steps, dtype=np.float32)
        if "vixy_zscore_390" in merged.columns:
            vz = pd.to_numeric(merged["vixy_zscore_390"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            vixy_change_since_entry = (vz[idx_arr] - float(vz[i])).astype(np.float32, copy=False)
        range_expansion_speed = (bar_ret_frac.astype(np.float64) / np.maximum(holds.astype(np.float64), 1.0)).astype(np.float32, copy=False)
        ret_last_3, ret_last_5, vol_trade, slope_fb = _l3_intra_episode_frac_derivatives(bar_ret_frac)
        mom_lead = _l3_episode_momentum_leading_block(
            n_steps,
            unreal_seg,
            drawdown_from_peak,
            live_mfe_seg,
            live_mae_seg,
            vol_surprise_seg,
            bar_ret_frac,
            holds,
            target_horizon,
            float(pred_mfe[i]),
        )
        close_seg = close_px[idx_arr].astype(np.float64, copy=False)
        atr_d = max(float(atr), 1e-6)
        vel3 = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            j0 = max(0, j - 3)
            vel3[j] = float((close_seg[j] - close_seg[j0]) / atr_d)
        reg_div_d = regime_div_seg.astype(np.float64, copy=False)
        mom_rd = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            j0 = max(0, j - 3)
            mom_rd[j] = float(reg_div_d[j] - reg_div_d[j0])
        vs = vol_surprise_seg.astype(np.float64, copy=False)
        vs_acc = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            if j >= 2:
                vs_acc[j] = float(vs[j] - 2.0 * vs[j - 1] + vs[j - 2])
        cr = current_regime[idx_arr].astype(np.float64, copy=False)
        rid = np.argmax(cr, axis=1).astype(np.int32, copy=False)
        stab = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            lo = max(0, j - 2)
            stab[j] = float(np.mean(rid[lo : j + 1] == rid[j]))
        aux_block = _l3_episode_aux_feature_block(
            n_steps=n_steps,
            idx_arr=idx_arr,
            entry_i=i,
            side=float(side),
            decision_class=decision_class,
            decision_conf=decision_conf,
            size=size,
            neutral=neutral,
            entry_regime_row=entry_regime[i],
            current_regime=current_regime,
            unreal_seg=unreal_seg,
            min_conf_arr=entry_policy_arrays["min_confidence"],
            min_size_arr=entry_policy_arrays["min_size"],
            drawdown_from_peak=drawdown_from_peak,
        )
        rng_i = float(merged["l2_range_pred"].iloc[i]) if "l2_range_pred" in merged.columns else 0.0
        gp_i = float(merged["l2_gate_prob"].iloc[i]) if "l2_gate_prob" in merged.columns else float(decision_conf[i])
        _stack_parts: list[np.ndarray] = [
            np.full(n_steps, float(straddle_on_arr[i]), dtype=np.float32),
            np.full(n_steps, rng_i, dtype=np.float32),
            np.full(n_steps, gp_i, dtype=np.float32),
            np.full(n_steps, decision_conf[i], dtype=np.float32),
            np.full(n_steps, size[i], dtype=np.float32),
            np.full(n_steps, pred_mfe[i], dtype=np.float32),
            np.full(n_steps, pred_mae[i], dtype=np.float32),
        ]
        for _xc in extra_merged:
            _stack_parts.append(merged.loc[idx_arr, _xc].to_numpy(dtype=np.float32, copy=False))
        _stack_parts.extend(
            [
                np.repeat(entry_regime[i : i + 1], n_steps, axis=0),
                np.full(n_steps, entry_vol[i], dtype=np.float32),
                current_regime[idx_arr],
                current_vol[idx_arr].astype(np.float32, copy=False),
                regime_div_seg,
                vol_surprise_seg,
                holds,
                unreal_seg.astype(np.float32, copy=False),
                live_mfe_seg,
                live_mae_seg,
                live_edge_seg,
                np.full(n_steps, side, dtype=np.float32),
                log_h_seg,
                h_sq_seg,
                h_bkt_seg,
                drawdown_from_peak,
                bar_ret_frac,
                drawdown_frac,
                ret_last_3,
                ret_last_5,
                vol_trade,
                slope_fb,
                mom_lead,
                vel3,
                mom_rd,
                vs_acc,
                stab,
                merged.loc[idx_arr, PA_STATE_FEATURES].to_numpy(dtype=np.float32, copy=False),
                aux_block,
                range_realization_ratio,
                theta_burn_fraction,
                vixy_change_since_entry,
                range_expansion_speed,
            ]
        )
        if _l3_exit_hold_interaction_features_enabled():
            hold_f = holds.astype(np.float64)
            unreal_f = unreal_seg.astype(np.float64)
            vel3_f = vel3.astype(np.float64)
            _stack_parts.extend(
                [
                    (hold_f * unreal_f).astype(np.float32, copy=False),
                    (hold_f * vel3_f).astype(np.float32, copy=False),
                ]
            )
        feat_block = np.column_stack(_stack_parts).astype(np.float32, copy=False)
        terminal_unreal = float(unreal_seg[-1])
        future_gain_left = (terminal_unreal - unreal_seg).astype(np.float32, copy=False)
        peak_thr = _l3_value_upside_threshold_atr()
        exit_mode = _l3_policy_exit_label_mode()
        if exit_mode in {"straddle_vol", "v2", "fwd_return"}:
            lk = _l3_policy_label_lookahead_bars()
            eps = _l3_policy_min_continuation_frac()
            if exit_mode == "straddle_vol" and play_straddle:
                exit_block = _l3_compute_exit_label_straddle_vol(
                    bar_ret_frac,
                    lookahead=lk,
                    vol_side=float(side),
                    implied_range_frac=implied_frac,
                )
                exit_block[-1] = 1
            elif exit_mode == "v2":
                exit_block = _l3_compute_exit_label_v2(bar_ret_frac, lookahead=lk, min_continuation_frac=eps)
                exit_block[-1] = 1
            elif play_straddle:
                exit_block = _l3_compute_exit_label_v2(bar_ret_frac, lookahead=lk, min_continuation_frac=eps)
                exit_block[-1] = 1
            else:
                atr_seg = safe_atr[idx_arr].astype(np.float64, copy=False)
                exit_block = _l3_compute_exit_label_fwd_return(
                    close_seg.astype(np.float64, copy=False),
                    side=float(side),
                    atr_seg=atr_seg,
                    cost_atr=_l3_exit_fwd_cost_atr(),
                    path_bad_atr=_l3_exit_fwd_path_bad_atr(),
                    path_weak_mult=_l3_exit_fwd_path_weak_mult(),
                )
            vtm = _l3_value_target_mode()
            atr_seg_row = safe_atr[idx_arr].astype(np.float64, copy=False)
            if vtm == "trade_outcome":
                value_block = np.zeros(n_steps, dtype=np.float32)
            elif vtm == "remaining_value":
                value_block = _l3_remaining_value_labels_from_unreal(unreal_seg)
            elif vtm == "remaining_value_atr":
                value_block = _l3_remaining_value_atr_from_unreal(unreal_seg, atr_seg_row)
            elif vtm == "peak_cls":
                value_block = _l3_peak_upside_value_labels(
                    close_seg=close_seg,
                    high_seg=high_seg,
                    low_seg=low_seg,
                    side=float(side),
                    play_straddle=play_straddle,
                    atr_scale=float(atr),
                    upside_threshold_atr=peak_thr,
                )
            else:
                value_block = _l3_compute_value_label_v2(bar_ret_frac, lookahead=lk)
        else:
            continuation_score = _l3_continuation_score(future_gain_left, live_edge_seg)
            step_trend = np.asarray(pa_state["pa_state_trend_strength"][idx_arr], dtype=np.float32)
            step_follow = np.asarray(pa_state["pa_state_followthrough_quality"][idx_arr], dtype=np.float32)
            step_range = np.asarray(pa_state["pa_state_range_risk"][idx_arr], dtype=np.float32)
            step_breakout = np.asarray(pa_state["pa_state_breakout_failure_risk"][idx_arr], dtype=np.float32)
            step_pullback = np.asarray(pa_state["pa_state_pullback_exhaustion"][idx_arr], dtype=np.float32)
            last_step = np.arange(n_steps) == (n_steps - 1)
            late_min = min(L3DEF.late_hold_min_bars(), max_hold)
            if L3DEF.late_hold_start_mode() in {"target_horizon_frac", "legacy", "horizon"}:
                late_hold_start = max(late_min, int(np.ceil(target_horizon * L3DEF.late_hold_frac())))
            else:
                late_hold_start = max(late_min, int(max_hold * L3DEF.late_hold_max_hold_ratio()))
            if _l3_pa_targets_enabled():
                eps_arr = (exit_epsilon_atr * pa_exit_eps_multiplier(step_range, step_breakout, step_pullback, step_trend, step_follow)).astype(
                    np.float32, copy=False
                )
                loss_buffer_arr = (
                    policy_loss_buffer_base * pa_exit_loss_buffer_multiplier(step_range, step_breakout, step_pullback, step_trend)
                ).astype(np.float32, copy=False)
                live_edge_floor_arr = (_l3_exit_live_edge_floor() * pa_exit_live_edge_floor_multiplier(step_range, step_breakout, step_trend)).astype(
                    np.float32, copy=False
                )
                late_start_scale = pa_exit_late_hold_entry_scale(pa_state, i)
                late_hold_start = max(1, int(np.ceil(late_hold_start * late_start_scale)))
            else:
                eps_arr = np.full(n_steps, exit_epsilon_atr, dtype=np.float32)
                loss_buffer_arr = np.full(n_steps, policy_loss_buffer_base, dtype=np.float32)
                live_edge_floor_arr = np.full(n_steps, _l3_exit_live_edge_floor(), dtype=np.float32)
            if L3DEF.late_hold_ramp():
                th_r = float(max(1, late_hold_start))
                span_r = max(float(max_hold) - th_r, 1.0)
                late_ramp = np.where(holds <= th_r, 1.0, np.clip(1.0 - (holds - th_r) / span_r, 0.0, 1.0)).astype(np.float32)
                eps_eff = (eps_arr / np.maximum(late_ramp, float(L3DEF.late_hold_ramp_eps_floor()))).astype(np.float32)
                weak_continuation = continuation_score <= eps_eff
            else:
                weak_continuation = continuation_score <= eps_arr
            clearly_spent = future_gain_left <= -loss_buffer_arr
            live_edge_faded = live_edge_seg <= live_edge_floor_arr
            late_flat = (holds >= late_hold_start) & (future_gain_left <= 0.0)
            exit_block = (last_step | clearly_spent | (weak_continuation & live_edge_faded) | late_flat).astype(np.int32, copy=False)
            vtm = _l3_value_target_mode()
            atr_seg_row2 = safe_atr[idx_arr].astype(np.float64, copy=False)
            if vtm == "trade_outcome":
                value_block = np.zeros(n_steps, dtype=np.float32)
            elif vtm == "remaining_value":
                value_block = _l3_remaining_value_labels_from_unreal(unreal_seg)
            elif vtm == "remaining_value_atr":
                value_block = _l3_remaining_value_atr_from_unreal(unreal_seg, atr_seg_row2)
            elif vtm == "peak_cls":
                value_block = _l3_peak_upside_value_labels(
                    close_seg=close_seg,
                    high_seg=high_seg,
                    low_seg=low_seg,
                    side=float(side),
                    play_straddle=play_straddle,
                    atr_scale=float(atr),
                    upside_threshold_atr=peak_thr,
                )
            else:
                value_block = future_gain_left
        rows_x_blocks.append(feat_block)
        rows_exit_blocks.append(exit_block)
        rows_value_blocks.append(value_block)
        rows_time_blocks.append(times[idx_arr])
        rows_merged_idx_blocks.append(idx_arr.astype(np.int64, copy=False))
        rows_entry_blocks.append(np.full(n_steps, int(i), dtype=np.int64))
        rows_from_model_blocks.append(np.full(n_steps, from_model, dtype=np.int32))
        if build_traj:
            traj_hist = np.zeros((_t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32)
            traj_len_cur = 0
            peak_unreal = -1e9
            prev_unreal = 0.0
            for local_idx, j in enumerate(idx_arr.tolist()):
                peak_unreal = max(peak_unreal, float(unreal_seg[local_idx]))
                tvec = l3_traj_step_features(
                    float(unreal_seg[local_idx]),
                    prev_unreal,
                    peak_unreal,
                    int(holds[local_idx]),
                    times[j],
                    float(close_px[j - 1]),
                    float(close_px[j]),
                    float(high_px[j]),
                    float(low_px[j]),
                    atr,
                    float(vol_surprise_seg[local_idx]),
                    float(regime_div_seg[local_idx]),
                    float(live_mfe_seg[local_idx]),
                    float(live_mae_seg[local_idx]),
                    max_seq_ref=_t_ref,
                    mfe_scale=_t_cfg_eff.mfe_norm_scale,
                    mae_scale=_t_cfg_eff.mae_norm_scale,
                )
                prev_unreal = float(unreal_seg[local_idx])
                if traj_len_cur < _t_max:
                    traj_hist[traj_len_cur] = tvec
                    traj_len_cur += 1
                else:
                    traj_hist[:-1] = traj_hist[1:]
                    traj_hist[-1] = tvec
                rows_traj.append(traj_hist.copy())
                rows_traj_len.append(traj_len_cur)
    if not rows_x_blocks:
        print(
            f"  [L3] policy dataset empty: no bars with L2 model trade (class≠neutral, size>0.05) "
            f"or label edge (|decision_edge_atr|>{tau_edge}) at a row with same-symbol next bar.",
            flush=True,
        )
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            feature_cols,
            np.empty(0, dtype=np.int64),
            np.empty((0, _t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            {
                "policy_state_vol_quantiles": policy_vol_quantiles,
                "l3_entry_policy": entry_policy_global,
                "l3_entry_policy_by_state": entry_policy_by_state,
                "l3_trust_l2_entry": bool(_l3_trust_l2_entry_enabled()),
                "l3_target_horizon_bars": target_horizon_global,
                "l3_target_horizon_bars_by_state": target_horizon_by_state,
                "pa_target_semantics": "PA-aware continuation thresholds, horizon scaling, and entry-quality scoring are applied inside dataset/target construction",
                "l3_traj_cfg_dict": asdict(_t_cfg_eff),
                "l3_policy_loss_buffer": dict(policy_loss_buffer_meta),
                "l3_trade_semantics": L3DEF.trade_semantics_default(),
                "policy_rows_merged_idx": np.empty(0, dtype=np.int64),
            },
        )
    print(
        f"  [L3] policy dataset: entry signals model={n_policy_signals_model:,} "
        f"truth_edge_fallback={n_policy_signals_truth:,}  policy_rows={sum(int(x.shape[0]) for x in rows_x_blocks):,}  "
        f"allow_truth_fallback={allow_truth_fallback}  entry_policy_states={len(entry_policy_by_state)}  "
        f"target_horizon_global={target_horizon_global}",
        flush=True,
    )
    total_entries = int(n_policy_signals_model + n_policy_signals_truth)
    total_rows = int(sum(int(x.shape[0]) for x in rows_x_blocks))
    avg_rows_all = float(total_rows / total_entries) if total_entries > 0 else float("nan")
    avg_rows_model = float(total_rows / n_policy_signals_model) if n_policy_signals_model > 0 else float("nan")
    print(
        f"  [L3] policy dataset detail: total_entries={total_entries:,}  "
        f"avg_rows_per_entry={avg_rows_all:.2f}  avg_rows_per_model_entry={avg_rows_model:.2f}",
        flush=True,
    )
    if n_policy_signals_truth and not n_policy_signals_model:
        print(
            "  [L3] NOTE: all policy entries from label edge fallback — L2 predictions rarely trade; "
            "L3 still uses merged L2 features at each signal bar.",
            flush=True,
        )
    return (
        np.concatenate(rows_x_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_exit_blocks, axis=0).astype(np.int32, copy=False),
        np.concatenate(rows_value_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_time_blocks, axis=0),
        feature_cols,
        np.concatenate(rows_entry_blocks, axis=0).astype(np.int64, copy=False),
        (
            np.stack(rows_traj, axis=0).astype(np.float32, copy=False)
            if build_traj and rows_traj
            else np.empty((0, _t_max, _t_cfg_eff.seq_feat_dim), dtype=np.float32)
        ),
        (np.asarray(rows_traj_len, dtype=np.int32) if build_traj and rows_traj_len else np.empty(0, dtype=np.int32)),
        np.concatenate(rows_from_model_blocks, axis=0).astype(np.int32, copy=False),
        {
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "l3_entry_policy": entry_policy_global,
            "l3_entry_policy_by_state": entry_policy_by_state,
            "l3_trust_l2_entry": bool(_l3_trust_l2_entry_enabled()),
            "l3_target_horizon_bars": target_horizon_global,
            "l3_target_horizon_bars_by_state": target_horizon_by_state,
            "l3_traj_cfg_dict": asdict(_t_cfg_eff),
            "l3_policy_loss_buffer": dict(policy_loss_buffer_meta),
            "l3_trade_semantics": L3DEF.trade_semantics_default(),
            "policy_rows_merged_idx": np.concatenate(rows_merged_idx_blocks, axis=0).astype(np.int64, copy=False),
        },
    )


def _l3_policy_mode_ablation_metrics(
    y_exit: np.ndarray,
    p_exit: np.ndarray,
    value_econ: np.ndarray,
    *,
    exit_prob_threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    p = np.asarray(p_exit, dtype=np.float64).ravel()
    v = np.asarray(value_econ, dtype=np.float64).ravel()
    thr = float(np.clip(exit_prob_threshold, 0.0, 1.0))
    pred_prob = (p >= thr).astype(np.int32)
    tail_weight = float(L3DEF.policy_utility_tail_weight())
    hold_mask = y == 0
    pred_hold = pred_prob == 0
    exit_rate = float(np.mean(pred_prob))
    hold_recall = float(np.mean(pred_prob[hold_mask] == 0)) if np.any(hold_mask) else float("nan")
    hold_precision = float(np.mean(y[pred_hold] == 0)) if np.any(pred_hold) else float("nan")
    econ = _l3_economic_uplift_score(pred_prob, v, tail_weight=tail_weight, sample_weight=None)
    m_prob: dict[str, float] = {
        "acc": float(accuracy_score(y, pred_prob)),
        "f1": float(f1_score(y, pred_prob, zero_division=0)),
        "exit_rate": exit_rate,
        "hold_recall": hold_recall,
        "hold_precision": hold_precision,
        "utility_mean": float(econ["uplift_mean"]),
        "utility_p10": float(econ["uplift_p10"]),
        "score": float(econ["score"]),
    }
    return {"prob_only": m_prob}


def _log_l3_val_extended(
    X: np.ndarray,
    y_exit: np.ndarray,
    y_value: np.ndarray,
    t_state: pd.Series | np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    exit_model: lgb.Booster,
    value_model: lgb.Booster | None,
    *,
    X_value: np.ndarray | None = None,
    value_nonzero_model: lgb.Booster | None = None,
    value_hurdle_prob_power: float = 1.0,
    value_prep: dict[str, Any] | None = None,
    exit_calibrator: Any = None,
    exit_prob_threshold: float = 0.5,
    exit_policy_summary: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    out = {
        "val_hold_recall": float("nan"),
        "val_exit_rate": float("nan"),
        "val_auc": float("nan"),
        "holdout_hold_recall": float("nan"),
        "holdout_auc": float("nan"),
    }
    vm = np.asarray(val_mask, dtype=bool)
    if int(vm.sum()) < 5:
        return out
    p_exit = _apply_l3_exit_calibrator(exit_model.predict(X[vm]).astype(np.float64), exit_calibrator)
    yv = y_exit[vm].astype(np.int32)
    ih_hold = feature_cols.index("l3_hold_bars")
    hold_vm = np.asarray(X[vm, ih_hold], dtype=np.int64)
    try:
        ll = float(log_loss(yv, p_exit))
    except ValueError:
        ll = float("nan")
    try:
        auc = float(roc_auc_score(yv, p_exit))
    except ValueError:
        auc = float("nan")
    br = brier_binary(yv.astype(np.float64), p_exit)
    ece = ece_binary(yv, p_exit)
    yhat = (p_exit >= 0.5).astype(np.int32)
    acc = float(accuracy_score(yv, yhat))
    f1 = float(f1_score(yv, yhat, zero_division=0))
    cm = confusion_matrix(yv, yhat, labels=[0, 1])
    hold_recall = float(np.mean(yhat[yv == 0] == 0)) if np.any(yv == 0) else float("nan")
    hold_precision = float(np.mean(yv[yhat == 0] == 0)) if np.any(yhat == 0) else float("nan")
    exit_recall = float(np.mean(yhat[yv == 1] == 1)) if np.any(yv == 1) else float("nan")
    exit_precision = float(np.mean(yv[yhat == 1] == 1)) if np.any(yhat == 1) else float("nan")
    exit_rate = float(np.mean(yhat))
    out["val_hold_recall"] = hold_recall
    out["val_exit_rate"] = exit_rate
    out["val_auc"] = auc
    print("\n  [L3] val — exit (extended)", flush=True)
    print(
        f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  acc@0.5={acc:.4f}  F1={f1:.4f}  "
        f"exit_rate={exit_rate:.3f}  hold_recall={hold_recall:.4f}  hold_precision={hold_precision:.4f}  "
        f"exit_recall={exit_recall:.4f}  exit_precision={exit_precision:.4f}",
        flush=True,
    )
    print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)
    if exit_policy_summary:
        print(
            "    policy-search decomposition: "
            f"utility_mean={float(exit_policy_summary.get('utility_mean', float('nan'))):.4f}  "
            f"utility_p10={float(exit_policy_summary.get('utility_p10', float('nan'))):.4f}  "
            f"hold_recall_contrib={float(exit_policy_summary.get('hold_recall_contrib', float('nan'))):.4f}  "
            f"exit_rate_penalty_contrib={float(exit_policy_summary.get('exit_rate_penalty_contrib', float('nan'))):.4f}",
            flush=True,
        )
    if np.isfinite(hold_recall) and hold_recall < 0.20:
        print(f"    WARNING: hold recall still very low ({hold_recall:.3f})", flush=True)

    print(" exit AUC by hold bucket (val):", flush=True)
    for lo, hi in [(0, 3), (3, 8), (8, 15), (15, 30), (30, 10_000)]:
        m_hold = (hold_vm >= lo) & (hold_vm < hi)
        n_sub = int(m_hold.sum())
        if n_sub < 50:
            continue
        yy = yv[m_hold]
        pp = p_exit[m_hold]
        if len(np.unique(yy)) < 2:
            continue
        try:
            auc_h = float(roc_auc_score(yy, pp))
        except ValueError:
            auc_h = float("nan")
        print(f"      hold [{lo:>3d}, {hi:>5d}): n={n_sub:>6d}  AUC={auc_h:.4f}", flush=True)

    reg_cols = sorted(
        [c for c in feature_cols if c.startswith("l2_entry_regime_")],
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if reg_cols:
        reg_idx = [feature_cols.index(c) for c in reg_cols]
        E = X[vm][:, reg_idx]
        reg_id = np.argmax(E, axis=1)
        for k in range(len(reg_cols)):
            m = reg_id == k
            n_k = int(m.sum())
            if n_k < 20:
                print(f"    entry-regime {k}  n={n_k:,}  [WARN tiny slice: skip AUC]", flush=True)
                continue
            yy = yv[m]
            pp = p_exit[m]
            if len(np.unique(yy)) < 2:
                continue
            try:
                auc_k = float(roc_auc_score(yy, pp))
                warn = "  [WARN n<50]" if n_k < 50 else ""
                print(f"    entry-regime {k}  AUC={auc_k:.4f}  n={n_k:,}{warn}", flush=True)
            except ValueError:
                pass

    if "l3_regime_divergence" in feature_cols:
        idx_div = feature_cols.index("l3_regime_divergence")
        dvm = np.asarray(X[vm, idx_div], dtype=np.float64)
        fin = np.isfinite(dvm)
        if fin.any():
            err_abs = np.abs(p_exit - yv.astype(np.float64))
            c_div_err = pearson_corr(dvm[fin], err_abs[fin])
            print(
                f"    l3_regime_divergence (val): mean={float(np.mean(dvm[fin])):.6f}  std={float(np.std(dvm[fin])):.6f}  "
                f"corr(div, |p-y|)={c_div_err:.4f}",
                flush=True,
            )

    c_ec = float("nan")
    if "l2_decision_confidence" in feature_cols:
        i_conf = feature_cols.index("l2_decision_confidence")
        c_ec = pearson_corr(X[vm, i_conf].astype(np.float64), p_exit)
    Xv = X if X_value is None else X_value
    vv_val_report = None
    if value_model is not None:
        vv_val_report = _l3_value_predict_hurdle(
            Xv[vm],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
            prep=value_prep,
        )
    if value_model is None:
        print(f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  (L3 value model disabled)", flush=True)
    else:
        c_sz_val = float("nan")
        if "l2_size" in feature_cols:
            i_size = feature_cols.index("l2_size")
            c_sz_val = pearson_corr(X[vm, i_size].astype(np.float64), vv_val_report)
        print(
            f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  corr(L2 size, L3 value pred)={c_sz_val:.4f}",
            flush=True,
        )
    mode_cmp = _l3_policy_mode_ablation_metrics(
        yv,
        p_exit,
        y_value[vm].astype(np.float64),
        exit_prob_threshold=float(exit_prob_threshold),
    )
    print(
        "    policy ablation(val_report, prob_only): "
        f"acc={mode_cmp['prob_only']['acc']:.4f}  F1={mode_cmp['prob_only']['f1']:.4f}  "
        f"exit_rate={mode_cmp['prob_only']['exit_rate']:.3f}  utility_mean={mode_cmp['prob_only']['utility_mean']:.4f}",
        flush=True,
    )

    t_vm = np.asarray(pd.to_datetime(t_state))[vm]
    order = np.argsort(t_vm)
    fr = flip_rate_sorted(p_exit, order)
    print(f"    exit prob flip_rate (time-sorted val): raw={fr:.6f}", flush=True)

    if value_model is None:
        print("\n  [L3] val — value (extended): skipped (L3_VALUE_MODE=disabled)", flush=True)
    else:
        vv_pred = vv_val_report
        vv_true = y_value[vm].astype(np.float64)
        print("\n  [L3] val — value (extended)", flush=True)
        vtm_log = str(value_prep.get("value_target_mode", "")).strip().lower() if value_prep is not None else ""
        if value_prep is not None and vtm_log in ("peak_cls", "trade_outcome", "remaining_value"):
            yv = vv_true.astype(np.int32)
            try:
                auc_v = float(roc_auc_score(yv, vv_pred))
            except ValueError:
                auc_v = float("nan")
            try:
                ll_v = float(log_loss(yv, np.clip(vv_pred, 1e-6, 1.0 - 1e-6)))
            except ValueError:
                ll_v = float("nan")
            br_v = brier_binary(yv.astype(np.float64), vv_pred)
            ece_v = ece_binary(yv, vv_pred)
            acc_v = float(accuracy_score(yv, (vv_pred >= 0.5).astype(np.int32)))
            std_v = float(np.std(vv_pred))
            print(
                f"    {vtm_log}  AUC={auc_v:.4f}  log_loss={ll_v:.4f}  Brier={br_v:.4f}  ECE={ece_v:.4f}  acc@0.5={acc_v:.4f}  "
                f"pred_std={std_v:.6f}",
                flush=True,
            )
            if reg_cols:
                for k in range(len(reg_cols)):
                    m = reg_id == k
                    n_k = int(m.sum())
                    if n_k < 15:
                        continue
                    yt = yv[m]
                    yp = vv_pred[m]
                    if len(np.unique(yt)) < 2:
                        continue
                    try:
                        auc_k = float(roc_auc_score(yt, yp))
                    except ValueError:
                        auc_k = float("nan")
                    print(f"    entry-regime {k}  AUC={auc_k:.4f}  n={n_k:,}", flush=True)
        else:
            mae_v = float(mean_absolute_error(vv_true, vv_pred))
            rmse_v = float(np.sqrt(mean_squared_error(vv_true, vv_pred)))
            r2_v = float(r2_score(vv_true, vv_pred)) if len(np.unique(vv_true)) > 1 else float("nan")
            c_v = pearson_corr(vv_true, vv_pred)
            std_v, degen_v = regression_degen_flag(vv_pred)
            dir_acc = directional_accuracy_regression(vv_true, vv_pred)
            print(
                f"    [{vtm_log or 'regression'}]  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}  corr={c_v:.4f}  "
                f"dir_acc={dir_acc:.4f}  pred_std={std_v:.6f}  degen={degen_v}",
                flush=True,
            )
            if reg_cols:
                for k in range(len(reg_cols)):
                    m = reg_id == k
                    n_k = int(m.sum())
                    if n_k < 15:
                        continue
                    yt = vv_true[m]
                    yp = vv_pred[m]
                    mae_k = float(mean_absolute_error(yt, yp))
                    r2_k = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
                    print(f"    entry-regime {k}  MAE={mae_k:.4f}  R2={r2_k:.4f}  n={n_k:,}", flush=True)

    tm = np.asarray(test_mask, dtype=bool)
    if int(tm.sum()) >= 5:
        p_t = _apply_l3_exit_calibrator(exit_model.predict(X[tm]).astype(np.float64), exit_calibrator)
        yt = y_exit[tm].astype(np.int32)
        try:
            auc_t = float(roc_auc_score(yt, p_t))
        except ValueError:
            auc_t = float("nan")
        br_t = brier_binary(yt.astype(np.float64), p_t)
        ece_t = ece_binary(yt, p_t)
        yhat_t = (p_t >= 0.5).astype(np.int32)
        hold_recall_t = float(np.mean(yhat_t[yt == 0] == 0)) if np.any(yt == 0) else float("nan")
        hold_precision_t = float(np.mean(yt[yhat_t == 0] == 0)) if np.any(yhat_t == 0) else float("nan")
        print(
            f"\n  [L3] holdout — exit AUC={auc_t:.4f}  Brier={br_t:.4f}  ECE={ece_t:.4f}  "
            f"acc@0.5={float(accuracy_score(yt, yhat_t)):.4f}  F1={float(f1_score(yt, yhat_t, zero_division=0)):.4f}  "
            f"hold_recall={hold_recall_t:.4f}  hold_precision={hold_precision_t:.4f}  n={int(tm.sum()):,}",
            flush=True,
        )
        Xv_ho = X if X_value is None else X_value
        vv_holdout = None
        if value_model is not None:
            vv_holdout = _l3_value_predict_hurdle(
                Xv_ho[tm],
                value_model,
                value_nonzero_model,
                prob_power=value_hurdle_prob_power,
                prep=value_prep,
            )
        holdout_cmp = _l3_policy_mode_ablation_metrics(
            yt,
            p_t,
            y_value[tm].astype(np.float64),
            exit_prob_threshold=float(exit_prob_threshold),
        )
        print(
            "    policy ablation(holdout, prob_only): "
            f"acc={holdout_cmp['prob_only']['acc']:.4f}  F1={holdout_cmp['prob_only']['f1']:.4f}  "
            f"exit_rate={holdout_cmp['prob_only']['exit_rate']:.3f}  utility_mean={holdout_cmp['prob_only']['utility_mean']:.4f}",
            flush=True,
        )
        out["holdout_hold_recall"] = hold_recall_t
        out["holdout_auc"] = auc_t
    return out


from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.trainers.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1B_META_FILE,
    L1B_OUTPUT_CACHE_FILE,
    L1B_SCHEMA_VERSION,
    MODEL_DIR,
    TRAIN_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _mfe_mae_atr_arrays,
    _numeric_feature_cols_for_matrix,
    _options_target_config,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.trainers.val_metrics_extra import brier_binary
from core.trainers.stack_v2_common import (
    build_stack_time_splits,
    compute_cross_asset_context,
    diagnose_l1b_leakage,
    log_label_baseline,
    save_output_cache,
)


L1B_MODEL_HEADS = [
    "l1b_pullback_setup",
    "l1b_failure_risk",
    "l1b_shock_risk",
]
L1B_DIRECT_SEMANTIC_COLS = [
    "l1b_breakout_quality",
    "l1b_mean_reversion_setup",
    "l1b_trend_strength",
    "l1b_range_reversal_setup",
    "l1b_failed_breakout_setup",
    "l1b_setup_alignment",
    "l1b_follow_through_score",
    "l1b_liquidity_score",
]
L1B_BINARY_HEADS: tuple[str, ...] = ()
L1B_DIRECT_CONTEXT_COLS = [
    "l1b_sector_relative_strength",
    "l1b_correlation_regime",
    "l1b_market_breadth",
]
L1B_OUTPUT_COLS = list(L1B_DIRECT_SEMANTIC_COLS) + list(L1B_MODEL_HEADS) + list(L1B_DIRECT_CONTEXT_COLS)


@dataclass
class L1BTrainingBundle:
    models: dict[str, lgb.Booster]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _select_l1b_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    keep = []
    orthogonal_stat_cols = [
        "pa_hmm_state",
        "pa_hmm_transition_pressure",
        "pa_garch_vol",
        "pa_garch_shock",
        "pa_garch_vol_of_vol",
        "pa_hsmm_duration_norm",
        "pa_hsmm_remaining_duration",
        "pa_hsmm_switch_hazard",
        "pa_hsmm_duration_percentile",
        "pa_egarch_leverage_effect",
        "pa_egarch_downside_vol_ratio",
        "pa_egarch_vol_asymmetry",
        "pa_egarch_std_residual",
    ]
    base_pref = [
        "bo_body_atr",
        "bo_range_atr",
        "bo_vol_spike",
        "bo_close_extremity",
        "bo_wick_imbalance",
        "bo_range_compress",
        "bo_body_growth",
        "bo_gap_signal",
        "bo_consec_dir",
        "bo_inside_prior",
        "bo_pressure_diff",
        "bo_or_dist",
        "bo_bb_width",
        "bo_atr_zscore",
        "pa_ctx_setup_long",
        "pa_ctx_setup_short",
        "pa_ctx_setup_trend_long",
        "pa_ctx_setup_trend_short",
        "pa_ctx_setup_pullback_long",
        "pa_ctx_setup_pullback_short",
        "pa_ctx_setup_range_long",
        "pa_ctx_setup_range_short",
        "pa_ctx_setup_failed_breakout_long",
        "pa_ctx_setup_failed_breakout_short",
        "pa_ctx_follow_through_long",
        "pa_ctx_follow_through_short",
        "pa_ctx_range_pressure",
        "pa_ctx_structure_veto",
        "pa_ctx_premise_break_long",
        "pa_ctx_premise_break_short",
        "pa_vol_rvol",
        "pa_vol_momentum",
        "pa_bo_wick_imbalance",
        "pa_bo_close_extremity",
        "pa_lead_macd_hist_slope",
        "pa_lead_rsi_slope",
        "pa_bo_dist_vwap",
        "pa_struct_swing_range_atr",
        "pa_vol_exhaustion_climax",
        "pa_vol_zscore_20",
        "pa_vol_evr_ratio",
        "pa_vol_absorption_bull",
        "pa_vol_absorption_bear",
    ]
    keep.extend([c for c in base_pref if c in df.columns])
    keep.extend([c for c in orthogonal_stat_cols if c in feat_cols and c in df.columns])
    ts = pd.to_datetime(df["time_key"])
    minutes = (ts.dt.hour * 60 + ts.dt.minute).astype(np.float32)
    df["l1b_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)
    keep.append("l1b_session_progress")
    return _numeric_feature_cols_for_matrix(df, keep)


def _l1b_cross_context_reliable(df: pd.DataFrame) -> bool:
    n_symbols = int(pd.Series(df["symbol"]).nunique(dropna=True))
    n_rows = len(df)
    return n_symbols >= 2 and n_rows >= 200


def _col_f32(df: pd.DataFrame, name: str) -> np.ndarray:
    """Read a numeric column as float32; missing column -> zeros (``df.get(..., 0.0)`` is a scalar and breaks ``fillna``)."""
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _stretch_score(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[fit]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.full_like(arr, 0.5, dtype=np.float32)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32, copy=False)


def _clip01(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _l1b_head_feature_cols(head: str, feature_cols: list[str]) -> tuple[list[str], bool]:
    preferred = {
        "l1b_breakout_quality": [
            "bo_body_atr",
            "bo_range_atr",
            "bo_vol_spike",
            "bo_close_extremity",
            "bo_wick_imbalance",
            "bo_body_growth",
            "bo_gap_signal",
            "bo_consec_dir",
            "bo_or_dist",
            "bo_atr_zscore",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "pa_vol_rvol",
            "pa_hsmm_switch_hazard",
            "pa_egarch_std_residual",
        ],
        "l1b_mean_reversion_setup": [
            "pa_ctx_range_pressure",
            "bo_inside_prior",
            "bo_bb_width",
            "bo_or_dist",
            "pa_struct_swing_range_atr",
            "pa_vol_zscore_20",
            "pa_hsmm_duration_norm",
            "pa_hsmm_switch_hazard",
        ],
        "l1b_trend_strength": [
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "bo_consec_dir",
            "bo_body_growth",
            "bo_atr_zscore",
            "pa_vol_momentum",
            "pa_lead_macd_hist_slope",
            "pa_hmm_transition_pressure",
            "pa_hsmm_switch_hazard",
        ],
        "l1b_pullback_setup": [
            "pa_ctx_setup_trend_long",
            "pa_ctx_setup_trend_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "bo_or_dist",
            "bo_inside_prior",
            "pa_struct_swing_range_atr",
            "pa_hsmm_remaining_duration",
            "pa_ctx_range_pressure",
            "bo_bb_width",
        ],
        "l1b_range_reversal_setup": [
            "pa_ctx_range_pressure",
            "bo_inside_prior",
            "bo_bb_width",
            "bo_or_dist",
            "bo_close_extremity",
            "pa_vol_absorption_bull",
            "pa_vol_absorption_bear",
            "pa_hsmm_duration_percentile",
        ],
        "l1b_failed_breakout_setup": [
            "bo_wick_imbalance",
            "bo_close_extremity",
            "bo_gap_signal",
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_garch_shock",
            "pa_egarch_vol_asymmetry",
            "pa_hmm_transition_pressure",
            "pa_ctx_structure_veto",
        ],
        "l1b_setup_alignment": [
            "pa_ctx_setup_long",
            "pa_ctx_setup_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_ctx_structure_veto",
            "pa_hsmm_switch_hazard",
            "pa_hsmm_remaining_duration",
            "pa_egarch_std_residual",
        ],
        "l1b_follow_through_score": [
            "bo_consec_dir",
            "bo_body_growth",
            "bo_close_extremity",
            "pa_vol_rvol",
            "pa_vol_momentum",
            "pa_lead_macd_hist_slope",
            "pa_hsmm_switch_hazard",
            "pa_egarch_std_residual",
            "bo_gap_signal",
            "bo_body_atr",
        ],
        "l1b_failure_risk": [
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_ctx_structure_veto",
            "pa_ctx_range_pressure",
            "bo_wick_imbalance",
            "pa_garch_shock",
            "pa_egarch_downside_vol_ratio",
            "pa_hsmm_switch_hazard",
        ],
        "l1b_shock_risk": [
            "pa_garch_shock",
            "pa_garch_vol",
            "pa_garch_vol_of_vol",
            "pa_egarch_leverage_effect",
            "pa_egarch_downside_vol_ratio",
            "pa_egarch_vol_asymmetry",
            "pa_egarch_std_residual",
            "pa_hsmm_switch_hazard",
            "pa_hsmm_duration_norm",
            "bo_atr_zscore",
            "bo_vol_spike",
            "pa_vol_zscore_20",
        ],
        "l1b_liquidity_score": [
            "pa_vol_rvol",
            "bo_wick_imbalance",
            "bo_body_atr",
            "bo_range_atr",
            "bo_close_extremity",
            "pa_vol_absorption_bull",
            "pa_vol_absorption_bear",
            "pa_vol_evr_ratio",
            "pa_vol_exhaustion_climax",
            "pa_egarch_std_residual",
        ],
    }.get(head)
    if preferred is None:
        return list(feature_cols), False
    chosen = [c for c in preferred if c in feature_cols]
    min_features = min(8, max(5, len(preferred) // 2))
    if len(chosen) >= min_features:
        return chosen, False
    safe_fallback = [
        c
        for c in feature_cols
        if c
        not in {
            "pa_ctx_setup_long",
            "pa_ctx_setup_short",
            "pa_ctx_setup_trend_long",
            "pa_ctx_setup_trend_short",
            "pa_ctx_setup_pullback_long",
            "pa_ctx_setup_pullback_short",
            "pa_ctx_setup_range_long",
            "pa_ctx_setup_range_short",
            "pa_ctx_setup_failed_breakout_long",
            "pa_ctx_setup_failed_breakout_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
        }
    ]
    augmented = list(dict.fromkeys(chosen + safe_fallback[: max(0, min_features - len(chosen) + 4)]))
    return augmented, True


def _build_l1b_direct_semantic_outputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    trend_long = _col_f32(df, "pa_ctx_setup_trend_long")
    trend_short = _col_f32(df, "pa_ctx_setup_trend_short")
    pullback_long = _col_f32(df, "pa_ctx_setup_pullback_long")
    pullback_short = _col_f32(df, "pa_ctx_setup_pullback_short")
    range_long = _col_f32(df, "pa_ctx_setup_range_long")
    range_short = _col_f32(df, "pa_ctx_setup_range_short")
    failed_breakout_long = _col_f32(df, "pa_ctx_setup_failed_breakout_long")
    failed_breakout_short = _col_f32(df, "pa_ctx_setup_failed_breakout_short")
    follow_long = _col_f32(df, "pa_ctx_follow_through_long")
    follow_short = _col_f32(df, "pa_ctx_follow_through_short")
    structure_veto = _col_f32(df, "pa_ctx_structure_veto")
    premise_break_long = _col_f32(df, "pa_ctx_premise_break_long")
    premise_break_short = _col_f32(df, "pa_ctx_premise_break_short")
    vol_rvol = _col_f32(df, "pa_vol_rvol")
    wick_imbalance = _col_f32(df, "bo_wick_imbalance")

    breakout_quality = _clip01(
        0.22 * np.clip(_col_f32(df, "bo_body_atr"), 0.0, 2.0) / 2.0
        + 0.18 * np.clip(_col_f32(df, "bo_range_atr"), 0.0, 2.5) / 2.5
        + 0.18 * np.clip(_col_f32(df, "bo_vol_spike"), 0.0, 2.0) / 2.0
        + 0.14 * np.clip(_col_f32(df, "bo_close_extremity"), 0.0, 1.0)
        + 0.12 * np.maximum(follow_long, follow_short)
        + 0.10 * np.maximum(_col_f32(df, "pa_ctx_setup_long"), _col_f32(df, "pa_ctx_setup_short"))
        - 0.10 * structure_veto
        - 0.08 * np.maximum(premise_break_long, premise_break_short)
    )
    mean_reversion_setup = _clip01(np.maximum(range_long, range_short))
    trend_strength = _clip01(np.maximum(trend_long, trend_short))
    range_reversal_setup = _clip01(np.maximum(range_long, range_short))
    failed_breakout_setup = _clip01(np.maximum(failed_breakout_long, failed_breakout_short))

    long_continuation = np.maximum(trend_long, pullback_long) * follow_long * (1.0 - premise_break_long)
    short_continuation = np.maximum(trend_short, pullback_short) * follow_short * (1.0 - premise_break_short)
    long_reversal = np.maximum(range_long, failed_breakout_long) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_long)
    short_reversal = np.maximum(range_short, failed_breakout_short) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_short)
    setup_alignment = _clip01(np.maximum.reduce([long_continuation, short_continuation, long_reversal, short_reversal]))
    follow_through_score = _clip01(np.maximum(follow_long, follow_short))
    liquidity_score = _clip01(0.5 + 0.15 * vol_rvol - 0.10 * wick_imbalance)

    return {
        "l1b_breakout_quality": breakout_quality.astype(np.float32, copy=False),
        "l1b_mean_reversion_setup": mean_reversion_setup.astype(np.float32, copy=False),
        "l1b_trend_strength": trend_strength.astype(np.float32, copy=False),
        "l1b_range_reversal_setup": range_reversal_setup.astype(np.float32, copy=False),
        "l1b_failed_breakout_setup": failed_breakout_setup.astype(np.float32, copy=False),
        "l1b_setup_alignment": setup_alignment.astype(np.float32, copy=False),
        "l1b_follow_through_score": follow_through_score.astype(np.float32, copy=False),
        "l1b_liquidity_score": liquidity_score.astype(np.float32, copy=False),
    }


def _build_l1b_targets(df: pd.DataFrame, *, fit_mask: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    direct_outputs = _build_l1b_direct_semantic_outputs(df)
    edge = np.clip(_decision_edge_atr_array(df), -4.0, 4.0).astype(np.float32)
    mfe, mae = _mfe_mae_atr_arrays(df)
    mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    cfg = _options_target_config()
    horizon = float(max(int(cfg["decision_horizon_bars"]), 1))
    if "decision_peak_bar" in df.columns:
        peak_bar = pd.to_numeric(df["decision_peak_bar"], errors="coerce").fillna(horizon).to_numpy(dtype=np.float32)
    else:
        peak_bar = np.full(len(df), horizon, dtype=np.float32)
    peak_bar = np.clip(peak_bar, 1.0, horizon)
    peak_frac = np.clip(peak_bar / horizon, 0.0, 1.0).astype(np.float32)
    late_peak_bonus = np.clip((peak_frac - 0.25) / 0.75, 0.0, 1.0).astype(np.float32)
    early_peak_bonus = np.clip(1.0 - peak_frac, 0.0, 1.0).astype(np.float32)
    rr = mfe / np.maximum(mae, 0.10)
    pullback_raw = np.clip(edge, 0.0, None) * np.exp(-0.55 * mae) * (0.35 + 0.65 * late_peak_bonus)
    failure_raw = np.clip(-edge, 0.0, None) + 0.55 * mae + 0.20 * np.clip(1.5 - rr, 0.0, 1.5)
    shock_raw = (mfe + mae) * (0.70 + 0.30 * early_peak_bonus) + 0.15 * np.abs(edge)
    targets = {
        "l1b_pullback_setup": _stretch_score(pullback_raw, fit_mask=fit_mask),
        "l1b_failure_risk": _stretch_score(failure_raw, fit_mask=fit_mask),
        "l1b_shock_risk": _stretch_score(shock_raw, fit_mask=fit_mask),
    }
    cross = compute_cross_asset_context(df)
    cross.index = df.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    return targets, direct_outputs, cross


def _compute_l1b_deterministic_outputs(
    direct_outputs: dict[str, np.ndarray],
    cross: pd.DataFrame,
    *,
    cross_context_reliable: bool,
) -> pd.DataFrame:
    n_rows = len(cross.index)
    out = pd.DataFrame(index=cross.index)
    for col in L1B_DIRECT_SEMANTIC_COLS:
        values = np.asarray(direct_outputs.get(col, np.zeros(n_rows, dtype=np.float32)), dtype=np.float32).ravel()
        out[col] = values
    for col in L1B_DIRECT_CONTEXT_COLS:
        values = pd.to_numeric(cross[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if not cross_context_reliable:
            values = np.zeros(n_rows, dtype=np.float32)
        out[col] = values
    return out


def _corr1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _l1b_val_report(head: str, y_t: np.ndarray, y_p: np.ndarray) -> None:
    """Validation metrics on calibration split (train=pre-CAL_END, val=cal window)."""
    y_t = np.asarray(y_t, dtype=np.float64).ravel()
    y_p = np.asarray(y_p, dtype=np.float64).ravel()
    n = len(y_t)
    print(f"\n  [L1b] val — {head}  (n={n:,})", flush=True)
    if n < 5:
        print("    (skip: too few val rows)", flush=True)
        return

    if head in L1B_BINARY_HEADS:
        yi = np.clip(np.round(y_t), 0, 1).astype(np.int32)
        y_pc = np.clip(y_p, 1e-7, 1.0 - 1e-7)
        try:
            auc = float(roc_auc_score(yi, y_p))
        except ValueError:
            auc = float("nan")
        try:
            ll = float(log_loss(yi, y_pc))
        except ValueError:
            ll = float("nan")
        yhat = (y_p >= 0.5).astype(np.int32)
        cm = confusion_matrix(yi, yhat, labels=[0, 1])
        br = brier_binary(yi.astype(np.float64), y_p)
        print(
            f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  acc@0.5={accuracy_score(yi, yhat):.4f}  "
            f"precision={precision_score(yi, yhat, zero_division=0):.4f}  "
            f"recall={recall_score(yi, yhat, zero_division=0):.4f}  "
            f"F1={f1_score(yi, yhat, zero_division=0):.4f}",
            flush=True,
        )
        print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)
        return

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else float("nan")
    cor = _corr1d(y_t, y_p)
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  corr(y,pred)={cor:.4f}", flush=True)

def train_l1b_market_descriptor(df: pd.DataFrame, feat_cols: list[str]) -> L1BTrainingBundle:
    work = df.copy()
    feature_cols = _select_l1b_feature_cols(work, feat_cols)
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(work["time_key"])
    train_mask = splits.train_mask
    val_mask = splits.l2_val_mask
    cal_mask = splits.cal_mask
    targets, direct_outputs, cross = _build_l1b_targets(work, fit_mask=train_mask)
    cross_context_reliable = _l1b_cross_context_reliable(work)

    log_layer_banner("[L1b] Tabular market descriptor")
    log_time_key_split(
        "L1b",
        work["time_key"],
        train_mask,
        val_mask,
        train_label="train (t < TRAIN_END)",
        val_label="val (l2_val)",
        extra_note=(
            f"Primary L1b early stopping/reporting uses l2_val end-bars inside [{TRAIN_END}, {CAL_END}); "
            f"full cal remains a secondary diagnostic."
        ),
    )
    log_numpy_x_stats("L1b", X[train_mask], label="X[train]")
    print(f"  [L1b] target/output schema L1B_OUTPUT_COLS count={len(L1B_OUTPUT_COLS)}: {L1B_OUTPUT_COLS}", flush=True)
    print(f"  [L1b] input feature count: {len(feature_cols)}", flush=True)
    print(f"  [L1b] artifact dir: {MODEL_DIR}", flush=True)
    print(f"  [L1b] will write meta/cache: {artifact_path(L1B_META_FILE)} | {artifact_path(L1B_OUTPUT_CACHE_FILE)}", flush=True)
    print(
        "  [L1b] note: L1b heads use in-memory X from this run (not OOF row-by-row unless you change data prep).",
        flush=True,
    )

    rounds = 300 if FAST_TRAIN_MODE else 1000
    es_rounds = 50 if FAST_TRAIN_MODE else 120
    model_map: dict[str, lgb.Booster] = {}
    constant_output_values: dict[str, float] = {}
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    outputs = pd.concat(
        [
            outputs,
            _compute_l1b_deterministic_outputs(
                direct_outputs,
                cross,
                cross_context_reliable=cross_context_reliable,
            ),
        ],
        axis=1,
    )
    print(f"  [L1b] model heads: {L1B_MODEL_HEADS}", flush=True)
    print(f"  [L1b] direct semantic heads: {L1B_DIRECT_SEMANTIC_COLS}", flush=True)
    print(f"  [L1b] direct context heads: {L1B_DIRECT_CONTEXT_COLS}", flush=True)
    if not cross_context_reliable:
        print(
            "  [L1b] cross-asset context deemed unreliable (need at least 2 aligned symbols/200 rows); direct context heads will be zeroed.",
            flush=True,
        )

    for name, y in tqdm(
        [(head_name, targets[head_name]) for head_name in L1B_MODEL_HEADS],
        desc="[L1b] heads",
        unit="head",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    ):
        binary = name in L1B_BINARY_HEADS
        log_label_baseline(name, y[train_mask], task="cls" if binary else "reg")
        head_feature_cols, used_fallback = _l1b_head_feature_cols(name, feature_cols)
        X_head = work[head_feature_cols].to_numpy(dtype=np.float32, copy=False)
        print(f"  [L1b] {name}: input_dim={len(head_feature_cols)}", flush=True)
        if used_fallback:
            print(
                f"  [L1b] {name}: head-specific feature subset was sparse; augmented with safe fallback features.",
                flush=True,
            )
        params = {
            "objective": "binary" if binary else "regression",
            "metric": "binary_logloss" if binary else "l2",
            "learning_rate": 0.03,
            "num_leaves": 48,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 80,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": 42,
            "n_jobs": _lgbm_n_jobs(),
        }
        if name == "l1b_shock_risk":
            params.update(
                {
                    "learning_rate": 0.02,
                    "num_leaves": 31,
                    "max_depth": 5,
                    "min_child_samples": 50,
                    "feature_fraction": 0.7,
                }
            )
        dtrain = lgb.Dataset(X_head[train_mask], label=y[train_mask], feature_name=head_feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_head[val_mask], label=y[val_mask], feature_name=head_feature_cols, free_raw_data=False)
        callbacks, cleanups = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, f"[L1b] {name}")
        try:
            model = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], callbacks=callbacks)
        finally:
            for fn in cleanups:
                fn()
        model_map[name] = model
        pred_full = model.predict(X_head).astype(np.float32)
        _l1b_val_report(name, y[val_mask], pred_full[val_mask])
        if cal_mask.sum() != val_mask.sum():
            _l1b_val_report(f"{name} [cal_full]", y[cal_mask], pred_full[cal_mask])
        outputs[name] = pred_full

    if model_map:
        diagnose_l1b_leakage(model_map, {name: _l1b_head_feature_cols(name, feature_cols)[0] for name in model_map})
    for col in L1B_OUTPUT_COLS:
        if col not in outputs.columns:
            outputs[col] = 0.0

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_files: dict[str, str] = {}
    for name, model in model_map.items():
        fname = f"{name}.txt"
        model.save_model(os.path.join(MODEL_DIR, fname))
        model_files[name] = fname

    meta = {
        "schema_version": L1B_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "output_cols": L1B_OUTPUT_COLS,
        "model_output_cols": sorted(model_map),
        "direct_output_cols": sorted(L1B_DIRECT_SEMANTIC_COLS + L1B_DIRECT_CONTEXT_COLS),
        "deterministic_output_cols": list(L1B_DIRECT_SEMANTIC_COLS + L1B_DIRECT_CONTEXT_COLS),
        "deprecated_output_cols": [],
        "constant_output_values": constant_output_values,
        "model_files": model_files,
        "head_feature_cols": {name: _l1b_head_feature_cols(name, feature_cols)[0] for name in model_map},
        "cross_context_reliable": cross_context_reliable,
        "weak_supervision_semantics": (
            "hybrid contract: deterministic semantic descriptors from current-bar context plus "
            "forward-looking predictive heads aligned to decision-window labels"
        ),
        "output_cache_file": L1B_OUTPUT_CACHE_FILE,
    }
    with open(os.path.join(MODEL_DIR, L1B_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    print(f"  [L1b] meta saved  -> {os.path.join(MODEL_DIR, L1B_META_FILE)}", flush=True)
    print(f"  [L1b] cache saved -> {cache_path}", flush=True)
    return L1BTrainingBundle(models=model_map, meta=meta, outputs=outputs)


def load_l1b_market_descriptor() -> tuple[dict[str, lgb.Booster], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L1B_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L1B_SCHEMA_VERSION:
        raise RuntimeError(
            "L1b checkpoint is incompatible with the current head contract. "
            f"Retrain L1b so artifacts match schema {L1B_SCHEMA_VERSION}."
        )
    models = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in (meta.get("model_files") or {}).items()
    }
    return models, meta


def infer_l1b_market_descriptor(models: dict[str, lgb.Booster], meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    feature_cols = list(meta["feature_cols"])
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    direct_outputs = _build_l1b_direct_semantic_outputs(work)
    cross = compute_cross_asset_context(work)
    cross.index = work.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    outputs = pd.concat(
        [
            outputs,
            _compute_l1b_deterministic_outputs(
                direct_outputs,
                cross,
                cross_context_reliable=bool(meta.get("cross_context_reliable", True)),
            ),
        ],
        axis=1,
    )
    for name, value in (meta.get("constant_output_values") or {}).items():
        outputs[name] = np.full(len(work), float(value), dtype=np.float32)
    head_feature_cols_map = meta.get("head_feature_cols") or {}
    for name, model in models.items():
        head_feature_cols = list(head_feature_cols_map.get(name) or feature_cols)
        for col in head_feature_cols:
            if col not in work.columns:
                work[col] = 0.0
        X_head = work[head_feature_cols].to_numpy(dtype=np.float32, copy=False)
        outputs[name] = model.predict(X_head).astype(np.float32)
    for col in meta.get("output_cols", L1B_OUTPUT_COLS):
        if col not in outputs.columns:
            outputs[col] = 0.0
    return outputs

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
    future_group_apply,
    log_label_baseline,
    save_output_cache,
)


L1B_OUTPUT_COLS = [
    "l1b_breakout_quality",
    "l1b_mean_reversion_setup",
    "l1b_trend_strength",
    "l1b_pullback_setup",
    "l1b_range_reversal_setup",
    "l1b_failed_breakout_setup",
    "l1b_setup_alignment",
    "l1b_follow_through_score",
    "l1b_failure_risk",
    "l1b_vol_expansion_prob",
    "l1b_vol_bucket",
    "l1b_shock_risk",
    "l1b_momentum_score",
    "l1b_liquidity_score",
    "l1b_sector_relative_strength",
    "l1b_correlation_regime",
    "l1b_market_breadth",
    "l1b_structure_integrity",
    "l1b_premise_break_risk",
]

L1B_DETERMINISTIC_HEADS = [
    "l1b_mean_reversion_setup",
    "l1b_trend_strength",
    "l1b_pullback_setup",
    "l1b_range_reversal_setup",
    "l1b_failed_breakout_setup",
    "l1b_setup_alignment",
    "l1b_follow_through_score",
    "l1b_failure_risk",
    "l1b_momentum_score",
    "l1b_liquidity_score",
    "l1b_structure_integrity",
    "l1b_premise_break_risk",
]
L1B_LEARNED_HEADS = [
    "l1b_breakout_quality",
    "l1b_vol_bucket",
    "l1b_shock_risk",
]
L1B_DEPRECATED_HEADS = [
    "l1b_vol_expansion_prob",
]
L1B_DIRECT_CONTEXT_COLS = [
    "l1b_sector_relative_strength",
    "l1b_correlation_regime",
    "l1b_market_breadth",
]


@dataclass
class L1BTrainingBundle:
    models: dict[str, lgb.Booster]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _select_l1b_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    keep = []
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
    keep.extend([c for c in feat_cols if c.startswith("pa_garch_") and c in df.columns][:5])
    keep.extend([c for c in feat_cols if c.startswith("pa_hmm_") and c in df.columns][:3])
    ts = pd.to_datetime(df["time_key"])
    minutes = (ts.dt.hour * 60 + ts.dt.minute).astype(np.float32)
    df["l1b_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)
    keep.append("l1b_session_progress")
    return _numeric_feature_cols_for_matrix(df, keep)


def _col_f32(df: pd.DataFrame, name: str) -> np.ndarray:
    """Read a numeric column as float32; missing column -> zeros (``df.get(..., 0.0)`` is a scalar and breaks ``fillna``)."""
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _build_l1b_parule_semantic_heads(df: pd.DataFrame) -> dict[str, np.ndarray]:
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

    pullback_setup = np.clip(np.maximum(pullback_long, pullback_short), 0.0, 1.0)
    range_reversal_setup = np.clip(np.maximum(range_long, range_short), 0.0, 1.0)
    failed_breakout_setup = np.clip(np.maximum(failed_breakout_long, failed_breakout_short), 0.0, 1.0)

    long_continuation = np.maximum(trend_long, pullback_long) * follow_long * (1.0 - premise_break_long)
    short_continuation = np.maximum(trend_short, pullback_short) * follow_short * (1.0 - premise_break_short)
    long_reversal = np.maximum(range_long, failed_breakout_long) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_long)
    short_reversal = np.maximum(range_short, failed_breakout_short) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_short)
    setup_alignment = np.clip(
        np.maximum.reduce([long_continuation, short_continuation, long_reversal, short_reversal]),
        0.0,
        1.0,
    )

    return {
        "l1b_pullback_setup": pullback_setup.astype(np.float32, copy=False),
        "l1b_range_reversal_setup": range_reversal_setup.astype(np.float32, copy=False),
        "l1b_failed_breakout_setup": failed_breakout_setup.astype(np.float32, copy=False),
        "l1b_setup_alignment": setup_alignment.astype(np.float32, copy=False),
    }


def _build_l1b_targets(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    cfg = _options_target_config()
    horizon = int(cfg["decision_horizon_bars"])
    safe_atr = np.where(pd.to_numeric(df["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, df["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
    mfe, mae = _mfe_mae_atr_arrays(df)
    future_range = future_group_apply(df, "high", horizon, "max") - future_group_apply(df, "low", horizon, "min")
    future_range = np.clip(future_range / safe_atr, 0.0, 5.0)

    breakout_quality = np.clip(
        np.maximum(_col_f32(df, "quality_bull_breakout"), _col_f32(df, "quality_bear_breakout")),
        0.0,
        1.0,
    )
    parule_heads = _build_l1b_parule_semantic_heads(df)
    mean_reversion_setup = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_setup_range_long"), _col_f32(df, "pa_ctx_setup_range_short")),
        0.0,
        1.0,
    )
    trend_strength = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_setup_trend_long"), _col_f32(df, "pa_ctx_setup_trend_short")),
        0.0,
        1.0,
    )
    follow_through = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_follow_through_long"), _col_f32(df, "pa_ctx_follow_through_short")),
        0.0,
        1.0,
    )
    structure_veto = _col_f32(df, "pa_ctx_structure_veto")
    premise_break = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_premise_break_long"), _col_f32(df, "pa_ctx_premise_break_short")),
        0.0,
        1.0,
    )
    failure_risk = np.clip(
        np.maximum.reduce(
            [
                _col_f32(df, "pa_ctx_setup_failed_breakout_long"),
                _col_f32(df, "pa_ctx_setup_failed_breakout_short"),
                premise_break,
                structure_veto,
            ]
        ),
        0.0,
        1.0,
    )
    vol_expansion = (future_range > 1.1).astype(np.float32)
    vol_bucket = np.digitize(future_range, bins=[0.6, 1.0, 1.5, 2.2]).astype(np.float32)
    shock_risk = (np.maximum(mfe, mae) > 1.5).astype(np.float32)
    momentum_score = np.clip(
        0.5 + 0.2 * _col_f32(df, "pa_vol_momentum") + 0.1 * _col_f32(df, "pa_lead_macd_hist_slope"),
        0.0,
        1.0,
    )
    liquidity_score = np.clip(
        0.5 + 0.15 * _col_f32(df, "pa_vol_rvol") - 0.10 * _col_f32(df, "bo_wick_imbalance"),
        0.0,
        1.0,
    )
    structure_integrity = np.clip(1.0 - structure_veto, 0.0, 1.0)

    cross = compute_cross_asset_context(df)
    cross.index = df.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    targets = {
        "l1b_breakout_quality": breakout_quality,
        "l1b_mean_reversion_setup": mean_reversion_setup,
        "l1b_trend_strength": trend_strength,
        **parule_heads,
        "l1b_follow_through_score": follow_through,
        "l1b_failure_risk": failure_risk,
        "l1b_vol_expansion_prob": vol_expansion,
        "l1b_vol_bucket": vol_bucket,
        "l1b_shock_risk": shock_risk,
        "l1b_momentum_score": momentum_score,
        "l1b_liquidity_score": liquidity_score,
        "l1b_structure_integrity": structure_integrity,
        "l1b_premise_break_risk": premise_break,
    }
    return targets, cross


def _compute_l1b_deterministic_outputs(df: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    parule_heads = _build_l1b_parule_semantic_heads(df)
    structure_veto = _col_f32(df, "pa_ctx_structure_veto")
    premise_break = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_premise_break_long"), _col_f32(df, "pa_ctx_premise_break_short")),
        0.0,
        1.0,
    )
    out["l1b_mean_reversion_setup"] = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_setup_range_long"), _col_f32(df, "pa_ctx_setup_range_short")),
        0.0,
        1.0,
    ).astype(np.float32)
    out["l1b_trend_strength"] = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_setup_trend_long"), _col_f32(df, "pa_ctx_setup_trend_short")),
        0.0,
        1.0,
    ).astype(np.float32)
    for col, values in parule_heads.items():
        out[col] = values
    out["l1b_follow_through_score"] = np.clip(
        np.maximum(_col_f32(df, "pa_ctx_follow_through_long"), _col_f32(df, "pa_ctx_follow_through_short")),
        0.0,
        1.0,
    ).astype(np.float32)
    out["l1b_failure_risk"] = np.clip(
        np.maximum.reduce(
            [
                _col_f32(df, "pa_ctx_setup_failed_breakout_long"),
                _col_f32(df, "pa_ctx_setup_failed_breakout_short"),
                premise_break,
                structure_veto,
            ]
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    out["l1b_momentum_score"] = np.clip(
        0.5 + 0.2 * _col_f32(df, "pa_vol_momentum") + 0.1 * _col_f32(df, "pa_lead_macd_hist_slope"),
        0.0,
        1.0,
    ).astype(np.float32)
    out["l1b_liquidity_score"] = np.clip(
        0.5 + 0.15 * _col_f32(df, "pa_vol_rvol") - 0.10 * _col_f32(df, "bo_wick_imbalance"),
        0.0,
        1.0,
    ).astype(np.float32)
    out["l1b_structure_integrity"] = np.clip(1.0 - structure_veto, 0.0, 1.0).astype(np.float32)
    out["l1b_premise_break_risk"] = premise_break.astype(np.float32)
    for col in L1B_DIRECT_CONTEXT_COLS:
        out[col] = pd.to_numeric(cross[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
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

    if head in ("l1b_vol_expansion_prob", "l1b_shock_risk"):
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

    if head == "l1b_vol_bucket":
        ti = np.clip(np.round(y_t), 0, 5).astype(int)
        pi = np.clip(np.round(y_p), 0, 5).astype(int)
        cm = confusion_matrix(ti, pi, labels=list(range(6)))
        print(f"    rounded-bucket 6x6 confusion (row=true, col=pred):\n    {cm}", flush=True)


def train_l1b_market_descriptor(df: pd.DataFrame, feat_cols: list[str]) -> L1BTrainingBundle:
    work = df.copy(deep=False)
    feature_cols = _select_l1b_feature_cols(work, feat_cols)
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(work["time_key"])
    targets, cross = _build_l1b_targets(work)
    train_mask = splits.train_mask
    val_mask = splits.cal_mask

    log_layer_banner("[L1b] Tabular market descriptor")
    log_time_key_split(
        "L1b",
        work["time_key"],
        train_mask,
        val_mask,
        train_label="train (t < TRAIN_END)",
        val_label="val (cal)",
        extra_note=f"Expected train t in (-inf, {TRAIN_END}), val in [{TRAIN_END}, {CAL_END}).",
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
    outputs = pd.concat([outputs, _compute_l1b_deterministic_outputs(work, cross)], axis=1)
    print(f"  [L1b] deterministic heads: {L1B_DETERMINISTIC_HEADS}", flush=True)
    print(f"  [L1b] learned heads: {L1B_LEARNED_HEADS}", flush=True)
    print(f"  [L1b] deprecated heads: {L1B_DEPRECATED_HEADS}", flush=True)
    for name in L1B_DETERMINISTIC_HEADS:
        log_label_baseline(name, targets[name][train_mask], task="reg")
        print(f"  [L1b] {name}: deterministic rule-computed; skipping model fit", flush=True)

    for name in L1B_DEPRECATED_HEADS:
        y = targets[name]
        task = "cls" if name in {"l1b_vol_expansion_prob", "l1b_shock_risk"} else "reg"
        log_label_baseline(name, y[train_mask], task=task)
        constant_value = float(np.nanmean(y[train_mask])) if int(train_mask.sum()) > 0 else 0.0
        outputs[name] = np.full(len(work), constant_value, dtype=np.float32)
        constant_output_values[name] = constant_value
        print(f"  [L1b] {name}: deprecated -> constant output {constant_value:.6f}", flush=True)

    for name, y in tqdm(
        [(head_name, targets[head_name]) for head_name in L1B_LEARNED_HEADS],
        desc="[L1b] heads",
        unit="head",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    ):
        binary = name in {"l1b_vol_expansion_prob", "l1b_shock_risk"}
        log_label_baseline(name, y[train_mask], task="cls" if binary else "reg")
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
            params["is_unbalance"] = True
        dtrain = lgb.Dataset(X[train_mask], label=y[train_mask], feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X[val_mask], label=y[val_mask], feature_name=feature_cols, free_raw_data=False)
        callbacks, cleanups = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, f"[L1b] {name}")
        try:
            model = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], callbacks=callbacks)
        finally:
            for fn in cleanups:
                fn()
        model_map[name] = model
        pred_full = model.predict(X).astype(np.float32)
        _l1b_val_report(name, y[val_mask], pred_full[val_mask])
        outputs[name] = pred_full

    if model_map:
        diagnose_l1b_leakage(model_map, feature_cols)
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
        "direct_output_cols": sorted(L1B_DETERMINISTIC_HEADS + L1B_DIRECT_CONTEXT_COLS + L1B_DEPRECATED_HEADS),
        "deterministic_output_cols": list(L1B_DETERMINISTIC_HEADS + L1B_DIRECT_CONTEXT_COLS),
        "deprecated_output_cols": list(L1B_DEPRECATED_HEADS),
        "constant_output_values": constant_output_values,
        "model_files": model_files,
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
    models = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in (meta.get("model_files") or {}).items()
    }
    return models, meta


def infer_l1b_market_descriptor(models: dict[str, lgb.Booster], meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy(deep=False)
    feature_cols = list(meta["feature_cols"])
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    cross = compute_cross_asset_context(work)
    cross.index = work.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    outputs = pd.concat([outputs, _compute_l1b_deterministic_outputs(work, cross)], axis=1)
    for name, value in (meta.get("constant_output_values") or {}).items():
        outputs[name] = np.full(len(work), float(value), dtype=np.float32)
    for name, model in models.items():
        outputs[name] = model.predict(X).astype(np.float32)
    for col in meta.get("output_cols", L1B_OUTPUT_COLS):
        if col not in outputs.columns:
            outputs[col] = 0.0
    return outputs

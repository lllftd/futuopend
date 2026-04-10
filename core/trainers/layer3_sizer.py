from __future__ import annotations

import gc
import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.data_prep import *
from core.trainers.layer2b_quality import (
    _layer3_fill_trade_stack_probs_gates,
    _l2b_nested_opp_models,
    _compute_opportunity_triplet,
    _l2b_triplet_from_trade_prob,
    _apply_cp_skip,
    _reconstruct_quality_classes,
    _build_trade_quality_targets,
)

def _regime_raw_prob_cols() -> list[str]:
    return [f"{c}_raw" for c in REGIME_NOW_PROB_COLS]


def _layer3_fill_regime_calibrated(
    regime_model: lgb.Booster,
    regime_calibrators: Any,
    work: pd.DataFrame,
    out: np.ndarray,
    chunk: int,
    *,
    raw_out: np.ndarray | None = None,
    tqdm_desc: str = "Layer3 regime→cal",
) -> None:
    n = len(work)
    n_cls = NUM_REGIME_CLASSES
    n_chunk = (n + chunk - 1) // chunk
    regime_cols = _lgbm_booster_feature_names(regime_model)
    for i in _tq(range(0, n, chunk), desc=tqdm_desc, total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_s = work[regime_cols].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
        raw = regime_model.predict(x_s)
        if raw_out is not None:
            raw_out[i:j] = raw.astype(np.float32, copy=False)
        
        if isinstance(regime_calibrators, list):
            row = np.empty((j - i, n_cls), dtype=np.float64)
            for c in range(n_cls):
                row[:, c] = regime_calibrators[c].predict(raw[:, c])
            row = np.maximum(row, 1e-12)
            row /= row.sum(axis=1, keepdims=True)
            out[i:j] = row.astype(np.float32, copy=False)
        else:
            eps = 1e-7
            l_p = np.log(np.clip(raw, eps, 1 - eps))
            row = regime_calibrators.predict_proba(l_p)
            out[i:j] = row.astype(np.float32, copy=False)
            
        del x_s, raw, row


def _layer3_attach_regime_probs_to_work(work: pd.DataFrame, cal_regime: np.ndarray) -> None:
    """Persist Layer-2a calibrated probs on ``work`` (L2b regression gate reads ``REGIME_NOW_PROB_COLS``)."""
    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        work[col] = cal_regime[:, j]
    work["regime_now_conf"] = cal_regime.max(axis=1)


def _layer3_attach_regime_raw_probs_to_work(work: pd.DataFrame, raw_regime: np.ndarray) -> None:
    """Persist uncalibrated Layer-2a probs for leakage-safe routing in downstream regression heads."""
    for j, col in enumerate(_regime_raw_prob_cols()):
        work[col] = raw_regime[:, j]


def _layer3_fill_trade_stack_probs(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
    chunk: int,
    *,
    tqdm_desc: str = "Layer3 trade stack",
) -> None:
    if not trade_quality_models.get("step1_regression"):
        raise RuntimeError("Layer 2b Step1 is regression-only; missing step1_regression in model bundle.")
    _layer3_fill_trade_stack_probs_gates(
        trade_quality_models, work, layer2_feats, p_long_gate, p_short_gate, chunk,
        tqdm_desc=tqdm_desc,
    )


def _layer3_fill_l2b_triplet_arrays(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_trade: np.ndarray,
    opp_out: np.ndarray,
    mfe_out: np.ndarray,
    mae_out: np.ndarray,
    chunk: int,
    *,
    tqdm_desc: str = "Layer3 L2b triplet (reg)",
) -> None:
    """Fill L2b regression outputs for Layer 3 (chunked). Uses Step1 regression if available."""
    regb = trade_quality_models.get("step1_regression")
    n = len(work)
    raw_cols = _regime_raw_prob_cols()
    route_cols = raw_cols if all(c in work.columns for c in raw_cols) else list(REGIME_NOW_PROB_COLS)
    regime_mat = work[route_cols].to_numpy(dtype=np.float32, copy=False)
    if regb:
        models = _l2b_nested_opp_models(regb)
        n_chunk = (n + chunk - 1) // chunk
        for i in _tq(range(0, n, chunk), desc=tqdm_desc, total=n_chunk, unit="chunk"):
            j = min(i + chunk, n)
            x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
            rp = regime_mat[i:j]
            o, mf, ma = _compute_opportunity_triplet(x_b, rp, models)
            opp_out[i:j] = o.astype(np.float32)
            mfe_out[i:j] = mf.astype(np.float32)
            mae_out[i:j] = ma.astype(np.float32)
        return
    o, mf, ma = _l2b_triplet_from_trade_prob(p_trade)
    opp_out[:] = o.astype(np.float32)
    mfe_out[:] = mf.astype(np.float32)
    mae_out[:] = ma.astype(np.float32)


def train_execution_sizer(
    df: pd.DataFrame,
    feat_cols: list[str],
    regime_model: lgb.Booster,
    regime_calibrators: Any,
    trade_quality_models: dict,
    thr_cp: float,
):
    print("\n" + "=" * 70)
    print("  LAYER 3: Execution Sizer v2 (L2b triplet × regime × TCN × GARCH × PA + gate×size)")
    print("=" * 70)

    l3_flat_tau = float(os.environ.get("L3_FLAT_TAU", "0.05"))
    l3_flat_w = float(os.environ.get("L3_FLAT_WEIGHT", "0.35"))
    l3_pos_multiplier = float(os.environ.get("L3_POS_MULTIPLIER", "3.0"))  # Amplify L3 raw prediction
    opt_cfg = _options_target_config()

    chunk = _layer3_chunk_rows()
    print(f"  Memory: chunked predicts (LAYER3_CHUNK={chunk}); shallow df, no full feature matrices")

    work = ensure_structure_context_features(df.copy(deep=False))

    n = len(work)
    cal_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    _layer3_fill_regime_calibrated(
        regime_model, regime_calibrators, work, cal_regime, chunk,
    )
    _layer3_attach_regime_probs_to_work(work, cal_regime)

    garch_cols = sorted([
        c for c in work.columns
        if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}
    ])
    layer2_feats = trade_quality_models["feature_cols"]
    thr = trade_quality_models["thresholds"]

    p_long_gate = np.empty(n, dtype=np.float32)
    p_short_gate = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(
        trade_quality_models, work, layer2_feats, p_long_gate, p_short_gate, chunk,
    )
    tcn_transition_prob_all = work["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in work.columns else None
    p_long_gate, _ = _apply_cp_skip(cal_regime, p_long_gate, thr_cp, tcn_transition_prob_all)
    p_short_gate, _ = _apply_cp_skip(cal_regime, p_short_gate, thr_cp, tcn_transition_prob_all)
    p_long_gate, p_short_gate = _apply_structure_veto_to_gates(work, p_long_gate, p_short_gate)

    p_trade_max = np.maximum(p_long_gate, p_short_gate)

    l2b_opp = np.empty(n, dtype=np.float32)
    l2b_mfe = np.empty(n, dtype=np.float32)
    l2b_mae = np.empty(n, dtype=np.float32)
    _layer3_fill_l2b_triplet_arrays(
        trade_quality_models, work, layer2_feats, p_trade_max, l2b_opp, l2b_mfe, l2b_mae, chunk,
    )

    p_range_mass = cal_regime[:, RANGE_REGIME_INDICES].sum(axis=1)
    y_cls_est = _reconstruct_quality_classes(
        p_long_gate=p_long_gate,
        p_short_gate=p_short_gate,
        p_range_mass=p_range_mass,
        thr_long=thr["long"],
        thr_short=thr["short"],
    )

    y_cls = _build_trade_quality_targets(work)
    class_size = np.array([1.0, 0.0, 0.0, -1.0], dtype=float)
    base_size = 0.7 * class_size[y_cls] + 0.3 * class_size[y_cls_est]

    edge = np.clip(_decision_edge_atr_array(work), -3.0, 3.0)
    edge_scale = np.clip(0.75 + 0.35 * edge, 0.0, 1.75)
    y_target = np.clip(base_size * edge_scale, -1.0, 1.0)

    opt_tp_atr, opt_sl_atr, _ = _optimal_exit_target_arrays(work)
    y_tp_target = np.clip(opt_tp_atr, 0.0, 6.0)
    y_sl_target = np.clip(opt_sl_atr, 0.0, 3.0)

    tcn_prob_cols = [c for c in TCN_REGIME_FUT_PROB_COLS if c in work.columns]
    mamba_prob_cols = [c for c in MAMBA_REGIME_FUT_PROB_COLS if c in work.columns]
    pa_key_cols = [c for c in LAYER3_PA_KEY_FEATURES if c in work.columns][:24]

    # Routed scalar opp (this bar's argmax regime head) × each regime's probability — 6 L3 interaction cols.
    inter_blk = (
        l2b_opp.astype(np.float64)[:, None] * cal_regime.astype(np.float64)
    ).astype(np.float32, copy=False)

    triplet_blk = np.hstack([
        l2b_opp.reshape(-1, 1),
        l2b_mfe.reshape(-1, 1),
        l2b_mae.reshape(-1, 1),
    ]).astype(np.float32, copy=False)
    sc_conf = cal_regime.max(axis=1, keepdims=True).astype(np.float32, copy=False)
    regime_blk = np.hstack([cal_regime, sc_conf]).astype(np.float32, copy=False)

    tcn_mat = work[tcn_prob_cols].to_numpy(dtype=np.float32, copy=False) if tcn_prob_cols else np.empty((n, 0), np.float32)
    mamba_mat = work[mamba_prob_cols].to_numpy(dtype=np.float32, copy=False) if mamba_prob_cols else np.empty((n, 0), np.float32)
    pa_mat = work[pa_key_cols].to_numpy(dtype=np.float32, copy=False) if pa_key_cols else np.empty((n, 0), np.float32)
    if garch_cols:
        g_mat = work[garch_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        g_mat = np.empty((n, 0), dtype=np.float32)

    X = np.hstack([triplet_blk, regime_blk, tcn_mat, mamba_mat, g_mat, pa_mat, inter_blk])
    exec_feat_cols = (
        ["l2b_opportunity_score", "l2b_pred_mfe", "l2b_pred_mae"]
        + REGIME_NOW_PROB_COLS
        + ["regime_now_conf"]
        + tcn_prob_cols
        + mamba_prob_cols
        + garch_cols
        + pa_key_cols
        + L2B_OPP_X_REGIME_COLS
    )
    _require_lgb_matrix_matches_names(X, exec_feat_cols, "Layer 3 (execution sizer v2)")

    del triplet_blk, regime_blk, tcn_mat, mamba_mat, pa_mat, g_mat, inter_blk, cal_regime
    del p_trade_max, p_long_gate, p_short_gate, sc_conf, work
    del l2b_opp, l2b_mfe, l2b_mae
    gc.collect()

    t = df["time_key"].values
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    # Split cal_mask temporally into training and validation sets for Layer 3
    cal_indices = np.where(cal_mask)[0]
    split_idx = int(len(cal_indices) * 0.8)
    l3_train_mask = np.zeros_like(cal_mask, dtype=bool)
    l3_train_mask[cal_indices[:split_idx]] = True
    l3_val_mask = np.zeros_like(cal_mask, dtype=bool)
    l3_val_mask[cal_indices[split_idx:]] = True

    X_train, y_train = X[l3_train_mask], y_target[l3_train_mask]
    X_val, y_val = X[l3_val_mask], y_target[l3_val_mask]
    X_test, y_test = X[test_mask], y_target[test_mask]

    y_tp_train, y_sl_train = y_tp_target[l3_train_mask], y_sl_target[l3_train_mask]
    y_tp_val, y_sl_val = y_tp_target[l3_val_mask], y_sl_target[l3_val_mask]
    y_tp_test, y_sl_test = y_tp_target[test_mask], y_sl_target[test_mask]

    y_gate_train = (np.abs(y_train) >= l3_flat_tau).astype(np.int32)
    y_gate_val = (np.abs(y_val) >= l3_flat_tau).astype(np.int32)
    pos_ct = int(y_gate_train.sum())
    neg_ct = int(len(y_gate_train) - pos_ct)
    spw = float(neg_ct / max(pos_ct, 1)) if pos_ct else 1.0

    w_size = np.where(np.abs(y_train) < l3_flat_tau, l3_flat_w, 1.0).astype(np.float64)
    w_size_val = np.where(np.abs(y_val) < l3_flat_tau, l3_flat_w, 1.0).astype(np.float64)
    y_gate_test = (np.abs(y_test) >= l3_flat_tau).astype(np.int32)

    print(
        f"  Features: {len(exec_feat_cols)} "
        f"(L2b triplet=3, regime_now={len(REGIME_NOW_PROB_COLS)}+conf, "
        f"tcn_fut={len(tcn_prob_cols)}, garch={len(garch_cols)}, pa_key={len(pa_key_cols)}, "
        f"opp×regime=3)",
    )
    print(
        f"  Train (cal, full rows): {len(y_train):,}  |  Valid/Test: {len(y_test):,}  "
        f"| flat weight={l3_flat_w}  τ={l3_flat_tau}",
    )
    print(
        f"  Target config — decision_h={opt_cfg['decision_horizon_bars']}  "
        f"theta_start={opt_cfg['theta_start_bars']}  theta_decay={opt_cfg['theta_decay_bars']:.1f}  "
        f"adv_penalty={opt_cfg['adverse_penalty']:.2f}"
    )
    print(
        f"  Active (|y|≥τ) — train: {(np.abs(y_train) >= l3_flat_tau).mean():.1%} | "
        f"test: {(np.abs(y_test) >= l3_flat_tau).mean():.1%}",
    )

    rounds = 1600 if FAST_TRAIN_MODE else 4000
    es_cb = _lgb_train_callbacks(90 if FAST_TRAIN_MODE else 120)

    gate_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 120,
        "lambda_l1": 0.15,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        "scale_pos_weight": spw,
    }
    d_gate_tr = lgb.Dataset(X_train, label=y_gate_train, feature_name=exec_feat_cols, free_raw_data=True)
    d_gate_va = lgb.Dataset(X_val, label=y_gate_val, feature_name=exec_feat_cols, free_raw_data=True)
    model_gate = lgb.train(
        gate_params,
        d_gate_tr,
        num_boost_round=rounds,
        valid_sets=[d_gate_va],
        callbacks=es_cb,
    )

    size_params = {
        "objective": "fair",
        "fair_c": 1.0,  # Fair loss parameter controlling transition from L2 to L1
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": 43,
        "n_jobs": _lgbm_n_jobs(),
    }
    d_sz_tr = lgb.Dataset(
        X_train, label=y_train, weight=w_size, feature_name=exec_feat_cols, free_raw_data=True,
    )
    d_sz_va = lgb.Dataset(X_val, label=y_val, weight=w_size_val, feature_name=exec_feat_cols, free_raw_data=True)
    model_size = lgb.train(
        size_params,
        d_sz_tr,
        num_boost_round=rounds,
        valid_sets=[d_sz_va],
        callbacks=es_cb,
    )

    pred_g = model_gate.predict(X_test)
    pred_s = model_size.predict(X_test)
    pred = np.clip(pred_g * pred_s * l3_pos_multiplier, -1.0, 1.0)

    mse = float(np.mean((pred - y_test) ** 2))
    nz = np.abs(y_test) >= l3_flat_tau
    sign_hit = float((np.sign(pred[nz]) == np.sign(y_test[nz])).mean()) if nz.sum() > 0 else float("nan")
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if len(pred) > 2 else float("nan")
    try:
        gate_auc = float(roc_auc_score(y_gate_test, pred_g)) if len(np.unique(y_gate_test)) > 1 else float("nan")
    except ValueError:
        gate_auc = float("nan")

    print("\n  Test metrics (combined = p_gate × size, clipped):")
    print(f"    MSE:         {mse:.5f}")
    print(f"    Gate AUC:    {gate_auc:.4f}  (|y|≥τ)")
    print(f"    Corr(y,p):   {corr:.4f}")
    print(f"    Sign hit:    {sign_hit:.4f}  (|target|≥τ)")
    print(f"    Mean |pos|:  {np.mean(np.abs(pred)):.3f}")

    imp_size = model_size.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": exec_feat_cols,
        "importance": imp_size,
    }).sort_values("importance", ascending=False)
    print("\n  Top 20 Layer-3 size-head features (gain):")
    print(imp_df.head(20).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_gate.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_GATE_FILE))
    model_size.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_SIZE_FILE))
    import pickle

    meta = {
        "l3_schema": 2,
        "type": "execution_sizer_two_stage",
        "feature_cols": exec_feat_cols,
        "position_clip": [-1.0, 1.0],
        "combine_rule": f"clip(p_gate * pred_size * {l3_pos_multiplier}, -1, 1)",
        "pos_multiplier": l3_pos_multiplier,
        "target_definition": "clip(class_blend * decision_edge_scale, -1, 1); decision_edge uses horizon-limited, theta-decayed net edge",
        "flat_tau": l3_flat_tau,
        "flat_sample_weight": l3_flat_w,
        "gate_metric": "auc",
        "size_objective": "huber",
        "decision_horizon_bars": opt_cfg["decision_horizon_bars"],
        "theta_start_bars": opt_cfg["theta_start_bars"],
        "theta_decay_bars": opt_cfg["theta_decay_bars"],
        "adverse_penalty": opt_cfg["adverse_penalty"],
        "uses_garch": bool(garch_cols),
        "garch_cols": garch_cols,
        "pa_key_cols": pa_key_cols,
        "tcn_prob_cols": tcn_prob_cols,
        "model_files": {
            "gate": EXECUTION_SIZER_GATE_FILE,
            "size": EXECUTION_SIZER_SIZE_FILE,
        },
    }
    with open(os.path.join(MODEL_DIR, "execution_sizer_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    bundle = {"gate": model_gate, "size": model_size, "meta": meta}
    print(
        f"\n  Models saved → {MODEL_DIR}/{EXECUTION_SIZER_GATE_FILE}, "
        f"{EXECUTION_SIZER_SIZE_FILE}"
    )
    print(f"  Meta saved  → {MODEL_DIR}/execution_sizer_meta.pkl")
    return bundle, meta, imp_df



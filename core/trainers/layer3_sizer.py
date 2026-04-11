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
    _attach_l2b_regime_context_features,
    _build_l2b_gate_meta_features,
    _build_l3_interaction_features,
    _compute_opportunity_triplet_with_regime_opp,
    _layer3_fill_trade_stack_probs_gates,
    _l2b_nested_opp_models,
    _compute_opportunity_triplet,
    _l2b_triplet_from_trade_prob,
    _apply_cp_skip,
    _build_l2b_directional_feature_matrix,
    _tcn_transition_risk_profile,
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
    opp_regime_out: np.ndarray | None,
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
            o, mf, ma, orp = _compute_opportunity_triplet_with_regime_opp(x_b, rp, models)
            opp_out[i:j] = o.astype(np.float32)
            mfe_out[i:j] = mf.astype(np.float32)
            mae_out[i:j] = ma.astype(np.float32)
            if opp_regime_out is not None:
                opp_regime_out[i:j] = orp.astype(np.float32)
        return
    o, mf, ma = _l2b_triplet_from_trade_prob(p_trade)
    opp_out[:] = o.astype(np.float32)
    mfe_out[:] = mf.astype(np.float32)
    mae_out[:] = ma.astype(np.float32)
    if opp_regime_out is not None:
        opp_regime_out[:] = np.repeat(o.reshape(-1, 1), len(REGIME_NOW_PROB_COLS), axis=1).astype(np.float32)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _l3_directional_ablation_settings() -> dict[str, Any]:
    drop_summary = [
        c.strip()
        for c in os.environ.get("L3_DROP_SUMMARY_COLS", "").split(",")
        if c.strip()
    ]
    return {
        "tag": os.environ.get("L3_ABLATION_TAG", "base").strip() or "base",
        "include_opp_x_regime": _env_flag("L3_INCLUDE_OPP_X_REGIME", True),
        "include_regime_probs": _env_flag("L3_INCLUDE_REGIME_PROBS", True),
        "include_opp_summary": _env_flag("L3_INCLUDE_OPP_SUMMARY", True),
        "include_per_regime_opp": _env_flag("L3_INCLUDE_PER_REGIME_OPP", True),
        "drop_summary_cols": drop_summary,
        "log_shap": _env_flag("L3_LOG_SHAP", False),
        "shap_sample": max(256, int(os.environ.get("L3_SHAP_SAMPLE", "4096"))),
    }


def _filter_l3_directional_block(
    blk: np.ndarray,
    cols: list[str],
    settings: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    keep = np.ones(len(cols), dtype=bool)
    drop_summary_cols = set(settings["drop_summary_cols"])
    for idx, col in enumerate(cols):
        if col in L2B_OPP_X_REGIME_COLS and not settings["include_opp_x_regime"]:
            keep[idx] = False
        elif col in REGIME_NOW_PROB_COLS and not settings["include_regime_probs"]:
            keep[idx] = False
        elif col in L2B_OPP_SUMMARY_COLS and not settings["include_opp_summary"]:
            keep[idx] = False
        elif col in L2B_PER_REGIME_OPP_COLS and not settings["include_per_regime_opp"]:
            keep[idx] = False
        elif col in drop_summary_cols:
            keep[idx] = False
    return blk[:, keep], [c for c, k in zip(cols, keep) if k]


def _log_l3_shap_diagnostics(
    model: lgb.Booster,
    X_val: np.ndarray,
    feature_names: list[str],
    settings: dict[str, Any],
) -> None:
    if not settings["log_shap"]:
        return
    try:
        import shap  # type: ignore
    except ImportError:
        print("  [L3 ablation] SHAP requested but package is not installed; skipping.", flush=True)
        return
    if len(X_val) == 0:
        return
    n = min(len(X_val), int(settings["shap_sample"]))
    X_s = X_val[:n]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_mat = np.asarray(shap_values, dtype=np.float64)
    mean_abs = np.mean(np.abs(shap_mat), axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    print("\n  [L3 ablation] Top 15 mean|SHAP| features:", flush=True)
    print(shap_df.head(15).to_string(index=False), flush=True)
    l2b_mask = [c.startswith("l2b_") for c in feature_names]
    l2b_cols = [c for c, ok in zip(feature_names, l2b_mask) if ok]
    if len(l2b_cols) >= 2:
        shap_l2b = pd.DataFrame(shap_mat[:, l2b_mask], columns=l2b_cols)
        corr = shap_l2b.corr().abs()
        pairs: list[tuple[str, str, float]] = []
        for i, left in enumerate(l2b_cols):
            for right in l2b_cols[i + 1:]:
                val = float(corr.loc[left, right])
                if np.isfinite(val) and val >= 0.90:
                    pairs.append((left, right, val))
        if pairs:
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:12]
            print("  [L3 ablation] High-correlation SHAP pairs (|corr|>=0.90):", flush=True)
            for left, right, val in pairs:
                print(f"    {left} ~ {right}: {val:.3f}", flush=True)


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

    ablation = _l3_directional_ablation_settings()
    l3_flat_tau = float(os.environ.get("L3_FLAT_TAU", "0.03"))
    l3_flat_w = float(os.environ.get("L3_FLAT_WEIGHT", "0.10"))
    l3_pos_multiplier = float(os.environ.get("L3_POS_MULTIPLIER", "3.0"))  # Base amplification before val tuning
    opt_cfg = _options_target_config()

    chunk = _layer3_chunk_rows()
    print(f"  Memory: chunked predicts (LAYER3_CHUNK={chunk}); shallow df, no full feature matrices")
    print(
        f"  L3 ablation tag={ablation['tag']}  include_opp_x_regime={ablation['include_opp_x_regime']}  "
        f"include_regime_probs={ablation['include_regime_probs']}  include_opp_summary={ablation['include_opp_summary']}  "
        f"include_per_regime_opp={ablation['include_per_regime_opp']}",
        flush=True,
    )
    if ablation["drop_summary_cols"]:
        print(f"  L3 ablation drop_summary_cols={', '.join(ablation['drop_summary_cols'])}", flush=True)

    work = ensure_structure_context_features(df.copy(deep=False))

    n = len(work)
    cal_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    _layer3_fill_regime_calibrated(
        regime_model, regime_calibrators, work, cal_regime, chunk,
    )
    _layer3_attach_regime_probs_to_work(work, cal_regime)
    _attach_l2b_regime_context_features(work, cal_regime)

    garch_cols = sorted([
        c for c in work.columns
        if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}
    ])
    layer2_feats = trade_quality_models["feature_cols"]

    p_long_gate = np.empty(n, dtype=np.float32)
    p_short_gate = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(
        trade_quality_models, work, layer2_feats, p_long_gate, p_short_gate, chunk,
    )
    tcn_transition_prob_all = work["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in work.columns else None
    p_long_gate, _ = _apply_cp_skip(cal_regime, p_long_gate, thr_cp, tcn_transition_prob_all, side="long")
    p_short_gate, _ = _apply_cp_skip(cal_regime, p_short_gate, thr_cp, tcn_transition_prob_all, side="short")
    p_long_gate, p_short_gate = _apply_structure_veto_to_gates(work, p_long_gate, p_short_gate)

    p_trade_max = np.maximum(p_long_gate, p_short_gate)

    l2b_opp = np.empty(n, dtype=np.float32)
    l2b_mfe = np.empty(n, dtype=np.float32)
    l2b_mae = np.empty(n, dtype=np.float32)
    l2b_opp_regime = np.empty((n, len(REGIME_NOW_PROB_COLS)), dtype=np.float32)
    _layer3_fill_l2b_triplet_arrays(
        trade_quality_models, work, layer2_feats, p_trade_max, l2b_opp, l2b_mfe, l2b_mae, l2b_opp_regime, chunk,
    )

    y_target = np.clip(_decision_edge_atr_array(work), -1.0, 1.0)

    opt_tp_atr, opt_sl_atr, _ = _optimal_exit_target_arrays(work)
    y_tp_target = np.clip(opt_tp_atr, 0.0, 6.0)
    y_sl_target = np.clip(opt_sl_atr, 0.0, 3.0)

    mamba_prob_cols = [c for c in MAMBA_REGIME_FUT_PROB_COLS if c in work.columns]
    pa_key_cols = [c for c in LAYER3_PA_KEY_FEATURES if c in work.columns][:24]
    tcn_feature_cols = [c for c in TCN_FEATURES_FOR_L2B if c in work.columns and c != TCN_TRANSITION_PROB_COL]

    l2b_dir_blk, l2b_dir_cols = _build_l2b_directional_feature_matrix(
        cal_regime, l2b_opp, l2b_mfe, l2b_mae, l2b_opp_regime, tcn_transition_prob_all
    )
    l2b_dir_blk, l2b_dir_cols = _filter_l3_directional_block(l2b_dir_blk, l2b_dir_cols, ablation)
    gate_meta_blk, gate_meta_cols = _build_l2b_gate_meta_features(p_long_gate, p_short_gate)
    interaction_blk, interaction_cols = _build_l3_interaction_features(
        l2b_opp, l2b_mfe, l2b_mae, p_long_gate, p_short_gate
    )

    tcn_mat = work[tcn_feature_cols].to_numpy(dtype=np.float32, copy=False) if tcn_feature_cols else np.empty((n, 0), np.float32)
    mamba_mat = work[mamba_prob_cols].to_numpy(dtype=np.float32, copy=False) if mamba_prob_cols else np.empty((n, 0), np.float32)
    pa_mat = work[pa_key_cols].to_numpy(dtype=np.float32, copy=False) if pa_key_cols else np.empty((n, 0), np.float32)
    if garch_cols:
        g_mat = work[garch_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        g_mat = np.empty((n, 0), dtype=np.float32)

    X = np.hstack([l2b_dir_blk, gate_meta_blk, interaction_blk, tcn_mat, mamba_mat, g_mat, pa_mat])
    exec_feat_cols = (
        l2b_dir_cols
        + gate_meta_cols
        + interaction_cols
        + tcn_feature_cols
        + mamba_prob_cols
        + garch_cols
        + pa_key_cols
    )
    _require_lgb_matrix_matches_names(X, exec_feat_cols, "Layer 3 (execution sizer v2)")

    del l2b_dir_blk, gate_meta_blk, interaction_blk, tcn_mat, mamba_mat, pa_mat, g_mat, cal_regime
    del p_trade_max, p_long_gate, p_short_gate, work
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

    y_gate_test = (np.abs(y_test) >= l3_flat_tau).astype(np.int32)

    print(
        f"  Features: {len(exec_feat_cols)} "
        f"(l2b_directional={len(l2b_dir_cols)}, l2b_gates={len(gate_meta_cols)}, "
        f"l3_interactions={len(interaction_cols)}, "
        f"tcn_risk={len(tcn_feature_cols) + int(tcn_transition_prob_all is not None)}, mamba={len(mamba_prob_cols)}, "
        f"garch={len(garch_cols)}, pa_key={len(pa_key_cols)})",
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
    size_train_mask = y_gate_train == 1
    size_val_mask = y_gate_val == 1
    if int(size_train_mask.sum()) < 200:
        raise RuntimeError("Layer 3 size head: too few active train rows after active-only filtering.")
    if int(size_val_mask.sum()) < 50:
        raise RuntimeError("Layer 3 size head: too few active validation rows after active-only filtering.")
    print(
        f"  Size head active-only rows — train={int(size_train_mask.sum()):,}  val={int(size_val_mask.sum()):,}",
        flush=True,
    )
    d_sz_tr = lgb.Dataset(
        X_train[size_train_mask], label=y_train[size_train_mask], feature_name=exec_feat_cols, free_raw_data=True,
    )
    d_sz_va = lgb.Dataset(
        X_val[size_val_mask], label=y_val[size_val_mask], feature_name=exec_feat_cols, free_raw_data=True,
    )
    model_size = lgb.train(
        size_params,
        d_sz_tr,
        num_boost_round=rounds,
        valid_sets=[d_sz_va],
        callbacks=es_cb,
    )

    def _risk_scale_from_tcn(tcn_arr: np.ndarray | None) -> np.ndarray | float:
        if tcn_arr is None:
            return 1.0
        risk_prof = _tcn_transition_risk_profile(tcn_arr.astype(np.float64, copy=False))
        return risk_prof["pos_scale"] if risk_prof is not None else 1.0

    def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) <= 2:
            return float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _pick_pos_multiplier(
        raw_base: np.ndarray,
        y_true: np.ndarray,
        base_multiplier: float,
    ) -> tuple[float, dict[str, float]]:
        raw = np.asarray(raw_base, dtype=np.float64)
        yt = np.asarray(y_true, dtype=np.float64)
        min_mult = float(os.environ.get("L3_POS_MULT_MIN", "1.0"))
        max_mult = float(os.environ.get("L3_POS_MULT_MAX", "12.0"))
        step = float(os.environ.get("L3_POS_MULT_STEP", "0.25"))
        corr_tol = float(os.environ.get("L3_POS_MULT_CORR_TOL", "0.002"))
        mult_grid = np.arange(min_mult, max_mult + 0.5 * step, step)
        best_mult = float(base_multiplier)
        best_row: tuple[float, float, float, float] | None = None
        rows: list[tuple[float, float, float, float]] = []
        nz = np.abs(yt) >= l3_flat_tau
        for mult in mult_grid:
            pred_c = np.clip(raw * mult, -1.0, 1.0)
            corr_c = _corr_safe(pred_c, yt)
            if not np.isfinite(corr_c):
                corr_c = -1.0
            sign_hit_c = float((np.sign(pred_c[nz]) == np.sign(yt[nz])).mean()) if nz.sum() > 0 else float("nan")
            mean_abs_c = float(np.mean(np.abs(pred_c)))
            mse_c = float(np.mean((pred_c - yt) ** 2))
            rows.append((float(mult), corr_c, mean_abs_c, mse_c))
            if best_row is None:
                best_row = rows[-1]
                best_mult = float(mult)
                continue
            _, best_corr, best_mean_abs, best_mse = best_row
            better_corr = corr_c > best_corr + corr_tol
            corr_close = abs(corr_c - best_corr) <= corr_tol
            better_mean_abs = mean_abs_c > best_mean_abs + 1e-6
            better_mse = mse_c < best_mse - 1e-6
            if better_corr or (corr_close and (better_mean_abs or better_mse)):
                best_row = rows[-1]
                best_mult = float(mult)
        assert best_row is not None
        _, best_corr, best_mean_abs, best_mse = best_row
        best_pred = np.clip(raw * best_mult, -1.0, 1.0)
        best_sign_hit = float((np.sign(best_pred[nz]) == np.sign(yt[nz])).mean()) if nz.sum() > 0 else float("nan")
        return best_mult, {
            "corr": float(best_corr),
            "mean_abs": float(best_mean_abs),
            "mse": float(best_mse),
            "sign_hit": float(best_sign_hit),
        }

    pred_g_val = model_gate.predict(X_val)
    pred_s_val = model_size.predict(X_val)
    pred_g = model_gate.predict(X_test)
    pred_s = model_size.predict(X_test)
    if tcn_transition_prob_all is not None:
        tcn_val = tcn_transition_prob_all[l3_val_mask].astype(np.float64, copy=False)
        tcn_test = tcn_transition_prob_all[test_mask].astype(np.float64, copy=False)
        risk_scale_val = _risk_scale_from_tcn(tcn_val)
        risk_scale = _risk_scale_from_tcn(tcn_test)
    else:
        risk_scale_val = 1.0
        risk_scale = 1.0
    raw_val = pred_g_val * pred_s_val * risk_scale_val
    tuned_pos_multiplier, tune_stats = _pick_pos_multiplier(raw_val, y_val, l3_pos_multiplier)
    print(
        f"  Validation-tuned pos_multiplier: base={l3_pos_multiplier:.2f} -> tuned={tuned_pos_multiplier:.2f}  "
        f"corr={tune_stats['corr']:.4f}  sign_hit={tune_stats['sign_hit']:.4f}  "
        f"mean|pos|={tune_stats['mean_abs']:.3f}  mse={tune_stats['mse']:.5f}"
    )

    pred = np.clip(pred_g * pred_s * tuned_pos_multiplier * risk_scale, -1.0, 1.0)

    mse = float(np.mean((pred - y_test) ** 2))
    nz = np.abs(y_test) >= l3_flat_tau
    sign_hit = float((np.sign(pred[nz]) == np.sign(y_test[nz])).mean()) if nz.sum() > 0 else float("nan")
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if len(pred) > 2 else float("nan")
    try:
        gate_auc = float(roc_auc_score(y_gate_test, pred_g)) if len(np.unique(y_gate_test)) > 1 else float("nan")
    except ValueError:
        gate_auc = float("nan")

    print("\n  Test metrics (combined = p_gate × size, clipped):")
    print(f"    Experiment:  {ablation['tag']}")
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
    _log_l3_shap_diagnostics(model_size, X_val, exec_feat_cols, ablation)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_gate.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_GATE_FILE))
    model_size.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_SIZE_FILE))
    import pickle

    meta = {
        "l3_schema": 3,
        "type": "execution_sizer_two_stage",
        "feature_cols": exec_feat_cols,
        "position_clip": [-1.0, 1.0],
        "combine_rule": f"clip(p_gate * pred_size * {tuned_pos_multiplier} * staged_tcn_pos_scale, -1, 1)",
        "pos_multiplier": float(tuned_pos_multiplier),
        "base_pos_multiplier": float(l3_pos_multiplier),
        "tcn_position_risk_coef": None,
        "target_definition": "clip(decision_edge_atr, -1, 1); gate learns act-vs-flat from |decision_edge| >= flat_tau and size learns signed edge directly",
        "flat_tau": l3_flat_tau,
        "flat_sample_weight": l3_flat_w,
        "size_training_scope": "active_only(|decision_edge|>=flat_tau)",
        "pos_multiplier_tuning": tune_stats,
        "ablation": {
            "tag": ablation["tag"],
            "include_opp_x_regime": ablation["include_opp_x_regime"],
            "include_regime_probs": ablation["include_regime_probs"],
            "include_opp_summary": ablation["include_opp_summary"],
            "include_per_regime_opp": ablation["include_per_regime_opp"],
            "drop_summary_cols": list(ablation["drop_summary_cols"]),
        },
        "gate_metric": "auc",
        "size_objective": "fair",
        "decision_horizon_bars": opt_cfg["decision_horizon_bars"],
        "theta_start_bars": opt_cfg["theta_start_bars"],
        "theta_decay_bars": opt_cfg["theta_decay_bars"],
        "adverse_penalty": opt_cfg["adverse_penalty"],
        "uses_garch": bool(garch_cols),
        "garch_cols": garch_cols,
        "pa_key_cols": pa_key_cols,
        "tcn_prob_cols": tcn_feature_cols,
        "tcn_feature_cols": tcn_feature_cols,
        "interaction_feature_cols": interaction_cols,
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



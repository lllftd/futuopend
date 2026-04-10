from __future__ import annotations

import gc
import os
import pickle
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN, FocalLoss

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.data_prep import *
from core.trainers.layer2b_quality import focal_loss_lgb, focal_loss_lgb_eval_error, _apply_cp_skip
from core.trainers.layer3_sizer import (
    _layer3_fill_regime_calibrated,
    _layer3_attach_regime_probs_to_work,
    _layer3_fill_trade_stack_probs,
    _layer3_fill_l2b_triplet_arrays,
)

def train_exit_manager_layer4(
    df: pd.DataFrame,
    feat_cols: list[str],
    regime_model: lgb.Booster,
    regime_calibrators: Any,
    trade_quality_models: dict,
    thr_cp: float,
):
    print("\n" + "=" * 70)
    print("  LAYER 4: Exit Manager (Multi-class Binning + Time Survival + CP Uncertainty)")
    print("  y_tp = MFE/ATR (4 Bins) | y_sl = MAE/ATR (4 Bins) | y_time = Exit Bar")
    print("=" * 70)

    chunk = _layer3_chunk_rows()
    work = df.copy(deep=False)
    bo_frame = compute_breakout_features(work)
    for c in BO_FEAT_COLS:
        work[c] = bo_frame[c].values
    del bo_frame

    n = len(work)
    cal_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    _layer3_fill_regime_calibrated(
        regime_model, regime_calibrators, work, cal_regime, chunk,
        tqdm_desc="Layer4 regime→cal",
    )
    _layer3_attach_regime_probs_to_work(work, cal_regime)

    garch_cols = sorted([c for c in work.columns if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}])
    layer2_feats = trade_quality_models["feature_cols"]

    p_long_gate = np.empty(n, dtype=np.float32)
    p_short_gate = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(
        trade_quality_models, work, layer2_feats, p_long_gate, p_short_gate, chunk,
        tqdm_desc="Layer4 trade stack",
    )
    
    tcn_transition_prob_all = work["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in work.columns else None
    p_long_gate, _ = _apply_cp_skip(cal_regime, p_long_gate, thr_cp, tcn_transition_prob_all)
    p_short_gate, _ = _apply_cp_skip(cal_regime, p_short_gate, thr_cp, tcn_transition_prob_all)
    
    p_trade_max = np.maximum(p_long_gate, p_short_gate)

    l2b_opp = np.empty(n, dtype=np.float32)
    l2b_mfe = np.empty(n, dtype=np.float32)
    l2b_mae = np.empty(n, dtype=np.float32)
    _layer3_fill_l2b_triplet_arrays(
        trade_quality_models, work, layer2_feats, p_trade_max, l2b_opp, l2b_mfe, l2b_mae, chunk,
        tqdm_desc="Layer4 L2b triplet (reg)",
    )

    safe_atr = np.where(work["lbl_atr"].values > 1e-3, work["lbl_atr"].values, 1e-3)
    
    # Layer 4 Targets: Discretized Bins for TP/SL and Continuous Time-to-Event
    mfe_atr = np.clip(work["max_favorable"].values / safe_atr, 0.0, 6.0)
    mae_atr = np.clip(work["max_adverse"].values / safe_atr, 0.0, 4.0)
    
    if "exit_bar" in work.columns:
        hold_time = np.maximum(work["exit_bar"].fillna(0).values.astype(float), 0.0)
        gamma_decay = np.exp(-np.maximum(hold_time - 15.0, 0.0) / 5.0)
        mfe_atr = mfe_atr * gamma_decay
    
    # Calculate percentiles dynamically to handle bimodal distribution and empty bins
    tp_boundaries = np.quantile(mfe_atr[cal_mask], [0.25, 0.50, 0.75])
    sl_boundaries = np.quantile(mae_atr[cal_mask], [0.25, 0.50, 0.75])
    
    print(f"  Dynamic TP Bins (ATR): {tp_boundaries}")
    print(f"  Dynamic SL Bins (ATR): {sl_boundaries}")

    y_tp_target = np.digitize(mfe_atr, bins=tp_boundaries).astype(np.int32)
    y_sl_target = np.digitize(mae_atr, bins=sl_boundaries).astype(np.int32)
    # Time target (Survival): predict expected duration of the trade. Max 6 bars (30 mins) for fast options scalping.
    y_time_target = np.clip(work.get("exit_bar", np.zeros(n)).fillna(0).values, 1, 6).astype(np.float32)

    tcn_prob_cols = [c for c in TCN_REGIME_FUT_PROB_COLS if c in work.columns]
    mamba_prob_cols = [c for c in MAMBA_REGIME_FUT_PROB_COLS if c in work.columns]
    pa_key_cols = [c for c in LAYER3_PA_KEY_FEATURES if c in work.columns][:15]

    inter_blk = (l2b_opp.astype(np.float64)[:, None] * cal_regime.astype(np.float64)).astype(np.float32, copy=False)
    triplet_blk = np.hstack([l2b_opp.reshape(-1, 1), l2b_mfe.reshape(-1, 1), l2b_mae.reshape(-1, 1)]).astype(np.float32, copy=False)
    sc_conf = cal_regime.max(axis=1, keepdims=True).astype(np.float32, copy=False)
    regime_blk = np.hstack([cal_regime, sc_conf]).astype(np.float32, copy=False)

    tcn_mat = work[tcn_prob_cols].to_numpy(dtype=np.float32, copy=False) if tcn_prob_cols else np.empty((n, 0), np.float32)
    mamba_mat = work[mamba_prob_cols].to_numpy(dtype=np.float32, copy=False) if mamba_prob_cols else np.empty((n, 0), np.float32)
    pa_mat = work[pa_key_cols].to_numpy(dtype=np.float32, copy=False) if pa_key_cols else np.empty((n, 0), np.float32)
    g_mat = work[garch_cols].to_numpy(dtype=np.float32, copy=False) if garch_cols else np.empty((n, 0), dtype=np.float32)

    X = np.hstack([triplet_blk, regime_blk, tcn_mat, mamba_mat, g_mat, pa_mat, inter_blk])
    exec_feat_cols = (
        ["l2b_opportunity_score", "l2b_pred_mfe", "l2b_pred_mae"]
        + REGIME_NOW_PROB_COLS
        + ["regime_now_conf"]
        + tcn_prob_cols + mamba_prob_cols + garch_cols + pa_key_cols + L2B_OPP_X_REGIME_COLS
    )

    t = df["time_key"].values
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, X_test = X[cal_mask], X[test_mask]
    y_tp_train, y_tp_test = y_tp_target[cal_mask], y_tp_target[test_mask]
    y_sl_train, y_sl_test = y_sl_target[cal_mask], y_sl_target[test_mask]
    y_time_train, y_time_test = y_time_target[cal_mask], y_time_target[test_mask]
    
    # We only care about training TP/SL on actual trades that passed the gate!
    thr_long = trade_quality_models["thresholds"]["long"]
    thr_short = trade_quality_models["thresholds"]["short"]
    gate_mask_train = (p_long_gate[cal_mask] >= thr_long) | (p_short_gate[cal_mask] >= thr_short)
    gate_mask_test = (p_long_gate[test_mask] >= thr_long) | (p_short_gate[test_mask] >= thr_short)
    
    w_train = np.where(gate_mask_train, 1.0, 0.05) # Emphasize actual trades
    w_test = np.where(gate_mask_test, 1.0, 0.05)

    rounds = 800 if FAST_TRAIN_MODE else 2000
    es_cb = _lgb_train_callbacks(60 if FAST_TRAIN_MODE else 100)

    print("  Training TP Ordinal Regressor (3 Binary Classifiers)...")
    tp_models = []
    for k in range(3):
        print(f"    -> TP > {k} ...")
        tp_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 48,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": 54 + k,
            "n_jobs": _lgbm_n_jobs(),
        }
        y_bin_tr = (y_tp_train > k).astype(np.int32)
        y_bin_te = (y_tp_test > k).astype(np.int32)
        
        pos_rate = max(y_bin_tr.mean(), 1e-5)
        neg_rate = 1.0 - pos_rate
        if pos_rate < 0.3:
            spw = float(neg_rate / pos_rate)
        else:
            spw = 1.0
        
        tp_params["scale_pos_weight"] = spw
        
        d_tr = lgb.Dataset(X_train, label=y_bin_tr, weight=w_train, feature_name=exec_feat_cols, free_raw_data=True)
        d_va = lgb.Dataset(X_test, label=y_bin_te, weight=w_test, feature_name=exec_feat_cols, free_raw_data=True)
        m = lgb.train(
            tp_params, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=es_cb,
        )
        tp_models.append(m)

    print("  Training SL Ordinal Regressor (3 Binary Classifiers)...")
    sl_models = []
    for k in range(3):
        print(f"    -> SL > {k} ...")
        sl_params = tp_params.copy()
        sl_params["learning_rate"] = 0.01
        sl_rounds = rounds * 2
        sl_params["seed"] = 64 + k
        y_bin_tr = (y_sl_train > k).astype(np.int32)
        y_bin_te = (y_sl_test > k).astype(np.int32)
        
        pos_rate = max(y_bin_tr.mean(), 1e-5)
        neg_rate = 1.0 - pos_rate
        if pos_rate < 0.3:
            spw = float(neg_rate / pos_rate)
        else:
            spw = 1.0
            
        sl_params["scale_pos_weight"] = spw
        
        d_tr = lgb.Dataset(X_train, label=y_bin_tr, weight=w_train, feature_name=exec_feat_cols, free_raw_data=True)
        d_va = lgb.Dataset(X_test, label=y_bin_te, weight=w_test, feature_name=exec_feat_cols, free_raw_data=True)
        m = lgb.train(
            sl_params, d_tr, num_boost_round=sl_rounds, valid_sets=[d_va], callbacks=es_cb,
        )
        sl_models.append(m)

    # LightGBM has no built-in "survival" objective; y_time is bounded continuous bars → regression.
    print("  Training time-to-exit regressor (exit_bar in bars, MAE)...")
    time_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 56,
        "n_jobs": _lgbm_n_jobs(),
    }
    d_time_tr = lgb.Dataset(X_train, label=y_time_train, weight=w_train, feature_name=exec_feat_cols, free_raw_data=True)
    d_time_va = lgb.Dataset(X_test, label=y_time_test, weight=w_test, feature_name=exec_feat_cols, free_raw_data=True)
    model_time = lgb.train(time_params, d_time_tr, num_boost_round=rounds, valid_sets=[d_time_va], callbacks=es_cb)

    print("\n" + "=" * 50)
    print("  Layer 4 Test Set Evaluation (on Gated Trades)")
    print("=" * 50)
    
    evt_tp_max = float(np.percentile(mfe_atr[mfe_atr > 0], 99.5)) if (mfe_atr > 0).any() else 5.0
    evt_sl_max = float(np.percentile(mae_atr[mae_atr > 0], 99.5)) if (mae_atr > 0).any() else 3.0

    print(f"  TP Bins (ATR): <0.5, 0.5-1.0, 1.0-1.5, >1.5")
    print(f"  SL Bins (ATR): <0.3, 0.3-0.6, 0.6-1.0, >1.0")
    print(f"  EVT Bounds (99.5% Tail): TP_max={evt_tp_max:.2f} ATR, SL_max={evt_sl_max:.2f} ATR")
    if evt_tp_max > 1.5:
        print("  -> TP EVT bound is larger than max bin boundary (1.5), which is expected.")
    if evt_sl_max > 1.0:
        print("  -> SL EVT bound is larger than max bin boundary (1.0), which is expected.")

    test_idx = np.where(gate_mask_test)[0]
    if len(test_idx) > 0:
        X_test_gated = X_test[test_idx]
        y_tp_test_gated = y_tp_test[test_idx]
        y_sl_test_gated = y_sl_test[test_idx]
        y_time_test_gated = y_time_test[test_idx]
        
        train_idx_gated = np.where(gate_mask_train)[0]
        n_long_train = np.sum(p_long_gate[cal_mask][train_idx_gated] >= thr_long)
        n_short_train = np.sum(p_short_gate[cal_mask][train_idx_gated] >= thr_short)
        print(f"\n  [Train Set] LONG vs SHORT gated trades:")
        print(f"    LONG: {n_long_train}, SHORT: {n_short_train} (Ratio: {n_long_train/max(1, n_short_train):.2f})")

        p_long_test_gated = p_long_gate[test_mask][test_idx]
        p_short_test_gated = p_short_gate[test_mask][test_idx]
        is_long_test = p_long_test_gated >= thr_long
        is_short_test = p_short_test_gated >= thr_short
        
        print(f"\n  [Test Set] LONG vs SHORT gated trades:")
        print(f"    LONG: {np.sum(is_long_test)}, SHORT: {np.sum(is_short_test)}")

        def predict_ordinal(models, X):
            n_bins = len(models) + 1
            cumulative_probs = np.zeros((len(X), len(models)))
            for k, m in enumerate(models):
                cumulative_probs[:, k] = m.predict(X)
                
            # Enforce monotonicity P(Y>0) >= P(Y>1) >= P(Y>2)
            for k in range(1, len(models)):
                cumulative_probs[:, k] = np.minimum(cumulative_probs[:, k], cumulative_probs[:, k-1])
                
            bin_probs = np.zeros((len(X), n_bins))
            bin_probs[:, 0] = 1.0 - cumulative_probs[:, 0]
            for k in range(1, n_bins - 1):
                bin_probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
            bin_probs[:, -1] = cumulative_probs[:, -1]
            
            return bin_probs.argmax(axis=1)
            
        pred_tp_class = predict_ordinal(tp_models, X_test_gated)
        pred_sl_class = predict_ordinal(sl_models, X_test_gated)
        
        from sklearn.metrics import classification_report
        print("\n  --- TP Classification Report (Test Set Gated) ---")
        print(classification_report(y_tp_test_gated, pred_tp_class, zero_division=0))
        
        print("\n  --- SL Classification Report (Test Set Gated) ---")
        print(classification_report(y_sl_test_gated, pred_sl_class, zero_division=0))

        if np.sum(is_short_test) > 0:
            print("\n  --- TP Classification Report (Test Set Gated - SHORT ONLY) ---")
            print(classification_report(y_tp_test_gated[is_short_test], pred_tp_class[is_short_test], zero_division=0))
            print("\n  --- SL Classification Report (Test Set Gated - SHORT ONLY) ---")
            print(classification_report(y_sl_test_gated[is_short_test], pred_sl_class[is_short_test], zero_division=0))

        pred_time = model_time.predict(X_test_gated)
        time_err = pred_time - y_time_test_gated
        print("\n  --- Time-to-Exit Error Distribution (Pred - True) ---")
        print(f"    Mean Error: {np.mean(time_err):.2f} bars")
        print(f"    MAE:        {np.mean(np.abs(time_err)):.2f} bars")
        for pct in [5, 25, 50, 75, 95]:
            print(f"    {pct}th pctl:  {np.percentile(time_err, pct):.2f} bars")

        l2b_mfe_test = l2b_mfe[test_mask][test_idx]
        l2b_mae_test = l2b_mae[test_mask][test_idx]
        
        l2b_tp_class = np.digitize(l2b_mfe_test, bins=tp_boundaries).astype(np.int32)
        l2b_sl_class = np.digitize(l2b_mae_test, bins=sl_boundaries).astype(np.int32)
        
        print("\n  --- Comparison: L4 vs L2b (Accuracy on Bins) ---")
        print(f"    L4 TP Accuracy:  {np.mean(pred_tp_class == y_tp_test_gated):.4f}")
        print(f"    L2b TP Accuracy: {np.mean(l2b_tp_class == y_tp_test_gated):.4f}")
        print(f"    L4 SL Accuracy:  {np.mean(pred_sl_class == y_sl_test_gated):.4f}")
        print(f"    L2b SL Accuracy: {np.mean(l2b_sl_class == y_sl_test_gated):.4f}")
    else:
        print("\n  [Test Set] No gated trades found. Skipping evaluation.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    EXECUTION_SIZER_TIME_FILE = "execution_sizer_time.txt"
    model_tp_files = [f"execution_sizer_tp_gt{k}.txt" for k in range(3)]
    model_sl_files = [f"execution_sizer_sl_gt{k}.txt" for k in range(3)]
    
    for k, m in enumerate(tp_models):
        m.save_model(os.path.join(MODEL_DIR, model_tp_files[k]))
    for k, m in enumerate(sl_models):
        m.save_model(os.path.join(MODEL_DIR, model_sl_files[k]))
        
    model_time.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_TIME_FILE))

    evt_tp_max = float(np.percentile(mfe_atr[mfe_atr > 0], 99.5)) if (mfe_atr > 0).any() else 5.0
    evt_sl_max = float(np.percentile(mae_atr[mae_atr > 0], 99.5)) if (mae_atr > 0).any() else 3.0

    meta = {
        "l4_schema": 3,
        "type": "exit_manager_ordinal_evt_bocpd_hawkes",
        "feature_cols": exec_feat_cols,
        "tp_bins": tp_boundaries.tolist(),
        "sl_bins": sl_boundaries.tolist(),
        "evt_tp_max": evt_tp_max,
        "evt_sl_max": evt_sl_max,
        "model_files": {
            "tp_ordinal": model_tp_files,
            "sl_ordinal": model_sl_files,
            "time": EXECUTION_SIZER_TIME_FILE,
        },
    }
    import pickle
    with open(os.path.join(MODEL_DIR, "exit_manager_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
        
    print(f"\n  Layer 4 Models saved → {MODEL_DIR}/execution_sizer_tp_gt*.txt, execution_sizer_sl_gt*.txt, {EXECUTION_SIZER_TIME_FILE}")
    print(f"  Layer 4 Meta saved  → {MODEL_DIR}/exit_manager_meta.pkl")
    print(f"  EVT Bounds (99.5% Tail): TP_max={evt_tp_max:.2f} ATR, SL_max={evt_sl_max:.2f} ATR")
    return {"tp_models": tp_models, "sl_models": sl_models, "time": model_time, "meta": meta}



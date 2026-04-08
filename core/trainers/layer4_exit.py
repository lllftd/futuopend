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
from core.trainers.layer2b_quality import _apply_cp_skip
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
    regime_calibrators: list,
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
    _layer3_fill_regime_calibrated(regime_model, regime_calibrators, work, cal_regime, chunk)
    _layer3_attach_regime_probs_to_work(work, cal_regime)

    garch_cols = sorted([c for c in work.columns if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}])
    layer2_feats = trade_quality_models["feature_cols"]

    p_trade = np.empty(n, dtype=np.float32)
    p_long = np.empty(n, dtype=np.float32)
    p_a = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(trade_quality_models, work, layer2_feats, p_trade, p_long, p_a, chunk)
    
    tcn_transition_prob_all = work["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in work.columns else None
    p_trade, _ = _apply_cp_skip(cal_regime, p_trade, thr_cp, tcn_transition_prob_all)

    l2b_opp = np.empty(n, dtype=np.float32)
    l2b_mfe = np.empty(n, dtype=np.float32)
    l2b_mae = np.empty(n, dtype=np.float32)
    _layer3_fill_l2b_triplet_arrays(trade_quality_models, work, layer2_feats, p_trade, l2b_opp, l2b_mfe, l2b_mae, chunk)

    safe_atr = np.where(work["lbl_atr"].values > 1e-3, work["lbl_atr"].values, 1e-3)
    
    # Layer 4 Targets: Discretized Bins for TP/SL and Continuous Time-to-Event
    mfe_atr = np.clip(work["max_favorable"].values / safe_atr, 0.0, 6.0)
    mae_atr = np.clip(work["max_adverse"].values / safe_atr, 0.0, 4.0)
    
    # Bins: 0: <0.5, 1: 0.5-1.2, 2: 1.2-2.5, 3: >2.5
    y_tp_target = np.digitize(mfe_atr, bins=[0.5, 1.2, 2.5]).astype(np.int32)
    # Bins: 0: <0.5, 1: 0.5-1.0, 2: 1.0-1.5, 3: >1.5
    y_sl_target = np.digitize(mae_atr, bins=[0.5, 1.0, 1.5]).astype(np.int32)
    # Time target (Survival): predict expected duration of the trade
    y_time_target = np.clip(work.get("exit_bar", np.zeros(n)).fillna(0).values, 1, 30).astype(np.float32)

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
    thr_trade = trade_quality_models["thresholds"]["trade"]
    gate_mask_train = p_trade[cal_mask] >= thr_trade
    gate_mask_test = p_trade[test_mask] >= thr_trade
    
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
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 5,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": 54 + k,
            "n_jobs": _lgbm_n_jobs(),
        }
        y_bin_tr = (y_tp_train > k).astype(np.int32)
        y_bin_te = (y_tp_test > k).astype(np.int32)
        
        pos_w = len(y_bin_tr) / max(y_bin_tr.sum(), 1)
        w_tr_adj = w_train * np.where(y_bin_tr == 1, pos_w, 1.0)
        
        d_tr = lgb.Dataset(X_train, label=y_bin_tr, weight=w_tr_adj, feature_name=exec_feat_cols, free_raw_data=True)
        d_va = lgb.Dataset(X_test, label=y_bin_te, weight=w_test, feature_name=exec_feat_cols, free_raw_data=True)
        m = lgb.train(tp_params, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=es_cb)
        tp_models.append(m)

    print("  Training SL Ordinal Regressor (3 Binary Classifiers)...")
    sl_models = []
    for k in range(3):
        print(f"    -> SL > {k} ...")
        sl_params = tp_params.copy()
        sl_params["seed"] = 64 + k
        y_bin_tr = (y_sl_train > k).astype(np.int32)
        y_bin_te = (y_sl_test > k).astype(np.int32)
        
        pos_w = len(y_bin_tr) / max(y_bin_tr.sum(), 1)
        w_tr_adj = w_train * np.where(y_bin_tr == 1, pos_w, 1.0)
        
        d_tr = lgb.Dataset(X_train, label=y_bin_tr, weight=w_tr_adj, feature_name=exec_feat_cols, free_raw_data=True)
        d_va = lgb.Dataset(X_test, label=y_bin_te, weight=w_test, feature_name=exec_feat_cols, free_raw_data=True)
        m = lgb.train(sl_params, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=es_cb)
        sl_models.append(m)

    print("  Training Time Survival Proxy (Poisson Regression on Holding Bars)...")
    time_params = {
        "objective": "poisson",
        "metric": "rmse",
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
        "tp_bins": [0.5, 1.2, 2.5],
        "sl_bins": [0.5, 1.0, 1.5],
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



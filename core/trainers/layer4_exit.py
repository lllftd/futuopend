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

def _symbol_segment_end_indices(symbols: np.ndarray) -> np.ndarray:
    ends = np.empty(len(symbols), dtype=np.int32)
    start = 0
    for i in range(1, len(symbols) + 1):
        if i == len(symbols) or symbols[i] != symbols[start]:
            ends[start:i] = i - 1
            start = i
    return ends


def _build_exit_policy_dataset(
    work: pd.DataFrame,
    base_X: np.ndarray,
    gate_mask: np.ndarray,
    side_arr: np.ndarray,
    feature_cols: list[str],
    *,
    exit_epsilon_atr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = _options_target_config()
    n = len(work)
    if n == 0:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
        )

    symbols = work["symbol"].values
    sym_end = _symbol_segment_end_indices(symbols)
    open_px = work["open"].values.astype(np.float64)
    high_px = work["high"].values.astype(np.float64)
    low_px = work["low"].values.astype(np.float64)
    close_px = work["close"].values.astype(np.float64)
    safe_atr = np.where(work["lbl_atr"].values > 1e-3, work["lbl_atr"].values.astype(np.float64), 1e-3)
    opt_exit = (
        work["optimal_exit_bar"].fillna(0).values.astype(np.int32)
        if "optimal_exit_bar" in work.columns
        else _optimal_exit_target_arrays(work)[2].astype(np.int32)
    )
    if "optimal_net_edge_atr" in work.columns:
        opt_net = work["optimal_net_edge_atr"].fillna(0.0).values.astype(np.float64)
    else:
        tp, sl, ex = _optimal_exit_target_arrays(work)
        opt_net = _net_edge_atr_from_state(tp, sl, ex).astype(np.float64)

    times = work["time_key"].values
    max_hold = int(cfg["max_hold_bars"])
    rows_x: list[np.ndarray] = []
    rows_exit: list[int] = []
    rows_value: list[float] = []
    rows_time: list[np.datetime64] = []

    candidate_entries = np.where(gate_mask & (side_arr != 0) & (opt_exit > 0))[0]
    for i in candidate_entries:
        if i + 1 >= n or symbols[i + 1] != symbols[i]:
            continue
        entry_price = float(open_px[i + 1])
        entry_atr = float(safe_atr[i])
        side = float(side_arr[i])
        live_mfe = 0.0
        live_mae = 0.0
        max_j = min(int(sym_end[i]), i + max_hold)
        opt_bar = max(1, int(opt_exit[i]))
        opt_edge = float(opt_net[i])
        for j in range(i + 1, max_j + 1):
            hold = j - i
            if side > 0:
                fav = max(0.0, (high_px[j] - entry_price) / entry_atr)
                adv = max(0.0, (entry_price - low_px[j]) / entry_atr)
                unreal = (close_px[j] - entry_price) / entry_atr
            else:
                fav = max(0.0, (entry_price - low_px[j]) / entry_atr)
                adv = max(0.0, (high_px[j] - entry_price) / entry_atr)
                unreal = (entry_price - close_px[j]) / entry_atr
            live_mfe = max(live_mfe, fav)
            live_mae = max(live_mae, adv)
            live_edge = float(_net_edge_atr_from_state(live_mfe, live_mae, hold))
            future_gain_left = float(opt_edge - live_edge)
            y_exit = 1 if (hold >= opt_bar or future_gain_left <= exit_epsilon_atr) else 0
            rows_x.append(
                _layer4_policy_state_vector(
                    base_X[j],
                    hold_bars=float(hold),
                    max_hold_bars=float(max_hold),
                    side=side,
                    unreal_pnl_atr=float(unreal),
                    mfe_atr_live=float(live_mfe),
                    mae_atr_live=float(live_mae),
                )
            )
            rows_exit.append(y_exit)
            rows_value.append(future_gain_left)
            rows_time.append(times[j])
            if hold >= opt_bar:
                break

    if not rows_x:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
        )
    return (
        np.asarray(rows_x, dtype=np.float32),
        np.asarray(rows_exit, dtype=np.int32),
        np.asarray(rows_value, dtype=np.float32),
        np.asarray(rows_time),
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
    print("  LAYER 4: Exit Policy (bar-by-bar hold vs exit)")
    print("  y_exit = whether future edge left is exhausted at this bar")
    print("=" * 70)

    chunk = _layer3_chunk_rows()
    opt_cfg = _options_target_config()
    exit_eps = float(os.environ.get("L4_EXIT_EPS_ATR", "0.03"))
    exit_prob_thr = float(os.environ.get("L4_EXIT_PROB_THRESHOLD", "0.55"))
    value_thr = float(os.environ.get("L4_VALUE_LEFT_THRESHOLD", "0.02"))
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

    base_X = np.hstack([triplet_blk, regime_blk, tcn_mat, mamba_mat, g_mat, pa_mat, inter_blk])
    base_feature_cols = (
        ["l2b_opportunity_score", "l2b_pred_mfe", "l2b_pred_mae"]
        + REGIME_NOW_PROB_COLS
        + ["regime_now_conf"]
        + tcn_prob_cols + mamba_prob_cols + garch_cols + pa_key_cols + L2B_OPP_X_REGIME_COLS
    )
    policy_feature_cols = _layer4_policy_feature_names(base_feature_cols)
    side_arr = np.where(p_long_gate >= p_short_gate, 1.0, -1.0).astype(np.float32)
    thr_long = trade_quality_models["thresholds"]["long"]
    thr_short = trade_quality_models["thresholds"]["short"]
    gate_mask = (p_long_gate >= thr_long) | (p_short_gate >= thr_short)

    X_policy, y_exit, y_value, t_state = _build_exit_policy_dataset(
        work,
        base_X,
        gate_mask,
        side_arr,
        policy_feature_cols,
        exit_epsilon_atr=exit_eps,
    )
    if len(X_policy) == 0:
        raise RuntimeError("Layer 4 policy dataset is empty. Check gating thresholds and labeled optimal exit columns.")

    cal_mask = (t_state >= np.datetime64(TRAIN_END)) & (t_state < np.datetime64(CAL_END))
    test_mask = (t_state >= np.datetime64(CAL_END)) & (t_state < np.datetime64(TEST_END))
    cal_indices = np.where(cal_mask)[0]
    split_idx = int(len(cal_indices) * 0.8)
    train_mask = np.zeros(len(X_policy), dtype=bool)
    val_mask = np.zeros(len(X_policy), dtype=bool)
    train_mask[cal_indices[:split_idx]] = True
    val_mask[cal_indices[split_idx:]] = True

    X_train, X_val, X_test = X_policy[train_mask], X_policy[val_mask], X_policy[test_mask]
    y_exit_train, y_exit_val, y_exit_test = y_exit[train_mask], y_exit[val_mask], y_exit[test_mask]
    y_value_train = np.clip(y_value[train_mask], -3.0, 3.0)
    y_value_val = np.clip(y_value[val_mask], -3.0, 3.0)
    y_value_test = np.clip(y_value[test_mask], -3.0, 3.0)

    pos_ct = int(y_exit_train.sum())
    neg_ct = int(len(y_exit_train) - pos_ct)
    spw = float(neg_ct / max(pos_ct, 1)) if pos_ct else 1.0
    w_value = np.where(y_exit_train == 1, 1.15, 1.0).astype(np.float64)
    w_value_val = np.where(y_exit_val == 1, 1.15, 1.0).astype(np.float64)

    print(
        f"  Policy dataset — train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} "
        f"exit_rate(train)={y_exit_train.mean():.1%}"
    )
    print(
        f"  Target config — max_hold={opt_cfg['max_hold_bars']}  theta_start={opt_cfg['theta_start_bars']}  "
        f"theta_decay={opt_cfg['theta_decay_bars']:.1f}  adv_penalty={opt_cfg['adverse_penalty']:.2f}  "
        f"exit_eps={exit_eps:.3f}"
    )

    rounds = 1200 if FAST_TRAIN_MODE else 3000
    es_cb = _lgb_train_callbacks(80 if FAST_TRAIN_MODE else 120)
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.1,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "seed": 71,
        "n_jobs": _lgbm_n_jobs(),
        "scale_pos_weight": spw,
    }
    value_params = {
        "objective": "fair",
        "fair_c": 1.0,
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 90,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": 72,
        "n_jobs": _lgbm_n_jobs(),
    }

    model_exit = lgb.train(
        exit_params,
        lgb.Dataset(X_train, label=y_exit_train, feature_name=policy_feature_cols, free_raw_data=True),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_val, label=y_exit_val, feature_name=policy_feature_cols, free_raw_data=True)],
        callbacks=es_cb,
    )
    model_value = lgb.train(
        value_params,
        lgb.Dataset(X_train, label=y_value_train, weight=w_value, feature_name=policy_feature_cols, free_raw_data=True),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_val, label=y_value_val, weight=w_value_val, feature_name=policy_feature_cols, free_raw_data=True)],
        callbacks=es_cb,
    )

    pred_exit = model_exit.predict(X_test) if len(X_test) else np.empty(0, dtype=np.float64)
    pred_value = model_value.predict(X_test) if len(X_test) else np.empty(0, dtype=np.float64)
    if len(X_test):
        try:
            auc = float(roc_auc_score(y_exit_test, pred_exit)) if len(np.unique(y_exit_test)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        exit_hit = float(((pred_exit >= exit_prob_thr).astype(np.int32) == y_exit_test).mean())
        mae = float(np.mean(np.abs(pred_value - y_value_test)))
        corr = float(np.corrcoef(pred_value, y_value_test)[0, 1]) if len(pred_value) > 2 else float("nan")
        print("\n  Layer 4 Test Metrics")
        print(f"    Exit AUC:      {auc:.4f}")
        print(f"    Exit Acc@thr:  {exit_hit:.4f}  (thr={exit_prob_thr:.2f})")
        print(f"    Value MAE:     {mae:.4f}")
        print(f"    Value Corr:    {corr:.4f}")
    else:
        print("\n  Layer 4 Test Metrics skipped: no test states.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    exit_file = "exit_policy_exit.txt"
    value_file = "exit_policy_value.txt"
    model_exit.save_model(os.path.join(MODEL_DIR, exit_file))
    model_value.save_model(os.path.join(MODEL_DIR, value_file))

    meta = {
        "l4_schema": 4,
        "type": "exit_policy_barwise",
        "feature_cols": policy_feature_cols,
        "base_feature_cols": base_feature_cols,
        "dynamic_feature_cols": L4_POLICY_DYNAMIC_FEATURES,
        "decision_horizon_bars": opt_cfg["decision_horizon_bars"],
        "theta_start_bars": opt_cfg["theta_start_bars"],
        "theta_decay_bars": opt_cfg["theta_decay_bars"],
        "max_hold_bars": opt_cfg["max_hold_bars"],
        "adverse_penalty": opt_cfg["adverse_penalty"],
        "exit_epsilon_atr": exit_eps,
        "exit_prob_threshold": exit_prob_thr,
        "value_left_threshold": value_thr,
        "model_files": {
            "exit": exit_file,
            "value": value_file,
        },
    }
    with open(os.path.join(MODEL_DIR, "exit_manager_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  Layer 4 Models saved → {MODEL_DIR}/{exit_file}, {value_file}")
    print(f"  Layer 4 Meta saved  → {MODEL_DIR}/exit_manager_meta.pkl")
    return {"exit": model_exit, "value": model_value, "meta": meta}



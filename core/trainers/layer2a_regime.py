from __future__ import annotations

import gc
import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    classification_report,
    roc_auc_score,
)

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.data_prep import *


def train_regime_classifier(df: pd.DataFrame, feat_cols: list[str]):
    print("\n" + "=" * 70)
    print("  LAYER 2a: 6-Class Regime Head (y = current market_state, no forward shift)")
    print(
        "  Predict: Bull/Bear/Range × Conv/Div | X = PA + GARCH only "
        "(no TCN; no pa_hmm_* — y matches HMM argmax, those cols would leak the label)"
    )
    print("=" * 70)

    raw_regime_feats = _regime_lgbm_feature_cols(feat_cols)
    pa_only_feats = _numeric_feature_cols_for_matrix(df, raw_regime_feats)
    if not pa_only_feats:
        raise ValueError("Layer 2a: no numeric PA features left after filtering string/tag cols.")

    X = df[pa_only_feats].values.astype(np.float32)
    y = df["state_label"].values.astype(int)
    t = df["time_key"].values

    train_mask = t < np.datetime64(TRAIN_END)
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    w_train = _compute_sample_weights(y_train, t[train_mask])

    print(f"  Dates — Train: < {TRAIN_END} | Cal: → {CAL_END} | Test: → {TEST_END}")

    base_params = {
        "objective": "multiclass",
        "num_class": NUM_REGIME_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.3,
        "lambda_l2": 2.0,
        "min_gain_to_split": 0.01,
        "path_smooth": 1.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
    }

    print(f"  Train: {len(y_train):,}  |  Cal: {len(y_cal):,}  |  Test: {len(y_test):,}")

    best_combo = _optuna_search_params(
        X_train, y_train, w_train, X_cal, y_cal, pa_only_feats, base_params, n_trials=30
    )
    params = {**base_params, **best_combo}

    print(f"\n  Training final model with best params: lr={params['learning_rate']:.4f}, colsample={params['feature_fraction']:.4f}…")
    _require_lgb_matrix_matches_names(X_train, pa_only_feats, "Layer 2a final train")
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,
                             feature_name=pa_only_feats, free_raw_data=False)
    # Split cal into early_stopping and calibration to prevent leaking labels to Layer 3
    split_idx = int(len(X_cal) * 0.5)
    X_cal_es, y_cal_es = X_cal[:split_idx], y_cal[:split_idx]
    X_cal_iso, y_cal_iso = X_cal[split_idx:], y_cal[split_idx:]

    valid_data = lgb.Dataset(X_cal_es, label=y_cal_es,
                             feature_name=pa_only_feats, free_raw_data=False)

    # Imbalanced 1-min labels: allow convergence (2000 rounds often still improving).
    state_rounds = 5000 if FAST_TRAIN_MODE else 6000
    state_es = 150
    model = lgb.train(
        params,
        train_data,
        num_boost_round=state_rounds,
        valid_sets=[valid_data],
        callbacks=_lgb_train_callbacks(state_es),
    )

    # ── Dirichlet Calibration on the calibration set ──
    # Optimizes logloss and joint distribution, replacing per-class Isotonic which breaks normalization.
    # Use the second half of the calibration set (X_cal_iso) which wasn't used for early stopping
    raw_cal_probs = model.predict(X_cal_iso)   # (n_cal_iso, NUM_REGIME_CLASSES)
    
    eps = 1e-7
    log_probs = np.log(np.clip(raw_cal_probs, eps, 1 - eps))
    calibrator = LogisticRegression(multi_class='multinomial', max_iter=2000, C=1.0)
    calibrator.fit(log_probs, y_cal_iso)

    def predict_calibrated(X_in):
        raw = model.predict(X_in)
        l_p = np.log(np.clip(raw, eps, 1 - eps))
        return calibrator.predict_proba(l_p)

    # ── Conformal Prediction Calibration (Score method) ──
    probs_cal = predict_calibrated(X_cal)
    p_true = probs_cal[np.arange(len(y_cal)), y_cal]
    scores = 1.0 - p_true
    alpha = 0.05
    n_cal = len(y_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    Q_hat = np.quantile(scores, q_level, method="higher")
    thr_cp = 1.0 - Q_hat
    
    print(f"\n  Conformal Prediction Calibration (alpha={alpha}):")
    print(f"    1-alpha Quantile Q_hat: {Q_hat:.4f}")
    print(f"    Probability Threshold (thr_cp):  {thr_cp:.4f}")

    # ── Evaluate on held-out test set ──
    probs = predict_calibrated(X_test)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, probs, labels=list(range(NUM_REGIME_CLASSES)))
    target_names = [STATE_NAMES[i] for i in range(NUM_REGIME_CLASSES)]

    print(f"\n  Accuracy (all): {acc:.4f}  |  Log-loss: {ll:.4f}")
    print(f"\n  Classification Report (all bars):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(NUM_REGIME_CLASSES)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_REGIME_CLASSES)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())

    # ── CP analysis on test set ──
    y_set_test = probs >= thr_cp
    set_size = y_set_test.sum(axis=1)
    contains_bull = y_set_test[:, 0] | y_set_test[:, 1]
    contains_bear = y_set_test[:, 2] | y_set_test[:, 3]
    is_conflicting = contains_bull & contains_bear
    skip_cp = (set_size >= 3) | is_conflicting | (set_size == 0)
    print(f"\n  Test set CP SKIP rate:  {skip_cp.mean():.2%} ({skip_cp.sum():,} bars)")
    print(f"  Test set_size distr:    {dict(zip(*np.unique(set_size, return_counts=True)))}")

    # ── Confidence-filtered accuracy ──
    print(f"\n  Confidence-Filtered Accuracy:")
    print(f"  {'Threshold':>10s}  {'Acc':>7s}  {'Coverage':>9s}  {'Bars':>9s}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = confidence >= thr
        if mask.sum() > 0:
            acc_f = accuracy_score(y_test[mask], y_pred[mask])
            cov = mask.mean()
            print(f"  {thr:>10.2f}  {acc_f:>7.4f}  {cov:>9.1%}  {mask.sum():>9,}")

    # ── Per-class probability calibration ──
    print(f"\n  Per-Class Probability Calibration (10 bins):")
    for c in range(NUM_REGIME_CLASSES):
        y_bin = (y_test == c).astype(int)
        p_c = probs[:, c]
        frac_pos, mean_pred = calibration_curve(y_bin, p_c, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_bin, p_c)
        print(f"\n    {STATE_NAMES[c]} (Brier={brier:.4f}):")
        print(f"    {'Predicted':>10s}  {'Actual':>8s}")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    {mp:>10.3f}  {fp:>8.3f}")

    # ── Feature importance ──
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": pa_only_feats, "importance": importance}).sort_values("importance", ascending=False)
    print(f"\n  Top 25 features (gain):")
    print(imp_df.head(25).to_string(index=False))

    # ── Save ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, STATE_CLASSIFIER_FILE))
    import pickle
    with open(os.path.join(MODEL_DIR, "state_calibrators.pkl"), "wb") as f:
        pickle.dump({
            "calibrators": calibrator,
            "thr_cp": float(thr_cp)
        }, f)
    print(f"\n  Model saved → {MODEL_DIR}/{STATE_CLASSIFIER_FILE}")
    print(f"  Calibrators saved → {MODEL_DIR}/state_calibrators.pkl")

    return model, calibrator, imp_df, float(thr_cp)



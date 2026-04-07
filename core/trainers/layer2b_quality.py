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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN, FocalLoss

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.data_prep import *

def _train_regime_opp_regression_models(
    X: np.ndarray,
    state_label: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    train_mask: np.ndarray,
    cal_mask: np.ndarray,
    all_bo_feats: list[str],
    regime_cal_probs_cal: np.ndarray,
    y_trade_cal: np.ndarray,
) -> tuple[dict[str, dict[str, lgb.Booster]], dict[str, float], np.ndarray]:
    """Train up to 6× (MFE + MAE) regressors (one pair per REGIMES_6 name); tune thresholds on cal.

    **Training rows** for regime ``REGIMES_6[k]``: ground-truth ``state_label == k`` (supervision).

    **Cal / inference routing**: pick head with ``argmax(regime_now probabilities) == k``; then
    ``score = pred_mfe / (pred_mae + 0.1)`` on those rows only; F1 grid-search yields ``thr_vec[k]``.

    Six regimes ⇒ fewer routed cal rows per bucket than old 3-way groups — default min row
    count is lower (``L2B_OPP_CAL_MIN_ROWS``).
    """
    reg_rounds = 800 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ROUNDS", "2000"))
    reg_es = 80 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ES", "150"))
    min_prec = float(os.environ.get("L2B_OPP_MIN_PRECISION", "0.30"))
    min_cal_for_opp_thr = int(os.environ.get("L2B_OPP_CAL_MIN_ROWS", "150"))
    base_reg = {
        "boosting_type": "gbdt",
        "learning_rate": float(os.environ.get("L2B_REG_LR", "0.01")),
        "num_leaves": int(os.environ.get("L2B_REG_NUM_LEAVES", "63")),
        "max_depth": int(os.environ.get("L2B_REG_MAX_DEPTH", "7")),
        "min_child_samples": int(os.environ.get("L2B_REG_MIN_CHILD", "100")),
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        **_l2b_reg_objective_params(),
    }
    cb = _lgb_train_callbacks(reg_es)
    models: dict[str, dict[str, lgb.Booster]] = {}

    X_tr = X[train_mask]
    X_ca = X[cal_mask]
    st_tr = state_label[train_mask]
    st_ca = state_label[cal_mask]
    mfe_tr = mfe[train_mask]
    mae_tr = mae[train_mask]
    mfe_ca = mfe[cal_mask]
    mae_ca = mae[cal_mask]

    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        mtr = st_tr == argmax_idx
        mca = st_ca == argmax_idx
        ntr, nca = int(mtr.sum()), int(mca.sum())
        print(f"  [L2b regression] {predicted_regime}: train_rows={ntr:,}  cal_rows(early_stop)={nca:,}")
        if ntr < 2000:
            print(f"    [warn] {predicted_regime}: too few train rows — skipping pair (needs ≥2000).")
            continue
        idx_sub = np.where(mtr)[0]
        X_g = X_tr[idx_sub]
        y_mfe_g = mfe_tr[idx_sub]
        y_mae_g = mae_tr[idx_sub]
        w_base = _opp_regression_sample_weights(y_mfe_g, predicted_regime)

        if nca >= 200:
            idx_va = np.where(mca)[0]
            X_va = X_ca[idx_va]
            y_mfe_va = mfe_ca[idx_va]
            y_mae_va = mae_ca[idx_va]
        else:
            tail = min(5000, ntr)
            X_va = X_g[-tail:]
            y_mfe_va = y_mfe_g[-tail:]
            y_mae_va = y_mae_g[-tail:]

        d_mfe_tr = lgb.Dataset(X_g, label=y_mfe_g, weight=w_base, feature_name=all_bo_feats, free_raw_data=False)
        d_mfe_va = lgb.Dataset(X_va, label=y_mfe_va, feature_name=all_bo_feats, free_raw_data=False)
        print(f"    [L2b regression] {predicted_regime}: train MFE head …", flush=True)
        m_mfe = lgb.train(
            base_reg, d_mfe_tr, num_boost_round=reg_rounds, valid_sets=[d_mfe_va], callbacks=cb,
        )
        w_mae = _opp_regression_sample_weights(y_mae_g, predicted_regime)
        d_mae_tr = lgb.Dataset(X_g, label=y_mae_g, weight=w_mae, feature_name=all_bo_feats, free_raw_data=False)
        d_mae_va = lgb.Dataset(X_va, label=y_mae_va, feature_name=all_bo_feats, free_raw_data=False)
        print(f"    [L2b regression] {predicted_regime}: train MAE head …", flush=True)
        m_mae = lgb.train(
            base_reg, d_mae_tr, num_boost_round=reg_rounds, valid_sets=[d_mae_va], callbacks=cb,
        )
        models[predicted_regime] = {"mfe": m_mfe, "mae": m_mae}

    if not models:
        raise RuntimeError("Regime opportunity regression: no group had enough data.")

    # L2a routing: argmax on cal probs → class index k == predicted_regime REGIMES_6[k]
    st_cal_pred = np.argmax(regime_cal_probs_cal, axis=1)
    gix_cal = st_cal_pred.astype(np.int64, copy=False)
    n_cal = len(y_trade_cal)
    opp_cal = np.zeros(n_cal, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = gix_cal == argmax_idx
        if not m.any():
            continue
        mfe_p = models[predicted_regime]["mfe"].predict(X_ca[m])
        mae_p = models[predicted_regime]["mae"].predict(X_ca[m])
        mfe_p = np.clip(mfe_p, 0.0, None)
        mae_p = np.clip(mae_p, 0.01, None)
        opp_cal[m] = np.log1p(mfe_p) - np.log1p(mae_p)

    opp_nonzero = opp_cal[opp_cal != 0]
    if len(opp_nonzero) > 0:
        print(f"  [L2b regression] opportunity(log-ratio) dist (cal): "
              f"min={np.min(opp_nonzero):.3f} 25%={np.percentile(opp_nonzero, 25):.3f} "
              f"median={np.median(opp_nonzero):.3f} 75%={np.percentile(opp_nonzero, 75):.3f} "
              f"max={np.max(opp_nonzero):.3f}")

    opp_thr: dict[str, float] = {}
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        m = gix_cal == argmax_idx
        n_routed = int(m.sum())
        if predicted_regime not in models or n_routed < min_cal_for_opp_thr:
            opp_thr[predicted_regime] = float(os.environ.get(f"L2B_OPP_THR_{predicted_regime.upper()}", "0.0"))
            why = "no model" if predicted_regime not in models else f"routed cal n={n_routed} < {min_cal_for_opp_thr}"
            print(
                f"  [L2b regression] {predicted_regime}: fallback opp_thr={opp_thr[predicted_regime]:.2f} ({why})",
            )
            continue
        y_true = y_trade_cal[m]
        o = opp_cal[m]
        best_f1, best_thr, best_row = 0.0, 0.0, None
        for thr in np.arange(-1.0, 3.0, 0.1):
            pred = (o >= thr).astype(int)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            f1v = f1_score(y_true, pred, zero_division=0)
            if prec >= min_prec and f1v > best_f1:
                best_f1, best_thr = f1v, float(thr)
                best_row = (prec, rec, f1v)
        if best_row is None:
            best_thr = float(os.environ.get(f"L2B_OPP_THR_{predicted_regime.upper()}", "0.0"))
            print(
                f"  [L2b regression] {predicted_regime}: no thr met prec>={min_prec}; fallback thr={best_thr:.2f}",
            )
        else:
            prec, rec, f1v = best_row
            print(
                f"  [L2b regression] {predicted_regime}: opp_thr={best_thr:.2f}  "
                f"F1={f1v:.4f}  prec={prec:.3f}  rec={rec:.3f}  (cal, routed by L2a)",
            )
        opp_thr[predicted_regime] = best_thr

    thr_vec = np.array([opp_thr.get(g, 1.0) for g in REGIMES_6], dtype=np.float64)
    return models, opp_thr, thr_vec


def _l2b_nested_opp_models(regb: dict) -> dict[str, dict[str, lgb.Booster]]:
    """``step1_regression`` bundle: flat ``{regime}_mfe`` / ``{regime}_mae`` → nested routing dict."""
    out: dict[str, dict[str, lgb.Booster]] = {}
    for regime in REGIMES_6:
        km, ka = f"{regime}_mfe", f"{regime}_mae"
        if km in regb and ka in regb:
            out[regime] = {"mfe": regb[km], "mae": regb[ka]}
    return out


def _compute_opportunity_triplet(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Routed dual-head: pred_mfe, pred_mae, opportunity = mfe/(mae+0.1)."""
    # st[i] = L2a argmax index k → use models[REGIMES_6[k]] (same k as REGIME_NOW_PROB_COLS[k]).
    st = np.argmax(regime_probs, axis=1).astype(np.int64, copy=False)
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = st == argmax_idx
        if not m.any():
            continue
        mf = models[predicted_regime]["mfe"].predict(X[m])
        ma = models[predicted_regime]["mae"].predict(X[m])
        mf = np.clip(mf, 0.0, None)
        ma = np.clip(ma, 0.01, None)
        opp[m] = np.log1p(mf) - np.log1p(ma)
        mfe_p[m] = mf
        mae_p[m] = ma
    return opp, mfe_p, mae_p


def _compute_opportunity_scores(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> np.ndarray:
    o, _, _ = _compute_opportunity_triplet(X, regime_probs, models)
    return o


def _l2b_triplet_from_trade_prob(p_trade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary Step1 fallback: synthetic MFE/MAE/opp from p_trade (no label leakage)."""
    pt = np.clip(p_trade.astype(np.float64), 0.0, 1.0)
    mf = np.clip(2.0 * pt, 0.0, 5.0)
    ma = np.clip(0.5 + 0.5 * (1.0 - pt), 0.01, 4.0)
    opp = np.log1p(mf) - np.log1p(ma)
    return opp, mf, ma


def _opp_to_synthetic_p_trade(opp: np.ndarray, thr_row: np.ndarray, kappa: float = 4.0) -> np.ndarray:
    """Map opportunity score to (0,1) so Layer3 / reconstruct can keep thr_trade=0.5."""
    z = np.clip(kappa * (opp - thr_row), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _layer3_fill_p_trade_from_regression(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_a: np.ndarray,
    chunk: int,
) -> None:
    s1 = trade_quality_models.get("step1_binary")
    s2 = trade_quality_models["step2"]
    s3 = trade_quality_models["step3"]
    regb = trade_quality_models.get("step1_regression")

    regime_mat = work[list(REGIME_NOW_PROB_COLS)].to_numpy(dtype=np.float32, copy=False)
    models = _l2b_nested_opp_models(regb) if regb else {}
    thr_vec = regb["thr_vec"] if regb else None

    n = len(work)
    n_chunk = (n + chunk - 1) // chunk
    for i in _tq(range(0, n, chunk), desc="Layer3 trade stack", total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)

        if s1 is not None:
            raw_p = s1.predict(x_b)
            rp = regime_mat[i:j]
            p_rm = rp[:, RANGE_REGIME_INDICES].sum(axis=1)
            p_trade[i:j] = raw_p * (1.0 - 0.7 * p_rm)
        else:
            rp = regime_mat[i:j]
            opp = _compute_opportunity_scores(x_b, rp, models)
            gix = np.argmax(rp, axis=1).astype(np.int64, copy=False)
            thr_row = thr_vec[gix]
            p_trade[i:j] = _opp_to_synthetic_p_trade(opp, thr_row)

        p_long[i:j] = s2.predict(x_b)
        p_a[i:j] = s3.predict(x_b)


def _print_quality_label_outcome_stats(df: pd.DataFrame, y6: np.ndarray) -> None:
    """Mean MFE/ATR, MAE/ATR, RR by KMeans-derived quality class (A/B sanity check)."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae_arr = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    rr = mfe / np.maximum(mae_arr, 0.1)
    print("  Outcome stats by quality label (MFE/ATR, MAE/ATR, RR):")
    for c in range(6):
        sel = y6 == c
        if not sel.any():
            continue
        print(
            f"    {QUALITY_CLASS_NAMES[c]:>8s}: n={int(sel.sum()):>9,}  "
            f"mfe={mfe[sel].mean():.3f}  mae={mae_arr[sel].mean():.3f}  rr={rr[sel].mean():.3f}"
        )
    for a, b, name in [(0, 1, "A_LONG vs B_LONG"), (5, 4, "A_SHORT vs B_SHORT"), (2, 3, "NEUTRAL vs CHOP")]:
        ma, mb = y6 == a, y6 == b
        if ma.sum() and mb.sum():
            dmfe = mfe[ma].mean() - mfe[mb].mean()
            drr = rr[ma].mean() - rr[mb].mean()
            print(f"    Δ({name}): Δmfe={dmfe:+.3f}  Δrr={drr:+.3f}")


def _build_trade_quality_targets(df: pd.DataFrame) -> np.ndarray:
    """
    Build 6-class joint discrete labels using Unsupervised Clustering (K-Means)
    on outcome metrics (MFE/ATR, MAE/ATR, RR, log1p(Hold_Time)) for breakouts.
    This replaces the rule-based approach with data-driven statistical groupings.
    """
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    rr = mfe / np.maximum(mae, 0.1)
    hold_time = np.maximum(df["exit_bar"].fillna(0).values.astype(float), 0.0)
    # Right-skewed hold lengths: log1p before Z-score so K-Means isn't dominated by rare long holds
    log_hold_time = np.log1p(hold_time)

    qbull = df["quality_bull_breakout"].fillna(0).values.astype(int)
    qbear = df["quality_bear_breakout"].fillna(0).values.astype(int)
    state = df["state_label"].fillna(2).values.astype(int)

    y = np.full(len(df), 2, dtype=int)  # default NEUTRAL

    both_breakout = (qbull == 1) & (qbear == 1)
    y[both_breakout] = 3  # CHOP

    long_mask = (qbull == 1) & (qbear == 0)
    short_mask = (qbear == 1) & (qbull == 0)
    print(
        f"  [L2b labels] KMeans setup — long={int(long_mask.sum()):,} short={int(short_mask.sum()):,} rows …",
        flush=True,
    )

    def _cluster_and_assign(mask: np.ndarray, is_long: bool):
        indices = np.where(mask)[0]
        if len(indices) < 3:
            # Fallback if too few samples
            y[mask] = 1 if is_long else 4
            return

        # Features: [MFE, MAE, RR, log1p(Hold_Time)] then Z-score below
        X_cluster = np.column_stack([
            mfe[indices],
            mae[indices],
            rr[indices],
            log_hold_time[indices],
        ])

        # Z-score normalization for clustering
        X_mean = X_cluster.mean(axis=0)
        X_std = X_cluster.std(axis=0) + 1e-6
        X_scaled = (X_cluster - X_mean) / X_std

        # GMM clustering into 3 classes (A, B, CHOP)
        # GMM captures the covariance between MFE and MAE better than spherical KMeans
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=5)
        clusters = gmm.fit_predict(X_scaled)

        # Interpret clusters based on heuristic score: MFE + RR - MAE
        cluster_scores = []
        for c in range(3):
            c_mask = (clusters == c)
            if not c_mask.any():
                cluster_scores.append(-1000)
                continue
            c_mfe = mfe[indices][c_mask].mean()
            c_mae = mae[indices][c_mask].mean()
            c_rr = rr[indices][c_mask].mean()
            score = c_mfe + c_rr - c_mae
            cluster_scores.append(score)

        # Sort clusters by score descending
        ranked_clusters = np.argsort(cluster_scores)[::-1]
        
        # Best score -> A grade
        # Middle score -> B grade
        # Lowest score -> CHOP
        grade_a_cluster = ranked_clusters[0]
        grade_b_cluster = ranked_clusters[1]
        chop_cluster = ranked_clusters[2]

        label_a = 0 if is_long else 5
        label_b = 1 if is_long else 4
        label_chop = 3

        for c, label in [(grade_a_cluster, label_a), 
                         (grade_b_cluster, label_b), 
                         (chop_cluster, label_chop)]:
            c_idx = indices[clusters == c]
            y[c_idx] = label

    # Apply clustering separately for longs and shorts
    if long_mask.any():
        print(
            f"  [L2b labels] KMeans long breakouts: n={int(long_mask.sum()):,} …",
            flush=True,
        )
        _cluster_and_assign(long_mask, is_long=True)
    if short_mask.any():
        print(
            f"  [L2b labels] KMeans short breakouts: n={int(short_mask.sum()):,} …",
            flush=True,
        )
        _cluster_and_assign(short_mask, is_long=False)

    # Non-breakout bars in trend regimes may still offer weak directional quality.
    no_breakout = (qbull == 0) & (qbear == 0)
    trend_long_weak = no_breakout & (state == 0) & (mfe >= 0.35) & (rr >= 0.40)
    trend_short_weak = no_breakout & (state == 1) & (mfe >= 0.35) & (rr >= 0.40)
    y[trend_long_weak] = 1
    y[trend_short_weak] = 4
    range_state = np.isin(state, (4, 5))
    y[no_breakout & range_state] = 3
    y[no_breakout & ~range_state & ~(trend_long_weak | trend_short_weak)] = 2
    return y


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Custom eval metric for Focal Loss to be used with LightGBM.
    Returns (eval_name, eval_result, is_higher_better).
    """
    label = dtrain.get_label()
    # Apply sigmoid to raw predictions
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Calculate the loss
    loss = - alpha * (1 - p)**gamma * label * np.log(np.maximum(p, 1e-7)) \
           - (1 - alpha) * p**gamma * (1 - label) * np.log(np.maximum(1 - p, 1e-7))
    
    return 'focal_loss', np.mean(loss), False


def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Custom Focal Loss objective function for LightGBM.
    y_pred is the raw margin (before sigmoid).
    """
    label = dtrain.get_label()
    # Apply sigmoid to get probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Gradient (first derivative)
    # df/dx = (p - y) * alpha_t * [ gamma * (1-p_t)^(gamma-1) * p_t * log(p_t) + (1-p_t)^gamma ]
    # A simplified but effective gradient approximation for focal loss:
    pt = np.where(label == 1, p, 1.0 - p)
    alpha_t = np.where(label == 1, alpha, 1.0 - alpha)
    
    grad = (p - label) * alpha_t * ((1.0 - pt)**gamma)
    
    # Hessian (second derivative) approximation
    # For stability in GBDT, using the standard binary cross-entropy hessian multiplied by the focal weight
    hess = alpha_t * ((1.0 - pt)**gamma) * p * (1.0 - p)
    
    return grad, hess


def _binary_weights(y: np.ndarray, timestamps: np.ndarray, pos_boost: float = 1.0) -> np.ndarray:
    """Balanced weights for binary tasks with recency adjustment."""
    ts = pd.to_datetime(timestamps)
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(n - pos, 1)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    w = np.where(y == 1, w_pos * pos_boost, w_neg).astype(float)

    days_from_end = (ts.max() - ts).total_seconds() / 86400
    max_days = max(days_from_end.max(), 1.0)
    recency = 0.85 + 0.15 * (1.0 - days_from_end / max_days)
    w *= recency.values
    return w


def _reconstruct_quality_classes(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_a: np.ndarray,
    p_range_mass: np.ndarray,
    thr_trade: float = 0.55,
    thr_long: float = 0.50,
    thr_a: float = 0.50,
) -> np.ndarray:
    """Map hierarchical binary outputs back into 6-class trade-quality space.

    ``p_range_mass`` must be the summed L2a probability for range regimes only
    (indices ``RANGE_REGIME_INDICES``, i.e. ``range_conv`` + ``range_div`` in ``REGIMES_6``),
    not a 3-way bull/bear/range bucket and not the sum of all six probs.
    """
    pred = np.full(len(p_trade), 2, dtype=int)  # NEUTRAL default
    skip = p_trade < thr_trade
    pred[skip & (p_range_mass >= 0.50)] = 3  # CHOP when range_* mass is high
    pred[skip & (p_range_mass < 0.50)] = 2   # NEUTRAL when bull/bear mass dominates

    trade = ~skip
    is_long = p_long >= thr_long
    is_a = p_a >= thr_a
    pred[trade & is_long & is_a] = 0
    pred[trade & is_long & ~is_a] = 1
    pred[trade & ~is_long & ~is_a] = 4
    pred[trade & ~is_long & is_a] = 5
    return pred


def _apply_cp_skip(regime_probs: np.ndarray, p_trade: np.ndarray, thr_cp: float, tcn_transition_prob: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply Conformal Prediction prediction sets and TCN transition signal to filter out uncertain/OOS trades."""
    y_set = regime_probs >= thr_cp
    set_size = y_set.sum(axis=1)
    contains_bull = y_set[:, 0] | y_set[:, 1]
    contains_bear = y_set[:, 2] | y_set[:, 3]
    is_conflicting = contains_bull & contains_bear
    skip_cp = (set_size >= 3) | is_conflicting | (set_size == 0)
    
    # Optional: Hard Skip if TCN predicts very high transition probability
    if tcn_transition_prob is not None:
        high_transition_risk = tcn_transition_prob > 0.70
        skip_cp = skip_cp | high_transition_risk
        
    p_trade_adj = p_trade.copy()
    p_trade_adj[skip_cp] = 0.0
    return p_trade_adj, skip_cp


def train_trade_quality_classifier(
    df: pd.DataFrame,
    feat_cols: list[str],
    regime_model: lgb.Booster,
    regime_calibrators: list,
    thr_cp: float,
):
    print("\n" + "=" * 70)
    print("  LAYER 2b: Hierarchical Trade-Quality Stack (regression Step1)")
    print("  y = trade_quality (KMeans outcomes); X excludes regime probabilities")
    print("  Step1 6-regime opp gate  |  Step2 LONG/SHORT  |  Step3 A/B")
    print("=" * 70)

    print(
        f"  [L2b prep] Copying dataframe ({len(df):,} rows) and building labels …",
        flush=True,
    )
    work = df.copy()
    work["trade_quality_label"] = _build_trade_quality_targets(work)

    print("  Label distribution (full):")
    for c in range(6):
        cnt = (work["trade_quality_label"] == c).sum()
        pct = 100.0 * cnt / max(len(work), 1)
        print(f"    {QUALITY_CLASS_NAMES[c]:>8s}: {cnt:>9,}  ({pct:>5.2f}%)")
    _print_quality_label_outcome_stats(work, work["trade_quality_label"].values)

    # Breakout-context features are computed for all bars, so NEUTRAL/CHOP are covered too.
    print("  [L2b prep] Breakout / bar-context features …", flush=True)
    if "symbol" in work.columns:
        bo_parts: list[pd.DataFrame] = []
        grps = list(work.groupby("symbol", sort=False))
        for _, g in _tq(
            grps,
            desc="  L2b BO feats",
            total=len(grps),
            unit="sym",
            leave=False,
        ):
            bo_parts.append(compute_breakout_features(g))
        bo_feats = pd.concat(bo_parts)
        bo_feats = bo_feats.loc[work.index]
    else:
        bo_feats = compute_breakout_features(work)
    for c in _tq(BO_FEAT_COLS, desc="  L2b BO → cols", leave=False, unit="col"):
        work[c] = bo_feats[c].values

    print(
        "  [L2b prep] Regime head predict + per-class calibration (all bars) …",
        flush=True,
    )
    regime_X_cols = _numeric_feature_cols_for_matrix(
        work, _regime_lgbm_feature_cols(feat_cols)
    )
    if not regime_X_cols:
        raise ValueError("Layer 2b prep: no numeric columns for regime head predict.")
    X_state = work[regime_X_cols].values.astype(np.float32)
    n_l2b = len(work)
    l2b_chunk = _layer3_chunk_rows()
    n_l2b_b = (n_l2b + l2b_chunk - 1) // l2b_chunk
    raw_regime = np.empty((n_l2b, NUM_REGIME_CLASSES), dtype=np.float64)
    for i in _tq(
        range(0, n_l2b, l2b_chunk),
        desc="  L2b regime raw",
        total=n_l2b_b,
        unit="chunk",
        leave=False,
    ):
        j = min(i + l2b_chunk, n_l2b)
        raw_regime[i:j] = regime_model.predict(X_state[i:j])
    cal_regime = np.empty((n_l2b, NUM_REGIME_CLASSES), dtype=np.float64)
    for i in _tq(
        range(0, n_l2b, l2b_chunk),
        desc="  L2b regime cal",
        total=n_l2b_b,
        unit="chunk",
        leave=False,
    ):
        j = min(i + l2b_chunk, n_l2b)
        row = raw_regime[i:j]
        cal_blk = np.column_stack([
            regime_calibrators[c].predict(row[:, c]) for c in range(NUM_REGIME_CLASSES)
        ])
        cal_blk = np.maximum(cal_blk, 1e-12)
        cal_blk /= cal_blk.sum(axis=1, keepdims=True)
        cal_regime[i:j] = cal_blk

    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        work[col] = cal_regime[:, j]
    work["regime_now_conf"] = cal_regime.max(axis=1)

    print("  [L2b prep] Resolving feature groups (PA / HMM / GARCH / TCN) …", flush=True)
    garch_cols = [
        c for c in work.columns
        if "garch" in c.lower() and str(work[c].dtype) not in {"object", "category"}
    ]
    hmm_cols = [
        c for c in work.columns
        if c.startswith("pa_hmm_") and str(work[c].dtype) not in {"object", "category"}
    ]

    all_bo_feats_raw = _unique_cols(feat_cols + BO_FEAT_COLS + sorted(garch_cols))
    all_bo_feats = _numeric_feature_cols_for_matrix(work, all_bo_feats_raw)
    if not all_bo_feats:
        raise ValueError("Layer 2b: no numeric features left after filtering.")
    pa_base, pa_hmm, pa_garch, pa_tcn = _split_feature_groups(feat_cols)
    print(f"  Feature set (deduped):")
    print(f"    Base PA/OR: {len(pa_base)}")
    print(f"    HMM-style:  {len(pa_hmm)}")
    print(f"    GARCH-style:{len(pa_garch)}")
    print(f"    TCN-derived:{len(pa_tcn)}")
    print(f"    Breakout:   {len(BO_FEAT_COLS)}")
    print(f"    Regime probs: not in X (Layer 2a outputs; used for reconstruct + Layer 3 only)")
    print(f"    TOTAL L2b:  {len(all_bo_feats)}")
    if hmm_cols:
        print(f"  HMM features included: {', '.join(sorted(hmm_cols)[:6])}"
              + (" ..." if len(hmm_cols) > 6 else ""))
    if garch_cols:
        print(f"  GARCH features included: {', '.join(garch_cols[:6])}"
              + (" ..." if len(garch_cols) > 6 else ""))

    X = work[all_bo_feats].values.astype(np.float32)
    _require_lgb_matrix_matches_names(X, all_bo_feats, "Layer 2b (trade-quality stack)")
    y6 = work["trade_quality_label"].values.astype(int)
    t = work["time_key"].values

    train_mask = t < np.datetime64(TRAIN_END)
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, y_train6 = X[train_mask], y6[train_mask]
    X_cal, y_cal6 = X[cal_mask], y6[cal_mask]
    X_test, y_test6 = X[test_mask], y6[test_mask]

    y_trade_train = np.isin(y_train6, list(TRADABLE_CLASS_IDS)).astype(int)
    y_trade_cal = np.isin(y_cal6, list(TRADABLE_CLASS_IDS)).astype(int)
    y_trade_test = np.isin(y_test6, list(TRADABLE_CLASS_IDS)).astype(int)

    tradable_train = y_trade_train == 1
    tradable_cal = y_trade_cal == 1
    tradable_test = y_trade_test == 1

    y_dir_train = np.isin(y_train6[tradable_train], [0, 1]).astype(int)   # 1=LONG, 0=SHORT
    y_dir_cal = np.isin(y_cal6[tradable_cal], [0, 1]).astype(int)
    y_dir_test = np.isin(y_test6[tradable_test], [0, 1]).astype(int)

    y_ab_train = np.isin(y_train6[tradable_train], [0, 5]).astype(int)     # 1=A, 0=B
    y_ab_cal = np.isin(y_cal6[tradable_cal], [0, 5]).astype(int)
    y_ab_test = np.isin(y_test6[tradable_test], [0, 5]).astype(int)

    print(f"  Dates — Train: < {TRAIN_END} | Cal: → {CAL_END} | Test: → {TEST_END}")
    print(f"  Train: {len(y_train6):,}  |  Cal: {len(y_cal6):,}  |  Test: {len(y_test6):,}")
    print(f"  Step1 TRADE rate (train/test): {y_trade_train.mean():.2%} / {y_trade_test.mean():.2%}")

    mc = np.zeros(len(all_bo_feats), dtype=int)
    # Impose monotonic constraints: higher values of these risk features MUST monotonically decrease the quality score.
    # This prevents the tree from learning irrational rules on historical black swan events.
    if "bo_wick_imbalance" in all_bo_feats:
        mc[all_bo_feats.index("bo_wick_imbalance")] = -1
    if "pa_bo_wick_imbalance" in all_bo_feats:
        mc[all_bo_feats.index("pa_bo_wick_imbalance")] = -1
    if "pa_vol_exhaustion_climax" in all_bo_feats:
        mc[all_bo_feats.index("pa_vol_exhaustion_climax")] = -1
    mc_tuple = tuple(mc)

    common_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": 7,
        "learning_rate": 0.02,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 60,
        "lambda_l1": 0.2,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        "monotone_constraints": mc_tuple,
        "monotone_constraints_method": "advanced",
    }
    rounds = 1800 if FAST_TRAIN_MODE else 4000
    es = 90 if FAST_TRAIN_MODE else 140

    X2_train, X2_cal, X2_test = X_train[tradable_train], X_cal[tradable_cal], X_test[tradable_test]
    t2_train, t2_cal = t[train_mask][tradable_train], t[cal_mask][tradable_cal]
    w2_train = _binary_weights(y_dir_train, t2_train, pos_boost=1.0)
    w2_cal = _binary_weights(y_dir_cal, t2_cal, pos_boost=1.0)
    d2_train = lgb.Dataset(X2_train, label=y_dir_train, weight=w2_train, feature_name=all_bo_feats, free_raw_data=False)
    d2_cal = lgb.Dataset(X2_cal, label=y_dir_cal, weight=w2_cal, feature_name=all_bo_feats, free_raw_data=False)
    w3_train = _binary_weights(y_ab_train, t2_train, pos_boost=1.3)
    w3_cal = _binary_weights(y_ab_cal, t2_cal, pos_boost=1.1)
    d3_train = lgb.Dataset(X2_train, label=y_ab_train, weight=w3_train, feature_name=all_bo_feats, free_raw_data=False)
    d3_cal = lgb.Dataset(X2_cal, label=y_ab_cal, weight=w3_cal, feature_name=all_bo_feats, free_raw_data=False)

    reg_models: dict[str, dict[str, lgb.Booster]] = {}
    thr_vec = np.ones(len(REGIMES_6), dtype=np.float64)

    print(
        "  [L2b train] Step1 = 6-regime dual regression (MFE & MAE @ ATR → opportunity) …",
        flush=True,
    )
    mfe_full, mae_full = _mfe_mae_atr_arrays(work)
    st_full = work["state_label"].values.astype(int)
    rp_full = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)
    reg_models, _, thr_vec = _train_regime_opp_regression_models(
        X,
        st_full,
        mfe_full,
        mae_full,
        train_mask,
        cal_mask,
        all_bo_feats,
        rp_full[cal_mask],
        y_trade_cal,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    for regime in REGIMES_6:
        if regime not in reg_models:
            continue
        reg_models[regime]["mfe"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mfe_{regime}.txt"))
        reg_models[regime]["mae"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mae_{regime}.txt"))
    step1_regression_bundle = {"thr_vec": thr_vec}
    for regime, pair in reg_models.items():
        step1_regression_bundle[f"{regime}_mfe"] = pair["mfe"]
        step1_regression_bundle[f"{regime}_mae"] = pair["mae"]

    print("  [L2b train] Dedicated Step1 TRADE/SKIP binary classifier …", flush=True)
    c1, clean1 = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step1 Binary")
    # Massive pos_boost to force model to learn the rare 3% signals instead of predicting CHOP for everything
    w1_train = _binary_weights(y_trade_train, t[train_mask], pos_boost=8.0)
    w1_cal = _binary_weights(y_trade_cal, t[cal_mask], pos_boost=4.0)
    d1_train = lgb.Dataset(X_train, label=y_trade_train, weight=w1_train, feature_name=all_bo_feats, free_raw_data=False)
    d1_cal = lgb.Dataset(X_cal, label=y_trade_cal, weight=w1_cal, feature_name=all_bo_feats, free_raw_data=False)
    try:
        step1_binary_model = lgb.train(
            common_params, d1_train, num_boost_round=rounds, valid_sets=[d1_cal], callbacks=c1,
        )
    finally:
        for fn in clean1:
            fn()

    print("  [L2b train] Step2 LONG/SHORT LightGBM …", flush=True)
    c2, clean2 = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step2 LONG/SHORT")
    try:
        step2_model = lgb.train(
            common_params, d2_train, num_boost_round=rounds, valid_sets=[d2_cal], callbacks=c2,
        )
    finally:
        for fn in clean2:
            fn()
    print("  [L2b train] Step3 A/B LightGBM (Custom Focal Loss) …", flush=True)
    c3, clean3 = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step3 A/B")
    
    # Use Focal Loss for Step 3 to drastically boost A-grade recall (dealing with extreme 0.06% class imbalance)
    # 1=A, 0=B. A is extremely rare, so we need alpha > 0.5 to weight it higher, and gamma >= 2.0 to focus on hard examples.
    def custom_focal_obj(preds, dtrain): return focal_loss_lgb(preds, dtrain, alpha=0.75, gamma=2.0)
    def custom_focal_eval(preds, dtrain): return focal_loss_lgb_eval_error(preds, dtrain, alpha=0.75, gamma=2.0)
    
    focal_params = common_params.copy()
    # Remove standard objective/metric to use custom ones
    focal_params.pop("objective", None)
    focal_params.pop("metric", None)
    
    try:
        step3_model = lgb.train(
            focal_params, d3_train, num_boost_round=rounds, valid_sets=[d3_cal], 
            callbacks=c3, fobj=custom_focal_obj, feval=custom_focal_eval
        )
    finally:
        for fn in clean3:
            fn()

    rp_cal = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[cal_mask]
    p_trade_cal = step1_binary_model.predict(X_cal)
    
    # 1. Range Regime Penalty: heavily slash probabilities in range regimes
    p_range_mass_cal = rp_cal[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_trade_cal = p_trade_cal * (1.0 - 0.7 * p_range_mass_cal)
    
    # 2. Threshold Search: Prioritize Precision via F0.5 score + strict Precision floor >= 10%
    best_f05, best_thr = 0.0, 0.7
    for thr_c in np.arange(0.15, 0.95, 0.02):
        pr = (p_trade_cal >= thr_c).astype(int)
        f05 = fbeta_score(y_trade_cal, pr, beta=0.5, zero_division=0)
        prec = precision_score(y_trade_cal, pr, zero_division=0)
        # Relaxed precision floor from 15% to 10% to improve recall for rare A_LONG / A_SHORT opportunities
        if f05 > best_f05 and prec >= 0.10:
            best_f05, best_thr = f05, thr_c
    thr_trade_cal = best_thr if best_f05 > 0 else 0.85
    
    tcn_transition_prob_cal = work["tcn_transition_prob"].values.astype(np.float32)[cal_mask] if "tcn_transition_prob" in work.columns else None
    p_trade_cal, _ = _apply_cp_skip(rp_cal, p_trade_cal, thr_cp, tcn_transition_prob_cal)
    pr_cal = (p_trade_cal >= thr_trade_cal).astype(int)
    f1_cal_at_thr = f1_score(y_trade_cal, pr_cal, zero_division=0)
    n_trade_pred_cal = int(pr_cal.sum())
    rec_cal_thr = recall_score(y_trade_cal, pr_cal, zero_division=0)
    prec_cal_thr = precision_score(y_trade_cal, pr_cal, zero_division=0)
    thr_rule_note = f"binary gate: opt F0.5 thr={thr_trade_cal:.2f} (prec>=0.10)"
    print(
        f"  Step1 cal: {thr_rule_note}  "
        f"F1={f1_cal_at_thr:.4f}  recall={rec_cal_thr:.3f}  precision={prec_cal_thr:.4f}  "
        f"n_trade={n_trade_pred_cal:,}/{len(y_trade_cal):,}"
    )

    # Evaluate cascade
    rp_test = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[test_mask]
    opp_te = _compute_opportunity_scores(X_test, rp_test, reg_models)
    p_trade = step1_binary_model.predict(X_test)
    
    # Apply identical range penalty to test set
    p_range_mass_test = rp_test[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_trade = p_trade * (1.0 - 0.7 * p_range_mass_test)
    
    tcn_transition_prob_test = work["tcn_transition_prob"].values.astype(np.float32)[test_mask] if "tcn_transition_prob" in work.columns else None
    p_trade, skip_cp_test = _apply_cp_skip(rp_test, p_trade, thr_cp, tcn_transition_prob_test)
    print(f"\n  CP Skip rate (Layer 2b routed test): {skip_cp_test.mean():.2%} (forced p_trade=0)")
    
    p_long = np.full(len(X_test), 0.5, dtype=float)
    p_a = np.full(len(X_test), 0.5, dtype=float)
    if len(X2_test) > 0:
        p_long[tradable_test] = step2_model.predict(X2_test)
        p_a[tradable_test] = step3_model.predict(X2_test)
    rp_rows = work.loc[test_mask, REGIME_NOW_PROB_COLS].to_numpy(dtype=np.float64, copy=False)
    p_range_mass = rp_rows[:, RANGE_REGIME_INDICES].sum(axis=1)
    y_pred6 = _reconstruct_quality_classes(
        p_trade, p_long, p_a, p_range_mass, thr_trade=thr_trade_cal
    )

    print("\n  Step metrics (test):")
    step1_pred = (p_trade >= thr_trade_cal).astype(int)
    s1_rec = recall_score(y_trade_test, step1_pred, zero_division=0)
    s1_prec = precision_score(y_trade_test, step1_pred, zero_division=0)
    print(
        f"    Step1 TRADE/SKIP  acc={accuracy_score(y_trade_test, step1_pred):.4f} "
        f"f1={f1_score(y_trade_test, step1_pred, zero_division=0):.4f}  "
        f"recall={s1_rec:.3f}  precision={s1_prec:.4f}  (thr={thr_trade_cal:.3f})"
    )
    if len(y_dir_test) > 0:
        step2_pred = (step2_model.predict(X2_test) >= 0.5).astype(int)
        print(f"    Step2 LONG/SHORT acc={accuracy_score(y_dir_test, step2_pred):.4f} "
              f"f1={f1_score(y_dir_test, step2_pred, zero_division=0):.4f}")
    if len(y_ab_test) > 0:
        step3_pred = (step3_model.predict(X2_test) >= 0.5).astype(int)
        print(f"    Step3 A/B        acc={accuracy_score(y_ab_test, step3_pred):.4f} "
              f"f1={f1_score(y_ab_test, step3_pred, zero_division=0):.4f}")

    acc = accuracy_score(y_test6, y_pred6)
    macro_f1 = f1_score(y_test6, y_pred6, average="macro")
    print("\n  Reconstructed 6-class metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test6, y_pred6, target_names=QUALITY_CLASS_ORDER, digits=4))
    cm = confusion_matrix(y_test6, y_pred6, labels=list(range(6)))
    print("  Confusion Matrix:")
    print(pd.DataFrame(cm, index=QUALITY_CLASS_ORDER, columns=QUALITY_CLASS_ORDER).to_string())

    _g = next(g for g in REGIMES_6 if g in reg_models)
    importance = reg_models[_g]["mfe"].feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": all_bo_feats, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    print("\n  Top 25 Step1 features (gain):")
    print(imp_df.head(25).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    step1_binary_model.save_model(os.path.join(MODEL_DIR, "trade_gate_step1.txt"))
    step2_model.save_model(os.path.join(MODEL_DIR, "trade_dir_step2.txt"))
    step3_model.save_model(os.path.join(MODEL_DIR, "trade_grade_step3.txt"))
    import pickle

    meta = {
        "type": "trade_quality_hier_binary_gate",
        "class_names": QUALITY_CLASS_NAMES,
        "feature_cols": all_bo_feats,
        "pa_base_feat_cols": pa_base,
        "pa_hmm_feat_cols": pa_hmm,
        "pa_garch_feat_cols": pa_garch,
        "tcn_feat_cols": pa_tcn,
        "bo_feat_cols": BO_FEAT_COLS,
        "regime_prob_cols_layer3": REGIME_PROB_COLS,
        "garch_cols": garch_cols,
        "hierarchy_thresholds": {
            "trade": float(thr_trade_cal), "long": 0.50, "grade_a": 0.50, 
            "cp_alpha": 0.05, "thr_cp": float(thr_cp)
        },
        "step1_calibration": {
            "best_trade_threshold": float(thr_trade_cal),
            "best_f1_cal": float(f1_cal_at_thr),
            "recall_cal": float(rec_cal_thr),
            "precision_cal": float(prec_cal_thr),
            "n_trade_pred_cal": int(n_trade_pred_cal),
            "threshold_selection_rule": thr_rule_note,
        },
        "model_files": {
            "step1_trade": "trade_gate_step1.txt",
            "step2_direction": "trade_dir_step2.txt",
            "step3_grade": "trade_grade_step3.txt",
        },
        "position_scale_map": {
            "A_LONG": 1.50,
            "B_LONG": 1.00,
            "NEUTRAL": 0.00,
            "CHOP": 0.00,
            "B_SHORT": -1.00,
            "A_SHORT": -1.50,
        },
        "regression_gate": {
            "groups": list(REGIMES_6),
            "thr_vec": thr_vec.tolist(),
            "model_files": {
                fk: fn
                for regime in REGIMES_6
                if regime in reg_models
                for fk, fn in (
                    (f"{regime}_mfe", f"l2b_opp_mfe_{regime}.txt"),
                    (f"{regime}_mae", f"l2b_opp_mae_{regime}.txt"),
                )
            },
        },
    }

    with open(os.path.join(MODEL_DIR, "trade_quality_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(
        f"\n  Models saved → trade_gate_step1.txt, trade_dir_step2.txt, trade_grade_step3.txt\n"
        f"  (L3 continuous features models saved to l2b_opp_mfe/mae_<regime>.txt)",
    )
    print(f"  Meta saved  → {MODEL_DIR}/trade_quality_meta.pkl")

    model_bundle = {
        "step1_regression": step1_regression_bundle,
        "step1_binary": step1_binary_model,
        "step2": step2_model,
        "step3": step3_model,
        "thresholds": meta["hierarchy_thresholds"],
        "feature_cols": all_bo_feats,
    }
    return model_bundle, meta, imp_df



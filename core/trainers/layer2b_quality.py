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
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix, precision_score, recall_score, fbeta_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features

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
    regime_route_probs: np.ndarray,
    y_trade_cal: np.ndarray,
) -> tuple[dict[str, dict[str, lgb.Booster]], dict[str, float], np.ndarray, float]:
    """Train up to 6× (MFE + MAE) regressors (one pair per REGIMES_6 name); tune thresholds on cal.

    Train / cal routing uses the base regime model's raw probabilities so downstream regression
    heads do not consume future information from the later calibration window. ``state_label`` is
    retained only for mismatch diagnostics.

    **Cal / inference routing**: pick head with ``argmax(regime_now probabilities) == k``; then
    ``score = pred_mfe / (pred_mae + 0.1)`` on those rows only; F1 grid-search yields ``thr_vec[k]``.

    Six regimes ⇒ fewer routed cal rows per bucket than old 3-way groups — default min row
    count is lower (``L2B_OPP_CAL_MIN_ROWS``). A global soft-mixture threshold is also
    calibrated on the full cal split for downstream mixture routing.
    """
    reg_rounds = 1500 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ROUNDS", "4000"))
    reg_es = 150 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ES", "300"))
    min_prec = float(os.environ.get("L2B_OPP_MIN_PRECISION", "0.20"))  # Relaxed precision floor to allow more opportunities
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
    rp_tr = regime_route_probs[train_mask]
    rp_ca = regime_route_probs[cal_mask]
    st_tr_pred = np.argmax(rp_tr, axis=1).astype(np.int64, copy=False)
    st_ca_pred = np.argmax(rp_ca, axis=1).astype(np.int64, copy=False)
    mfe_tr = mfe[train_mask]
    mae_tr = mae[train_mask]
    mfe_ca = mfe[cal_mask]
    mae_ca = mae[cal_mask]

    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        mtr = st_tr_pred == argmax_idx
        mca = st_ca_pred == argmax_idx
        ntr, nca = int(mtr.sum()), int(mca.sum())
        train_match = float((st_tr[mtr] == argmax_idx).mean()) if ntr else float("nan")
        cal_match = float((st_ca[mca] == argmax_idx).mean()) if nca else float("nan")
        print(
            f"  [L2b regression] {predicted_regime}: "
            f"train_rows(routed)={ntr:,}  cal_rows(routed)={nca:,}  "
            f"label_match(train/cal)={train_match:.1%}/{cal_match:.1%}"
        )
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

    # L2a routing: argmax on raw regime probs → class index k == predicted_regime REGIMES_6[k]
    gix_cal = st_ca_pred
    n_cal = len(y_trade_cal)
    opp_cal = np.zeros(n_cal, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = gix_cal == argmax_idx
        if not m.any():
            continue
        mf = models[predicted_regime]["mfe"].predict(X_ca[m])
        ma = models[predicted_regime]["mae"].predict(X_ca[m])
        mf = np.clip(mf, 0.0, None)
        ma = np.clip(ma, 0.01, None)
        opp_cal[m] = np.log1p(mf) - np.log1p(ma)

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

    best_soft_f1 = 0.0
    best_soft_thr = float(os.environ.get("L2B_OPP_SOFT_THR", "0.0"))
    for thr in np.arange(-1.0, 3.0, 0.1):
        pred = (opp_cal >= thr).astype(int)
        prec = precision_score(y_trade_cal, pred, zero_division=0)
        f1v = f1_score(y_trade_cal, pred, zero_division=0)
        if prec >= min_prec and f1v > best_soft_f1:
            best_soft_f1 = f1v
            best_soft_thr = float(thr)
    print(
        f"  [L2b regression] soft_mix_thr={best_soft_thr:.2f}  "
        f"F1={best_soft_f1:.4f}  (cal, full mixture)"
    )

    thr_vec = np.array([opp_thr.get(g, 1.0) for g in REGIMES_6], dtype=np.float64)
    return models, opp_thr, thr_vec, best_soft_thr


def _l2b_nested_opp_models(regb: dict) -> dict[str, dict[str, lgb.Booster]]:
    """``step1_regression`` bundle: flat ``{regime}_mfe`` / ``{regime}_mae`` → nested routing dict."""
    out: dict[str, dict[str, lgb.Booster]] = {}
    for regime in REGIMES_6:
        km, ka = f"{regime}_mfe", f"{regime}_mae"
        if km in regb and ka in regb:
            out[regime] = {"mfe": regb[km], "mae": regb[ka]}
    return out


def _available_l2b_regime_indices(models: dict[str, dict[str, lgb.Booster]]) -> list[int]:
    return [idx for idx, regime in enumerate(REGIMES_6) if regime in models]


def _compute_opportunity_triplet(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft mixture over regime experts: expected MFE/MAE then log opportunity."""
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    if n == 0 or not models:
        return opp, mfe_p, mae_p

    available = _available_l2b_regime_indices(models)
    if not available:
        return opp, mfe_p, mae_p

    mfe_stack: list[np.ndarray] = []
    mae_stack: list[np.ndarray] = []
    for idx in available:
        regime = REGIMES_6[idx]
        mf = np.clip(models[regime]["mfe"].predict(X), 0.0, None)
        ma = np.clip(models[regime]["mae"].predict(X), 0.01, None)
        mfe_stack.append(mf)
        mae_stack.append(ma)

    weights = np.clip(regime_probs[:, available].astype(np.float64, copy=False), 0.0, None)
    if weights.shape[1] == 0:
        return opp, mfe_p, mae_p
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        denom,
        out=np.full_like(weights, 1.0 / weights.shape[1]),
        where=denom > 1e-12,
    )
    mfe_mat = np.column_stack(mfe_stack)
    mae_mat = np.column_stack(mae_stack)
    mfe_p = np.sum(weights * mfe_mat, axis=1)
    mae_p = np.sum(weights * mae_mat, axis=1)
    opp = np.log1p(mfe_p) - np.log1p(mae_p)
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


def _layer3_fill_trade_stack_probs_gates(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
    chunk: int,
    *,
    tqdm_desc: str = "Layer3 trade stack",
) -> None:
    s1_long = trade_quality_models.get("step1_long")
    s1_short = trade_quality_models.get("step1_short")
    regb = trade_quality_models.get("step1_regression")

    regime_mat = work[list(REGIME_NOW_PROB_COLS)].to_numpy(dtype=np.float32, copy=False)
    models = _l2b_nested_opp_models(regb) if regb else {}
    thr_vec = regb["thr_vec"] if regb else None

    n = len(work)
    n_chunk = (n + chunk - 1) // chunk
    for i in _tq(range(0, n, chunk), desc=tqdm_desc, total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)

        if s1_long is not None and s1_short is not None:
            pl = 1.0 / (1.0 + np.exp(-s1_long.predict(x_b)))
            ps = 1.0 / (1.0 + np.exp(-s1_short.predict(x_b)))
            rp = regime_mat[i:j]
            p_rm = rp[:, RANGE_REGIME_INDICES].sum(axis=1)
            p_long_gate[i:j] = pl * (1.0 - 0.7 * p_rm)
            p_short_gate[i:j] = ps * (1.0 - 0.7 * p_rm)
        else:
            rp = regime_mat[i:j]
            opp = _compute_opportunity_scores(x_b, rp, models)
            soft_thr = float(regb.get("soft_opp_threshold", np.nan)) if regb else float("nan")
            if np.isfinite(soft_thr):
                thr_row = np.full(len(rp), soft_thr, dtype=np.float64)
            else:
                gix = np.argmax(rp, axis=1).astype(np.int64, copy=False)
                thr_row = thr_vec[gix]
            p_trade = _opp_to_synthetic_p_trade(opp, thr_row)
            p_long_gate[i:j] = p_trade
            p_short_gate[i:j] = p_trade


def _print_quality_label_outcome_stats(df: pd.DataFrame, y6: np.ndarray) -> None:
    """Mean MFE/ATR, MAE/ATR, RR by KMeans-derived quality class (A/B sanity check)."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae_arr = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    rr = mfe / np.maximum(mae_arr, 0.1)
    print("  Outcome stats by quality label (MFE/ATR, MAE/ATR, RR):")
    for c in range(4):
        sel = y6 == c
        if not sel.any():
            continue
        print(
            f"    {QUALITY_CLASS_NAMES[c]:>8s}: n={int(sel.sum()):>9,}  "
            f"mfe={mfe[sel].mean():.3f}  mae={mae_arr[sel].mean():.3f}  rr={rr[sel].mean():.3f}"
        )
    for a, b, name in [(0, 3, "LONG vs SHORT"), (1, 2, "NEUTRAL vs CHOP")]:
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
    hold_time = np.maximum(df["exit_bar"].fillna(0).values.astype(float), 0.0)
    
    # ---------------------------------------------------------
    # OPTIONS GAMMA SCALPING: Time-decay on MFE
    # ---------------------------------------------------------
    # Penalize slow favorable moves so trade-quality labels reflect theta-sensitive intraday options.
    gamma_decay = _theta_decay_from_bars(hold_time)
    mfe = mfe * gamma_decay
    
    rr = mfe / np.maximum(mae, 0.1)
    # Right-skewed hold lengths: log1p before Z-score so K-Means isn't dominated by rare long holds
    log_hold_time = np.log1p(hold_time)

    qbull = df["quality_bull_breakout"].fillna(0).values.astype(int)
    qbear = df["quality_bear_breakout"].fillna(0).values.astype(int)
    state = df["state_label"].fillna(2).values.astype(int)

    y = np.full(len(df), 1, dtype=int)  # default NEUTRAL

    both_breakout = (qbull == 1) & (qbear == 1)
    y[both_breakout] = 2  # CHOP

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
            y[mask] = 0 if is_long else 3
            return

        # Features: [MFE, MAE, RR, log1p(Hold_Time)] then Z-score below
        X_cluster = np.column_stack([
            mfe[indices],
            mae[indices],
            rr[indices],
            log_hold_time[indices],
        ])

        # Find train indices within this mask to avoid data leakage
        train_indices_mask = mask & (df["time_key"].values < np.datetime64(TRAIN_END))
        
        # If train set is too small, fallback to using all indices for stats
        if train_indices_mask.sum() < 5:
            train_indices = indices
            X_train_cluster = X_cluster
        else:
            train_indices = np.where(train_indices_mask)[0]
            X_train_cluster = np.column_stack([
                mfe[train_indices],
                mae[train_indices],
                rr[train_indices],
                log_hold_time[train_indices],
            ])

        # Z-score normalization using train statistics
        X_mean = X_train_cluster.mean(axis=0)
        X_std = X_train_cluster.std(axis=0) + 1e-6
        X_scaled = (X_cluster - X_mean) / X_std
        X_train_scaled = (X_train_cluster - X_mean) / X_std

        # GMM clustering into 2 classes (TRADE, CHOP)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=5)
        gmm.fit(X_train_scaled)
        clusters = gmm.predict(X_scaled)
        train_clusters = clusters[np.isin(indices, train_indices)]

        # Interpret clusters based on heuristic score on train data
        cluster_scores = []
        for c in range(2):
            c_mask = (train_clusters == c)
            if not c_mask.any():
                cluster_scores.append(-1000)
                continue
            c_mfe = mfe[train_indices][c_mask].mean()
            c_mae = mae[train_indices][c_mask].mean()
            c_rr = rr[train_indices][c_mask].mean()
            score = c_mfe + c_rr - c_mae
            cluster_scores.append(score)

        # Sort clusters by score descending
        ranked_clusters = np.argsort(cluster_scores)[::-1]
        
        # Best score -> TRADE
        # Lowest score -> CHOP
        trade_cluster = ranked_clusters[0]
        chop_cluster = ranked_clusters[1]

        label_trade = 0 if is_long else 3
        label_chop = 2

        for c, label in [(trade_cluster, label_trade), 
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
    y[trend_long_weak] = 0
    y[trend_short_weak] = 3
    range_state = np.isin(state, (4, 5))
    y[no_breakout & range_state] = 2
    y[no_breakout & ~range_state & ~(trend_long_weak | trend_short_weak)] = 1
    return y


def _binary_recency_weights(timestamps: np.ndarray) -> np.ndarray:
    """Recency-only weights for native BCE training with scale_pos_weight."""
    ts = pd.to_datetime(timestamps)
    days_from_end = (ts.max() - ts).total_seconds() / 86400
    max_days = max(days_from_end.max(), 1.0)
    recency = 0.85 + 0.15 * (1.0 - days_from_end / max_days)
    return recency.values.astype(float)


def _binary_scale_pos_weight(y: np.ndarray, *, max_spw: float = 50.0) -> float:
    """Native LightGBM class prior correction for rare positive gates."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return max_spw
    if neg <= 0:
        return 1.0
    return float(np.clip(neg / max(pos, 1), 1.0, max_spw))


def _reconstruct_quality_classes(
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
    p_range_mass: np.ndarray,
    thr_long: float = 0.55,
    thr_short: float = 0.55,
) -> np.ndarray:
    """Map hierarchical binary outputs back into 4-class trade-quality space."""
    pred = np.full(len(p_long_gate), 1, dtype=int)  # NEUTRAL default
    
    is_long = p_long_gate >= thr_long
    is_short = p_short_gate >= thr_short
    
    skip = ~(is_long | is_short)
    pred[skip & (p_range_mass >= 0.50)] = 2  # CHOP when range_* mass is high
    pred[skip & (p_range_mass < 0.50)] = 1   # NEUTRAL when bull/bear mass dominates

    both = is_long & is_short
    conflict_long = both & (p_long_gate > p_short_gate)
    conflict_short = both & (p_short_gate >= p_long_gate)
    
    pred[is_long & ~both] = 0
    pred[is_short & ~both] = 3
    pred[conflict_long] = 0
    pred[conflict_short] = 3
    
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
    regime_calibrators: Any,
    thr_cp: float,
):
    print("\n" + "=" * 70)
    print("  LAYER 2b: Hierarchical Trade-Quality Stack (regression Step1)")
    print("  y = trade_quality (KMeans outcomes); X excludes regime probabilities")
    print("  Step1 6-regime opp gate  |  Step2 LONG/SHORT")
    print("=" * 70)

    print(
        f"  [L2b prep] Copying dataframe ({len(df):,} rows) and building labels …",
        flush=True,
    )
    work = df.copy()
    work["trade_quality_label"] = _build_trade_quality_targets(work)

    print("  Label distribution (full):")
    for c in range(4):
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
        if isinstance(regime_calibrators, list):
            cal_blk = np.column_stack([
                regime_calibrators[c].predict(row[:, c]) for c in range(NUM_REGIME_CLASSES)
            ])
            cal_blk = np.maximum(cal_blk, 1e-12)
            cal_blk /= cal_blk.sum(axis=1, keepdims=True)
            cal_regime[i:j] = cal_blk
        else:
            eps = 1e-7
            l_p = np.log(np.clip(row, eps, 1 - eps))
            cal_blk = regime_calibrators.predict_proba(l_p)
            cal_regime[i:j] = cal_blk

    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        work[col] = cal_regime[:, j]
    work["regime_now_conf"] = cal_regime.max(axis=1)
    raw_regime = raw_regime.astype(np.float32, copy=False)

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
    pa_base, pa_hmm, pa_garch, pa_tcn, pa_mamba = _split_feature_groups(feat_cols)
    print(f"  Feature set (deduped):")
    print(f"    Base PA/OR: {len(pa_base)}")
    print(f"    HMM-style:  {len(pa_hmm)}")
    print(f"    GARCH-style:{len(pa_garch)}")
    print(f"    TCN-derived:{len(pa_tcn)}")
    print(f"    Mamba-deriv:{len(pa_mamba)}")
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

    y_long_train = (y_train6 == 0).astype(int)
    y_long_cal = (y_cal6 == 0).astype(int)
    y_long_test = (y_test6 == 0).astype(int)

    y_short_train = (y_train6 == 3).astype(int)
    y_short_cal = (y_cal6 == 3).astype(int)
    y_short_test = (y_test6 == 3).astype(int)

    print(f"  Dates — Train: < {TRAIN_END} | Cal: → {CAL_END} | Test: → {TEST_END}")
    print(f"  Train: {len(y_train6):,}  |  Cal: {len(y_cal6):,}  |  Test: {len(y_test6):,}")
    print(f"  LONG rate (train/test): {y_long_train.mean():.2%} / {y_long_test.mean():.2%}")
    print(f"  SHORT rate (train/test): {y_short_train.mean():.2%} / {y_short_test.mean():.2%}")

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
        "metric": "None",
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
    
    reg_models: dict[str, dict[str, lgb.Booster]] = {}
    thr_vec = np.ones(len(REGIMES_6), dtype=np.float64)

    print(
        "  [L2b train] Step1 = 6-regime dual regression (MFE & MAE @ ATR → opportunity) …",
        flush=True,
    )
    mfe_full, mae_full = _mfe_mae_atr_arrays(work)
    st_full = work["state_label"].values.astype(int)
    rp_route_full = raw_regime
    reg_models, _, thr_vec, soft_opp_thr = _train_regime_opp_regression_models(
        X,
        st_full,
        mfe_full,
        mae_full,
        train_mask,
        cal_mask,
        all_bo_feats,
        rp_route_full,
        y_trade_cal,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    for regime in REGIMES_6:
        if regime not in reg_models:
            continue
        reg_models[regime]["mfe"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mfe_{regime}.txt"))
        reg_models[regime]["mae"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mae_{regime}.txt"))
    step1_regression_bundle = {
        "thr_vec": thr_vec,
        "soft_opp_threshold": float(soft_opp_thr),
        "gating_mode": "soft_mixture",
    }
    for regime, pair in reg_models.items():
        step1_regression_bundle[f"{regime}_mfe"] = pair["mfe"]
        step1_regression_bundle[f"{regime}_mae"] = pair["mae"]

    print("  [L2b train] Dedicated Step1 LONG gate binary classifier …", flush=True)
    c_long, clean_long = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step1 LONG")
    w_long_train = _binary_recency_weights(t[train_mask])
    w_long_cal = _binary_recency_weights(t[cal_mask])
    d_long_train = lgb.Dataset(X_train, label=y_long_train, weight=w_long_train, feature_name=all_bo_feats, free_raw_data=False)
    d_long_cal = lgb.Dataset(X_cal, label=y_long_cal, weight=w_long_cal, feature_name=all_bo_feats, free_raw_data=False)
    long_spw = _binary_scale_pos_weight(y_long_train)
    print(f"    [L2b gate] LONG pos_rate={y_long_train.mean():.3%} scale_pos_weight={long_spw:.2f}")

    common_params_long = common_params.copy()
    common_params_long.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": long_spw,
    })

    try:
        step1_long_model = lgb.train(
            common_params_long, d_long_train, num_boost_round=rounds, valid_sets=[d_long_cal], callbacks=c_long,
        )
    finally:
        for fn in clean_long:
            fn()

    print("  [L2b train] Dedicated Step1 SHORT gate binary classifier …", flush=True)
    c_short, clean_short = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step1 SHORT")
    w_short_train = _binary_recency_weights(t[train_mask])
    w_short_cal = _binary_recency_weights(t[cal_mask])
    d_short_train = lgb.Dataset(X_train, label=y_short_train, weight=w_short_train, feature_name=all_bo_feats, free_raw_data=False)
    d_short_cal = lgb.Dataset(X_cal, label=y_short_cal, weight=w_short_cal, feature_name=all_bo_feats, free_raw_data=False)
    short_spw = _binary_scale_pos_weight(y_short_train)
    print(f"    [L2b gate] SHORT pos_rate={y_short_train.mean():.3%} scale_pos_weight={short_spw:.2f}")

    common_params_short = common_params.copy()
    common_params_short.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": short_spw,
    })

    try:
        step1_short_model = lgb.train(
            common_params_short, d_short_train, num_boost_round=rounds, valid_sets=[d_short_cal], callbacks=c_short,
        )
    finally:
        for fn in clean_short:
            fn()
    
    rp_cal = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[cal_mask]
    p_long_cal = step1_long_model.predict(X_cal)
    p_short_cal = step1_short_model.predict(X_cal)
    
    # 1. Range Regime Penalty: heavily slash probabilities in range regimes
    p_range_mass_cal = rp_cal[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_long_cal = p_long_cal * (1.0 - 0.7 * p_range_mass_cal)
    p_short_cal = p_short_cal * (1.0 - 0.7 * p_range_mass_cal)
    
    # 2. Threshold Search: Prioritize Precision via F0.5 score + strict Precision floor >= 10%
    best_f05_l, best_thr_l = 0.0, 0.7
    for thr_c in np.arange(0.08, 0.95, 0.02):
        pr = (p_long_cal >= thr_c).astype(int)
        f05 = fbeta_score(y_long_cal, pr, beta=0.5, zero_division=0)
        prec = precision_score(y_long_cal, pr, zero_division=0)
        # Relaxed precision floor from 10% to 5% to force SHORT/LONG discovery at the cost of lower hit rate
        if f05 > best_f05_l and prec >= 0.05:
            best_f05_l, best_thr_l = f05, thr_c
    thr_long_cal = best_thr_l if best_f05_l > 0 else 0.85

    best_f05_s, best_thr_s = 0.0, 0.7
    for thr_c in np.arange(0.08, 0.95, 0.02):
        pr = (p_short_cal >= thr_c).astype(int)
        f05 = fbeta_score(y_short_cal, pr, beta=0.5, zero_division=0)
        prec = precision_score(y_short_cal, pr, zero_division=0)
        if f05 > best_f05_s and prec >= 0.05:
            best_f05_s, best_thr_s = f05, thr_c
    thr_short_cal = best_thr_s if best_f05_s > 0 else 0.85
    
    tcn_transition_prob_cal = work["tcn_transition_prob"].values.astype(np.float32)[cal_mask] if "tcn_transition_prob" in work.columns else None
    p_long_cal, _ = _apply_cp_skip(rp_cal, p_long_cal, thr_cp, tcn_transition_prob_cal)
    p_short_cal, _ = _apply_cp_skip(rp_cal, p_short_cal, thr_cp, tcn_transition_prob_cal)
    
    pr_l_cal = (p_long_cal >= thr_long_cal).astype(int)
    pr_s_cal = (p_short_cal >= thr_short_cal).astype(int)
    pr_trade_cal = pr_l_cal | pr_s_cal
    
    f1_cal_at_thr = f1_score(y_trade_cal, pr_trade_cal, zero_division=0)
    n_trade_pred_cal = int(pr_trade_cal.sum())
    rec_cal_thr = recall_score(y_trade_cal, pr_trade_cal, zero_division=0)
    prec_cal_thr = precision_score(y_trade_cal, pr_trade_cal, zero_division=0)
    thr_rule_note = f"long_thr={thr_long_cal:.2f} short_thr={thr_short_cal:.2f}"
    print(
        f"  Step1 cal: {thr_rule_note}  "
        f"F1={f1_cal_at_thr:.4f}  recall={rec_cal_thr:.3f}  precision={prec_cal_thr:.4f}  "
        f"n_trade={n_trade_pred_cal:,}/{len(y_trade_cal):,}"
    )

    # Evaluate cascade
    rp_test = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[test_mask]
    rp_route_test = raw_regime[test_mask]
    opp_te = _compute_opportunity_scores(X_test, rp_route_test, reg_models)
    p_long_te = step1_long_model.predict(X_test)
    p_short_te = step1_short_model.predict(X_test)
    
    # Apply identical range penalty to test set
    p_range_mass_test = rp_test[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_long_te = p_long_te * (1.0 - 0.7 * p_range_mass_test)
    p_short_te = p_short_te * (1.0 - 0.7 * p_range_mass_test)
    
    tcn_transition_prob_test = work["tcn_transition_prob"].values.astype(np.float32)[test_mask] if "tcn_transition_prob" in work.columns else None
    p_long_te, skip_cp_test = _apply_cp_skip(rp_test, p_long_te, thr_cp, tcn_transition_prob_test)
    p_short_te, _ = _apply_cp_skip(rp_test, p_short_te, thr_cp, tcn_transition_prob_test)
    print(f"\n  CP Skip rate (Layer 2b soft-mixture test): {skip_cp_test.mean():.2%} (forced p_trade=0)")
    
    y_pred6 = _reconstruct_quality_classes(
        p_long_te, p_short_te, p_range_mass_test, thr_long=thr_long_cal, thr_short=thr_short_cal
    )

    print("\n  Step metrics (test):")
    step1_pred = (p_long_te >= thr_long_cal) | (p_short_te >= thr_short_cal)
    s1_rec = recall_score(y_trade_test, step1_pred, zero_division=0)
    s1_prec = precision_score(y_trade_test, step1_pred, zero_division=0)
    print(
        f"    Step1 LONG/SHORT gates  acc={accuracy_score(y_trade_test, step1_pred):.4f} "
        f"f1={f1_score(y_trade_test, step1_pred, zero_division=0):.4f}  "
        f"recall={s1_rec:.3f}  precision={s1_prec:.4f}"
    )
    
    acc = accuracy_score(y_test6, y_pred6)
    macro_f1 = f1_score(y_test6, y_pred6, average="macro")
    print("\n  Reconstructed 4-class metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test6, y_pred6, target_names=QUALITY_CLASS_ORDER, digits=4))
    cm = confusion_matrix(y_test6, y_pred6, labels=list(range(4)))
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
    step1_long_model.save_model(os.path.join(MODEL_DIR, "trade_gate_long.txt"))
    step1_short_model.save_model(os.path.join(MODEL_DIR, "trade_gate_short.txt"))
    import pickle

    meta = {
        "type": "trade_quality_split_long_short_gates",
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
            "long": float(thr_long_cal), "short": float(thr_short_cal), "cp_alpha": 0.05, "thr_cp": float(thr_cp)
        },
        "step1_calibration": {
            "best_f1_cal": float(f1_cal_at_thr),
            "recall_cal": float(rec_cal_thr),
            "precision_cal": float(prec_cal_thr),
            "n_trade_pred_cal": int(n_trade_pred_cal),
            "threshold_selection_rule": thr_rule_note,
        },
        "model_files": {
            "step1_long": "trade_gate_long.txt",
            "step1_short": "trade_gate_short.txt",
            },
        "position_scale_map": {
            "LONG": 1.00,
            "NEUTRAL": 0.00,
            "CHOP": 0.00,
            "SHORT": -1.00,
        },
        "regression_gate": {
            "groups": list(REGIMES_6),
            "thr_vec": thr_vec.tolist(),
            "soft_opp_threshold": float(soft_opp_thr),
            "gating_mode": "soft_mixture",
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
        f"\n  Models saved → trade_gate_long.txt, trade_gate_short.txt\n"
        f"  (L3 continuous features models saved to l2b_opp_mfe/mae_<regime>.txt)",
    )
    print(f"  Meta saved  → {MODEL_DIR}/trade_quality_meta.pkl")

    model_bundle = {
        "step1_regression": step1_regression_bundle,
        "step1_long": step1_long_model,
        "step1_short": step1_short_model,
        "thresholds": meta["hierarchy_thresholds"],
        "feature_cols": all_bo_feats,
    }
    return model_bundle, meta, imp_df



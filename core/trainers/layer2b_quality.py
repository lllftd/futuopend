from __future__ import annotations

import gc
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.lgbm_utils import _theta_decay_from_bars
from core.trainers.data_prep import *


def _train_one_regime_opp_pair(
    predicted_regime: str,
    argmax_idx: int,
    X_tr: np.ndarray,
    X_ca: np.ndarray,
    st_tr: np.ndarray,
    st_ca: np.ndarray,
    rp_tr: np.ndarray,
    rp_ca: np.ndarray,
    st_tr_pred: np.ndarray,
    st_ca_pred: np.ndarray,
    mfe_tr: np.ndarray,
    mae_tr: np.ndarray,
    mfe_ca: np.ndarray,
    mae_ca: np.ndarray,
    all_bo_feats: list[str],
    base_reg: dict[str, Any],
    reg_rounds: int,
    reg_es: int,
    route_power: float,
    min_route_weight_sum: float,
) -> tuple[str, dict[str, lgb.Booster] | None]:
    """One regime: MFE + MAE heads on full train with soft route weights. Returns (name, models or None)."""
    mtr = st_tr_pred == argmax_idx
    mca = st_ca_pred == argmax_idx
    ntr, nca = int(mtr.sum()), int(mca.sum())
    route_w_tr = np.power(np.clip(rp_tr[:, argmax_idx], 1e-6, 1.0), route_power)
    route_w_ca = np.power(np.clip(rp_ca[:, argmax_idx], 1e-6, 1.0), route_power)
    eff_w_tr = float(route_w_tr.sum())
    eff_w_ca = float(route_w_ca.sum())
    eff_n_tr = int((rp_tr[:, argmax_idx] >= 0.15).sum())
    eff_n_ca = int((rp_ca[:, argmax_idx] >= 0.15).sum())
    train_match = float((st_tr[mtr] == argmax_idx).mean()) if ntr else float("nan")
    cal_match = float((st_ca[mca] == argmax_idx).mean()) if nca else float("nan")
    print(
        f"  [L2b regression] {predicted_regime}: "
        f"train_rows(routed)={ntr:,}  cal_rows(routed)={nca:,}  "
        f"soft_weight_sum(train/cal)={eff_w_tr:.1f}/{eff_w_ca:.1f}  "
        f"effective_rows(train/cal)={eff_n_tr:,}/{eff_n_ca:,}  "
        f"label_match(train/cal)={train_match:.1%}/{cal_match:.1%}",
        flush=True,
    )
    if eff_w_tr < min_route_weight_sum:
        print(
            f"    [warn] {predicted_regime}: too little soft-routed weight "
            f"(sum={eff_w_tr:.1f} < {min_route_weight_sum:.1f}) — skipping pair.",
            flush=True,
        )
        return predicted_regime, None

    X_g = X_tr
    y_mfe_g = mfe_tr
    y_mae_g = mae_tr
    w_base = _opp_regression_sample_weights(y_mfe_g, predicted_regime) * route_w_tr

    if eff_w_ca >= 50.0:
        X_va = X_ca
        y_mfe_va = mfe_ca
        y_mae_va = mae_ca
        route_w_val = route_w_ca
    else:
        tail = min(5000, len(X_g))
        X_va = X_g[-tail:]
        y_mfe_va = y_mfe_g[-tail:]
        y_mae_va = y_mae_g[-tail:]
        route_w_val = route_w_tr[-tail:]

    cb = _lgb_train_callbacks(reg_es)
    d_mfe_tr = lgb.Dataset(X_g, label=y_mfe_g, weight=w_base, feature_name=all_bo_feats, free_raw_data=False)
    d_mfe_va = lgb.Dataset(X_va, label=y_mfe_va, weight=route_w_val, feature_name=all_bo_feats, free_raw_data=False)
    print(f"    [L2b regression] {predicted_regime}: train MFE head …", flush=True)
    m_mfe = lgb.train(
        base_reg, d_mfe_tr, num_boost_round=reg_rounds, valid_sets=[d_mfe_va], callbacks=cb,
    )
    w_mae = _opp_regression_sample_weights(y_mae_g, predicted_regime) * route_w_tr
    d_mae_tr = lgb.Dataset(X_g, label=y_mae_g, weight=w_mae, feature_name=all_bo_feats, free_raw_data=False)
    d_mae_va = lgb.Dataset(X_va, label=y_mae_va, weight=route_w_val, feature_name=all_bo_feats, free_raw_data=False)
    print(f"    [L2b regression] {predicted_regime}: train MAE head …", flush=True)
    m_mae = lgb.train(
        base_reg, d_mae_tr, num_boost_round=reg_rounds, valid_sets=[d_mae_va], callbacks=cb,
    )
    return predicted_regime, {"mfe": m_mfe, "mae": m_mae}


def _resolve_l2b_reg_parallel_workers() -> tuple[int, str]:
    """How many regime heads to train concurrently. Empty / ``auto`` ⇒ ``~sqrt(cpu)``, capped at 6."""
    raw = os.environ.get("L2B_REG_PARALLEL_WORKERS", "").strip()
    n_reg = len(REGIMES_6)
    cpu = max(1, os.cpu_count() or 4)
    if not raw or raw.lower() == "auto":
        w = int(max(1, round(cpu**0.5)))
        w = min(w, n_reg)
        return w, f"auto (cpu={cpu})"
    n = int(raw)
    if n <= 1:
        return 1, "sequential"
    return min(n, n_reg), f"manual={n}"


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

    Parallelism: by default picks a worker count from ``cpu_count`` (roughly ``sqrt(n_cpu)``, max 6).
    Override with ``L2B_REG_PARALLEL_WORKERS``: ``0`` / ``1`` = sequential; a positive int = cap;
    ``auto`` = same as unset.
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
    models: dict[str, dict[str, lgb.Booster]] = {}

    X_tr = X[train_mask]
    X_ca = X[cal_mask]
    st_tr = state_label[train_mask]
    st_ca = state_label[cal_mask]
    rp_tr = np.clip(regime_route_probs[train_mask].astype(np.float64, copy=False), 0.0, None)
    rp_ca = np.clip(regime_route_probs[cal_mask].astype(np.float64, copy=False), 0.0, None)
    st_tr_pred = np.argmax(rp_tr, axis=1).astype(np.int64, copy=False)
    st_ca_pred = np.argmax(rp_ca, axis=1).astype(np.int64, copy=False)
    mfe_tr = mfe[train_mask]
    mae_tr = mae[train_mask]
    mfe_ca = mfe[cal_mask]
    mae_ca = mae[cal_mask]
    route_power = float(os.environ.get("L2B_ROUTE_WEIGHT_POWER", "0.75"))
    min_route_weight_sum = float(os.environ.get("L2B_ROUTE_WEIGHT_MIN_SUM", "600.0"))

    n_par, par_how = _resolve_l2b_reg_parallel_workers()
    base_train = base_reg
    if n_par > 1:
        nj0 = int(base_reg.get("n_jobs", _lgbm_n_jobs()))
        nj = max(1, min(nj0, max(1, (os.cpu_count() or 8) // n_par)))
        base_train = {**base_reg, "n_jobs": nj}
        print(
            f"  [L2b regression] parallel regime training: workers={n_par}  "
            f"LightGBM n_jobs per head={nj}  ({par_how}; sequential: L2B_REG_PARALLEL_WORKERS=1)",
            flush=True,
        )

    def _regime_task(t: tuple[int, str]) -> tuple[str, dict[str, lgb.Booster] | None]:
        argmax_idx, predicted_regime = t
        return _train_one_regime_opp_pair(
            predicted_regime,
            argmax_idx,
            X_tr,
            X_ca,
            st_tr,
            st_ca,
            rp_tr,
            rp_ca,
            st_tr_pred,
            st_ca_pred,
            mfe_tr,
            mae_tr,
            mfe_ca,
            mae_ca,
            all_bo_feats,
            base_train,
            reg_rounds,
            reg_es,
            route_power,
            min_route_weight_sum,
        )

    if n_par <= 1:
        for argmax_idx, predicted_regime in enumerate(REGIMES_6):
            name, pair = _regime_task((argmax_idx, predicted_regime))
            if pair is not None:
                models[name] = pair
    else:
        tasks = [(i, REGIMES_6[i]) for i in range(len(REGIMES_6))]
        with ThreadPoolExecutor(max_workers=n_par) as ex:
            for name, pair in ex.map(_regime_task, tasks):
                if pair is not None:
                    models[name] = pair

    if not models:
        raise RuntimeError("Regime opportunity regression: no group had enough data.")

    # --- A: per-regime regression quality on cal (routed by raw L2a argmax) ---
    print("  [L2b diagnostics] A. Per-regime heads (cal, routed):", flush=True)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = st_ca_pred == argmax_idx
        n_r = int(m.sum())
        if n_r < 10:
            print(f"    {predicted_regime}: skip (n_cal_routed={n_r})", flush=True)
            continue
        X_r = X_ca[m]
        y_mfe_r = mfe_ca[m]
        y_mae_r = mae_ca[m]
        pred_mfe = np.clip(models[predicted_regime]["mfe"].predict(X_r), 0.0, None)
        pred_mae = np.clip(models[predicted_regime]["mae"].predict(X_r), 0.01, None)
        r2_mfe = float(r2_score(y_mfe_r, pred_mfe)) if n_r > 2 else float("nan")
        r2_mae = float(r2_score(y_mae_r, pred_mae)) if n_r > 2 else float("nan")
        print(
            f"    {predicted_regime}: MFE R²={r2_mfe:.3f}, MAE R²={r2_mae:.3f}  (n={n_r:,})",
            flush=True,
        )
        print(
            f"      MFE pred [{pred_mfe.min():.4f}, {pred_mfe.max():.4f}]  "
            f"true [{y_mfe_r.min():.4f}, {y_mfe_r.max():.4f}]",
            flush=True,
        )
        print(
            f"      MAE pred [{pred_mae.min():.4f}, {pred_mae.max():.4f}]  "
            f"true [{y_mae_r.min():.4f}, {y_mae_r.max():.4f}]",
            flush=True,
        )
        t_spread = float(y_mfe_r.max() - y_mfe_r.min())
        p_spread = float(pred_mfe.max() - pred_mfe.min())
        if t_spread > 1e-6:
            print(
                f"      MFE spread ratio pred/true={p_spread / t_spread:.3f} "
                f"(<<1 often means compressed preds)",
                flush=True,
            )

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


def _compute_opportunity_triplet_with_regime_opp(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
    *,
    tqdm_regime_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soft-mixture triplet plus per-regime opportunity matrix for directional features."""
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    opp_regime = np.zeros((n, len(REGIMES_6)), dtype=np.float64)
    if n == 0 or not models:
        return opp, mfe_p, mae_p, opp_regime

    available = _available_l2b_regime_indices(models)
    if not available:
        return opp, mfe_p, mae_p, opp_regime

    mfe_stack: list[np.ndarray] = []
    mae_stack: list[np.ndarray] = []
    regime_iter = available
    if tqdm_regime_desc:
        regime_iter = _tq(
            available,
            desc=tqdm_regime_desc,
            total=len(available),
            unit="regime",
            leave=False,
        )
    for idx in regime_iter:
        regime = REGIMES_6[idx]
        mf = np.clip(models[regime]["mfe"].predict(X), 0.0, None)
        ma = np.clip(models[regime]["mae"].predict(X), 0.01, None)
        mfe_stack.append(mf)
        mae_stack.append(ma)

    weights = np.clip(regime_probs[:, available].astype(np.float64, copy=False), 0.0, None)
    if weights.shape[1] == 0:
        return opp, mfe_p, mae_p, opp_regime
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        denom,
        out=np.full_like(weights, 1.0 / weights.shape[1]),
        where=denom > 1e-12,
    )
    mfe_mat = np.column_stack(mfe_stack)
    mae_mat = np.column_stack(mae_stack)
    opp_mat = np.log1p(mfe_mat) - np.log1p(mae_mat)
    opp_regime[:, available] = opp_mat
    mfe_p = np.sum(weights * mfe_mat, axis=1)
    mae_p = np.sum(weights * mae_mat, axis=1)
    opp = np.log1p(mfe_p) - np.log1p(mae_p)
    return opp, mfe_p, mae_p, opp_regime


def _compute_opportunity_triplet(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
    *,
    tqdm_regime_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft mixture over regime experts: expected MFE/MAE then log opportunity."""
    opp, mfe_p, mae_p, _ = _compute_opportunity_triplet_with_regime_opp(
        X, regime_probs, models, tqdm_regime_desc=tqdm_regime_desc
    )
    return opp, mfe_p, mae_p


def _compute_opportunity_scores(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> np.ndarray:
    o, _, _ = _compute_opportunity_triplet(X, regime_probs, models)
    return o


def _regime_context_from_probs(regime_probs: np.ndarray) -> dict[str, np.ndarray]:
    rp = np.clip(regime_probs.astype(np.float64, copy=False), 0.0, None)
    denom = rp.sum(axis=1, keepdims=True)
    rp = np.divide(
        rp,
        denom,
        out=np.full_like(rp, 1.0 / max(rp.shape[1], 1)),
        where=denom > 1e-12,
    )
    bull_mass = rp[:, 0] + rp[:, 1]
    bear_mass = rp[:, 2] + rp[:, 3]
    range_mass = rp[:, 4] + rp[:, 5]
    conf = rp.max(axis=1)
    rp_sorted = np.sort(rp, axis=1)
    margin = rp_sorted[:, -1] - rp_sorted[:, -2]
    entropy = -np.sum(rp * np.log(np.clip(rp, 1e-12, None)), axis=1)
    return {
        "bull_mass": bull_mass,
        "bear_mass": bear_mass,
        "range_mass": range_mass,
        "conf": conf,
        "margin": margin,
        "entropy": entropy,
    }


def _attach_l2b_regime_context_features(work: pd.DataFrame, regime_probs: np.ndarray) -> None:
    ctx = _regime_context_from_probs(regime_probs)
    work["l2b_bull_mass"] = ctx["bull_mass"]
    work["l2b_bear_mass"] = ctx["bear_mass"]
    work["l2b_range_mass"] = ctx["range_mass"]
    work["l2b_regime_margin"] = ctx["margin"]
    work["l2b_regime_entropy"] = ctx["entropy"]


def _build_l2b_directional_feature_matrix(
    regime_probs: np.ndarray,
    opp: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    regime_opp: np.ndarray | None = None,
    tcn_transition_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    rp = np.clip(regime_probs.astype(np.float64, copy=False), 0.0, None)
    denom = rp.sum(axis=1, keepdims=True)
    rp = np.divide(
        rp,
        denom,
        out=np.full_like(rp, 1.0 / max(rp.shape[1], 1)),
        where=denom > 1e-12,
    )
    ctx = _regime_context_from_probs(rp)
    opp64 = opp.astype(np.float64, copy=False)
    mfe64 = mfe.astype(np.float64, copy=False)
    mae64 = mae.astype(np.float64, copy=False)
    mfe_mae_gap = mfe64 - mae64
    mfe_mae_ratio = np.divide(mfe64, np.maximum(mae64, 0.05))
    bull_edge = opp64 * ctx["bull_mass"]
    bear_edge = opp64 * ctx["bear_mass"]
    range_drag = opp64 * ctx["range_mass"]
    opp_x_regime = opp64[:, None] * rp
    if regime_opp is None:
        regime_opp64 = np.repeat(opp64.reshape(-1, 1), rp.shape[1], axis=1)
    else:
        regime_opp64 = np.asarray(regime_opp, dtype=np.float64)
    opp_best = regime_opp64.max(axis=1)
    opp_worst = regime_opp64.min(axis=1)
    opp_std = regime_opp64.std(axis=1)
    opp_range = opp_best - opp_worst
    top2_idx = np.argsort(rp, axis=1)[:, -2:]
    top2_weights = np.take_along_axis(rp, top2_idx, axis=1)
    top2_opp = np.take_along_axis(regime_opp64, top2_idx, axis=1)
    top2_denom = np.maximum(top2_weights.sum(axis=1), 1e-12)
    opp_top2_weighted = np.sum(top2_weights * top2_opp, axis=1) / top2_denom
    conf_x_opp = ctx["conf"] * opp64

    blocks = [
        opp64.reshape(-1, 1),
        mfe64.reshape(-1, 1),
        mae64.reshape(-1, 1),
        mfe_mae_gap.reshape(-1, 1),
        mfe_mae_ratio.reshape(-1, 1),
        bull_edge.reshape(-1, 1),
        bear_edge.reshape(-1, 1),
        range_drag.reshape(-1, 1),
        ctx["conf"].reshape(-1, 1),
        ctx["bull_mass"].reshape(-1, 1),
        ctx["bear_mass"].reshape(-1, 1),
        ctx["range_mass"].reshape(-1, 1),
        ctx["margin"].reshape(-1, 1),
        ctx["entropy"].reshape(-1, 1),
        opp_best.reshape(-1, 1),
        opp_worst.reshape(-1, 1),
        opp_std.reshape(-1, 1),
        opp_range.reshape(-1, 1),
        opp_top2_weighted.reshape(-1, 1),
        conf_x_opp.reshape(-1, 1),
        rp,
        opp_x_regime,
        regime_opp64,
    ]
    cols = (
        list(L2B_DIRECTIONAL_BASE_COLS)
        + list(L2B_OPP_SUMMARY_COLS)
        + list(REGIME_NOW_PROB_COLS)
        + list(L2B_OPP_X_REGIME_COLS)
        + list(L2B_PER_REGIME_OPP_COLS)
    )
    if tcn_transition_prob is not None:
        blocks.append(np.asarray(tcn_transition_prob, dtype=np.float64).reshape(-1, 1))
        cols.append("tcn_transition_prob")
    return np.hstack(blocks).astype(np.float32, copy=False), cols


def _align_feature_matrix_to_names(
    X: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
) -> np.ndarray:
    if feature_names == target_names:
        return X
    pos = {name: idx for idx, name in enumerate(feature_names)}
    out = np.zeros((len(X), len(target_names)), dtype=np.float32)
    for j, name in enumerate(target_names):
        idx = pos.get(name)
        if idx is not None:
            out[:, j] = X[:, idx]
    return out


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


def _compose_two_stage_gate_probs(
    p_trade: np.ndarray,
    p_long_given_trade: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose Stage1(has_trade) and Stage2(direction) into LONG/SHORT gate probabilities."""
    pt = np.clip(np.asarray(p_trade, dtype=np.float64), 0.0, 1.0)
    pd = np.clip(np.asarray(p_long_given_trade, dtype=np.float64), 0.0, 1.0)
    p_long = pt * pd
    p_short = pt * (1.0 - pd)
    return p_long, p_short


def _build_l2b_gate_meta_features(
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compact gate-derived features for Layer 3."""
    pl = np.clip(np.asarray(p_long_gate, dtype=np.float64), 0.0, 1.0)
    ps = np.clip(np.asarray(p_short_gate, dtype=np.float64), 0.0, 1.0)
    spread = pl - ps
    gate_max = np.maximum(pl, ps)
    denom = np.maximum(pl + ps, 1e-9)
    p_dir = np.clip(pl / denom, 1e-9, 1.0 - 1e-9)
    entropy = -(p_dir * np.log(p_dir) + (1.0 - p_dir) * np.log(1.0 - p_dir))
    blk = np.hstack([
        pl.reshape(-1, 1),
        ps.reshape(-1, 1),
        spread.reshape(-1, 1),
        gate_max.reshape(-1, 1),
        entropy.reshape(-1, 1),
    ]).astype(np.float32, copy=False)
    return blk, list(L2B_META_GATE_ALL_COLS)


def _build_l3_interaction_features(
    opp: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Small set of explicit L3 interactions between gate confidence and L2b economics."""
    opp64 = np.asarray(opp, dtype=np.float64)
    mfe64 = np.asarray(mfe, dtype=np.float64)
    mae64 = np.asarray(mae, dtype=np.float64)
    pl = np.clip(np.asarray(p_long_gate, dtype=np.float64), 0.0, 1.0)
    ps = np.clip(np.asarray(p_short_gate, dtype=np.float64), 0.0, 1.0)
    gate_max = np.maximum(pl, ps)
    gate_spread = pl - ps
    rr_ratio = np.divide(mfe64, np.maximum(mae64, 0.05))
    gate_x_mfe = gate_max * mfe64
    gate_x_rr = gate_max * rr_ratio
    gate_spread_x_opp = gate_spread * opp64
    signal_agree = ((gate_max >= 0.20) & (rr_ratio >= 1.5)).astype(np.float64)
    blk = np.hstack([
        gate_x_mfe.reshape(-1, 1),
        gate_x_rr.reshape(-1, 1),
        gate_spread_x_opp.reshape(-1, 1),
        signal_agree.reshape(-1, 1),
    ]).astype(np.float32, copy=False)
    return blk, list(L3_INTERACTION_FEATURE_COLS)


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
    s1_trade = trade_quality_models.get("step1_has_trade")
    s2_dir = trade_quality_models.get("step2_direction")
    regb = trade_quality_models.get("step1_regression")

    regime_mat = work[list(REGIME_NOW_PROB_COLS)].to_numpy(dtype=np.float32, copy=False)
    models = _l2b_nested_opp_models(regb) if regb else {}
    thr_vec = regb["thr_vec"] if regb else None
    gate_feature_cols = trade_quality_models.get("gate_feature_cols")
    has_two_stage = s1_trade is not None and s2_dir is not None
    has_legacy = s1_long is not None and s1_short is not None
    tcn_full = work["tcn_transition_prob"].to_numpy(dtype=np.float32, copy=False) if "tcn_transition_prob" in work.columns else None

    n = len(work)
    n_chunk = (n + chunk - 1) // chunk
    for i in _tq(range(0, n, chunk), desc=tqdm_desc, total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
        rp = regime_mat[i:j]
        tcn_chunk = tcn_full[i:j] if tcn_full is not None else None

        if has_two_stage or has_legacy:
            target_gate_cols = gate_feature_cols
            if not target_gate_cols:
                model_for_schema = s1_trade if has_two_stage else s1_long
                target_gate_cols = [str(c) for c in model_for_schema.feature_name()]
            if set(target_gate_cols).issubset(set(layer2_feats)):
                x_gate = _align_feature_matrix_to_names(x_b, layer2_feats, target_gate_cols)
            else:
                opp, mfe_p, mae_p, opp_regime = _compute_opportunity_triplet_with_regime_opp(x_b, rp, models)
                x_gate_full, x_gate_cols = _build_l2b_directional_feature_matrix(
                    rp, opp, mfe_p, mae_p, opp_regime, tcn_chunk
                )
                x_gate = _align_feature_matrix_to_names(x_gate_full, x_gate_cols, target_gate_cols)
            if s1_trade is not None and s2_dir is not None:
                pt = s1_trade.predict(x_gate)
                pd = s2_dir.predict(x_gate)
                pl, ps = _compose_two_stage_gate_probs(pt, pd)
            else:
                pl = s1_long.predict(x_gate)
                ps = s1_short.predict(x_gate)
            p_rm = rp[:, RANGE_REGIME_INDICES].sum(axis=1)
            p_long_gate[i:j] = pl * (1.0 - 0.7 * p_rm)
            p_short_gate[i:j] = ps * (1.0 - 0.7 * p_rm)
        else:
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
    """Mean MFE/ATR, MAE/ATR, RR by trade-quality class."""
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


def _quality_label_rule_params() -> dict[str, Any]:
    """Economically meaningful label thresholds for Layer 2b."""
    mode = os.environ.get("L2B_LABEL_DIRECTION_MODE", "hybrid_ratio").strip().lower() or "hybrid_ratio"
    if mode not in {"strict", "ratio", "hybrid_ratio"}:
        mode = "hybrid_ratio"
    return {
        "direction_mode": mode,
        "rr_thr": float(os.environ.get("L2B_LABEL_RR_THR", "2.0")),
        "min_mfe": float(os.environ.get("L2B_LABEL_MIN_MFE", "0.5")),
        "ratio_thr": float(os.environ.get("L2B_LABEL_RATIO_THR", "1.20")),
        "ratio_min_mfe": float(os.environ.get("L2B_LABEL_RATIO_MIN_MFE", "0.20")),
        "weak_rr_thr": float(os.environ.get("L2B_LABEL_WEAK_RR_THR", "1.25")),
        "weak_min_mfe": float(os.environ.get("L2B_LABEL_WEAK_MIN_MFE", "0.35")),
        "weak_ratio_thr": float(os.environ.get("L2B_LABEL_WEAK_RATIO_THR", "1.10")),
        "weak_ratio_min_mfe": float(os.environ.get("L2B_LABEL_WEAK_RATIO_MIN_MFE", "0.15")),
    }


def _trade_quality_rule_state(df: pd.DataFrame) -> dict[str, Any]:
    """Precompute label-rule metrics and masks so training and diagnostics stay consistent."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    hold_time = np.maximum(df["exit_bar"].fillna(0).values.astype(float), 0.0)

    # Penalize slow favorable moves so labels reflect theta-sensitive intraday options.
    gamma_decay = _theta_decay_from_bars(hold_time)
    mfe = mfe * gamma_decay
    rr = mfe / np.maximum(mae, 0.1)
    ratio = mfe / np.maximum(mae, 0.1)

    qbull = df["quality_bull_breakout"].fillna(0).values.astype(int)
    qbear = df["quality_bear_breakout"].fillna(0).values.astype(int)
    state = df["state_label"].fillna(2).values.astype(int)
    params = _quality_label_rule_params()

    both_breakout = (qbull == 1) & (qbear == 1)
    long_mask = (qbull == 1) & (qbear == 0)
    short_mask = (qbear == 1) & (qbull == 0)
    no_breakout = (qbull == 0) & (qbear == 0)
    bull_state = np.isin(state, (0, 1))
    bear_state = np.isin(state, (2, 3))
    range_state = np.isin(state, (4, 5))
    directional_chop = mae > mfe

    breakout_trade_strict = (rr >= params["rr_thr"]) & (mfe >= params["min_mfe"])
    breakout_trade_ratio = (ratio >= params["ratio_thr"]) & (mfe >= params["ratio_min_mfe"])
    weak_trade_strict = (rr >= params["weak_rr_thr"]) & (mfe >= params["weak_min_mfe"])
    weak_trade_ratio = (ratio >= params["weak_ratio_thr"]) & (mfe >= params["weak_ratio_min_mfe"])

    mode = str(params["direction_mode"])
    if mode == "strict":
        breakout_trade_final = breakout_trade_strict
        weak_trade_final = weak_trade_strict
    elif mode == "ratio":
        breakout_trade_final = breakout_trade_ratio
        weak_trade_final = weak_trade_ratio
    else:
        breakout_trade_final = breakout_trade_strict | breakout_trade_ratio
        weak_trade_final = weak_trade_strict | weak_trade_ratio

    breakout_long_strict = long_mask & breakout_trade_strict
    breakout_short_strict = short_mask & breakout_trade_strict
    breakout_long_ratio = long_mask & breakout_trade_ratio
    breakout_short_ratio = short_mask & breakout_trade_ratio
    breakout_long_final = long_mask & breakout_trade_final
    breakout_short_final = short_mask & breakout_trade_final

    trend_long_strict = no_breakout & bull_state & weak_trade_strict
    trend_short_strict = no_breakout & bear_state & weak_trade_strict
    trend_long_ratio = no_breakout & bull_state & weak_trade_ratio
    trend_short_ratio = no_breakout & bear_state & weak_trade_ratio
    trend_long_final = no_breakout & bull_state & weak_trade_final
    trend_short_final = no_breakout & bear_state & weak_trade_final

    strict_trade_mask = breakout_long_strict | breakout_short_strict | trend_long_strict | trend_short_strict
    ratio_trade_mask = breakout_long_ratio | breakout_short_ratio | trend_long_ratio | trend_short_ratio
    final_trade_mask = breakout_long_final | breakout_short_final | trend_long_final | trend_short_final

    breakout_ambiguous = (long_mask | short_mask) & ~(breakout_long_final | breakout_short_final)
    relaxed_only_breakout = (long_mask | short_mask) & breakout_trade_ratio & ~breakout_trade_strict
    relaxed_only_weak = no_breakout & weak_trade_ratio & ~weak_trade_strict & (bull_state | bear_state)

    return {
        "params": params,
        "mfe": mfe,
        "mae": mae,
        "rr": rr,
        "ratio": ratio,
        "qbull": qbull,
        "qbear": qbear,
        "state": state,
        "both_breakout": both_breakout,
        "long_mask": long_mask,
        "short_mask": short_mask,
        "no_breakout": no_breakout,
        "range_state": range_state,
        "directional_chop": directional_chop,
        "breakout_long_strict": breakout_long_strict,
        "breakout_short_strict": breakout_short_strict,
        "breakout_long_ratio": breakout_long_ratio,
        "breakout_short_ratio": breakout_short_ratio,
        "breakout_long_final": breakout_long_final,
        "breakout_short_final": breakout_short_final,
        "trend_long_strict": trend_long_strict,
        "trend_short_strict": trend_short_strict,
        "trend_long_ratio": trend_long_ratio,
        "trend_short_ratio": trend_short_ratio,
        "trend_long_final": trend_long_final,
        "trend_short_final": trend_short_final,
        "strict_trade_mask": strict_trade_mask,
        "ratio_trade_mask": ratio_trade_mask,
        "final_trade_mask": final_trade_mask,
        "breakout_ambiguous": breakout_ambiguous,
        "relaxed_only_breakout": relaxed_only_breakout,
        "relaxed_only_weak": relaxed_only_weak,
    }


def _build_trade_quality_targets(df: pd.DataFrame) -> np.ndarray:
    """
    Build 4-class trade-quality labels from deterministic MFE/MAE economics.

    LONG / SHORT:
      directional breakout (or weak directional rescue) with sufficient
      risk-reward and absolute favorable excursion.
    CHOP:
      ambiguous direction or adverse move dominates favorable move.
    NEUTRAL:
      everything else.
    """
    rule = _trade_quality_rule_state(df)
    params = rule["params"]
    both_breakout = rule["both_breakout"]
    long_mask = rule["long_mask"]
    short_mask = rule["short_mask"]
    breakout_ambiguous = rule["breakout_ambiguous"]
    directional_chop = rule["directional_chop"]
    no_breakout = rule["no_breakout"]
    range_state = rule["range_state"]

    y = np.full(len(df), 1, dtype=int)  # default NEUTRAL

    y[both_breakout] = 2  # CHOP
    print(
        "  [L2b labels] directional setup — "
        f"mode={params['direction_mode']}  "
        f"rr_thr={params['rr_thr']:.2f}  min_mfe={params['min_mfe']:.2f}  "
        f"ratio_thr={params['ratio_thr']:.2f}  ratio_min_mfe={params['ratio_min_mfe']:.2f}  "
        f"weak_rr_thr={params['weak_rr_thr']:.2f}  weak_min_mfe={params['weak_min_mfe']:.2f}  "
        f"weak_ratio_thr={params['weak_ratio_thr']:.2f}  weak_ratio_min_mfe={params['weak_ratio_min_mfe']:.2f}  "
        f"long={int(long_mask.sum()):,} short={int(short_mask.sum()):,} rows …",
        flush=True,
    )

    y[rule["breakout_long_final"]] = 0
    y[rule["breakout_short_final"]] = 3
    y[breakout_ambiguous & directional_chop] = 2
    y[breakout_ambiguous & ~directional_chop] = 1

    # Non-breakout bars in trend regimes may still offer weak directional quality.
    y[rule["trend_long_final"]] = 0
    y[rule["trend_short_final"]] = 3
    y[no_breakout & range_state] = 2
    y[no_breakout & ~range_state & ~(rule["trend_long_final"] | rule["trend_short_final"])] = 1

    print(
        "  [L2b labels] assignment summary — "
        f"breakout_long={int(rule['breakout_long_final'].sum()):,}  "
        f"breakout_short={int(rule['breakout_short_final'].sum()):,}  "
        f"ambiguous_to_chop={int((breakout_ambiguous & directional_chop).sum()):,}  "
        f"weak_long={int(rule['trend_long_final'].sum()):,}  "
        f"weak_short={int(rule['trend_short_final'].sum()):,}",
        flush=True,
    )
    print(
        "  [L2b labels] funnel wideners — "
        f"strict_trade={int(rule['strict_trade_mask'].sum()):,}  "
        f"relaxed_trade={int(rule['ratio_trade_mask'].sum()):,}  "
        f"final_trade={int(rule['final_trade_mask'].sum()):,}  "
        f"relaxed_only_breakout={int(rule['relaxed_only_breakout'].sum()):,}  "
        f"relaxed_only_weak={int(rule['relaxed_only_weak'].sum()):,}",
        flush=True,
    )
    return y


def _diagnose_l2b_label_funnel(df: pd.DataFrame, y6: np.ndarray, train_end: str) -> None:
    """Print label funnel diagnostics across train/cal/test splits."""
    rule = _trade_quality_rule_state(df)
    params = rule["params"]
    qbull = rule["qbull"]
    qbear = rule["qbear"]
    mfe = rule["mfe"]
    mae = rule["mae"]
    rr = rule["rr"]
    ratio = rule["ratio"]
    t = df["time_key"].values

    split_masks = [
        ("full", np.ones(len(df), dtype=bool)),
        ("train", t < np.datetime64(train_end)),
        ("cal", (t >= np.datetime64(train_end)) & (t < np.datetime64(CAL_END))),
        ("test", (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))),
    ]

    print("\n[L2b label funnel]")
    print(
        "  Directional thresholds: "
        f"mode={params['direction_mode']}  "
        f"rr_thr={params['rr_thr']:.2f}  min_mfe={params['min_mfe']:.2f}  "
        f"ratio_thr={params['ratio_thr']:.2f}  ratio_min_mfe={params['ratio_min_mfe']:.2f}  "
        f"weak_rr_thr={params['weak_rr_thr']:.2f}  weak_min_mfe={params['weak_min_mfe']:.2f}  "
        f"weak_ratio_thr={params['weak_ratio_thr']:.2f}  weak_ratio_min_mfe={params['weak_ratio_min_mfe']:.2f}",
        flush=True,
    )
    for split_name, mask in split_masks:
        n = int(mask.sum())
        if n <= 0:
            continue
        long_breakout = mask & (qbull == 1) & (qbear == 0)
        short_breakout = mask & (qbear == 1) & (qbull == 0)
        both_breakout = mask & (qbull == 1) & (qbear == 1)
        long_label = mask & (y6 == 0)
        short_label = mask & (y6 == 3)
        chop_label = mask & (y6 == 2)
        neutral_label = mask & (y6 == 1)
        print(
            f"  {split_name:>5s}: rows={n:,}  "
            f"LONG={int(long_label.sum()):,} ({long_label.mean():.2%})  "
            f"SHORT={int(short_label.sum()):,} ({short_label.mean():.2%})  "
            f"CHOP={int(chop_label.sum()):,} ({chop_label.mean():.2%})  "
            f"NEUTRAL={int(neutral_label.sum()):,} ({neutral_label.mean():.2%})",
            flush=True,
        )
        print(
            f"        breakouts: long={int(long_breakout.sum()):,}  "
            f"short={int(short_breakout.sum()):,}  both={int(both_breakout.sum()):,}",
            flush=True,
        )
        strict_long = mask & (rule["breakout_long_strict"] | rule["trend_long_strict"])
        strict_short = mask & (rule["breakout_short_strict"] | rule["trend_short_strict"])
        ratio_long = mask & (rule["breakout_long_ratio"] | rule["trend_long_ratio"])
        ratio_short = mask & (rule["breakout_short_ratio"] | rule["trend_short_ratio"])
        final_long = mask & long_label
        final_short = mask & short_label
        strict_trade = strict_long | strict_short
        ratio_trade = ratio_long | ratio_short
        final_trade = mask & np.isin(y6, list(TRADABLE_CLASS_IDS))
        print(
            f"        strict  LONG={strict_long.mean():.2%} SHORT={strict_short.mean():.2%} TRADE={strict_trade.mean():.2%}",
            flush=True,
        )
        print(
            f"        relaxed LONG={ratio_long.mean():.2%} SHORT={ratio_short.mean():.2%} TRADE={ratio_trade.mean():.2%}",
            flush=True,
        )
        print(
            f"        final   LONG={final_long.mean():.2%} SHORT={final_short.mean():.2%} TRADE={final_trade.mean():.2%}",
            flush=True,
        )
        if long_breakout.any():
            strict_conv = strict_long[long_breakout].mean()
            ratio_conv = ratio_long[long_breakout].mean()
            final_conv = final_long[long_breakout].mean()
            print(
                f"        long breakout -> tradable strict/relaxed/final={strict_conv:.2%}/{ratio_conv:.2%}/{final_conv:.2%}",
                flush=True,
            )
        if short_breakout.any():
            strict_conv = strict_short[short_breakout].mean()
            ratio_conv = ratio_short[short_breakout].mean()
            final_conv = final_short[short_breakout].mean()
            print(
                f"        short breakout -> tradable strict/relaxed/final={strict_conv:.2%}/{ratio_conv:.2%}/{final_conv:.2%}",
                flush=True,
            )
        flat_mask = mask & ~final_trade
        if final_trade.any() and flat_mask.any():
            for name, arr in (("mfe", mfe), ("mae", mae), ("rr", rr), ("ratio", ratio)):
                trade_q = np.percentile(arr[final_trade], [50, 75, 90])
                flat_q = np.percentile(arr[flat_mask], [50, 75, 90])
                print(
                    f"        {name}: trade p50/p75/p90={trade_q[0]:.3f}/{trade_q[1]:.3f}/{trade_q[2]:.3f}  "
                    f"flat={flat_q[0]:.3f}/{flat_q[1]:.3f}/{flat_q[2]:.3f}",
                    flush=True,
                )


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


def _regression_quality_bucket(spearman_r: float, r2: float) -> str:
    if np.isfinite(spearman_r) and np.isfinite(r2):
        if spearman_r > 0.30 and r2 > 0.10:
            return "good_regression"
        if 0.15 <= spearman_r <= 0.30 and 0.02 <= r2 <= 0.10:
            return "coarse_ranking_only"
        if spearman_r < 0.15 and r2 < 0.02:
            return "mostly_noise"
    return "mixed_signal"


def _print_regression_quality_report(pred: np.ndarray, actual: np.ndarray, name: str) -> None:
    pred_s = pd.Series(np.asarray(pred, dtype=np.float64))
    actual_s = pd.Series(np.asarray(actual, dtype=np.float64))
    mask = pred_s.notna() & actual_s.notna()
    n = int(mask.sum())
    if n < 10:
        print(f"    {name}: skip (n={n})", flush=True)
        return
    pred_v = pred_s[mask].to_numpy(dtype=np.float64, copy=False)
    actual_v = actual_s[mask].to_numpy(dtype=np.float64, copy=False)
    pearson_r = float(np.corrcoef(pred_v, actual_v)[0, 1]) if n > 2 and np.std(pred_v) > 1e-12 and np.std(actual_v) > 1e-12 else float("nan")
    spearman_r = float(pred_s[mask].corr(actual_s[mask], method="spearman"))
    r2 = float(r2_score(actual_v, pred_v)) if n > 2 else float("nan")
    bucket = _regression_quality_bucket(spearman_r, r2)
    print(
        f"    {name}: Pearson r={pearson_r:.4f}  Spearman rho={spearman_r:.4f}  "
        f"R2={r2:.4f}  n={n:,}  verdict={bucket}",
        flush=True,
    )


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


def _tcn_transition_risk_profile(tcn_transition_prob: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if tcn_transition_prob is None:
        return None
    risk = np.clip(np.asarray(tcn_transition_prob, dtype=np.float64), 0.0, 1.0)
    prob_weight = float(os.environ.get("TCN_RISK_PROB_WEIGHT", "0.85"))
    pos_weight = float(os.environ.get("TCN_RISK_POS_WEIGHT", "0.95"))
    thr_weight = float(os.environ.get("TCN_RISK_THR_WEIGHT", "0.18"))
    min_prob_scale = float(os.environ.get("TCN_RISK_MIN_PROB_SCALE", "0.05"))
    min_pos_scale = float(os.environ.get("TCN_RISK_MIN_POS_SCALE", "0.10"))
    hard_veto_thr = float(os.environ.get("TCN_RISK_HARD_VETO_THRESHOLD", "0.98"))
    mid_thr = float(os.environ.get("TCN_RISK_MID_THRESHOLD", "0.45"))
    high_thr = float(os.environ.get("TCN_RISK_HIGH_THRESHOLD", "0.70"))

    prob_scale = np.clip(1.0 - prob_weight * risk, min_prob_scale, 1.0)
    pos_scale = np.clip(1.0 - pos_weight * risk, min_pos_scale, 1.0)
    thr_add = thr_weight * risk
    is_high = risk >= hard_veto_thr
    is_mid = (risk >= mid_thr) & (risk < high_thr) & ~is_high
    return {
        "risk": risk,
        "is_mid": is_mid,
        "is_high": is_high,
        "prob_scale": prob_scale,
        "pos_scale": pos_scale,
        "thr_add": thr_add,
    }


def _transition_threshold_add(tcn_transition_prob: np.ndarray | None, n: int) -> np.ndarray:
    prof = _tcn_transition_risk_profile(tcn_transition_prob)
    if prof is None:
        return np.zeros(n, dtype=np.float64)
    return prof["thr_add"]


def _apply_cp_skip(
    regime_probs: np.ndarray,
    p_trade: np.ndarray,
    thr_cp: float,
    tcn_transition_prob: np.ndarray = None,
    *,
    side: str = "trade",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply direction-aware CP skip plus continuous TCN risk shrink."""
    rp = np.clip(np.asarray(regime_probs, dtype=np.float64), 0.0, None)
    denom = rp.sum(axis=1, keepdims=True)
    rp = np.divide(
        rp,
        denom,
        out=np.full_like(rp, 1.0 / max(rp.shape[1], 1)),
        where=denom > 1e-12,
    )
    y_set = rp >= thr_cp
    set_size = y_set.sum(axis=1)
    contains_bull = y_set[:, 0] | y_set[:, 1]
    contains_bear = y_set[:, 2] | y_set[:, 3]
    is_conflicting = contains_bull & contains_bear
    ctx = _regime_context_from_probs(rp)
    bull_mass = ctx["bull_mass"]
    bear_mass = ctx["bear_mass"]
    range_mass = ctx["range_mass"]
    conf = ctx["conf"]
    margin = ctx["margin"]
    entropy = ctx["entropy"]
    entropy_skip_thr = float(os.environ.get("L2B_CP_ENTROPY_SKIP_THR", "1.45"))
    margin_skip_thr = float(os.environ.get("L2B_CP_MARGIN_SKIP_THR", "0.08"))
    conf_skip_thr = float(os.environ.get("L2B_CP_CONF_SKIP_THR", "0.34"))
    range_skip_thr = float(os.environ.get("L2B_CP_RANGE_SKIP_THR", "0.60"))
    opp_mass_buffer = float(os.environ.get("L2B_CP_OPPOSITE_BUFFER", "0.03"))
    ambiguous = (entropy >= entropy_skip_thr) | (margin <= margin_skip_thr) | (conf <= conf_skip_thr)
    skip_cp = (set_size >= 3) | is_conflicting | (set_size == 0) | (ambiguous & (range_mass >= range_skip_thr))
    if side == "long":
        opposite_hit = contains_bear | (bear_mass >= bull_mass - opp_mass_buffer)
        skip_cp = skip_cp | (opposite_hit & ambiguous)
    elif side == "short":
        opposite_hit = contains_bull | (bull_mass >= bear_mass - opp_mass_buffer)
        skip_cp = skip_cp | (opposite_hit & ambiguous)
    else:
        skip_cp = skip_cp | (ambiguous & (np.abs(bull_mass - bear_mass) <= 0.05))

    p_trade_adj = p_trade.copy()
    prof = _tcn_transition_risk_profile(tcn_transition_prob)
    if prof is not None:
        p_trade_adj = p_trade_adj * prof["prob_scale"]
        skip_cp = skip_cp | prof["is_high"]
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
    print("  y = trade_quality (deterministic RR outcomes); X excludes regime probabilities")
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
    _diagnose_l2b_label_funnel(work, work["trade_quality_label"].values, TRAIN_END)

    # Breakout-context features are computed for all bars, so NEUTRAL/CHOP are covered too.
    print("  [L2b prep] Breakout / bar-context features …", flush=True)
    work = ensure_structure_context_features(work)

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
    _attach_l2b_regime_context_features(work, cal_regime)
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

    tcn_whitelist = [c for c in TCN_FEATURES_FOR_L2B if c in work.columns]
    if TCN_FEATURES_ENABLED:
        l2b_base_feats = [c for c in feat_cols if (not c.startswith("tcn_")) or (c in tcn_whitelist)]
    else:
        l2b_base_feats = [c for c in feat_cols if not c.startswith("tcn_")]
    all_bo_feats_raw = _unique_cols(
        l2b_base_feats
        + BO_FEAT_COLS
        + sorted(garch_cols)
        + list(REGIME_NOW_PROB_COLS)
        + ["regime_now_conf"]
        + list(L2B_REGIME_CONTEXT_COLS)
    )
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
    print(f"    Regime probs/context: {len(REGIME_NOW_PROB_COLS) + 1 + len(L2B_REGIME_CONTEXT_COLS)}")
    print(f"    TOTAL L2b:  {len(all_bo_feats)}")
    print(
        f"    TCN whitelist active: {TCN_FEATURES_ENABLED}  "
        f"selected={', '.join(tcn_whitelist) if tcn_whitelist else 'none'}"
    )
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
    es_trade = 90 if FAST_TRAIN_MODE else int(os.environ.get("L2B_TRADE_ES", "200"))
    es_dir = 90 if FAST_TRAIN_MODE else 140
    
    reg_models: dict[str, dict[str, lgb.Booster]] = {}
    thr_vec = np.ones(len(REGIMES_6), dtype=np.float64)

    print(
        "  [L2b train] Step1 = 6-regime dual regression (MFE & MAE @ ATR → opportunity) …",
        flush=True,
    )
    mfe_full, mae_full = _mfe_mae_atr_arrays(work)
    st_full = work["state_label"].values.astype(int)
    rp_route_full = cal_regime.astype(np.float32, copy=False)
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

    # --- B: soft-mixture opportunity triplet on train/cal/test (used by meta gates / Layer 3) ---
    rp_train_soft = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[train_mask]
    rp_cal_soft = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[cal_mask]
    rp_test_soft = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[test_mask]
    opp_train_t, mfe_train_t, mae_train_t, opp_regime_train = _compute_opportunity_triplet_with_regime_opp(
        X_train, rp_train_soft, reg_models, tqdm_regime_desc="L2b soft-mix triplet (train)"
    )
    opp_cal_t, mfe_cal_t, mae_cal_t, opp_regime_cal = _compute_opportunity_triplet_with_regime_opp(
        X_cal, rp_cal_soft, reg_models, tqdm_regime_desc="L2b soft-mix triplet (cal)"
    )
    opp_test_t, mfe_test_t, mae_test_t, opp_regime_test = _compute_opportunity_triplet_with_regime_opp(
        X_test, rp_test_soft, reg_models, tqdm_regime_desc="L2b soft-mix triplet (test)"
    )
    y_cal_6 = y6[cal_mask]
    print("  [L2b diagnostics] B. Opportunity triplet (soft mixture, cal):", flush=True)
    print(
        f"    opp=log1p(mfe_mix)-log1p(mae_mix): mean={opp_cal_t.mean():.4f}  "
        f"P95={np.percentile(opp_cal_t, 95):.4f}  min/max=[{opp_cal_t.min():.4f}, {opp_cal_t.max():.4f}]",
        flush=True,
    )
    print(
        f"    mfe_mix: mean={mfe_cal_t.mean():.4f}  mae_mix: mean={mae_cal_t.mean():.4f}",
        flush=True,
    )
    m_l = y_cal_6 == 0
    m_s = y_cal_6 == 3
    if m_l.any():
        ol = opp_cal_t[m_l]
        print(
            f"    LONG-class rows: mean={ol.mean():.4f}  P95={np.percentile(ol, 95):.4f}  n={m_l.sum():,}",
            flush=True,
        )
    if m_s.any():
        os_ = opp_cal_t[m_s]
        print(
            f"    SHORT-class rows: mean={os_.mean():.4f}  P95={np.percentile(os_, 95):.4f}  n={m_s.sum():,}",
            flush=True,
        )
    print("  [L2b diagnostics] B2. Regression quality (cal, soft mixture):", flush=True)
    _print_regression_quality_report(mfe_cal_t, mfe_full[cal_mask], "pred_mfe vs actual_mfe")
    _print_regression_quality_report(mae_cal_t, mae_full[cal_mask], "pred_mae vs actual_mae")

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

    tcn_transition_prob_train = work["tcn_transition_prob"].values.astype(np.float32)[train_mask] if "tcn_transition_prob" in work.columns else None
    tcn_transition_prob_cal = work["tcn_transition_prob"].values.astype(np.float32)[cal_mask] if "tcn_transition_prob" in work.columns else None
    tcn_transition_prob_test = work["tcn_transition_prob"].values.astype(np.float32)[test_mask] if "tcn_transition_prob" in work.columns else None
    thr_add_cal = _transition_threshold_add(tcn_transition_prob_cal, len(y_trade_cal))
    thr_add_test = _transition_threshold_add(tcn_transition_prob_test, len(y_trade_test))
    X_gate_train, gate_feature_cols = _build_l2b_directional_feature_matrix(
        rp_train_soft, opp_train_t, mfe_train_t, mae_train_t, opp_regime_train, tcn_transition_prob_train
    )
    X_gate_cal, gate_feature_cols_cal = _build_l2b_directional_feature_matrix(
        rp_cal_soft, opp_cal_t, mfe_cal_t, mae_cal_t, opp_regime_cal, tcn_transition_prob_cal
    )
    X_gate_test, gate_feature_cols_test = _build_l2b_directional_feature_matrix(
        rp_test_soft, opp_test_t, mfe_test_t, mae_test_t, opp_regime_test, tcn_transition_prob_test
    )
    if gate_feature_cols != gate_feature_cols_cal or gate_feature_cols != gate_feature_cols_test:
        raise ValueError("Layer 2b gate feature schema mismatch across train/cal/test.")

    print("  [L2b train] Stage1 has-trade binary classifier …", flush=True)
    c_trade, clean_trade = _lgb_train_callbacks_with_round_tqdm(es_trade, rounds, "L2b Stage1 TRADE")
    w_trade_train = _binary_recency_weights(t[train_mask])
    w_trade_cal = _binary_recency_weights(t[cal_mask])
    d_trade_train = lgb.Dataset(X_gate_train, label=y_trade_train, weight=w_trade_train, feature_name=gate_feature_cols, free_raw_data=False)
    d_trade_cal = lgb.Dataset(X_gate_cal, label=y_trade_cal, weight=w_trade_cal, feature_name=gate_feature_cols, free_raw_data=False)
    trade_spw = _binary_scale_pos_weight(y_trade_train)
    print(f"    [L2b gate] TRADE pos_rate={y_trade_train.mean():.3%} scale_pos_weight={trade_spw:.2f}")

    common_params_trade = common_params.copy()
    common_params_trade.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": trade_spw,
        "min_child_samples": int(os.environ.get("L2B_TRADE_MIN_CHILD", "24")),
        "lambda_l1": float(os.environ.get("L2B_TRADE_L1", "0.05")),
        "lambda_l2": float(os.environ.get("L2B_TRADE_L2", "0.75")),
    })
    # Meta gate uses gate_feature_cols, not all_bo_feats — regression monotone_constraints length must not be reused.
    common_params_trade.pop("monotone_constraints", None)
    common_params_trade.pop("monotone_constraints_method", None)

    try:
        step1_trade_model = lgb.train(
            common_params_trade, d_trade_train, num_boost_round=rounds, valid_sets=[d_trade_cal], callbacks=c_trade,
        )
    finally:
        for fn in clean_trade:
            fn()

    dir_train_mask = y_trade_train == 1
    dir_cal_mask = y_trade_cal == 1
    if dir_train_mask.sum() < 200:
        raise RuntimeError("Layer 2b direction gate: too few tradable train rows for Stage2.")
    X_dir_train = X_gate_train[dir_train_mask]
    y_dir_train = y_long_train[dir_train_mask]
    w_dir_train = w_trade_train[dir_train_mask]
    if dir_cal_mask.sum() >= 50:
        X_dir_cal = X_gate_cal[dir_cal_mask]
        y_dir_cal = y_long_cal[dir_cal_mask]
        w_dir_cal = w_trade_cal[dir_cal_mask]
    else:
        tail = min(5000, len(X_dir_train))
        X_dir_cal = X_dir_train[-tail:]
        y_dir_cal = y_dir_train[-tail:]
        w_dir_cal = w_dir_train[-tail:]

    print("  [L2b train] Stage2 long-vs-short direction classifier …", flush=True)
    c_dir, clean_dir = _lgb_train_callbacks_with_round_tqdm(es_dir, rounds, "L2b Stage2 DIRECTION")
    d_dir_train = lgb.Dataset(X_dir_train, label=y_dir_train, weight=w_dir_train, feature_name=gate_feature_cols, free_raw_data=False)
    d_dir_cal = lgb.Dataset(X_dir_cal, label=y_dir_cal, weight=w_dir_cal, feature_name=gate_feature_cols, free_raw_data=False)
    long_share = float(y_dir_train.mean()) if len(y_dir_train) else float("nan")
    print(
        f"    [L2b gate] DIRECTION tradable_rows(train/cal)={int(dir_train_mask.sum()):,}/{int(dir_cal_mask.sum()):,}  "
        f"long_share(train)={long_share:.3%}",
        flush=True,
    )

    common_params_dir = common_params.copy()
    common_params_dir.update({
        "objective": "binary",
        "metric": "binary_logloss",
    })
    common_params_dir.pop("monotone_constraints", None)
    common_params_dir.pop("monotone_constraints_method", None)

    try:
        step2_direction_model = lgb.train(
            common_params_dir, d_dir_train, num_boost_round=rounds, valid_sets=[d_dir_cal], callbacks=c_dir,
        )
    finally:
        for fn in clean_dir:
            fn()

    rp_cal = rp_cal_soft
    p_trade_cal_raw = step1_trade_model.predict(X_gate_cal)
    p_dir_long_cal = step2_direction_model.predict(X_gate_cal)
    p_long_cal, p_short_cal = _compose_two_stage_gate_probs(p_trade_cal_raw, p_dir_long_cal)

    # 1. Range Regime Penalty: heavily slash probabilities in range regimes
    p_range_mass_cal = rp_cal[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_long_cal = p_long_cal * (1.0 - 0.7 * p_range_mass_cal)
    p_short_cal = p_short_cal * (1.0 - 0.7 * p_range_mass_cal)
    p_trade_cal = p_trade_cal_raw * (1.0 - 0.7 * p_range_mass_cal)

    # 2. Threshold Search: Prioritize Precision via F0.5 score + strict Precision floor >= 10%
    best_trade_f1, best_trade_thr = 0.0, 0.7
    for thr_c in np.arange(0.08, 0.95, 0.02):
        pr = (p_trade_cal >= (thr_c + thr_add_cal)).astype(int)
        f1v = f1_score(y_trade_cal, pr, zero_division=0)
        prec = precision_score(y_trade_cal, pr, zero_division=0)
        if f1v > best_trade_f1 and prec >= 0.05:
            best_trade_f1, best_trade_thr = f1v, thr_c
    thr_trade_cal = best_trade_thr if best_trade_f1 > 0 else 0.85

    best_f05_l, best_thr_l = 0.0, 0.7
    for thr_c in np.arange(0.08, 0.95, 0.02):
        pr = (p_long_cal >= (thr_c + thr_add_cal)).astype(int)
        f05 = fbeta_score(y_long_cal, pr, beta=0.5, zero_division=0)
        prec = precision_score(y_long_cal, pr, zero_division=0)
        # Relaxed precision floor from 10% to 5% to force SHORT/LONG discovery at the cost of lower hit rate
        if f05 > best_f05_l and prec >= 0.05:
            best_f05_l, best_thr_l = f05, thr_c
    thr_long_cal = best_thr_l if best_f05_l > 0 else 0.85

    best_f05_s, best_thr_s = 0.0, 0.7
    for thr_c in np.arange(0.08, 0.95, 0.02):
        pr = (p_short_cal >= (thr_c + thr_add_cal)).astype(int)
        f05 = fbeta_score(y_short_cal, pr, beta=0.5, zero_division=0)
        prec = precision_score(y_short_cal, pr, zero_division=0)
        if f05 > best_f05_s and prec >= 0.05:
            best_f05_s, best_thr_s = f05, thr_c
    thr_short_cal = best_thr_s if best_f05_s > 0 else 0.85

    p_trade_cal, _ = _apply_cp_skip(rp_cal, p_trade_cal, thr_cp, tcn_transition_prob_cal, side="trade")
    p_long_cal, _ = _apply_cp_skip(rp_cal, p_long_cal, thr_cp, tcn_transition_prob_cal, side="long")
    p_short_cal, _ = _apply_cp_skip(rp_cal, p_short_cal, thr_cp, tcn_transition_prob_cal, side="short")

    def _gate_prob_dist_report(p: np.ndarray, name: str) -> None:
        if len(p) == 0:
            return
        q50, q90, q99 = np.percentile(p, [50, 90, 99])
        print(
            f"  [L2b diagnostics] C. {name}: mean={p.mean():.4f}  p50={q50:.4f}  "
            f"p90={q90:.4f}  p99={q99:.4f}  max={p.max():.4f}",
            flush=True,
        )

    def _gate_thr_band_report(p: np.ndarray, thr: float, name: str, stage_note: str, band: float = 0.02) -> None:
        if len(p) == 0:
            return
        near = np.abs(p - thr) <= band
        below = p < thr - band
        above = p > thr + band
        print(
            f"  [L2b diagnostics] C. {name} gate probs ({stage_note}): "
            f"thr={thr:.3f}  mean={p.mean():.4f}  "
            f"in (thr±{band})={near.mean():.2%}  below={below.mean():.2%}  above={above.mean():.2%}",
            flush=True,
        )

    _gate_prob_dist_report(p_trade_cal_raw, "TRADE raw probs (cal, pre-range)")
    _gate_prob_dist_report(p_trade_cal, "TRADE probs (cal, post range+CP)")
    _gate_thr_band_report(p_trade_cal_raw, thr_trade_cal, "TRADE raw", "cal, pre-range")
    _gate_thr_band_report(p_trade_cal, thr_trade_cal, "TRADE", "cal, post range+CP")
    _gate_thr_band_report(p_long_cal, thr_long_cal, "LONG", "cal, post range+CP")
    _gate_thr_band_report(p_short_cal, thr_short_cal, "SHORT", "cal, post range+CP")

    pr_trade_stage1_cal = (p_trade_cal >= (thr_trade_cal + thr_add_cal)).astype(int)
    pr_l_cal = (p_long_cal >= (thr_long_cal + thr_add_cal)).astype(int)
    pr_s_cal = (p_short_cal >= (thr_short_cal + thr_add_cal)).astype(int)
    pr_trade_cal = pr_l_cal | pr_s_cal

    if dir_cal_mask.any():
        dir_pred_cal = (p_dir_long_cal[dir_cal_mask] >= 0.50).astype(int)
        dir_acc_cal = accuracy_score(y_dir_cal, dir_pred_cal)
        dir_f1_cal = f1_score(y_dir_cal, dir_pred_cal, zero_division=0)
        print(
            f"  [L2b diagnostics] Stage2 direction (cal, tradable only): "
            f"acc={dir_acc_cal:.4f}  f1={dir_f1_cal:.4f}",
            flush=True,
        )

    f1_trade_stage1 = f1_score(y_trade_cal, pr_trade_stage1_cal, zero_division=0)
    rec_trade_stage1 = recall_score(y_trade_cal, pr_trade_stage1_cal, zero_division=0)
    prec_trade_stage1 = precision_score(y_trade_cal, pr_trade_stage1_cal, zero_division=0)
    f1_cal_at_thr = f1_score(y_trade_cal, pr_trade_cal, zero_division=0)
    n_trade_pred_cal = int(pr_trade_cal.sum())
    rec_cal_thr = recall_score(y_trade_cal, pr_trade_cal, zero_division=0)
    prec_cal_thr = precision_score(y_trade_cal, pr_trade_cal, zero_division=0)
    thr_rule_note = (
        f"trade_thr={thr_trade_cal:.2f} "
        f"long_thr={thr_long_cal:.2f} short_thr={thr_short_cal:.2f}"
    )
    print(
        f"  Stage1 cal: trade_thr={thr_trade_cal:.2f}  "
        f"F1={f1_trade_stage1:.4f}  recall={rec_trade_stage1:.3f}  precision={prec_trade_stage1:.4f}",
        flush=True,
    )
    print(
        f"  Step2 cal: {thr_rule_note}  "
        f"F1={f1_cal_at_thr:.4f}  recall={rec_cal_thr:.3f}  precision={prec_cal_thr:.4f}  "
        f"n_trade={n_trade_pred_cal:,}/{len(y_trade_cal):,}"
    )

    # Evaluate cascade
    rp_test = rp_test_soft
    p_trade_te_raw = step1_trade_model.predict(X_gate_test)
    p_dir_long_te = step2_direction_model.predict(X_gate_test)
    p_long_te, p_short_te = _compose_two_stage_gate_probs(p_trade_te_raw, p_dir_long_te)

    # Apply identical range penalty to test set
    p_range_mass_test = rp_test[:, RANGE_REGIME_INDICES].sum(axis=1)
    p_long_te = p_long_te * (1.0 - 0.7 * p_range_mass_test)
    p_short_te = p_short_te * (1.0 - 0.7 * p_range_mass_test)
    p_trade_te = p_trade_te_raw * (1.0 - 0.7 * p_range_mass_test)

    p_trade_te, skip_cp_test_trade = _apply_cp_skip(rp_test, p_trade_te, thr_cp, tcn_transition_prob_test, side="trade")
    p_long_te, skip_cp_test = _apply_cp_skip(rp_test, p_long_te, thr_cp, tcn_transition_prob_test, side="long")
    p_short_te, _ = _apply_cp_skip(rp_test, p_short_te, thr_cp, tcn_transition_prob_test, side="short")
    print(f"\n  CP Skip rate (Layer 2b soft-mixture test): {skip_cp_test.mean():.2%} (forced p_trade=0)")

    print("  [L2b diagnostics] C. Gate probs vs threshold (test, post range+CP; used before 4-class reconstruct):", flush=True)
    _gate_prob_dist_report(p_trade_te_raw, "TRADE raw probs (test, pre-range)")
    _gate_prob_dist_report(p_trade_te, "TRADE probs (test, post range+CP)")
    _gate_thr_band_report(p_trade_te_raw, thr_trade_cal, "TRADE raw", "test, pre-range")
    _gate_thr_band_report(p_trade_te, thr_trade_cal, "TRADE", "test, post range+CP")
    _gate_thr_band_report(p_long_te, thr_long_cal, "LONG", "test, post range+CP")
    _gate_thr_band_report(p_short_te, thr_short_cal, "SHORT", "test, post range+CP")

    y_pred6 = _reconstruct_quality_classes(
        p_long_te,
        p_short_te,
        p_range_mass_test,
        thr_long=(thr_long_cal + thr_add_test),
        thr_short=(thr_short_cal + thr_add_test),
    )

    print("\n  Step metrics (test):")
    stage1_pred = p_trade_te >= (thr_trade_cal + thr_add_test)
    stage1_rec = recall_score(y_trade_test, stage1_pred, zero_division=0)
    stage1_prec = precision_score(y_trade_test, stage1_pred, zero_division=0)
    print(
        f"    Stage1 has-trade gate    acc={accuracy_score(y_trade_test, stage1_pred):.4f} "
        f"f1={f1_score(y_trade_test, stage1_pred, zero_division=0):.4f}  "
        f"recall={stage1_rec:.3f}  precision={stage1_prec:.4f}"
    )
    step1_pred = (p_long_te >= (thr_long_cal + thr_add_test)) | (p_short_te >= (thr_short_cal + thr_add_test))
    s1_rec = recall_score(y_trade_test, step1_pred, zero_division=0)
    s1_prec = precision_score(y_trade_test, step1_pred, zero_division=0)
    print(
        f"    Stage2 LONG/SHORT gates acc={accuracy_score(y_trade_test, step1_pred):.4f} "
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
    step1_trade_model.save_model(os.path.join(MODEL_DIR, "trade_gate_has_trade.txt"))
    step2_direction_model.save_model(os.path.join(MODEL_DIR, "trade_gate_direction.txt"))
    import pickle

    meta = {
        "l2b_schema": 2,
        "type": "trade_quality_two_stage_gate",
        "gate_architecture": "two_stage_trade_direction",
        "class_names": QUALITY_CLASS_NAMES,
        "feature_cols": all_bo_feats,
        "gate_feature_cols": gate_feature_cols,
        "pa_base_feat_cols": pa_base,
        "pa_hmm_feat_cols": pa_hmm,
        "pa_garch_feat_cols": pa_garch,
        "tcn_feat_cols": pa_tcn,
        "bo_feat_cols": BO_FEAT_COLS,
        "regime_prob_cols_layer3": REGIME_PROB_COLS,
        "garch_cols": garch_cols,
        "hierarchy_thresholds": {
            "trade": float(thr_trade_cal),
            "long": float(thr_long_cal),
            "short": float(thr_short_cal),
            "cp_alpha": 0.05,
            "thr_cp": float(thr_cp),
        },
        "step1_calibration": {
            "best_f1_cal": float(f1_cal_at_thr),
            "recall_cal": float(rec_cal_thr),
            "precision_cal": float(prec_cal_thr),
            "n_trade_pred_cal": int(n_trade_pred_cal),
            "stage1_trade_f1_cal": float(f1_trade_stage1),
            "stage1_trade_recall_cal": float(rec_trade_stage1),
            "stage1_trade_precision_cal": float(prec_trade_stage1),
            "threshold_selection_rule": thr_rule_note,
        },
        "model_files": {
            "step1_has_trade": "trade_gate_has_trade.txt",
            "step2_direction": "trade_gate_direction.txt",
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
        f"\n  Models saved → trade_gate_has_trade.txt, trade_gate_direction.txt\n"
        f"  (L3 continuous features models saved to l2b_opp_mfe/mae_<regime>.txt)",
    )
    print(f"  Meta saved  → {MODEL_DIR}/trade_quality_meta.pkl")

    model_bundle = {
        "l2b_schema": int(meta["l2b_schema"]),
        "step1_regression": step1_regression_bundle,
        "step1_has_trade": step1_trade_model,
        "step2_direction": step2_direction_model,
        "thresholds": meta["hierarchy_thresholds"],
        "feature_cols": all_bo_feats,
        "gate_feature_cols": gate_feature_cols,
    }
    return model_bundle, meta, imp_df



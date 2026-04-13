from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.trainers.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1A_MODEL_FILE,
    L1A_REGIME_COLS,
    L1B_META_FILE,
    L2_DIRECTION_FILE,
    L2_ENTRY_REGIME_COLS,
    L2_GATE_FILE,
    L2_MAE_FILE,
    L2_META_FILE,
    L2_MFE_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
    L2_SIZE_FILE,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    TRAIN_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _mfe_mae_atr_arrays,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.trainers.stack_v2_common import build_stack_time_splits, l2_val_start_time, log_label_baseline, save_output_cache
from core.trainers.val_metrics_extra import (
    brier_multiclass,
    ece_multiclass_maxprob,
    pearson_corr,
    regression_degen_flag,
    tail_mae_truth_upper,
)


def _l2_no_direction_gate_bump_cap() -> tuple[float, float]:
    """When direction head is skipped, raise gate threshold by bump (capped) — see meta direction_available."""
    bump = float(os.environ.get("L2_NO_DIRECTION_GATE_BUMP", "0.05"))
    cap = float(os.environ.get("L2_NO_DIRECTION_GATE_CAP", "0.55"))
    return bump, cap


def _l2_early_stopping_rounds_from_env(key: str, fallback: int) -> int:
    """Parse early_stopping_rounds from env; unset/empty uses fallback. Min 1."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return max(1, int(fallback))
    return max(1, int(raw))


# Drop from L2 without retraining upstream layers: failed L1b heads / redundant pairs.
# Override with L2_SKIP_FEATURE_HARD_DROP=1; extend via L2_EXTRA_HARD_DROP=col1,col2
L2_FEATURE_HARD_DROP_DEFAULT = frozenset(
    {
    }
)


def _l2_select_features_for_training(
    X: np.ndarray,
    feature_cols: list[str],
    train_mask: np.ndarray,
    *,
    min_std: float,
    hard_drop: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Remove hard-listed columns, near-constants, and near-duplicate train features."""
    tm = np.asarray(train_mask, dtype=bool)
    if not tm.any():
        raise RuntimeError("L2: empty train_mask for feature selection.")
    xt = X[tm].astype(np.float64, copy=False)
    dropped: list[str] = []
    keep_idx: list[int] = []
    for j, name in enumerate(feature_cols):
        if name in hard_drop:
            dropped.append(f"{name}(hard_drop)")
            continue
        col = xt[:, j]
        sd = float(np.nanstd(col))
        if not np.isfinite(sd) or sd < min_std:
            dropped.append(f"{name}(std={sd:.2e})")
            continue
        keep_idx.append(j)

    corr_thr = float(os.environ.get("L2_MAX_PAIRWISE_CORR", "0.995"))
    if len(keep_idx) >= 2 and corr_thr < 0.999999:
        kept_after_corr: list[int] = []
        kept_cols: list[np.ndarray] = []
        for j in keep_idx:
            col = xt[:, j]
            drop_name = None
            for prev_j, prev_col in zip(kept_after_corr, kept_cols):
                corr = np.corrcoef(col, prev_col)[0, 1]
                if np.isfinite(corr) and abs(float(corr)) >= corr_thr:
                    drop_name = feature_cols[prev_j]
                    break
            if drop_name is not None:
                dropped.append(f"{feature_cols[j]}(corr~{drop_name})")
                continue
            kept_after_corr.append(j)
            kept_cols.append(col)
        keep_idx = kept_after_corr

    keep = [feature_cols[j] for j in keep_idx]
    return keep, dropped


L2_OUTPUT_COLS = [
    "l2_decision_class",
    "l2_decision_long",
    "l2_decision_neutral",
    "l2_decision_short",
    "l2_decision_confidence",
    "l2_size",
    "l2_pred_mfe",
    "l2_pred_mae",
    *L2_ENTRY_REGIME_COLS,
    "l2_entry_vol",
    "l2_expected_edge",
    "l2_rr_proxy",
]


@dataclass
class L2TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _l2_build_two_stage_labels(y_decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_decision, dtype=np.int64).ravel()
    y_gate = (y != 1).astype(np.int64)
    y_dir = np.full(len(y), -1, dtype=np.int64)
    y_dir[y == 0] = 1
    y_dir[y == 2] = 0
    return y_gate, y_dir


def _l2_compose_probs_from_gate_dir(gate_p: np.ndarray, dir_p: np.ndarray) -> np.ndarray:
    """Columns order: long, neutral, short. Sums to 1 row-wise."""
    g = np.clip(np.asarray(gate_p, dtype=np.float64).ravel(), 0.0, 1.0)
    d = np.clip(np.asarray(dir_p, dtype=np.float64).ravel(), 0.0, 1.0)
    p_long = g * d
    p_short = g * (1.0 - d)
    p_neu = 1.0 - g
    return np.column_stack([p_long, p_neu, p_short]).astype(np.float32)


def _l2_direction_margin_to_prob(direction_margin: np.ndarray, *, temperature: float) -> np.ndarray:
    margin = np.asarray(direction_margin, dtype=np.float64).ravel()
    temp = max(float(temperature), 1e-3)
    z = np.clip(margin / temp, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)


def _split_mask_for_tuning_and_report(
    time_key: pd.Series | np.ndarray,
    base_mask: np.ndarray,
    *,
    tune_frac: float,
    min_rows_each: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_mask, dtype=bool)
    idx = np.flatnonzero(base)
    tune_mask = np.zeros_like(base, dtype=bool)
    report_mask = np.zeros_like(base, dtype=bool)
    if idx.size == 0:
        return tune_mask, report_mask
    ts = np.asarray(pd.to_datetime(time_key))
    idx = idx[np.argsort(ts[idx])]
    if idx.size < 2 * min_rows_each:
        tune_mask[idx] = True
        report_mask[idx] = True
        return tune_mask, report_mask
    split = int(round(idx.size * float(np.clip(tune_frac, 0.2, 0.8))))
    split = max(min_rows_each, min(idx.size - min_rows_each, split))
    tune_mask[idx[:split]] = True
    report_mask[idx[split:]] = True
    return tune_mask, report_mask


def _fit_binary_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> IsotonicRegression | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return calib


def _apply_binary_calibrator(p: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)


def _l2_hard_decision_from_gate_dir(gate_p: np.ndarray, dir_p: np.ndarray, thr: float) -> np.ndarray:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    dir_p = np.asarray(dir_p, dtype=np.float64).ravel()
    out = np.ones(len(gate_p), dtype=np.int64)
    trade = gate_p >= thr
    out[trade & (dir_p >= 0.5)] = 0
    out[trade & (dir_p < 0.5)] = 2
    return out


def _l2_predict_gate_dir_probs(
    gate_model: lgb.Booster,
    direction_model: lgb.Booster | None,
    X: np.ndarray,
    *,
    trade_threshold: float,
    direction_head_type: str = "probability",
    direction_temperature: float = 0.35,
    gate_calibrator: IsotonicRegression | None = None,
    gate_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_scores = gate_model.predict(X).astype(np.float64) if gate_raw is None else np.asarray(gate_raw, dtype=np.float64).ravel()
    gate_p = _apply_binary_calibrator(gate_scores, gate_calibrator)
    dir_p = np.full(len(X), 0.5, dtype=np.float64)
    if direction_model is not None:
        m = gate_p >= trade_threshold
        if m.any():
            raw = direction_model.predict(X[m]).astype(np.float64)
            if direction_head_type == "signed_edge_regression":
                dir_p[m] = _l2_direction_margin_to_prob(raw, temperature=direction_temperature)
            else:
                dir_p[m] = raw
    decision_probs = _l2_compose_probs_from_gate_dir(gate_p, dir_p)
    return gate_p.astype(np.float32), dir_p.astype(np.float32), decision_probs


def _search_l2_trade_threshold(
    gate_p: np.ndarray,
    *,
    target_trade_rate: float = 0.10,
    min_trade_rate: float = 0.08,
    max_trade_rate: float = 0.12,
) -> float:
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    if gate_p.size == 0:
        return 0.35
    target = float(np.clip(target_trade_rate, min_trade_rate, max_trade_rate))
    thr = float(np.quantile(gate_p, 1.0 - target))
    realized = float(np.mean(gate_p >= thr))
    print(
        "\n  [L2] trade_threshold search on gate_p only (target trade-rate percentile)",
        flush=True,
    )
    for rate in sorted({min_trade_rate, target, max_trade_rate}):
        cand = float(np.quantile(gate_p, 1.0 - rate))
        cand_rate = float(np.mean(gate_p >= cand))
        mark = "  *" if abs(rate - target) < 1e-9 else ""
        print(f"    target_trade_rate={rate:.3f}  threshold={cand:.4f}  realized={cand_rate:.3f}{mark}", flush=True)
    print(f"  [L2] selected trade_threshold={thr:.4f}  target_trade_rate={target:.3f}  realized={realized:.3f}", flush=True)
    return thr


def _log_l2_two_stage_val_diagnostics(
    gate_p: np.ndarray,
    dir_p: np.ndarray,
    y_gate: np.ndarray,
    y_dir: np.ndarray,
    y_decision: np.ndarray,
    trade_threshold: float,
) -> None:
    y_gate = np.asarray(y_gate, dtype=np.int64).ravel()
    y_decision = np.asarray(y_decision, dtype=np.int64).ravel()
    gate_p = np.asarray(gate_p, dtype=np.float64).ravel()
    dir_p = np.asarray(dir_p, dtype=np.float64).ravel()
    print("\n  [L2] val — two-stage gate", flush=True)
    try:
        auc_g = float(roc_auc_score(y_gate, gate_p))
    except ValueError:
        auc_g = float("nan")
    pred_gate = (gate_p >= trade_threshold).astype(np.int32)
    print(f"    gate AUC={auc_g:.4f}  threshold={trade_threshold:.2f}", flush=True)
    print(
        "    gate classification_report:\n"
        + classification_report(y_gate, pred_gate, target_names=["no_trade", "trade"], digits=4, zero_division=0),
        flush=True,
    )
    if (y_gate == 1).any():
        recall_trade = float(np.mean(pred_gate[y_gate == 1] == 1))
        if recall_trade < 0.15:
            print(f"    WARNING: trade recall={recall_trade:.3f} < 0.15 — consider lowering threshold", flush=True)
    act = y_dir >= 0
    gated = act & (gate_p >= trade_threshold)
    if gated.sum() >= 20:
        print("\n  [L2] val — direction (deploy path: gated active bars)", flush=True)
        yda = y_dir[gated].astype(np.int32)
        dpa = dir_p[gated]
        try:
            auc_d = float(roc_auc_score(yda, dpa))
        except ValueError:
            auc_d = float("nan")
        pred_d = (dpa >= 0.5).astype(np.int32)
        print(
            f"    direction AUC={auc_d:.4f}  n_gated_active={int(gated.sum()):,}  "
            f"coverage_of_true_active={float(np.mean(gate_p[act] >= trade_threshold)) if act.any() else 0.0:.3f}",
            flush=True,
        )
        print(
            "    direction classification_report:\n"
            + classification_report(yda, pred_d, target_names=["short", "long"], digits=4, zero_division=0),
            flush=True,
        )
    elif act.sum() >= 20:
        print(
            f"\n  [L2] val — direction: gated subset too small for deploy-path diagnostics "
            f"(gated_active={int(gated.sum())}, true_active={int(act.sum())})",
            flush=True,
        )
    pred_hard = _l2_hard_decision_from_gate_dir(gate_p, dir_p, trade_threshold)
    print(
        f"\n  [L2] val — hard two-stage vs truth: pred_active={float(np.mean(pred_hard != 1)):.3f}  "
        f"true_active={float(np.mean(y_decision != 1)):.3f}",
        flush=True,
    )


def _log_l2_extended_val_metrics(
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    decision_probs: np.ndarray,
    y_size: np.ndarray,
    size_pred: np.ndarray,
    y_mfe: np.ndarray,
    mfe_pred: np.ndarray,
    y_mae: np.ndarray,
    mae_pred: np.ndarray,
) -> None:
    """Extra val diagnostics: multiclass Brier/ECE, lift, L1 entropy↔L2 conf, regression tails & degen."""
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return
    yv = y_decision[vm]
    Pv = np.asarray(decision_probs[vm], dtype=np.float64)
    Pv = np.clip(Pv, 1e-15, 1.0)
    Pv = Pv / Pv.sum(axis=1, keepdims=True)
    try:
        ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
    except ValueError:
        ll = float("nan")
    br = brier_multiclass(yv, Pv, 3)
    ece = ece_multiclass_maxprob(yv, Pv)
    pred = np.argmax(Pv, axis=1)
    acc = float(accuracy_score(yv, pred))
    f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
    cm = confusion_matrix(yv, pred, labels=[0, 1, 2])
    conf = np.max(Pv, axis=1)
    f1w = float(f1_score(yv, pred, average="weighted", zero_division=0))
    print("\n  [L2] val — decision (extended)", flush=True)
    print(
        f"    log_loss={ll:.4f}  Brier={br:.4f}  ECE(max-prob)={ece:.4f}  acc={acc:.4f}  F1_macro={f1m:.4f}  F1_weighted={f1w:.4f}",
        flush=True,
    )
    print(f"    confusion [rows=true 0/1/2, cols=pred]:\n    {cm}", flush=True)
    print(
        "    classification report:\n"
        + classification_report(
            yv,
            pred,
            labels=[0, 1, 2],
            target_names=["long", "neutral", "short"],
            digits=4,
            zero_division=0,
        ),
        flush=True,
    )
    pct_rows = np.percentile(Pv, [5, 25, 50, 75, 95], axis=0)
    print(
        f"    P(long/neutral/short) pct [5,25,50,75,95] rows:\n"
        f"      long:    {pct_rows[:, 0]}\n"
        f"      neutral: {pct_rows[:, 1]}\n"
        f"      short:   {pct_rows[:, 2]}",
        flush=True,
    )
    try:
        dfq = pd.DataFrame({"conf": conf, "ok": (pred == yv).astype(np.float64)})
        dfq["bin"] = pd.qcut(dfq["conf"], 10, duplicates="drop")
        lift = dfq.groupby("bin", observed=True)["ok"].agg(["mean", "count"])
        print(f"    accuracy lift by confidence decile:\n{lift}", flush=True)
    except Exception as ex:
        print(f"    (skip lift table: {ex})", flush=True)
    for cls_idx, cls_name in ((0, "long"), (2, "short")):
        mask = yv == cls_idx
        if not mask.any():
            continue
        recall = float(np.mean(pred[mask] == cls_idx))
        pcts = np.percentile(Pv[mask, cls_idx], [5, 25, 50, 75, 95])
        print(
            f"    {cls_name}: n={int(mask.sum()):,}  recall={recall:.4f}  prob_pcts={np.round(pcts, 4).tolist()}",
            flush=True,
        )
    neutral_pred_rate = float(np.mean(pred == 1))
    if neutral_pred_rate > 0.95:
        print(f"    WARNING: still collapsing toward neutral ({100.0 * neutral_pred_rate:.1f}% predicted neutral)", flush=True)
    elif neutral_pred_rate > 0.85:
        print(f"    WARNING: borderline neutral-heavy predictions ({100.0 * neutral_pred_rate:.1f}%)", flush=True)
    else:
        print(f"    neutral prediction rate={100.0 * neutral_pred_rate:.1f}%", flush=True)
    R = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float64)[vm]
    R = np.clip(R, 1e-12, 1.0)
    R = R / R.sum(axis=1, keepdims=True)
    ent = -np.sum(R * np.log(R), axis=1)
    c_l1_l2 = pearson_corr(ent, conf)
    print(f"    corr(L1 regime entropy, L2 max prob)={c_l1_l2:.4f}", flush=True)

    av = vm & (y_decision != 1)
    if av.sum() >= 5:
        yt = y_size[av].astype(np.float64)
        yp = size_pred[av].astype(np.float64)
        mae_s = float(mean_absolute_error(yt, yp))
        rmse_s = float(np.sqrt(mean_squared_error(yt, yp)))
        r2_s = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
        c_s = pearson_corr(yt, yp)
        std_s, degen_s = regression_degen_flag(yp)
        print("\n  [L2] val — size (active bars only, y_decision≠neutral)", flush=True)
        print(
            f"    MAE={mae_s:.4f}  RMSE={rmse_s:.4f}  R2={r2_s:.4f}  corr={c_s:.4f}  pred_std={std_s:.6f}  degen={degen_s}",
            flush=True,
        )
    else:
        print("\n  [L2] val — size: (skip: too few active val rows)", flush=True)

    for name, yt_a, yp_a in (
        ("mfe", y_mfe, mfe_pred),
        ("mae", y_mae, mae_pred),
    ):
        mask = av
        yt = yt_a[mask].astype(np.float64)
        yp = yp_a[mask].astype(np.float64)
        if yt.size < 5:
            print(f"\n  [L2] val — {name} head: (skip: too few active val rows)", flush=True)
            continue
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
        c = pearson_corr(yt, yp)
        tail = tail_mae_truth_upper(yt, yp, 90.0)
        print(f"\n  [L2] val — {name} head (active bars only)", flush=True)
        print(
            f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  corr={c:.4f}  tail_MAE(P90+)={tail:.4f}",
            flush=True,
        )


def _log_l2_l1b_ablation(
    X: np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    y_decision: np.ndarray,
    gate_model: lgb.Booster,
    direction_model: lgb.Booster | None,
    *,
    trade_threshold: float,
    direction_head_type: str,
    direction_temperature: float,
    gate_calibrator: IsotonicRegression | None,
) -> None:
    l1b_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1b_")]
    if not l1b_idx:
        print("\n  [L2] l1b ablation: skip (no l1b_* columns after selection)", flush=True)
        return
    vm = np.asarray(val_mask, dtype=bool)
    if not vm.any():
        return

    def _eval_probs(prob_mat: np.ndarray) -> tuple[float, float, float]:
        Pv = np.asarray(prob_mat[vm], dtype=np.float64)
        Pv = np.clip(Pv, 1e-15, 1.0)
        Pv = Pv / Pv.sum(axis=1, keepdims=True)
        yv = np.asarray(y_decision[vm], dtype=np.int64)
        try:
            ll = float(log_loss(yv, Pv, labels=[0, 1, 2]))
        except ValueError:
            ll = float("nan")
        pred = np.argmax(Pv, axis=1)
        f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
        trade_rate = float(np.mean(pred != 1))
        return ll, f1m, trade_rate

    _, _, base_probs = _l2_predict_gate_dir_probs(
        gate_model,
        direction_model,
        X,
        trade_threshold=trade_threshold,
        direction_head_type=direction_head_type,
        direction_temperature=direction_temperature,
        gate_calibrator=gate_calibrator,
    )
    X_no_l1b = np.array(X, copy=True)
    X_no_l1b[:, l1b_idx] = 0.0
    _, _, ablated_probs = _l2_predict_gate_dir_probs(
        gate_model,
        direction_model,
        X_no_l1b,
        trade_threshold=trade_threshold,
        direction_head_type=direction_head_type,
        direction_temperature=direction_temperature,
        gate_calibrator=gate_calibrator,
    )
    base_ll, base_f1, base_trade = _eval_probs(base_probs)
    abl_ll, abl_f1, abl_trade = _eval_probs(ablated_probs)
    print("\n  [L2] l1b val ablation (zero l1b_* at inference)", flush=True)
    print(
        f"    baseline:      log_loss={base_ll:.4f}  F1_macro={base_f1:.4f}  trade_rate={base_trade:.3f}",
        flush=True,
    )
    print(
        f"    without_l1b:   log_loss={abl_ll:.4f}  F1_macro={abl_f1:.4f}  trade_rate={abl_trade:.3f}",
        flush=True,
    )
    print(
        f"    delta(no_l1b-base):  log_loss={abl_ll - base_ll:+.4f}  F1_macro={abl_f1 - base_f1:+.4f}  "
        f"trade_rate={abl_trade - base_trade:+.3f}",
        flush=True,
    )


def _session_context(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["time_key"])
    out = pd.DataFrame(index=df.index)
    out["l2_session_progress"] = ((ts.dt.hour * 60 + ts.dt.minute) / (24.0 * 60.0)).astype(np.float32)
    out["l2_is_opening_hour"] = (ts.dt.hour <= 10).astype(np.float32)
    return out


def _l2_target_trade_rate() -> float:
    return float(np.clip(float(os.environ.get("L2_TARGET_TRADE_RATE", "0.10")), 0.08, 0.12))


def _quantile_rescale_01(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[fit]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _residual_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in [
        "pa_ctx_structure_veto",
        "pa_ctx_premise_break_long",
        "pa_ctx_premise_break_short",
        "pa_ctx_range_pressure",
        "bo_wick_imbalance",
        "bo_or_dist",
    ]:
        out[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    out = pd.concat([out, _session_context(df)], axis=1)
    return out


def _build_l2_frame(df: pd.DataFrame, l1a_outputs: pd.DataFrame, l1b_outputs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    merged = (
        df[["symbol", "time_key"]]
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l1b_outputs, on=["symbol", "time_key"], how="left")
    )
    residual = _residual_feature_frame(df)
    merged = pd.concat([merged.reset_index(drop=True), residual.reset_index(drop=True)], axis=1)
    feature_cols = [
        c
        for c in merged.columns
        if c not in {"symbol", "time_key"}
    ]
    merged[feature_cols] = merged[feature_cols].fillna(0.0).astype(np.float32)
    return merged, feature_cols


def train_l2_trade_decision(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
) -> L2TrainingBundle:
    frame, feature_cols = _build_l2_frame(df, l1a_outputs, l1b_outputs)
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(df["time_key"])
    l2_val_start = l2_val_start_time()

    train_mask = splits.l2_train_mask
    val_mask = splits.l2_val_mask
    test_mask = splits.test_mask
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("L2: calibration split is empty for train/val.")
    tune_frac = float(os.environ.get("L2_TUNE_FRAC_WITHIN_VAL", "0.5"))
    val_tune_mask, val_report_mask = _split_mask_for_tuning_and_report(
        df["time_key"], val_mask, tune_frac=tune_frac, min_rows_each=50
    )
    if not val_tune_mask.any() or not val_report_mask.any():
        raise RuntimeError("L2: failed to create non-empty tuning/report masks inside l2_val.")

    min_std = float(os.environ.get("L2_MIN_FEATURE_STD", "1e-4"))
    skip_hard = os.environ.get("L2_SKIP_FEATURE_HARD_DROP", "").strip().lower() in {"1", "true", "yes"}
    hard_drop = frozenset() if skip_hard else set(L2_FEATURE_HARD_DROP_DEFAULT)
    if not skip_hard:
        _extra = os.environ.get("L2_EXTRA_HARD_DROP", "").strip()
        if _extra:
            hard_drop |= {s.strip() for s in _extra.split(",") if s.strip()}
    hard_drop = frozenset(hard_drop)
    feature_cols, l2_dropped_features = _l2_select_features_for_training(
        X, feature_cols, train_mask, min_std=min_std, hard_drop=hard_drop
    )
    if l2_dropped_features:
        print(
            f"  [L2] feature selection: dropped {len(l2_dropped_features)} cols "
            f"(hard_drop={not skip_hard}, min_train_std={min_std:g})",
            flush=True,
        )
        for line in l2_dropped_features[:35]:
            print(f"       {line}", flush=True)
        if len(l2_dropped_features) > 35:
            print(f"       ... {len(l2_dropped_features) - 35} more", flush=True)
    if not feature_cols:
        raise RuntimeError(
            "L2: all features removed by selection; set L2_MIN_FEATURE_STD lower or "
            "L2_SKIP_FEATURE_HARD_DROP=1."
        )
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)

    edge = _decision_edge_atr_array(df)
    mfe, mae = _mfe_mae_atr_arrays(df)
    tau = 0.05
    y_decision = np.full(len(df), 1, dtype=np.int64)
    y_decision[edge > tau] = 0
    y_decision[edge < -tau] = 2
    rr = np.clip(mfe / np.maximum(mae, 0.10), 0.0, 4.0)
    size_raw = np.clip(np.abs(edge), 0.0, 1.5) * (rr / (1.0 + rr)) * np.exp(-0.35 * np.clip(mae, 0.0, 4.0))
    active_train = train_mask & (y_decision != 1)
    active_val = val_mask & (y_decision != 1)
    y_size = _quantile_rescale_01(size_raw, fit_mask=active_train)
    y_size[y_decision == 1] = 0.0
    y_mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    y_mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    y_dir_margin = np.clip(edge / float(os.environ.get("L2_DIRECTION_EDGE_SCALE", "0.75")), -1.0, 1.0).astype(np.float32)

    log_layer_banner("[L2] Trade decision (LGBM)")
    log_time_key_split(
        "L2",
        df["time_key"],
        train_mask,
        val_mask,
        train_label=f"l2_train [{TRAIN_END}, {str(l2_val_start)})",
        val_label=f"l2_val [{str(l2_val_start)}, {CAL_END})",
        extra_note=(
            f"Strict time split inside cal window: train in [{TRAIN_END}, {str(l2_val_start)}), "
            f"val in [{str(l2_val_start)}, {CAL_END})."
        ),
    )
    log_time_key_split(
        "L2(threshold/calibration)",
        df["time_key"],
        val_tune_mask,
        val_report_mask,
        train_label="val_tune (threshold/calibration)",
        val_label="val_report (metrics)",
        extra_note="L2 thresholds and probability calibration are fit on val_tune; headline validation metrics are reported on val_report.",
    )
    log_numpy_x_stats("L2", X[train_mask], label="X[l2_train]")
    l1a_cols = [c for c in feature_cols if c.startswith("l1a_")]
    l1b_cols = [c for c in feature_cols if c.startswith("l1b_")]
    res_cols = [c for c in feature_cols if c not in l1a_cols and c not in l1b_cols]
    print(
        f"  [L2] feature_cols total={len(feature_cols)} (expect ~51)  "
        f"l1a_*={len(l1a_cols)}  l1b_*={len(l1b_cols)}  residual/other={len(res_cols)}",
        flush=True,
    )
    print(f"  [L2] residual columns (n={len(res_cols)}): {res_cols}", flush=True)
    print(
        f"  [L2] upstream artifact refs: L1a={artifact_path(L1A_MODEL_FILE)}  L1b meta={artifact_path(L1B_META_FILE)}",
        flush=True,
    )
    print(
        "  [L2] note: l1a_*/l1b_* here come from in-memory outputs (full-history forward in this run), not time-OOF caches.",
        flush=True,
    )
    print(f"  [L2] will write: {artifact_path(L2_META_FILE)} | {artifact_path(L2_OUTPUT_CACHE_FILE)}", flush=True)
    log_label_baseline("l2_size", y_size[train_mask & (y_decision != 1)], task="reg")
    log_label_baseline("l2_mfe", y_mfe[train_mask & (y_decision != 1)], task="reg")
    log_label_baseline("l2_mae", y_mae[train_mask & (y_decision != 1)], task="reg")

    rounds = 250 if FAST_TRAIN_MODE else 1200
    # Gate: default 120 — not tied to FAST_TRAIN_MODE (set L2_GATE_EARLY_STOPPING_ROUNDS to override).
    gate_es_rounds = _l2_early_stopping_rounds_from_env("L2_GATE_EARLY_STOPPING_ROUNDS", 120)
    aux_es_fallback = 40 if FAST_TRAIN_MODE else 120
    aux_es_base = _l2_early_stopping_rounds_from_env("L2_EARLY_STOPPING_ROUNDS", aux_es_fallback)
    direction_es_rounds = _l2_early_stopping_rounds_from_env("L2_DIRECTION_EARLY_STOPPING_ROUNDS", aux_es_base)
    size_es_rounds = _l2_early_stopping_rounds_from_env("L2_SIZE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mfe_es_rounds = _l2_early_stopping_rounds_from_env("L2_MFE_EARLY_STOPPING_ROUNDS", aux_es_base)
    mae_es_rounds = _l2_early_stopping_rounds_from_env("L2_MAE_EARLY_STOPPING_ROUNDS", aux_es_base)
    y_gate, y_dir = _l2_build_two_stage_labels(y_decision)
    log_label_baseline("l2_gate", y_gate[train_mask], task="cls")
    mdir = y_dir[train_mask] >= 0
    if mdir.any():
        log_label_baseline("l2_direction_margin", y_dir_margin[train_mask][mdir], task="reg")
    pr_trade = float(np.mean(y_gate[train_mask]))
    gate_lr = float(os.environ.get("L2_GATE_LEARNING_RATE", "0.01"))
    gate_leaves = int(os.environ.get("L2_GATE_NUM_LEAVES", "15"))
    gate_depth = int(os.environ.get("L2_GATE_MAX_DEPTH", "5"))
    gate_mcs = int(os.environ.get("L2_GATE_MIN_CHILD_SAMPLES", "80"))
    gate_bag = float(os.environ.get("L2_GATE_BAGGING_FRACTION", "0.7"))
    gate_bag_freq = int(os.environ.get("L2_GATE_BAGGING_FREQ", "1"))
    gate_params = {
        "objective": "binary",
        # AUC first so early_stopping (first_metric_only) tracks ranking, not logloss under imbalance.
        "metric": ["auc", "binary_logloss"],
        "learning_rate": gate_lr,
        "num_leaves": gate_leaves,
        "max_depth": gate_depth,
        "feature_fraction": float(os.environ.get("L2_GATE_FEATURE_FRACTION", "0.8")),
        "bagging_fraction": gate_bag,
        "bagging_freq": gate_bag_freq,
        "min_child_samples": gate_mcs,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        "is_unbalance": True,
    }
    print(
        f"  [L2] gate: pos_rate={pr_trade:.3f}  is_unbalance=True  lr={gate_lr}  "
        f"num_leaves={gate_leaves}  max_depth={gate_depth}  min_child_samples={gate_mcs}  "
        f"bagging={gate_bag}/{gate_bag_freq}  early_stopping_rounds={gate_es_rounds}  "
        f"early_stop_metric=auc (first)",
        flush=True,
    )
    print(
        f"  [L2] early_stopping_rounds: direction={direction_es_rounds}  size={size_es_rounds}  "
        f"mfe={mfe_es_rounds}  mae={mae_es_rounds}  (aux base={aux_es_base}; "
        f"override via L2_EARLY_STOPPING_ROUNDS / L2_*_EARLY_STOPPING_ROUNDS)",
        flush=True,
    )
    direction_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 44,
        "n_jobs": _lgbm_n_jobs(),
    }
    reg_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 60,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 43,
        "n_jobs": _lgbm_n_jobs(),
    }
    dtrain_gate = lgb.Dataset(
        X[train_mask],
        label=y_gate[train_mask],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    dval_gate = lgb.Dataset(
        X[val_mask],
        label=y_gate[val_mask],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    l2_outer = tqdm(
        total=5,
        desc="[L2] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    direction_model: lgb.Booster | None = None
    try:
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(
            gate_es_rounds, rounds, "[L2] gate", first_metric_only=True
        )
        try:
            gate_model = lgb.train(gate_params, dtrain_gate, num_boost_round=rounds, valid_sets=[dval_gate], callbacks=cbs)
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("gate", refresh=False)
        l2_outer.update(1)

        mt = train_mask & (y_dir >= 0)
        mv = val_mask & (y_dir >= 0)
        if int(mt.sum()) >= 100:
            cbs, cl = _lgb_train_callbacks_with_round_tqdm(direction_es_rounds, rounds, "[L2] direction")
            try:
                direction_model = lgb.train(
                    direction_params,
                    lgb.Dataset(X[mt], label=y_dir_margin[mt], feature_name=feature_cols, free_raw_data=False),
                    num_boost_round=rounds,
                    valid_sets=[
                        lgb.Dataset(X[mv], label=y_dir_margin[mv], feature_name=feature_cols, free_raw_data=False)
                    ],
                    callbacks=cbs,
                )
            finally:
                for fn in cl:
                    fn()
        else:
            print(f"  [L2] direction: skip (only {int(mt.sum())} active train rows; need >= 100)", flush=True)

        l2_outer.set_postfix_str("direction", refresh=False)
        l2_outer.update(1)

        if int(active_train.sum()) < 100 or int(active_val.sum()) < 25:
            raise RuntimeError(
                "L2: too few active rows for size/MFE/MAE heads after strict time split. "
                f"active_train={int(active_train.sum())}, active_val={int(active_val.sum())}"
            )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(size_es_rounds, rounds, "[L2] size")
        try:
            size_model = lgb.train(
                reg_params,
                lgb.Dataset(X[active_train], label=y_size[active_train], feature_name=feature_cols, free_raw_data=False),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_size[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("size", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(mfe_es_rounds, rounds, "[L2] mfe")
        try:
            mfe_model = lgb.train(
                reg_params,
                lgb.Dataset(X[active_train], label=y_mfe[active_train], feature_name=feature_cols, free_raw_data=False),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_mfe[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("mfe", refresh=False)
        l2_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(mae_es_rounds, rounds, "[L2] mae")
        try:
            mae_model = lgb.train(
                reg_params,
                lgb.Dataset(X[active_train], label=y_mae[active_train], feature_name=feature_cols, free_raw_data=False),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X[active_val], label=y_mae[active_val], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l2_outer.set_postfix_str("mae", refresh=False)
        l2_outer.update(1)
    finally:
        l2_outer.close()

    gate_raw_all = gate_model.predict(X).astype(np.float64)
    gate_calibrator = _fit_binary_calibrator(y_gate[val_tune_mask], gate_raw_all[val_tune_mask])
    gate_p_tune = _apply_binary_calibrator(gate_raw_all[val_tune_mask], gate_calibrator)
    trade_threshold = _search_l2_trade_threshold(
        gate_p_tune,
        target_trade_rate=_l2_target_trade_rate(),
    )
    gate_thr_after_search = float(trade_threshold)
    if direction_model is None:
        bump, cap = _l2_no_direction_gate_bump_cap()
        trade_threshold = float(min(cap, gate_thr_after_search + bump))
        print(
            f"  [L2] direction head skipped: raising gate trade_threshold "
            f"{gate_thr_after_search:.2f} -> {trade_threshold:.2f} "
            f"(bump={bump:.2f}, cap={cap:.2f}); composed long/short split is degenerate at d=0.5",
            flush=True,
        )
    gate_p_report, dir_p_report, _ = _l2_predict_gate_dir_probs(
        gate_model,
        direction_model,
        X[val_report_mask],
        trade_threshold=trade_threshold,
        direction_head_type="signed_edge_regression",
        direction_temperature=0.35,
        gate_calibrator=gate_calibrator,
    )
    _log_l2_two_stage_val_diagnostics(
        gate_p_report,
        dir_p_report,
        y_gate[val_report_mask],
        y_dir[val_report_mask],
        y_decision[val_report_mask],
        trade_threshold,
    )
    _, _, decision_probs = _l2_predict_gate_dir_probs(
        gate_model,
        direction_model,
        X,
        trade_threshold=trade_threshold,
        direction_head_type="signed_edge_regression",
        direction_temperature=0.35,
        gate_calibrator=gate_calibrator,
        gate_raw=gate_raw_all,
    )
    size_pred = np.clip(size_model.predict(X).astype(np.float32), 0.0, 1.0)
    mfe_pred = np.clip(mfe_model.predict(X).astype(np.float32), 0.0, 5.0)
    mae_pred = np.clip(mae_model.predict(X).astype(np.float32), 0.0, 4.0)
    _log_l2_extended_val_metrics(
        frame,
        val_report_mask,
        y_decision,
        decision_probs,
        y_size,
        size_pred,
        y_mfe,
        mfe_pred,
        y_mae,
        mae_pred,
    )
    _log_l2_l1b_ablation(
        X,
        feature_cols,
        val_report_mask,
        y_decision,
        gate_model,
        direction_model,
        trade_threshold=trade_threshold,
        direction_head_type="signed_edge_regression",
        direction_temperature=0.35,
        gate_calibrator=gate_calibrator,
    )
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_decision_class"] = np.argmax(decision_probs, axis=1).astype(np.int64)
    outputs["l2_decision_long"] = decision_probs[:, 0]
    outputs["l2_decision_neutral"] = decision_probs[:, 1]
    outputs["l2_decision_short"] = decision_probs[:, 2]
    outputs["l2_decision_confidence"] = np.max(decision_probs, axis=1).astype(np.float32)
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_expected_edge"] = (outputs["l2_decision_long"] - outputs["l2_decision_short"]) * outputs["l2_size"]
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)

    os.makedirs(MODEL_DIR, exist_ok=True)
    gate_model.save_model(os.path.join(MODEL_DIR, L2_GATE_FILE))
    if direction_model is not None:
        direction_model.save_model(os.path.join(MODEL_DIR, L2_DIRECTION_FILE))
    size_model.save_model(os.path.join(MODEL_DIR, L2_SIZE_FILE))
    mfe_model.save_model(os.path.join(MODEL_DIR, L2_MFE_FILE))
    mae_model.save_model(os.path.join(MODEL_DIR, L2_MAE_FILE))
    model_files: dict[str, str] = {
        "gate": L2_GATE_FILE,
        "size": L2_SIZE_FILE,
        "mfe": L2_MFE_FILE,
        "mae": L2_MAE_FILE,
    }
    if direction_model is not None:
        model_files["direction"] = L2_DIRECTION_FILE
    if gate_calibrator is not None:
        gate_calib_file = "l2_gate_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, gate_calib_file), "wb") as f:
            pickle.dump(gate_calibrator, f)
        model_files["gate_calibrator"] = gate_calib_file
    meta = {
        "schema_version": L2_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "output_cols": L2_OUTPUT_COLS,
        "decision_mode": "two_stage",
        "decision_tau": tau,
        "trade_threshold": trade_threshold,
        "trade_threshold_applies_to": "gate_probability",
        "direction_available": direction_model is not None,
        "direction_head_type": "signed_edge_regression" if direction_model is not None else "none",
        "direction_temperature": 0.35,
        "confidence_semantics": "max(composed long, neutral, short); composed from gate*dir",
        "size_semantics": "risk_adjusted_position_fraction",
        "model_files": model_files,
        "output_cache_file": L2_OUTPUT_CACHE_FILE,
        "target_trade_rate": _l2_target_trade_rate(),
        "l2_val_tune_frac": tune_frac,
        "l2_min_feature_std": min_std,
        "l2_feature_hard_drop_skipped": skip_hard,
        "l2_feature_selection_dropped": l2_dropped_features,
        "l2_train_boost_rounds": int(rounds),
        "l2_early_stopping_rounds": {
            "gate": int(gate_es_rounds),
            "direction": int(direction_es_rounds),
            "size": int(size_es_rounds),
            "mfe": int(mfe_es_rounds),
            "mae": int(mae_es_rounds),
        },
        "l2_gate_config": {
            "learning_rate": float(gate_lr),
            "num_leaves": int(gate_leaves),
            "max_depth": int(gate_depth),
            "min_child_samples": int(gate_mcs),
            "bagging_fraction": float(gate_bag),
            "bagging_freq": int(gate_bag_freq),
            "is_unbalance": True,
            "metric_eval_order": ["auc", "binary_logloss"],
            "early_stopping_on": "first_metric (auc)",
            "early_stopping_rounds": int(gate_es_rounds),
        },
    }
    if direction_model is None:
        meta["trade_threshold_gate_search"] = gate_thr_after_search
        meta["direction_degenerate_note"] = (
            "direction head absent: inference uses d=0.5 so p_long=p_short=gate/2; "
            "hard class ties break to long (dir>=0.5). Gate threshold raised to compensate."
        )
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    if test_mask.any():
        test_edge = outputs.loc[test_mask, "l2_expected_edge"].to_numpy(dtype=np.float32)
        corr = np.corrcoef(test_edge, edge[test_mask])[0, 1] if int(test_mask.sum()) > 2 else float("nan")
        print(f"  [L2] test corr(expected_edge, decision_edge_atr)={corr:.4f}", flush=True)
    print(f"  [L2] meta saved  -> {os.path.join(MODEL_DIR, L2_META_FILE)}", flush=True)
    print(f"  [L2] cache saved -> {cache_path}", flush=True)
    bundle_models: dict[str, Any] = {
        "gate": gate_model,
        "size": size_model,
        "mfe": mfe_model,
        "mae": mae_model,
    }
    if direction_model is not None:
        bundle_models["direction"] = direction_model
    if gate_calibrator is not None:
        bundle_models["gate_calibrator"] = gate_calibrator
    return L2TrainingBundle(models=bundle_models, meta=meta, outputs=outputs)


def load_l2_trade_decision() -> tuple[dict[str, Any], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L2_SCHEMA_VERSION:
        raise RuntimeError(
            f"L2 schema mismatch: artifact has {meta.get('schema_version')} but code expects {L2_SCHEMA_VERSION}. "
            f"Retrain L2 so artifacts match schema {L2_SCHEMA_VERSION}."
        )
    model_files = meta.get("model_files", {})
    models: dict[str, Any] = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in model_files.items()
        if name != "gate_calibrator"
    }
    gate_calib_file = model_files.get("gate_calibrator")
    if gate_calib_file:
        with open(os.path.join(MODEL_DIR, gate_calib_file), "rb") as f:
            models["gate_calibrator"] = pickle.load(f)
    return models, meta


def infer_l2_trade_decision(
    models: dict[str, Any],
    meta: dict[str, Any],
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l1b_outputs: pd.DataFrame,
) -> pd.DataFrame:
    frame, feature_cols = _build_l2_frame(df, l1a_outputs, l1b_outputs)
    target_cols = list(meta["feature_cols"])
    for col in target_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    X = frame[target_cols].to_numpy(dtype=np.float32, copy=False)
    mode = meta.get("decision_mode", "multiclass")
    if mode == "two_stage":
        thr = float(meta.get("trade_threshold", 0.35))
        gate_m = models["gate"]
        dir_m = models.get("direction")
        gate_calibrator = models.get("gate_calibrator")
        # thr applies to gate_p >= thr (not argmax on composed 3-way probs); see meta trade_threshold_applies_to
        _, _, decision_probs = _l2_predict_gate_dir_probs(
            gate_m,
            dir_m,
            X,
            trade_threshold=thr,
            direction_head_type=str(meta.get("direction_head_type", "probability")),
            direction_temperature=float(meta.get("direction_temperature", 0.35)),
            gate_calibrator=gate_calibrator,
        )
    elif "decision" in models:
        decision_probs = models["decision"].predict(X).astype(np.float32)
    else:
        raise RuntimeError("L2 meta missing two_stage gate or legacy decision model.")
    size_pred = np.clip(models["size"].predict(X).astype(np.float32), 0.0, 1.0)
    mfe_pred = np.clip(models["mfe"].predict(X).astype(np.float32), 0.0, 5.0)
    mae_pred = np.clip(models["mae"].predict(X).astype(np.float32), 0.0, 4.0)
    outputs = df[["symbol", "time_key"]].copy()
    outputs["l2_decision_class"] = np.argmax(decision_probs, axis=1).astype(np.int64)
    outputs["l2_decision_long"] = decision_probs[:, 0]
    outputs["l2_decision_neutral"] = decision_probs[:, 1]
    outputs["l2_decision_short"] = decision_probs[:, 2]
    outputs["l2_decision_confidence"] = np.max(decision_probs, axis=1).astype(np.float32)
    outputs["l2_size"] = size_pred
    outputs["l2_pred_mfe"] = mfe_pred
    outputs["l2_pred_mae"] = mae_pred
    entry_regime = frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    for idx in range(NUM_REGIME_CLASSES):
        outputs[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
    outputs["l2_entry_vol"] = frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    outputs["l2_expected_edge"] = (outputs["l2_decision_long"] - outputs["l2_decision_short"]) * outputs["l2_size"]
    outputs["l2_rr_proxy"] = outputs["l2_pred_mfe"] / np.maximum(outputs["l2_pred_mae"], 0.05)
    return outputs

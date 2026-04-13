from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
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
    L1A_REGIME_COLS,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L3_EXIT_FILE,
    L3_META_FILE,
    L3_SCHEMA_VERSION,
    L3_TRAJECTORY_ENCODER_FILE,
    L3_VALUE_FILE,
    MODEL_DIR,
    TEST_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _live_trade_state_from_bar,
    _lgbm_n_jobs,
    _net_edge_atr_from_state,
    _options_target_config,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.trainers.stack_v2_common import log_label_baseline
from core.trainers.val_metrics_extra import (
    brier_binary,
    directional_accuracy_regression,
    ece_binary,
    flip_rate_sorted,
    pearson_corr,
    regression_degen_flag,
)
from core.trainers.l3_trajectory_hybrid import (
    L3TrajectoryConfig,
    l3_encode_trajectories,
    l3_trajectory_embed_importance_ratio,
    l3_traj_step_features,
    train_l3_trajectory_encoder,
)
from core.trainers.tcn_constants import DEVICE as TORCH_DEVICE


@dataclass
class L3TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]


def _l3_oot_train_val_masks_by_trade(
    t_state: np.ndarray | pd.Series,
    rows_entry: np.ndarray,
    oot_mask: np.ndarray,
    *,
    train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Split OOT policy rows into train/val by whole trade (entry signal row id).

    Avoids putting prefixes of the same position in train and later bars in val, which
    would leak trajectory structure into GRU/LGBM validation.
    """
    n = int(rows_entry.shape[0])
    ts = np.asarray(pd.to_datetime(t_state))
    oot = np.asarray(oot_mask, dtype=bool)
    oot_idx = np.flatnonzero(oot)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    if oot_idx.size == 0:
        return train_mask, val_mask

    unique_e = np.unique(rows_entry[oot_idx])
    trade_t0: list[tuple[np.datetime64, int]] = []
    for e in unique_e:
        e = int(e)
        m = (rows_entry == e) & oot
        if not np.any(m):
            continue
        ix = np.flatnonzero(m)
        tmin = np.min(ts[ix])
        trade_t0.append((tmin, e))
    trade_t0.sort(key=lambda x: x[0])
    n_tr = len(trade_t0)
    if n_tr == 0:
        return train_mask, val_mask

    if n_tr >= 2:
        split = max(1, min(n_tr - 1, int(round(n_tr * train_frac))))
        train_entries = {e for _, e in trade_t0[:split]}
        val_entries = {e for _, e in trade_t0[split:]}
        for idx in oot_idx:
            e = int(rows_entry[idx])
            if e in train_entries:
                train_mask[idx] = True
            elif e in val_entries:
                val_mask[idx] = True
    else:
        # Single OOT trade: cannot separate whole trades; fall back to time-ordered bar split.
        print(
            "  [L3] WARNING: only one distinct entry in OOT window; val split is intra-trade "
            "(prefix leakage possible for GRU). Prefer more OOT trades or longer calendar span.",
            flush=True,
        )
        order = oot_idx[np.argsort(ts[oot_idx])]
        split = max(1, min(int(order.size) - 1, int(round(float(order.size) * train_frac))))
        train_mask[order[:split]] = True
        val_mask[order[split:]] = True

    if not val_mask.any() and oot_idx.size:
        val_mask[oot_idx[-1:]] = True
        train_mask[oot_idx[-1:]] = False
    return train_mask, val_mask


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0)
    q = np.clip(q, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1).astype(np.float32)


def _split_l3_val_for_calibration(
    t_state: pd.Series | np.ndarray,
    val_mask: np.ndarray,
    *,
    tune_frac: float,
    min_rows_each: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(val_mask, dtype=bool)
    idx = np.flatnonzero(base)
    tune_mask = np.zeros_like(base, dtype=bool)
    report_mask = np.zeros_like(base, dtype=bool)
    if idx.size == 0:
        return tune_mask, report_mask
    ts = np.asarray(pd.to_datetime(t_state))
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


def _fit_l3_exit_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> IsotonicRegression | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return calib


def _apply_l3_exit_calibrator(p: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)


def _l3_entry_policy_config() -> tuple[float, float]:
    min_conf = float(np.clip(float(os.environ.get("L3_ENTRY_MIN_CONFIDENCE", "0.0")), 0.0, 1.0))
    min_size = float(max(0.0, float(os.environ.get("L3_ENTRY_MIN_SIZE", "0.05"))))
    return min_conf, min_size


def _l3_target_horizon_bars(max_hold: int) -> int:
    cfg = _options_target_config()
    raw = int(os.environ.get("L3_TARGET_HORIZON_BARS", str(cfg["decision_horizon_bars"])))
    return max(1, min(int(max_hold), raw))


def l3_entry_side_from_l2(decision_class: int, decision_confidence: float, size: float, *, min_confidence: float, min_size: float) -> float:
    if float(size) < float(min_size) or float(decision_confidence) < float(min_confidence):
        return 0.0
    if int(decision_class) == 0:
        return 1.0
    if int(decision_class) == 2:
        return -1.0
    return 0.0


def _build_l3_policy_dataset(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
    *,
    max_hold: int = 30,
    exit_epsilon_atr: float = 0.03,
    traj_cfg: L3TrajectoryConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    merged = (
        df.reset_index(drop=True)
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l2_outputs, on=["symbol", "time_key"], how="left")
    )
    safe_atr = np.where(pd.to_numeric(merged["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, merged["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
    open_px = merged["open"].to_numpy(dtype=np.float64)
    high_px = merged["high"].to_numpy(dtype=np.float64)
    low_px = merged["low"].to_numpy(dtype=np.float64)
    close_px = merged["close"].to_numpy(dtype=np.float64)
    symbols = merged["symbol"].to_numpy()
    times = pd.to_datetime(merged["time_key"]).to_numpy()
    current_regime = merged[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    entry_regime = merged[[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32, copy=False)
    current_vol = merged["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    entry_vol = merged["l2_entry_vol"].to_numpy(dtype=np.float32, copy=False)
    decision_class = (
        pd.to_numeric(merged["l2_decision_class"], errors="coerce").fillna(1).astype(np.int64).to_numpy()
    )
    decision_conf = merged["l2_decision_confidence"].to_numpy(dtype=np.float32, copy=False)
    size = merged["l2_size"].to_numpy(dtype=np.float32, copy=False)
    edge_atr = _decision_edge_atr_array(merged).astype(np.float64)
    tau_edge = 0.05  # same as L2 y_decision threshold
    pred_mfe = merged["l2_pred_mfe"].to_numpy(dtype=np.float32, copy=False)
    pred_mae = merged["l2_pred_mae"].to_numpy(dtype=np.float32, copy=False)
    _t_cfg = traj_cfg or L3TrajectoryConfig()
    _t_max = _t_cfg.max_seq_len
    _t_ref = max(_t_max, int(max_hold))
    target_horizon = _l3_target_horizon_bars(max_hold)

    rows_x: list[np.ndarray] = []
    rows_exit: list[int] = []
    rows_value: list[float] = []
    rows_time: list[np.datetime64] = []
    rows_entry: list[int] = []
    rows_from_model: list[int] = []
    rows_traj: list[np.ndarray] = []
    rows_traj_len: list[int] = []
    n_policy_signals_model = 0
    n_policy_signals_truth = 0
    allow_truth_fallback = os.environ.get("L3_ALLOW_TRUTH_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
    _hold_bin_edges = np.array([3, 8, 15, 30, 999], dtype=np.int64)
    feature_cols = [
        "l2_decision_confidence",
        "l2_size",
        "l2_pred_mfe",
        "l2_pred_mae",
        *[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))],
        "l2_entry_vol",
        *L1A_REGIME_COLS,
        "l1a_vol_forecast",
        "l3_regime_divergence",
        "l3_vol_surprise",
        "l3_hold_bars",
        "l3_unreal_pnl_atr",
        "l3_live_mfe",
        "l3_live_mae",
        "l3_live_edge",
        "l3_side",
        "l3_log_hold_bars",
        "l3_hold_bars_sq",
        "l3_hold_bucket",
    ]
    min_confidence, min_size = _l3_entry_policy_config()

    row_it = range(len(merged))
    if _lgb_round_tqdm_enabled():
        row_it = tqdm(
            row_it,
            desc="[L3] policy dataset",
            unit="bar",
            leave=False,
            file=TQDM_FILE,
            mininterval=1.0,
        )
    for i in row_it:
        if i + 1 >= len(merged) or symbols[i + 1] != symbols[i]:
            continue
        sz = float(size[i])
        model_side = l3_entry_side_from_l2(
            int(decision_class[i]),
            float(decision_conf[i]),
            sz,
            min_confidence=min_confidence,
            min_size=min_size,
        )
        model_active = model_side != 0.0
        ed = float(edge_atr[i])
        truth_dir = 1
        if ed > tau_edge:
            truth_dir = 0
        elif ed < -tau_edge:
            truth_dir = 2
        truth_active = truth_dir != 1
        if model_active:
            side = float(model_side)
            n_policy_signals_model += 1
            from_model = 1
        elif truth_active and allow_truth_fallback:
            # Optional legacy fallback when L2 is too sparse; disabled by default to match deployment.
            side = 1.0 if truth_dir == 0 else -1.0
            n_policy_signals_truth += 1
            from_model = 0
        else:
            continue
        entry_price = float(open_px[i + 1])
        atr = float(safe_atr[i])
        end = min(len(merged), i + target_horizon + 1, i + max_hold + 1)
        live_mfe = 0.0
        live_mae = 0.0
        entry_regime_row = entry_regime[i : i + 1]
        prev_unreal = 0.0
        peak_unreal = -1e9
        traj_hist: list[np.ndarray] = []
        entry_rows_x: list[np.ndarray] = []
        entry_rows_time: list[np.datetime64] = []
        entry_rows_entry: list[int] = []
        entry_rows_traj: list[np.ndarray] = []
        entry_rows_traj_len: list[int] = []
        entry_rows_unreal: list[float] = []
        entry_rows_from_model: list[int] = []
        for j in range(i + 1, end):
            if symbols[j] != symbols[i]:
                break
            hold = j - i
            fav, adv, unreal = _live_trade_state_from_bar(
                side=side,
                entry_price=entry_price,
                atr=atr,
                high_price=float(high_px[j]),
                low_price=float(low_px[j]),
                close_price=float(close_px[j]),
            )
            live_mfe = max(live_mfe, fav)
            live_mae = max(live_mae, adv)
            live_edge = float(_net_edge_atr_from_state(live_mfe, live_mae, hold))
            regime_div = float(_kl_divergence(entry_regime_row, current_regime[j : j + 1])[0])
            vol_surprise = float(current_vol[j] / max(float(entry_vol[i]), 1e-3))
            close_prev = float(close_px[j - 1])
            peak_unreal = max(peak_unreal, float(unreal))
            tvec = l3_traj_step_features(
                float(unreal),
                prev_unreal,
                peak_unreal,
                int(hold),
                times[j],
                close_prev,
                float(close_px[j]),
                float(high_px[j]),
                float(low_px[j]),
                atr,
                vol_surprise,
                regime_div,
                float(live_mfe),
                float(live_mae),
                max_seq_ref=_t_ref,
            )
            traj_hist.append(tvec)
            prev_unreal = float(unreal)
            window = traj_hist[-_t_max:]
            sl = len(window)
            seq_pad = np.zeros((_t_max, _t_cfg.seq_feat_dim), dtype=np.float32)
            if sl:
                seq_pad[:sl] = np.stack(window, axis=0)
            entry_rows_traj.append(seq_pad)
            entry_rows_traj_len.append(sl)
            log_h = float(np.log1p(hold))
            h_sq = float(hold * hold) / 100.0
            h_bkt = float(np.searchsorted(_hold_bin_edges, int(hold), side="right"))
            entry_rows_x.append(
                np.asarray(
                    [
                        decision_conf[i],
                        size[i],
                        pred_mfe[i],
                        pred_mae[i],
                        *entry_regime[i].tolist(),
                        entry_vol[i],
                        *current_regime[j].tolist(),
                        current_vol[j],
                        regime_div,
                        vol_surprise,
                        float(hold),
                        float(unreal),
                        float(live_mfe),
                        float(live_mae),
                        float(live_edge),
                        float(side),
                        log_h,
                        h_sq,
                        h_bkt,
                    ],
                    dtype=np.float32,
                )
            )
            entry_rows_unreal.append(float(unreal))
            entry_rows_time.append(times[j])
            entry_rows_entry.append(int(i))
            entry_rows_from_model.append(from_model)
        if entry_rows_x:
            terminal_unreal = float(entry_rows_unreal[-1])
            last_idx = len(entry_rows_x) - 1
            for row_idx, feat_row in enumerate(entry_rows_x):
                future_gain_left = float(terminal_unreal - entry_rows_unreal[row_idx])
                rows_x.append(feat_row)
                rows_exit.append(1 if row_idx == last_idx or future_gain_left <= exit_epsilon_atr else 0)
                rows_value.append(future_gain_left)
                rows_time.append(entry_rows_time[row_idx])
                rows_entry.append(entry_rows_entry[row_idx])
                rows_traj.append(entry_rows_traj[row_idx])
                rows_traj_len.append(entry_rows_traj_len[row_idx])
                rows_from_model.append(entry_rows_from_model[row_idx])
    if not rows_x:
        print(
            f"  [L3] policy dataset empty: no bars with L2 model trade (class≠neutral, size>0.05) "
            f"or label edge (|decision_edge_atr|>{tau_edge}) at a row with same-symbol next bar.",
            flush=True,
        )
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            feature_cols,
            np.empty(0, dtype=np.int64),
            np.empty((0, _t_max, _t_cfg.seq_feat_dim), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    print(
        f"  [L3] policy dataset: entry signals model={n_policy_signals_model:,} "
        f"truth_edge_fallback={n_policy_signals_truth:,}  policy_rows={len(rows_x):,}  "
        f"allow_truth_fallback={allow_truth_fallback}  entry_min_conf={min_confidence:.2f}  "
        f"entry_min_size={min_size:.2f}  target_horizon={target_horizon}",
        flush=True,
    )
    if n_policy_signals_truth and not n_policy_signals_model:
        print(
            "  [L3] NOTE: all policy entries from label edge fallback — L2 predictions rarely trade; "
            "L3 still uses merged L2 features at each signal bar.",
            flush=True,
        )
    return (
        np.asarray(rows_x, dtype=np.float32),
        np.asarray(rows_exit, dtype=np.int32),
        np.asarray(rows_value, dtype=np.float32),
        np.asarray(rows_time),
        feature_cols,
        np.asarray(rows_entry, dtype=np.int64),
        np.stack(rows_traj, axis=0).astype(np.float32, copy=False),
        np.asarray(rows_traj_len, dtype=np.int32),
        np.asarray(rows_from_model, dtype=np.int32),
    )


def l3_survival_from_hazard(hazard_probs: np.ndarray) -> np.ndarray:
    """Discrete survival S(t)=prod_{s<=t}(1-h(s)) for ordered hazard per episode."""
    h = np.clip(np.asarray(hazard_probs, dtype=np.float64).ravel(), 0.0, 1.0)
    return np.cumprod(1.0 - h)


def l3_group_hazard_by_entry(entry_row_idx: np.ndarray, hazard_probs: np.ndarray) -> dict[int, np.ndarray]:
    """Map entry bar index -> hazard sequence in row order (contiguous per entry in builder)."""
    order: dict[int, list[float]] = {}
    for e, p in zip(np.asarray(entry_row_idx).tolist(), np.asarray(hazard_probs).tolist()):
        order.setdefault(int(e), []).append(float(p))
    return {k: np.asarray(v, dtype=np.float64) for k, v in order.items()}


def _log_l3_val_extended(
    X: np.ndarray,
    y_exit: np.ndarray,
    y_value: np.ndarray,
    t_state: pd.Series | np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    exit_model: lgb.Booster,
    value_model: lgb.Booster,
    *,
    exit_calibrator: IsotonicRegression | None = None,
) -> None:
    vm = np.asarray(val_mask, dtype=bool)
    if int(vm.sum()) < 5:
        return
    p_exit = _apply_l3_exit_calibrator(exit_model.predict(X[vm]).astype(np.float64), exit_calibrator)
    yv = y_exit[vm].astype(np.int32)
    try:
        ll = float(log_loss(yv, p_exit))
    except ValueError:
        ll = float("nan")
    try:
        auc = float(roc_auc_score(yv, p_exit))
    except ValueError:
        auc = float("nan")
    br = brier_binary(yv.astype(np.float64), p_exit)
    ece = ece_binary(yv, p_exit)
    yhat = (p_exit >= 0.5).astype(np.int32)
    acc = float(accuracy_score(yv, yhat))
    f1 = float(f1_score(yv, yhat, zero_division=0))
    cm = confusion_matrix(yv, yhat, labels=[0, 1])
    print("\n  [L3] val — exit (extended)", flush=True)
    print(
        f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  acc@0.5={acc:.4f}  F1={f1:.4f}",
        flush=True,
    )
    print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)

    ih = feature_cols.index("l3_hold_bars")
    hold_vm = np.asarray(X[vm, ih], dtype=np.int64)
    print(" exit AUC by hold bucket (val):", flush=True)
    for lo, hi in [(0, 3), (3, 8), (8, 15), (15, 30), (30, 10_000)]:
        m_hold = (hold_vm >= lo) & (hold_vm < hi)
        n_sub = int(m_hold.sum())
        if n_sub < 50:
            continue
        yy = yv[m_hold]
        pp = p_exit[m_hold]
        if len(np.unique(yy)) < 2:
            continue
        try:
            auc_h = float(roc_auc_score(yy, pp))
        except ValueError:
            auc_h = float("nan")
        print(f"      hold [{lo:>3d}, {hi:>5d}): n={n_sub:>6d}  AUC={auc_h:.4f}", flush=True)

    reg_cols = sorted(
        [c for c in feature_cols if c.startswith("l2_entry_regime_")],
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if reg_cols:
        reg_idx = [feature_cols.index(c) for c in reg_cols]
        E = X[vm][:, reg_idx]
        reg_id = np.argmax(E, axis=1)
        for k in range(len(reg_cols)):
            m = reg_id == k
            n_k = int(m.sum())
            if n_k < 20:
                continue
            yy = yv[m]
            pp = p_exit[m]
            if len(np.unique(yy)) < 2:
                continue
            try:
                auc_k = float(roc_auc_score(yy, pp))
                print(f"    entry-regime {k}  AUC={auc_k:.4f}  n={n_k:,}", flush=True)
            except ValueError:
                pass

    idx_div = feature_cols.index("l3_regime_divergence")
    dvm = np.asarray(X[vm, idx_div], dtype=np.float64)
    fin = np.isfinite(dvm)
    if fin.any():
        err_abs = np.abs(p_exit - yv.astype(np.float64))
        c_div_err = pearson_corr(dvm[fin], err_abs[fin])
        print(
            f"    l3_regime_divergence (val): mean={float(np.mean(dvm[fin])):.6f}  std={float(np.std(dvm[fin])):.6f}  "
            f"corr(div, |p-y|)={c_div_err:.4f}",
            flush=True,
        )

    i_conf = feature_cols.index("l2_decision_confidence")
    i_size = feature_cols.index("l2_size")
    c_ec = pearson_corr(X[vm, i_conf].astype(np.float64), p_exit)
    vv_pred = value_model.predict(X[vm]).astype(np.float64)
    c_sz_val = pearson_corr(X[vm, i_size].astype(np.float64), vv_pred)
    print(f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  corr(L2 size, L3 value pred)={c_sz_val:.4f}", flush=True)

    t_vm = np.asarray(pd.to_datetime(t_state))[vm]
    order = np.argsort(t_vm)
    fr = flip_rate_sorted(p_exit, order)
    print(f"    exit prob flip_rate (time-sorted val)={fr:.6f}", flush=True)

    vv_true = y_value[vm].astype(np.float64)
    mae_v = float(mean_absolute_error(vv_true, vv_pred))
    rmse_v = float(np.sqrt(mean_squared_error(vv_true, vv_pred)))
    r2_v = float(r2_score(vv_true, vv_pred)) if len(np.unique(vv_true)) > 1 else float("nan")
    c_v = pearson_corr(vv_true, vv_pred)
    std_v, degen_v = regression_degen_flag(vv_pred)
    dir_acc = directional_accuracy_regression(vv_true, vv_pred)
    print("\n  [L3] val — value (extended)", flush=True)
    print(
        f"    MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}  corr={c_v:.4f}  "
        f"dir_acc={dir_acc:.4f}  pred_std={std_v:.6f}  degen={degen_v}",
        flush=True,
    )
    if reg_cols:
        for k in range(len(reg_cols)):
            m = reg_id == k
            n_k = int(m.sum())
            if n_k < 15:
                continue
            yt = vv_true[m]
            yp = vv_pred[m]
            mae_k = float(mean_absolute_error(yt, yp))
            r2_k = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
            print(f"    entry-regime {k}  MAE={mae_k:.4f}  R2={r2_k:.4f}  n={n_k:,}", flush=True)

    tm = np.asarray(test_mask, dtype=bool)
    if int(tm.sum()) >= 5:
        p_t = _apply_l3_exit_calibrator(exit_model.predict(X[tm]).astype(np.float64), exit_calibrator)
        yt = y_exit[tm].astype(np.int32)
        try:
            auc_t = float(roc_auc_score(yt, p_t))
        except ValueError:
            auc_t = float("nan")
        print(f"\n  [L3] holdout — exit AUC={auc_t:.4f}  n={int(tm.sum()):,}", flush=True)


def train_l3_exit_manager(df: pd.DataFrame, l1a_outputs: pd.DataFrame, l2_outputs: pd.DataFrame) -> L3TrainingBundle:
    traj_cfg = L3TrajectoryConfig()
    X, y_exit, y_value, t_state, feature_cols, rows_entry, traj_seq, traj_len, rows_from_model = _build_l3_policy_dataset(
        df, l1a_outputs, l2_outputs, traj_cfg=traj_cfg
    )
    if len(X) == 0:
        raise RuntimeError("L3: policy dataset is empty.")
    t_state = pd.to_datetime(t_state)
    oot_mask = (t_state >= np.datetime64(CAL_END)) & (t_state < np.datetime64(TEST_END))
    holdout_mask = t_state >= np.datetime64(TEST_END)
    oot_idx = np.flatnonzero(oot_mask)
    if len(oot_idx) < 20:
        raise RuntimeError("L3: not enough post-CAL_END state rows for strict OOT training.")
    train_mask, val_mask = _l3_oot_train_val_masks_by_trade(
        t_state, rows_entry, oot_mask, train_frac=0.7
    )
    val_tune_frac = float(os.environ.get("L3_VAL_TUNE_FRAC", "0.5"))
    val_tune_mask, val_report_mask = _split_l3_val_for_calibration(
        t_state, val_mask, tune_frac=val_tune_frac, min_rows_each=40
    )
    n_oot_tr = len(np.unique(rows_entry[oot_idx]))
    print(
        f"  [L3] OOT train/val split by trade_id (rows_entry): {n_oot_tr:,} distinct entries in OOT window",
        flush=True,
    )
    test_mask = np.asarray(holdout_mask, dtype=bool)

    log_layer_banner("[L3] Exit / policy (LGBM)")
    log_time_key_arrays(
        "L3",
        pd.Series(t_state[train_mask]),
        pd.Series(t_state[val_mask]),
        train_label="policy train (OOT trades, ~70% of distinct entries)",
        val_label="policy val (OOT trades, ~30% of distinct entries)",
        extra_note=f"Split by rows_entry (signal bar), not by policy row index. Times in [{CAL_END}, {TEST_END}); holdout t>={TEST_END}.",
    )
    log_time_key_arrays(
        "L3(calibration/report)",
        pd.Series(t_state[val_tune_mask]),
        pd.Series(t_state[val_report_mask]),
        train_label="val_tune (exit calibration)",
        val_label="val_report (headline metrics)",
        extra_note="Exit probability calibration is fit on val_tune; headline validation metrics use val_report.",
    )
    if test_mask.any():
        tt_hold = pd.Series(t_state[test_mask])
        print(
            f"  [L3] holdout samples: {int(test_mask.sum()):,}  time_key: [{tt_hold.min()}, {tt_hold.max()}]",
            flush=True,
        )
    model_row_rate = float(np.mean(rows_from_model.astype(np.float64))) if len(rows_from_model) else float("nan")
    print(
        f"  [L3] policy rows from model entries={model_row_rate:.3f}  "
        f"(fallback rows={int(np.sum(rows_from_model == 0)):,})",
        flush=True,
    )
    idx_div = feature_cols.index("l3_regime_divergence")
    div_all = np.asarray(X[:, idx_div], dtype=np.float64)
    fin = np.isfinite(div_all)
    print(
        f"  [L3] l3_regime_divergence: min={np.nanmin(div_all):.6f}  max={np.nanmax(div_all):.6f}  "
        f"mean={float(np.nanmean(div_all)):.6f}  finite={int(fin.sum()):,}/{len(div_all):,}",
        flush=True,
    )
    if (~fin).any():
        print(f"  [L3] l3_regime_divergence non-finite count: {int((~fin).sum())}", flush=True)
    log_numpy_x_stats("L3", X[train_mask], label="X[policy_train]")
    print(
        f"  [L3] L2 artifact ref (features come from l2_outputs merge): {artifact_path(L2_META_FILE)} / cache {artifact_path(L2_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    print(
        "  [L3] note: l2_* / l1a_* in policy rows are from merged pipeline outputs (same run), not a separate OOF run.",
        flush=True,
    )
    print(
        "  [L3] target semantics: continuation value to a fixed deadline (non-oracle), not optimal future exit.",
        flush=True,
    )
    print(f"  [L3] will write: {artifact_path(L3_EXIT_FILE)} | {artifact_path(L3_VALUE_FILE)} | {artifact_path(L3_META_FILE)}", flush=True)
    log_label_baseline("l3_exit", y_exit[train_mask], task="cls")
    log_label_baseline("l3_value", y_value[train_mask], task="reg")

    use_hybrid = os.environ.get("L3_TRAJ_GRU", "1").strip().lower() in {"1", "true", "yes"}
    use_hybrid = (
        use_hybrid
        and int(train_mask.sum()) >= 300
        and int(val_mask.sum()) >= 30
        and traj_seq.shape[0] == len(X)
    )
    static_cols = list(feature_cols)
    n_static = len(static_cols)
    if use_hybrid:
        print("  [L3] hybrid: training trajectory GRU encoder (L3_TRAJ_GRU=1)...", flush=True)
        encoder = train_l3_trajectory_encoder(
            traj_seq[train_mask],
            traj_len[train_mask],
            y_exit[train_mask].astype(np.float32),
            y_value[train_mask].astype(np.float32),
            traj_seq[val_mask],
            traj_len[val_mask],
            y_exit[val_mask].astype(np.float32),
            y_value[val_mask].astype(np.float32),
            cfg=traj_cfg,
            device=TORCH_DEVICE,
        )
        emb_all = l3_encode_trajectories(encoder, traj_seq, traj_len, TORCH_DEVICE)
        emb_names = [f"l3_traj_emb_{k}" for k in range(traj_cfg.embed_dim)]
        X_lgb = np.hstack([X, emb_all.astype(np.float32, copy=False)])
        feature_cols = static_cols + emb_names
        print(
            f"  [L3] hybrid layout: static={n_static}  traj_step_dim={traj_cfg.seq_feat_dim}  "
            f"embed_dim={traj_cfg.embed_dim}  lgbm_cols={len(feature_cols)}",
            flush=True,
        )
    else:
        if not os.environ.get("L3_TRAJ_GRU", "1").strip().lower() in {"1", "true", "yes"}:
            print("  [L3] hybrid disabled (L3_TRAJ_GRU=0); LightGBM on static features only.", flush=True)
        else:
            print(
                f"  [L3] hybrid skipped (need train>=300 & val>=30); "
                f"got train={int(train_mask.sum())} val={int(val_mask.sum())}.",
                flush=True,
            )
        encoder = None
        X_lgb = X
        feature_cols = static_cols

    rounds = 250
    es_rounds = 40
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 71,
        "n_jobs": _lgbm_n_jobs(),
    }
    value_params = {**exit_params, "objective": "regression", "metric": "l2", "seed": 72}
    l3_outer = tqdm(
        total=2,
        desc="[L3] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    try:
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, "[L3] exit")
        try:
            ih_tr = feature_cols.index("l3_hold_bars")
            hold_tr = X_lgb[train_mask, ih_tr].astype(np.float64)
            w_exit = (np.log1p(hold_tr) + 1.0) * np.where(y_exit[train_mask].astype(np.int32) == 1, 2.0, 1.0)
            exit_model = lgb.train(
                exit_params,
                lgb.Dataset(
                    X_lgb[train_mask],
                    label=y_exit[train_mask],
                    weight=w_exit.astype(np.float32),
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X_lgb[val_mask], label=y_exit[val_mask], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l3_outer.set_postfix_str("exit", refresh=False)
        l3_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, "[L3] value")
        try:
            value_model = lgb.train(
                value_params,
                lgb.Dataset(X_lgb[train_mask], label=y_value[train_mask], feature_name=feature_cols, free_raw_data=False),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X_lgb[val_mask], label=y_value[val_mask], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l3_outer.set_postfix_str("value", refresh=False)
        l3_outer.update(1)
    finally:
        l3_outer.close()
    exit_calibrator = _fit_l3_exit_calibrator(
        y_exit[val_tune_mask],
        exit_model.predict(X_lgb[val_tune_mask]).astype(np.float64),
    )
    if use_hybrid:
        st_sh, em_sh = l3_trajectory_embed_importance_ratio(exit_model, n_static, traj_cfg.embed_dim)
        st_v, em_v = l3_trajectory_embed_importance_ratio(value_model, n_static, traj_cfg.embed_dim)
        print(
            f"  [L3] gain importance share — exit: static={st_sh:.1%}  traj_emb={em_sh:.1%}  |  "
            f"value: static={st_v:.1%}  traj_emb={em_v:.1%}",
            flush=True,
        )
    _log_l3_val_extended(
        X_lgb,
        y_exit,
        y_value,
        t_state,
        feature_cols,
        val_report_mask,
        test_mask,
        exit_model,
        value_model,
        exit_calibrator=exit_calibrator,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    exit_model.save_model(os.path.join(MODEL_DIR, L3_EXIT_FILE))
    value_model.save_model(os.path.join(MODEL_DIR, L3_VALUE_FILE))
    model_files: dict[str, str] = {"exit": L3_EXIT_FILE, "value": L3_VALUE_FILE}
    if exit_calibrator is not None:
        exit_calib_file = "l3_exit_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, exit_calib_file), "wb") as f:
            pickle.dump(exit_calibrator, f)
        model_files["exit_calibrator"] = exit_calib_file
    entry_min_confidence, entry_min_size = _l3_entry_policy_config()
    meta = {
        "schema_version": L3_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "model_files": model_files,
        "derived_features": [
            "l3_regime_divergence",
            "l3_vol_surprise",
            "l3_log_hold_bars",
            "l3_hold_bars_sq",
            "l3_hold_bucket",
        ],
        "l3_hybrid": bool(use_hybrid),
        "l3_val_tune_frac": val_tune_frac,
        "l3_allow_truth_fallback": os.environ.get("L3_ALLOW_TRUTH_FALLBACK", "0").strip().lower() in {"1", "true", "yes"},
        "l3_entry_min_confidence": entry_min_confidence,
        "l3_entry_min_size": entry_min_size,
        "l3_target_horizon_bars": _l3_target_horizon_bars(30),
        "l3_target_semantics": "continuation_value_to_fixed_deadline",
    }
    if use_hybrid:
        import torch

        meta["l3_traj_encoder_file"] = L3_TRAJECTORY_ENCODER_FILE
        meta["l3_traj_cfg"] = asdict(traj_cfg)
        torch.save(
            encoder.state_dict(),
            os.path.join(MODEL_DIR, L3_TRAJECTORY_ENCODER_FILE),
        )
    with open(os.path.join(MODEL_DIR, L3_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    print(
        f"  [L3] strict OOT split: train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} "
        f"(tune={int(val_tune_mask.sum()):,}, report={int(val_report_mask.sum()):,}) holdout={int(test_mask.sum()):,}",
        flush=True,
    )
    if test_mask.any():
        prob = _apply_l3_exit_calibrator(exit_model.predict(X_lgb[test_mask]), exit_calibrator)
        print(f"  [L3] test mean exit prob={float(np.mean(prob)):.4f}", flush=True)
    print(f"  [L3] meta saved -> {os.path.join(MODEL_DIR, L3_META_FILE)}", flush=True)
    bundle_models: dict[str, Any] = {"exit": exit_model, "value": value_model}
    if exit_calibrator is not None:
        bundle_models["exit_calibrator"] = exit_calibrator
    return L3TrainingBundle(models=bundle_models, meta=meta)


def load_l3_exit_manager() -> tuple[dict[str, Any], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L3_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L3_SCHEMA_VERSION:
        raise RuntimeError(
            f"L3 schema mismatch: artifact has {meta.get('schema_version')} but code expects {L3_SCHEMA_VERSION}. "
            f"Retrain L3 so artifacts match schema {L3_SCHEMA_VERSION}."
        )
    model_files = meta.get("model_files", {})
    models: dict[str, Any] = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in model_files.items()
        if name != "exit_calibrator"
    }
    exit_calib_file = model_files.get("exit_calibrator")
    if exit_calib_file:
        with open(os.path.join(MODEL_DIR, exit_calib_file), "rb") as f:
            models["exit_calibrator"] = pickle.load(f)
    return models, meta


def load_l3_trajectory_encoder_for_infer(meta: dict[str, Any]) -> tuple[Any, L3TrajectoryConfig | None]:
    """Load GRU encoder when meta['l3_hybrid'] is True; else (None, None)."""
    if not meta.get("l3_hybrid"):
        return None, None
    import torch

    from core.trainers.l3_trajectory_hybrid import L3TrajectoryEncoder

    cfg_d = meta.get("l3_traj_cfg")
    if not cfg_d:
        return None, None
    cfg = L3TrajectoryConfig(**cfg_d)
    enc = L3TrajectoryEncoder(cfg)
    path = os.path.join(MODEL_DIR, meta.get("l3_traj_encoder_file", L3_TRAJECTORY_ENCODER_FILE))
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc, cfg

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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.data_prep import *
from core.trainers.l4_sequence import L4ExitSequenceModel
from core.trainers.layer2b_quality import _apply_cp_skip
from core.trainers.layer3_sizer import (
    _layer3_fill_regime_calibrated,
    _layer3_attach_regime_probs_to_work,
    _layer3_attach_regime_raw_probs_to_work,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = _options_target_config()
    n = len(work)
    if n == 0:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
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
    split_code = np.full(n, -1, dtype=np.int8)
    split_code[times < np.datetime64(TRAIN_END)] = 0
    split_code[(times >= np.datetime64(TRAIN_END)) & (times < np.datetime64(CAL_END))] = 1
    split_code[(times >= np.datetime64(CAL_END)) & (times < np.datetime64(TEST_END))] = 2
    max_hold = int(cfg["max_hold_bars"])
    rows_x: list[np.ndarray] = []
    rows_exit: list[int] = []
    rows_value: list[float] = []
    rows_time: list[np.datetime64] = []
    seq_starts: list[int] = []
    seq_ends: list[int] = []
    seq_times: list[np.datetime64] = []

    candidate_entries = np.where(gate_mask & (side_arr != 0) & (opt_exit > 0))[0]
    for i in candidate_entries:
        if i + 1 >= n or symbols[i + 1] != symbols[i]:
            continue
        entry_split = int(split_code[i])
        if entry_split < 0:
            continue
        entry_price = float(open_px[i + 1])
        entry_atr = float(safe_atr[i])
        side = float(side_arr[i])
        live_mfe = 0.0
        live_mae = 0.0
        max_j = min(int(sym_end[i]), i + max_hold)
        opt_bar = max(1, int(opt_exit[i]))
        trade_end_j = min(max_j, i + opt_bar)
        if trade_end_j <= i or int(split_code[trade_end_j]) != entry_split:
            continue
        opt_edge = float(opt_net[i])
        seq_start = len(rows_x)
        for j in range(i + 1, trade_end_j + 1):
            if int(split_code[j]) != entry_split:
                break
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
        seq_end = len(rows_x)
        if seq_end > seq_start:
            seq_starts.append(seq_start)
            seq_ends.append(seq_end)
            seq_times.append(rows_time[seq_start])

    if not rows_x:
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype="datetime64[ns]"),
        )
    return (
        np.asarray(rows_x, dtype=np.float32),
        np.asarray(rows_exit, dtype=np.int32),
        np.asarray(rows_value, dtype=np.float32),
        np.asarray(rows_time),
        np.asarray(seq_starts, dtype=np.int32),
        np.asarray(seq_ends, dtype=np.int32),
        np.asarray(seq_times),
    )


def _resolve_l4_seq_device() -> torch.device:
    raw = os.environ.get("TORCH_DEVICE", "").strip()
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _slice_policy_sequences(
    X_policy: np.ndarray,
    y_exit: np.ndarray,
    y_value: np.ndarray,
    seq_starts: np.ndarray,
    seq_ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start, end in zip(seq_starts.tolist(), seq_ends.tolist()):
        if end <= start:
            continue
        seqs.append((
            X_policy[start:end].astype(np.float32, copy=False),
            y_exit[start:end].astype(np.float32, copy=False),
            y_value[start:end].astype(np.float32, copy=False),
        ))
    return seqs


def _collate_policy_sequences(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [torch.from_numpy(item[0]) for item in batch]
    ys_exit = [torch.from_numpy(item[1]) for item in batch]
    ys_value = [torch.from_numpy(item[2]) for item in batch]
    x_pad = pad_sequence(xs, batch_first=True)
    exit_pad = pad_sequence(ys_exit, batch_first=True)
    value_pad = pad_sequence(ys_value, batch_first=True)
    mask_pad = pad_sequence(
        [torch.ones(len(item[1]), dtype=torch.bool) for item in batch],
        batch_first=True,
        padding_value=False,
    )
    return x_pad, exit_pad, value_pad, mask_pad


def _evaluate_l4_sequence_model(
    model: L4ExitSequenceModel,
    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    if not seqs:
        return {
            "exit_pred": np.empty(0, dtype=np.float32),
            "value_pred": np.empty(0, dtype=np.float32),
            "y_exit": np.empty(0, dtype=np.float32),
            "y_value": np.empty(0, dtype=np.float32),
            "loss": float("nan"),
        }
    loader = DataLoader(seqs, batch_size=batch_size, shuffle=False, collate_fn=_collate_policy_sequences)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    huber = torch.nn.SmoothL1Loss(reduction="sum")
    model.eval()
    losses = []
    exit_preds: list[np.ndarray] = []
    value_preds: list[np.ndarray] = []
    y_exit_all: list[np.ndarray] = []
    y_value_all: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb_exit, yb_value, mask in loader:
            xb = ((xb.to(device) - mean_t) / std_t)
            yb_exit = yb_exit.to(device)
            yb_value = yb_value.to(device)
            mask = mask.to(device)
            exit_logits, value_pred, _ = model(xb)
            valid_exit = exit_logits[mask]
            valid_value = value_pred[mask]
            target_exit = yb_exit[mask]
            target_value = yb_value[mask]
            loss = bce(valid_exit, target_exit) + 0.5 * huber(valid_value, target_value)
            losses.append(float(loss.item() / max(int(mask.sum().item()), 1)))
            exit_preds.append(torch.sigmoid(valid_exit).cpu().numpy())
            value_preds.append(valid_value.cpu().numpy())
            y_exit_all.append(target_exit.cpu().numpy())
            y_value_all.append(target_value.cpu().numpy())
    return {
        "exit_pred": np.concatenate(exit_preds) if exit_preds else np.empty(0, dtype=np.float32),
        "value_pred": np.concatenate(value_preds) if value_preds else np.empty(0, dtype=np.float32),
        "y_exit": np.concatenate(y_exit_all) if y_exit_all else np.empty(0, dtype=np.float32),
        "y_value": np.concatenate(y_value_all) if y_value_all else np.empty(0, dtype=np.float32),
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


def _train_l4_sequence_model(
    train_seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    input_size: int,
    exit_prob_thr: float,
) -> tuple[L4ExitSequenceModel | None, dict[str, Any]]:
    if not train_seqs or not val_seqs:
        return None, {"trained": False, "reason": "missing train/val sequences"}

    flat_train = np.vstack([seq[0] for seq in train_seqs]).astype(np.float32, copy=False)
    mean = flat_train.mean(axis=0).astype(np.float32, copy=False)
    std = flat_train.std(axis=0).astype(np.float32, copy=False)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32, copy=False)

    device = _resolve_l4_seq_device()
    hidden_size = int(os.environ.get("L4_SEQ_HIDDEN", "48"))
    dropout = float(os.environ.get("L4_SEQ_DROPOUT", "0.10"))
    batch_size = max(8, int(os.environ.get("L4_SEQ_BATCH", "128")))
    max_epochs = 8 if FAST_TRAIN_MODE else int(os.environ.get("L4_SEQ_EPOCHS", "18"))
    patience = 3 if FAST_TRAIN_MODE else int(os.environ.get("L4_SEQ_PATIENCE", "5"))

    model = L4ExitSequenceModel(input_size=input_size, hidden_size=hidden_size, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("L4_SEQ_LR", "0.001")), weight_decay=1e-4)
    train_loader = DataLoader(train_seqs, batch_size=batch_size, shuffle=True, collate_fn=_collate_policy_sequences)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb_exit, yb_value, mask in train_loader:
            xb = ((xb.to(device) - mean_t) / std_t)
            yb_exit = yb_exit.to(device)
            yb_value = yb_value.to(device)
            mask = mask.to(device)
            opt.zero_grad()
            exit_logits, value_pred, _ = model(xb)
            valid_exit = exit_logits[mask]
            valid_value = value_pred[mask]
            target_exit = yb_exit[mask]
            target_value = yb_value[mask]
            loss_exit = torch.nn.functional.binary_cross_entropy_with_logits(valid_exit, target_exit)
            loss_value = torch.nn.functional.smooth_l1_loss(valid_value, target_value)
            loss = loss_exit + 0.5 * loss_value
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_eval = _evaluate_l4_sequence_model(model, val_seqs, mean, std, device, batch_size)
        val_loss = float(val_eval["loss"])
        print(f"  [L4 seq] epoch={epoch + 1}/{max_epochs}  val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        return None, {"trained": False, "reason": "no valid checkpoint"}

    model.load_state_dict(best_state)
    test_eval = _evaluate_l4_sequence_model(model, test_seqs, mean, std, device, batch_size)
    seq_exit_pred = np.asarray(test_eval["exit_pred"], dtype=np.float32)
    seq_value_pred = np.asarray(test_eval["value_pred"], dtype=np.float32)
    seq_y_exit = np.asarray(test_eval["y_exit"], dtype=np.float32)
    seq_y_value = np.asarray(test_eval["y_value"], dtype=np.float32)
    seq_auc = float(roc_auc_score(seq_y_exit, seq_exit_pred)) if len(np.unique(seq_y_exit)) > 1 else float("nan")
    seq_acc = float(((seq_exit_pred >= exit_prob_thr).astype(np.int32) == seq_y_exit.astype(np.int32)).mean()) if len(seq_y_exit) else float("nan")
    seq_mae = float(np.mean(np.abs(seq_value_pred - seq_y_value))) if len(seq_y_value) else float("nan")
    seq_corr = float(np.corrcoef(seq_value_pred, seq_y_value)[0, 1]) if len(seq_value_pred) > 2 else float("nan")

    print("\n  Layer 4 Sequence Test Metrics")
    print(f"    Exit AUC:      {seq_auc:.4f}")
    print(f"    Exit Acc@thr:  {seq_acc:.4f}  (thr={exit_prob_thr:.2f})")
    print(f"    Value MAE:     {seq_mae:.4f}")
    print(f"    Value Corr:    {seq_corr:.4f}")

    return model, {
        "trained": True,
        "device": str(device),
        "hidden_size": hidden_size,
        "dropout": dropout,
        "input_mean": mean,
        "input_std": std,
        "test_exit_auc": seq_auc,
        "test_exit_acc": seq_acc,
        "test_value_mae": seq_mae,
        "test_value_corr": seq_corr,
    }


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
    time_arr = work["time_key"].values
    train_rows = time_arr < np.datetime64(TRAIN_END)
    cal_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    raw_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    _layer3_fill_regime_calibrated(
        regime_model, regime_calibrators, work, cal_regime, chunk,
        raw_out=raw_regime,
        tqdm_desc="Layer4 regime→cal",
    )
    _layer3_attach_regime_probs_to_work(work, cal_regime)
    _layer3_attach_regime_raw_probs_to_work(work, raw_regime)

    garch_cols = sorted([c for c in work.columns if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}])
    layer2_feats = trade_quality_models["feature_cols"]

    p_long_gate = np.empty(n, dtype=np.float32)
    p_short_gate = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(
        trade_quality_models, work, layer2_feats, p_long_gate, p_short_gate, chunk,
        tqdm_desc="Layer4 trade stack",
    )

    tcn_transition_prob_all = work["tcn_transition_prob"].values.astype(np.float32) if "tcn_transition_prob" in work.columns else None
    non_train_rows = ~train_rows
    if non_train_rows.any():
        tcn_transition_prob_eval = tcn_transition_prob_all[non_train_rows] if tcn_transition_prob_all is not None else None
        p_long_adj, _ = _apply_cp_skip(cal_regime[non_train_rows], p_long_gate[non_train_rows], thr_cp, tcn_transition_prob_eval)
        p_short_adj, _ = _apply_cp_skip(cal_regime[non_train_rows], p_short_gate[non_train_rows], thr_cp, tcn_transition_prob_eval)
        p_long_gate[non_train_rows] = p_long_adj
        p_short_gate[non_train_rows] = p_short_adj

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
    regime_feat = cal_regime.copy()
    regime_feat[train_rows] = raw_regime[train_rows]
    inter_blk = (l2b_opp.astype(np.float64)[:, None] * regime_feat.astype(np.float64)).astype(np.float32, copy=False)
    triplet_blk = np.hstack([l2b_opp.reshape(-1, 1), l2b_mfe.reshape(-1, 1), l2b_mae.reshape(-1, 1)]).astype(np.float32, copy=False)
    sc_conf = regime_feat.max(axis=1, keepdims=True).astype(np.float32, copy=False)
    regime_blk = np.hstack([regime_feat, sc_conf]).astype(np.float32, copy=False)
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
    train_thr_long = float(os.environ.get("L4_TRAIN_GATE_THRESHOLD_LONG", "0.50"))
    train_thr_short = float(os.environ.get("L4_TRAIN_GATE_THRESHOLD_SHORT", "0.50"))
    gate_mask = np.zeros(n, dtype=bool)
    gate_mask[train_rows] = (p_long_gate[train_rows] >= train_thr_long) | (p_short_gate[train_rows] >= train_thr_short)
    gate_mask[non_train_rows] = (p_long_gate[non_train_rows] >= thr_long) | (p_short_gate[non_train_rows] >= thr_short)

    X_policy, y_exit, y_value, t_state, seq_starts, seq_ends, seq_times = _build_exit_policy_dataset(
        work,
        base_X,
        gate_mask,
        side_arr,
        policy_feature_cols,
        exit_epsilon_atr=exit_eps,
    )
    if len(X_policy) == 0:
        raise RuntimeError("Layer 4 policy dataset is empty. Check gating thresholds and labeled optimal exit columns.")

    train_mask = t_state < np.datetime64(TRAIN_END)
    val_mask = (t_state >= np.datetime64(TRAIN_END)) & (t_state < np.datetime64(CAL_END))
    test_mask = (t_state >= np.datetime64(CAL_END)) & (t_state < np.datetime64(TEST_END))

    # Fallback: if a window is unexpectedly empty, split all pre-test rows chronologically.
    if (not train_mask.any()) or (not val_mask.any()):
        pretest_mask = t_state < np.datetime64(CAL_END)
        pretest_indices = np.where(pretest_mask)[0]
        split_idx = int(len(pretest_indices) * 0.85)
        split_idx = min(max(split_idx, 1), max(len(pretest_indices) - 1, 1))
        train_mask = np.zeros(len(X_policy), dtype=bool)
        val_mask = np.zeros(len(X_policy), dtype=bool)
        if len(pretest_indices) >= 2:
            train_mask[pretest_indices[:split_idx]] = True
            val_mask[pretest_indices[split_idx:]] = True
        else:
            train_mask = pretest_mask.copy()
            val_mask = np.zeros(len(X_policy), dtype=bool)

    X_train, X_val, X_test = X_policy[train_mask], X_policy[val_mask], X_policy[test_mask]
    y_exit_train, y_exit_val, y_exit_test = y_exit[train_mask], y_exit[val_mask], y_exit[test_mask]
    y_value_train = np.clip(y_value[train_mask], -3.0, 3.0)
    y_value_val = np.clip(y_value[val_mask], -3.0, 3.0)
    y_value_test = np.clip(y_value[test_mask], -3.0, 3.0)

    seq_train_mask = seq_times < np.datetime64(TRAIN_END)
    seq_val_mask = (seq_times >= np.datetime64(TRAIN_END)) & (seq_times < np.datetime64(CAL_END))
    seq_test_mask = (seq_times >= np.datetime64(CAL_END)) & (seq_times < np.datetime64(TEST_END))
    if (not seq_train_mask.any()) or (not seq_val_mask.any()):
        pretest_seq_mask = seq_times < np.datetime64(CAL_END)
        pretest_seq_idx = np.where(pretest_seq_mask)[0]
        split_idx = int(len(pretest_seq_idx) * 0.85)
        split_idx = min(max(split_idx, 1), max(len(pretest_seq_idx) - 1, 1))
        seq_train_mask = np.zeros(len(seq_times), dtype=bool)
        seq_val_mask = np.zeros(len(seq_times), dtype=bool)
        if len(pretest_seq_idx) >= 2:
            seq_train_mask[pretest_seq_idx[:split_idx]] = True
            seq_val_mask[pretest_seq_idx[split_idx:]] = True
        else:
            seq_train_mask = pretest_seq_mask.copy()
            seq_val_mask = np.zeros(len(seq_times), dtype=bool)

    train_seqs = _slice_policy_sequences(X_policy, y_exit, np.clip(y_value, -3.0, 3.0), seq_starts[seq_train_mask], seq_ends[seq_train_mask])
    val_seqs = _slice_policy_sequences(X_policy, y_exit, np.clip(y_value, -3.0, 3.0), seq_starts[seq_val_mask], seq_ends[seq_val_mask])
    test_seqs = _slice_policy_sequences(X_policy, y_exit, np.clip(y_value, -3.0, 3.0), seq_starts[seq_test_mask], seq_ends[seq_test_mask])

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
        f"  Policy windows — train:<{TRAIN_END}  val:[{TRAIN_END}, {CAL_END})  "
        f"test:[{CAL_END}, {TEST_END})"
    )
    print(
        f"  Policy sequences — train={len(train_seqs):,} val={len(val_seqs):,} test={len(test_seqs):,}"
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

    seq_model, seq_meta = _train_l4_sequence_model(
        train_seqs,
        val_seqs,
        test_seqs,
        input_size=len(policy_feature_cols),
        exit_prob_thr=exit_prob_thr,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    exit_file = "exit_policy_exit.txt"
    value_file = "exit_policy_value.txt"
    seq_file = "exit_policy_seq.pt"
    model_exit.save_model(os.path.join(MODEL_DIR, exit_file))
    model_value.save_model(os.path.join(MODEL_DIR, value_file))
    if seq_model is not None and seq_meta.get("trained"):
        torch.save(seq_model.state_dict(), os.path.join(MODEL_DIR, seq_file))

    meta = {
        "l4_schema": 5 if seq_model is not None and seq_meta.get("trained") else 4,
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
        "exit_backend": os.environ.get("L4_DEFAULT_EXIT_BACKEND", "tree"),
        "value_backend": os.environ.get("L4_DEFAULT_VALUE_BACKEND", "tree"),
        "available_backends": ["tree"] + (["gru"] if seq_model is not None and seq_meta.get("trained") else []),
        "model_files": {
            "exit": exit_file,
            "value": value_file,
        },
    }
    if seq_model is not None and seq_meta.get("trained"):
        meta["model_files"]["seq"] = seq_file
        meta["seq_hidden_size"] = int(seq_meta["hidden_size"])
        meta["seq_dropout"] = float(seq_meta["dropout"])
        meta["seq_device"] = str(seq_meta["device"])
        meta["seq_norm_mean"] = np.asarray(seq_meta["input_mean"], dtype=np.float32).tolist()
        meta["seq_norm_std"] = np.asarray(seq_meta["input_std"], dtype=np.float32).tolist()
        meta["seq_metrics"] = {
            "exit_auc": float(seq_meta["test_exit_auc"]),
            "exit_acc": float(seq_meta["test_exit_acc"]),
            "value_mae": float(seq_meta["test_value_mae"]),
            "value_corr": float(seq_meta["test_value_corr"]),
        }
    with open(os.path.join(MODEL_DIR, "exit_manager_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    if seq_model is not None and seq_meta.get("trained"):
        print(f"\n  Layer 4 Models saved → {MODEL_DIR}/{exit_file}, {value_file}, {seq_file}")
    else:
        print(f"\n  Layer 4 Models saved → {MODEL_DIR}/{exit_file}, {value_file}")
    print(f"  Layer 4 Meta saved  → {MODEL_DIR}/exit_manager_meta.pkl")
    return {"exit": model_exit, "value": model_value, "seq": seq_model, "meta": meta}



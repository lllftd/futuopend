from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from core.trainers.val_metrics_extra import pearson_corr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 3:
        return float("nan")
    s = pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman")
    return float(s) if np.isfinite(s) else float("nan")


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    m = (y_true >= 0) & (y_true <= 1) & (y_pred >= 0) & (y_pred <= 1)
    if not m.any():
        return float("nan")
    tp = int(np.sum((y_true == 1) & (y_pred == 1) & m))
    fp = int(np.sum((y_true == 0) & (y_pred == 1) & m))
    fn = int(np.sum((y_true == 1) & (y_pred == 0) & m))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec <= 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def evaluate_l1c(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Val metrics for binary direction: probs vs labels; spearman vs raw forward return."""
    model.eval()
    probs_l: list[np.ndarray] = []
    y_bin_l: list[np.ndarray] = []
    y_ret_l: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].cpu().numpy().ravel()
            yr = batch[3].cpu().numpy().ravel()
            logits = model(xb)
            pr = torch.sigmoid(logits.view(-1)).cpu().numpy()
            probs_l.append(pr.astype(np.float64))
            y_bin_l.append(yb.astype(np.float64))
            y_ret_l.append(yr.astype(np.float64))
    prob = np.concatenate(probs_l)
    y_bin = np.concatenate(y_bin_l)
    y_ret = np.concatenate(y_ret_l)
    fin = np.isfinite(prob) & np.isfinite(y_bin) & np.isfinite(y_ret)
    prob = prob[fin]
    y_bin = y_bin[fin]
    y_ret = y_ret[fin]
    score = 2.0 * prob - 1.0
    y_int = (y_bin > 0.5).astype(np.int64)
    pred_int = (prob >= 0.5).astype(np.int64)

    acc = float(np.mean(y_int == pred_int)) if y_int.size else float("nan")
    f1 = _binary_f1(y_int, pred_int)
    pear = pearson_corr(prob, y_bin.astype(np.float64))
    spear_pb = _safe_spearman(prob, y_bin)
    spear_ret = _safe_spearman(prob, y_ret)
    spear_score_ret = _safe_spearman(score, y_ret)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_int, prob)) if len(np.unique(y_int)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    abs_ret = np.abs(y_ret)
    med = float(np.median(abs_ret)) if abs_ret.size else 0.0
    strong = abs_ret > med
    acc_strong = float(np.mean(pred_int[strong] == y_int[strong])) if np.any(strong) else float("nan")
    confident = np.abs(prob - 0.5) > 0.2
    acc_conf = float(np.mean(pred_int[confident] == y_int[confident])) if np.any(confident) else float("nan")

    return {
        "binary_accuracy": acc,
        "binary_f1": float(f1) if np.isfinite(f1) else float("nan"),
        "auc_up": auc,
        "pearson_prob_label": float(np.nan_to_num(pear, nan=0.0)),
        "spearman_prob_label": float(spear_pb) if np.isfinite(spear_pb) else float("nan"),
        "spearman_prob_raw_return": float(spear_ret) if np.isfinite(spear_ret) else float("nan"),
        "spearman_score_raw_return": float(spear_score_ret) if np.isfinite(spear_score_ret) else float("nan"),
        "acc_abs_ret_gt_median": acc_strong,
        "acc_prob_confident": acc_conf,
        "n": int(len(prob)),
    }


def print_l1c_eval_report(metrics: dict[str, Any]) -> None:
    print(
        f"  [L1c] val binary_acc={metrics.get('binary_accuracy', float('nan')):.4f}  "
        f"binary_f1={metrics.get('binary_f1', float('nan')):.4f}  auc={metrics.get('auc_up', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"  [L1c] val spearman: prob~label={metrics.get('spearman_prob_label', float('nan')):.4f}  "
        f"prob~raw_ret={metrics.get('spearman_prob_raw_return', float('nan')):.4f}  "
        f"score~raw_ret={metrics.get('spearman_score_raw_return', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"  [L1c] val subsets: acc(|ret|>median)={metrics.get('acc_abs_ret_gt_median', float('nan')):.4f}  "
        f"acc(|p-0.5|>0.2)={metrics.get('acc_prob_confident', float('nan')):.4f}",
        flush=True,
    )

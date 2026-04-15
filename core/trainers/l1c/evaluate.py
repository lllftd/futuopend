from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.trainers.val_metrics_extra import brier_binary, ece_binary, pearson_corr


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


def _binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _coverage_accuracy_table(
    prob: np.ndarray,
    y_int: np.ndarray,
    *,
    thresholds: tuple[float, ...] | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pred_int = (np.asarray(prob, dtype=np.float64).ravel() >= 0.5).astype(np.int64)
    y = np.asarray(y_int, dtype=np.int64).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    conf = np.abs(p - 0.5)
    n = len(p)
    if thresholds is None:
        qs = [0.50, 0.65, 0.75, 0.85, 0.90]
        thresholds = tuple(float(np.quantile(conf, q)) for q in qs) if conf.size else (0.05, 0.10, 0.15, 0.20, 0.25)
    for thr in thresholds:
        mask = conf > float(thr)
        cnt = int(np.sum(mask))
        rows.append(
            {
                "threshold": float(thr),
                "n": cnt,
                "coverage": float(cnt / n) if n > 0 else float("nan"),
                "accuracy": _binary_accuracy(y[mask], pred_int[mask]) if cnt > 0 else float("nan"),
                "mean_confidence": float(np.mean(conf[mask])) if cnt > 0 else float("nan"),
                "mean_prob": float(np.mean(p[mask])) if cnt > 0 else float("nan"),
                "up_rate": float(np.mean(y[mask])) if cnt > 0 else float("nan"),
            }
        )
    return rows


def _probability_bin_table(prob: np.ndarray, y_int: np.ndarray, *, n_bins: int = 10) -> list[dict[str, float]]:
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y_int, dtype=np.int64).ravel()
    if p.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for idx in range(n_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "n": cnt,
                "coverage": float(cnt / p.size),
                "mean_prob": float(np.mean(p[mask])) if cnt > 0 else float("nan"),
                "up_rate": float(np.mean(y[mask])) if cnt > 0 else float("nan"),
                "accuracy": _binary_accuracy(y[mask], (p[mask] >= 0.5).astype(np.int64)) if cnt > 0 else float("nan"),
            }
        )
    return rows


def _quintile_summary(prob: np.ndarray, y_int: np.ndarray) -> list[dict[str, float]]:
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y_int, dtype=np.int64).ravel()
    if p.size == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    y_s = y[order]
    n = len(p_s)
    rows: list[dict[str, float]] = []
    for q in range(5):
        lo = int(np.floor(q * n / 5))
        hi = int(np.floor((q + 1) * n / 5))
        if hi <= lo:
            continue
        pq = p_s[lo:hi]
        yq = y_s[lo:hi]
        rows.append(
            {
                "quintile": int(q + 1),
                "n": int(len(pq)),
                "mean_prob": float(np.mean(pq)),
                "up_rate": float(np.mean(yq)),
                "accuracy": _binary_accuracy(yq, (pq >= 0.5).astype(np.int64)),
            }
        )
    return rows


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
            out = model(xb)
            logits = out["direction_logits"] if isinstance(out, dict) else out
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
    brier = brier_binary(y_int.astype(np.float64), prob)
    ece = ece_binary(y_int.astype(np.float64), prob)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_int, prob)) if len(np.unique(y_int)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    abs_ret = np.abs(y_ret)
    med = float(np.median(abs_ret)) if abs_ret.size else 0.0
    strong = abs_ret > med
    strong_n = int(np.sum(strong))
    acc_strong = float(np.mean(pred_int[strong] == y_int[strong])) if strong_n > 0 else float("nan")
    conf_abs = np.abs(prob - 0.5)
    conf_target = float(np.clip(float(os.environ.get("L1C_CONFIDENT_COVERAGE_TARGET", "0.20")), 0.05, 0.80))
    conf_min_margin = float(np.clip(float(os.environ.get("L1C_CONFIDENT_MIN_MARGIN", "0.05")), 0.01, 0.25))
    conf_mode = (os.environ.get("L1C_CONF_MODE", "cost_based") or "cost_based").strip().lower()
    conf_thr_q = float(np.quantile(conf_abs, 1.0 - conf_target)) if conf_abs.size else 0.2
    avg_move = float(np.median(np.abs(y_ret[np.isfinite(y_ret)]))) if np.isfinite(y_ret).any() else 0.0
    tx_cost = float(max(0.0, float(os.environ.get("L1C_TX_COST_RATE", "0.0005"))))
    win_move = float(np.median(y_ret[y_ret > 0])) if np.any(y_ret > 0) else avg_move
    loss_move = float(np.median(np.abs(y_ret[y_ret < 0]))) if np.any(y_ret < 0) else avg_move
    if not np.isfinite(win_move) or win_move <= 1e-8:
        win_move = max(avg_move, 1e-6)
    if not np.isfinite(loss_move) or loss_move <= 1e-8:
        loss_move = max(avg_move, 1e-6)
    cost_based_thr = float(np.clip((loss_move + tx_cost) / max(win_move + loss_move, 1e-8), 0.5, 0.99))
    conf_thr_cost = float(max(conf_min_margin, cost_based_thr - 0.5))
    conf_thr = float(max(conf_thr_q, conf_min_margin))
    conf_fallback_reason = ""
    if avg_move <= 1e-6 or (win_move + loss_move) <= 2e-6:
        conf_fallback_reason = "near_zero_move_use_coverage"
        conf_thr = float(max(conf_thr_q, conf_min_margin))
    if conf_mode == "cost_based":
        conf_thr = float(max(conf_thr_cost, conf_min_margin)) if not conf_fallback_reason else float(max(conf_thr_q, conf_min_margin))
    confident = conf_abs > conf_thr
    conf_n = int(np.sum(confident))
    acc_conf = float(np.mean(pred_int[confident] == y_int[confident])) if conf_n > 0 else float("nan")
    conf_table = _coverage_accuracy_table(prob, y_int)
    prob_bin_table = _probability_bin_table(prob, y_int, n_bins=10)
    quintiles = _quintile_summary(prob, y_int)

    return {
        "binary_accuracy": acc,
        "binary_f1": float(f1) if np.isfinite(f1) else float("nan"),
        "auc_up": auc,
        "brier": float(brier) if np.isfinite(brier) else float("nan"),
        "ece": float(ece) if np.isfinite(ece) else float("nan"),
        "pearson_prob_label": float(np.nan_to_num(pear, nan=0.0)),
        "spearman_prob_label": float(spear_pb) if np.isfinite(spear_pb) else float("nan"),
        "spearman_prob_raw_return": float(spear_ret) if np.isfinite(spear_ret) else float("nan"),
        "spearman_score_raw_return": float(spear_score_ret) if np.isfinite(spear_score_ret) else float("nan"),
        "acc_abs_ret_gt_median": acc_strong,
        "n_abs_ret_gt_median": strong_n,
        "acc_prob_confident": acc_conf,
        "n_prob_confident": conf_n,
        "coverage_prob_confident": float(conf_n / len(prob)) if len(prob) > 0 else float("nan"),
        "confident_threshold": float(conf_thr),
        "confident_threshold_quantile": float(conf_thr_q),
        "confident_min_margin": float(conf_min_margin),
        "confident_mode": str(conf_mode),
        "confident_threshold_cost_based": float(conf_thr_cost),
        "cost_based_probability_threshold": float(conf_thr_cost + 0.5),
        "confident_fallback_reason": str(conf_fallback_reason),
        "tx_cost_rate": float(tx_cost),
        "avg_move": float(avg_move),
        "win_move": float(win_move),
        "loss_move": float(loss_move),
        "confidence_threshold_table": conf_table,
        "probability_bin_table": prob_bin_table,
        "probability_quintiles": quintiles,
        "n": int(len(prob)),
    }


def print_l1c_eval_report(metrics: dict[str, Any]) -> None:
    print(
        f"  [L1c] val binary_acc={metrics.get('binary_accuracy', float('nan')):.4f}  "
        f"binary_f1={metrics.get('binary_f1', float('nan')):.4f}  auc={metrics.get('auc_up', float('nan')):.4f}  "
        f"Brier={metrics.get('brier', float('nan')):.4f}  ECE={metrics.get('ece', float('nan')):.4f}",
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
        f"n={metrics.get('n_abs_ret_gt_median', 0):,}  "
        f"acc(|p-0.5|>{metrics.get('confident_threshold', 0.2):.4f})={metrics.get('acc_prob_confident', float('nan')):.4f}  "
        f"n={metrics.get('n_prob_confident', 0):,}  coverage={metrics.get('coverage_prob_confident', float('nan')):.2%}",
        flush=True,
    )
    print(
        f"  [L1c] confident threshold policy: quantile_thr={metrics.get('confident_threshold_quantile', float('nan')):.4f}  "
        f"min_margin={metrics.get('confident_min_margin', float('nan')):.4f}  "
        f"cost_thr={metrics.get('confident_threshold_cost_based', float('nan')):.4f}  "
        f"mode={metrics.get('confident_mode', 'coverage')}",
        flush=True,
    )
    print(
        f"  [L1c] confidence economics: tx_cost={metrics.get('tx_cost_rate', float('nan')):.6f}  "
        f"avg_move={metrics.get('avg_move', float('nan')):.6f}  "
        f"win={metrics.get('win_move', float('nan')):.6f}  loss={metrics.get('loss_move', float('nan')):.6f}  "
        f"p_thr={metrics.get('cost_based_probability_threshold', float('nan')):.4f}  "
        f"fallback={metrics.get('confident_fallback_reason', '')}",
        flush=True,
    )
    conf_rows = metrics.get("confidence_threshold_table") or []
    if conf_rows:
        print("  [L1c] confidence coverage/accuracy:", flush=True)
        for row in conf_rows:
            print(
                f"    |p-0.5|>{row['threshold']:.2f}: n={int(row['n']):,}  coverage={row['coverage']:.2%}  "
                f"acc={row['accuracy']:.4f}  mean|p-0.5|={row['mean_confidence']:.4f}",
                flush=True,
            )
    quint_rows = metrics.get("probability_quintiles") or []
    if quint_rows:
        print("  [L1c] probability quintiles (monotonicity check):", flush=True)
        for row in quint_rows:
            print(
                f"    Q{int(row['quintile'])}: n={int(row['n']):,}  mean_prob={row['mean_prob']:.4f}  "
                f"up_rate={row['up_rate']:.4f}  acc={row['accuracy']:.4f}",
                flush=True,
            )
    bin_rows = metrics.get("probability_bin_table") or []
    if bin_rows:
        print("  [L1c] reliability by probability bin:", flush=True)
        for row in bin_rows:
            mean_prob = row["mean_prob"]
            up_rate = row["up_rate"]
            mean_prob_s = f"{mean_prob:.4f}" if np.isfinite(mean_prob) else "nan"
            up_rate_s = f"{up_rate:.4f}" if np.isfinite(up_rate) else "nan"
            print(
                f"    [{row['lo']:.1f},{row['hi']:.1f}{']' if row['hi'] >= 1.0 else ')'}: "
                f"n={int(row['n']):,}  coverage={row['coverage']:.2%}  mean_prob={mean_prob_s}  up_rate={up_rate_s}",
                flush=True,
            )

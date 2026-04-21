"""Extra validation metrics: Brier, ECE, tail MAE, multiclass helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def brier_binary(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).ravel().astype(np.float64)
    p = np.clip(np.asarray(p).ravel().astype(np.float64), 0.0, 1.0)
    return float(np.mean((y - p) ** 2))


def brier_multiclass(y: np.ndarray, P: np.ndarray, n_classes: int) -> float:
    y = np.asarray(y).astype(int).ravel()
    P = np.asarray(P, dtype=np.float64)
    n = len(y)
    if n == 0:
        return float("nan")
    oh = np.zeros((n, n_classes), dtype=np.float64)
    oh[np.arange(n), np.clip(y, 0, n_classes - 1)] = 1.0
    return float(np.mean(np.sum((P - oh) ** 2, axis=1)))


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary labels {0,1}."""
    y_true = np.asarray(y_true).ravel().astype(int)
    p = np.clip(np.asarray(p).ravel().astype(np.float64), 1e-9, 1.0 - 1e-9)
    order = np.argsort(p)
    y_s = y_true[order]
    p_s = p[order]
    n = len(y_s)
    if n == 0:
        return float("nan")
    ece = 0.0
    bin_size = max(1, n // n_bins)
    for b in range(n_bins):
        sl = slice(b * bin_size, (b + 1) * bin_size if b < n_bins - 1 else n)
        if sl.start >= n:
            break
        yt = y_s[sl]
        pp = p_s[sl]
        if len(yt) == 0:
            continue
        acc = float(np.mean(yt))
        conf = float(np.mean(pp))
        w = len(yt) / n
        ece += w * abs(acc - conf)
    return float(ece)


def ece_multiclass_maxprob(y_true: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    """ECE: bins by max predicted probability; accuracy vs mean confidence per bin."""
    y_true = np.asarray(y_true).astype(int).ravel()
    P = np.asarray(P, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return float("nan")
    pred = np.argmax(P, axis=1)
    conf = np.max(P, axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (conf > lo) & (conf <= hi)
        else:
            m = (conf > lo) & (conf <= hi) | (conf == hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        w = cnt / n
        ece += w * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def tail_mae_truth_upper(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 90.0) -> float:
    """MAE on rows where y_true is at or above the given percentile of y_true."""
    y_true = np.asarray(y_true).ravel().astype(np.float64)
    y_pred = np.asarray(y_pred).ravel().astype(np.float64)
    if len(y_true) < 3:
        return float("nan")
    thr = np.percentile(y_true, percentile)
    m = y_true >= thr
    if not m.any():
        return float("nan")
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def regression_degen_flag(y_pred: np.ndarray, eps: float = 1e-5) -> tuple[float, bool]:
    std = float(np.std(np.asarray(y_pred).ravel()))
    return std, std < eps


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def flip_rate_sorted(probs: np.ndarray, time_sorted_idx: np.ndarray) -> float:
    """Mean |Δp| along time order (higher = chattier signal)."""
    if len(time_sorted_idx) < 2:
        return float("nan")
    s = probs[time_sorted_idx]
    return float(np.mean(np.abs(np.diff(s))))


def directional_accuracy_regression(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where sign(pred) matches sign(true), ignoring ~zero true."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = np.abs(y_true) > 1e-8
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[m]) == np.sign(y_pred[m])))

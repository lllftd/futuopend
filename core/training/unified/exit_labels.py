"""Exit / value supervision helpers for unified L2+L3 Phase-2 (optional path-level labels)."""

from __future__ import annotations

import numpy as np


def remaining_bars_to_episode_end(
    rows_entry: np.ndarray,
    hold_bars: np.ndarray,
) -> np.ndarray:
    """
    For each L3 policy row, number of hold bars from this bar to the end of the episode
    (0 on the last bar of the trade). ``hold_bars`` is used only to order rows within a trade
    (typically ``l3_hold_bars`` from the policy matrix).
    """
    re = np.asarray(rows_entry, dtype=np.int64)
    h = np.asarray(hold_bars, dtype=np.float64)
    n = int(re.size)
    out = np.zeros(n, dtype=np.float32)
    for eid in np.unique(re):
        m = re == eid
        idxs = np.flatnonzero(m)
        if idxs.size == 0:
            continue
        sub = h[idxs]
        order = idxs[np.argsort(sub, kind="mergesort")]
        l_g = int(order.size)
        for j, ix in enumerate(order):
            out[int(ix)] = float(l_g - 1 - j)
    return out


def urgency_from_y_exit(
    y_exit: np.ndarray,
    rows_entry: np.ndarray,
) -> np.ndarray:
    """
    Heuristic urgency in [0,1]: 1.0 on last hold bar of each trade, else follow exit label.
    """
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    re = np.asarray(rows_entry, dtype=np.int64).ravel()
    n = y.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    u = y.astype(np.float32)
    last_mark = np.zeros(n, dtype=bool)
    for eid in np.unique(re):
        m = re == eid
        idxs = np.flatnonzero(m)
        if idxs.size:
            last_mark[idxs[-1]] = True
    u[last_mark] = 1.0
    return np.clip(u, 0.0, 1.0)



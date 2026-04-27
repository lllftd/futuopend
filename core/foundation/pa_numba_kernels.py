"""
Hot loops for :mod:`core.foundation.pa_rules` — optional Numba acceleration.

Disable with env ``PA_NUMBA=0`` (or ``false`` / ``off``). If ``numba`` is not
installed, the same logic runs as plain Python (slower).
"""
from __future__ import annotations

import os

import numpy as np

_PA_NUMBA = (os.environ.get("PA_NUMBA", "1") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

try:
    if _PA_NUMBA:
        from numba import njit
    else:
        raise ImportError("PA_NUMBA disabled")
except ImportError:

    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        return _decorator


@njit(cache=True)
def _pullback_counting_core(
    highs: np.ndarray,
    lows: np.ndarray,
    direction: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_count = np.zeros(n, dtype=np.int32)
    l_count = np.zeros(n, dtype=np.int32)
    cur_h = 0
    cur_l = 0
    in_pullback_up = False
    in_pullback_down = False
    for i in range(1, n):
        if direction[i] == 1:
            if lows[i] < lows[i - 1]:
                in_pullback_up = True
                cur_h = 0
            if in_pullback_up and highs[i] > highs[i - 1]:
                cur_h += 1
            if cur_h >= 1 and highs[i] > highs[i - 1] and not (lows[i] < lows[i - 1]):
                pass
            h_count[i] = cur_h
            cur_l = 0
            in_pullback_down = False
        elif direction[i] == -1:
            if highs[i] > highs[i - 1]:
                in_pullback_down = True
                cur_l = 0
            if in_pullback_down and lows[i] < lows[i - 1]:
                cur_l += 1
            l_count[i] = cur_l
            cur_h = 0
            in_pullback_up = False
        else:
            cur_h = 0
            cur_l = 0
            in_pullback_up = False
            in_pullback_down = False
    return h_count, l_count


def pullback_counting_fast(
    highs: np.ndarray,
    lows: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.ascontiguousarray(highs, dtype=np.float64)
    lo = np.ascontiguousarray(lows, dtype=np.float64)
    d = np.ascontiguousarray(direction, dtype=np.int64)
    return _pullback_counting_core(h, lo, d, h.shape[0])

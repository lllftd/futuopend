"""
Hot loops for :mod:`core.pa_rules` — optional Numba acceleration.

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
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

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
    n = h.shape[0]
    return _pullback_counting_core(h, lo, d, n)


@njit(cache=True)
def _mag_bar_core(highs: np.ndarray, lows: np.ndarray, ema: np.ndarray, n: int) -> np.ndarray:
    is_mag = np.zeros(n, dtype=np.uint8)
    above_count = 0
    below_count = 0
    for i in range(n):
        if np.isnan(ema[i]):
            above_count = 0
            below_count = 0
            continue
        bar_above = lows[i] > ema[i]
        bar_below = highs[i] < ema[i]
        if bar_above:
            above_count += 1
            below_count = 0
        elif bar_below:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0
        if bar_above and above_count == 20:
            is_mag[i] = 1
        elif bar_below and below_count == 20:
            is_mag[i] = 1
    return is_mag


def mag_bar_fast(highs: np.ndarray, lows: np.ndarray, ema: np.ndarray) -> np.ndarray:
    h = np.ascontiguousarray(highs, dtype=np.float64)
    lo = np.ascontiguousarray(lows, dtype=np.float64)
    e = np.ascontiguousarray(ema, dtype=np.float64)
    return _mag_bar_core(h, lo, e, h.shape[0]).astype(bool)


@njit(cache=True)
def _fill_causal_stops_core(
    sl_confirm: np.ndarray,
    sl_val: np.ndarray,
    sh_confirm: np.ndarray,
    sh_val: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    stop_long = np.empty(n, dtype=np.float64)
    stop_short = np.empty(n, dtype=np.float64)
    for i in range(n):
        stop_long[i] = np.nan
        stop_short[i] = np.nan
    n_sl = sl_confirm.shape[0]
    n_sh = sh_confirm.shape[0]
    for i in range(n):
        if n_sl > 0:
            idx_l = np.searchsorted(sl_confirm, i, side="right") - 1
            if idx_l >= 0:
                stop_long[i] = sl_val[idx_l]
        if n_sh > 0:
            idx_h = np.searchsorted(sh_confirm, i, side="right") - 1
            if idx_h >= 0:
                stop_short[i] = sh_val[idx_h]
    return stop_long, stop_short


def fill_causal_stops_fast(
    sl_confirm: np.ndarray,
    sl_val: np.ndarray,
    sh_confirm: np.ndarray,
    sh_val: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    sl_c = np.ascontiguousarray(sl_confirm, dtype=np.int64)
    sl_v = np.ascontiguousarray(sl_val, dtype=np.float64)
    sh_c = np.ascontiguousarray(sh_confirm, dtype=np.int64)
    sh_v = np.ascontiguousarray(sh_val, dtype=np.float64)
    return _fill_causal_stops_core(sl_c, sl_v, sh_c, sh_v, n)


@njit(cache=True)
def _causal_support_resistance_core(
    close: np.ndarray,
    atr: np.ndarray,
    sh_confirm: np.ndarray,
    sh_bar: np.ndarray,
    sh_val: np.ndarray,
    sl_confirm: np.ndarray,
    sl_bar: np.ndarray,
    sl_val: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nearest_resist = np.full(n, np.nan)
    nearest_support = np.full(n, np.nan)
    sr_position = np.full(n, 0.5)
    retrace_50 = np.full(n, np.nan)
    at_50_retrace = np.zeros(n, dtype=np.bool_)
    round_num_dist = np.full(n, np.nan)

    nh = sh_confirm.shape[0]
    nl = sl_confirm.shape[0]
    sh_b = np.empty(nh, dtype=np.int64)
    sh_vbuf = np.empty(nh, dtype=np.float64)
    sl_b = np.empty(nl, dtype=np.int64)
    sl_vbuf = np.empty(nl, dtype=np.float64)

    sh_ptr = 0
    sl_ptr = 0
    n_sh = 0
    n_sl = 0

    for i in range(n):
        while sh_ptr < nh and sh_confirm[sh_ptr] <= i:
            sh_b[n_sh] = sh_bar[sh_ptr]
            sh_vbuf[n_sh] = sh_val[sh_ptr]
            n_sh += 1
            sh_ptr += 1
        while sl_ptr < nl and sl_confirm[sl_ptr] <= i:
            sl_b[n_sl] = sl_bar[sl_ptr]
            sl_vbuf[n_sl] = sl_val[sl_ptr]
            n_sl += 1
            sl_ptr += 1

        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr <= 0.0:
            cur_atr = 1e-6

        ci = close[i]

        if n_sh > 0:
            min_above = np.inf
            max_all = -np.inf
            has_above = False
            for j in range(n_sh):
                v = sh_vbuf[j]
                if v > max_all:
                    max_all = v
                if v > ci:
                    if v < min_above:
                        min_above = v
                    has_above = True
            if has_above:
                nearest_resist[i] = min_above
            else:
                nearest_resist[i] = max_all

        if n_sl > 0:
            max_below = -np.inf
            min_all = np.inf
            has_below = False
            for j in range(n_sl):
                v = sl_vbuf[j]
                if v < min_all:
                    min_all = v
                if v < ci:
                    if v > max_below:
                        max_below = v
                    has_below = True
            if has_below:
                nearest_support[i] = max_below
            else:
                nearest_support[i] = min_all

        sup = nearest_support[i]
        res = nearest_resist[i]
        if np.isfinite(sup) and np.isfinite(res) and res > sup:
            denom = res - sup
            sp = (ci - sup) / denom
            if sp < 0.0:
                sp = 0.0
            elif sp > 1.0:
                sp = 1.0
            sr_position[i] = sp

        if n_sh > 0 and n_sl > 0:
            last_sh_v = sh_vbuf[n_sh - 1]
            last_sl_v = sl_vbuf[n_sl - 1]
            r50 = (last_sh_v + last_sl_v) / 2.0
            retrace_50[i] = r50
            at_50_retrace[i] = abs(ci - r50) < 0.2 * cur_atr

        r5 = round(ci / 5.0) * 5.0
        r1 = round(ci)
        if abs(ci - r5) < abs(ci - r1):
            nearest_round = r5
        else:
            nearest_round = r1
        round_num_dist[i] = abs(ci - nearest_round) / cur_atr

    return nearest_resist, nearest_support, sr_position, retrace_50, at_50_retrace, round_num_dist


def causal_support_resistance_fast(
    close: np.ndarray,
    atr: np.ndarray,
    swing_highs: list[tuple[int, int, float]],
    swing_lows: list[tuple[int, int, float]],
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.ascontiguousarray(close, dtype=np.float64)
    a = np.ascontiguousarray(atr, dtype=np.float64)
    if swing_highs:
        sh_confirm = np.array([t[0] for t in swing_highs], dtype=np.int64)
        sh_bar = np.array([t[1] for t in swing_highs], dtype=np.int64)
        sh_val = np.array([t[2] for t in swing_highs], dtype=np.float64)
    else:
        sh_confirm = np.empty(0, dtype=np.int64)
        sh_bar = np.empty(0, dtype=np.int64)
        sh_val = np.empty(0, dtype=np.float64)
    if swing_lows:
        sl_confirm = np.array([t[0] for t in swing_lows], dtype=np.int64)
        sl_bar = np.array([t[1] for t in swing_lows], dtype=np.int64)
        sl_val = np.array([t[2] for t in swing_lows], dtype=np.float64)
    else:
        sl_confirm = np.empty(0, dtype=np.int64)
        sl_bar = np.empty(0, dtype=np.int64)
        sl_val = np.empty(0, dtype=np.float64)
    if not _HAVE_NUMBA:
        return _causal_support_resistance_core_py(c, a, sh_confirm, sh_val, sl_confirm, sl_val, n)
    return _causal_support_resistance_core(
        c, a, sh_confirm, sh_bar, sh_val, sl_confirm, sl_bar, sl_val, n
    )


def _bit_add(tree: np.ndarray, idx0: int, delta: int) -> None:
    i = idx0 + 1
    n = tree.shape[0] - 1
    while i <= n:
        tree[i] += delta
        i += i & -i


def _bit_sum(tree: np.ndarray, idx_exclusive: int) -> int:
    s = 0
    i = idx_exclusive
    while i > 0:
        s += int(tree[i])
        i -= i & -i
    return s


def _bit_find_kth(tree: np.ndarray, k: int) -> int:
    """Return 0-based index of the kth (1-based) active value."""
    n = tree.shape[0] - 1
    idx = 0
    bit = 1 << (n.bit_length() - 1)
    while bit:
        nxt = idx + bit
        if nxt <= n and tree[nxt] < k:
            idx = nxt
            k -= int(tree[nxt])
        bit >>= 1
    return idx


def _causal_support_resistance_core_py(
    close: np.ndarray,
    atr: np.ndarray,
    sh_confirm: np.ndarray,
    sh_val: np.ndarray,
    sl_confirm: np.ndarray,
    sl_val: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python fallback optimized for large histories when numba is unavailable."""
    nearest_resist = np.full(n, np.nan)
    nearest_support = np.full(n, np.nan)
    sr_position = np.full(n, 0.5)
    retrace_50 = np.full(n, np.nan)
    at_50_retrace = np.zeros(n, dtype=np.bool_)
    round_num_dist = np.full(n, np.nan)

    vals_h = np.unique(sh_val) if sh_val.size else np.empty(0, dtype=np.float64)
    vals_l = np.unique(sl_val) if sl_val.size else np.empty(0, dtype=np.float64)
    tree_h = np.zeros(vals_h.size + 1, dtype=np.int64)
    tree_l = np.zeros(vals_l.size + 1, dtype=np.int64)
    rank_h = np.searchsorted(vals_h, sh_val).astype(np.int64, copy=False) if sh_val.size else np.empty(0, dtype=np.int64)
    rank_l = np.searchsorted(vals_l, sl_val).astype(np.int64, copy=False) if sl_val.size else np.empty(0, dtype=np.int64)

    nh = sh_confirm.shape[0]
    nl = sl_confirm.shape[0]
    sh_ptr = 0
    sl_ptr = 0
    n_sh = 0
    n_sl = 0
    last_sh_v = np.nan
    last_sl_v = np.nan

    for i in range(n):
        while sh_ptr < nh and sh_confirm[sh_ptr] <= i:
            _bit_add(tree_h, int(rank_h[sh_ptr]), 1)
            n_sh += 1
            last_sh_v = sh_val[sh_ptr]
            sh_ptr += 1
        while sl_ptr < nl and sl_confirm[sl_ptr] <= i:
            _bit_add(tree_l, int(rank_l[sl_ptr]), 1)
            n_sl += 1
            last_sl_v = sl_val[sl_ptr]
            sl_ptr += 1

        ci = close[i]
        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr <= 0.0:
            cur_atr = 1e-6

        if n_sh > 0:
            idx_gt = int(np.searchsorted(vals_h, ci, side="right"))
            cnt_le = _bit_sum(tree_h, idx_gt)
            if cnt_le < n_sh:
                rank_idx = _bit_find_kth(tree_h, cnt_le + 1)
            else:
                rank_idx = _bit_find_kth(tree_h, n_sh)
            nearest_resist[i] = vals_h[rank_idx]

        if n_sl > 0:
            idx_ge = int(np.searchsorted(vals_l, ci, side="left"))
            cnt_lt = _bit_sum(tree_l, idx_ge)
            if cnt_lt > 0:
                rank_idx = _bit_find_kth(tree_l, cnt_lt)
            else:
                rank_idx = _bit_find_kth(tree_l, 1)
            nearest_support[i] = vals_l[rank_idx]

        sup = nearest_support[i]
        res = nearest_resist[i]
        if np.isfinite(sup) and np.isfinite(res) and res > sup:
            sp = (ci - sup) / (res - sup)
            if sp < 0.0:
                sp = 0.0
            elif sp > 1.0:
                sp = 1.0
            sr_position[i] = sp

        if n_sh > 0 and n_sl > 0:
            r50 = (last_sh_v + last_sl_v) / 2.0
            retrace_50[i] = r50
            at_50_retrace[i] = abs(ci - r50) < 0.2 * cur_atr

        r5 = round(ci / 5.0) * 5.0
        r1 = round(ci)
        nearest_round = r5 if abs(ci - r5) < abs(ci - r1) else r1
        round_num_dist[i] = abs(ci - nearest_round) / cur_atr

    return nearest_resist, nearest_support, sr_position, retrace_50, at_50_retrace, round_num_dist


NUMBA_AVAILABLE: bool = _HAVE_NUMBA


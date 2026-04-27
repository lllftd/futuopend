"""Map L3 policy feature rows to a fixed-dim position_x for UnifiedL2L3Net.position_enc."""

from __future__ import annotations

import numpy as np

# Order matters: first n_position columns are used (rest padded if n_position is larger).
POSITION_FEATURE_COLS: tuple[str, ...] = (
    "l3_hold_bars",
    "l3_unreal_pnl_atr",
    "l3_live_mfe",
    "l3_live_mae",
    "l3_live_edge",
    "l3_drawdown_from_peak_atr",
    "l3_unreal_pnl_frac",
    "l3_drawdown_from_peak_frac",
    "l3_regime_divergence",
    "l3_vol_surprise",
    "l3_log_hold_bars",
    "l3_hold_bars_sq",
    "l3_side",
    "l3_bars_since_peak",
    "l3_at_new_high",
    "l3_would_enter_now",
)

_COL_CACHE: dict[tuple[str, ...], dict[str, int]] = {}


def _col_index_map(feature_cols: list[str]) -> dict[str, int]:
    key = tuple(feature_cols)
    if key not in _COL_CACHE:
        _COL_CACHE[key] = {c: i for i, c in enumerate(feature_cols)}
    return _COL_CACHE[key]


def build_position_matrix(
    X: np.ndarray,
    feature_cols: list[str],
    n_pos: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Return ``(position_x, used_names)`` with shape (N, n_pos). Unknown columns are 0; names list
    documents which slot maps to which policy column (or a padding token).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    idx_map = _col_index_map(feature_cols)
    n = int(X.shape[0])
    out = np.zeros((n, n_pos), dtype=np.float32)
    used: list[str] = []
    for j in range(n_pos):
        if j < len(POSITION_FEATURE_COLS):
            name = POSITION_FEATURE_COLS[j]
            if name in idx_map:
                out[:, j] = np.asarray(X[:, idx_map[name]], dtype=np.float32)
                used.append(name)
            else:
                used.append(f"__missing__:{name}")
        else:
            used.append(f"__pad_{j}")
    return out, used


def build_position_matrix_from_dataframe(
    df,
    n_pos: int,
) -> np.ndarray:
    """
    OOS: build position_x from a row-wise DataFrame using :data:`POSITION_FEATURE_COLS` order.
    Missing columns are 0.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    n = int(len(df))
    out = np.zeros((n, n_pos), dtype=np.float32)
    cols = set(df.columns)
    for j in range(n_pos):
        if j < len(POSITION_FEATURE_COLS):
            name = POSITION_FEATURE_COLS[j]
            if name in cols:
                out[:, j] = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float32, copy=False
                )
    return out

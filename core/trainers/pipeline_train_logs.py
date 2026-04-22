"""Standard pipeline validation prints (time splits, shapes, NaN/Inf) for L1a–L3 training.

Use in each trainer *after* train/val masks and feature matrices are finalized:
  - ``log_layer_banner``, ``log_time_key_split`` / ``log_time_key_arrays``, ``log_numpy_x_stats``
  - ``artifact_path`` for MODEL_DIR-relative filenames
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.trainers.constants import MODEL_DIR


def _as_series_time(time_values: pd.Series | np.ndarray) -> pd.Series:
    return pd.to_datetime(pd.Series(np.asarray(time_values)))


def log_time_key_arrays(
    layer: str,
    tk_train: pd.Series | np.ndarray | None,
    tk_val: pd.Series | np.ndarray | None,
    *,
    train_label: str = "train",
    val_label: str = "val",
    extra_note: str = "",
) -> None:
    """Print counts and min/max when masks are not row-aligned (e.g. L1a window indices)."""
    print(f"\n  [{layer}] --- split: {train_label} / {val_label} ---", flush=True)
    if tk_train is not None:
        tt = _as_series_time(tk_train)
        n_t = len(tt)
        print(f"  [{layer}] {train_label} samples: {n_t:,}", flush=True)
        if n_t > 0:
            v = tt.values
            print(f"  [{layer}] {train_label} time_key: [{pd.Timestamp(np.min(v))}, {pd.Timestamp(np.max(v))}]", flush=True)
    if tk_val is not None:
        vv = _as_series_time(tk_val)
        n_v = len(vv)
        print(f"  [{layer}] {val_label} samples:   {n_v:,}", flush=True)
        if n_v > 0:
            v = vv.values
            print(f"  [{layer}] {val_label} time_key:   [{pd.Timestamp(np.min(v))}, {pd.Timestamp(np.max(v))}]", flush=True)
    if extra_note:
        print(f"  [{layer}] note: {extra_note}", flush=True)


def log_time_key_split(
    layer: str,
    time_values: pd.Series | np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    train_label: str = "train",
    val_label: str = "val",
    extra_note: str = "",
) -> None:
    """Print sample counts and min/max time_key for two boolean masks (same length as time_values)."""
    tk = _as_series_time(time_values)
    tm = np.asarray(train_mask, dtype=bool)
    vm = np.asarray(val_mask, dtype=bool)
    n_t, n_v = int(tm.sum()), int(vm.sum())
    print(f"\n  [{layer}] --- split: {train_label} / {val_label} ---", flush=True)
    print(f"  [{layer}] {train_label} samples: {n_t:,}", flush=True)
    if n_t > 0:
        tr = tk.values[tm]
        print(f"  [{layer}] {train_label} time_key: [{pd.Timestamp(tr.min())}, {pd.Timestamp(tr.max())}]", flush=True)
    print(f"  [{layer}] {val_label} samples:   {n_v:,}", flush=True)
    if n_v > 0:
        vr = tk.values[vm]
        print(f"  [{layer}] {val_label} time_key:   [{pd.Timestamp(vr.min())}, {pd.Timestamp(vr.max())}]", flush=True)
    if extra_note:
        print(f"  [{layer}] note: {extra_note}", flush=True)


def log_numpy_x_stats(layer: str, X: np.ndarray, *, label: str = "X") -> None:
    """NaN / Inf counts for a float feature matrix."""
    X = np.asarray(X)
    if X.dtype == object:
        print(f"  [{layer}] {label}: object dtype (skipped NaN/Inf scan)", flush=True)
        return
    # Avoid huge temporary allocations on large tensors (e.g. L1a windows flattened).
    if X.size <= 5_000_000:
        xf = X.astype(np.float32, copy=False)
        nan_c = int(np.isnan(xf).sum())
        inf_c = int(np.isinf(xf).sum())
    else:
        flat = X.reshape(-1)
        chunk = max(1_000_000, int(os.environ.get("LOG_NANINF_CHUNK", "4000000")))
        nan_c = 0
        inf_c = 0
        for i in range(0, flat.size, chunk):
            part = np.asarray(flat[i : i + chunk], dtype=np.float32)
            nan_c += int(np.isnan(part).sum())
            inf_c += int(np.isinf(part).sum())
    print(f"  [{layer}] {label} shape: {X.shape}  NaN: {nan_c:,}  Inf: {inf_c:,}", flush=True)


def log_layer_banner(layer: str) -> None:
    print(f"\n{'=' * 50}", flush=True)
    print(f"  {layer}", flush=True)
    print(f"{'=' * 50}", flush=True)


def artifact_path(*parts: str) -> str:
    return os.path.join(MODEL_DIR, *parts)


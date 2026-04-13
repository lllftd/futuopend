from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.trainers.constants import CAL_END, MODEL_DIR, TEST_END, TRAIN_END


@dataclass(frozen=True)
class StackTimeSplits:
    train_mask: np.ndarray
    cal_mask: np.ndarray
    test_mask: np.ndarray
    l2_train_mask: np.ndarray
    l2_val_mask: np.ndarray


def l2_val_start_time() -> np.datetime64:
    raw = os.environ.get("L2_VAL_START", "2023-05-01").strip() or "2023-05-01"
    return np.datetime64(raw)


def build_stack_time_splits(time_key: pd.Series | np.ndarray) -> StackTimeSplits:
    ts = pd.to_datetime(np.asarray(time_key))
    train_mask = ts < np.datetime64(TRAIN_END)
    cal_mask = (ts >= np.datetime64(TRAIN_END)) & (ts < np.datetime64(CAL_END))
    test_mask = (ts >= np.datetime64(CAL_END)) & (ts < np.datetime64(TEST_END))

    l2_val_start = l2_val_start_time()
    if not (np.datetime64(TRAIN_END) < l2_val_start < np.datetime64(CAL_END)):
        raise ValueError(
            f"L2_VAL_START={str(l2_val_start)!r} must lie strictly inside calibration window "
            f"[{TRAIN_END}, {CAL_END})."
        )
    l2_train_mask = (ts >= np.datetime64(TRAIN_END)) & (ts < l2_val_start)
    l2_val_mask = (ts >= l2_val_start) & (ts < np.datetime64(CAL_END))
    if not l2_train_mask.any() or not l2_val_mask.any():
        raise RuntimeError(
            "L2 strict time split produced an empty train or val mask. "
            f"Check TRAIN_END={TRAIN_END}, L2_VAL_START={str(l2_val_start)}, CAL_END={CAL_END}."
        )

    return StackTimeSplits(
        train_mask=train_mask,
        cal_mask=cal_mask,
        test_mask=test_mask,
        l2_train_mask=l2_train_mask,
        l2_val_mask=l2_val_mask,
    )


def future_group_apply(
    df: pd.DataFrame,
    column: str,
    horizon: int,
    reducer: str,
) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=np.float32)
    for _, grp in df.groupby("symbol", sort=False):
        vals = pd.to_numeric(grp[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        n = len(vals)
        res = np.zeros(n, dtype=np.float32)
        for i in range(n):
            j = min(n, i + 1 + horizon)
            future = vals[i + 1 : j]
            if future.size == 0:
                res[i] = vals[i]
            elif reducer == "mean":
                res[i] = float(np.nanmean(future))
            elif reducer == "max":
                res[i] = float(np.nanmax(future))
            elif reducer == "min":
                res[i] = float(np.nanmin(future))
            elif reducer == "last":
                res[i] = float(future[-1])
            else:
                raise ValueError(f"Unsupported reducer={reducer!r}")
        out[grp.index.to_numpy()] = res
    return out


def compute_transition_risk_labels(state_label: np.ndarray, symbols: np.ndarray, *, horizon: int = 10) -> np.ndarray:
    labels = np.zeros(len(state_label), dtype=np.float32)
    state_label = np.asarray(state_label, dtype=np.int64)
    symbols = np.asarray(symbols)
    for sym in np.unique(symbols):
        idx = np.flatnonzero(symbols == sym)
        s = state_label[idx]
        n = len(s)
        for i in range(n):
            future = s[i + 1 : min(n, i + 1 + horizon)]
            if future.size == 0:
                labels[idx[i]] = 0.0
                continue
            changes = np.flatnonzero(future != s[i])
            if changes.size == 0:
                labels[idx[i]] = 0.0
            else:
                first_change = int(changes[0])
                labels[idx[i]] = float(1.0 - (first_change / max(horizon, 1)))
    return labels


def compute_transition_event_labels(state_label: np.ndarray, symbols: np.ndarray, *, horizon: int = 10) -> np.ndarray:
    labels = np.zeros(len(state_label), dtype=np.float32)
    state_label = np.asarray(state_label, dtype=np.int64)
    symbols = np.asarray(symbols)
    horizon = max(int(horizon), 1)
    for sym in np.unique(symbols):
        idx = np.flatnonzero(symbols == sym)
        s = state_label[idx]
        n = len(s)
        for i in range(n):
            future = s[i + 1 : min(n, i + 1 + horizon)]
            labels[idx[i]] = float(np.any(future != s[i])) if future.size else 0.0
    return labels


def compute_cross_asset_context(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["symbol", "time_key", "close"]].copy()
    work["ret_1"] = work.groupby("symbol")["close"].pct_change().fillna(0.0)
    pivot = work.pivot(index="time_key", columns="symbol", values="ret_1").fillna(0.0)
    market_mean = pivot.mean(axis=1)
    breadth = (pivot.gt(0).mean(axis=1) * 2.0 - 1.0).astype(np.float32)
    centered = pivot.sub(market_mean, axis=0)
    out = work[["symbol", "time_key"]].copy()
    out["sector_relative_strength"] = centered.stack(dropna=False).reindex(
        pd.MultiIndex.from_frame(work[["time_key", "symbol"]])
    ).to_numpy(dtype=np.float32, copy=False)
    out["market_breadth"] = work["time_key"].map(breadth).astype(np.float32)
    corr_map: dict[str, float] = {}
    if pivot.shape[1] >= 2:
        corr = pivot.rolling(20, min_periods=5).corr()
        for ts, block in corr.groupby(level=0):
            vals = block.droplevel(0).to_numpy(dtype=np.float32, copy=False)
            corr_map[ts] = float(np.nanmean(vals)) if vals.size else 0.0
    out["correlation_regime"] = work["time_key"].map(corr_map).fillna(0.0).astype(np.float32)
    return out


def log_label_baseline(head_name: str, y: np.ndarray, task: str = "auto") -> None:
    """Print compact label diagnostics before fitting a head."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    nan_n = int(np.isnan(y).sum())
    inf_n = int(np.isinf(y).sum())
    finite = y[np.isfinite(y)]
    if task == "auto":
        nuniq = len(np.unique(finite)) if finite.size else 0
        task = "cls" if nuniq <= 20 else "reg"
    print(f"  [label-baseline] {head_name}  n={n:,}  NaN={nan_n}  Inf={inf_n}  task={task}", flush=True)
    if finite.size == 0:
        print("    (skip: no finite labels)", flush=True)
        return
    if task == "cls":
        vals, counts = np.unique(finite, return_counts=True)
        total = int(counts.sum())
        for v, c in zip(vals.tolist(), counts.tolist()):
            pct = 100.0 * float(c) / max(total, 1)
            flag = "  <- skew" if pct > 95.0 or pct < 1.0 else ""
            print(f"    class={v:g}: {c:>8,} ({pct:5.1f}%){flag}", flush=True)
        if len(vals) < 2:
            print("    WARNING: single-class label; model cannot learn a separator.", flush=True)
        return
    pcts = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    print(
        f"    mean={float(np.mean(finite)):.6f}  std={float(np.std(finite)):.6f}  "
        f"min={float(np.min(finite)):.6f}  max={float(np.max(finite)):.6f}",
        flush=True,
    )
    print(f"    pcts[1/5/25/50/75/95/99]={np.round(pcts, 6).tolist()}", flush=True)
    nuniq = len(np.unique(np.round(finite, 6)))
    if nuniq <= 3:
        print(f"    WARNING: only {nuniq} unique rounded values; label may be degenerate.", flush=True)


def diagnose_l1b_leakage(models: dict, feat_names: dict[str, list[str]] | list[str]) -> tuple[list[str], list[str]]:
    """Summarize top feature concentration for trained L1b models."""
    print("\n  [L1b] leakage diagnostic", flush=True)
    print(f"  {'head':<30s} {'top1_feat':<30s} {'top1%':>7s} {'top3%':>7s}", flush=True)
    print(f"  {'-' * 80}", flush=True)
    deterministic_heads: list[str] = []
    learned_heads: list[str] = []
    for head_name, model in models.items():
        if isinstance(feat_names, dict):
            head_feat_names = list(feat_names.get(head_name) or [])
        else:
            head_feat_names = list(feat_names)
        imp = np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64)
        total = float(np.sum(imp))
        if total <= 0.0:
            print(f"  {head_name:<30s} {'(zero importance)':<30s} {'n/a':>7s} {'n/a':>7s}", flush=True)
            continue
        order = np.argsort(imp)[::-1]
        top1_idx = int(order[0])
        top1_pct = float(imp[top1_idx] / total)
        top3_pct = float(np.sum(imp[order[:3]]) / total)
        top1_name = head_feat_names[top1_idx] if 0 <= top1_idx < len(head_feat_names) else f"feat[{top1_idx}]"
        print(f"  {head_name:<30s} {top1_name:<30s} {top1_pct:>6.1%} {top3_pct:>6.1%}", flush=True)
        if top3_pct > 0.85:
            deterministic_heads.append(head_name)
        else:
            learned_heads.append(head_name)
    print(f"  [L1b] diag deterministic-ish={deterministic_heads}", flush=True)
    print(f"  [L1b] diag learned-ish={learned_heads}", flush=True)
    return deterministic_heads, learned_heads


def save_output_cache(df: pd.DataFrame, filename: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return path


def load_output_cache(filename: str) -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Cache {path} is not a pandas DataFrame.")
    return obj

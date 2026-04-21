"""Train-window distribution fingerprints for prepared tabular features (drift guard)."""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from core.trainers.constants import MODEL_DIR, TRAIN_END

PREP_FEATURE_FINGERPRINT_FILE = "prep_feature_fingerprint.pkl"


def _finite_numeric(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return x[np.isfinite(x)]


def compute_train_window_stats(
    df: pd.DataFrame,
    columns: list[str],
    *,
    train_end: str = TRAIN_END,
) -> tuple[dict[str, dict[str, float]], int]:
    ts = pd.to_datetime(df["time_key"])
    mask = ts < pd.Timestamp(train_end)
    n = int(mask.sum())
    stats: dict[str, dict[str, float]] = {}
    sub = df.loc[mask]
    for c in columns:
        if c not in df.columns:
            continue
        x = _finite_numeric(sub[c])
        if x.size < 20:
            continue
        stats[c] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "null_rate": float(sub[c].isna().mean()) if hasattr(sub[c], "isna") else 0.0,
            "q01": float(np.quantile(x, 0.01)),
            "q50": float(np.quantile(x, 0.50)),
            "q99": float(np.quantile(x, 0.99)),
            "n": float(x.size),
        }
    return stats, n


def fingerprint_path() -> str:
    return os.path.join(MODEL_DIR, PREP_FEATURE_FINGERPRINT_FILE)


def load_prep_fingerprint() -> dict[str, Any] | None:
    path = fingerprint_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_prep_fingerprint(feat_cols: list[str], stats: dict[str, dict[str, float]], train_n: int) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = fingerprint_path()
    payload = {
        "feat_cols": list(feat_cols),
        "train_end": TRAIN_END,
        "train_n": int(train_n),
        "stats": stats,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def report_prep_drift_vs_saved(
    df: pd.DataFrame,
    *,
    train_end: str = TRAIN_END,
    z_threshold: float = 3.0,
    max_warnings: int = 24,
) -> int:
    """Compare current train-window stats to last saved fingerprint; print warnings. Returns warning count."""
    zt = float(os.environ.get("PREP_DRIFT_Z_THRESHOLD", str(z_threshold)))
    prev = load_prep_fingerprint()
    if not prev or not isinstance(prev.get("stats"), dict):
        return 0
    prev_stats: dict[str, dict[str, float]] = prev["stats"]
    cols = [c for c in prev_stats if c in df.columns]
    cur_stats, n_cur = compute_train_window_stats(df, cols, train_end=train_end)
    print(
        f"\n  === prep feature drift vs saved fingerprint (train t<{train_end}, rows={n_cur:,}) ===",
        flush=True,
    )
    warned = 0
    for c in cols:
        if c not in cur_stats or c not in prev_stats:
            continue
        a, b = prev_stats[c], cur_stats[c]
        sd = max(float(a.get("std", 0.0)), 1e-9)
        z_mean = abs(float(b["mean"]) - float(a["mean"])) / sd
        z_med = abs(float(b["q50"]) - float(a["q50"])) / sd
        flag = z_mean > zt or z_med > zt
        nr_a, nr_b = float(a.get("null_rate", 0.0)), float(b.get("null_rate", 0.0))
        if abs(nr_b - nr_a) > 0.05:
            flag = True
        if flag and warned < max_warnings:
            print(
                f"    [drift] {c}: z_mean={z_mean:.2f} z_median={z_med:.2f} "
                f"null {nr_a:.3f}->{nr_b:.3f}",
                flush=True,
            )
            warned += 1
    if warned == 0:
        print("    no material drift vs saved fingerprint", flush=True)
    elif warned >= max_warnings:
        print(f"    ... truncated ({max_warnings}+ columns flagged)", flush=True)
    return warned

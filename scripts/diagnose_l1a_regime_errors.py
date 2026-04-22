#!/usr/bin/env python3
"""
Merge l1a_outputs.pkl with prepared_lgbm_dataset (market_state) and report:
  - calibration-slice accuracy sanity (should be ~0.5–0.7 if artifacts match training log)
  - worst confusion pairs, max_prob / entropy, bar-in-run vs accuracy, transition_risk vs wrong.

Run from repo root: PYTHONPATH=. python3 scripts/diagnose_l1a_regime_errors.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.trainers.constants import (  # noqa: E402
    CAL_END,
    L1A_OUTPUT_CACHE_FILE,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    PREPARED_DATASET_CACHE_FILE,
    REGIME_NOW_PROB_COLS,
    TRAIN_END,
)

NAMES = list(REGIME_NOW_PROB_COLS)


def main() -> int:
    out_path = os.path.join(MODEL_DIR, L1A_OUTPUT_CACHE_FILE)
    prep_path = os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE)
    if not os.path.isfile(out_path) or not os.path.isfile(prep_path):
        print("Need:", out_path, "and", prep_path)
        return 1

    with open(out_path, "rb") as f:
        out = pickle.load(f)
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)["df"]

    regime_cols = [c for c in out.columns if c.startswith("l1a_regime_prob_")]
    if len(regime_cols) != NUM_REGIME_CLASSES:
        print(f"Unexpected regime col count: {len(regime_cols)}")
        return 1

    m = out.merge(prep[["symbol", "time_key", "market_state"]], on=["symbol", "time_key"], how="inner")
    if len(m) != len(out):
        print(f"WARN: merge rows {len(m)} != l1a_outputs {len(out)}")
    m["tk"] = pd.to_datetime(m["time_key"])
    warm = m["l1a_is_warm"].to_numpy(dtype=float) > 0.5
    cal_mask = warm & (m["tk"] >= pd.Timestamp(TRAIN_END)) & (m["tk"] < pd.Timestamp(CAL_END))

    yt = pd.to_numeric(m["market_state"], errors="coerce").astype("Int64").to_numpy()
    pr = m[regime_cols].to_numpy(dtype=np.float64)
    pred = np.argmax(pr, axis=1)

    valid = warm & np.isfinite(yt.astype(float))
    yt_i = yt.astype(np.int64, copy=False)

    acc_cal = float((pred[cal_mask] == yt_i[cal_mask]).mean()) if cal_mask.any() else float("nan")
    acc_warm = float((pred[valid] == yt_i[valid]).mean())
    maj = int(pd.Series(yt_i[cal_mask]).mode().iloc[0]) if cal_mask.any() else 0
    maj_freq = float((yt_i[cal_mask] == maj).mean()) if cal_mask.any() else float("nan")

    print("=== Artifact sanity (vs layer1a.log oof_cal_full acc ~0.65) ===")
    print(f"  warm rows: {int(valid.sum()):,}  cal-window warm: {int(cal_mask.sum()):,}")
    print(f"  accuracy warm (all): {acc_warm:.4f}")
    print(f"  accuracy cal [TRAIN_END, CAL_END): {acc_cal:.4f}  majority-class freq: {maj_freq:.4f} (class {maj})")
    if acc_cal < 0.35:
        print(
            "  *** WARN: cal accuracy far below training log — check that prepared_lgbm_dataset.pkl, "
            "l1a_outputs.pkl, and l1a_market_tcn.pt are from the same pipeline run; "
            "re-materialize l1a_outputs from current .pt or rebuild prep + retrain.",
            flush=True,
        )

    cm = confusion_matrix(yt_i[valid], pred[valid], labels=list(range(NUM_REGIME_CLASSES)))
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(np.float64) / row_sum
    off = cm_norm.copy()
    np.fill_diagonal(off, 0.0)
    flat_idx = int(np.argmax(off))
    i, j = divmod(flat_idx, NUM_REGIME_CLASSES)
    pairs: list[tuple[int, int, int, float]] = []
    for a in range(NUM_REGIME_CLASSES):
        for b in range(NUM_REGIME_CLASSES):
            if a == b:
                continue
            pairs.append((cm[a, b], a, b, cm_norm[a, b]))
    pairs.sort(reverse=True)

    print("\n=== Q1 最严重混淆（warm，按 P(pred|true) 最大单格）===")
    print(
        f"  true={NAMES[i]:12s} pred={NAMES[j]:12s}  "
        f"P(pred|true)={off[i, j]:.4f}  count={cm[i, j]:,}",
    )
    print("  Top 5 by count:")
    for cnt, a, b, pcond in pairs[:5]:
        print(f"    {NAMES[a]:12s} -> {NAMES[b]:12s}  n={cnt:7,}  P(pred|true)={pcond:.4f}")

    max_prob = m[regime_cols].max(axis=1)
    eps = 1e-12
    ent = -(pr * np.log(np.clip(pr, eps, 1.0))).sum(axis=1)
    max_ent = float(np.log(NUM_REGIME_CLASSES))
    print("\n=== Q2 置信度 / 模糊度（warm）===")
    mw = max_prob[valid]
    print(f"  max_prob mean={mw.mean():.4f} median={mw.median():.4f}")
    for thr in (0.3, 0.4, 0.5, 0.7):
        print(f"  max_prob < {thr}: {(mw < thr).mean():.2%}")
    print(f"  max_prob > 0.7: {(mw > 0.7).mean():.2%}")
    print(f"  normalized entropy mean={(ent[valid] / max_ent).mean():.4f} (1≈均匀)")

    df = m.loc[valid, ["symbol", "time_key", "market_state"]].copy()
    df["yt"] = yt_i[valid]
    df["pred"] = pred[valid]
    df = df.sort_values(["symbol", "time_key"])
    g = df.groupby("symbol", sort=False)["market_state"]
    new_run = g.diff().ne(0)
    first = df.groupby("symbol", sort=False).cumcount() == 0
    new_run = new_run | first
    run_id = new_run.groupby(df["symbol"]).cumsum()
    df["bar_in_run"] = df.groupby(["symbol", run_id], sort=False).cumcount() + 1
    ok = df["yt"].to_numpy() == df["pred"].to_numpy()

    print("\n=== Q3 regime 切换点 vs 稳态（warm）===")
    tmask = df["bar_in_run"].to_numpy() == 1
    mid = df["bar_in_run"].to_numpy() >= 10
    print(f"  bar_in_run==1  n={tmask.sum():,}  acc={ok[tmask].mean():.4f}")
    print(f"  bar_in_run>=10  n={mid.sum():,}  acc={ok[mid].mean():.4f}")
    for lo, hi, lab in [(2, 5, "[2,5]"), (6, 15, "[6,15]"), (16, 10**9, "[16,inf)")]:
        mm = (df["bar_in_run"].to_numpy() >= lo) & (df["bar_in_run"].to_numpy() <= hi)
        if mm.any():
            print(f"  {lab:10s} n={mm.sum():,}  acc={ok[mm].mean():.4f}")

    tr = m.loc[valid, "l1a_transition_risk"].to_numpy(dtype=np.float64)
    print("\n=== transition_risk（warm）===")
    print(f"  correct mean={tr[ok].mean():.4f}  wrong mean={tr[~ok].mean():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

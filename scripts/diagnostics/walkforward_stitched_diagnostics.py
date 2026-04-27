#!/usr/bin/env python3
"""
Plot stitched cumulative return (by exit_time) and Fold-1 daily PnL / drawdown post-mortem.

  PYTHONPATH=. python3 scripts/diagnostics/walkforward_stitched_diagnostics.py \\
    --stitched results/walkforward_true_oos_4f/stitched_trades_ALL.csv \\
    --out-dir results/walkforward_true_oos_4f
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib required") from e


def _max_dd_window(returns: np.ndarray) -> tuple[float, int, int]:
    """Cumulative sum of `returns` (1d); return (mdd, start_idx, end_idx) of worst run."""
    c = np.cumsum(returns, dtype=np.float64)
    peak = c[0]
    peak_i = 0
    best_mdd = 0.0
    best_s, best_e = 0, 0
    for i in range(1, len(c)):
        if c[i] > peak:
            peak = c[i]
            peak_i = i
        d = float(c[i] - peak)
        if d < best_mdd:
            best_mdd = d
            best_s, best_e = peak_i, i
    return best_mdd, best_s, best_e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitched", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=1, help="Fold for daily PnL panel (default 1)")
    args = ap.parse_args()

    df = pd.read_csv(args.stitched, parse_dates=["entry_time", "exit_time"])
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # --- 1) Stitched: all trades, exit order ---
    t = df.sort_values("exit_time", kind="mergesort")
    r = t["return"].to_numpy(np.float64)
    cum = np.cumsum(r)
    t_idx = t["exit_time"]
    mdd, i0, i1 = _max_dd_window(r)
    mdd_pct = 100.0 * mdd

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=120)
    ax.plot(t_idx, 100.0 * cum, lw=0.8, color="C0", label="cumulative (simple sum, % notional)")
    ax.axhline(0.0, color="0.4", lw=0.5)
    ax.set_title("Stitched walk-forward: cumulative return by exit_time (all folds)")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("exit_time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p1 = out / "stitched_cumulative_by_exit.png"
    fig.savefig(p1)
    plt.close()

    # --- 2) Fold-1: daily PnL (sum returns by exit calendar date) + trade-level mdd window ---
    d1 = df[df["walkforward_fold"] == int(args.fold)].copy()
    d1["exit_d"] = d1["exit_time"].dt.normalize()
    daily = d1.groupby("exit_d", sort=True)["return"].sum()
    d_c = daily.cumsum()
    rm = d_c.cummax()
    dd = d_c - rm
    worst_d = float(dd.min())
    tr_d = dd.idxmin()

    r1 = d1.sort_values("exit_time")["return"].to_numpy(np.float64)
    mdd_t, t0, t1 = _max_dd_window(r1)
    t_series = d1.sort_values("exit_time").reset_index(drop=True)
    w_start = t_series["exit_time"].iloc[t0]
    w_trough = t_series["exit_time"].iloc[t1]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=120, sharex=True)
    ax0, ax1 = axes
    ax0.bar(daily.index, 100.0 * daily.values, width=0.8, color="0.4", alpha=0.7)
    ax0.axhline(0.0, color="k", lw=0.4)
    ax0.set_ylabel("Daily PnL sum (% )")
    ax0.set_title(f"Fold {args.fold}: daily PnL (sum of per-trade return by exit date)")
    ax1.plot(d_c.index, 100.0 * d_c.values, color="C0", lw=1.0, label="cum daily")
    ax1.fill_between(d_c.index, 100.0 * d_c.values, 100.0 * rm.values, alpha=0.25, color="C1", label="drawdown vs peak")
    ax1.set_ylabel("Cumulative daily PnL (%)")
    ax1.set_xlabel("date")
    ax1.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Fold {args.fold}: worst daily-agg drawdown = {100.0 * worst_d:.2f}% on {tr_d.date()!s}"
        f"  |  trade-level max DD = {100.0 * mdd_t:.2f}% (window exit_time {w_start} -> {w_trough})",
        fontsize=9,
    )
    fig.tight_layout()
    p2 = out / f"fold0{args.fold}_daily_pnl.png"
    fig.savefig(p2)
    plt.close()

    # Worst days (fold-1) by daily sum
    worst_days = daily.nsmallest(20)
    lines = [
        f"stitched: n={len(df)} trades, cumulative sum_ret={float(cum[-1]):.6f}  global maxDD%={mdd_pct:.2f} (on stitched series)",
        f"fold {args.fold} trades={len(d1)}",
        f"fold {args.fold} trade-level max drawdown: {100.0 * mdd_t:.4f}%  window exit_time: {w_start} .. {w_trough}  (trades {t0}..{t1})",
        f"fold {args.fold} daily-agg: worst underperformance vs running peak: {100.0 * worst_d:.4f}% on {tr_d}",
        "",
        "Worst 20 days by daily PnL sum (exit date, %):",
    ]
    for d_i, v in worst_days.items():
        lines.append(f"  {d_i.date()!s}  {100.0 * float(v):+.4f}")
    rep = out / "walkforward_stitched_diagnostics.txt"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {p1}\nWrote {p2}\nWrote {rep}\n", flush=True)
    print("\n".join(lines[:12]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

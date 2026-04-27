#!/usr/bin/env python3
"""
Walkforward fold: **1m K 线 5 行诊断**；按 **每 5 个自然日** 切片，在
``fold_*/regime_charts/{sym}_week_....png`` 输出。

  PYTHONPATH=. python3 scripts/walkforward/walkforward_regime_chart.py \\
    --trades results/walkforward_true_oos_4f/fold_01/trades_ALL.csv \\
    --out-dir results/walkforward_true_oos_4f/fold_01 \\
    --title-suffix "fold_01 [2025-01-01, 2025-05-01)"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtests.oos_regime_figure import plot_regime_candles_for_walkforward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        type=Path,
        required=True,
        help="trades_ALL.csv (single fold or single OOS result dir)",
    )
    ap.add_argument("--data-dir", type=Path, default=None, help="Default: <repo>/data")
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Fold 根目录；其下会建 week_YYYYMMDD_YYYYMMDD/ 子目录并写入 PNG",
    )
    ap.add_argument("--symbols", type=str, default="QQQ,SPY", help="Comma symbols")
    ap.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="E.g. fold_01 [2025-01-01, 2025-05-01) (shown in figure title)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or (repo / "data")
    df = pd.read_csv(args.trades, parse_dates=["entry_time", "exit_time"])
    for c in ("walkforward_fold",):
        if c in df.columns:
            df = df.drop(columns=[c])
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        ok = plot_regime_candles_for_walkforward(
            sym=sym,
            trades=df,
            data_dir=data_dir,
            out_dir=args.out_dir,
            title_suffix=(args.title_suffix or "").strip(),
        )
        if not ok:
            print(f"  [walkforward_regime_chart] skip or fail: {sym}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

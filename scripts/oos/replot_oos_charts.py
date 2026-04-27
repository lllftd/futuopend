#!/usr/bin/env python3
"""Re-render ``oos_chart_{SYM}.png`` from existing OOS results (trades + 1m CSV), no L1a–L3 re-run.

Reads ``oos_summary.txt`` in ``--results-dir`` for OOS window and ``OOS_CUMULATIVE_PLOT_MODE``,
then loads minute OHLC from ``data/{SYM}.csv`` and trades from ``trades_{SYM}.csv``.

  PYTHONPATH=. python3 scripts/oos/replot_oos_charts.py \\
    --results-dir results/modeloos_fullsample --symbols QQQ,SPY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def _load_oos_config(summary_path: Path) -> dict:
    text = summary_path.read_text(encoding="utf-8")
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/modeloos"),
        help="Dir with oos_summary.txt, trades_*.csv, oos_chart_*.png output",
    )
    ap.add_argument(
        "--symbols",
        type=str,
        default="QQQ,SPY",
        help="Comma-separated, same as OOS_SYMBOLS",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Override path to oos JSON (default: <results-dir>/oos_summary.txt)",
    )
    args = ap.parse_args()
    rdir: Path = args.results_dir
    summ_path = args.summary or (rdir / "oos_summary.txt")
    if not summ_path.is_file():
        print(f"[replot] missing {summ_path}", file=sys.stderr, flush=True)
        return 1
    meta = _load_oos_config(summ_path)
    oos_start = str(meta.get("oos_start", "2018-01-01"))
    oos_end = str(meta.get("oos_end", "2035-01-01"))
    cpm = (meta.get("backtest_config") or {}).get("OOS_CUMULATIVE_PLOT_MODE") or "all"
    cpm = str(cpm).strip() or "all"

    os.environ["OOS_START"] = oos_start
    os.environ["OOS_END"] = oos_end
    os.environ["OOS_RESULTS_DIR"] = str(rdir.resolve())
    os.environ["OOS_CUMULATIVE_PLOT_MODE"] = cpm

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import pandas as pd

    from backtests.oos_backtest import (  # noqa: E402
        _resolve_symbol_csv_path,
        filter_trades_for_cumulative_plot_mode,
        plot_oos_price_and_cumulative_return,
        summarize_trade_returns,
    )
    from backtests.oos_regime_figure import _load_ohlc_1m_from_csv  # noqa: E402

    t0, t1 = pd.Timestamp(oos_start), pd.Timestamp(oos_end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        tp = rdir / f"trades_{sym}.csv"
        if not tp.is_file():
            print(f"[replot] skip {sym}: no {tp.name}", file=sys.stderr, flush=True)
            continue
        raw_path = _resolve_symbol_csv_path(sym)
        print(f"[replot] {sym} OHLC from {raw_path}  window [{t0}, {t1}) ...", flush=True)
        price_df = _load_ohlc_1m_from_csv(raw_path, t0, t1)
        tr = pd.read_csv(tp, parse_dates=["entry_time", "exit_time"])
        tr_plot = filter_trades_for_cumulative_plot_mode(tr, cpm)
        m_plot = (
            summarize_trade_returns(tr_plot, label=sym)
            if tr_plot is not None and not tr_plot.empty
            else {"label": sym, "n_trades": 0}
        )
        tsub: str | None = None
        if cpm == "l3_learned":
            tsub = (
                f"累计曲线: 仅 L3 分类器平仓 (OOS_CUMULATIVE_PLOT_MODE={cpm}, "
                f"n={int(m_plot.get('n_trades', 0) or 0)})"
            )
        elif cpm == "synthetic_rules":
            tsub = (
                f"累计曲线: 仅合成规则平仓 (OOS_CUMULATIVE_PLOT_MODE={cpm}, "
                f"n={int(m_plot.get('n_trades', 0) or 0)})"
            )
        out_p = (
            rdir / f"oos_chart_{sym}.png" if cpm == "all" else rdir / f"oos_chart_{sym}_{cpm}.png"
        )
        plot_oos_price_and_cumulative_return(
            sym, price_df, tr_plot, m_plot, out_p, title_subtitle=tsub
        )
        print(f"[replot] wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

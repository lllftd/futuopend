"""
Build full-period V3 report from whatever is in data/*.csv (no manual end date required).

Resolves [start, end] as the intersection of all symbol files, clipped to optional --end.
Then runs the same backtest + chart export as the standalone scripts.

Usage:
  python -m reports.report_v3_from_data
  python -m reports.report_v3_from_data --start 2020-03-28 --end 2026-03-30
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.backtest_v3_range import OUT_DIR as DEFAULT_BACKTEST_DIR
from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import run_intraday_rule
from reports.plot_v3_range_charts import OUT_DIR as DEFAULT_CHART_DIR
from reports.plot_v3_range_charts import plot_one
from core.utils import resolve_v3_backtest_window
from core.v3_data_utils import prepare_featured as _prepare_featured
from core.v3_data_utils import slice_range as _slice_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3 backtest + charts using full available data/ range.")
    p.add_argument("--start", default="2020-03-28", help="Requested first calendar day (clipped to CSV start).")
    p.add_argument(
        "--end",
        default=None,
        help="Last calendar day inclusive (default: latest common time in all CSVs).",
    )
    p.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    p.add_argument("--backtest-dir", default=str(DEFAULT_BACKTEST_DIR))
    p.add_argument("--chart-dir", default=str(DEFAULT_CHART_DIR))
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--skip-plots", action="store_true", help="Only write summary / JSON / trades.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbols = tuple(s.upper() for s in args.symbols)
    missing = [s for s in symbols if s not in V3_BEST_PARAMS]
    if missing:
        raise SystemExit(f"No V3_BEST_PARAMS for: {missing}")

    start, end, meta = resolve_v3_backtest_window(symbols, args.start, args.end)
    bt_dir = Path(args.backtest_dir)
    bt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_to_{end}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, object]] = []
    for sym in symbols:
        params = V3_BEST_PARAMS[sym]
        featured = _prepare_featured(sym, params)
        tcol = featured["time_key"]
        data_min, data_max = tcol.min(), tcol.max()
        sliced = _slice_range(featured, start, end)
        if sliced.empty:
            raise RuntimeError(f"{sym}: no bars in resolved window {start}..{end}")

        res = run_intraday_rule(sliced, sym, params)
        sdict = res.summary
        rows.append(
            {
                "symbol": sym,
                "backtest_start": start,
                "backtest_end": end,
                "source_data_min": str(data_min),
                "source_data_max": str(data_max),
                "bars_in_range": len(sliced),
                "total_return": sdict.get("total_return"),
                "annual_return": sdict.get("annual_return"),
                "sharpe": sdict.get("sharpe"),
                "win_rate": sdict.get("win_rate"),
                "max_drawdown": sdict.get("max_drawdown"),
                "trade_count": sdict.get("trade_count"),
                "avg_holding_minutes": sdict.get("avg_holding_minutes"),
                "take_profit_rate": sdict.get("take_profit_rate"),
                "stop_loss_rate": sdict.get("stop_loss_rate"),
                "time_stop_rate": sdict.get("time_stop_rate"),
            }
        )
        trade_path = bt_dir / f"trades_{sym}_{tag}_FULL_{ts}.csv"
        res.trade_log.to_csv(trade_path, index=False)
        print(f"[{sym}] trades -> {trade_path}")

    summary_df = pd.DataFrame(rows)
    summary_csv = bt_dir / f"summary_{tag}_FULL_{ts}.csv"
    summary_df.to_csv(summary_csv, index=False)

    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    agg = {
        "backtest_start": start,
        "backtest_end": end,
        "approx_years": round(years, 3),
        "symbol_count": len(symbols),
        "mean_sharpe": float(summary_df["sharpe"].mean()),
        "mean_total_return": float(summary_df["total_return"].mean()),
        "mean_max_drawdown": float(summary_df["max_drawdown"].mean()),
        "total_trades_all_symbols": int(summary_df["trade_count"].sum()),
    }

    payload = {
        "resolved_window_meta": meta,
        "per_symbol": rows,
        "aggregate": agg,
        "artifacts": {
            "summary_csv": summary_csv.as_posix(),
            "tag": tag,
            "timestamp": ts,
        },
    }
    json_path = bt_dir / f"full_summary_{tag}_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    overview = bt_dir / f"overview_{tag}_{ts}.txt"
    lines = [
        f"V3 backtest (data-aligned window)",
        f"Period: {start} .. {end}  (~{agg['approx_years']} y)",
        f"Symbols: {', '.join(symbols)}",
        "",
        str(meta.get("common_data_range")),
        f"start_clipped_to_data: {meta.get('start_clipped_to_data')}",
        f"end_clipped_to_data: {meta.get('end_clipped_to_data')}",
        "",
        summary_df.to_string(index=False),
        "",
        "Aggregate:",
        json.dumps(agg, indent=2),
        "",
        f"CSV: {summary_csv}",
        f"JSON: {json_path}",
    ]
    overview.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")
    print(f"JSON:        {json_path}")
    print(f"Overview:    {overview}")

    if not args.skip_plots:
        chart_dir = Path(args.chart_dir)
        for sym in symbols:
            out_png = chart_dir / f"{sym}_minute_k_equity_{tag}_FULL.png"
            print(f"Plotting {sym} -> {out_png} ...")
            plot_one(sym, start, end, out_png, args.dpi)
            print(f"  wrote {out_png}")
            payload["artifacts"][f"chart_{sym}"] = out_png.as_posix()
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

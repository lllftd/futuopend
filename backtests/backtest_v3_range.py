"""
Run V3 strategy (config.V3_BEST_PARAMS) on a calendar date range using data/*.csv.

Indicators are computed on the full loaded history, then the simulation uses only
bars in [start, end] so warm-up is correct.

Usage:
  python -m backtests.backtest_v3_range --start 2020-03-28 --end 2024-03-27
  python -m backtests.backtest_v3_range --start 2020-03-28 --end 2024-03-27 --symbols QQQ
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import run_intraday_rule
from core.v3_data_utils import prepare_featured as _prepare_featured
from core.v3_data_utils import slice_range as _slice_range

OUT_DIR = Path("results") / "v3_range_backtest"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3 backtest on a date range (uses V3_BEST_PARAMS from config.py).")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["QQQ", "SPY"],
        help="Symbols matching data/{SYMBOL}.csv (default: QQQ SPY).",
    )
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Directory for CSV exports.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise SystemExit("--start must be <= --end")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start}_to_{args.end}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, object]] = []
    for sym in args.symbols:
        sym_u = sym.upper()
        if sym_u not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS entry for {sym_u}. Edit config.py.")
        params = V3_BEST_PARAMS[sym_u]

        featured = _prepare_featured(sym_u, params)
        tcol = featured["time_key"]
        data_min, data_max = tcol.min(), tcol.max()
        sliced = _slice_range(featured, args.start, args.end)
        if sliced.empty:
            print(f"[{sym_u}] NO BARS in [{args.start}, {args.end}] — CSV covers {data_min} .. {data_max}")
            rows.append(
                {
                    "symbol": sym_u,
                    "start": args.start,
                    "end": args.end,
                    "data_min": str(data_min),
                    "data_max": str(data_max),
                    "bars_in_range": 0,
                    "total_return": np.nan,
                    "sharpe": np.nan,
                    "trade_count": 0,
                }
            )
            continue

        res = run_intraday_rule(sliced, sym_u, params)
        s = res.summary
        rows.append(
            {
                "symbol": sym_u,
                "start": args.start,
                "end": args.end,
                "data_min": str(data_min),
                "data_max": str(data_max),
                "bars_in_range": len(sliced),
                "total_return": s.get("total_return"),
                "annual_return": s.get("annual_return"),
                "sharpe": s.get("sharpe"),
                "win_rate": s.get("win_rate"),
                "max_drawdown": s.get("max_drawdown"),
                "trade_count": s.get("trade_count"),
                "avg_holding_minutes": s.get("avg_holding_minutes"),
            }
        )

        trade_path = out_dir / f"trades_{sym_u}_{tag}_{ts}.csv"
        res.trade_log.to_csv(trade_path, index=False)
        print(f"[{sym_u}] trades -> {trade_path}")

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"summary_{tag}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    print("")
    print(summary_df.to_string(index=False))
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

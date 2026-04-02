"""
Compare current monitor version vs monitor+PA environment execution.

Baseline:
- Uses V3_BEST_PARAMS as-is (same as live monitor core params)
- Uses CE signal validity expansion with 5 bars (same as monitor)

Candidate:
- Baseline + pa_use_env_execution=True

Usage:
  python3 -m backtests.backtest_pa_env_vs_monitor --start 2024-01-01 --end 2026-12-31
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import RuleParams, run_intraday_rule
from core.v3_data_utils import prepare_featured_with_monitor_ce as _prepare_featured_with_monitor_ce
from core.v3_data_utils import slice_range as _slice_range


OUT_DIR = Path("results") / "pa_env_vs_monitor"
CE_VALID_BARS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare monitor baseline vs monitor+PA env execution.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols (default: SPY QQQ).")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    return p.parse_args()


def _env_params(params: RuleParams) -> RuleParams:
    return replace(
        params,
        pa_use_env_execution=True,
        pa_mtf_trend_filter=True,
        pa_mtf_min_rr=2.5,
        pa_use_advanced_pa_triggers=True,
    )


def main() -> int:
    args = parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise SystemExit("--start must be <= --end")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, object]] = []
    for sym in [s.upper() for s in args.symbols]:
        if sym not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS entry for {sym}")

        monitor_params = V3_BEST_PARAMS[sym]
        env_params = _env_params(monitor_params)

        featured = _prepare_featured_with_monitor_ce(sym, monitor_params, CE_VALID_BARS)
        sliced = _slice_range(featured, args.start, args.end)
        if sliced.empty:
            rows.append({"symbol": sym, "mode": "monitor_baseline", "trade_count": 0})
            rows.append({"symbol": sym, "mode": "monitor_plus_pa_env", "trade_count": 0})
            continue

        baseline = run_intraday_rule(sliced, sym, monitor_params).summary
        candidate = run_intraday_rule(sliced, sym, env_params).summary

        rows.append(
            {
                "symbol": sym,
                "mode": "monitor_baseline",
                "start": args.start,
                "end": args.end,
                "total_return": baseline.get("total_return"),
                "annual_return": baseline.get("annual_return"),
                "sharpe": baseline.get("sharpe"),
                "win_rate": baseline.get("win_rate"),
                "max_drawdown": baseline.get("max_drawdown"),
                "trade_count": baseline.get("trade_count"),
                "avg_holding_minutes": baseline.get("avg_holding_minutes"),
            }
        )
        rows.append(
            {
                "symbol": sym,
                "mode": "monitor_plus_pa_env",
                "start": args.start,
                "end": args.end,
                "total_return": candidate.get("total_return"),
                "annual_return": candidate.get("annual_return"),
                "sharpe": candidate.get("sharpe"),
                "win_rate": candidate.get("win_rate"),
                "max_drawdown": candidate.get("max_drawdown"),
                "trade_count": candidate.get("trade_count"),
                "avg_holding_minutes": candidate.get("avg_holding_minutes"),
                "pa_env_skip_count": candidate.get("pa_env_skip_count"),
                "pa_env_neutral_entry_count": candidate.get("pa_env_neutral_entry_count"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["symbol", "mode"]).reset_index(drop=True)
    summary_path = out_dir / f"pa_env_vs_monitor_summary_{ts}.csv"
    summary.to_csv(summary_path, index=False)

    deltas: list[dict[str, object]] = []
    for sym, grp in summary.groupby("symbol"):
        b = grp.loc[grp["mode"] == "monitor_baseline"]
        c = grp.loc[grp["mode"] == "monitor_plus_pa_env"]
        if b.empty or c.empty:
            continue
        b0 = b.iloc[0]
        c0 = c.iloc[0]
        deltas.append(
            {
                "symbol": sym,
                "start": args.start,
                "end": args.end,
                "sharpe_delta_candidate_minus_baseline": float(c0["sharpe"]) - float(b0["sharpe"]),
                "return_delta_candidate_minus_baseline": float(c0["total_return"]) - float(b0["total_return"]),
                "win_rate_delta_candidate_minus_baseline": float(c0["win_rate"]) - float(b0["win_rate"]),
                "max_dd_delta_candidate_minus_baseline": float(c0["max_drawdown"]) - float(b0["max_drawdown"]),
                "trade_count_delta_candidate_minus_baseline": int(c0["trade_count"]) - int(b0["trade_count"]),
            }
        )
    delta_df = pd.DataFrame(deltas).sort_values(["symbol"]).reset_index(drop=True)
    delta_path = out_dir / f"pa_env_vs_monitor_delta_{ts}.csv"
    delta_df.to_csv(delta_path, index=False)

    print(summary.to_string(index=False))
    print("")
    if not delta_df.empty:
        print(delta_df.to_string(index=False))
        print("")
    print(f"Summary: {summary_path}")
    print(f"Delta:   {delta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


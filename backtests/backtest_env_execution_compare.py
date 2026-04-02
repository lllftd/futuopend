"""
Compare baseline (CE+ZLSMA+KAMA only) vs environment-execution-enhanced model.

Usage:
  python3 -m backtests.backtest_env_execution_compare --start 2024-01-01 --end 2026-12-31
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
from core.v3_data_utils import prepare_featured as _prepare_featured
from core.v3_data_utils import slice_range as _slice_range


OUT_DIR = Path("results") / "env_execution_compare"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs env-execution backtests.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols (default: SPY QQQ).")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    return p.parse_args()


def _baseline_params(params: RuleParams) -> RuleParams:
    # Keep CE + ZLSMA + KAMA core logic, disable all optional PA execution layers.
    return replace(
        params,
        pa_or_filter=False,
        pa_require_signal_bar=False,
        pa_require_h2_l2=False,
        pa_pressure_min=0.0,
        pa_use_mm_target=False,
        pa_use_pa_stops=False,
        pa_mag_bar_exit=False,
        pa_exhaustion_gap_exit=False,
        pa_regime_filter=False,
        pa_20bar_neutral=False,
        pa_ii_breakout_entry=False,
        pa_position_sizing_mode="fixed",
        pa_use_env_execution=False,
    )


def _env_params_from_baseline(params: RuleParams) -> RuleParams:
    return replace(params, pa_use_env_execution=True)


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

        base_params = _baseline_params(V3_BEST_PARAMS[sym])
        env_params = _env_params_from_baseline(base_params)

        featured = _prepare_featured(sym, base_params)
        sliced = _slice_range(featured, args.start, args.end)
        if sliced.empty:
            rows.append(
                {
                    "symbol": sym,
                    "mode": "baseline",
                    "start": args.start,
                    "end": args.end,
                    "trade_count": 0,
                }
            )
            rows.append(
                {
                    "symbol": sym,
                    "mode": "env_execution",
                    "start": args.start,
                    "end": args.end,
                    "trade_count": 0,
                }
            )
            continue

        base_res = run_intraday_rule(sliced, sym, base_params).summary
        env_res = run_intraday_rule(sliced, sym, env_params).summary

        rows.append(
            {
                "symbol": sym,
                "mode": "baseline",
                "start": args.start,
                "end": args.end,
                "total_return": base_res.get("total_return"),
                "annual_return": base_res.get("annual_return"),
                "sharpe": base_res.get("sharpe"),
                "win_rate": base_res.get("win_rate"),
                "max_drawdown": base_res.get("max_drawdown"),
                "trade_count": base_res.get("trade_count"),
                "avg_holding_minutes": base_res.get("avg_holding_minutes"),
            }
        )
        rows.append(
            {
                "symbol": sym,
                "mode": "env_execution",
                "start": args.start,
                "end": args.end,
                "total_return": env_res.get("total_return"),
                "annual_return": env_res.get("annual_return"),
                "sharpe": env_res.get("sharpe"),
                "win_rate": env_res.get("win_rate"),
                "max_drawdown": env_res.get("max_drawdown"),
                "trade_count": env_res.get("trade_count"),
                "avg_holding_minutes": env_res.get("avg_holding_minutes"),
                "pa_env_skip_count": env_res.get("pa_env_skip_count"),
                "pa_env_neutral_entry_count": env_res.get("pa_env_neutral_entry_count"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["symbol", "mode"]).reset_index(drop=True)
    out_summary = out_dir / f"env_execution_summary_{ts}.csv"
    summary.to_csv(out_summary, index=False)

    delta_rows: list[dict[str, object]] = []
    for sym, grp in summary.groupby("symbol"):
        b = grp.loc[grp["mode"] == "baseline"]
        e = grp.loc[grp["mode"] == "env_execution"]
        if b.empty or e.empty:
            continue
        b0 = b.iloc[0]
        e0 = e.iloc[0]
        delta_rows.append(
            {
                "symbol": sym,
                "start": args.start,
                "end": args.end,
                "sharpe_delta_env_minus_base": float(e0["sharpe"]) - float(b0["sharpe"]),
                "return_delta_env_minus_base": float(e0["total_return"]) - float(b0["total_return"]),
                "win_rate_delta_env_minus_base": float(e0["win_rate"]) - float(b0["win_rate"]),
                "max_dd_delta_env_minus_base": float(e0["max_drawdown"]) - float(b0["max_drawdown"]),
                "trade_count_delta_env_minus_base": int(e0["trade_count"]) - int(b0["trade_count"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows).sort_values(["symbol"]).reset_index(drop=True)
    out_delta = out_dir / f"env_execution_delta_{ts}.csv"
    delta_df.to_csv(out_delta, index=False)

    print(summary.to_string(index=False))
    print("")
    if not delta_df.empty:
        print(delta_df.to_string(index=False))
        print("")
    print(f"Summary: {out_summary}")
    print(f"Delta:   {out_delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


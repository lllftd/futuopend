"""
Grid search grouped PA-environment thresholds on top of current monitor baseline.

Usage:
  python3 -m backtests.grid_search_env_group_thresholds \
    --start 2025-01-01 --end 2025-12-31 --symbols SPY QQQ --max-workers 4
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import RuleParams, run_intraday_rule
from core.v3_data_utils import prepare_featured_with_monitor_ce as _prepare_featured_with_monitor_ce
from core.v3_data_utils import slice_range as _slice_range


OUT_DIR = Path("results") / "env_group_threshold_search"
CE_VALID_BARS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search grouped PA-env thresholds.")
    p.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD.")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols.")
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Worker count.")
    p.add_argument("--min-trades", type=int, default=150, help="Minimum trades to mark eligible.")
    p.add_argument("--top-k", type=int, default=10, help="Rows to print per symbol.")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    return p.parse_args()


def _build_params(base: RuleParams, combo: tuple[int, float, float, int, float, float]) -> RuleParams:
    w_lb, w_pt, w_st, tr_lb, tr_pt, tr_st = combo
    return replace(
        base,
        pa_use_env_execution=True,
        pa_mtf_trend_filter=True,
        pa_mtf_min_rr=2.5,
        pa_use_advanced_pa_triggers=True,
        pa_env_ce_lookback_wide=int(w_lb),
        pa_env_price_tol_wide=float(w_pt),
        pa_env_slope_tol_wide=float(w_st),
        pa_env_ce_lookback_tr=int(tr_lb),
        pa_env_price_tol_tr=float(tr_pt),
        pa_env_slope_tol_tr=float(tr_st),
        pa_env_ce_lookback_neutral=int(tr_lb),
        pa_env_price_tol_neutral=float(tr_pt),
        pa_env_slope_tol_neutral=float(tr_st),
    )


def _score(summary: dict[str, object], min_trades: int) -> tuple[bool, float]:
    sharpe = float(summary.get("sharpe", np.nan))
    ret = float(summary.get("total_return", np.nan))
    win = float(summary.get("win_rate", np.nan))
    dd = float(summary.get("max_drawdown", np.nan))
    trades = int(summary.get("trade_count", 0))
    eligible = bool(np.isfinite(sharpe) and np.isfinite(ret) and trades >= min_trades and ret > 0)
    val = -1e9
    if np.isfinite(sharpe):
        val = sharpe + 0.7 * (ret if np.isfinite(ret) else 0.0) + 0.15 * (win if np.isfinite(win) else 0.0) - 1.2 * abs(dd if np.isfinite(dd) else 0.0)
        if not eligible:
            val -= 2.0
    return eligible, float(val)


def _eval_one(
    symbol: str,
    start: str,
    end: str,
    combo: tuple[int, float, float, int, float, float],
    min_trades: int,
) -> dict[str, object]:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Downcasting object dtype arrays on \\.fillna.*",
    )
    base = V3_BEST_PARAMS[symbol]
    params = _build_params(base, combo)
    featured = _prepare_featured_with_monitor_ce(symbol, params, CE_VALID_BARS)
    sliced = _slice_range(featured, start, end)
    if sliced.empty:
        return {"symbol": symbol, "trade_count": 0, "eligible": False, "rank_score": -1e9}
    summary = run_intraday_rule(sliced, symbol, params).summary
    eligible, rank_score = _score(summary, min_trades)
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "wide_ce_lookback": combo[0],
        "wide_price_tol": combo[1],
        "wide_slope_tol": combo[2],
        "tr_ce_lookback": combo[3],
        "tr_price_tol": combo[4],
        "tr_slope_tol": combo[5],
        "trade_count": int(summary.get("trade_count", 0)),
        "total_return": float(summary.get("total_return", np.nan)),
        "sharpe": float(summary.get("sharpe", np.nan)),
        "win_rate": float(summary.get("win_rate", np.nan)),
        "max_drawdown": float(summary.get("max_drawdown", np.nan)),
        "eligible": bool(eligible),
        "rank_score": float(rank_score),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    symbols = [s.upper() for s in args.symbols]
    for s in symbols:
        if s not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS entry for {s}")

    wide_lb = [1, 2]
    wide_pt = [0.0012, 0.0018]
    wide_st = [0.00015, 0.00025]
    tr_lb = [0, 1]
    tr_pt = [0.0010]
    tr_st = [0.00030]

    combos = list(product(wide_lb, wide_pt, wide_st, tr_lb, tr_pt, tr_st))
    jobs: list[tuple[str, tuple[int, float, float, int, float, float]]] = []
    for sym in symbols:
        for c in combos:
            jobs.append((sym, c))

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [
            pool.submit(_eval_one, sym, args.start, args.end, c, args.min_trades)
            for sym, c in jobs
        ]
        pbar = tqdm(total=len(futs), desc="Env threshold grid", unit="job", dynamic_ncols=True)
        for fut in as_completed(futs):
            rows.append(fut.result())
            pbar.update(1)
        pbar.close()

    all_df = pd.DataFrame(rows).sort_values(
        ["symbol", "eligible", "rank_score", "sharpe", "total_return", "trade_count"],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)
    best_df = all_df.groupby("symbol", as_index=False).head(1).reset_index(drop=True)

    out_all = out_dir / f"env_group_thresholds_all_{ts}.csv"
    out_best = out_dir / f"env_group_thresholds_best_{ts}.csv"
    all_df.to_csv(out_all, index=False)
    best_df.to_csv(out_best, index=False)

    print("[BEST]")
    print(best_df.to_string(index=False))
    print("")
    for sym in symbols:
        top = all_df[all_df["symbol"] == sym].head(args.top_k)
        print(f"[TOP {args.top_k}] {sym}")
        print(top.to_string(index=False))
        print("")
    print(f"All:  {out_all}")
    print(f"Best: {out_best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


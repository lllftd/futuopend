"""
Async grid search for CE + ZLSMA + KAMA under 5m-PA execution framework.

Usage example:
  python3 -m backtests.async_grid_search_pa_framework \
    --start 2024-01-01 --end 2026-12-31 --symbols SPY QQQ
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import os
import warnings
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import RuleParams, run_intraday_rule
from core.v3_data_utils import prepare_featured_with_monitor_ce as _prepare_featured_with_monitor_ce
from core.v3_data_utils import slice_range as _slice_range


OUT_DIR = Path("results") / "async_pa_grid_search"
DEFAULT_CE_LENGTHS = "1,2,3"
DEFAULT_CE_MULTIPLIERS = "2.0,2.2,2.5"
DEFAULT_ZLSMA_LENGTHS = "35,40,45"
DEFAULT_KAMA_ER_LENGTHS = "9,11"
DEFAULT_KAMA_FAST_LENGTHS = "2"
DEFAULT_KAMA_SLOW_LENGTHS = "30,40"


def _parse_int_list(value: str) -> list[int]:
    out = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not out:
        raise ValueError("Empty int list")
    return out


def _parse_float_list(value: str) -> list[float]:
    out = [float(v.strip()) for v in value.split(",") if v.strip()]
    if not out:
        raise ValueError("Empty float list")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async grid search in 5m-PA framework.")
    p.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", default="2026-12-31", help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols.")
    p.add_argument("--ce-valid-bars", type=int, default=5, help="CE valid bars expansion (monitor style).")
    p.add_argument("--ce-lengths", default=DEFAULT_CE_LENGTHS, help="Comma-separated CE lengths.")
    p.add_argument("--ce-multipliers", default=DEFAULT_CE_MULTIPLIERS, help="Comma-separated CE multipliers.")
    p.add_argument("--zlsma-lengths", default=DEFAULT_ZLSMA_LENGTHS, help="Comma-separated ZLSMA lengths.")
    p.add_argument("--kama-er-lengths", default=DEFAULT_KAMA_ER_LENGTHS, help="Comma-separated KAMA ER lengths.")
    p.add_argument("--kama-fast-lengths", default=DEFAULT_KAMA_FAST_LENGTHS, help="Comma-separated KAMA fast lengths.")
    p.add_argument("--kama-slow-lengths", default=DEFAULT_KAMA_SLOW_LENGTHS, help="Comma-separated KAMA slow lengths.")
    p.add_argument("--min-trades", type=int, default=200, help="Minimum trades for candidate eligibility.")
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="Process workers.")
    p.add_argument("--top-k", type=int, default=10, help="Top rows to print per symbol.")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    return p.parse_args()


def _build_params(base: RuleParams, combo: tuple[int, float, int, int, int, int]) -> RuleParams:
    ce_len, ce_mul, zlsma_len, kama_er, kama_fast, kama_slow = combo
    return replace(
        base,
        ce_length=int(ce_len),
        ce_multiplier=float(ce_mul),
        zlsma_length=int(zlsma_len),
        kama_er_length=int(kama_er),
        kama_fast_length=int(kama_fast),
        kama_slow_length=int(kama_slow),
        pa_use_env_execution=True,
        pa_mtf_trend_filter=True,
        pa_mtf_min_rr=2.5,
        pa_use_advanced_pa_triggers=True,
    )


def _score(summary: dict[str, object], min_trades: int) -> tuple[bool, float]:
    sharpe = float(summary.get("sharpe", np.nan))
    total_return = float(summary.get("total_return", np.nan))
    win_rate = float(summary.get("win_rate", np.nan))
    max_dd = float(summary.get("max_drawdown", np.nan))
    trades = int(summary.get("trade_count", 0))
    eligible = bool(np.isfinite(sharpe) and trades >= min_trades and np.isfinite(total_return) and total_return > 0)
    rank_score = -1e9
    if np.isfinite(sharpe):
        dd_penalty = abs(max_dd) if np.isfinite(max_dd) else 0.0
        wr = win_rate if np.isfinite(win_rate) else 0.0
        ret = total_return if np.isfinite(total_return) else -1.0
        rank_score = sharpe + 0.6 * ret + 0.2 * wr - 1.5 * dd_penalty
        if not eligible:
            rank_score -= 2.0
    return eligible, float(rank_score)


def _eval_combo(
    symbol: str,
    start: str,
    end: str,
    base: RuleParams,
    combo: tuple[int, float, int, int, int, int],
    ce_valid_bars: int,
    min_trades: int,
) -> dict[str, object]:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Downcasting object dtype arrays on \\.fillna.*",
    )
    params = _build_params(base, combo)
    featured = _prepare_featured_with_monitor_ce(symbol, params, ce_valid_bars)
    sliced = _slice_range(featured, start, end)
    if sliced.empty:
        return {
            "symbol": symbol,
            "start": start,
            "end": end,
            "ce_length": combo[0],
            "ce_multiplier": combo[1],
            "zlsma_length": combo[2],
            "kama_er_length": combo[3],
            "kama_fast_length": combo[4],
            "kama_slow_length": combo[5],
            "trade_count": 0,
            "sharpe": np.nan,
            "total_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
            "eligible": False,
            "rank_score": -1e9,
        }

    summary = run_intraday_rule(sliced, symbol, params).summary
    eligible, rank_score = _score(summary, min_trades=min_trades)
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "ce_length": int(combo[0]),
        "ce_multiplier": float(combo[1]),
        "zlsma_length": int(combo[2]),
        "kama_er_length": int(combo[3]),
        "kama_fast_length": int(combo[4]),
        "kama_slow_length": int(combo[5]),
        "trade_count": int(summary.get("trade_count", 0)),
        "sharpe": float(summary.get("sharpe", np.nan)),
        "total_return": float(summary.get("total_return", np.nan)),
        "win_rate": float(summary.get("win_rate", np.nan)),
        "max_drawdown": float(summary.get("max_drawdown", np.nan)),
        "eligible": bool(eligible),
        "rank_score": float(rank_score),
    }


async def _run_async_grid(args: argparse.Namespace) -> pd.DataFrame:
    symbols = [s.upper() for s in args.symbols]
    for s in symbols:
        if s not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS entry for {s}")

    ce_lengths = _parse_int_list(args.ce_lengths)
    ce_multipliers = _parse_float_list(args.ce_multipliers)
    zlsma_lengths = _parse_int_list(args.zlsma_lengths)
    kama_er_lengths = _parse_int_list(args.kama_er_lengths)
    kama_fast_lengths = _parse_int_list(args.kama_fast_lengths)
    kama_slow_lengths = _parse_int_list(args.kama_slow_lengths)

    combos = list(product(ce_lengths, ce_multipliers, zlsma_lengths, kama_er_lengths, kama_fast_lengths, kama_slow_lengths))
    total_jobs = len(combos) * len(symbols)
    if total_jobs == 0:
        return pd.DataFrame()

    loop = asyncio.get_running_loop()
    rows: list[dict[str, object]] = []
    done = 0
    print(f"[INFO] Running {total_jobs} jobs with {args.max_workers} workers ...")

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        tasks = []
        for sym in symbols:
            base = V3_BEST_PARAMS[sym]
            for combo in combos:
                fut = loop.run_in_executor(
                    pool,
                    _eval_combo,
                    sym,
                    args.start,
                    args.end,
                    base,
                    combo,
                    args.ce_valid_bars,
                    args.min_trades,
                )
                tasks.append(fut)

        with tqdm(total=total_jobs, desc="Grid Search", unit="job", dynamic_ncols=True) as pbar:
            for fut in asyncio.as_completed(tasks):
                row = await fut
                rows.append(row)
                done += 1
                pbar.update(1)

    return pd.DataFrame(rows)


def _pick_best(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return all_rows
    ranked = all_rows.sort_values(
        by=["symbol", "eligible", "rank_score", "sharpe", "total_return", "trade_count"],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)
    best = ranked.groupby("symbol", as_index=False).head(1).reset_index(drop=True)
    return best


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows = asyncio.run(_run_async_grid(args))
    if all_rows.empty:
        print("No rows generated.")
        return 0

    all_rows = all_rows.sort_values(
        by=["symbol", "eligible", "rank_score", "sharpe", "total_return", "trade_count"],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)
    best = _pick_best(all_rows)

    out_all = out_dir / f"pa_framework_grid_all_{ts}.csv"
    out_best = out_dir / f"pa_framework_grid_best_{ts}.csv"
    all_rows.to_csv(out_all, index=False)
    best.to_csv(out_best, index=False)

    print("")
    print("[BEST PARAMS BY SYMBOL]")
    print(best.to_string(index=False))
    print("")
    for sym in best["symbol"].tolist():
        top_df = all_rows[all_rows["symbol"] == sym].head(args.top_k)
        print(f"[TOP {args.top_k}] {sym}")
        print(top_df.to_string(index=False))
        print("")

    print(f"All rows: {out_all.as_posix()}")
    print(f"Best:     {out_best.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Test Relaxed Indicators gate under PA environments, with different CE params by environment.

Usage:
  python3 -m backtests.grid_search_env_relaxed_indicators --start 2025-01-01 --end 2025-12-31 --symbols SPY QQQ
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
from core.optimize_ce_zlsma_kama_rule import RuleParams, apply_ce_features, run_intraday_rule
from core.v3_data_utils import prepare_featured as _prepare_featured
from core.v3_data_utils import expand_signal_same_day, slice_range as _slice_range


OUT_DIR = Path("results") / "env_relaxed_indicators_search"
CE_VALID_BARS = 5
_WORKER_FEATURED_BY_SYMBOL: dict[str, pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search env-specific CE under PA framework.")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 6) - 2))
    p.add_argument("--min-trades", type=int, default=80)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def _init_worker(symbols: list[str]) -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Downcasting object dtype arrays on \\.fillna.*",
    )
    global _WORKER_FEATURED_BY_SYMBOL
    _WORKER_FEATURED_BY_SYMBOL = {}
    for sym in symbols:
        base = V3_BEST_PARAMS[sym]
        params = _build_params(base)
        _WORKER_FEATURED_BY_SYMBOL[sym] = _prepare_featured(sym, params)


def _score(summary: dict[str, object], min_trades: int) -> tuple[bool, float]:
    sharpe = float(summary.get("sharpe", np.nan))
    ret = float(summary.get("total_return", np.nan))
    wr = float(summary.get("win_rate", np.nan))
    dd = float(summary.get("max_drawdown", np.nan))
    trades = int(summary.get("trade_count", 0))
    eligible = bool(np.isfinite(sharpe) and trades >= min_trades and np.isfinite(ret) and ret > 0)
    rank = -1e9
    if np.isfinite(sharpe):
        rank = sharpe + 0.8 * ret + 0.2 * (wr if np.isfinite(wr) else 0.0) - 1.1 * abs(dd if np.isfinite(dd) else 0.0)
        if not eligible:
            rank -= 1.5
    return eligible, float(rank)


def _build_params(base: RuleParams) -> RuleParams:
    # Keep PA environment execution, but use Relaxed Indicators gate.
    return replace(
        base,
        pa_use_env_execution=True,
        pa_mtf_trend_filter=False,
        pa_use_advanced_pa_triggers=False,
        pa_env_ce_only_gate=False,
        pa_env_relaxed_indicators=True,
        cvd_classic_divergence=False,
        cvd_slope_divergence=False,
        pa_require_signal_bar=False,
        pa_require_h2_l2=False,
        pa_pressure_min=0.0,
    )


def _eval_one(
    symbol: str,
    start: str,
    end: str,
    trend_ce_len: int,
    trend_ce_mul: float,
    range_ce_len: int,
    range_ce_mul: float,
    min_trades: int,
) -> dict[str, object]:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Downcasting object dtype arrays on \\.fillna.*",
    )
    base = V3_BEST_PARAMS[symbol]
    params = _build_params(base)
    featured = _WORKER_FEATURED_BY_SYMBOL.get(symbol)
    if featured is None:
        featured = _prepare_featured(symbol, params)
    sliced = _slice_range(featured, start, end)
    if sliced.empty:
        return {"symbol": symbol, "eligible": False, "rank_score": -1e9, "trade_count": 0}

    # Drop existing CE columns so apply_ce_features doesn't overlap
    ce_cols = [c for c in sliced.columns if c.startswith("ce_")]
    clean_sliced = sliced.drop(columns=ce_cols)

    tr_df = apply_ce_features(clean_sliced, trend_ce_len, trend_ce_mul)
    rg_df = apply_ce_features(clean_sliced, range_ce_len, range_ce_mul)

    env_state = sliced["pa_env_state"].astype(str)
    is_trend = env_state.isin(["tight_bull_channel", "wide_bull_channel", "tight_bear_channel", "wide_bear_channel"])

    work = sliced.copy()
    work["ce_buy_signal"] = np.where(is_trend.to_numpy(dtype=bool), tr_df["ce_buy_signal"].fillna(False), rg_df["ce_buy_signal"].fillna(False))
    work["ce_sell_signal"] = np.where(is_trend.to_numpy(dtype=bool), tr_df["ce_sell_signal"].fillna(False), rg_df["ce_sell_signal"].fillna(False))
    work["ce_buy_signal"] = expand_signal_same_day(work["ce_buy_signal"], work["time_key"], CE_VALID_BARS)
    work["ce_sell_signal"] = expand_signal_same_day(work["ce_sell_signal"], work["time_key"], CE_VALID_BARS)

    summary = run_intraday_rule(work, symbol, params).summary
    eligible, rank_score = _score(summary, min_trades=min_trades)
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "trend_ce_length": int(trend_ce_len),
        "trend_ce_multiplier": float(trend_ce_mul),
        "range_ce_length": int(range_ce_len),
        "range_ce_multiplier": float(range_ce_mul),
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

    trend_len = [1, 2]
    trend_mul = [1.8, 2.0, 2.2]
    range_len = [1, 2, 3]
    range_mul = [2.0, 2.3, 2.6]
    combos = list(product(trend_len, trend_mul, range_len, range_mul))

    futures = []
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers, initializer=_init_worker, initargs=(symbols,)) as pool:
        for sym in symbols:
            for c in combos:
                futures.append(
                    pool.submit(
                        _eval_one,
                        sym,
                        args.start,
                        args.end,
                        c[0],
                        c[1],
                        c[2],
                        c[3],
                        args.min_trades,
                    )
                )
        pbar = tqdm(total=len(futures), desc="Env Relaxed grid", unit="job", dynamic_ncols=True)
        for fut in as_completed(futures):
            rows.append(fut.result())
            pbar.update(1)
        pbar.close()

    all_df = pd.DataFrame(rows).sort_values(
        ["symbol", "eligible", "rank_score", "sharpe", "total_return", "trade_count"],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)
    best_df = all_df.groupby("symbol", as_index=False).head(1).reset_index(drop=True)

    out_all = out_dir / f"env_relaxed_all_{ts}.csv"
    out_best = out_dir / f"env_relaxed_best_{ts}.csv"
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


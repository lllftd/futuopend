"""
Backtest CE signal validity window while keeping all other rules unchanged.

Idea:
- Keep V3_BEST_PARAMS unchanged.
- Only modify ce_buy_signal / ce_sell_signal to stay valid for N bars after trigger.
- N=0 means baseline behavior.

Usage:
  python -m backtests.backtest_ce_signal_validity
  python -m backtests.backtest_ce_signal_validity --windows 0 1 2 3 5 8 --symbols SPY QQQ
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import run_intraday_rule
from core.v3_data_utils import expand_signal_same_day, prepare_featured


OUT_DIR = Path("results") / "ce_validity_backtest"
DEFAULT_MODES = ["0", "1", "2", "3", "4", "5", "until_reverse"]
_FEATURED_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest CE signal validity window (others fixed).",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help="CE validity modes, e.g. 0 1 2 3 4 5 until_reverse",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ"],
        help="Symbols to run (default: SPY QQQ).",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Output directory for comparison CSVs.",
    )
    p.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD (inclusive). Default 2024-01-01.")
    p.add_argument("--end", default="2026-12-31", help="End date YYYY-MM-DD (inclusive). Default 2026-12-31.")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2), help="Parallel worker processes.")
    return p.parse_args()


def _apply_ce_validity_window(featured: pd.DataFrame, keep_bars: int) -> pd.DataFrame:
    df = featured.copy()
    if keep_bars <= 0:
        return df
    df["ce_buy_signal"] = expand_signal_same_day(df["ce_buy_signal"], df["time_key"], keep_bars)
    df["ce_sell_signal"] = expand_signal_same_day(df["ce_sell_signal"], df["time_key"], keep_bars)
    return df


def _apply_ce_validity_until_reverse(featured: pd.DataFrame) -> pd.DataFrame:
    """
    Keep CE side valid until opposite CE trigger appears (same day).
    """
    df = featured.copy()
    buy_src = df["ce_buy_signal"].fillna(False).astype(bool)
    sell_src = df["ce_sell_signal"].fillna(False).astype(bool)
    days = df["time_key"].dt.date

    buy_out = pd.Series(False, index=df.index)
    sell_out = pd.Series(False, index=df.index)

    for day in days.unique():
        idx = df.index[days == day]
        b = buy_src.loc[idx].to_numpy(dtype=bool)
        s = sell_src.loc[idx].to_numpy(dtype=bool)
        bo = np.zeros(len(idx), dtype=bool)
        so = np.zeros(len(idx), dtype=bool)
        state = 0  # 1=buy valid, -1=sell valid, 0=none
        for i in range(len(idx)):
            if b[i]:
                state = 1
            if s[i]:
                state = -1
            bo[i] = state == 1
            so[i] = state == -1
        buy_out.loc[idx] = bo
        sell_out.loc[idx] = so

    df["ce_buy_signal"] = buy_out
    df["ce_sell_signal"] = sell_out
    return df


def _slice_date_range(featured: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = featured
    if start:
        start_ts = pd.Timestamp(start)
        out = out.loc[out["time_key"] >= start_ts]
    if end:
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        out = out.loc[out["time_key"] < end_exclusive]
    return out.reset_index(drop=True)


def _get_featured_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    key = (symbol, start, end)
    cached = _FEATURED_CACHE.get(key)
    if cached is not None:
        return cached

    if symbol not in V3_BEST_PARAMS:
        raise RuntimeError(f"No V3_BEST_PARAMS entry for {symbol}")
    params = V3_BEST_PARAMS[symbol]
    featured = prepare_featured(symbol, params)
    featured = _slice_date_range(featured, start, end)
    _FEATURED_CACHE[key] = featured
    return featured


def _run_one_task(symbol: str, mode: str, start: str, end: str) -> dict[str, object] | None:
    featured = _get_featured_cached(symbol, start, end)
    if featured.empty:
        return None
    params = V3_BEST_PARAMS[symbol]

    mode_s = str(mode).strip().lower()
    if mode_s == "until_reverse":
        df_test = _apply_ce_validity_until_reverse(featured)
        ce_valid_bars = np.nan
    else:
        keep_bars = int(mode_s)
        if keep_bars < 0:
            raise RuntimeError(f"Invalid mode: {mode}")
        df_test = _apply_ce_validity_window(featured, keep_bars)
        ce_valid_bars = keep_bars

    res = run_intraday_rule(df_test, symbol, params)
    s = res.summary
    return {
        "symbol": symbol,
        "ce_valid_mode": mode_s,
        "ce_valid_bars": ce_valid_bars,
        "start": start,
        "end": end,
        "total_return": s.get("total_return"),
        "annual_return": s.get("annual_return"),
        "sharpe": s.get("sharpe"),
        "win_rate": s.get("win_rate"),
        "max_drawdown": s.get("max_drawdown"),
        "trade_count": s.get("trade_count"),
        "avg_holding_minutes": s.get("avg_holding_minutes"),
        "take_profit_rate": s.get("take_profit_rate"),
        "stop_loss_rate": s.get("stop_loss_rate"),
        "time_stop_rate": s.get("time_stop_rate"),
    }


def main() -> int:
    args = parse_args()
    pd.Timestamp(args.start)
    pd.Timestamp(args.end)
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise SystemExit("--start must be <= --end")

    raw_modes = [str(m).strip().lower() for m in args.modes]
    modes: list[str] = []
    for m in raw_modes:
        if m == "until_reverse":
            modes.append(m)
            continue
        try:
            iv = int(m)
        except ValueError as exc:
            raise SystemExit(f"Invalid mode: {m}") from exc
        if iv < 0:
            raise SystemExit(f"Invalid mode: {m}")
        modes.append(str(iv))
    modes = list(dict.fromkeys(modes))
    if not modes:
        raise SystemExit("No valid --modes provided.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    symbols = [s.upper() for s in args.symbols]
    rows: list[dict[str, object]] = []
    tasks = [(symbol, mode) for symbol in symbols for mode in modes]
    workers = max(1, min(int(args.workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_run_one_task, symbol, mode, args.start, args.end)
            for symbol, mode in tasks
        ]
        for fut in as_completed(futs):
            row = fut.result()
            if row is not None:
                rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["symbol", "ce_valid_mode"]).reset_index(drop=True)
    out_path = out_dir / f"ce_validity_comparison_{ts}.csv"
    summary.to_csv(out_path, index=False)

    # Delta table versus baseline window=0
    deltas: list[dict[str, object]] = []
    for symbol, sub in summary.groupby("symbol"):
        base_row = sub.loc[sub["ce_valid_mode"] == "0"]
        if base_row.empty:
            continue
        b = base_row.iloc[0]
        for _, r in sub.iterrows():
            deltas.append(
                {
                    "symbol": symbol,
                    "ce_valid_mode": str(r["ce_valid_mode"]),
                    "sharpe_delta_vs_w0": float(r["sharpe"]) - float(b["sharpe"]),
                    "return_delta_vs_w0": float(r["total_return"]) - float(b["total_return"]),
                    "trade_count_delta_vs_w0": int(r["trade_count"]) - int(b["trade_count"]),
                    "max_dd_delta_vs_w0": float(r["max_drawdown"]) - float(b["max_drawdown"]),
                }
            )
    delta_df = pd.DataFrame(deltas).sort_values(["symbol", "ce_valid_mode"]).reset_index(drop=True)
    delta_path = out_dir / f"ce_validity_delta_vs_w0_{ts}.csv"
    delta_df.to_csv(delta_path, index=False)

    print(summary.to_string(index=False))
    print("")
    print(delta_df.to_string(index=False))
    print("")
    print(f"Comparison: {out_path}")
    print(f"Deltas:     {delta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

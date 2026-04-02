"""
Evaluate 1-min CE+ZLSMA+KAMA signals under different PA environments/scores,
with PA structure points (pa_stop / pa_mm_target) as risk-reward anchors.

Usage:
  python3 -m backtests.backtest_env_score_signal_pa_points \
    --start 2025-01-01 --end 2025-12-31 --symbols SPY QQQ
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
from core.optimize_ce_zlsma_kama_rule import build_entry_signals
from core.v3_data_utils import prepare_featured_with_monitor_ce as _prepare_featured_with_monitor_ce
from core.v3_data_utils import slice_range as _slice_range


OUT_DIR = Path("results") / "env_score_signal_pa_points"
CE_VALID_BARS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test 1m CE/ZLSMA/KAMA signals by PA environment score with PA points.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"], help="Symbols.")
    p.add_argument("--horizon-bars", type=int, default=30, help="Forward bars to evaluate each signal.")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    return p.parse_args()


def _score_bucket(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score >= 6:
        return "tight_bull_6_8"
    if score >= 3:
        return "wide_bull_3_5"
    if score >= -2:
        return "tr_-2_2"
    if score >= -5:
        return "wide_bear_-5_-3"
    return "tight_bear_-8_-6"


def _simulate_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
    side: int,
    entry: float,
    stop: float,
    target: float,
) -> tuple[str, int, float]:
    """
    Returns:
      outcome: target_hit / stop_hit / timeout
      bars_held: holding bars
      realized_rr: realized R multiple
    """
    risk = abs(entry - stop)
    if risk <= 1e-12:
        return "invalid", 0, np.nan

    for j in range(start_idx, end_idx + 1):
        if side == 1:
            stop_hit = low[j] <= stop
            target_hit = high[j] >= target
        else:
            stop_hit = high[j] >= stop
            target_hit = low[j] <= target

        if stop_hit and target_hit:
            # Conservative sequencing: stop first in same bar.
            return "stop_hit", j - start_idx + 1, -1.0
        if stop_hit:
            return "stop_hit", j - start_idx + 1, -1.0
        if target_hit:
            rr = abs(target - entry) / risk
            return "target_hit", j - start_idx + 1, rr

    final_px = close[end_idx]
    if side == 1:
        rr = (final_px - entry) / risk
    else:
        rr = (entry - final_px) / risk
    return "timeout", end_idx - start_idx + 1, float(rr)


def _evaluate_symbol(
    symbol: str,
    start: str,
    end: str,
    horizon_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = V3_BEST_PARAMS[symbol]
    featured = _prepare_featured_with_monitor_ce(symbol, params, CE_VALID_BARS)
    df = _slice_range(featured, start, end)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    long_sig, short_sig = build_entry_signals(df, params)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    opn = df["open"].to_numpy(dtype=float)
    pa_stop_long = df["pa_stop_long"].to_numpy(dtype=float) if "pa_stop_long" in df.columns else np.full(len(df), np.nan)
    pa_stop_short = df["pa_stop_short"].to_numpy(dtype=float) if "pa_stop_short" in df.columns else np.full(len(df), np.nan)
    pa_mm_up = df["pa_mm_target_up"].to_numpy(dtype=float) if "pa_mm_target_up" in df.columns else np.full(len(df), np.nan)
    pa_mm_down = df["pa_mm_target_down"].to_numpy(dtype=float) if "pa_mm_target_down" in df.columns else np.full(len(df), np.nan)
    env_state = df["pa_env_state"].astype(str).to_numpy() if "pa_env_state" in df.columns else np.array([""] * len(df))
    env_score = pd.to_numeric(df.get("pa_env_score_total", np.nan), errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    n = len(df)
    for idx in range(n - 1):
        entry_idx = idx + 1
        eval_end = min(n - 1, entry_idx + horizon_bars - 1)
        if eval_end < entry_idx:
            continue

        if bool(long_sig.iloc[idx]):
            entry = opn[entry_idx]
            stop = pa_stop_long[idx]
            target = pa_mm_up[idx]
            if np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target) and stop < entry < target:
                outcome, bars_held, realized_rr = _simulate_trade(
                    high, low, close, entry_idx, eval_end, 1, entry, stop, target
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "time_key": df.iloc[idx]["time_key"],
                        "side": "long",
                        "env_state": env_state[idx],
                        "env_score_total": env_score[idx],
                        "env_score_bucket": _score_bucket(env_score[idx]),
                        "entry_price": entry,
                        "pa_stop": stop,
                        "pa_target": target,
                        "pa_rr_target": abs(target - entry) / abs(entry - stop),
                        "outcome": outcome,
                        "bars_held": bars_held,
                        "realized_rr": realized_rr,
                    }
                )

        if bool(short_sig.iloc[idx]):
            entry = opn[entry_idx]
            stop = pa_stop_short[idx]
            target = pa_mm_down[idx]
            if np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target) and target < entry < stop:
                outcome, bars_held, realized_rr = _simulate_trade(
                    high, low, close, entry_idx, eval_end, -1, entry, stop, target
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "time_key": df.iloc[idx]["time_key"],
                        "side": "short",
                        "env_state": env_state[idx],
                        "env_score_total": env_score[idx],
                        "env_score_bucket": _score_bucket(env_score[idx]),
                        "entry_price": entry,
                        "pa_stop": stop,
                        "pa_target": target,
                        "pa_rr_target": abs(entry - target) / abs(stop - entry),
                        "outcome": outcome,
                        "bars_held": bars_held,
                        "realized_rr": realized_rr,
                    }
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    detail["is_target_hit"] = detail["outcome"] == "target_hit"
    detail["is_stop_hit"] = detail["outcome"] == "stop_hit"
    summary = (
        detail.groupby(["symbol", "env_state", "env_score_bucket", "side"], as_index=False)
        .agg(
            signal_count=("outcome", "size"),
            target_hit_rate=("is_target_hit", "mean"),
            stop_hit_rate=("is_stop_hit", "mean"),
            avg_realized_rr=("realized_rr", "mean"),
            median_realized_rr=("realized_rr", "median"),
            avg_target_rr=("pa_rr_target", "mean"),
            avg_bars_held=("bars_held", "mean"),
        )
        .sort_values(["symbol", "signal_count", "avg_realized_rr"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return detail, summary


def main() -> int:
    args = parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise SystemExit("--start must be <= --end")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_all: list[pd.DataFrame] = []
    summary_all: list[pd.DataFrame] = []
    for sym in [s.upper() for s in args.symbols]:
        if sym not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS entry for {sym}")
        d, s = _evaluate_symbol(sym, args.start, args.end, args.horizon_bars)
        if not d.empty:
            detail_all.append(d)
        if not s.empty:
            summary_all.append(s)

    detail_df = pd.concat(detail_all, ignore_index=True) if detail_all else pd.DataFrame()
    summary_df = pd.concat(summary_all, ignore_index=True) if summary_all else pd.DataFrame()

    detail_path = out_dir / f"env_score_signal_pa_points_detail_{ts}.csv"
    summary_path = out_dir / f"env_score_signal_pa_points_summary_{ts}.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if summary_df.empty:
        print("No valid signals with PA stop/target anchors found in this range.")
    else:
        print(summary_df.to_string(index=False))
        print("")
    print(f"Detail:  {detail_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


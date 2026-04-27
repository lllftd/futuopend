#!/usr/bin/env python3
"""Estimate share of L1c-style forward windows that cross a session gap (US RTH).

Example:
  PYTHONPATH=. python archive/diagnose_l1c_crossday.py --symbols QQQ SPY --horizon 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.training.common.constants import DATA_DIR
from core.utils.session import mark_session_boundaries


def load_symbol_csv(symbol: str) -> pd.DataFrame:
    path = Path(DATA_DIR) / f"{symbol.upper()}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=lambda c: c in {"time_key", "close", "symbol"})
    df["time_key"] = pd.to_datetime(df["time_key"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = symbol.upper()
    df = df.dropna(subset=["time_key", "close"]).sort_values("time_key")
    return df


def diagnose(df: pd.DataFrame, horizon: int, gap_seconds: float) -> pd.DataFrame:
    d = df.copy()
    mark_session_boundaries(
        d,
        overnight_gap_seconds=gap_seconds,
        open_skip_bars=0,
        close_cutoff_bars=0,
    )
    rows: list[dict[str, float | int | str]] = []
    for sym, grp in d.groupby("symbol", sort=False):
        sess = grp["session_id"].to_numpy(dtype=np.int32)
        c = pd.to_numeric(grp["close"], errors="coerce").to_numpy(dtype=np.float64)
        n = len(c)
        if n <= horizon:
            continue
        total = 0
        cross = 0
        cross_abs: list[float] = []
        intra_abs: list[float] = []
        for i in range(0, n - horizon):
            total += 1
            r = (c[i + horizon] / (abs(c[i]) + 1e-9) - 1.0) * 10000.0
            if sess[i + horizon] != sess[i]:
                cross += 1
                cross_abs.append(abs(float(r)))
            else:
                intra_abs.append(abs(float(r)))
        ca = np.asarray(cross_abs, dtype=np.float64)
        ia = np.asarray(intra_abs, dtype=np.float64)
        rows.append(
            {
                "symbol": str(sym),
                "total_windows": total,
                "cross_session": cross,
                "cross_pct": 100.0 * cross / max(total, 1),
                "cross_abs_ret_bps_mean": float(np.mean(ca)) if ca.size else float("nan"),
                "intra_abs_ret_bps_mean": float(np.mean(ia)) if ia.size else float("nan"),
                "cross_abs_ret_bps_std": float(np.std(ca)) if ca.size else float("nan"),
                "intra_abs_ret_bps_std": float(np.std(ia)) if ia.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose cross-session forward returns for L1c horizon")
    p.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"], help="Tickers (CSV under data/)")
    p.add_argument("--horizon", type=int, default=10, help="Forward bars (match L1c predict_horizon on your bar size)")
    p.add_argument("--gap-seconds", type=float, default=4 * 3600, help="New session if bar gap exceeds this")
    args = p.parse_args()

    parts = [load_symbol_csv(s) for s in args.symbols]
    df = pd.concat(parts, ignore_index=True)
    out = diagnose(df, horizon=int(args.horizon), gap_seconds=float(args.gap_seconds))

    print("=" * 72)
    print("L1c cross-session forward window diagnosis (pre-filter)")
    print(f"  horizon={args.horizon} bars session_gap>{args.gap_seconds:.0f}s")
    print("=" * 72)
    if out.empty:
        print("No data.")
        return 1
    for _, row in out.iterrows():
        print(
            f"  {row['symbol']:<6} cross={row['cross_pct']:5.2f}%  "
            f"|ret| mean cross={row['cross_abs_ret_bps_mean']:7.1f} bps  intra={row['intra_abs_ret_bps_mean']:7.1f} bps  "
            f"n={int(row['total_windows']):,}"
        )
    ratio = out["cross_abs_ret_bps_mean"] / out["intra_abs_ret_bps_mean"].replace(0, np.nan)
    if ratio.notna().any():
        print(f"\n  Mean |ret| ratio (cross / intra): {float(np.nanmean(ratio.values)):.2f}x")
    print("\n  Enable intraday-only labels: default L1C_INTRADAY_LABELS=1 in training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
MINUTES_PER_YEAR = 252 * 390


def load_price_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def resolve_v3_backtest_window(
    symbols: tuple[str, ...],
    start_req: str,
    end_req: str | None,
) -> tuple[str, str, dict[str, object]]:
    """
    Calendar dates (YYYY-MM-DD) inclusive for backtest_v3_range / plot_v3_range_charts.

    Uses the intersection of all symbol CSV time ranges, then clips to optional end_req.
    If end_req is None, uses the latest common timestamp available in data.
    """
    bounds: dict[str, dict[str, str]] = {}
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for sym in symbols:
        path = DATA_DIR / f"{sym}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        t = pd.read_csv(path, usecols=["time_key"])
        ts = pd.to_datetime(t["time_key"])
        tmin, tmax = ts.min(), ts.max()
        bounds[sym] = {"min": str(tmin), "max": str(tmax)}
        mins.append(tmin)
        maxs.append(tmax)
    common_lo = max(mins)
    common_hi = min(maxs)
    rs = pd.Timestamp(start_req).normalize()
    re_hi = pd.Timestamp(end_req).normalize() if end_req else common_hi.normalize()
    w_lo = max(rs, common_lo.normalize())
    w_hi = min(re_hi, common_hi.normalize())
    if w_lo > w_hi:
        raise ValueError(
            f"No usable overlap for symbols={symbols}: common {common_lo}..{common_hi}, "
            f"requested {start_req}..{end_req or 'max'}"
        )
    start_date = w_lo.date().isoformat()
    end_date = w_hi.date().isoformat()
    meta: dict[str, object] = {
        "requested_start": start_req,
        "requested_end": end_req,
        "per_symbol_bounds": bounds,
        "common_data_range": {"min": str(common_lo), "max": str(common_hi)},
        "backtest_start": start_date,
        "backtest_end": end_date,
        "start_clipped_to_data": common_lo.normalize() > rs,
        "end_clipped_to_data": bool(end_req) and re_hi.normalize() > common_hi.normalize(),
    }
    return start_date, end_date, meta


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0

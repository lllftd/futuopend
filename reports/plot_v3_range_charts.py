"""
Plot minute-resolution price (high-low band + close) and strategy cumulative return.

Uses the same slice and V3_BEST_PARAMS as backtest_v3_range.py. Does not aggregate bars
to daily or higher — every row in data/*.csv is one minute.

Usage:
  python -m reports.plot_v3_range_charts --start 2020-03-28 --end 2024-03-27
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import run_intraday_rule
from core.v3_data_utils import prepare_featured as _prepare_featured
from core.v3_data_utils import slice_range as _slice_range

OUT_DIR = Path("results") / "v3_range_backtest" / "charts"


def _stats_text(summary: dict[str, object]) -> str:
    def fnum(k: str, nd: int = 4) -> str:
        v = summary.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.{nd}f}"

    tr = float(summary.get("total_return", np.nan))
    ar = float(summary.get("annual_return", np.nan))
    sh = float(summary.get("sharpe", np.nan))
    wr = float(summary.get("win_rate", np.nan))
    dd = float(summary.get("max_drawdown", np.nan))
    tc = int(summary.get("trade_count", 0))
    ah = float(summary.get("avg_holding_minutes", np.nan))

    lines = [
        f"Sharpe: {fnum('sharpe', 3)}",
        f"Total return: {tr * 100:.2f}%",
        f"Ann. return: {ar * 100:.2f}%",
        f"Max DD: {dd * 100:.2f}%",
        f"Win rate: {wr * 100:.2f}%",
        f"Trades: {tc}",
        f"Avg hold (min): {fnum('avg_holding_minutes', 2)}",
    ]
    return "\n".join(lines)


def plot_one(
    symbol: str,
    start: str,
    end: str,
    out_path: Path,
    dpi: int,
) -> None:
    params = V3_BEST_PARAMS[symbol.upper()]
    featured = _prepare_featured(symbol.upper(), params)
    sliced = _slice_range(featured, start, end)
    if sliced.empty:
        raise RuntimeError(f"No bars for {symbol} in range.")

    result = run_intraday_rule(sliced, symbol.upper(), params)
    s = result.summary
    eq = result.equity_curve
    cum_pct = (eq - 1.0) * 100.0

    times = pd.to_datetime(sliced["time_key"], utc=False)
    high = sliced["high"].to_numpy(dtype=float)
    low = sliced["low"].to_numpy(dtype=float)
    close = sliced["close"].to_numpy(dtype=float)
    tnum = mdates.date2num(times)

    fig, ax_p = plt.subplots(figsize=(18, 7), dpi=dpi)
    # Minute range: low–high band (full resolution, no aggregation)
    ax_p.fill_between(tnum, low, high, facecolor="#94a3b8", edgecolor="none", alpha=0.22, linewidth=0)
    ax_p.plot(tnum, close, color="#0f172a", linewidth=0.35, alpha=0.9, label="Close (1m)")
    ax_p.set_ylabel(f"{symbol.upper()} price", color="#0f172a")
    ax_p.tick_params(axis="y", labelcolor="#0f172a")
    ax_p.grid(True, alpha=0.25)
    ax_p.set_title(f"{symbol.upper()} — minute OHLC + V3 cumulative return  ({start} ~ {end})")

    ax_r = ax_p.twinx()
    ax_r.plot(mdates.date2num(pd.to_datetime(eq.index)), cum_pct.to_numpy(dtype=float), color="#16a34a", linewidth=0.85, label="Cum. return %")
    ax_r.set_ylabel("Strategy cumulative return (%)", color="#16a34a")
    ax_r.tick_params(axis="y", labelcolor="#16a34a")

    ax_p.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    txt = _stats_text(s)
    fig.text(
        0.99,
        0.98,
        txt,
        transform=ax_p.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cbd5e1", alpha=0.92),
        family="monospace",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot minute K + cumulative return for V3 backtest range.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    out_dir = Path(args.output_dir)
    tag = f"{args.start}_to_{args.end}"

    for sym in args.symbols:
        sym_u = sym.upper()
        if sym_u not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS for {sym_u}")
        out_png = out_dir / f"{sym_u}_minute_k_equity_{tag}.png"
        print(f"Plotting {sym_u} -> {out_png} (re-running backtest for equity curve, may take ~2–3 min)...")
        plot_one(sym_u, args.start, args.end, out_png, args.dpi)
        print(f"  wrote {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

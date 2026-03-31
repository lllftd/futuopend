"""
V3.5 daily charts: V3_BEST_PARAMS + same-day CE validity as live/monitor (CE_SIGNAL_VALID_BARS).

For each trading day in [start, end], runs run_intraday_rule on that day’s 1m bars only
(indicators were computed on full history, then CE expanded on full history, then sliced by day).

Usage:
  python -m reports.v35_daily_backtest_charts
  python -m reports.v35_daily_backtest_charts --start 2025-03-28 --end 2026-03-30 --ce-valid-bars 5
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from core.config import V3_BEST_PARAMS
from core.optimize_ce_zlsma_kama_rule import run_intraday_rule
from core.v3_data_utils import prepare_featured_with_monitor_ce, slice_range
from reports.plot_v3_range_charts import _stats_text

DEFAULT_OUT = Path("results") / "v3.5backtest"


def _draw_candles(ax, tnum: np.ndarray, opens, highs, lows, closes) -> None:
    n = len(tnum)
    if n == 0:
        return
    width = (tnum[1] - tnum[0]) * 0.55 if n > 1 else 0.00035
    for i in range(n):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        body_color = "#22c55e" if c >= o else "#ef4444"
        ax.plot([tnum[i], tnum[i]], [l, h], color="#475569", linewidth=0.45, solid_capstyle="round", zorder=2)
        bot, top = sorted([o, c])
        if top - bot < 1e-8:
            ax.plot([tnum[i] - width / 2, tnum[i] + width / 2], [c, c], color=body_color, linewidth=0.8, zorder=3)
        else:
            ax.bar(
                tnum[i],
                top - bot,
                width=width,
                bottom=bot,
                color=body_color,
                edgecolor="#0f172a",
                linewidth=0.15,
                zorder=3,
            )


def _annotate_trades(ax_p, trade_log: pd.DataFrame) -> None:
    if trade_log is None or trade_log.empty:
        return
    for k, (_, tr) in enumerate(trade_log.iterrows()):
        et = pd.to_datetime(tr["entry_time"])
        xt = pd.to_datetime(tr["exit_time"])
        en, xn = mdates.date2num(et), mdates.date2num(xt)
        entry_p = float(tr["entry_price"])
        exit_p = float(tr["exit_price"])
        is_long = str(tr.get("side", "")).lower() == "long"
        reason = str(tr.get("exit_reason", ""))

        ax_p.axvline(en, color="#2563eb", alpha=0.35, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
        ax_p.axvline(xn, color="#a855f7", alpha=0.45, linewidth=0.85, linestyle=(0, (2, 2)), zorder=4)

        ax_p.scatter(
            [en],
            [entry_p],
            marker="^" if is_long else "v",
            s=55,
            color="#2563eb" if is_long else "#ea580c",
            edgecolors="#0f172a",
            linewidths=0.35,
            zorder=6,
        )
        ax_p.scatter(
            [xn],
            [exit_p],
            marker="x",
            s=56,
            color="#be123c",
            linewidths=0.9,
            zorder=6,
        )

        time_lbl = et.strftime("%H:%M")
        ax_p.annotate(
            f"in {time_lbl}",
            xy=(en, entry_p),
            xytext=(0, 8 + (k % 4) * 3),
            textcoords="offset points",
            fontsize=5.5,
            ha="center",
            color="#1d4ed8",
            alpha=0.9,
        )
        ax_p.annotate(
            f"out {reason}\n{xt.strftime('%H:%M')}",
            xy=(xn, exit_p),
            xytext=(0, -(10 + (k % 5) * 4)),
            textcoords="offset points",
            fontsize=5.5,
            ha="center",
            va="top",
            color="#6b21a8",
            alpha=0.95,
            linespacing=0.95,
        )


def _plot_one_day(
    symbol: str,
    day: date,
    day_df: pd.DataFrame,
    result,
    out_path: Path,
    dpi: int,
    ce_valid_bars: int,
) -> None:
    s = result.summary
    eq = result.equity_curve
    cum_pct = (eq - 1.0) * 100.0
    trade_log = result.trade_log

    times = pd.to_datetime(day_df["time_key"], utc=False)
    opens = day_df["open"].to_numpy(dtype=float)
    high = day_df["high"].to_numpy(dtype=float)
    low = day_df["low"].to_numpy(dtype=float)
    close = day_df["close"].to_numpy(dtype=float)
    tnum = mdates.date2num(times)

    fig, ax_p = plt.subplots(figsize=(15, 7), dpi=dpi)
    _draw_candles(ax_p, tnum, opens, high, low, close)
    y_lo, y_hi = float(np.nanmin(low)), float(np.nanmax(high))
    _annotate_trades(ax_p, trade_log)

    ax_p.set_ylabel(f"{symbol.upper()} price (1m candles)", color="#0f172a")
    ax_p.tick_params(axis="y", labelcolor="#0f172a")
    ax_p.grid(True, alpha=0.22)
    ax_p.set_title(
        f"{symbol.upper()} {day.isoformat()} — V3.5 candles + CE +{ce_valid_bars} bar (monitor-aligned)"
    )

    legend_elements = [
        Line2D([0], [0], color="#2563eb", linestyle=(0, (3, 2)), label="Entry time"),
        Line2D([0], [0], color="#a855f7", linestyle=(0, (2, 2)), label="Exit time"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2563eb", markersize=8, label="Long entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#ea580c", markersize=8, label="Short entry"),
        Line2D([0], [0], marker="x", color="#be123c", linestyle="None", markersize=8, label="Exit (reason text)"),
    ]
    ax_p.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)

    ax_r = ax_p.twinx()
    ax_r.plot(
        mdates.date2num(pd.to_datetime(eq.index)),
        cum_pct.to_numpy(dtype=float),
        color="#16a34a",
        linewidth=1.0,
        label="Day cum. return %",
    )
    ax_r.set_ylabel("Intraday cumulative return (%)", color="#16a34a")
    ax_r.tick_params(axis="y", labelcolor="#16a34a")

    ax_p.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    txt = _stats_text(s) + f"\nCE valid bars: {ce_valid_bars}"
    fig.text(
        0.99,
        0.98,
        txt,
        transform=ax_p.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.92),
        family="monospace",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3.5 per-day backtest PNGs (CE validity aligned with monitor).")
    p.add_argument("--start", default="2025-03-28", help="First calendar day (inclusive).")
    p.add_argument("--end", default="2026-03-30", help="Last calendar day (inclusive).")
    p.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ce-valid-bars", type=int, default=5, help="Same as live.monitor CE_SIGNAL_VALID_BARS.")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    out_dir: Path = args.output_dir
    ce_bars = int(args.ce_valid_bars)

    start_d = pd.Timestamp(args.start).date()
    end_d = pd.Timestamp(args.end).date()
    if start_d > end_d:
        raise SystemExit("--start must be <= --end")

    rows: list[dict[str, object]] = []

    for sym in args.symbols:
        sym_u = sym.upper()
        if sym_u not in V3_BEST_PARAMS:
            raise SystemExit(f"No V3_BEST_PARAMS for {sym_u}")
        params = V3_BEST_PARAMS[sym_u]

        featured_full = prepare_featured_with_monitor_ce(sym_u, params, ce_bars)
        featured = slice_range(featured_full, args.start, args.end)
        if featured.empty:
            print(f"[{sym_u}] no bars in {args.start}..{args.end}, skip")
            continue

        days = sorted(featured["time_key"].dt.date.unique())
        sym_dir = out_dir / sym_u
        for day in tqdm(days, desc=f"{sym_u} days"):
            if day < start_d or day > end_d:
                continue
            day_df = featured.loc[featured["time_key"].dt.date == day].reset_index(drop=True)
            if day_df.empty:
                continue
            result = run_intraday_rule(day_df, sym_u, params)
            s = result.summary
            out_png = sym_dir / f"{day.isoformat()}.png"
            _plot_one_day(sym_u, day, day_df, result, out_png, args.dpi, ce_bars)
            rows.append(
                {
                    "symbol": sym_u,
                    "date": day.isoformat(),
                    "bars": len(day_df),
                    "total_return": s.get("total_return"),
                    "sharpe": s.get("sharpe"),
                    "trade_count": s.get("trade_count"),
                    "win_rate": s.get("win_rate"),
                    "max_drawdown": s.get("max_drawdown"),
                    "ce_valid_bars": ce_bars,
                    "png": out_png.as_posix(),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"daily_summary_{args.start}_to_{args.end}_{ts}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nWrote {len(rows)} day charts. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Export trade logs and per-day candlestick + trade markers + intraday cumulative return charts (v3).
Includes Opening Range shading, MM target levels, holding/return distribution plots,
and price vs cumulative return equity chart.
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import time as dt_time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams.setdefault("font.sans-serif", ["DejaVu Sans"])
plt.rcParams.setdefault("axes.unicode_minus", False)
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm

from core.equity_chart_utils import create_equity_chart as _create_equity_chart
from core.optimize_ce_zlsma_kama_rule import (
    RuleParams,
    OptimizationResult,
    apply_ce_features,
    build_base_features,
    run_intraday_rule,
)
from reports.plot_trade_distributions import plot_trade_distribution
from core.ranking_utils import load_best_params_and_rows as load_best_params_by_symbol
from core.utils import load_price_data

EXIT_REASON_EN = {
    "take_profit": "take profit",
    "stop_loss": "stop loss",
    "time_stop": "time stop",
    "end_of_day": "end of day",
    "mag_bar_exit": "MAG bar",
    "exhaustion_gap_exit": "exhaust gap",
}

def run_backtest_symbol(symbol: str, params: RuleParams) -> tuple[pd.DataFrame, OptimizationResult]:
    raw = load_price_data(symbol)
    base = build_base_features(
        raw,
        zlsma_length=params.zlsma_length,
        zlsma_offset=params.zlsma_offset,
        kama_er_length=params.kama_er_length,
        kama_fast_length=params.kama_fast_length,
        kama_slow_length=params.kama_slow_length,
        atr_percentile_lookback=params.atr_percentile_lookback,
        pseudo_cvd_method=params.pseudo_cvd_method,
        cvd_lookback=params.cvd_lookback,
        cvd_slope_lookback=params.cvd_slope_lookback,
    )
    featured = apply_ce_features(base, params.ce_length, params.ce_multiplier)
    return featured, run_intraday_rule(featured, symbol, params)


def unique_trading_days_desc(df: pd.DataFrame, last_n: int) -> list[pd.Timestamp]:
    dates = pd.Series(df["time_key"].dt.normalize().unique()).sort_values()
    tail = dates.tail(last_n)
    return [pd.Timestamp(d) for d in tail]


def draw_candles(ax: plt.Axes, day_df: pd.DataFrame) -> None:
    if len(day_df) < 1:
        return
    t = pd.to_datetime(day_df["time_key"])
    x = mdates.date2num(t)
    width = float(np.median(np.diff(x))) * 0.65 if len(x) > 1 else 0.00025
    o = day_df["open"].astype(float).to_numpy()
    h = day_df["high"].astype(float).to_numpy()
    l_ = day_df["low"].astype(float).to_numpy()
    c = day_df["close"].astype(float).to_numpy()
    for i in range(len(day_df)):
        color = "#2ca02c" if c[i] >= o[i] else "#d62728"
        ax.plot([x[i], x[i]], [l_[i], h[i]], color=color, linewidth=0.7, solid_capstyle="round")
        bh = max(abs(c[i] - o[i]), (h[i] - l_[i]) * 0.02 if h[i] != l_[i] else 0.02)
        bottom = min(o[i], c[i])
        ax.add_patch(
            Rectangle(
                (x[i] - width / 2.0, bottom),
                width,
                bh,
                facecolor=color,
                edgecolor=color,
                linewidth=0.3,
                zorder=3,
            )
        )


def _draw_opening_range(ax: plt.Axes, day_bars: pd.DataFrame, day: pd.Timestamp) -> None:
    if day_bars.empty:
        return
    times = pd.to_datetime(day_bars["time_key"])
    bar_time = times.dt.time
    or_mask = (bar_time >= dt_time(9, 30)) & (bar_time < dt_time(11, 0))
    or_bars = day_bars.loc[or_mask]
    if or_bars.empty:
        return

    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())
    or_start = mdates.date2num(pd.to_datetime(or_bars["time_key"].iloc[0]))
    or_end = mdates.date2num(pd.to_datetime(or_bars["time_key"].iloc[-1]))

    ax.add_patch(
        Rectangle(
            (or_start, or_low),
            or_end - or_start,
            or_high - or_low,
            facecolor="#1f77b4",
            edgecolor="#1f77b4",
            alpha=0.10,
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
    )
    ax.axhline(or_high, color="#1f77b4", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)
    ax.axhline(or_low, color="#1f77b4", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)
    mid = (or_start + or_end) / 2
    ax.annotate(
        f"OR {or_high:.2f}/{or_low:.2f}",
        (mid, or_high),
        textcoords="offset points",
        xytext=(0, 6),
        fontsize=6,
        ha="center",
        color="#1f77b4",
        alpha=0.8,
    )


def _draw_mm_targets(ax: plt.Axes, day_bars: pd.DataFrame) -> None:
    if day_bars.empty:
        return
    if "pa_mm_target_up" not in day_bars.columns:
        return

    mm_up_vals = day_bars["pa_mm_target_up"].dropna().unique()
    mm_down_vals = day_bars["pa_mm_target_down"].dropna().unique()

    y_low = float(day_bars["low"].min())
    y_high = float(day_bars["high"].max())
    y_range = y_high - y_low

    for v in mm_up_vals[-3:]:
        if y_low - y_range * 0.5 < v < y_high + y_range * 0.5:
            ax.axhline(v, color="#17becf", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)

    for v in mm_down_vals[-3:]:
        if y_low - y_range * 0.5 < v < y_high + y_range * 0.5:
            ax.axhline(v, color="#e377c2", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)


def render_day_chart_fixed(
    *,
    symbol: str,
    day: pd.Timestamp,
    day_bars: pd.DataFrame,
    day_trades: pd.DataFrame,
    out_path: Path,
    featured_day_bars: pd.DataFrame | None = None,
) -> None:
    day_date = day.date()
    fig, (ax_k, ax_eq) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        height_ratios=[0.62, 0.38],
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    if len(day_bars) > 1:
        xnums = mdates.date2num(pd.to_datetime(day_bars["time_key"]))
        halfw = float(np.median(np.diff(xnums))) * 0.65
    else:
        halfw = 0.0003

    draw_candles(ax_k, day_bars)

    _draw_opening_range(ax_k, day_bars, day)

    if featured_day_bars is not None and not featured_day_bars.empty:
        _draw_mm_targets(ax_k, featured_day_bars)

    if len(day_bars):
        y_low = float(day_bars["low"].min())
        y_high = float(day_bars["high"].max())
        pad = (y_high - y_low) * 0.08 + 1e-6
        ax_k.set_ylim(y_low - pad, y_high + pad)

    if not day_trades.empty:
        dt = day_trades.copy()
        dt["entry_time"] = pd.to_datetime(dt["entry_time"])
        dt["exit_time"] = pd.to_datetime(dt["exit_time"])
        dt = dt.sort_values("exit_time")
        for _, tr in dt.iterrows():
            ent = tr["entry_time"]
            ext = tr["exit_time"]
            side = str(tr["side"])
            ep = float(tr["entry_price"])
            xp = float(tr["exit_price"])
            x_e = mdates.date2num(ent)
            x_x = mdates.date2num(ext)
            reason = str(tr["exit_reason"])
            reason_en = EXIT_REASON_EN.get(reason, reason.replace("_", " "))

            ax_k.plot([x_e, x_x], [ep, xp], color="#555555", linewidth=0.9, linestyle="--", alpha=0.75, zorder=4)
            if side == "long":
                ax_k.scatter([x_e], [ep], marker="^", s=55, c="#1f77b4", zorder=5, edgecolors="k", linewidths=0.4)
                ax_k.annotate("Long entry", (x_e, ep), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
            else:
                ax_k.scatter([x_e], [ep], marker="v", s=55, c="#ff7f0e", zorder=5, edgecolors="k", linewidths=0.4)
                ax_k.annotate("Short entry", (x_e, ep), textcoords="offset points", xytext=(0, -8), fontsize=7, ha="center")

            ax_k.scatter([x_x], [xp], marker="s", s=36, c="#9467bd", zorder=5, edgecolors="k", linewidths=0.3)
            ax_k.annotate(
                f"Exit: {reason_en}",
                (x_x, xp),
                textcoords="offset points",
                xytext=(0, 10 if side == "short" else -12),
                fontsize=7,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.86, edgecolor="#cccccc"),
            )

    ax_k.set_ylabel("Price")
    ax_k.set_title(f"{symbol} {day_date} — intraday candles & trades")
    ax_k.grid(True, linestyle="--", alpha=0.25)
    ax_k.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if len(day_bars):
        t_start = mdates.date2num(day_bars["time_key"].min())
        t_end = mdates.date2num(day_bars["time_key"].max())
    else:
        t_start = mdates.date2num(pd.Timestamp(day_date))
        t_end = t_start + 0.05

    eq_times: list[float] = []
    eq_vals: list[float] = []
    if len(day_bars):
        eq_times.append(mdates.date2num(day_bars["time_key"].iloc[0]))
        eq_vals.append(0.0)

    if not day_trades.empty:
        dt = day_trades.copy()
        dt["exit_time"] = pd.to_datetime(dt["exit_time"])
        dt = dt.sort_values("exit_time")
        cum = 1.0
        for _, tr in dt.iterrows():
            r = float(tr["trade_return"])
            cum *= 1.0 + r
            tx = mdates.date2num(tr["exit_time"])
            eq_times.append(tx)
            eq_vals.append(cum - 1.0)
    elif len(day_bars):
        eq_times.append(mdates.date2num(day_bars["time_key"].iloc[-1]))
        eq_vals.append(0.0)

    if len(eq_times) >= 2:
        ax_eq.step(eq_times, eq_vals, where="post", color="#d62728", linewidth=1.2)
        ax_eq.scatter(eq_times[1:], eq_vals[1:], color="#d62728", s=14, zorder=3)
    else:
        ax_eq.axhline(0.0, color="#d62728", linewidth=1.0)

    ax_eq.axhline(0.0, color="#999999", linewidth=0.6, linestyle="-")
    ax_eq.set_ylabel("Cumulative return (day, from 0)")
    ax_eq.set_xlabel("Time (US/Eastern)")
    ax_eq.grid(True, linestyle="--", alpha=0.25)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    margin = halfw * 3
    ax_k.set_xlim(t_start - margin, t_end + margin)
    ax_eq.set_xlim(t_start - margin, t_end + margin)

    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 trade export + daily review charts")
    p.add_argument(
        "--ranking",
        type=Path,
        default=Path("results") / "refined_ce_zlsma_kama_rule" / "ce_zlsma_kama_rule_refined_optimization.csv",
    )
    p.add_argument("--out", type=Path, default=Path("results") / "refined_ce_zlsma_kama_rule")
    p.add_argument("--lookback-days", type=int, default=252, help="Number of trading days to chart (from data end)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root: Path = args.out
    trades_dir = out_root / "trades"
    charts_root = out_root / "charts"
    trades_dir.mkdir(parents=True, exist_ok=True)
    charts_root.mkdir(parents=True, exist_ok=True)

    params_by_symbol = load_best_params_by_symbol(args.ranking, ("QQQ", "SPY"))
    print("Loaded best params for:", list(params_by_symbol.keys()), flush=True)

    raw_by_symbol: dict[str, pd.DataFrame] = {}
    featured_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, (params, best_row) in params_by_symbol.items():
        print(f"\n{'='*60}\n{symbol}: running backtest with best params...", flush=True)
        featured, result = run_backtest_symbol(symbol, params)
        featured_by_symbol[symbol] = featured
        raw = load_price_data(symbol)
        raw_by_symbol[symbol] = raw
        log = result.trade_log.copy()
        if log.empty:
            print(f"Warning: {symbol} has no trades", flush=True)
        else:
            log["entry_time"] = pd.to_datetime(log["entry_time"])
            log["exit_time"] = pd.to_datetime(log["exit_time"])

        trades_path = trades_dir / f"{symbol.lower()}_trades.csv"
        log.to_csv(trades_path, index=False)
        print(f"Wrote {trades_path} ({len(log)} rows)", flush=True)

        # Equity chart
        eq_path = _create_equity_chart(symbol, best_row, raw, result, charts_root)
        print(f"Wrote equity chart: {eq_path}", flush=True)

        # Daily charts
        days = unique_trading_days_desc(raw, args.lookback_days)
        start_d = days[0].date()
        end_d = days[-1].date()
        print(f"{symbol} chart window: {start_d} .. {end_d} ({len(days)} sessions)", flush=True)

        sym_charts = charts_root / symbol.lower()
        sym_charts.mkdir(parents=True, exist_ok=True)
        raw["date_only"] = raw["time_key"].dt.normalize()
        feat = featured_by_symbol[symbol]
        feat["date_only"] = feat["time_key"].dt.normalize()

        for day in tqdm(days, desc=f"{symbol} charts", unit="day"):
            day_bars = raw.loc[raw["date_only"] == day.normalize()].copy()
            day_trades = log.loc[log["exit_time"].dt.normalize() == day.normalize()].copy() if not log.empty else log.copy()
            feat_day = feat.loc[feat["date_only"] == day.normalize()].copy()
            png = sym_charts / f"{day.strftime('%Y-%m-%d')}.png"
            render_day_chart_fixed(
                symbol=symbol,
                day=day,
                day_bars=day_bars,
                day_trades=day_trades,
                out_path=png,
                featured_day_bars=feat_day,
            )

        # Distribution plot
        if not log.empty:
            try:
                plot_trade_distribution(trades_path, charts_root)
                print(f"Wrote distribution plot for {symbol}", flush=True)
            except Exception as exc:
                print(f"Warning: distribution plot for {symbol} failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

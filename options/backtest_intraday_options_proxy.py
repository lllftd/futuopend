from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.utils import MINUTES_PER_YEAR, calculate_max_drawdown, load_price_data
from core.optimize_ce_zlsma_kama_rule import (
    RuleParams,
    apply_ce_features,
    build_base_features,
    build_entry_signals,
    session_entry_allowed,
)
from core.ranking_utils import load_best_rule_params as load_ranked_rule_params


DEFAULT_SYMBOLS = ("QQQ", "SPY")
DEFAULT_REFINED_RANKING = (
    Path("results") / "refined_ce_zlsma_kama_rule" / "ce_zlsma_kama_rule_refined_optimization.csv"
)
DEFAULT_OUTPUT_DIR = Path("results") / "options_proxy_intraday"


@dataclass(frozen=True)
class ProxyOptionConfig:
    initial_premium: float = 2.0
    leverage: float = 12.0
    theta_per_minute: float = 0.00025
    premium_floor_ratio: float = 0.05
    take_profit_pct: float = 0.25
    stop_loss_pct: float = 0.20


@dataclass
class ProxyBacktestResult:
    summary: dict[str, object]
    equity_curve: pd.Series
    trade_log: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest intraday option proxy execution using underlying signals."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to backtest.")
    parser.add_argument(
        "--ranking-path",
        default=str(DEFAULT_REFINED_RANKING),
        help="CSV ranking used to pick the current best underlying rule per symbol.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--initial-premium",
        type=float,
        default=2.0,
        help="Starting proxy option premium for each new trade.",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=12.0,
        help="Underlying-return multiplier used to approximate option convexity.",
    )
    parser.add_argument(
        "--theta-per-minute",
        type=float,
        default=0.00025,
        help="Time-decay penalty applied to proxy returns each minute.",
    )
    parser.add_argument(
        "--premium-floor-ratio",
        type=float,
        default=0.05,
        help="Minimum premium as a ratio of initial premium to avoid negative prices.",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.25,
        help="Take-profit threshold on proxy premium, expressed as decimal return.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.20,
        help="Stop-loss threshold on proxy premium, expressed as decimal loss.",
    )
    return parser.parse_args()

def load_best_rule_params(ranking_path: Path, symbols: tuple[str, ...]) -> dict[str, RuleParams]:
    try:
        return load_ranked_rule_params(ranking_path, symbols)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Refined ranking file not found: {ranking_path}. Run python -m research.refine_ce_zlsma_kama_rule first."
        ) from exc


def locate_underlying_index(times: pd.Series, timestamp: pd.Timestamp) -> int:
    values = times.to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(timestamp.to_datetime64())
    idx = int(values.searchsorted(target, side="right") - 1)
    if idx < 0:
        idx = 0
    if idx >= len(times):
        idx = len(times) - 1
    return idx


def proxy_price_from_underlying(
    *,
    entry_underlying_price: float,
    observed_underlying_price: float,
    minutes_held: int,
    side_sign: int,
    config: ProxyOptionConfig,
) -> float:
    directional_return = side_sign * (observed_underlying_price / entry_underlying_price - 1.0)
    proxy_return = config.leverage * directional_return - config.theta_per_minute * minutes_held
    floor_price = config.initial_premium * config.premium_floor_ratio
    return max(config.initial_premium * (1.0 + proxy_return), floor_price)


def run_proxy_backtest(
    symbol: str,
    params: RuleParams,
    config: ProxyOptionConfig,
) -> tuple[ProxyBacktestResult, pd.DataFrame]:
    raw = load_price_data(symbol)
    base = build_base_features(raw)
    featured = apply_ce_features(base, params.ce_length, params.ce_multiplier)
    long_signal, short_signal = build_entry_signals(featured, params.trend_mode)
    entry_allowed = session_entry_allowed(featured["time_key"], params.session_filter)
    is_last_bar = featured["time_key"].dt.date.ne(featured["time_key"].dt.date.shift(-1)).fillna(True)

    trade_records: list[dict[str, object]] = []
    exit_events: dict[int, list[float]] = {}
    idx = 0
    progress = tqdm(total=max(len(featured) - 1, 0), desc=f"{symbol} proxy", unit="bar")

    while idx < len(featured) - 1:
        if is_last_bar.iloc[idx] or not entry_allowed.iloc[idx]:
            idx += 1
            progress.update(1)
            continue

        signal_side = None
        side_sign = 0
        proxy_type = ""
        if bool(long_signal.iloc[idx]):
            signal_side = "long"
            side_sign = 1
            proxy_type = "call_proxy"
        elif bool(short_signal.iloc[idx]):
            signal_side = "short"
            side_sign = -1
            proxy_type = "put_proxy"

        if signal_side is None:
            idx += 1
            progress.update(1)
            continue

        entry_idx = idx + 1
        if entry_idx >= len(featured):
            break

        entry_time = featured.iloc[entry_idx]["time_key"]
        entry_underlying_price = float(featured.iloc[entry_idx]["open"])
        entry_proxy_price = config.initial_premium
        take_profit_price = entry_proxy_price * (1.0 + config.take_profit_pct)
        stop_price = entry_proxy_price * (1.0 - config.stop_loss_pct)

        exit_idx = entry_idx
        exit_time = featured.iloc[entry_idx]["time_key"]
        exit_proxy_price = proxy_price_from_underlying(
            entry_underlying_price=entry_underlying_price,
            observed_underlying_price=float(featured.iloc[entry_idx]["close"]),
            minutes_held=0,
            side_sign=side_sign,
            config=config,
        )
        exit_reason = "end_of_day"

        for current_idx in range(entry_idx, len(featured)):
            row = featured.iloc[current_idx]
            minutes_held = max(int((pd.Timestamp(row["time_key"]) - pd.Timestamp(entry_time)).total_seconds() / 60), 0)
            proxy_at_high = proxy_price_from_underlying(
                entry_underlying_price=entry_underlying_price,
                observed_underlying_price=float(row["high"]),
                minutes_held=minutes_held,
                side_sign=side_sign,
                config=config,
            )
            proxy_at_low = proxy_price_from_underlying(
                entry_underlying_price=entry_underlying_price,
                observed_underlying_price=float(row["low"]),
                minutes_held=minutes_held,
                side_sign=side_sign,
                config=config,
            )
            best_proxy_price = max(proxy_at_high, proxy_at_low)
            worst_proxy_price = min(proxy_at_high, proxy_at_low)

            if worst_proxy_price <= stop_price:
                exit_idx = current_idx
                exit_time = pd.Timestamp(row["time_key"])
                exit_proxy_price = stop_price
                exit_reason = "stop_loss"
                break
            if best_proxy_price >= take_profit_price:
                exit_idx = current_idx
                exit_time = pd.Timestamp(row["time_key"])
                exit_proxy_price = take_profit_price
                exit_reason = "take_profit"
                break
            if is_last_bar.iloc[current_idx]:
                exit_idx = current_idx
                exit_time = pd.Timestamp(row["time_key"])
                exit_proxy_price = proxy_price_from_underlying(
                    entry_underlying_price=entry_underlying_price,
                    observed_underlying_price=float(row["close"]),
                    minutes_held=minutes_held,
                    side_sign=side_sign,
                    config=config,
                )
                exit_reason = "end_of_day"
                break

        trade_return = float(exit_proxy_price / entry_proxy_price - 1.0)
        holding_minutes = max(int((exit_time - pd.Timestamp(entry_time)).total_seconds() / 60), 0)
        exit_underlying_price = float(featured.iloc[exit_idx]["close"])

        trade_records.append(
            {
                "symbol": symbol,
                "strategy": params.strategy_name,
                "signal_time": featured.iloc[idx]["time_key"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": signal_side,
                "proxy_type": proxy_type,
                "underlying_signal_price": float(featured.iloc[idx]["close"]),
                "underlying_entry_price": entry_underlying_price,
                "underlying_exit_price": exit_underlying_price,
                "proxy_entry_price": entry_proxy_price,
                "proxy_exit_price": exit_proxy_price,
                "trade_return": trade_return,
                "holding_minutes": holding_minutes,
                "exit_reason": exit_reason,
                "proxy_leverage": config.leverage,
                "theta_per_minute": config.theta_per_minute,
                "entry_underlying_idx": entry_idx,
                "exit_underlying_idx": exit_idx,
            }
        )

        exit_events.setdefault(exit_idx, []).append(trade_return)
        next_idx = max(exit_idx + 1, idx + 1)
        progress.update(max(next_idx - idx, 1))
        idx = next_idx

    progress.close()

    equity = 1.0
    equity_values: list[float] = []
    for row_idx in range(len(featured)):
        for trade_return in exit_events.get(row_idx, []):
            equity *= 1.0 + trade_return
        equity_values.append(equity)

    trade_log = pd.DataFrame(trade_records)
    equity_curve = pd.Series(equity_values, index=featured["time_key"], name="equity_curve")
    period_returns = equity_curve.pct_change().fillna(0.0)
    periods = len(period_returns)
    total_return = float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(period_returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * period_returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float((trade_log["trade_return"] > 0).mean()) if not trade_log.empty else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)
    avg_holding_minutes = float(trade_log["holding_minutes"].mean()) if not trade_log.empty else np.nan
    avg_proxy_entry = float(trade_log["proxy_entry_price"].mean()) if not trade_log.empty else np.nan

    summary = {
        "dataset": symbol,
        "strategy": params.strategy_name,
        "ce_length": params.ce_length,
        "ce_multiplier": params.ce_multiplier,
        "trend_mode": params.trend_mode,
        "tp_atr_multiple": params.tp_atr_multiple,
        "sl_atr_multiple": params.sl_atr_multiple,
        "session_filter": params.session_filter,
        "proxy_initial_premium": config.initial_premium,
        "proxy_leverage": config.leverage,
        "proxy_theta_per_minute": config.theta_per_minute,
        "proxy_take_profit_pct": config.take_profit_pct,
        "proxy_stop_loss_pct": config.stop_loss_pct,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": int(len(trade_log)),
        "max_drawdown": max_drawdown,
        "avg_holding_minutes": avg_holding_minutes,
        "avg_proxy_entry": avg_proxy_entry,
    }
    return ProxyBacktestResult(summary=summary, equity_curve=equity_curve, trade_log=trade_log), featured


def create_proxy_overview_visual(
    symbol: str,
    price_frame: pd.DataFrame,
    result: ProxyBacktestResult,
    output_dir: Path,
) -> Path:
    fig, ax_price = plt.subplots(figsize=(17, 8.5))
    normalized_price = price_frame["close"] / price_frame["close"].iloc[0]
    cumulative_return = result.equity_curve - 1.0

    ax_price.plot(price_frame["time_key"], normalized_price, color="#1F77B4", linewidth=1.0, label="Underlying price")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Normalized Price")
    ax_price.grid(True, linestyle="--", alpha=0.25)

    ax_return = ax_price.twinx()
    ax_return.plot(price_frame["time_key"], cumulative_return, color="#D62728", linewidth=1.15, label="Proxy cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    summary = result.summary
    stats_text = (
        f"Strategy: {summary['strategy']}\n"
        f"CE: length={int(summary['ce_length'])}, mult={summary['ce_multiplier']}\n"
        f"Trend: {summary['trend_mode']}\n"
        f"Session: {summary['session_filter']}\n"
        f"Premium: {summary['proxy_initial_premium']:.2f}\n"
        f"Leverage: {summary['proxy_leverage']:.2f}\n"
        f"Theta/min: {summary['proxy_theta_per_minute']:.5f}\n"
        f"TP/SL: {summary['proxy_take_profit_pct']:.0%}/{summary['proxy_stop_loss_pct']:.0%}\n"
        f"Return: {summary['total_return'] * 100:.2f}%\n"
        f"Sharpe: {summary['sharpe']:.3f}\n"
        f"Win Rate: {summary['win_rate'] * 100:.2f}%\n"
        f"Trades: {int(summary['trade_count'])}\n"
        f"Avg Hold: {summary['avg_holding_minutes']:.1f} min\n"
        f"Max DD: {summary['max_drawdown'] * 100:.2f}%"
    )
    ax_price.text(
        0.015,
        0.985,
        stats_text,
        transform=ax_price.transAxes,
        va="top",
        ha="left",
        fontsize=9.3,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_return.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    output_path = output_dir / f"{symbol.lower()}_intraday_options_proxy_overview.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_proxy_trade_visual(
    symbol: str,
    price_frame: pd.DataFrame,
    result: ProxyBacktestResult,
    output_dir: Path,
) -> Path:
    fig, ax_price = plt.subplots(figsize=(18, 9))
    ax_price.plot(price_frame["time_key"], price_frame["close"], color="#1F77B4", linewidth=0.9, label="Underlying close")
    ax_price.set_title(f"{symbol} Option Proxy Execution With Trade Markers")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Underlying Price")
    ax_price.grid(True, linestyle="--", alpha=0.25)

    trade_log = result.trade_log.copy()
    if not trade_log.empty:
        trade_log["entry_time"] = pd.to_datetime(trade_log["entry_time"])
        trade_log["exit_time"] = pd.to_datetime(trade_log["exit_time"])

        long_entries = trade_log.loc[trade_log["side"] == "long"]
        short_entries = trade_log.loc[trade_log["side"] == "short"]
        ax_price.scatter(
            long_entries["entry_time"],
            long_entries["underlying_entry_price"],
            marker="^",
            s=24,
            color="#2ECC71",
            label="Long signal -> Call proxy",
            alpha=0.8,
        )
        ax_price.scatter(
            short_entries["entry_time"],
            short_entries["underlying_entry_price"],
            marker="v",
            s=24,
            color="#E74C3C",
            label="Short signal -> Put proxy",
            alpha=0.8,
        )
        ax_price.scatter(
            trade_log["exit_time"],
            trade_log["underlying_exit_price"],
            marker="o",
            s=12,
            color="#555555",
            label="Exit",
            alpha=0.45,
        )

    ax_return = ax_price.twinx()
    cumulative_return = result.equity_curve - 1.0
    ax_return.plot(price_frame["time_key"], cumulative_return, color="#D62728", linewidth=1.1, label="Proxy cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_return.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)

    fig.tight_layout()
    output_path = output_dir / f"{symbol.lower()}_intraday_options_proxy_trades.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_results(
    ranking: pd.DataFrame,
    results: dict[str, ProxyBacktestResult],
    price_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    visuals_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    ranking_path = output_dir / "intraday_options_proxy_backtest.csv"
    ranking.to_csv(ranking_path, index=False)
    generated.append(ranking_path)

    for symbol, result in results.items():
        trade_path = output_dir / f"{symbol.lower()}_intraday_options_proxy_trades.csv"
        result.trade_log.to_csv(trade_path, index=False)
        generated.extend(
            [
                trade_path,
                create_proxy_overview_visual(symbol, price_frames[symbol], result, visuals_dir),
                create_proxy_trade_visual(symbol, price_frames[symbol], result, visuals_dir),
            ]
        )

    return generated


def main() -> int:
    args = parse_args()
    symbols = tuple(symbol.upper() for symbol in args.symbols)
    params_by_symbol = load_best_rule_params(Path(args.ranking_path), symbols)
    config = ProxyOptionConfig(
        initial_premium=float(args.initial_premium),
        leverage=float(args.leverage),
        theta_per_minute=float(args.theta_per_minute),
        premium_floor_ratio=float(args.premium_floor_ratio),
        take_profit_pct=float(args.take_profit_pct),
        stop_loss_pct=float(args.stop_loss_pct),
    )

    results: dict[str, ProxyBacktestResult] = {}
    price_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        print(f"Running proxy options backtest for {symbol}...", flush=True)
        result, price_frame = run_proxy_backtest(symbol=symbol, params=params_by_symbol[symbol], config=config)
        results[symbol] = result
        price_frames[symbol] = price_frame

    ranking = pd.DataFrame([result.summary for result in results.values()])
    ranking = ranking.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False])
    generated = export_results(ranking, results, price_frames, Path(args.output_dir))

    print("\nProxy options execution results:", flush=True)
    print(ranking.to_string(index=False), flush=True)
    print("\nGenerated files:", flush=True)
    for path in generated:
        print(path.as_posix(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

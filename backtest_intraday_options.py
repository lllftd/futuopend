from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import MINUTES_PER_YEAR, calculate_max_drawdown, load_price_data
from futu_options_data import FutuOptionDataClient, OptionHistoryRequest
from optimize_ce_zlsma_kama_rule import (
    RuleParams,
    apply_ce_features,
    build_base_features,
    build_entry_signals,
    session_entry_allowed,
)
from options_execution_model import ContractSelectorConfig, select_option_contract_for_signal


DEFAULT_SYMBOLS = ("QQQ", "SPY")
DEFAULT_REFINED_RANKING = (
    Path("results") / "refined_ce_zlsma_kama_rule" / "ce_zlsma_kama_rule_refined_optimization.csv"
)
DEFAULT_OUTPUT_DIR = Path("results") / "options_intraday"


@dataclass(frozen=True)
class OptionRiskConfig:
    take_profit_pct: float = 0.25
    stop_loss_pct: float = 0.20


@dataclass
class OptionsBacktestResult:
    summary: dict[str, object]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    skipped_signals: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map underlying intraday signals to real option execution using Futu OpenD."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to backtest.")
    parser.add_argument(
        "--ranking-path",
        default=str(DEFAULT_REFINED_RANKING),
        help="CSV ranking used to pick the current best underlying rule per symbol.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Futu OpenD host.")
    parser.add_argument("--port", type=int, default=11111, help="Futu OpenD port.")
    parser.add_argument("--max-count", type=int, default=1000, help="Max bars per Futu page request.")
    parser.add_argument(
        "--otm-steps",
        type=int,
        default=1,
        help="Fixed OTM distance in strike steps. 1 means the first OTM strike.",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.25,
        help="Take-profit threshold on option premium, expressed as decimal return.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.20,
        help="Stop-loss threshold on option premium, expressed as decimal loss.",
    )
    parser.add_argument("--refresh-expirations", action="store_true", help="Refresh expiration cache.")
    parser.add_argument("--refresh-chains", action="store_true", help="Refresh chain cache.")
    parser.add_argument("--refresh-bars", action="store_true", help="Refresh option bar cache.")
    parser.add_argument(
        "--max-dte-days",
        type=int,
        default=14,
        help="Maximum number of days-to-expiry allowed when selecting contracts.",
    )
    parser.add_argument(
        "--chain-window-days",
        type=int,
        default=7,
        help="Window size used when scanning historical option chains around each signal date.",
    )
    return parser.parse_args()


def row_to_rule_params(row: pd.Series) -> RuleParams:
    return RuleParams(
        ce_length=int(row["ce_length"]),
        ce_multiplier=float(row["ce_multiplier"]),
        trend_mode=str(row["trend_mode"]),
        tp_atr_multiple=float(row["tp_atr_multiple"]),
        sl_atr_multiple=float(row["sl_atr_multiple"]),
        session_filter=str(row["session_filter"]),
    )


def load_best_rule_params(ranking_path: Path, symbols: tuple[str, ...]) -> dict[str, RuleParams]:
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Refined ranking file not found: {ranking_path}. Run refine_ce_zlsma_kama_rule.py first."
        )

    ranking = pd.read_csv(ranking_path)
    required = {
        "dataset",
        "ce_length",
        "ce_multiplier",
        "trend_mode",
        "tp_atr_multiple",
        "sl_atr_multiple",
        "session_filter",
    }
    missing = required.difference(ranking.columns)
    if missing:
        raise RuntimeError(f"Ranking file is missing required columns: {sorted(missing)}")

    params_by_symbol: dict[str, RuleParams] = {}
    for symbol in symbols:
        subset = ranking.loc[ranking["dataset"].astype(str).str.upper() == symbol.upper()].copy()
        if subset.empty:
            raise RuntimeError(f"No ranking rows found for symbol {symbol} in {ranking_path}.")
        subset = subset.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False])
        params_by_symbol[symbol.upper()] = row_to_rule_params(subset.iloc[0])
    return params_by_symbol


def locate_underlying_index(times: pd.Series, timestamp: pd.Timestamp) -> int:
    values = times.to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(timestamp.to_datetime64())
    idx = int(values.searchsorted(target, side="right") - 1)
    if idx < 0:
        idx = 0
    if idx >= len(times):
        idx = len(times) - 1
    return idx


def _first_bar_at_or_after(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    subset = frame.loc[frame["time_key"] >= timestamp]
    if subset.empty:
        return None
    return subset.iloc[0]


def classify_option_error(prefix: str, error: Exception | str) -> str:
    message = str(error).lower()
    if "too frequent" in message or "30 seconds" in message:
        return f"{prefix}_rate_limited"
    if "returned no option bars" in message:
        return f"{prefix}_no_bars"
    if (
        "option expiration list is empty" in message
        or "no expiries within" in message
        or "option chain lookup failed for window" in message
    ):
        return f"{prefix}_historical_chain_unavailable"
    if "option chain lookup failed for" in message:
        return f"{prefix}_expiry_chain_unavailable"
    return prefix


def simulate_option_trade(
    df: pd.DataFrame,
    symbol: str,
    strategy_name: str,
    signal_idx: int,
    signal_side: str,
    client: FutuOptionDataClient,
    selector_config: ContractSelectorConfig,
    risk_config: OptionRiskConfig,
    refresh_expirations: bool,
    refresh_chains: bool,
    refresh_bars: bool,
    option_bar_cache: dict[tuple[str, str], pd.DataFrame],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    signal_row = df.iloc[signal_idx]
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None, {"reason": "no_next_underlying_bar", "signal_time": signal_row["time_key"], "side": signal_side}

    entry_time = df.iloc[entry_idx]["time_key"]
    trade_date = entry_time.strftime("%Y-%m-%d")
    underlying_signal_price = float(signal_row["close"])
    underlying_entry_price = float(df.iloc[entry_idx]["open"])

    try:
        contract = select_option_contract_for_signal(
            client=client,
            underlying=symbol,
            signal_time=signal_row["time_key"],
            entry_bar_time=entry_time,
            signal_side=signal_side,
            underlying_price=underlying_signal_price,
            config=selector_config,
            refresh_expirations=refresh_expirations,
            refresh_chains=refresh_chains,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime data availability
        return None, {
            "reason": classify_option_error("contract_selection_failed", exc),
            "error": str(exc),
            "signal_time": signal_row["time_key"],
            "side": signal_side,
        }

    cache_key = (contract.option_code, trade_date)
    if cache_key not in option_bar_cache or refresh_bars:
        try:
            option_bar_cache[cache_key] = client.get_option_history_bars(
                OptionHistoryRequest(
                    underlying=contract.underlying,
                    option_code=contract.option_code,
                    expiry=contract.expiry,
                    option_type=contract.option_type,
                    strike_price=contract.strike_price,
                    start=trade_date,
                    end=trade_date,
                ),
                refresh=refresh_bars,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime data availability
            return None, {
                "reason": classify_option_error("option_history_failed", exc),
                "error": str(exc),
                "signal_time": signal_row["time_key"],
                "side": signal_side,
                "option_code": contract.option_code,
            }

    option_bars = option_bar_cache[cache_key].copy()
    option_bars["time_key"] = pd.to_datetime(option_bars["time_key"])
    option_bars = option_bars.sort_values("time_key").reset_index(drop=True)
    entry_bar = _first_bar_at_or_after(option_bars, entry_time)
    if entry_bar is None:
        return None, {
            "reason": "missing_entry_option_bar",
            "signal_time": signal_row["time_key"],
            "entry_time": entry_time,
            "side": signal_side,
            "option_code": contract.option_code,
        }

    entry_price = float(entry_bar["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None, {
            "reason": "invalid_entry_option_price",
            "signal_time": signal_row["time_key"],
            "entry_time": entry_bar["time_key"],
            "side": signal_side,
            "option_code": contract.option_code,
        }

    take_profit = entry_price * (1.0 + risk_config.take_profit_pct)
    stop_price = entry_price * (1.0 - risk_config.stop_loss_pct)
    active_bars = option_bars.loc[option_bars["time_key"] >= entry_bar["time_key"]].reset_index(drop=True)

    exit_row = active_bars.iloc[-1]
    exit_time = pd.Timestamp(exit_row["time_key"])
    exit_price = float(exit_row["close"])
    exit_reason = "end_of_day"

    for _, row in active_bars.iterrows():
        low_price = float(row["low"])
        high_price = float(row["high"])
        if np.isfinite(low_price) and low_price <= stop_price:
            exit_time = pd.Timestamp(row["time_key"])
            exit_price = stop_price
            exit_reason = "stop_loss"
            break
        if np.isfinite(high_price) and high_price >= take_profit:
            exit_time = pd.Timestamp(row["time_key"])
            exit_price = take_profit
            exit_reason = "take_profit"
            break

    exit_idx = locate_underlying_index(df["time_key"], exit_time)
    trade_return = float(exit_price / entry_price - 1.0)
    holding_minutes = max(int((exit_time - pd.Timestamp(entry_bar["time_key"])).total_seconds() / 60), 0)
    underlying_exit_price = float(df.iloc[exit_idx]["close"])

    trade_record = {
        "symbol": symbol,
        "strategy": strategy_name,
        "signal_time": signal_row["time_key"],
        "entry_time": pd.Timestamp(entry_bar["time_key"]),
        "exit_time": exit_time,
        "side": signal_side,
        "underlying_signal_price": underlying_signal_price,
        "underlying_entry_price": underlying_entry_price,
        "underlying_exit_price": underlying_exit_price,
        "option_code": contract.option_code,
        "option_type": contract.option_type,
        "expiry": contract.expiry,
        "strike_price": contract.strike_price,
        "selection_note": contract.selection_note,
        "option_entry_price": entry_price,
        "option_exit_price": exit_price,
        "trade_return": trade_return,
        "holding_minutes": holding_minutes,
        "exit_reason": exit_reason,
        "entry_underlying_idx": entry_idx,
        "exit_underlying_idx": exit_idx,
    }
    return trade_record, None


def run_options_backtest(
    symbol: str,
    params: RuleParams,
    client: FutuOptionDataClient,
    selector_config: ContractSelectorConfig,
    risk_config: OptionRiskConfig,
    refresh_expirations: bool = False,
    refresh_chains: bool = False,
    refresh_bars: bool = False,
) -> tuple[OptionsBacktestResult, pd.DataFrame]:
    raw = load_price_data(symbol)
    base = build_base_features(raw)
    featured = apply_ce_features(base, params.ce_length, params.ce_multiplier)
    long_signal, short_signal = build_entry_signals(featured, params.trend_mode)
    entry_allowed = session_entry_allowed(featured["time_key"], params.session_filter)
    is_last_bar = featured["time_key"].dt.date.ne(featured["time_key"].dt.date.shift(-1)).fillna(True)

    trade_records: list[dict[str, object]] = []
    skipped_records: list[dict[str, object]] = []
    exit_events: dict[int, list[float]] = {}
    option_bar_cache: dict[tuple[str, str], pd.DataFrame] = {}

    idx = 0
    progress = tqdm(total=max(len(featured) - 1, 0), desc=f"{symbol} options", unit="bar")
    while idx < len(featured) - 1:
        if is_last_bar.iloc[idx] or not entry_allowed.iloc[idx]:
            idx += 1
            progress.update(1)
            continue

        signal_side = None
        if bool(long_signal.iloc[idx]):
            signal_side = "long"
        elif bool(short_signal.iloc[idx]):
            signal_side = "short"

        if signal_side is None:
            idx += 1
            progress.update(1)
            continue

        trade_record, skipped = simulate_option_trade(
            df=featured,
            symbol=symbol,
            strategy_name=params.strategy_name,
            signal_idx=idx,
            signal_side=signal_side,
            client=client,
            selector_config=selector_config,
            risk_config=risk_config,
            refresh_expirations=refresh_expirations,
            refresh_chains=refresh_chains,
            refresh_bars=refresh_bars,
            option_bar_cache=option_bar_cache,
        )

        if trade_record is None:
            skipped_records.append(skipped or {"reason": "unknown", "signal_time": featured.iloc[idx]["time_key"]})
            idx += 1
            progress.update(1)
            continue

        trade_records.append(trade_record)
        exit_idx = int(trade_record["exit_underlying_idx"])
        exit_events.setdefault(exit_idx, []).append(float(trade_record["trade_return"]))
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
    skipped_signals = pd.DataFrame(skipped_records)
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
    avg_premium = float(trade_log["option_entry_price"].mean()) if not trade_log.empty else np.nan

    summary = {
        "dataset": symbol,
        "strategy": params.strategy_name,
        "ce_length": params.ce_length,
        "ce_multiplier": params.ce_multiplier,
        "trend_mode": params.trend_mode,
        "tp_atr_multiple": params.tp_atr_multiple,
        "sl_atr_multiple": params.sl_atr_multiple,
        "session_filter": params.session_filter,
        "option_take_profit_pct": risk_config.take_profit_pct,
        "option_stop_loss_pct": risk_config.stop_loss_pct,
        "otm_steps": selector_config.otm_steps,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": int(len(trade_log)),
        "max_drawdown": max_drawdown,
        "avg_holding_minutes": avg_holding_minutes,
        "avg_premium": avg_premium,
        "contract_selection_failures": int(len(skipped_signals)),
    }
    return OptionsBacktestResult(summary=summary, equity_curve=equity_curve, trade_log=trade_log, skipped_signals=skipped_signals), featured


def create_options_visual(
    symbol: str,
    price_frame: pd.DataFrame,
    result: OptionsBacktestResult,
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
    ax_return.plot(price_frame["time_key"], cumulative_return, color="#D62728", linewidth=1.15, label="Option cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    summary = result.summary
    ax_price.set_title(f"{symbol} Intraday Options Execution")
    stats_text = (
        f"Strategy: {summary['strategy']}\n"
        f"CE: length={int(summary['ce_length'])}, mult={summary['ce_multiplier']}\n"
        f"Trend: {summary['trend_mode']}\n"
        f"Session: {summary['session_filter']}\n"
        f"Option TP/SL: {summary['option_take_profit_pct']:.0%}/{summary['option_stop_loss_pct']:.0%}\n"
        f"OTM steps: {int(summary['otm_steps'])}\n"
        f"Return: {summary['total_return'] * 100:.2f}%\n"
        f"Sharpe: {summary['sharpe']:.3f}\n"
        f"Win Rate: {summary['win_rate'] * 100:.2f}%\n"
        f"Trades: {int(summary['trade_count'])}\n"
        f"Avg Premium: {summary['avg_premium']:.2f}\n"
        f"Avg Hold: {summary['avg_holding_minutes']:.1f} min\n"
        f"Skipped: {int(summary['contract_selection_failures'])}\n"
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
    output_path = output_dir / f"{symbol.lower()}_intraday_options_overview.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_option_trade_marker_visual(
    symbol: str,
    price_frame: pd.DataFrame,
    result: OptionsBacktestResult,
    output_dir: Path,
) -> Path:
    fig, ax_price = plt.subplots(figsize=(18, 9))
    ax_price.plot(price_frame["time_key"], price_frame["close"], color="#1F77B4", linewidth=0.9, label="Underlying close")
    ax_price.set_title(f"{symbol} Options Execution With Trade Markers")
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
            label="Long signal -> Call buy",
            alpha=0.8,
        )
        ax_price.scatter(
            short_entries["entry_time"],
            short_entries["underlying_entry_price"],
            marker="v",
            s=24,
            color="#E74C3C",
            label="Short signal -> Put buy",
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
    ax_return.plot(price_frame["time_key"], cumulative_return, color="#D62728", linewidth=1.1, label="Option cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_return.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)

    fig.tight_layout()
    output_path = output_dir / f"{symbol.lower()}_intraday_options_trades.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_results(
    ranking: pd.DataFrame,
    results: dict[str, OptionsBacktestResult],
    price_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    visuals_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    ranking_path = output_dir / "intraday_options_backtest.csv"
    ranking.to_csv(ranking_path, index=False)
    generated.append(ranking_path)

    for symbol, result in results.items():
        trade_path = output_dir / f"{symbol.lower()}_intraday_options_trades.csv"
        skipped_path = output_dir / f"{symbol.lower()}_intraday_options_skipped_signals.csv"
        result.trade_log.to_csv(trade_path, index=False)
        result.skipped_signals.to_csv(skipped_path, index=False)
        generated.extend(
            [
                trade_path,
                skipped_path,
                create_options_visual(symbol, price_frames[symbol], result, visuals_dir),
                create_option_trade_marker_visual(symbol, price_frames[symbol], result, visuals_dir),
            ]
        )

    return generated


def main() -> int:
    args = parse_args()
    symbols = tuple(symbol.upper() for symbol in args.symbols)
    params_by_symbol = load_best_rule_params(Path(args.ranking_path), symbols)
    selector_config = ContractSelectorConfig(
        otm_steps=max(int(args.otm_steps), 1),
        max_dte_days=max(int(args.max_dte_days), 0),
        chain_window_days=max(int(args.chain_window_days), 1),
    )
    risk_config = OptionRiskConfig(
        take_profit_pct=float(args.take_profit_pct),
        stop_loss_pct=float(args.stop_loss_pct),
    )

    results: dict[str, OptionsBacktestResult] = {}
    price_frames: dict[str, pd.DataFrame] = {}
    with FutuOptionDataClient(host=args.host, port=args.port, max_count=args.max_count) as client:
        for symbol in symbols:
            print(f"Running options execution backtest for {symbol}...", flush=True)
            result, price_frame = run_options_backtest(
                symbol=symbol,
                params=params_by_symbol[symbol],
                client=client,
                selector_config=selector_config,
                risk_config=risk_config,
                refresh_expirations=args.refresh_expirations,
                refresh_chains=args.refresh_chains,
                refresh_bars=args.refresh_bars,
            )
            results[symbol] = result
            price_frames[symbol] = price_frame

    ranking = pd.DataFrame([result.summary for result in results.values()])
    ranking = ranking.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False])
    generated = export_results(ranking, results, price_frames, Path(args.output_dir))

    print("\nOptions execution results:", flush=True)
    print(ranking.to_string(index=False), flush=True)
    print("\nGenerated files:", flush=True)
    for path in generated:
        print(path.as_posix(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from archive.backtest_signal_combos import MINUTES_PER_YEAR, calculate_max_drawdown, load_price_data
from core.foundation.indicators import add_chandelier_exit, kama, zlsma


RESULTS_DIR = Path("results")


@dataclass
class StrategyResult:
    summary: dict[str, object]
    equity_curve: pd.Series
    trade_returns: list[float]
    trade_log: pd.DataFrame


def build_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["zlsma"] = zlsma(result, length=50, offset=0, source="hlc3")
    result["kama"] = kama(result, er_length=9, fast_length=2, slow_length=30, source="hlc3")
    result = add_chandelier_exit(result, length=1, multiplier=2.0, use_close=True, prefix="ce")
    result["long_entry_signal"] = (
        result["ce_buy_signal"].fillna(False)
        & (result["close"] > result["zlsma"])
        & (result["close"] > result["kama"])
    )
    result["short_entry_signal"] = (
        result["ce_sell_signal"].fillna(False)
        & (result["close"] < result["zlsma"])
        & (result["close"] < result["kama"])
    )
    return result


def run_intraday_rule_backtest(df: pd.DataFrame, symbol: str) -> StrategyResult:
    session_date = df["time_key"].dt.date
    is_first_bar = session_date.ne(session_date.shift(1)).fillna(True)
    is_last_bar = session_date.ne(session_date.shift(-1)).fillna(True)

    equity = 1.0
    equity_values: list[float] = []
    trade_returns: list[float] = []
    trade_records: list[dict[str, object]] = []

    position = 0
    entry_price = np.nan
    entry_time = None
    stop_price = np.nan
    take_profit = np.nan

    open_prices = df["open"].to_numpy(dtype=float)
    high_prices = df["high"].to_numpy(dtype=float)
    low_prices = df["low"].to_numpy(dtype=float)
    close_prices = df["close"].to_numpy(dtype=float)
    atr_values = df["ce_atr"].to_numpy(dtype=float)

    for idx in range(len(df)):
        row = df.iloc[idx]
        current_time = row["time_key"]

        if is_first_bar.iloc[idx]:
            position = 0
            entry_price = np.nan
            stop_price = np.nan
            take_profit = np.nan
            entry_time = None

        bar_return = 0.0
        exit_reason = None
        exit_price = np.nan

        if position != 0:
            if position == 1:
                if low_prices[idx] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif high_prices[idx] >= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                elif is_last_bar.iloc[idx]:
                    exit_price = close_prices[idx]
                    exit_reason = "end_of_day"
            else:
                if high_prices[idx] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif low_prices[idx] <= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                elif is_last_bar.iloc[idx]:
                    exit_price = close_prices[idx]
                    exit_reason = "end_of_day"

            if exit_reason is not None:
                trade_return = position * (exit_price / entry_price - 1.0)
                bar_return = trade_return
                equity *= 1.0 + trade_return
                trade_returns.append(trade_return)
                trade_records.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "side": "long" if position == 1 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "trade_return": trade_return,
                        "exit_reason": exit_reason,
                    }
                )
                position = 0
                entry_price = np.nan
                stop_price = np.nan
                take_profit = np.nan
                entry_time = None

        if position == 0 and not is_last_bar.iloc[idx]:
            atr_now = atr_values[idx]
            if np.isfinite(atr_now) and atr_now > 0:
                if row["long_entry_signal"]:
                    position = 1
                    entry_price = open_prices[idx + 1] if idx + 1 < len(df) else close_prices[idx]
                    stop_price = entry_price - atr_now
                    take_profit = entry_price + atr_now
                    entry_time = df.iloc[idx + 1]["time_key"] if idx + 1 < len(df) else current_time
                elif row["short_entry_signal"]:
                    position = -1
                    entry_price = open_prices[idx + 1] if idx + 1 < len(df) else close_prices[idx]
                    stop_price = entry_price + atr_now
                    take_profit = entry_price - atr_now
                    entry_time = df.iloc[idx + 1]["time_key"] if idx + 1 < len(df) else current_time

        equity_values.append(equity)

    equity_curve = pd.Series(equity_values, index=df["time_key"], name="equity_curve")
    period_returns = equity_curve.pct_change().fillna(0.0)
    periods = len(period_returns)
    total_return = float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(period_returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * period_returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)

    summary = {
        "dataset": symbol,
        "strategy": "ce_price_above_below_zlsma_kama_atr_exit",
        "description": "CE trigger with price above/below ZLSMA and KAMA, 1x CE ATR take-profit and 1x CE ATR stop-loss, intraday only",
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "max_drawdown": max_drawdown,
    }
    return StrategyResult(
        summary=summary,
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        trade_log=pd.DataFrame(trade_records),
    )


def run_for_symbols() -> tuple[pd.DataFrame, dict[str, StrategyResult]]:
    results: dict[str, StrategyResult] = {}
    for symbol in ("QQQ", "SPY"):
        print(f"Running rule backtest for {symbol}...")
        raw = load_price_data(symbol)
        featured = build_rule_features(raw)
        results[symbol] = run_intraday_rule_backtest(featured, symbol)

    ranking = pd.DataFrame([result.summary for result in results.values()])
    ranking = ranking.sort_values("sharpe", ascending=False)
    return ranking, results


def export_results(ranking: pd.DataFrame, results: dict[str, StrategyResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(RESULTS_DIR / "ce_zlsma_kama_rule_backtest.csv", index=False)
    for symbol, result in results.items():
        result.trade_log.to_csv(RESULTS_DIR / f"{symbol.lower()}_ce_zlsma_kama_rule_trades.csv", index=False)


def main() -> int:
    ranking, results = run_for_symbols()
    export_results(ranking, results)
    print(ranking.to_string(index=False))
    print("\nSaved rule backtest results to results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

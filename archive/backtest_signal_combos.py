from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from archive.smc import add_smc_signals
from core.foundation.indicators import add_chandelier_exit, kama, zlsma
from core.utils import MINUTES_PER_YEAR, calculate_max_drawdown, load_price_data


RESULTS_DIR = Path("results")
SMC_RECENT_WINDOW = 10
TREND_STRATEGIES = (
    "ce_reversal",
    "zlsma_kama_cross",
    "ce_plus_trend",
    "trend_plus_structure",
    "trend_plus_fvg",
    "trend_plus_ob",
    "trend_plus_smc_any",
    "ce_trend_structure",
    "ce_trend_fvg",
    "ce_trend_ob",
    "ce_trend_smc_any",
)


@dataclass
class BacktestResult:
    summary: dict[str, object]
    returns: pd.Series
    trade_returns: list[float]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["zlsma"] = zlsma(result, length=50, offset=0, source="hlc3")
    result["kama"] = kama(result, er_length=9, fast_length=2, slow_length=30, source="hlc3")
    result = add_chandelier_exit(result, length=1, multiplier=2.0, use_close=True, prefix="ce")
    result = add_smc_signals(
        result,
        swing_length=50,
        internal_length=5,
        atr_length=200,
        order_block_filter="atr",
        order_block_mitigation="highlow",
        detect_order_blocks=True,
        detect_fvg=True,
        fvg_auto_threshold=True,
    )
    return result


def bool_recent(series: pd.Series, window: int = SMC_RECENT_WINDOW) -> pd.Series:
    return series.astype(int).rolling(window, min_periods=1).max().astype(bool)


def signal_from_sides(long_side: pd.Series, short_side: pd.Series) -> pd.Series:
    long_side = long_side.fillna(False).astype(bool)
    short_side = short_side.fillna(False).astype(bool)
    signal = long_side.astype(int) - short_side.astype(int)
    signal[(long_side & short_side)] = 0
    return signal.astype(int)


def build_strategy_signals(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, str]]:
    trend_bull = (df["zlsma"] > df["kama"]).fillna(False)
    trend_bear = (df["zlsma"] < df["kama"]).fillna(False)
    trend_cross_up = trend_bull & ~trend_bull.shift(1, fill_value=False)
    trend_cross_down = trend_bear & ~trend_bear.shift(1, fill_value=False)

    bull_structure = (
        df["swing_bullish_bos"]
        | df["swing_bullish_choch"]
        | df["internal_bullish_bos"]
        | df["internal_bullish_choch"]
    )
    bear_structure = (
        df["swing_bearish_bos"]
        | df["swing_bearish_choch"]
        | df["internal_bearish_bos"]
        | df["internal_bearish_choch"]
    )
    bull_ob = df["bullish_order_block_created"].fillna(False)
    bear_ob = df["bearish_order_block_created"].fillna(False)
    bull_fvg = df["bullish_fvg"].fillna(False)
    bear_fvg = df["bearish_fvg"].fillna(False)

    bull_smc_any = bull_structure | bull_ob | bull_fvg
    bear_smc_any = bear_structure | bear_ob | bear_fvg

    bull_structure_recent = bool_recent(bull_structure)
    bear_structure_recent = bool_recent(bear_structure)
    bull_ob_recent = bool_recent(bull_ob)
    bear_ob_recent = bool_recent(bear_ob)
    bull_fvg_recent = bool_recent(bull_fvg)
    bear_fvg_recent = bool_recent(bear_fvg)
    bull_smc_any_recent = bool_recent(bull_smc_any)
    bear_smc_any_recent = bool_recent(bear_smc_any)

    strategies = {
        "ce_reversal": signal_from_sides(df["ce_buy_signal"], df["ce_sell_signal"]),
        "zlsma_kama_cross": signal_from_sides(trend_cross_up, trend_cross_down),
        "smc_structure": signal_from_sides(bull_structure, bear_structure),
        "smc_fvg": signal_from_sides(bull_fvg, bear_fvg),
        "smc_order_block": signal_from_sides(bull_ob, bear_ob),
        "smc_any_event": signal_from_sides(bull_smc_any, bear_smc_any),
        "ce_plus_trend": signal_from_sides(df["ce_buy_signal"] & trend_bull, df["ce_sell_signal"] & trend_bear),
        "ce_plus_structure": signal_from_sides(
            df["ce_buy_signal"] & bull_structure_recent,
            df["ce_sell_signal"] & bear_structure_recent,
        ),
        "ce_plus_fvg": signal_from_sides(df["ce_buy_signal"] & bull_fvg_recent, df["ce_sell_signal"] & bear_fvg_recent),
        "ce_plus_ob": signal_from_sides(df["ce_buy_signal"] & bull_ob_recent, df["ce_sell_signal"] & bear_ob_recent),
        "ce_plus_smc_any": signal_from_sides(
            df["ce_buy_signal"] & bull_smc_any_recent,
            df["ce_sell_signal"] & bear_smc_any_recent,
        ),
        "trend_plus_structure": signal_from_sides(trend_cross_up & bull_structure_recent, trend_cross_down & bear_structure_recent),
        "trend_plus_fvg": signal_from_sides(trend_cross_up & bull_fvg_recent, trend_cross_down & bear_fvg_recent),
        "trend_plus_ob": signal_from_sides(trend_cross_up & bull_ob_recent, trend_cross_down & bear_ob_recent),
        "trend_plus_smc_any": signal_from_sides(
            trend_cross_up & bull_smc_any_recent,
            trend_cross_down & bear_smc_any_recent,
        ),
        "ce_trend_structure": signal_from_sides(
            df["ce_buy_signal"] & trend_bull & bull_structure_recent,
            df["ce_sell_signal"] & trend_bear & bear_structure_recent,
        ),
        "ce_trend_fvg": signal_from_sides(
            df["ce_buy_signal"] & trend_bull & bull_fvg_recent,
            df["ce_sell_signal"] & trend_bear & bear_fvg_recent,
        ),
        "ce_trend_ob": signal_from_sides(
            df["ce_buy_signal"] & trend_bull & bull_ob_recent,
            df["ce_sell_signal"] & trend_bear & bear_ob_recent,
        ),
        "ce_trend_smc_any": signal_from_sides(
            df["ce_buy_signal"] & trend_bull & bull_smc_any_recent,
            df["ce_sell_signal"] & trend_bear & bear_smc_any_recent,
        ),
    }

    descriptions = {
        "ce_reversal": "CE standalone reversal",
        "zlsma_kama_cross": "ZLSMA crosses KAMA",
        "smc_structure": "SMC bullish/bearish BOS or CHoCH",
        "smc_fvg": "SMC bullish/bearish FVG event",
        "smc_order_block": "SMC bullish/bearish order block creation",
        "smc_any_event": "Any bullish/bearish SMC event",
        "ce_plus_trend": "CE gated by ZLSMA >/< KAMA trend state",
        "ce_plus_structure": "CE gated by recent SMC structure event",
        "ce_plus_fvg": "CE gated by recent SMC FVG event",
        "ce_plus_ob": "CE gated by recent SMC order block event",
        "ce_plus_smc_any": "CE gated by any recent SMC event",
        "trend_plus_structure": "ZLSMA/KAMA cross gated by recent SMC structure event",
        "trend_plus_fvg": "ZLSMA/KAMA cross gated by recent SMC FVG event",
        "trend_plus_ob": "ZLSMA/KAMA cross gated by recent SMC order block event",
        "trend_plus_smc_any": "ZLSMA/KAMA cross gated by any recent SMC event",
        "ce_trend_structure": "CE plus trend filter plus recent SMC structure event",
        "ce_trend_fvg": "CE plus trend filter plus recent SMC FVG event",
        "ce_trend_ob": "CE plus trend filter plus recent SMC order block event",
        "ce_trend_smc_any": "CE plus trend filter plus any recent SMC event",
    }
    return strategies, descriptions


def run_strategy_backtest(df: pd.DataFrame, signal: pd.Series, symbol: str, strategy_name: str, description: str) -> BacktestResult:
    if len(df) != len(signal):
        raise ValueError("Signal length must match price data length")

    desired_position = signal.replace(0, np.nan).ffill().shift(1).fillna(0).astype(int)
    open_prices = df["open"].to_numpy(dtype=float)
    close_prices = df["close"].to_numpy(dtype=float)
    times = df["time_key"]

    open_to_open = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        open_to_open[:-1] = desired_position.iloc[:-1].to_numpy(dtype=float) * (
            open_prices[1:] / open_prices[:-1] - 1.0
        )
    open_to_open[-1] = desired_position.iloc[-1] * (close_prices[-1] / open_prices[-1] - 1.0)

    returns = pd.Series(open_to_open, index=times, name="strategy_return")
    equity_curve = (1.0 + returns).cumprod()

    trade_returns: list[float] = []
    current_position = 0
    entry_price = np.nan
    for idx, position in enumerate(desired_position.to_numpy(dtype=int)):
        if position == current_position:
            continue
        if current_position != 0:
            exit_price = open_prices[idx]
            trade_returns.append(current_position * (exit_price / entry_price - 1.0))
        if position != 0:
            entry_price = open_prices[idx]
        current_position = position

    if current_position != 0:
        trade_returns.append(current_position * (close_prices[-1] / entry_price - 1.0))

    total_return = float(equity_curve.iloc[-1] - 1.0)
    periods = len(returns)
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)

    summary = {
        "dataset": symbol,
        "strategy": strategy_name,
        "description": description,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "max_drawdown": max_drawdown,
    }
    return BacktestResult(summary=summary, returns=returns, trade_returns=trade_returns)


def run_intraday_trend_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    strategy_name: str,
    description: str,
) -> BacktestResult:
    if len(df) != len(signal):
        raise ValueError("Signal length must match price data length")

    session_date = df["time_key"].dt.date
    is_last_bar_of_day = session_date.ne(session_date.shift(-1)).fillna(True)

    desired_position = signal.replace(0, np.nan).ffill()
    desired_position[is_last_bar_of_day] = 0
    desired_position = desired_position.shift(1).fillna(0).astype(int)
    desired_position[session_date.ne(session_date.shift(1)).fillna(True)] = 0

    open_prices = df["open"].to_numpy(dtype=float)
    close_prices = df["close"].to_numpy(dtype=float)
    times = df["time_key"]

    intraday_returns = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        next_session_same_day = session_date.eq(session_date.shift(-1)).fillna(False).to_numpy(dtype=bool)
        positions = desired_position.to_numpy(dtype=float)
        intraday_returns[:-1] = np.where(
            next_session_same_day[:-1],
            positions[:-1] * (open_prices[1:] / open_prices[:-1] - 1.0),
            positions[:-1] * (close_prices[:-1] / open_prices[:-1] - 1.0),
        )
    intraday_returns[-1] = desired_position.iloc[-1] * (close_prices[-1] / open_prices[-1] - 1.0)

    returns = pd.Series(intraday_returns, index=times, name="strategy_return")
    equity_curve = (1.0 + returns).cumprod()

    trade_returns: list[float] = []
    current_position = 0
    entry_price = np.nan
    for idx, position in enumerate(desired_position.to_numpy(dtype=int)):
        if current_position == 0 and position != 0:
            current_position = position
            entry_price = open_prices[idx]
        elif current_position != 0 and position != current_position:
            exit_price = open_prices[idx]
            trade_returns.append(current_position * (exit_price / entry_price - 1.0))
            current_position = position
            entry_price = open_prices[idx] if position != 0 else np.nan

        if is_last_bar_of_day.iloc[idx] and current_position != 0:
            exit_price = close_prices[idx]
            trade_returns.append(current_position * (exit_price / entry_price - 1.0))
            current_position = 0
            entry_price = np.nan

    total_return = float(equity_curve.iloc[-1] - 1.0)
    periods = len(returns)
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)

    summary = {
        "dataset": symbol,
        "strategy": strategy_name,
        "description": description,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "max_drawdown": max_drawdown,
        "mode": "intraday_trend",
    }
    return BacktestResult(summary=summary, returns=returns, trade_returns=trade_returns)


def combine_symbol_results(symbol_results: dict[str, BacktestResult], strategy_name: str, description: str) -> BacktestResult:
    merged_returns = pd.concat([result.returns.rename(symbol) for symbol, result in symbol_results.items()], axis=1)
    portfolio_returns = merged_returns.fillna(0.0).mean(axis=1)
    equity_curve = (1.0 + portfolio_returns).cumprod()
    trade_returns = [trade for result in symbol_results.values() for trade in result.trade_returns]

    total_return = float(equity_curve.iloc[-1] - 1.0)
    periods = len(portfolio_returns)
    annual_return = float((equity_curve.iloc[-1] ** (MINUTES_PER_YEAR / periods) - 1.0)) if periods > 0 else 0.0
    volatility = float(portfolio_returns.std(ddof=0))
    sharpe = float(np.sqrt(MINUTES_PER_YEAR) * portfolio_returns.mean() / volatility) if volatility > 0 else np.nan
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else np.nan
    max_drawdown = calculate_max_drawdown(equity_curve)

    summary = {
        "dataset": "COMBINED",
        "strategy": strategy_name,
        "description": description,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": len(trade_returns),
        "max_drawdown": max_drawdown,
    }
    return BacktestResult(summary=summary, returns=portfolio_returns, trade_returns=trade_returns)


def run_batch_backtests() -> tuple[pd.DataFrame, dict[str, dict[str, BacktestResult]]]:
    symbol_frames = {}
    signal_maps = {}
    descriptions = None

    for symbol in ("QQQ", "SPY"):
        print(f"Loading and computing features for {symbol}...")
        raw = load_price_data(symbol)
        featured = build_feature_frame(raw)
        strategies, descriptions = build_strategy_signals(featured)
        symbol_frames[symbol] = featured
        signal_maps[symbol] = strategies

    all_results: list[dict[str, object]] = []
    detailed_results: dict[str, dict[str, BacktestResult]] = {symbol: {} for symbol in ("QQQ", "SPY")}
    combined_results: dict[str, BacktestResult] = {}

    assert descriptions is not None
    strategy_names = list(descriptions.keys())
    for strategy_name in strategy_names:
        description = descriptions[strategy_name]
        per_symbol_results = {}
        for symbol in ("QQQ", "SPY"):
            result = run_strategy_backtest(
                symbol_frames[symbol],
                signal_maps[symbol][strategy_name],
                symbol=symbol,
                strategy_name=strategy_name,
                description=description,
            )
            detailed_results[symbol][strategy_name] = result
            per_symbol_results[symbol] = result
            all_results.append(result.summary)

        combined = combine_symbol_results(per_symbol_results, strategy_name, description)
        combined_results[strategy_name] = combined
        all_results.append(combined.summary)

    ranking = pd.DataFrame(all_results)
    ranking = ranking.sort_values(["dataset", "sharpe", "total_return", "win_rate"], ascending=[True, False, False, False])
    detailed_results["COMBINED"] = combined_results
    return ranking, detailed_results


def run_intraday_trend_batch_backtests() -> tuple[pd.DataFrame, dict[str, dict[str, BacktestResult]], dict[str, pd.DataFrame]]:
    symbol_frames: dict[str, pd.DataFrame] = {}
    signal_maps: dict[str, dict[str, pd.Series]] = {}
    descriptions: dict[str, str] | None = None

    for symbol in ("QQQ", "SPY"):
        print(f"Loading and computing features for {symbol}...")
        raw = load_price_data(symbol)
        featured = build_feature_frame(raw)
        strategies, descriptions = build_strategy_signals(featured)
        symbol_frames[symbol] = featured
        signal_maps[symbol] = {name: strategies[name] for name in TREND_STRATEGIES}

    all_results: list[dict[str, object]] = []
    detailed_results: dict[str, dict[str, BacktestResult]] = {symbol: {} for symbol in ("QQQ", "SPY")}

    assert descriptions is not None
    for symbol in ("QQQ", "SPY"):
        for strategy_name in TREND_STRATEGIES:
            result = run_intraday_trend_backtest(
                symbol_frames[symbol],
                signal_maps[symbol][strategy_name],
                symbol=symbol,
                strategy_name=strategy_name,
                description=descriptions[strategy_name],
            )
            detailed_results[symbol][strategy_name] = result
            all_results.append(result.summary)

    ranking = pd.DataFrame(all_results)
    ranking = ranking.sort_values(["dataset", "sharpe", "total_return", "win_rate"], ascending=[True, False, False, False])
    return ranking, detailed_results, symbol_frames


def export_results(ranking: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(RESULTS_DIR / "strategy_ranking.csv", index=False)
    for dataset in ranking["dataset"].unique():
        subset = ranking.loc[ranking["dataset"] == dataset].copy()
        subset.to_csv(RESULTS_DIR / f"strategy_ranking_{dataset.lower()}.csv", index=False)


def print_best_of_each_metric(ranking: pd.DataFrame) -> None:
    for dataset in ("QQQ", "SPY", "COMBINED"):
        subset = ranking.loc[ranking["dataset"] == dataset].copy()
        if subset.empty:
            continue
        best_return = subset.sort_values("total_return", ascending=False).iloc[0]
        best_sharpe = subset.sort_values("sharpe", ascending=False, na_position="last").iloc[0]
        best_win = subset.sort_values("win_rate", ascending=False, na_position="last").iloc[0]
        print(f"\n=== {dataset} ===")
        print(
            f"Best total return: {best_return['strategy']} | "
            f"return={best_return['total_return']:.4f} | sharpe={best_return['sharpe']:.4f} | "
            f"win_rate={best_return['win_rate']:.4f} | trades={int(best_return['trade_count'])}"
        )
        print(
            f"Best sharpe: {best_sharpe['strategy']} | "
            f"return={best_sharpe['total_return']:.4f} | sharpe={best_sharpe['sharpe']:.4f} | "
            f"win_rate={best_sharpe['win_rate']:.4f} | trades={int(best_sharpe['trade_count'])}"
        )
        print(
            f"Best win rate: {best_win['strategy']} | "
            f"return={best_win['total_return']:.4f} | sharpe={best_win['sharpe']:.4f} | "
            f"win_rate={best_win['win_rate']:.4f} | trades={int(best_win['trade_count'])}"
        )


def main() -> int:
    ranking, _ = run_batch_backtests()
    export_results(ranking)
    print_best_of_each_metric(ranking)
    print("\nSaved ranking files to results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

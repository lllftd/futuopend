from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from backtest_signal_combos import run_intraday_trend_batch_backtests


VISUALS_DIR = Path("results") / "visuals"


def select_best_strategy(ranking: pd.DataFrame, dataset: str) -> pd.Series:
    subset = ranking.loc[ranking["dataset"] == dataset].copy()
    return subset.sort_values(["sharpe", "total_return", "win_rate"], ascending=[False, False, False]).iloc[0]


def create_symbol_chart(
    symbol: str,
    price_frame: pd.DataFrame,
    best_row: pd.Series,
    returns: pd.Series,
) -> Path:
    fig, ax_price = plt.subplots(figsize=(16, 8))

    price_norm = price_frame["close"] / price_frame["close"].iloc[0]
    equity = (1.0 + returns).cumprod()

    ax_price.plot(price_frame["time_key"], price_norm, color="#1F77B4", linewidth=1.2, label=f"{symbol} normalized price")
    ax_price.set_ylabel("Normalized Price")
    ax_price.set_xlabel("Time")
    ax_price.grid(True, linestyle="--", alpha=0.25)

    ax_return = ax_price.twinx()
    ax_return.plot(price_frame["time_key"], equity - 1.0, color="#D62728", linewidth=1.2, label="Strategy cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    title = f"{symbol} Intraday Trend Strategy: {best_row['strategy']}"
    ax_price.set_title(title)

    stats_text = (
        f"Return: {best_row['total_return'] * 100:.2f}%\n"
        f"Sharpe: {best_row['sharpe']:.3f}\n"
        f"Win Rate: {best_row['win_rate'] * 100:.2f}%\n"
        f"Trades: {int(best_row['trade_count'])}\n"
        f"Max DD: {best_row['max_drawdown'] * 100:.2f}%"
    )
    ax_price.text(
        0.015,
        0.98,
        stats_text,
        transform=ax_price.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_return.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    output_path = VISUALS_DIR / f"{symbol.lower()}_intraday_trend_best.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    ranking, detailed_results, price_frames = run_intraday_trend_batch_backtests()

    ranking.to_csv(Path("results") / "intraday_trend_ranking.csv", index=False)

    generated_paths: list[Path] = []
    for symbol in ("QQQ", "SPY"):
        best_row = select_best_strategy(ranking, symbol)
        returns = detailed_results[symbol][best_row["strategy"]].returns
        generated_paths.append(create_symbol_chart(symbol, price_frames[symbol], best_row, returns))

    print("Generated intraday trend charts:")
    for path in generated_paths:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

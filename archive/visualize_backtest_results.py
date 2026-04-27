from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from archive.backtest_signal_combos import run_batch_backtests


VISUALS_DIR = Path("results") / "visuals"
DATASETS = ("QQQ", "SPY", "COMBINED")
TOP_N_EQUITY = 5


def format_percent(series: pd.Series) -> pd.Series:
    return series * 100.0


def wrap_label(text: str, width: int = 28) -> str:
    if len(text) <= width:
        return text
    parts = text.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def create_metric_ranking_figure(ranking: pd.DataFrame, dataset: str) -> Path:
    subset = ranking.loc[ranking["dataset"] == dataset].copy()
    subset["strategy_label"] = subset["strategy"].map(wrap_label)

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    metrics = [
        ("total_return", "Total Return (%)", "#2E86DE", True),
        ("sharpe", "Sharpe Ratio", "#28B463", False),
        ("win_rate", "Win Rate (%)", "#AF7AC5", True),
    ]

    for ax, (column, title, color, as_percent) in zip(axes, metrics):
        data = subset.sort_values(column, ascending=True)
        values = format_percent(data[column]) if as_percent else data[column]
        ax.barh(data["strategy_label"], values, color=color, alpha=0.85)
        ax.set_title(f"{dataset} {title}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        if as_percent:
            ax.set_xlabel(title)
        else:
            ax.set_xlabel(column)

    fig.suptitle(f"{dataset} Strategy Ranking Overview", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = VISUALS_DIR / f"{dataset.lower()}_ranking_overview.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_trade_profile_figure(ranking: pd.DataFrame, dataset: str) -> Path:
    subset = ranking.loc[ranking["dataset"] == dataset].copy()
    subset["strategy_label"] = subset["strategy"].map(wrap_label)
    subset["win_rate_pct"] = subset["win_rate"] * 100.0
    subset["max_drawdown_pct"] = subset["max_drawdown"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].scatter(
        subset["trade_count"],
        subset["win_rate_pct"],
        s=90,
        c=subset["sharpe"],
        cmap="viridis",
        alpha=0.85,
    )
    axes[0].set_title(f"{dataset} Trade Count vs Win Rate")
    axes[0].set_xlabel("Trade Count")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    for _, row in subset.iterrows():
        axes[0].annotate(row["strategy"], (row["trade_count"], row["win_rate_pct"]), fontsize=7, alpha=0.8)

    ordered = subset.sort_values("max_drawdown_pct")
    axes[1].barh(ordered["strategy_label"], ordered["max_drawdown_pct"], color="#E74C3C", alpha=0.85)
    axes[1].set_title(f"{dataset} Max Drawdown by Strategy")
    axes[1].set_xlabel("Max Drawdown (%)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(f"{dataset} Trade Profile", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = VISUALS_DIR / f"{dataset.lower()}_trade_profile.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_equity_curve_figure(
    ranking: pd.DataFrame,
    detailed_results: dict[str, dict[str, object]],
    dataset: str,
    top_n: int = TOP_N_EQUITY,
) -> Path:
    subset = ranking.loc[ranking["dataset"] == dataset].copy()
    top = subset.sort_values(["sharpe", "total_return"], ascending=[False, False]).head(top_n)

    fig, ax = plt.subplots(figsize=(16, 8))
    for _, row in top.iterrows():
        strategy = row["strategy"]
        result = detailed_results[dataset][strategy]
        equity = (1.0 + result.returns).cumprod()
        ax.plot(equity.index, equity.values, linewidth=1.5, label=strategy)

    ax.set_title(f"{dataset} Top {len(top)} Equity Curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity Curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    output_path = VISUALS_DIR / f"{dataset.lower()}_equity_curves_top{len(top)}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_summary_table_figure(ranking: pd.DataFrame) -> Path:
    rows = []
    for dataset in DATASETS:
        subset = ranking.loc[ranking["dataset"] == dataset].copy()
        best_return = subset.sort_values("total_return", ascending=False).iloc[0]
        best_sharpe = subset.sort_values("sharpe", ascending=False, na_position="last").iloc[0]
        best_win = subset.sort_values("win_rate", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "best_return": best_return["strategy"],
                "return_pct": f"{best_return['total_return'] * 100:.2f}%",
                "best_sharpe": best_sharpe["strategy"],
                "sharpe": f"{best_sharpe['sharpe']:.3f}",
                "best_win_rate": best_win["strategy"],
                "win_rate": f"{best_win['win_rate'] * 100:.2f}%",
            }
        )

    summary_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Best Strategies by Dataset", fontsize=15, pad=12)
    fig.tight_layout()
    output_path = VISUALS_DIR / "best_strategy_summary.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    ranking, detailed_results = run_batch_backtests()

    generated_files: list[Path] = []
    generated_files.append(create_summary_table_figure(ranking))
    for dataset in DATASETS:
        generated_files.append(create_metric_ranking_figure(ranking, dataset))
        generated_files.append(create_trade_profile_figure(ranking, dataset))
        generated_files.append(create_equity_curve_figure(ranking, detailed_results, dataset))

    print("Generated visualization files:")
    for path in generated_files:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

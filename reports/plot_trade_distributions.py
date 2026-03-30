from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def build_output_path(csv_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = csv_path.parent / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = csv_path.stem.split("_", 1)[0]
    return output_dir / f"{symbol}_holding_return_relationship.png"


def plot_trade_distribution(csv_path: Path, output_dir: Path | None) -> Path:
    trades = pd.read_csv(csv_path)
    if trades.empty:
        raise ValueError(f"No trades found in {csv_path}")

    symbol = str(trades["symbol"].iloc[0])
    strategy = str(trades["strategy"].iloc[0])
    plot_df = trades.copy()
    plot_df["holding_minutes"] = pd.to_numeric(plot_df["holding_minutes"], errors="coerce")
    plot_df["trade_return_pct"] = pd.to_numeric(plot_df["trade_return"], errors="coerce") * 100.0
    plot_df = plot_df.dropna(subset=["holding_minutes", "trade_return_pct"]).copy()
    plot_df = plot_df.loc[plot_df["holding_minutes"] <= 30].copy()
    if plot_df.empty:
        raise ValueError(f"No <=30 minute trades found in {csv_path}")
    holding = plot_df["holding_minutes"]
    returns = plot_df["trade_return_pct"]

    long_trades = plot_df.loc[plot_df["side"] == "long"]
    short_trades = plot_df.loc[plot_df["side"] == "short"]
    eod_trades = plot_df.loc[plot_df["exit_reason"] == "end_of_day"]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(
        long_trades["holding_minutes"],
        long_trades["trade_return_pct"],
        s=20,
        alpha=0.42,
        color="#2CA02C",
        label="Long",
    )
    ax.scatter(
        short_trades["holding_minutes"],
        short_trades["trade_return_pct"],
        s=20,
        alpha=0.42,
        color="#D62728",
        label="Short",
    )
    if not eod_trades.empty:
        ax.scatter(
            eod_trades["holding_minutes"],
            eod_trades["trade_return_pct"],
            s=36,
            facecolors="none",
            edgecolors="#9467BD",
            linewidths=1.0,
            alpha=0.8,
            label="End of Day",
        )

    ax.axhline(0.0, color="#555555", linestyle="-", linewidth=1.0)
    ax.set_title(f"{symbol} Holding Time vs Return")
    ax.set_xlabel("Holding Minutes")
    ax.set_ylabel("Trade Return (%)")
    ax.grid(True, linestyle="--", alpha=0.25)

    x_upper = 30.5
    y_min = float(returns.min())
    y_max = float(returns.max())
    y_span = max(y_max - y_min, 0.01)
    y_top = y_max + 0.12 * y_span
    y_bottom = y_min - 0.05 * y_span

    bands = [
        ("XS 0-10m", (holding >= 0) & (holding <= 10), 0, 10, "#E8F1FB"),
        ("S 11-30m", (holding >= 11) & (holding <= 30), 11, 30, "#EAF7EA"),
    ]

    for label, mask, left, right, color in bands:
        ax.axvspan(left, right, color=color, alpha=0.22, ec="#BBBBBB", lw=0.8)
        subset = plot_df.loc[mask]
        if subset.empty:
            stats = f"{label}\nN=0"
        else:
            stats = (
                f"{label}\n"
                f"N={len(subset)}\n"
                f"W={subset['trade_return_pct'].gt(0).mean() * 100:.1f}%\n"
                f"R={subset['trade_return_pct'].mean():.3f}%\n"
                f"H={subset['holding_minutes'].mean():.1f}m"
            )
        x_mid = left + (right - left) / 2.0
        band_width = right - left
        font_size = 6.6 if band_width <= 12 else 7.2 if band_width <= 24 else 8.2
        ax.text(
            x_mid,
            y_top,
            stats,
            ha="center",
            va="top",
            fontsize=font_size,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
        )

    ax.set_xlim(-0.5, x_upper)
    ax.set_ylim(y_bottom, y_top + 0.02 * y_span)
    ax.legend(loc="lower right")

    stats_text = (
        f"N={len(plot_df)}\n"
        f"W={(returns.gt(0).mean() * 100):.1f}%\n"
        f"H={holding.mean():.1f}m\n"
        f"R={returns.mean():.3f}%"
    )
    fig.text(
        0.015,
        0.98,
        stats_text,
        va="top",
        ha="left",
        fontsize=8.4,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )

    fig.suptitle(f"{symbol} Holding Time / Return Relationship", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = build_output_path(csv_path, output_dir)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot holding-time and return distributions from trade logs.")
    parser.add_argument("csv_paths", nargs="+", help="Trade CSV files to visualize.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save generated plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = [plot_trade_distribution(Path(csv_path), args.output_dir) for csv_path in args.csv_paths]
    for path in generated:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

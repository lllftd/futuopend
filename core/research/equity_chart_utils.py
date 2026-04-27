from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from core.research.optimize_ce_zlsma_kama_rule import OptimizationResult


def create_equity_chart(
    symbol: str,
    best_row: pd.Series,
    price_frame: pd.DataFrame,
    result: OptimizationResult,
    output_dir: Path,
) -> Path:
    fig, ax_price = plt.subplots(figsize=(16, 8))
    normalized_price = price_frame["close"] / price_frame["close"].iloc[0]
    cumulative_return = result.equity_curve - 1.0

    ax_price.plot(price_frame["time_key"], normalized_price, color="#1F77B4", linewidth=1.1, label="Normalized price")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Normalized Price")
    ax_price.grid(True, linestyle="--", alpha=0.25)

    ax_return = ax_price.twinx()
    ax_return.plot(price_frame["time_key"], cumulative_return, color="#D62728", linewidth=1.2, label="Strategy cumulative return")
    ax_return.set_ylabel("Cumulative Return")

    ax_price.set_title(str(best_row["strategy"]), fontsize=9)

    def _g(key: str, default: object = "") -> object:
        value = best_row.get(key, default)
        return default if pd.isna(value) else value

    mag_r = _g("mag_bar_exit_rate", 0)
    eg_r = _g("exhaustion_gap_exit_rate", 0)

    stats = (
        f"{symbol}\n"
        f"CE {int(best_row['ce_length'])} x {best_row['ce_multiplier']}\n"
        f"Trend: {best_row['trend_mode']}\n"
        f"Z/K: {int(best_row['zlsma_length'])} | "
        f"{int(best_row['kama_er_length'])}/{int(best_row['kama_fast_length'])}/{int(best_row['kama_slow_length'])}\n"
        f"Exit: {best_row['exit_model']}\n"
        f"TP ATR/Pct: {best_row['tp_atr_multiple']}/{best_row['tp_pct']*100:.2f}%\n"
        f"SL ATR/Pct: {best_row['sl_atr_multiple']}/{best_row['sl_pct']*100:.2f}%\n"
        f"Session: {best_row['session_filter']}\n"
        f"Time stop: {int(best_row['time_stop_minutes'])}m\n"
        f"--- PA Rules ---\n"
        f"OR Filter: {_g('pa_or_filter')}\n"
        f"MM Target: {_g('pa_use_mm_target')}\n"
        f"Regime Filter: {_g('pa_regime_filter')}\n"
        f"MAG+EG Exit: {_g('pa_mag_bar_exit')}\n"
        f"PA Stops: {_g('pa_use_pa_stops')}\n"
        f"Signal Bar: {_g('pa_require_signal_bar')}\n"
        f"Sizing: {_g('pa_position_sizing_mode')}\n"
        f"--- Performance ---\n"
        f"Return: {best_row['total_return']*100:.2f}%\n"
        f"Sharpe: {best_row['sharpe']:.3f}\n"
        f"Win: {best_row['win_rate']*100:.2f}%\n"
        f"Trades: {int(best_row['trade_count'])}\n"
        f"TP/SL/TS: {best_row['take_profit_rate']*100:.1f}%/"
        f"{best_row['stop_loss_rate']*100:.1f}%/{best_row['time_stop_rate']*100:.1f}%\n"
        f"MAG/EG: {float(mag_r)*100:.1f}%/{float(eg_r)*100:.1f}%\n"
        f"Hold: {best_row['avg_holding_minutes']:.1f}/{best_row['max_holding_minutes']:.0f}m\n"
        f"Max DD: {best_row['max_drawdown']*100:.2f}%"
    )
    fig.text(
        0.82,
        0.90,
        stats,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#CCCCCC"},
    )

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_return.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    out = output_dir / f"{symbol.lower()}_v3_price_vs_return.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out

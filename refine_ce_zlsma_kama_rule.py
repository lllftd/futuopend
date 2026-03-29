from __future__ import annotations

import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from itertools import product

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from optimize_ce_zlsma_kama_rule import (
    RuleParams,
    OptimizationResult,
    apply_ce_features,
    build_base_features,
    build_entry_signals,
    build_exit_prices,
    session_entry_allowed,
    run_intraday_rule,
)
from utils import load_price_data

# ---------------------------------------------------------------------------
# Output directory — v3
# ---------------------------------------------------------------------------
V3_DIR = Path("results/20260329v3")

# ---------------------------------------------------------------------------
# Fixed best base params (from prior rounds)
# ---------------------------------------------------------------------------
ZLSMA_SLOPE = 2e-4

# ---------------------------------------------------------------------------
# Expanded search grids
# ---------------------------------------------------------------------------
SPY_TP_ATR_MULTIPLES = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
QQQ_TP_PCTS = (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008)

SPY_SL_PCT = 0.002
QQQ_SL_PCT = 0.001

TIME_STOPS = (10, 15, 30)

# ---------------------------------------------------------------------------
# PA parameter search — 6 binary dimensions = 64 combos
# ---------------------------------------------------------------------------
PA_OR_FILTER = (True, False)
PA_USE_MM_TARGET = (True, False)
PA_REGIME_FILTER = (True, False)
PA_MAG_EG_EXIT = (True, False)
PA_USE_PA_STOPS = (True, False)
PA_REQUIRE_SIGNAL_BAR = (True, False)

MAX_WORKERS = min(os.cpu_count() - 2, 24) if os.cpu_count() and os.cpu_count() > 4 else 4


def _base_params(
    time_stop_minutes: int,
    pa_or_filter: bool,
    pa_use_mm_target: bool,
    pa_regime_filter: bool,
    pa_mag_eg: bool,
    pa_use_pa_stops: bool,
    pa_require_signal_bar: bool,
) -> dict:
    return dict(
        trend_mode="price_above_both",
        session_filter="before_1230",
        confirmation_mode="none",
        zlsma_slope_threshold=ZLSMA_SLOPE,
        atr_percentile_lookback=60,
        atr_percentile_min=0.0,
        pseudo_cvd_method="clv_body_volume",
        cvd_lookback=20,
        cvd_slope_lookback=3,
        cvd_classic_divergence=True,
        cvd_slope_divergence=True,
        time_stop_minutes=time_stop_minutes,
        force_time_stop=True,
        time_progress_threshold=0.5,
        profit_lock_trigger_pct=0.0,
        profit_lock_fraction=0.0,
        pa_or_filter=pa_or_filter,
        pa_or_wide_tp_scale=0.7,
        pa_require_signal_bar=pa_require_signal_bar,
        pa_require_h2_l2=False,
        pa_pressure_min=0.0,
        pa_use_mm_target=pa_use_mm_target,
        pa_use_pa_stops=pa_use_pa_stops,
        pa_mag_bar_exit=pa_mag_eg,
        pa_exhaustion_gap_exit=pa_mag_eg,
        pa_regime_filter=pa_regime_filter,
        pa_20bar_neutral=False,
        pa_ii_breakout_entry=False,
        pa_position_sizing_mode="risk_based" if pa_use_pa_stops else "fixed",
    )


def refined_parameter_grid(symbol: str) -> list[RuleParams]:
    pa_combos = list(product(
        PA_OR_FILTER,
        PA_USE_MM_TARGET,
        PA_REGIME_FILTER,
        PA_MAG_EG_EXIT,
        PA_USE_PA_STOPS,
        PA_REQUIRE_SIGNAL_BAR,
    ))

    if symbol == "QQQ":
        return [
            RuleParams(
                ce_length=2,
                ce_multiplier=2.5,
                exit_model="pct",
                tp_atr_multiple=0.0,
                sl_atr_multiple=0.0,
                tp_pct=tp_pct,
                sl_pct=QQQ_SL_PCT,
                zlsma_length=45,
                zlsma_offset=0,
                kama_er_length=11,
                kama_fast_length=2,
                kama_slow_length=30,
                **_base_params(ts, pa_or, pa_mm, pa_rf, pa_mag, pa_ps, pa_sb),
            )
            for tp_pct, ts, (pa_or, pa_mm, pa_rf, pa_mag, pa_ps, pa_sb) in product(
                QQQ_TP_PCTS, TIME_STOPS, pa_combos,
            )
        ]

    if symbol == "SPY":
        return [
            RuleParams(
                ce_length=1,
                ce_multiplier=2.0,
                exit_model="atr_tp_pct_sl",
                tp_atr_multiple=tp_atr,
                sl_atr_multiple=0.0,
                tp_pct=0.0,
                sl_pct=SPY_SL_PCT,
                zlsma_length=40,
                zlsma_offset=0,
                kama_er_length=11,
                kama_fast_length=2,
                kama_slow_length=40,
                **_base_params(ts, pa_or, pa_mm, pa_rf, pa_mag, pa_ps, pa_sb),
            )
            for tp_atr, ts, (pa_or, pa_mm, pa_rf, pa_mag, pa_ps, pa_sb) in product(
                SPY_TP_ATR_MULTIPLES, TIME_STOPS, pa_combos,
            )
        ]

    raise ValueError(f"Unsupported symbol: {symbol}")


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------
_WORKER_DF: pd.DataFrame | None = None
_WORKER_SYMBOL: str = ""


def _init_worker(pkl_path: str, symbol: str) -> None:
    global _WORKER_DF, _WORKER_SYMBOL
    with open(pkl_path, "rb") as f:
        _WORKER_DF = pickle.load(f)
    _WORKER_SYMBOL = symbol


def _run_one(params: RuleParams) -> dict:
    result = run_intraday_rule(_WORKER_DF, _WORKER_SYMBOL, params)
    return result.summary


# ---------------------------------------------------------------------------
# Equity chart
# ---------------------------------------------------------------------------

def _create_equity_chart(
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

    def _g(key, default=""):
        v = best_row.get(key, default)
        return default if pd.isna(v) else v

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
        0.82, 0.90, stats, va="top", ha="left", fontsize=8.5, family="monospace",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_refined_optimization() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for symbol in ("QQQ", "SPY"):
        print(f"\nLoading features for {symbol}...", flush=True)
        raw = load_price_data(symbol)
        frames[symbol] = raw

        base = build_base_features(
            raw,
            zlsma_length=45 if symbol == "QQQ" else 40,
            zlsma_offset=0,
            kama_er_length=11,
            kama_fast_length=2,
            kama_slow_length=30 if symbol == "QQQ" else 40,
            atr_percentile_lookback=60,
            pseudo_cvd_method="clv_body_volume",
            cvd_lookback=20,
            cvd_slope_lookback=3,
        )
        ce_len = 2 if symbol == "QQQ" else 1
        ce_mult = 2.5 if symbol == "QQQ" else 2.0
        featured = apply_ce_features(base, ce_len, ce_mult)
        print(f"  Features computed: {len(featured)} rows", flush=True)

        pkl_path = tempfile.mktemp(suffix=f"_{symbol}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(featured, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"  Cached to {pkl_path} ({pkl_size_mb:.0f} MB)", flush=True)

        params_list = refined_parameter_grid(symbol)
        print(f"  Grid: {len(params_list)} param sets, {MAX_WORKERS} workers", flush=True)

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(pkl_path, symbol),
        ) as executor:
            futures = {executor.submit(_run_one, p): p for p in params_list}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{symbol} parallel",
                unit="set",
            ):
                summaries.append(future.result())

        try:
            os.remove(pkl_path)
        except OSError:
            pass

    ranking = pd.DataFrame(summaries)
    ranking = ranking.sort_values(
        ["dataset", "sharpe", "total_return", "win_rate"],
        ascending=[True, False, False, False],
    )
    return ranking, frames


def export_refined_outputs(
    ranking: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> list[Path]:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    trades_dir = V3_DIR / "trades"
    charts_dir = V3_DIR / "charts"
    trades_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    ranking_path = V3_DIR / "ce_zlsma_kama_rule_refined_optimization.csv"
    ranking.to_csv(ranking_path, index=False)
    generated.append(ranking_path)
    print(f"\nWrote ranking: {ranking_path} ({len(ranking)} rows)", flush=True)

    from export_v3_daily_review import ranking_row_to_rule_params

    for symbol in ("QQQ", "SPY"):
        best_row = ranking.loc[ranking["dataset"] == symbol].iloc[0]
        params = ranking_row_to_rule_params(best_row)
        print(f"\n{symbol} best: sharpe={best_row['sharpe']:.4f} return={best_row['total_return']*100:.2f}%", flush=True)

        raw = frames[symbol]
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
        result = run_intraday_rule(featured, symbol, params)

        trades_path = trades_dir / f"{symbol.lower()}_trades.csv"
        result.trade_log.to_csv(trades_path, index=False)
        generated.append(trades_path)
        print(f"Wrote {trades_path} ({len(result.trade_log)} rows)", flush=True)

        eq_path = _create_equity_chart(symbol, best_row, raw, result, charts_dir)
        generated.append(eq_path)
        print(f"Wrote equity chart: {eq_path}", flush=True)

    return generated


def main() -> int:
    ranking, frames = run_refined_optimization()
    generated = export_refined_outputs(ranking, frames)

    print("\n" + "=" * 70, flush=True)
    print("BEST STRATEGIES:", flush=True)
    for symbol in ("QQQ", "SPY"):
        best = ranking.loc[ranking["dataset"] == symbol].iloc[0]
        print(
            f"\n{symbol}: sharpe={best['sharpe']:.4f} | return={best['total_return']*100:.2f}% | "
            f"win={best['win_rate']*100:.1f}% | trades={int(best['trade_count'])} | "
            f"dd={best['max_drawdown']*100:.2f}% | ts={int(best['time_stop_minutes'])}m\n"
            f"  TP: atr={best['tp_atr_multiple']} pct={best['tp_pct']*100:.3f}% | "
            f"SL: pct={best['sl_pct']*100:.3f}%\n"
            f"  PA: OR={best.get('pa_or_filter','')} MM={best.get('pa_use_mm_target','')} "
            f"RF={best.get('pa_regime_filter','')} MAG={best.get('pa_mag_bar_exit','')} "
            f"PS={best.get('pa_use_pa_stops','')} SB={best.get('pa_require_signal_bar','')}",
            flush=True,
        )

    print(f"\nGenerated {len(generated)} files:", flush=True)
    for p in generated:
        print(f"  {p.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

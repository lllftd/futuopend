from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.equity_chart_utils import create_equity_chart as _create_equity_chart
from core.optimize_ce_zlsma_kama_rule import (
    RuleParams,
    OptimizationResult,
    apply_ce_features,
    build_base_features,
    build_entry_signals,
    build_exit_prices,
    session_entry_allowed,
    run_intraday_rule,
)
from core.ranking_utils import ranking_row_to_rule_params
from core.utils import load_price_data

# ---------------------------------------------------------------------------
# Output directory — v3
# ---------------------------------------------------------------------------
V3_DIR = Path("results") / "refined_ce_zlsma_kama_rule"

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


def _refine_base_params(zlsma_slope_threshold: float) -> dict[str, object]:
    params = _base_params(
        time_stop_minutes=30,
        pa_or_filter=False,
        pa_use_mm_target=False,
        pa_regime_filter=False,
        pa_mag_eg=False,
        pa_use_pa_stops=False,
        pa_require_signal_bar=False,
    )
    params["zlsma_slope_threshold"] = zlsma_slope_threshold
    return params


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the refined CE/ZLSMA/KAMA search and export the top-ranked artifacts."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
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

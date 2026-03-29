"""Grid search Chandelier Exit (ce_length, ce_multiplier) only; all other params fixed to current RR/CVD refine best."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from tqdm import tqdm

from optimize_ce_zlsma_kama_rule import (
    CE_LENGTHS,
    CE_MULTIPLIERS,
    RESULTS_DIR,
    RuleParams,
    apply_ce_features,
    build_base_features,
    load_price_data,
    run_intraday_rule,
)
from refine_ce_zlsma_kama_rule import _refine_base_params

OUTPUT_DIR = RESULTS_DIR / "ce_param_grid"

# Extend multipliers beyond global optimize (1.5, 2.0) to cover refine defaults (2.5 / 2.75) and nearby values.
CE_MULTIPLIERS_GRID = tuple(sorted(set(CE_MULTIPLIERS + (2.25, 2.5, 2.75, 3.0))))

# Fixed to 20260329_rr_cvd best-ranked combo per symbol (tp/sl/slope/zlsma/kama).
FIXED_ZLSMA_SLOPE = 5e-4


def _qqq_params(ce_length: int, ce_multiplier: float) -> RuleParams:
    return RuleParams(
        ce_length=ce_length,
        ce_multiplier=ce_multiplier,
        exit_model="pct",
        tp_atr_multiple=0.0,
        sl_atr_multiple=0.0,
        tp_pct=0.0015,
        sl_pct=0.001,
        zlsma_length=45,
        zlsma_offset=0,
        kama_er_length=11,
        kama_fast_length=2,
        kama_slow_length=30,
        **_refine_base_params(FIXED_ZLSMA_SLOPE),
    )


def _spy_params(ce_length: int, ce_multiplier: float) -> RuleParams:
    return RuleParams(
        ce_length=ce_length,
        ce_multiplier=ce_multiplier,
        exit_model="atr_tp_pct_sl",
        tp_atr_multiple=0.5,
        sl_atr_multiple=0.0,
        tp_pct=0.0,
        sl_pct=0.002,
        zlsma_length=40,
        zlsma_offset=0,
        kama_er_length=11,
        kama_fast_length=2,
        kama_slow_length=40,
        **_refine_base_params(FIXED_ZLSMA_SLOPE),
    )


def ce_parameter_grid(symbol: str) -> list[RuleParams]:
    builder = _qqq_params if symbol == "QQQ" else _spy_params
    return [builder(L, M) for L, M in product(CE_LENGTHS, CE_MULTIPLIERS_GRID)]


def run_ce_grid() -> pd.DataFrame:
    summaries: list[dict[str, object]] = []

    for symbol in ("QQQ", "SPY"):
        print(f"CE grid: loading features for {symbol}...", flush=True)
        raw = load_price_data(symbol)
        feature_key = (
            45 if symbol == "QQQ" else 40,
            0,
            11,
            2,
            30 if symbol == "QQQ" else 40,
            60,
            "clv_body_volume",
            20,
            3,
        )
        base = build_base_features(
            raw,
            zlsma_length=feature_key[0],
            zlsma_offset=0,
            kama_er_length=11,
            kama_fast_length=2,
            kama_slow_length=feature_key[4],
            atr_percentile_lookback=60,
            pseudo_cvd_method="clv_body_volume",
            cvd_lookback=20,
            cvd_slope_lookback=3,
        )
        ce_cache: dict[tuple[int, float], pd.DataFrame] = {}

        for params in tqdm(ce_parameter_grid(symbol), desc=f"{symbol} CE grid", unit="ce"):
            ck = (params.ce_length, params.ce_multiplier)
            if ck not in ce_cache:
                ce_cache[ck] = apply_ce_features(base, params.ce_length, params.ce_multiplier)
            result = run_intraday_rule(ce_cache[ck], symbol, params)
            summaries.append(result.summary)

    ranking = pd.DataFrame(summaries)
    ranking = ranking.sort_values(["dataset", "sharpe", "total_return", "win_rate"], ascending=[True, False, False, False])
    return ranking


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "ce_grid_optimization.csv"

    print(
        f"CE lengths {CE_LENGTHS}, multipliers {CE_MULTIPLIERS_GRID} "
        f"({len(CE_LENGTHS) * len(CE_MULTIPLIERS_GRID)} combos per symbol)\n",
        flush=True,
    )

    ranking = run_ce_grid()
    ranking.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.as_posix()}", flush=True)

    print("\nBest CE per symbol (by sharpe, total_return, win_rate):", flush=True)
    for symbol in ("QQQ", "SPY"):
        sub = ranking.loc[ranking["dataset"] == symbol]
        best = sub.iloc[0]
        print(
            f"  {symbol}: ce_length={int(best['ce_length'])} ce_multiplier={best['ce_multiplier']} | "
            f"return={best['total_return']:.4f} sharpe={best['sharpe']:.4f} "
            f"win_rate={best['win_rate']:.4f} trades={int(best['trade_count'])}",
            flush=True,
        )
        print(f"    strategy: {best['strategy']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

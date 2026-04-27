# Project Structure

This repository mixes active training/backtest/live code with historical research
scripts. Treat model-facing code as result sensitive: refactors are fine only when
they preserve feature columns, row alignment, labels, thresholds, calibration, and
simulation math.

## Active Entry Points

- `run_oos_backtest.py`: thin wrapper around `backtests.oos_backtest.main`.
- `run_in_sample_backtest.py`: in-sample wrapper around the same OOS engine with
  environment defaults.
- `backtests/train_pipeline.py`: full training orchestration.
- `backtests/train_layer1a_only.py`, `backtests/train_layer1b_only.py`: layer-only
  training entry points.
- `scripts/walkforward/run_walkforward.py`, `scripts/oos/replot_oos_charts.py`,
  `scripts/walkforward/*.py`: walk-forward and chart diagnostics.
- `live/monitor.py`: live Futu monitor for the V3 rule strategy.
- `live/pa_monitor.py`: PA-specific monitor subclass used by the live monitor.

## Directory Classes

| Path | Class | Notes |
| --- | --- | --- |
| `core/` | Shared library | Indicators, PA rules, V3 rule engine, configuration, model state classes, and trainers. |
| `core/training/` | Model-sensitive training stack | L1a/L1b/L2/L3/unified training, labels, feature registry, simulations, calibration, and threshold metadata. |
| `core/utils/` | Shared utilities | Data loading, drawdown helpers, session boundaries, and signal expansion. Keep utilities behavior-preserving. |
| `backtests/` | Model-sensitive backtest and orchestration | OOS engine, training launchers, diagnostics, and figures. Avoid changing lazy import order in `oos_backtest.py`. |
| `live/` | Runtime monitoring | Live rule execution and signal/position logging. |
| `scripts/` | Operational scripts | Walk-forward, plotting, and diagnostics. |
| `tests/` | Automated tests | Currently narrow; expand with parity tests before numeric refactors. |
| `archive/` | Historical research | Old experiments and data tools. Not on the active stack unless run manually. |
| `docs/` | Documentation | Setup, PA notes, and this structure map. |

## Data Paths

The default data directory is repo-root based. `core.utils.DATA_DIR`,
`core.training.common.constants.DATA_DIR`, and `backtests.oos_backtest.DATA_DIR` should
all resolve to `data/` under the repository root when `DATA_DIR` is unset. Set
the `DATA_DIR` environment variable to point all path users at an external data
bundle.

## Dependency Flow

```text
run_*_backtest.py
  -> backtests.oos_backtest
     -> core.training.*, core.foundation.pa_rules, core.foundation.indicators

backtests/train_pipeline.py
  -> core.training.prep.data_prep / tcn_data_prep
  -> core.training.l1a / l1b / l2 / unified

live/monitor.py
  -> core.config.v3
  -> core.research.optimize_ce_zlsma_kama_rule
  -> core.utils
```

`archive/` scripts may import active `core` modules, but active `core`,
`backtests`, and `live` code should not depend on `archive`.

## Model-Result Sensitive Zones

Do not change these without golden/parity checks:

- Feature engineering and column contracts: `core/foundation/pa_rules.py`,
  `core/training/prep/data_prep.py`, `core/training/prep/tcn_data_prep.py`,
  `core/training/prep/feature_registry.py`, `core/training/unified/features.py`,
  `core/training/unified/position_features.py`.
- Labels and split definitions: `core/training/labels/vol_regime_labels.py`,
  `core/training/labels/straddle_edge_labels.py`, `core/training/common/stack_v2_common.py`,
  `core/training/common/constants.py`.
- Training and inference math: `core/training/l1a/`, `core/training/l1b/`,
  `core/training/l2/`, `core/training/unified/`, `core/models/tcn_pa_state.py`,
  `core/models/mamba_pa_state.py`.
- Calibration and simulation: `core/training/l2/calibration.py`,
  `core/training/unified/simulation/`.
- OOS execution: `backtests/oos_backtest.py`.
- V3 live rule path: `core/research/optimize_ce_zlsma_kama_rule.py`, `core/config/v3.py`,
  `live/monitor.py`.

## Safe Cleanup Rules

- Safe by default: unused imports, comments/docs, import-path fixes for broken
  archive scripts, and utility deduplication that keeps identical inputs and
  outputs.
- Needs tests: path resolution changes, vectorized performance changes, and any
  function used by both live and backtests.
- Avoid without a baseline: changing rolling-window calculations, feature column
  order, label joins, train/cal/test masks, threshold selection, or OOS exit
  sequencing.

## Verification Baseline

Run from the repository root with the intended environment:

```bash
python3 -m compileall -q core backtests live scripts tests archive run_in_sample_backtest.py run_oos_backtest.py
python3 -m pytest tests/ -q
```

If `pytest` is not installed, `compileall` plus import/parity smoke tests are the
minimum check. Do not treat a successful syntax check as proof that model results
are unchanged.

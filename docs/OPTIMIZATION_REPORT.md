# Optimization Report

## Completed

- Added `docs/PROJECT_STRUCTURE.md` with project layers, active entry points,
  dependency flow, and model-result sensitive zones.
- Centralized archive helper usage:
  - `archive/backtest_signal_combos.py` now reuses `core.utils` for
    `load_price_data`, `MINUTES_PER_YEAR`, and `calculate_max_drawdown`.
  - Archive plotting/backtest scripts now use explicit `archive.*` imports and
    add the repo root to `sys.path` when run directly.
- Restored `archive/grid_search_ce_params.py` imports by keeping historical CE
  grid constants local to the script and importing `load_price_data` from
  `core.utils`.
- Aligned `core.utils.DATA_DIR` with the repo-root default used by trainers and
  OOS while still honoring the `DATA_DIR` environment variable.
- Added `tests/test_utils_parity.py` to guard shared utility behavior:
  `DATA_DIR` resolution, same-day CE signal expansion, and max drawdown.

## Verification

- `python3 -m compileall -q core backtests live scripts tests archive run_in_sample_backtest.py run_oos_backtest.py`
  passed.
- IDE lints reported no errors for edited files.
- `python3 -m pytest tests/ -q` could not run in the current environment because
  `pytest` is not installed.

## Preserved Boundaries

No model artifacts were changed. The following areas were intentionally not
optimized because they require golden/parity baselines:

- Rolling percentile logic in `core/research/optimize_ce_zlsma_kama_rule.py`.
- Feature and label builders under `core/foundation/pa_rules.py` and `core/training/`.
- Training/inference math under `core/training/l1a`, `core/training/l1b`,
  `core/training/l2`, and `core/training/unified`.
- Straddle simulation under `core/training/unified/simulation`.
- OOS sequencing and lazy import order in `backtests/oos_backtest.py`.

## Recommended Next Guards

- Install test dependencies in the project environment and run `pytest`.
- Add a small golden feature fixture before changing PA features, labels, OOS
  exits, or rolling-window calculations.
- Add an OOS smoke baseline JSON before refactoring `backtests/oos_backtest.py`.

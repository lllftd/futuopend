# Scripts

Operational scripts are grouped by workflow:

- `training/`: training launchers and GPU presets.
- `oos/`: OOS backtest runners and chart re-rendering.
- `walkforward/`: walk-forward OOS runners and per-fold regime charts.
- `diagnostics/`: post-run diagnostics and reporting helpers.
- `config/`: JSON configs consumed by scripts.

Run scripts from the repository root. Each launcher resolves the repo root from
its own location and sets `PYTHONPATH` where needed.

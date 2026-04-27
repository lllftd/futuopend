#!/usr/bin/env bash
# Full-sample OOS: only L1a **R1** (long straddle / 做多波动). Same pattern as run_oos_fullsample_r0r1r3.sh.
# For R0+R1+R3 use:  ./scripts/oos/run_oos_fullsample_r0r1r3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export OOS_START="${OOS_START:-2018-01-01}"
export OOS_END="${OOS_END:-2035-01-01}"
export OOS_ALLOW_ENTRY_L1A_REGIMES="${OOS_ALLOW_ENTRY_L1A_REGIMES:-1}"
export OOS_RESULTS_DIR="${OOS_RESULTS_DIR:-${ROOT}/results/modeloos_fullsample_r1_longvol}"
export OOS_CHART_TITLE="${OOS_CHART_TITLE:-OOS {symbol}  R1 only  |  [{oos_start}, {oos_end})}"
export OOS_SYMBOLS="${OOS_SYMBOLS:-QQQ,SPY}"
mkdir -p "$OOS_RESULTS_DIR"
echo "Logging to: ${OOS_RESULTS_DIR}/oos_run.log" >&2
exec > >(tee -a "${OOS_RESULTS_DIR}/oos_run.log")
exec 2>&1
echo "[run_oos_fullsample_r1_longvol] OOS_ALLOW_ENTRY_L1A_REGIMES=$OOS_ALLOW_ENTRY_L1A_REGIMES" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
python3 -u backtests/oos_backtest.py
echo "[run_oos_fullsample_r1_longvol] done" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

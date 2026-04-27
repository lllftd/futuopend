#!/usr/bin/env bash
# Full-sample OOS (same style as modeloos_fullsample) but only allow L1a **R0, R1, R3** entries:
#   R0 → short straddle (or IBF if OOS_R0_STRATEGY), R1 → long straddle, R3 → short straddle.
#   R2 (gamma scalp) and R4 (iron condor) are blocked.
#
# Output: results/modeloos_fullsample_r0r1r3/  (override with OOS_RESULTS_DIR)
#
# Usage: from repo root
#   chmod +x scripts/oos/run_oos_fullsample_r0r1r3.sh
#   ./scripts/oos/run_oos_fullsample_r0r1r3.sh
#
# Equivalent: OOS_ALLOW_ENTRY_L1A_REGIMES=0,1,3
# Block form:  OOS_BLOCK_ENTRY_L1A_REGIMES=2,4
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
export OOS_START="${OOS_START:-2018-01-01}"
export OOS_END="${OOS_END:-2035-01-01}"
export OOS_ALLOW_ENTRY_L1A_REGIMES="${OOS_ALLOW_ENTRY_L1A_REGIMES:-0,1,3}"
export OOS_RESULTS_DIR="${OOS_RESULTS_DIR:-${ROOT}/results/modeloos_fullsample_r0r1r3}"
export OOS_CHART_TITLE="${OOS_CHART_TITLE:-OOS {symbol}  L1a R0,R1,R3 only  |  [{oos_start}, {oos_end})}"
export OOS_SYMBOLS="${OOS_SYMBOLS:-QQQ,SPY}"
mkdir -p "$OOS_RESULTS_DIR"
echo "Logging to: ${OOS_RESULTS_DIR}/oos_run.log" >&2
exec > >(tee -a "${OOS_RESULTS_DIR}/oos_run.log")
exec 2>&1
echo "[run_oos_fullsample_r0r1r3] OOS_START=$OOS_START OOS_END=$OOS_END OOS_ALLOW_ENTRY_L1A_REGIMES=$OOS_ALLOW_ENTRY_L1A_REGIMES OOS_RESULTS_DIR=$OOS_RESULTS_DIR" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
python3 -u backtests/oos_backtest.py
echo "[run_oos_fullsample_r0r1r3] done" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

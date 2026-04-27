#!/usr/bin/env bash
# Parallel OOS for SPY + QQQ with a shared exit-prob threshold (default 0.35).
# Each process uses its own OOS_RESULTS_DIR to avoid races on trades_ALL / oos_summary.
# Usage:
#   ./scripts/oos/run_oos_parallel_spy_qqq.sh
#   OOS_L3_EXIT_PROB_THRESHOLD=0.40 ./scripts/oos/run_oos_parallel_spy_qqq.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT"
THR="${OOS_L3_EXIT_PROB_THRESHOLD:-0.35}"
export OOS_L3_EXIT_PROB_THRESHOLD="$THR"
PY="${ROOT}/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/results/modeloos_thr${THR}_${STAMP}"
mkdir -p "$OUT/qqq" "$OUT/spy"
echo "OOS_L3_EXIT_PROB_THRESHOLD=$THR  output root: $OUT"
(
  export OOS_SYMBOLS=QQQ
  export OOS_RESULTS_DIR="$OUT/qqq"
  "$PY" -u "$ROOT/backtests/oos_backtest.py"
) 2>&1 | tee "$OUT/qqq/oos.log" &
PID1=$!
(
  export OOS_SYMBOLS=SPY
  export OOS_RESULTS_DIR="$OUT/spy"
  "$PY" -u "$ROOT/backtests/oos_backtest.py"
) 2>&1 | tee "$OUT/spy/oos.log" &
PID2=$!
wait "$PID1" "$PID2"
echo "Done. QQQ -> $OUT/qqq  SPY -> $OUT/spy"

#!/usr/bin/env bash
# Run TCN training; stdout/stderr go to one log file overwritten each run (no -a).
# Override path: TCN_TRAIN_LOG=/path/to/log ./scripts/run_train_tcn.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${TCN_TRAIN_LOG:-train_tcn.log}"
: >"$LOG"
FORCE_TQDM=1 python3 -m backtests.train_tcn_pa_state 2>&1 | tee "$LOG"

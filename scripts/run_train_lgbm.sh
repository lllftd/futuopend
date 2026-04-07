#!/usr/bin/env bash
# Run layered LGBM training; stdout/stderr are copied to a single log file that is
# overwritten every run (no -a). Override path: LGBM_TRAIN_LOG=/path/to/log ./scripts/run_train_lgbm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${LGBM_TRAIN_LOG:-train_lgbm.log}"
: >"$LOG"
# tqdm uses stderr.isatty(); 2>&1 | tee makes stderr a pipe — force bars + log capture.
FORCE_TQDM=1 python3 -m backtests.train_lgbm_pa_state 2>&1 | tee "$LOG"

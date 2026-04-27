#!/usr/bin/env bash
# Walk-forward OOS: **same 4 default folds** as ``results/walkforward_true_oos_4f`` (see ``run_walkforward.py`` DEFAULT_FOLDS)
# + **L1a entry** limited to R0, R1, R3  (GAMMA R2 与 IRON_CONDOR R4 在入场时屏蔽, = ``OOS_BLOCK_ENTRY_L1A_REGIMES=2,4``).
#
# From repo root:
#   chmod +x scripts/walkforward/run_walkforward_oos_4f_r0r1r3.sh
#   ./scripts/walkforward/run_walkforward_oos_4f_r0r1r3.sh
#
# Override output dir:  WALKFORWARD_OUT=results/my_wf  ./scripts/walkforward/run_walkforward_oos_4f_r0r1r3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
OUT="${WALKFORWARD_OUT:-${ROOT}/results/walkforward_true_oos_4f_r0r1r3}"
echo "[run_walkforward_oos_4f_r0r1r3] out=${OUT}  (allow 0,1,3  ~=  block 2,4)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
python3 -u scripts/walkforward/run_walkforward.py \
  --out "${OUT}" \
  --oos-allow-l1a-regimes 0,1,3
echo "[run_walkforward_oos_4f_r0r1r3] done" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

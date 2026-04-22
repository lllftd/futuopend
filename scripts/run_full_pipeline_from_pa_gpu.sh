#!/usr/bin/env bash
# Full stack from PA cache invalidation + train_pipeline layer1 (see PowerShell script for details).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"
# Default off: Numba JIT can hard-exit on some Windows stacks; set PA_NUMBA=1 to re-enable.
export PA_NUMBA="${PA_NUMBA:-0}"
export FORCE_TQDM="${FORCE_TQDM:-1}"
export TORCH_DEVICE="${TORCH_DEVICE:-cuda}"
export L1A_AMP="${L1A_AMP:-1}"
export L1A_AMP_DTYPE="${L1A_AMP_DTYPE:-bf16}"
export TCN_AMP="${TCN_AMP:-1}"
export TCN_TRAIN_BATCH_SIZE="${TCN_TRAIN_BATCH_SIZE:-8192}"
export TCN_BATCH_SIZE="${TCN_BATCH_SIZE:-8192}"
export TORCH_ALLOW_TF32="${TORCH_ALLOW_TF32:-1}"
unset LAYER1A_USE_PREPARED_CACHE || true
unset PREPARED_DATASET_CACHE_REBUILD || true

NCPU="$( (command -v nproc >/dev/null && nproc) || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
NCPU="${NCPU:-8}"
TORCH_THREADS=$(( (NCPU + 1) / 2 ))
[[ "$TORCH_THREADS" -lt 4 ]] && TORCH_THREADS=4
[[ "$TORCH_THREADS" -gt 16 ]] && TORCH_THREADS=16
export TORCH_CPU_THREADS="${TORCH_CPU_THREADS:-$TORCH_THREADS}"
WORKERS=$(( NCPU - 2 ))
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
[[ "$WORKERS" -gt 8 ]] && WORKERS=8
export L1A_DATALOADER_WORKERS="${L1A_DATALOADER_WORKERS:-$WORKERS}"
export L1A_PREFETCH_FACTOR="${L1A_PREFETCH_FACTOR:-4}"
PREP_CPU=$(( NCPU - 2 ))
[[ "$PREP_CPU" -lt 4 ]] && PREP_CPU=4
[[ "$PREP_CPU" -gt 16 ]] && PREP_CPU=16
export PREP_CPU_WORKERS="${PREP_CPU_WORKERS:-$PREP_CPU}"

DATA_DIR="$ROOT/data"
PA_CACHE="$DATA_DIR/.pa_feature_cache"
MODEL_DIR="$ROOT/lgbm_models"

if [[ "${KEEP_PA_CACHE:-}" != "1" ]] && [[ -d "$PA_CACHE" ]]; then
  find "$PA_CACHE" -maxdepth 1 -name "*.pkl" -type f -print -delete
fi

if [[ "${KEEP_PREPARED_CACHE:-}" != "1" ]] && [[ -f "$MODEL_DIR/prepared_lgbm_dataset.pkl" ]]; then
  rm -f "$MODEL_DIR/prepared_lgbm_dataset.pkl"
  echo "Removed prepared_lgbm_dataset.pkl"
fi

if [[ "${KEEP_STACK_MODELS:-}" != "1" ]]; then
  for f in \
    l1a_market_tcn.pt l1a_market_tcn_meta.pkl l1a_outputs.pkl \
    l1b_descriptor_meta.pkl l1b_outputs.pkl l1b_edge_pred.txt l1b_dq_pred.txt \
    l2_trade_gate.txt l2_direction.txt l2_trade_gate_calibrator.pkl l2_direction_calibrator.pkl \
    l2_mfe.txt l2_mae.txt l2_decision_meta.pkl l2_outputs.pkl \
    l3_exit.txt l3_value.txt l3_exit_meta.pkl l3_trajectory_encoder.pt \
    l3_policy_dataset.pkl l3_cox_time_varying.pkl; do
    [[ -f "$MODEL_DIR/$f" ]] && rm -f "$MODEL_DIR/$f" && echo "Removed $f"
  done
fi

PY=""
for CAND in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$ROOT/quickvenv/bin/python" "$ROOT/testenv/bin/python"; do
  if [[ -x "$CAND" ]]; then PY="$CAND"; break; fi
done
[[ -n "$PY" ]] || PY="python3"

echo "Using: $PY | TORCH_DEVICE=$TORCH_DEVICE L1A_AMP=$L1A_AMP TORCH_CPU_THREADS=$TORCH_CPU_THREADS"
exec "$PY" -u -m backtests.train_pipeline --start-from layer1

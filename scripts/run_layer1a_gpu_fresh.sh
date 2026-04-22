#!/usr/bin/env bash
# Train L1a from scratch (CUDA by default; use TORCH_DEVICE=mps on Apple Silicon).
# Same intent as run_layer1a_gpu_fresh.ps1.
# Usage from repo root:
#   ./scripts/run_layer1a_gpu_fresh.sh
#   USE_PREPARED=1 ./scripts/run_layer1a_gpu_fresh.sh
#   REBUILD_PREP=1 ./scripts/run_layer1a_gpu_fresh.sh
#
# Fast wall-clock preset (env-only; target under ~2h on MPS — tune to your machine):
#   L1A_FAST=1 TORCH_DEVICE=mps ./scripts/run_layer1a_gpu_fresh.sh
#   (Optional) USE_PREPARED=1 to reuse prepared_lgbm_dataset.pkl — skips long PA+prep when unchanged.
#   Expanding OOF (default L1_OOF_MODE=expanding): fold count is len(L1_EXPAND_OOF_VAL_WINDOWS) or
#   L1a-only override L1A_EXPAND_OOF_VAL_WINDOWS — NOT L1_OOF_FOLDS (that applies only when
#   L1_OOF_MODE=blocked). The fast preset sets 2 L1a-only expanding segments + skips cal_full/oof_cal_full
#   metric passes (L1A_SKIP_CAL_FULL_METRICS), enables OOF warmstart (L1A_OOF_WARMSTART), and raises
#   L1A_MATERIALIZE_BATCH_SIZE for faster l1a_outputs materialization.
#
# Mid-term: shorter context L1A_SEQ_LEN=30 needs an L1b ablation before adopting in the stack.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export FORCE_TQDM="${FORCE_TQDM:-1}"
export TORCH_DEVICE="${TORCH_DEVICE:-cuda}"
export L1A_AMP="${L1A_AMP:-1}"

if [[ "${L1A_FAST:-}" == "1" ]]; then
  : "${L1A_EXPAND_OOF_VAL_WINDOWS:=2022-07-01:2024-01-01,2024-01-01:2024-07-01}"
  export L1A_EXPAND_OOF_VAL_WINDOWS
  : "${L1A_SKIP_CAL_FULL_METRICS:=1}"
  export L1A_SKIP_CAL_FULL_METRICS
  : "${L1A_OOF_WARMSTART:=1}"
  export L1A_OOF_WARMSTART
  : "${L1A_MATERIALIZE_BATCH_SIZE:=2048}"
  export L1A_MATERIALIZE_BATCH_SIZE
fi

# Expanding OOF only: fold 2+ warm-starts from previous fold (faster). Example:
#   L1A_OOF_WARMSTART=1 L1A_WARMSTART_OOF_MAX_EPOCHS=24 L1A_WARMSTART_OOF_LR_SCALE=0.5 ./scripts/run_layer1a_gpu_fresh.sh
# TCN stack: default is 5 blocks (larger RF). Slimmer: L1A_TCN_CHANNELS=48,48,96  or  legacy 3-block: 64,64,128
# Fewer L1a OOF folds under expanding mode: set L1A_EXPAND_OOF_VAL_WINDOWS (not L1_OOF_FOLDS); or
# L1_OOF_MODE=blocked L1_OOF_FOLDS=2 for legacy contiguous blocks inside train+cal.

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

if [[ "${USE_PREPARED:-}" == "1" ]]; then
  export LAYER1A_USE_PREPARED_CACHE=1
else
  unset LAYER1A_USE_PREPARED_CACHE || true
fi

if [[ "${REBUILD_PREP:-}" == "1" ]]; then
  export PREPARED_DATASET_CACHE_REBUILD=1
else
  unset PREPARED_DATASET_CACHE_REBUILD || true
fi

MODEL_DIR="$ROOT/lgbm_models"
if [[ "${SKIP_REMOVE_L1A:-}" != "1" ]]; then
  for f in l1a_market_tcn.pt l1a_market_tcn_meta.pkl l1a_outputs.pkl; do
    if [[ -f "$MODEL_DIR/$f" ]]; then
      rm -f "$MODEL_DIR/$f"
      echo "Removed $MODEL_DIR/$f"
    fi
  done
fi

PY=""
for CAND in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$ROOT/quickvenv/bin/python" "$ROOT/testenv/bin/python"; do
  if [[ -x "$CAND" ]]; then PY="$CAND"; break; fi
done
[[ -n "$PY" ]] || PY="python3"

echo "Using: $PY"
echo "TORCH_DEVICE=$TORCH_DEVICE L1A_AMP=$L1A_AMP L1A_FAST=${L1A_FAST:-0} TORCH_CPU_THREADS=$TORCH_CPU_THREADS L1A_DATALOADER_WORKERS=$L1A_DATALOADER_WORKERS"
exec "$PY" -u "$ROOT/backtests/train_layer1a_only.py"

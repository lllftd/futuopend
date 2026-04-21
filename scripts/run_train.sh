#!/usr/bin/env bash
# Run the dual-view stack training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
#   layer1 | layer1a — train L1a → L1b → L2 → L3
#   layer1b — load l1a_outputs.pkl, train L1b → L2 → L3
#   layer1c — prepared dataset cache + train L1c only
#   layer2 — load l1a + l1b caches, train L2 → L3
#   layer3 — load l1a + l2 caches, train L3 only
#   Note: L3_TRAJ_GRU=1 requires L3_OOF_FOLDS=1 (default OOF folds disable fold-wise GRU).
#
# Speed (optional env, export before calling this script):
#     Windows CUDA L1a-only fresh train: .\scripts\run_layer1a_gpu_fresh.ps1 (sets TORCH_DEVICE, L1A_AMP, threads, workers).
#     Full stack from PA recompute: .\scripts\run_full_pipeline_from_pa_gpu.ps1 (clears data/.pa_feature_cache, prepared cache, L1–L3 artifacts; then layer1).
#     TCN prep CPU: PREP_CPU_WORKERS (barrier + memmap normalize), TCN_MEMMAP_NORM_WORKERS, TCN_BARRIER_WORKERS / TCN_BARRIER_SERIAL=1, TCN_MEMMAP_NORM_SERIAL=1.
#     GPU throughput: TORCH_ALLOW_TF32=1, TORCH_MATMUL_PRECISION=high, L1A_AMP=1 + L1A_AMP_DTYPE=bf16|fp16, TCN_AMP=1, TCN_TRAIN_BATCH_SIZE / TCN_BATCH_SIZE (VRAM), TCN_DATALOADER_WORKERS (Windows CUDA default >0).
#     LAYER1A_USE_PREPARED_CACHE=1 — layer1a loads lgbm_models/prepared_lgbm_dataset.pkl and skips
#       hours of PA+prep (use after one full prep; PREPARED_DATASET_CACHE_REBUILD=1 to rebuild).
#     PA_LIGHT_1M_EXTRAS=1 — skip heaviest 1m PA blocks (wavelet/hawkes/kalman/hurst/entropy/jump).
#     Unset PA_TIMEFRAMES for legacy single-TF PA (multi-TF is slower).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_LAYER="${1:-layer1}"
VALID_LAYERS="layer1 layer1a layer1b layer1c layer2 layer3"

if [[ ! " $VALID_LAYERS " =~ " $START_LAYER " ]]; then
    echo "Error: Invalid start layer '$START_LAYER'."
    echo "Valid options: layer1, layer1a, layer1b, layer1c, layer2, layer3"
    exit 1
fi

echo "================================================================="
echo "  Starting Dual-View Training Pipeline from: $START_LAYER"
echo "  Layer logs (fixed paths under $ROOT/logs/, opened per stage by train_pipeline):"
echo "    layer1a.log layer1b.log layer1c.log layer2.log layer3.log"
echo "================================================================="

mkdir -p "$ROOT/logs"

PY=""
for CANDIDATE in \
  "${ROOT}/venv/bin/python" \
  "${ROOT}/.venv/bin/python" \
  "${ROOT}/quickvenv/bin/python" \
  "${ROOT}/testenv/bin/python"
do
  if [[ -x "$CANDIDATE" ]]; then
    PY="$CANDIDATE"
    break
  fi
done
if [[ -z "$PY" ]]; then
  PY="python3"
fi
echo "  Python: $PY"

if [[ "$START_LAYER" == "layer3" || "$START_LAYER" == "layer2" || "$START_LAYER" == "layer1" || "$START_LAYER" == "layer1a" || "$START_LAYER" == "layer1b" ]]; then
  if "$PY" - <<'PYEOF' >/dev/null 2>&1
import lifelines  # noqa: F401
PYEOF
  then
    echo "  [L3] Cox dependency check: lifelines available"
  else
    echo "  [L3] Cox dependency check: lifelines missing (run: $PY -m pip install -r requirements.txt)"
  fi
fi

FORCE_TQDM=1 "$PY" -u -m backtests.train_pipeline --start-from "$START_LAYER"

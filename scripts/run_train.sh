#!/usr/bin/env bash
# Run the unified PA state training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
# Example: ./scripts/run_train.sh layer1 (runs everything from TCN)
# Example: ./scripts/run_train.sh layer3 (skips TCN, L2a, L2b and runs L3, L4)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_LAYER="${1:-layer1}"
# layer2 = layer2a (train LGBM from regime; skips TCN/Mamba). To ignore Mamba weights if present:
#   DISABLE_MAMBA_FEATURES=1 ./scripts/run_train.sh layer2
# TCN OOF already done and tcn_oof_cache.pkl exists — train final TCN only (saves time & RAM):
#   TCN_SKIP_OOF=1 ./scripts/run_train.sh layer1
VALID_LAYERS="layer1 layer1a layer1b layer2 layer2a layer2b layer3 layer4"

if [[ ! " $VALID_LAYERS " =~ " $START_LAYER " ]]; then
    echo "Error: Invalid start layer '$START_LAYER'."
    echo "Valid options: layer1, layer1a, layer1b, layer2, layer2a, layer2b, layer3, layer4"
    exit 1
fi

echo "================================================================="
echo "  Starting Unified Training Pipeline from: $START_LAYER"
echo "  Log files will be written to: $ROOT/logs/"
if [[ "$START_LAYER" == "layer2" || "$START_LAYER" == "layer2a" ]]; then
  echo "  (TCN+meta must exist under lgbm_models/.) Mamba: auto if checkpoint present;"
  echo "   to force no Mamba columns: DISABLE_MAMBA_FEATURES=1 $0 $START_LAYER"
fi
echo "================================================================="

mkdir -p "$ROOT/logs"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

FORCE_TQDM=1 "$PY" -m backtests.train_pipeline --start-from "$START_LAYER"

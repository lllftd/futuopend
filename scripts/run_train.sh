#!/usr/bin/env bash
# Run the unified PA state training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
# Example: ./scripts/run_train.sh layer1 (runs everything from TCN)
# Example: ./scripts/run_train.sh layer3 (skips TCN, L2a, L2b and runs L3, L4)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_LAYER="${1:-layer1}"
VALID_LAYERS="layer1 layer1a layer1b layer2a layer2b layer3 layer4"

if [[ ! " $VALID_LAYERS " =~ " $START_LAYER " ]]; then
    echo "Error: Invalid start layer '$START_LAYER'."
    echo "Valid options: layer1, layer1a, layer1b, layer2a, layer2b, layer3, layer4"
    exit 1
fi

echo "================================================================="
echo "  Starting Unified Training Pipeline from: $START_LAYER"
echo "  Log files will be written to: $ROOT/logs/"
echo "================================================================="

mkdir -p "$ROOT/logs"

FORCE_TQDM=1 python3 -m backtests.train_pipeline --start-from "$START_LAYER"

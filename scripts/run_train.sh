#!/usr/bin/env bash
# Run the dual-view stack training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
# Example: ./scripts/run_train.sh layer1 (runs L1a -> L1b -> L2 -> L3)
# Example: ./scripts/run_train.sh layer3 (reuses cached L1a/L2 outputs and trains only L3)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_LAYER="${1:-layer1}"
VALID_LAYERS="layer1 layer1a layer1b layer2 layer3"

if [[ ! " $VALID_LAYERS " =~ " $START_LAYER " ]]; then
    echo "Error: Invalid start layer '$START_LAYER'."
    echo "Valid options: layer1, layer1a, layer1b, layer2, layer3"
    exit 1
fi

echo "================================================================="
echo "  Starting Dual-View Training Pipeline from: $START_LAYER"
echo "  Logs (overwrite each run, under $ROOT/logs/):"
echo "    layer1a.log layer1b.log layer2.log layer3.log"
if [[ "$START_LAYER" != "layer3" ]]; then
  echo "  Note: current pipeline still retrains the full stack unless you start from layer3."
fi
echo "================================================================="

mkdir -p "$ROOT/logs"
# Pre-create log files so `tail -f` works immediately.
case "$START_LAYER" in
  layer1)
    : > "$ROOT/logs/layer1a.log"
    : > "$ROOT/logs/layer1b.log"
    : > "$ROOT/logs/layer2.log"
    : > "$ROOT/logs/layer3.log"
    ;;
  layer1a|layer1b)
    : > "$ROOT/logs/layer1a.log"
    : > "$ROOT/logs/layer1b.log"
    : > "$ROOT/logs/layer2.log"
    : > "$ROOT/logs/layer3.log"
    ;;
  layer2)
    : > "$ROOT/logs/layer2.log"
    : > "$ROOT/logs/layer3.log"
    ;;
  layer3)
    : > "$ROOT/logs/layer3.log"
    ;;
esac

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

FORCE_TQDM=1 "$PY" -m backtests.train_pipeline --start-from "$START_LAYER"

#!/usr/bin/env bash
# Run the dual-view stack training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
#   layer1 | layer1a — train L1a → L1b → L2 → L3
#   layer1b — load l1a_outputs.pkl, train L1b → L2 → L3
#   layer1c — prepared dataset cache + train L1c only
#   layer2 — load l1a + l1b caches, train L2 → L3
#   layer3 — load l1a + l2 caches, train L3 only

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
echo "  Logs (overwrite for stages that run, under $ROOT/logs/):"
echo "    layer1a.log layer1b.log layer1c.log layer2.log layer3.log"
echo "================================================================="

mkdir -p "$ROOT/logs"
# Pre-create log files so `tail -f` works immediately (only stages that will run).
case "$START_LAYER" in
  layer1|layer1a)
    : > "$ROOT/logs/layer1a.log"
    : > "$ROOT/logs/layer1b.log"
    : > "$ROOT/logs/layer1c.log"
    : > "$ROOT/logs/layer2.log"
    : > "$ROOT/logs/layer3.log"
    ;;
  layer1b)
    : > "$ROOT/logs/layer1b.log"
    : > "$ROOT/logs/layer1c.log"
    : > "$ROOT/logs/layer2.log"
    : > "$ROOT/logs/layer3.log"
    ;;
  layer1c)
    : > "$ROOT/logs/layer1c.log"
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

#!/usr/bin/env bash
# Run L1b × L1a feature ablation (B0 / B1 / B2) sequentially, saving logs + meta snapshots.
#
# Requires:
#   - lgbm_models/prepared_lgbm_dataset.pkl
#   - lgbm_models/l1a_outputs.pkl + l1a_market_tcn.pt + l1a_market_tcn_meta.pkl (train L1a first)
#
# Optional: PY=/path/to/python

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODEL_DIR="${ROOT}/lgbm_models"
OUT_ROOT="${MODEL_DIR}/l1b_ablation"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

if [[ ! -f "${MODEL_DIR}/l1a_outputs.pkl" ]]; then
  echo "ERROR: ${MODEL_DIR}/l1a_outputs.pkl not found."
  echo "  Train L1a first, e.g.: ./scripts/training/run_train.sh layer1a"
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/prepared_lgbm_dataset.pkl" ]]; then
  echo "ERROR: ${MODEL_DIR}/prepared_lgbm_dataset.pkl not found."
  exit 1
fi

PY="${PY:-}"
if [[ -z "$PY" ]]; then
  for C in "${ROOT}/venv/bin/python" "${ROOT}/.venv/bin/python" "${ROOT}/quickvenv/bin/python"; do
    if [[ -x "$C" ]]; then PY="$C"; break; fi
  done
fi
[[ -z "$PY" ]] && PY="python3"

export PYTHONPATH="${ROOT}"

echo "Using Python: $PY"
echo "Ablation outputs: $OUT_ROOT"
echo "================================================================="

run_one() {
  local tag="$1"
  local log="${LOG_DIR}/l1b_ablation_${tag}.log"
  echo ""
  echo "[*] === ${tag} === logging to ${log}"
  FORCE_TQDM=1 "$PY" -u -m backtests.train_pipeline --start-from layer1b 2>&1 | tee "$log"

  local dest="${OUT_ROOT}/${tag}"
  mkdir -p "$dest"
  for f in l1b_descriptor_meta.pkl l1b_outputs.pkl l1b_edge_pred.txt l1b_dq_pred.txt \
           l1b_edge_candidate.txt drift_ref_l1b.pkl; do
    [[ -f "${MODEL_DIR}/${f}" ]] && cp -a "${MODEL_DIR}/${f}" "${dest}/"
  done
  echo "[*] ${tag} artifacts copied -> ${dest}"
}

unset L1B_EXTRA_FEATURE_ALLOWLIST || true

export L1B_USE_L1A_FEATURES=0
unset L1B_L1A_FEATURE_TIER || true
run_one b0

export L1B_USE_L1A_FEATURES=1
export L1B_L1A_FEATURE_TIER=scalar
run_one b1

export L1B_USE_L1A_FEATURES=1
export L1B_L1A_FEATURE_TIER=full
run_one b2

echo ""
echo "================================================================="
echo "DONE. Compare meta + logs under ${OUT_ROOT} and ${LOG_DIR}/l1b_ablation_*.log"

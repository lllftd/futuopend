#!/usr/bin/env bash
# Run the dual-view stack training pipeline
# Usage: ./scripts/run_train.sh [layer_name]
#   layer1 | layer1a — train L1a → L1b → L2 → L3
#   layer1b — load l1a_outputs.pkl, train L1b → L2 → L3
#   layer2 — load l1a + l1b caches, train L2 → L3
#   layer3 — load l1a + l2 caches, train L3 only
#   oos_backtest — run backtests/oos_backtest.py (Sharpe / PnL / drawdown; needs L1–L3 artifacts in lgbm_models/)
#   Note: L3_TRAJ_GRU=1 requires L3_OOF_FOLDS=1 (default OOF folds disable fold-wise GRU).
#   L2 opens / L3 exits only: L3_TRUST_L2_ENTRY=1 skips L3 entry-policy grid (uses L3_ENTRY_MIN_CONFIDENCE / L3_ENTRY_MIN_SIZE, default 0/0).
#   L2 straddle policy (defaults favor higher signal rate): L2_TARGET_TRADE_RATE=0.10; L2_TRADE_RATE_BAND=0.07,0.15
#     (or L2_TRADE_THR_SEARCH_MIN/MAX); L2_SORTINO_LAMBDA=0.15; L2_SORTINO_TARGET_RATE_PENALTY=0.05.
#     L2_REGIME_STRADDLE_ADJUST=1 — vol×trend buckets + per-regime min_profit/size_mult; optional L2_REGIME_STRADDLE_CONFIG_JSON.
#
# Speed (optional env, export before calling this script):
#     Windows CUDA L1a-only fresh train: .\scripts\run_layer1a_gpu_fresh.ps1 (sets TORCH_DEVICE, L1A_AMP, threads, workers).
#     Full stack from PA recompute: .\scripts\run_full_pipeline_from_pa_gpu.ps1 (clears data/.pa_feature_cache, prepared cache, L1–L3 artifacts; then layer1).
#     TCN prep CPU: PREP_CPU_WORKERS (barrier + memmap normalize), TCN_MEMMAP_NORM_WORKERS, TCN_BARRIER_WORKERS / TCN_BARRIER_SERIAL=1, TCN_MEMMAP_NORM_SERIAL=1.
#     GPU throughput: TORCH_ALLOW_TF32=1, TORCH_MATMUL_PRECISION=high, L1A_AMP=1 + L1A_AMP_DTYPE=bf16|fp16, TCN_AMP=1, TCN_TRAIN_BATCH_SIZE / TCN_BATCH_SIZE (VRAM), TCN_DATALOADER_WORKERS (Windows CUDA default >0).
#     layer1 / layer1a: script defaults LAYER1A_USE_PREPARED_CACHE=1 and L1_OOF_MODE=blocked L1_OOF_FOLDS=1 (override if needed).
#     LAYER1A_USE_PREPARED_CACHE=0 — force full PA+prep (PREPARED_DATASET_CACHE_REBUILD=1 rebuilds cache after prep).
#     L1a fast train (env-only / Mac MPS): L1A_FAST=1 ./scripts/run_layer1a_gpu_fresh.sh  (or set the same
#       vars by hand: L1A_EXPAND_OOF_VAL_WINDOWS two segments, L1A_SKIP_CAL_FULL_METRICS=1, L1A_OOF_WARMSTART=1;
#       TORCH_DEVICE=mps). Retrain L1a only when upstream PA/prep or feature contract changes — reuse pkl otherwise.
#     PA_LIGHT_1M_EXTRAS=1 — skip heaviest 1m PA blocks (wavelet/hawkes/kalman/hurst/entropy/jump).
#     Unset PA_TIMEFRAMES for legacy single-TF PA (multi-TF is slower).
#
# L1a regime is fixed 5-class vol lifecycle (no env toggle). Optional straddle-edge head: L1A_STRADDLE_EDGE_HEAD=0 to disable.
#   TCN depth/width (receptive field): default 5 layers L1A_TCN_CHANNELS=64,64,64,128,128 — override with one env line if needed.
#   L1a training progress: tqdm on stderr (epoch/fold). Per-batch tqdm off by default (speed); L1A_TQDM_BATCH=1 to enable.
#   Hierarchical aux on base_regime head: 5→2 CE; aux enters Kendall UW as task regime_aux (× L1A_REGIME_AUX_COEF).
#   Multi-task scaling: Kendall uncertainty (per-task learnable log σ²); L1A_UW_LR_RATIO scales UW params vs model.
#   Schedule: L1A_MAX_EPOCHS (default 24), L1A_PATIENCE (default 10).
#   Edge label: L1A_STRADDLE_EDGE_HOLD_BARS (default 10), L1A_STRADDLE_EDGE_CLIP_ABS (default 0.1).
#   Prep always adds vol_regime_label + l1a_straddle_edge_target; HMM state_label stays 6-class for PA features.
#
# L1b ablation (three runs; same fixed lgbm_models/l1a_outputs.pkl; only L1b L1a feature tier changes):
#   Columns — scalar tier = 5× l1a_regime_prob_* + transition_risk, vol_forecast, vol_trend, time_in_regime,
#     state_persistence (+ l1a_straddle_edge when edge head on); full adds embed_0..D-1.
#   Metrics — logs/layer1b.log + trainer stdout: L1b OOF (AUC / MAE / corr / staged val; not Sharpe — use oos_backtest).
#   Backup — each run overwrites lgbm_models/l1b_* and logs/layer1b.log; copy aside between runs to compare.
#
#   L1b-only (recommended for ablation — no L2/L3); from repo root:
#     (1) PYTHONPATH=. L1B_USE_L1A_FEATURES=0 python3 -u backtests/train_layer1b_only.py
#     (1′) Strict baseline (same ~682k rows × 3 expanding folds as L1a/full — no sample-size confound):
#          PYTHONPATH=. L1B_USE_L1A_FEATURES=0 L1B_BASELINE_ALIGN_TO_L1A_POOL=1 python3 -u backtests/train_layer1b_only.py
#     (2) PYTHONPATH=. L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=scalar python3 -u backtests/train_layer1b_only.py
#     (3) PYTHONPATH=. L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=full  python3 -u backtests/train_layer1b_only.py
#
#   Full stack from L1b onward (L1b then L2 then L3 — slower; not needed if you only need L1b OOF):
#     (1) L1B_USE_L1A_FEATURES=0 ./scripts/run_train.sh layer1b
#     (1′) Strict baseline: L1B_USE_L1A_FEATURES=0 L1B_BASELINE_ALIGN_TO_L1A_POOL=1 ./scripts/run_train.sh layer1b
#     (2) L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=scalar ./scripts/run_train.sh layer1b
#     (3) L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=full ./scripts/run_train.sh layer1b
#   Train L2/L3 after choosing L1b tier: L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=full ./scripts/run_train.sh layer2
#
#   Optional: L1B_EXTRA_FEATURE_ALLOWLIST=comma,separated,... overrides tier-derived list when set (see l1a_bridge).
#   When L1a features are on (scalar/full), honest fit mask + shifted L1b expanding OOF: L1B_EXPAND_OOF_VAL_WINDOWS in constants.py
#
# OOS / Sharpe & PnL (full stack artifacts in lgbm_models/: L1a→L3 trained to match L1b tier):
#     ./scripts/run_train.sh oos_backtest
#   Writes under results/modeloos/ (or OOS_RESULTS_DIR): trades_*.csv, oos_summary.json, oos_equity_curves.csv, oos_chart_*.png.
#   Window: OOS_START / OOS_END (default OOS_START=TEST_END from constants). Costs: OOS_SLIPPAGE_BPS, OOS_COMMISSION_BPS_PER_SIDE.
#   ATR exit ablation (no L3 exit model; compare Calmar/DD to learned L3): OOS_L3_EXIT_MODE=atr_trailing
#     with OOS_ATR_TRAIL_MULT=1.2  OOS_ATR_TRAIL_MIN_HOLD=1
#   Block entries by L1a argmax regime (e.g. drop regime 1): OOS_BLOCK_ENTRY_L1A_REGIMES=1
#
# L1b snapshot before a run that would overwrite l1b_*.pkl (artifacts are flat files under lgbm_models/, not a lgbm_models/l1b/ dir):
#     mkdir -p lgbm_models/snapshots/l1b_full_YYYYMMDD && cp lgbm_models/l1b_descriptor_meta.pkl lgbm_models/l1b_outputs.pkl \
#       lgbm_models/l1b_edge_candidate.txt lgbm_models/l1b_edge_pred.txt lgbm_models/l1b_dq_pred.txt lgbm_models/snapshots/l1b_full_YYYYMMDD/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_LAYER="${1:-layer1}"
VALID_LAYERS="layer1 layer1a layer1b layer2 layer3 oos_backtest"

if [[ ! " $VALID_LAYERS " =~ " $START_LAYER " ]]; then
    echo "Error: Invalid start layer '$START_LAYER'."
    echo "Valid options: layer1, layer1a, layer1b, layer2, layer3, oos_backtest"
    exit 1
fi

# L1a entrypoints: reuse prepared_lgbm_dataset.pkl when present, and legacy single train path (no expanding OOF).
# Override: LAYER1A_USE_PREPARED_CACHE=0  L1_OOF_MODE=expanding  (unset L1_OOF_FOLDS or set folds >1 as needed).
if [[ "$START_LAYER" == "layer1a" || "$START_LAYER" == "layer1" ]]; then
  export LAYER1A_USE_PREPARED_CACHE="${LAYER1A_USE_PREPARED_CACHE:-1}"
  export L1_OOF_MODE="${L1_OOF_MODE:-blocked}"
  export L1_OOF_FOLDS="${L1_OOF_FOLDS:-1}"
  echo "  [run_train] layer1a defaults: LAYER1A_USE_PREPARED_CACHE=$LAYER1A_USE_PREPARED_CACHE  L1_OOF_MODE=$L1_OOF_MODE  L1_OOF_FOLDS=$L1_OOF_FOLDS"
fi

if [[ "$START_LAYER" == "oos_backtest" ]]; then
  echo "================================================================="
  echo "  OOS backtest (L1a→L3 stack, sample >= TEST_END by default)"
  echo "================================================================="
  PY_OOS=""
  for CANDIDATE in \
    "${ROOT}/venv/bin/python" \
    "${ROOT}/.venv/bin/python" \
    "${ROOT}/quickvenv/bin/python" \
    "${ROOT}/testenv/bin/python"
  do
    if [[ -x "$CANDIDATE" ]]; then
      PY_OOS="$CANDIDATE"
      break
    fi
  done
  if [[ -z "$PY_OOS" ]]; then
    PY_OOS="python3"
  fi
  export PYTHONPATH="$ROOT"
  exec "$PY_OOS" -u "$ROOT/backtests/oos_backtest.py"
fi

echo "================================================================="
echo "  Starting Dual-View Training Pipeline from: $START_LAYER"
echo "  Layer logs (fixed paths under $ROOT/logs/, opened per stage by train_pipeline):"
echo "    layer1a.log layer1b.log layer2.log layer3.log"
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

# tqdm: non-TTY (tee logs/*.log) hides bars unless forced. Default ON here; set FORCE_TQDM=0 for quieter logs.
export FORCE_TQDM="${FORCE_TQDM:-1}"
"$PY" -u -m backtests.train_pipeline --start-from "$START_LAYER"

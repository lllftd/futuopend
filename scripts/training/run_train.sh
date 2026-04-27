#!/usr/bin/env bash
# Run the dual-view stack training pipeline
# Usage: ./scripts/training/run_train.sh [layer_name]
#   layer1 | layer1a — train L1a → L1b → L2 → L3
#   layer1b — load l1a_outputs.pkl, train L1b → L2 → L3
#   layer2 — load l1a + l1b caches, train L2 → L3
#   layer3 — load l1a + l2 caches, train L3 only
#   oos_backtest — run backtests/oos_backtest.py (Sharpe / PnL / drawdown; needs L1–L3 artifacts in lgbm_models/)
#   Note: L3_TRAJ_GRU=1 requires L3_OOF_FOLDS=1 (default OOF folds disable fold-wise GRU).
#   L2 opens / L3 exits only: L3_TRUST_L2_ENTRY=1 skips L3 entry-policy grid (uses L3_ENTRY_MIN_CONFIDENCE / L3_ENTRY_MIN_SIZE, default 0/0).
#   L2 straddle vol-only policy defaults are centralized in core/training/common/constants.py
#     (L2_VOL_ONLY_DEFAULTS / L3_VOL_EXIT_DEFAULTS). Script only sets generic training flags.
#     L2_TARGET_TRADE_RATE=0.10; L2_TRADE_RATE_BAND=0.07,0.15
#     (or L2_TRADE_THR_SEARCH_MIN/MAX); L2_SORTINO_LAMBDA=0.15; L2_SORTINO_TARGET_RATE_PENALTY=0.05.
#     L2_REGIME_STRADDLE_ADJUST=1 — vol×trend buckets + per-regime size_mult; optional L2_REGIME_STRADDLE_CONFIG_JSON.
#
# Speed (optional env, export before calling this script):
#     Windows CUDA L1a-only fresh train: .\scripts\run_layer1a_gpu_fresh.ps1 (sets TORCH_DEVICE, L1A_AMP, threads, workers).
#     Full stack from PA recompute: .\scripts\run_full_pipeline_from_pa_gpu.ps1 (clears data/.pa_feature_cache, prepared cache, L1–L3 artifacts; then layer1).
#     TCN prep CPU: PREP_CPU_WORKERS (barrier + memmap normalize), TCN_MEMMAP_NORM_WORKERS, TCN_BARRIER_WORKERS / TCN_BARRIER_SERIAL=1, TCN_MEMMAP_NORM_SERIAL=1.
#     GPU throughput: TORCH_ALLOW_TF32=1, TORCH_MATMUL_PRECISION=high, L1A_AMP=1 + L1A_AMP_DTYPE=bf16|fp16, TCN_AMP=1, TCN_TRAIN_BATCH_SIZE / TCN_BATCH_SIZE (VRAM), TCN_DATALOADER_WORKERS (Windows CUDA default >0).
#     layer1 / layer1a: script defaults LAYER1A_USE_PREPARED_CACHE=1 and L1_OOF_MODE=blocked L1_OOF_FOLDS=1 (override if needed).
#     LAYER1A_USE_PREPARED_CACHE=0 — force full PA+prep (PREPARED_DATASET_CACHE_REBUILD=1 rebuilds cache after prep).
#     L1a fast train (env-only / Mac MPS): L1A_FAST=1 ./scripts/training/run_layer1a_gpu_fresh.sh  (or set the same
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
#     (1) L1B_USE_L1A_FEATURES=0 ./scripts/training/run_train.sh layer1b
#     (1′) Strict baseline: L1B_USE_L1A_FEATURES=0 L1B_BASELINE_ALIGN_TO_L1A_POOL=1 ./scripts/training/run_train.sh layer1b
#     (2) L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=scalar ./scripts/training/run_train.sh layer1b
#     (3) L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=full ./scripts/training/run_train.sh layer1b
#   Train L2/L3 after choosing L1b tier: L1B_USE_L1A_FEATURES=1 L1B_L1A_FEATURE_TIER=full ./scripts/training/run_train.sh layer2
#
#   Optional: L1B_EXTRA_FEATURE_ALLOWLIST=comma,separated,... overrides tier-derived list when set (see l1a_bridge).
#   When L1a features are on (scalar/full), honest fit mask + shifted L1b expanding OOF: L1B_EXPAND_OOF_VAL_WINDOWS in constants.py
#
# OOS / Sharpe & PnL (full stack artifacts in lgbm_models/: L1a→L3 trained to match L1b tier):
#     ./scripts/training/run_train.sh oos_backtest
#   Writes under results/modeloos/ (or OOS_RESULTS_DIR): trades_*.csv, oos_summary.txt, oos_equity_curves.csv, oos_chart_*.png.
#   Baseline (NOT env): mtm_adaptive, straddle, no L1a regime block, no portfolio overlap cap, OOS_MTM_STATE_HOLD_FRAC=0.55
#     — edit backtests/oos_backtest.py ``OOS_BASELINE_*`` constants. Env OOS_L3_EXIT_MODE / OOS_STRATEGY_ROUTER / OOS_BLOCK_*
#     / OOS_PORTFOLIO_MAX_OPEN_POSITIONS / OOS_MTM_STATE_HOLD_* is ignored for those keys.
#   Symbols: OOS_SYMBOLS=QQQ,SPY (comma list; default both). SPY+QQQ in parallel + exit thr: ./scripts/oos/run_oos_parallel_spy_qqq.sh
#   Window: OOS_START / OOS_END (default OOS_START=TEST_END from constants). Costs: OOS_SLIPPAGE_BPS, OOS_COMMISSION_BPS_PER_SIDE.
#   ATR (still env): OOS_ATR_TRAIL_MULT  OOS_ATR_TRAIL_MIN_HOLD
#   Value-head tuning (env): OOS_L3_VALUE_EXIT_UNREAL_FRAC, OOS_L3_VALUE_EXIT_MIN_HOLD, OOS_L3_VALUE_EXIT_CONFIRM_BARS
#   MTM adaptive tunables (L3 exit head still runs; multi-signal vote; OOS_MTM_STATE_HOLD_FRAC is code-only, see OOS_BASELINE_*) —  OOS_MTM_FAST/SLOW/MIN_HOLD  OOS_MTM_VOTE_K  OOS_MTM_VOTE_K_PROFIT/LOSS
#     OOS_MTM_DD_FRACTIONAL  OOS_MTM_UNDERWATER_BARS  OOS_MTM_PROB_LEVEL  OOS_MTM_SAFETY_REGIME_P90  (optional OOS_MTM_REGIME_PROFILES_JSON)
#
# Vol-regime / synthetic path (legacy; current OOS router is straddle-only in code). When router were vol_regime, the bar loop
#   would `continue` so the L3 exit head was not applied while a synthetic position was open (rule-only exits).
#   LONG trail: OOS_LONG_TRAIL_ACTIVATE_PCT OOS_LONG_TRAIL_DRAWDOWN_FRAC OOS_LONG_TRAIL_TIGHTEN_AFTER_BAR OOS_LONG_TRAIL_TIGHTEN_SCALE
#   toggles: OOS_SYNTH_ENABLE_LONG_RULES OOS_SYNTH_ENABLE_GAMMA_RULES OOS_GAMMA_USE_LONG_TRAIL OOS_LONG_ENABLE_PROFIT_TAKE OOS_LONG_ENABLE_STOP_LOSS
#   SHORT defaults: OOS_SHORT_STRADDLE_R0_* / R3_* profit_take_frac stop_credit_mult; R3 cap: OOS_R3_MAX_HOLD_BARS
#   Gamma rolls: OOS_GAMMA_ROLL_V1 OOS_GAMMA_ROLL_TAKE_FRAC OOS_GAMMA_ROLL_MAX_ROLLS (re-enter after roll take, same signal)
#   R0 Iron Butterfly vs short straddle: OOS_R0_STRATEGY=ibf (or iron_butterfly|butterfly)
#   Position scale (linear on return): OOS_SIZE_MULT_R0..R4 OOS_SIZE_USE_SIGNAL
# Iron Condor (R4 or regime_ic): global OOS_IC_SHORT_DELTA OOS_IC_WING_WIDTH_PCT OOS_IC_MIN_WING_WIDTH OOS_IC_PROFIT_TAKE_FRAC
#   OOS_IC_STOP_CREDIT_MULT OOS_IC_STOP_MAX_LOSS_FRAC — or per-strategy OOS_IRON_CONDOR_R4_* keys via _oos_synthetic_strategy_params.
#
# L1b snapshot before a run that would overwrite l1b_*.pkl (artifacts are flat files under lgbm_models/, not a lgbm_models/l1b/ dir):
#     mkdir -p lgbm_models/snapshots/l1b_full_YYYYMMDD && cp lgbm_models/l1b_descriptor_meta.pkl lgbm_models/l1b_outputs.pkl \
#       lgbm_models/l1b_edge_candidate.txt lgbm_models/l1b_edge_pred.txt lgbm_models/l1b_dq_pred.txt lgbm_models/snapshots/l1b_full_YYYYMMDD/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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

# L3 value head: default continuous ATR forward residual (remaining_value_atr; aliases: rv_atr, forward_atr).
# Exit calibration: isotonic on val_tune only; calibrator is stored inside l3_exit_meta.pkl (no separate l3_exit_calibrator.pkl).
# Policy search (val_tune) optimizes trade-level economic uplift when row metadata allows; otherwise bar-level uplift is the fallback.
#   L3_POLICY_ADAPTIVE_GRID=1 — threshold grid from val_tune percentiles (L3_POLICY_ADAPTIVE_PERCENTILES).
#   L3_POLICY_UTILITY_TAIL_WEIGHT — mean + w*p10 trade/bar uplift objective.
#   L3_POLICY_MIN_EXIT_RATE / L3_POLICY_MIN_EXIT_RATE_ENFORCE — optional search floor on implied exit rate.
#   L3_EXIT_HOLD_INTERACTIONS=0 — turn off l3_hold_bars_x_* features (default on; changes the policy feature matrix).
#   L3_RM_ARTIFACTS=1 (or L3_CLEAN_TRAIN=1) — before layer3, delete lgbm_models/l3_*.pkl so training cannot reuse stale boosters/meta.
# L2 PyTorch unified — speed & device (no GPU: leave L2_UNIFIED_DEVICE unset; avoid forcing cpu on CUDA/MPS)
#   L2_UNIFIED_CHECKPOINT=0 — disable per-epoch resume writes (faster; no mid-run resume)
#   L2_UNIFIED_CHECKPOINT_EVERY=1 (default) — 1=each Phase1/2 epoch, 0=only last epoch of each phase, N=every N epochs (last always written)
#   L2_UNIFIED_EPOCHS / L2_UNIFIED_PHASE2_EPOCHS (default 8) — reduce for dev; L2_UNIFIED_PHASE2=0 skips exit/value heads
#   L2_UNIFIED_P2_EXIT_POS_WEIGHT — default/omit = 1.0 (no exit BCE reweight; prefer OOS threshold tune); auto|lgbm = LGBM scale_pos_weight rule
#   OOS: OOS_L3_EXIT_PROB_THRESHOLD=0.35 — override l3 meta l3_exit_prob_threshold for backtest (per-state map still can override)
#   L2_UNIFIED_CAL_MAX_ROWS>0 — sub-sample train rows in calibration summary (faster, approximate)
#   L2_UNIFIED_AMP=1 — autocast on cuda/mps (opt-in; verify metrics)
#   L2_UNIFIED_TORCH_COMPILE=1 — torch.compile(net) when supported (opt-in; first epoch may be slow)
# L2 PyTorch unified checkpoint (lgbm_models/l2_unified_train_ckpt.pt by default):
#   L2_UNIFIED_CHECKPOINT=1 — write resume file per L2_UNIFIED_CHECKPOINT_EVERY (default on, every 1)
#   L2_UNIFIED_RESUME=1 — load checkpoint and continue (Phase1 mid-epoch, after Phase1, or Phase2 mid-epoch)
#   L2_UNIFIED_CHECKPOINT_FILE=custom.pt  L2_UNIFIED_CHECKPOINT_KEEP=1 — keep file after a successful run
#   L2_AUTO_L3_META=1 (default) — after L2+recomputed l2_outputs, run L3 policy registry (same I/O as old layer3; set 0 to skip)
#
# Straddle L3 dataset build: default multi-IV paths are very slow on full history (weeks+ ETA is possible).
#   L3_STRADDLE_IV_PATH_MODE=deterministic — single flat IV path per entry (much faster; less diversity)
#   L3_STRADDLE_SCENARIO_COUNT=1  L3_STRADDLE_DTE_GRID=30 — fewer branches when not using deterministic
# Label modes: trade_outcome, remaining_value, peak_cls, regression — set L3_VALUE_TARGET_MODE.
if [[ "$START_LAYER" == "layer2" || "$START_LAYER" == "layer3" ]]; then
  export L3_VALUE_TARGET_MODE="${L3_VALUE_TARGET_MODE:-remaining_value_atr}"
  echo "  [run_train] L3 value labels: L3_VALUE_TARGET_MODE=$L3_VALUE_TARGET_MODE"
fi

# Layer3: child Python inherits this shell's env. Inline vars work: VAR=val ./scripts/training/run_train.sh layer3
# Optional: remove all L3 artifacts before training so pkls cannot be stale (same as manual rm).
if [[ "$START_LAYER" == "layer3" ]]; then
  echo "  [run_train] L3 env snapshot (empty means unset in THIS shell — export/inline before calling this script):"
  echo "    L3_POLICY_ADAPTIVE_GRID=${L3_POLICY_ADAPTIVE_GRID:-<unset>}"
  echo "    L3_POLICY_UTILITY_TAIL_WEIGHT=${L3_POLICY_UTILITY_TAIL_WEIGHT:-<unset>}"
  if [[ "${L3_RM_ARTIFACTS:-}" == "1" || "${L3_CLEAN_TRAIN:-}" == "1" ]]; then
    echo "  [run_train] L3_RM_ARTIFACTS=1 (or L3_CLEAN_TRAIN=1): removing ${ROOT}/lgbm_models/l3_*.pkl"
    shopt -s nullglob
    for _l3p in "${ROOT}/lgbm_models"/l3_*.pkl; do
      rm -f "$_l3p"
    done
    shopt -u nullglob
  fi
fi

if [[ "$START_LAYER" == "oos_backtest" ]]; then
  echo "================================================================="
  echo "  OOS backtest — repo: $ROOT (set VOLATILITY_MODEL_DIR for artifact dir if not default lgbm_models/)"
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
  export VOLATILITY_MODEL_DIR="${VOLATILITY_MODEL_DIR:-$ROOT/lgbm_models}"
  exec "$PY_OOS" -u "$ROOT/backtests/oos_backtest.py"
fi

echo "================================================================="
echo "  Starting Dual-View Training Pipeline from: $START_LAYER"
echo "  Layer logs (fixed paths under $ROOT/logs/, opened per stage by train_pipeline):"
echo "    layer1a.log layer1b.log  l2l3_unified.log  (L2+auto-L3+meta in one file; OOS: logs/oos_unified.log when enabled)"
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

# tqdm: LGBM may hide round bars to non-TTY unless FORCE_TQDM=1. L3 bar-scan is ON by default and writes to
# console/stderr (not into logs/layer3.log) when using train_pipeline. Avoid ``2>&1 | tee`` if you do not want tqdm in the file.
# L3 bar-scan tqdm: disabled when DISABLE_TQDM=1. FORCE_TQDM=0: quieter LGBM round bars.
export FORCE_TQDM="${FORCE_TQDM:-1}"
"$PY" -u -m backtests.train_pipeline --start-from "$START_LAYER"

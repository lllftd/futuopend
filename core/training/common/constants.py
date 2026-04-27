from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_DEFAULT_LGBM = os.path.join(_REPO_ROOT, "lgbm_models")
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
# External bundle (e.g. ~/Desktop/volatilitymodel/lgbm_models) without moving this repo.
MODEL_DIR = os.path.normpath(os.environ.get("VOLATILITY_MODEL_DIR", _DEFAULT_LGBM))
DATA_DIR = os.path.normpath(os.environ.get("DATA_DIR", _DEFAULT_DATA))
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")
# PA bar size for legacy `add_pa_features` / cache when PA_TIMEFRAMES is unset:
# "1min" (native 1m rules) or "5min" (closed5m buckets, causal map to 1m).
PA_TIMEFRAME = (os.environ.get("PA_TIMEFRAME", "1min").strip() or "1min")
# Multi-TF PA: comma-separated, e.g. "1min,5min,15min". When non-empty, overrides single PA_TIMEFRAME
# and produces prefixed columns (pa_1m_*, pa_5m_*, pa_15m_*). Empty = legacy single-TF mode.
PA_TIMEFRAMES_RAW = os.environ.get("PA_TIMEFRAMES", "").strip()


def pa_feature_cache_timeframe_key() -> str:
    """Cache filename token for `load_or_build_pa_features` (includes multi-TF when enabled)."""
    if PA_TIMEFRAMES_RAW:
        parts = [
            p.strip().replace("/", "_")
            for p in PA_TIMEFRAMES_RAW.split(",")
            if p.strip()
        ]
        return "mtf_" + "_".join(parts)
    return str(PA_TIMEFRAME).replace("/", "_")

def l1a_straddle_edge_head_enabled() -> bool:
    """Extra regression head on shared_repr (straddle-centric L1a). Disable via L1A_STRADDLE_EDGE_HEAD=0."""
    v = (os.environ.get("L1A_STRADDLE_EDGE_HEAD", "1") or "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def l1a_vol_trend_head_enabled() -> bool:
    """Train a vol_trend readout head. Default off: ``l1a_vol_trend`` is prep-derived; enable via L1A_VOL_TREND_HEAD=1."""
    v = (os.environ.get("L1A_VOL_TREND_HEAD", "0") or "0").strip().lower()
    return v in {"1", "true", "yes", "on"}


def l1a_time_in_regime_head_enabled() -> bool:
    """Train a time_in_regime readout head. Default off: ``l1a_time_in_regime`` is prep-derived; enable via L1A_TIME_IN_REGIME_HEAD=1."""
    v = (os.environ.get("L1A_TIME_IN_REGIME_HEAD", "0") or "0").strip().lower()
    return v in {"1", "true", "yes", "on"}


# CSV / pa_hmm / market_state imputation — always 6 directional regimes (independent of L1a head).
MARKET_STATE_NAMES: dict[int, str] = {
    0: "bull_conv",
    1: "bull_div",
    2: "bear_conv",
    3: "bear_div",
    4: "range_conv",
    5: "range_div",
}
HMM_NUM_CLASSES = len(MARKET_STATE_NAMES)

VOL_REGIME_NAMES: tuple[str, ...] = (
    "vol_compress",
    "vol_breakout",
    "vol_trending",
    "vol_exhaust",
    "vol_mean_revert",
)

# L1a / stack regime probabilities: always 5-class vol lifecycle (HMM state_label stays 6-class for PA).
NUM_REGIME_CLASSES = len(VOL_REGIME_NAMES)
REGIME_NOW_PROB_COLS = list(VOL_REGIME_NAMES)
RANGE_REGIME_INDICES = [0, 4]  # low-vol-ish: compress + mean-revert

REGIME_PROB_COLS = REGIME_NOW_PROB_COLS
STATE_NAMES: dict[int, str] = {i: REGIME_NOW_PROB_COLS[i] for i in range(NUM_REGIME_CLASSES)}

# Sequence-derived feature contracts reused by data prep.
TCN_REGIME_FUT_PROB_COLS = ["tcn_barrier_hit_up", "tcn_barrier_hit_dn", "tcn_barrier_chop"]
MAMBA_REGIME_FUT_PROB_COLS = ["mamba_transition_same", "mamba_transition_prob"]
TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "16")))
TCN_TRANSITION_PROB_COL = "tcn_transition_prob"
TCN_BARRIER_DIR_DIFF_COL = "tcn_barrier_dir_diff"
TCN_FEATURES_ENABLED = os.environ.get("TCN_FEATURES_ENABLED", "1").strip().lower() in {"1", "true", "yes"}

LGBM_EXCLUDE_PA_STRING_COLS = {
    "symbol", "time_key", "date", "market_state", "code", "name",
    "time_key_day", "market_state_shifted",
}

# Stack calendar: see stack_v2_common.L1_OOF_MODE.
# Expanding OOF (default): TRAIN_END = first expanding val start; calibration = [TRAIN_END, CAL_END);
# L2 train = union of all expanding val windows except the last; L2 val = last window; test = [CAL_END, TEST_END).
TRAIN_END = "2021-01-01"
CAL_END = "2024-07-01"
TEST_END = "2025-01-01"

# Expanding-window OOF validation segments [start, end) for L1; train for fold k is all rows with t < start_k.
# Fold count in expanding mode is len(...) below; L1_OOF_FOLDS is used only when L1_OOF_MODE=blocked.
# L1a-only: L1A_EXPAND_OOF_VAL_WINDOWS can shorten L1a OOF without changing L1b/L2 windows (see stack_v2_common).
L1_EXPAND_OOF_VAL_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
)
# L1a-only override (faster fewer folds): env ``L1A_EXPAND_OOF_VAL_WINDOWS`` comma-separated start:end pairs,
# e.g. two folds: 2022-07-01:2024-01-01,2024-01-01:2024-07-01 — see ``stack_v2_common.l1a_expand_oof_val_windows_from_env``.
# L1b supervised OOF when L1a columns are in the L1b matrix: fit pool starts at TRAIN_END, so the first
# L1_EXPAND_OOF_VAL_WINDOWS fold would have empty train. Use these shifted windows (3 folds) instead.
# Baseline ablation vs full tier (same rows/folds): L1B_BASELINE_ALIGN_TO_L1A_POOL=1 with L1B_USE_L1A_FEATURES=0.
# Env ablation: L1B_USE_L1A_FEATURES=1, L1B_L1A_FEATURE_TIER=scalar|full; see core/training/l1b/l1a_bridge.py.
L1B_EXPAND_OOF_VAL_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
)
FAST_TRAIN_MODE = os.environ.get("FAST_TRAIN", "").strip().lower() in {"1", "true", "yes"}

RNG = np.random.default_accuracy if hasattr(np.random, "default_accuracy") else np.random.RandomState(42)

BO_FEAT_COLS = [
    "bo_body_atr", "bo_range_atr", "bo_vol_spike", "bo_close_extremity",
    "bo_wick_imbalance", "bo_range_compress", "bo_body_growth", "bo_gap_signal",
    "bo_consec_dir", "bo_inside_prior", "bo_pressure_diff", "bo_or_dist",
    "bo_bb_width", "bo_atr_zscore",
]

PA_CTX_FEATURES = [
    "pa_ctx_setup_trend_long",
    "pa_ctx_setup_trend_short",
    "pa_ctx_setup_pullback_long",
    "pa_ctx_setup_pullback_short",
    "pa_ctx_setup_range_long",
    "pa_ctx_setup_range_short",
    "pa_ctx_setup_failed_breakout_long",
    "pa_ctx_setup_failed_breakout_short",
    "pa_ctx_range_pressure",
    "pa_ctx_structure_veto",
    "pa_ctx_premise_break_long",
    "pa_ctx_premise_break_short",
]

# Straddle-centric PA (1m); computed in pa_rules._straddle_focused_features.
PA_STRADDLE_FEATURES = [
    "pa_iv_rv_spread",
    "rv_iv_spread_z_60",
    "pa_rv_zscore",
    "pa_rv_acceleration",
    "vol_momentum_diff_60",
    "rv_iv_ratio_change_60",
    "pa_bb_width_pctile",
    "pa_atr_ratio_5_20",
    "pa_rv_term_slope",
    "pa_vol_regime_duration",
    "pa_realized_vs_expected",
    "pa_volume_zscore",
    "pa_volume_price_corr",
]

# Drop from LGBM feature list if still present in older caches (directional / OR location).
PA_FEATURES_EXCLUDE_FROM_LGBM = [
    "pa_ctx_setup_long",
    "pa_ctx_setup_short",
    "pa_ctx_follow_through_long",
    "pa_ctx_follow_through_short",
    "pa_or_position",
]

PA_STATE_FEATURES = [
    "pa_state_trend_strength",
    "pa_state_followthrough_quality",
    "pa_state_range_risk",
    "pa_state_pullback_exhaustion",
    "pa_state_breakout_failure_risk",
    "pa_state_always_in_bias",
]

REGIMES_6 = tuple(MARKET_STATE_NAMES[i] for i in range(HMM_NUM_CLASSES))

# New dual-view stack schema versions
L1A_SCHEMA_VERSION = "1.27.0"
L1B_SCHEMA_VERSION = "1.24.0"
L2_SCHEMA_VERSION = "1.48.0"
L3_SCHEMA_VERSION = "1.25.0"

# New artifact names
L1A_MODEL_FILE = "l1a_market_tcn.pt"
L1A_META_FILE = "l1a_market_tcn_meta.pkl"
L1A_OUTPUT_CACHE_FILE = "l1a_outputs.pkl"
# Written after val metrics, before full-table materialize — use L1A_MATERIALIZE_ONLY=1 to finish without retraining.
L1A_TRAIN_RESUME_FILE = "l1a_train_resume.pkl"

L1B_META_FILE = "l1b_descriptor_meta.pkl"
L1B_OUTPUT_CACHE_FILE = "l1b_outputs.pkl"
L1B_EDGE_PRED_FILE = "l1b_edge_pred.txt"
L1B_DQ_PRED_FILE = "l1b_dq_pred.txt"

L1C_SCHEMA_VERSION = "2.0.0"
L1C_MODEL_FILE = "l1c_direction.pt"
L1C_META_FILE = "l1c_direction_meta.pkl"
L1C_OUTPUT_CACHE_FILE = "l1c_outputs.pkl"

L2_GATE_FILE = "l2_trade_gate.txt"
L2_DIRECTION_FILE = "l2_direction.txt"
L2_TRADE_GATE_CALIBRATOR_FILE = "l2_trade_gate_calibrator.pkl"
L2_DIRECTION_CALIBRATOR_FILE = "l2_direction_calibrator.pkl"
L2_MFE_FILE = "l2_mfe.txt"
L2_MAE_FILE = "l2_mae.txt"
L2_RANGE_FILE = "l2_range.txt"
L2_RANGE10_FILE = "l2_range_10.txt"
L2_RANGE20_FILE = "l2_range_20.txt"
L2_TTP90_FILE = "l2_ttp90.txt"
L2_META_FILE = "l2_decision_meta.pkl"
L2_OUTPUT_CACHE_FILE = "l2_outputs.pkl"
# PyTorch unified L2 (replaces LGBM trade_gate/range/mfe/mae when L2_LEGACY_LGBM is unset)
L2_UNIFIED_MODEL_FILE = "l2_unified_l2l3.pt"
# Training resume (``L2_UNIFIED_RESUME=1``); see ``core.training.unified.train``.
L2_UNIFIED_TRAIN_CHECKPOINT_FILE = "l2_unified_train_ckpt.pt"

# VIXY 1m straddle features (aligned in L2 by time_key); disable via L2_USE_VIXY=0
VIXY_DATA_PATH = os.path.join(DATA_DIR, "VIXY.csv")
_L2_USE_VIXY_RAW = os.environ.get("L2_USE_VIXY", "1").strip().lower()
L2_USE_VIXY = _L2_USE_VIXY_RAW not in {"0", "false", "no", "off"}

# Vol-only L2 decision defaults (kept centralized; persisted into L2 meta at train time).
# ``rv_to_iv_mult``: IV proxy ≈ RV * mult when RV is 1-bar return std; common VRP shorthand RV/IV≈0.85 ⇒ mult=1/0.85.
# (Legacy key ``vrp_base_mult`` is accepted if ``rv_to_iv_mult`` is absent.)
L2_VOL_ONLY_DEFAULTS: dict[str, float | int] = {
    "long_edge_thr": 0.10,
    "short_edge_thr": 0.10,
    "iv_proxy_rv_lookback": 390,
    "iv_proxy_vixy_z_beta": 0.05,
    "implied_range_sigma_mult": 1.0,
    "rv_to_iv_mult": float(1.0 / 0.85),
}

# Optional sweep list for manual / external grid search (not read automatically).
L2_RV_TO_IV_MULT_SWEEP: tuple[float, ...] = (1.0 / 0.75, 1.0 / 0.80, 1.0 / 0.85, 1.0 / 0.90, 1.0 / 0.95)

# Straddle vol exit defaults (centralized and persisted into L3 meta).
L3_VOL_EXIT_DEFAULTS: dict[str, float | str] = {
    "label_mode": "straddle_vol",
    "short_vol_range_explosion_mult": 1.50,
    "long_vol_theta_frac": 0.08,
}

# L2 straddle dynamic cost (optional): set L2_DYNAMIC_STRADDLE_COST=1.
# Per-bar cost = base × clip(atr / rolling_quantile(atr), min_mult, max_mult), causal per symbol.
# Env: L2_DYNAMIC_COST_LOOKBACK (default 390), L2_DYNAMIC_COST_REF_QUANTILE (default 0.7),
# L2_DYNAMIC_COST_MIN_MULT / L2_DYNAMIC_COST_MAX_MULT, L2_DYNAMIC_ABS_MIN / L2_DYNAMIC_ABS_MAX.
#
# L2 straddle range-roll threshold (optional, overrides ATR-dynamic when on): L2_STRADDLE_RANGE_ROLL_COST=1
# cost[i] = percentile_p( decision_forward_range_atr[i-L:i) ) — causal via shift(1).rolling.
# Env: L2_RANGE_ROLL_LOOKBACK (1950), L2_RANGE_ROLL_PERCENTILE (50), L2_RANGE_ROLL_FLOOR/CAP (2/8).

L3_EXIT_FILE = "l3_exit.txt"
L3_VALUE_FILE = "l3_value.txt"
# Excluded from L3 value head only (exit head unchanged). Kept minimal: true trade_outcome
# label proxies. Path/hold features are *allowed* for ``remaining_value_atr`` regression
# and are re-included via ``L3_VALUE_EXTRA_ALLOWED`` when present in ``feature_cols``.
L3_VALUE_FEATURE_BLACKLIST: tuple[str, ...] = (
    "l3_cox_baseline_cumhaz_at_stop",
    "l3_cox_log_partial_hazard",
    # Value head only: raw L2 gate dominates gain; exit head still uses it.
    "l3_l2_gate_current",
)
# Always union these into the value head column set if they exist in the policy feature matrix.
L3_VALUE_EXTRA_ALLOWED: tuple[str, ...] = (
    "l3_unreal_pnl_atr",
    "l3_hold_bars",
    "l3_bars_remaining",
    "l3_live_mfe",
    "l3_drawdown_from_peak_atr",
    "l3_unreal_pnl_vel_3bar",
    "l3_unreal_pnl_frac",
    "l3_live_mae",
    "l3_live_edge",
    "l3_regret_ratio",
    "l3_bars_since_peak",
    "l3_at_new_high",
    "l3_regret_velocity",
    "l3_trade_quality_bayes",
)
L3_META_FILE = "l3_exit_meta.pkl"
PREPARED_DATASET_CACHE_FILE = "prepared_lgbm_dataset.pkl"

# New stack feature / contract names
L1A_REGIME_COLS = [f"l1a_regime_prob_{name}" for name in REGIME_NOW_PROB_COLS]
L2_ENTRY_REGIME_COLS = [f"l2_entry_regime_{idx}" for idx in range(NUM_REGIME_CLASSES)]

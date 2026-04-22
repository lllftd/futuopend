import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lgbm_models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
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
TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))
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
# Env ablation: L1B_USE_L1A_FEATURES=1, L1B_L1A_FEATURE_TIER=scalar|full; see core/trainers/l1b/l1a_bridge.py.
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
    "pa_rv_zscore",
    "pa_rv_acceleration",
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
L1A_SCHEMA_VERSION = "1.26.0"
L1B_SCHEMA_VERSION = "1.24.0"
L2_SCHEMA_VERSION = "1.46.4"
L3_SCHEMA_VERSION = "1.21.0"

# New artifact names
L1A_MODEL_FILE = "l1a_market_tcn.pt"
L1A_META_FILE = "l1a_market_tcn_meta.pkl"
L1A_OUTPUT_CACHE_FILE = "l1a_outputs.pkl"

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

# VIXY 1m straddle features (aligned in L2 by time_key); disable via L2_USE_VIXY=0
VIXY_DATA_PATH = os.path.join(DATA_DIR, "VIXY.csv")
_L2_USE_VIXY_RAW = os.environ.get("L2_USE_VIXY", "1").strip().lower()
L2_USE_VIXY = _L2_USE_VIXY_RAW not in {"0", "false", "no", "off"}

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
L3_META_FILE = "l3_exit_meta.pkl"
L3_TRAJECTORY_ENCODER_FILE = "l3_trajectory_encoder.pt"
L3_POLICY_DATASET_CACHE_FILE = "l3_policy_dataset.pkl"
L3_COX_FILE = "l3_cox_time_varying.pkl"
PREPARED_DATASET_CACHE_FILE = "prepared_lgbm_dataset.pkl"

# New stack feature / contract names
L1A_REGIME_COLS = [f"l1a_regime_prob_{name}" for name in REGIME_NOW_PROB_COLS]
L2_ENTRY_REGIME_COLS = [f"l2_entry_regime_{idx}" for idx in range(NUM_REGIME_CLASSES)]

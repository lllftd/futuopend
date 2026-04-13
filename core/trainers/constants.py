import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lgbm_models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")

NUM_REGIME_CLASSES = 6
STATE_NAMES: dict[int, str] = {
    0: "bull_conv",
    1: "bull_div",
    2: "bear_conv",
    3: "bear_div",
    4: "range_conv",
    5: "range_div",
}
REGIME_NOW_PROB_COLS = ["bull_conv", "bull_div", "bear_conv", "bear_div", "range_conv", "range_div"]
REGIME_PROB_COLS = REGIME_NOW_PROB_COLS
RANGE_REGIME_INDICES = [4, 5]

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

TRAIN_END = "2023-01-01"
CAL_END = "2023-07-01"
TEST_END = "2025-01-01"
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
    "pa_ctx_setup_long",
    "pa_ctx_setup_short",
    "pa_ctx_follow_through_long",
    "pa_ctx_follow_through_short",
    "pa_ctx_range_pressure",
    "pa_ctx_structure_veto",
    "pa_ctx_premise_break_long",
    "pa_ctx_premise_break_short",
]

REGIMES_6 = tuple(REGIME_NOW_PROB_COLS)

# New dual-view stack schema versions
L1A_SCHEMA_VERSION = "1.2.0"
L1B_SCHEMA_VERSION = "1.4.0"
L2_SCHEMA_VERSION = "1.6.0"
L3_SCHEMA_VERSION = "1.7.0"

# New artifact names
L1A_MODEL_FILE = "l1a_market_tcn.pt"
L1A_META_FILE = "l1a_market_tcn_meta.pkl"
L1A_OUTPUT_CACHE_FILE = "l1a_outputs.pkl"

L1B_META_FILE = "l1b_descriptor_meta.pkl"
L1B_OUTPUT_CACHE_FILE = "l1b_outputs.pkl"

L2_DECISION_FILE = "l2_decision.txt"
L2_GATE_FILE = "l2_trade_gate.txt"
L2_DIRECTION_FILE = "l2_direction.txt"
L2_SIZE_FILE = "l2_size.txt"
L2_MFE_FILE = "l2_mfe.txt"
L2_MAE_FILE = "l2_mae.txt"
L2_META_FILE = "l2_decision_meta.pkl"
L2_OUTPUT_CACHE_FILE = "l2_outputs.pkl"

L3_EXIT_FILE = "l3_exit.txt"
L3_VALUE_FILE = "l3_value.txt"
L3_META_FILE = "l3_exit_meta.pkl"
L3_TRAJECTORY_ENCODER_FILE = "l3_trajectory_encoder.pt"
L3_POLICY_DATASET_CACHE_FILE = "l3_policy_dataset.pkl"
PREPARED_DATASET_CACHE_FILE = "prepared_lgbm_dataset.pkl"

# New stack feature / contract names
L1A_REGIME_COLS = [f"l1a_regime_prob_{name}" for name in REGIME_NOW_PROB_COLS]
L2_ENTRY_REGIME_COLS = [f"l2_entry_regime_{idx}" for idx in range(NUM_REGIME_CLASSES)]

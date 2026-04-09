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
STATE_CLASSIFIER_FILE = "state_classifier_6c.txt"
REGIME_NOW_PROB_COLS = ["bull_conv", "bull_div", "bear_conv", "bear_div", "range_conv", "range_div"]
REGIME_PROB_COLS = REGIME_NOW_PROB_COLS
RANGE_REGIME_INDICES = [4, 5]
TCN_REGIME_FUT_PROB_COLS = ["tcn_barrier_hit_up", "tcn_barrier_hit_dn", "tcn_barrier_chop"]
MAMBA_REGIME_FUT_PROB_COLS = ["mamba_transition_same", "mamba_transition_prob"]
TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))

LGBM_EXCLUDE_PA_STRING_COLS = {
    "symbol", "time_key", "date", "market_state", "code", "name",
    "time_key_day", "market_state_shifted",
}

TRAIN_END = "2023-01-01"
CAL_END = "2023-07-01"
TEST_END = "2025-01-01"
FAST_TRAIN_MODE = os.environ.get("FAST_TRAIN", "").strip().lower() in {"1", "true", "yes"}

RNG = np.random.default_accuracy if hasattr(np.random, "default_accuracy") else np.random.RandomState(42)

QUALITY_CLASS_NAMES = ["LONG", "NEUTRAL", "CHOP", "SHORT"]
QUALITY_CLASS_ORDER = {name: i for i, name in enumerate(QUALITY_CLASS_NAMES)}
TRADABLE_CLASS_IDS = [0, 3]

EXECUTION_SIZER_GATE_FILE = "execution_sizer_gate.txt"
EXECUTION_SIZER_SIZE_FILE = "execution_sizer_size.txt"
EXECUTION_SIZER_TP_FILE = "execution_sizer_tp.txt"
EXECUTION_SIZER_SL_FILE = "execution_sizer_sl.txt"
EXECUTION_SIZER_LEGACY_V1_FILE = "execution_sizer_v1.txt"

BO_FEAT_COLS = [
    "bo_body_atr", "bo_range_atr", "bo_vol_spike", "bo_close_extremity",
    "bo_wick_imbalance", "bo_range_compress", "bo_body_growth", "bo_gap_signal",
    "bo_consec_dir", "bo_inside_prior", "bo_pressure_diff", "bo_or_dist",
    "bo_bb_width", "bo_atr_zscore",
]

L2B_OPP_X_REGIME_COLS = [f"l2b_opp_x_{r}" for r in REGIME_NOW_PROB_COLS]

LAYER3_PA_KEY_FEATURES = (
    "pa_vol_rvol", "pa_vol_momentum", "pa_bo_wick_imbalance", "pa_bo_close_extremity",
    "pa_lead_macd_hist_slope", "pa_lead_rsi_slope", "pa_bo_dist_vwap",
    "pa_struct_swing_range_atr", "pa_tr_mm_target_up", "pa_tr_mm_target_down",
    "pa_vol_exhaustion_climax", "pa_vol_zscore_20", "pa_vol_evr_ratio",
    "pa_vol_absorption_bull", "pa_vol_absorption_bear",
)

REGIMES_6 = tuple(REGIME_NOW_PROB_COLS)

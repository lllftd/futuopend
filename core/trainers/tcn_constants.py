import os
import torch

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lgbm_models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")

TCN_MEMMAP_CHUNK = 200_000
TCN_STATS_CHUNK = 100_000

TRAIN_END = "2023-01-01"
CAL_END = "2023-07-01"
TEST_END = "2025-01-01"

SEQ_LEN = 30

TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.25
SLIM_CHANNELS = [64, 64, 64, 64]

TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))

TCN_CE_WEIGHT_POWER = 0.8

TCN_MAX_EPOCHS = 120
TCN_ES_PATIENCE = 15

TCN_OOF_FOLDS = 4

STATE_NAMES = ["bull_conv", "bull_div", "bear_conv", "bear_div", "range_conv", "range_div"]
NUM_REGIME_CLASSES = len(STATE_NAMES)

STATE_CLASSIFIER_FILE = "tcn_state_classifier_6c"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

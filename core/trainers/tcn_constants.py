import os
import torch

from core.trainers.constants import CAL_END, TEST_END, TRAIN_END

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lgbm_models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")

TCN_MEMMAP_CHUNK = 50_000
TCN_STATS_CHUNK = 100_000

SEQ_LEN = max(20, int(os.environ.get("TCN_SEQ_LEN", "60")))

TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.25
SLIM_CHANNELS = [32, 32, 32]

TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "16")))

TCN_CE_WEIGHT_POWER = 0.8

TCN_MAX_EPOCHS = 120
TCN_ES_PATIENCE = 15

TCN_OOF_FOLDS = 4

STATE_NAMES = ["bull_conv", "bull_div", "bear_conv", "bear_div", "range_conv", "range_div"]
NUM_REGIME_CLASSES = len(STATE_NAMES)

STATE_CLASSIFIER_FILE = "tcn_state_classifier_6c"

def _default_torch_device() -> torch.device:
    forced = os.environ.get("TORCH_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _default_torch_device()

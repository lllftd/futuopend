import argparse
import os
import sys

import torch

from core.trainers.constants import L1A_OUTPUT_CACHE_FILE, L2_OUTPUT_CACHE_FILE
from core.trainers.data_prep import prepare_dataset as prepare_lgbm_data
from core.trainers.layer1a_market import train_l1a_market_encoder
from core.trainers.layer1b_descriptor import train_l1b_market_descriptor
from core.trainers.layer2_decision import train_l2_trade_decision
from core.trainers.layer3_exit import train_l3_exit_manager
from core.trainers.lgbm_utils import configure_compute_threads, _lgbm_n_jobs
from core.trainers.stack_v2_common import load_output_cache


class Logger:
    """Redirects stdout and stderr to both the terminal and a specified log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal.write(message)
        # tqdm should use core.trainers.lgbm_utils.TQDM_FILE (sys.__stderr__) so bars never hit this tee.
        # Fallback: redraw lines use '\r'; skip those so logs stay readable if something writes to sys.stderr.
        if message and "\r" in message:
            return
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()

# Fixed log filenames (overwrite each run): only these five under logs/
_LAYER_LOG_FILES = {
    "layer1a": "layer1a.log",
    "layer1b": "layer1b.log",
    "layer2": "layer2.log",
    "layer3": "layer3.log",
}


def setup_logger(layer_key: str):
    """Tee stdout/stderr to terminal and exactly one overwrite log per stage."""
    if layer_key not in _LAYER_LOG_FILES:
        raise ValueError(f"setup_logger: unknown layer_key={layer_key!r}; expected one of {sorted(_LAYER_LOG_FILES)}")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, _LAYER_LOG_FILES[layer_key])
    print(f"\n>> Layer log (overwrite): {log_file}")
    return Logger(log_file)


def run_lgbm_layers(start_from: str = "layer1"):
    configure_compute_threads()
    n_th = torch.get_num_threads()
    print(f"\n  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}")

    print("=" * 70)
    print("  Dual-View Stack: L1a + L1b + L2 + L3")
    print("=" * 70)

    print(f"\n[*] Preparing LGBM dataset...")
    df, feat_cols = prepare_lgbm_data(["QQQ", "SPY"])

    if start_from == "layer3":
        print("\n[*] L3-only: loading cached L1a / L2 outputs from lgbm_models/ ...")
        l1a_outputs = load_output_cache(L1A_OUTPUT_CACHE_FILE)
        l2_outputs = load_output_cache(L2_OUTPUT_CACHE_FILE)
        logger = setup_logger("layer3")
        try:
            print("\n[3] --- Training L3 (Exit Manager) ---")
            train_l3_exit_manager(df, l1a_outputs, l2_outputs)
        finally:
            logger.close()
        print("\n" + "=" * 70)
        print("  DONE — L3 models saved in lgbm_models/")
        print("=" * 70)
        return

    logger = setup_logger("layer1a")
    try:
        print("\n[1a] --- Training L1a (Sequence Market Encoder) ---")
        l1a_bundle = train_l1a_market_encoder(df, feat_cols)
    finally:
        logger.close()

    logger = setup_logger("layer1b")
    try:
        print("\n[1b] --- Training L1b (Tabular Market Descriptor) ---")
        l1b_bundle = train_l1b_market_descriptor(df, feat_cols)
    finally:
        logger.close()

    logger = setup_logger("layer2")
    try:
        print("\n[2] --- Training L2 (Trade Decision) ---")
        l2_bundle = train_l2_trade_decision(df, l1a_bundle.outputs, l1b_bundle.outputs)
    finally:
        logger.close()

    logger = setup_logger("layer3")
    try:
        print("\n[3] --- Training L3 (Exit Manager) ---")
        train_l3_exit_manager(df, l1a_bundle.outputs, l2_bundle.outputs)
    finally:
        logger.close()

    print("\n" + "=" * 70)
    print("  DONE — Models saved in lgbm_models/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dual-view training pipeline (L1a/L1b/L2/L3)")
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["layer1", "layer1a", "layer1b", "layer2", "layer3"],
        default="layer1",
        help="layer3: prepare data + train L3 only (requires l1a_outputs.pkl & l2_outputs.pkl in lgbm_models/). "
        "Other values still run the full pipeline (CLI compatibility).",
    )
    args = parser.parse_args()
    run_lgbm_layers(start_from=args.start_from)


if __name__ == "__main__":
    main()

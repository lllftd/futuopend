import argparse
import os
import pickle
import sys
import torch
import lightgbm as lgb
import numpy as np
import datetime

# Layer 1 (TCN) Imports
from core.trainers.tcn_constants import (
    DEVICE,
    MODEL_DIR as TCN_MODEL_DIR,
    STATE_CLASSIFIER_FILE as TCN_STATE_FILE,
    SEQ_LEN,
    SLIM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    TCN_BOTTLENECK_DIM,
)
from core.trainers.tcn_data_prep import prepare_data as prepare_tcn_data
from core.trainers.layer1_tcn import train_tcn
from core.trainers.layer1_mamba import train_mamba, MAMBA_STATE_CLASSIFIER_FILE

# Layer 2-4 (LGBM) Imports
from core.trainers.constants import (
    MODEL_DIR,
    STATE_CLASSIFIER_FILE,
    EXECUTION_SIZER_GATE_FILE,
    EXECUTION_SIZER_SIZE_FILE,
    EXECUTION_SIZER_TP_FILE,
    EXECUTION_SIZER_SL_FILE,
    TCN_REGIME_FUT_PROB_COLS,
)
from core.trainers.lgbm_utils import configure_compute_threads, _lgbm_n_jobs
from core.trainers.data_prep import prepare_dataset as prepare_lgbm_data
from core.trainers.layer2a_regime import train_regime_classifier
from core.trainers.layer2b_quality import train_trade_quality_classifier
from core.trainers.layer3_sizer import train_execution_sizer
from core.trainers.layer4_exit import train_exit_manager_layer4

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

def setup_logger(layer_name):
    """Sets up a logger for the given layer."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{layer_name}.log")
    print(f"\n>> Redirecting output to: {log_file}")
    return Logger(log_file)

def run_layer1_tcn():
    logger = setup_logger("layer1")
    try:
        print("\n" + "=" * 70)
        print("  [1] TCN — Future Transition Signal (binary, +15 bars)")
        print("=" * 70)
        print(f"  Device: {DEVICE}  (override: TORCH_DEVICE=cuda:0 | mps | cpu)")

        mm = None
        mmap_path = ""
        try:
            mm, y, train_idx, cal_idx, test_idx, norm_stats = prepare_tcn_data()
            mmap_path = str(norm_stats.get("memmap_path", ""))
            n_features = int(mm.shape[2])
            print("  Starting optimization (batches stream from memmap)…", flush=True)

            tcn_model = train_tcn(
                mm, y, train_idx, cal_idx, test_idx, n_features,
                norm_stats["ts"], norm_stats["syms"]
            )

            os.makedirs(TCN_MODEL_DIR, exist_ok=True)
            torch.save(tcn_model.state_dict(), os.path.join(TCN_MODEL_DIR, TCN_STATE_FILE))

            meta = {
                "mean": norm_stats["mean"],
                "std": norm_stats["std"],
                "feat_cols": norm_stats["feat_cols"],
                "seq_len": int(norm_stats.get("seq_len_used", SEQ_LEN)),
                "input_size": n_features,
                "num_channels": list(SLIM_CHANNELS),
                "kernel_size": TCN_KERNEL_SIZE,
                "dropout": TCN_DROPOUT,
                "bottleneck_dim": TCN_BOTTLENECK_DIM,
                "is_dual_head": False,
                "tcn_head": "regime_transition_15_bottleneck",
                "num_regime_classes": len(TCN_REGIME_FUT_PROB_COLS),
                "state_names": ["same", "transition"],
            }
            with open(os.path.join(TCN_MODEL_DIR, "tcn_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            print(f"\n  TCN binary future-transition model saved → {TCN_MODEL_DIR}/{TCN_STATE_FILE}")
            print(f"  Meta saved → {TCN_MODEL_DIR}/tcn_meta.pkl")
        finally:
            if mm is not None:
                try:
                    mm._mmap.close()
                except Exception:
                    pass
                del mm
            if mmap_path and os.path.isfile(mmap_path) and not os.environ.get("TCN_KEEP_MEMMAP"):
                try:
                    os.remove(mmap_path)
                    print(f"  Removed temp memmap {mmap_path}", flush=True)
                except OSError as exc:
                    print(f"  Warning: could not remove memmap {mmap_path}: {exc}", flush=True)
    finally:
        logger.close()

def run_layer1_mamba():
    logger = setup_logger("layer1_mamba")
    try:
        print("\n" + "=" * 70)
        print("  [1b] Mamba — Future Transition Signal (binary, +15 bars)")
        print("=" * 70)
        print(f"  Device: {DEVICE}  (override: TORCH_DEVICE=cuda:0 | mps | cpu)")

        mm = None
        mmap_path = ""
        try:
            mm, y, train_idx, cal_idx, test_idx, norm_stats = prepare_tcn_data()
            mmap_path = str(norm_stats.get("memmap_path", ""))
            n_features = int(mm.shape[2])
            print("  Starting optimization (batches stream from memmap)…", flush=True)

            mamba_model = train_mamba(
                mm, y, train_idx, cal_idx, test_idx, n_features,
                norm_stats["ts"], norm_stats["syms"]
            )

            os.makedirs(TCN_MODEL_DIR, exist_ok=True)
            torch.save(mamba_model.state_dict(), os.path.join(TCN_MODEL_DIR, MAMBA_STATE_CLASSIFIER_FILE))

            meta = {
                "mean": norm_stats["mean"],
                "std": norm_stats["std"],
                "feat_cols": norm_stats["feat_cols"],
                "seq_len": int(norm_stats.get("seq_len_used", SEQ_LEN)),
                "input_size": n_features,
                "d_model": 64,
                "n_layers": 4,
                "dropout": TCN_DROPOUT,
                "bottleneck_dim": TCN_BOTTLENECK_DIM,
                "is_dual_head": False,
                "mamba_head": "regime_transition_15_bottleneck",
                "num_regime_classes": len(TCN_REGIME_FUT_PROB_COLS),
                "state_names": ["same", "transition"],
            }
            with open(os.path.join(TCN_MODEL_DIR, "mamba_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            print(f"\n  Mamba binary future-transition model saved → {TCN_MODEL_DIR}/{MAMBA_STATE_CLASSIFIER_FILE}")
            print(f"  Meta saved → {TCN_MODEL_DIR}/mamba_meta.pkl")
        finally:
            if mm is not None:
                try:
                    mm._mmap.close()
                except Exception:
                    pass
                del mm
            if mmap_path and os.path.isfile(mmap_path) and not os.environ.get("TCN_KEEP_MEMMAP"):
                try:
                    os.remove(mmap_path)
                    print(f"  Removed temp memmap {mmap_path}", flush=True)
                except OSError as exc:
                    print(f"  Warning: could not remove memmap {mmap_path}: {exc}", flush=True)
    finally:
        logger.close()


def load_layer2a_artifacts():
    model_path = os.path.join(MODEL_DIR, STATE_CLASSIFIER_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Layer 2a model at {model_path}. Cannot skip Layer 2a.")
        
    regime_model = lgb.Booster(model_file=model_path)
    
    cal_path = os.path.join(MODEL_DIR, "state_calibrators.pkl")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Missing Layer 2a calibrators at {cal_path}. Cannot skip Layer 2a.")
        
    with open(cal_path, "rb") as f:
        meta = pickle.load(f)
        regime_cal = meta["calibrators"]
        thr_cp = meta.get("thr_cp", 0.95)
    
    return regime_model, regime_cal, thr_cp

def load_layer2b_artifacts():
    model_path = os.path.join(MODEL_DIR, "trade_gate_step1.txt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Layer 2b model at {model_path}. Cannot skip Layer 2b.")
    
    tq_model = lgb.Booster(model_file=model_path)
    return tq_model

def run_lgbm_layers(start_from="layer2a"):
    configure_compute_threads()
    n_th = torch.get_num_threads()
    print(f"\n  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}")

    print("=" * 70)
    print("  LightGBM Layer 2 (regime/trade) + Layer 3 + Layer 4")
    print("=" * 70)

    print(f"\n[*] Preparing LGBM dataset...")
    df, feat_cols = prepare_lgbm_data(["QQQ", "SPY"])

    # Layer 2a
    if start_from in ["layer1", "layer2a"]:
        logger = setup_logger("layer2a")
        try:
            print("\n[2a] --- Training Layer 2a (Regime) ---")
            regime_model, regime_cal, regime_imp, thr_cp = train_regime_classifier(df, feat_cols)
        finally:
            logger.close()
    else:
        print("\n[2a] --- Skipping Layer 2a, loading from disk ---")
        regime_model, regime_cal, thr_cp = load_layer2a_artifacts()

    # Layer 2b
    if start_from in ["layer1", "layer2a", "layer2b"]:
        logger = setup_logger("layer2b")
        try:
            print("\n[2b] --- Training Layer 2b (Trade Stack) ---")
            tq_model, tq_meta, tq_imp = train_trade_quality_classifier(df, feat_cols, regime_model, regime_cal, thr_cp)
        finally:
            logger.close()
    else:
        print("\n[2b] --- Skipping Layer 2b, loading from disk ---")
        tq_model = load_layer2b_artifacts()

    # Layer 3
    if start_from in ["layer1", "layer2a", "layer2b", "layer3"]:
        logger = setup_logger("layer3")
        try:
            print("\n[3] --- Training Layer 3 (Execution Sizer) ---")
            train_execution_sizer(df, feat_cols, regime_model, regime_cal, tq_model, thr_cp)
        finally:
            logger.close()
    else:
        print("\n[3] --- Skipping Layer 3 ---")

    # Layer 4
    if start_from in ["layer1", "layer2a", "layer2b", "layer3", "layer4"]:
        logger = setup_logger("layer4")
        try:
            print("\n[4] --- Training Layer 4 (Exit Manager) ---")
            train_exit_manager_layer4(df, feat_cols, regime_model, regime_cal, tq_model, thr_cp)
        finally:
            logger.close()
        
    print("\n" + "=" * 70)
    print("  DONE — Models saved in lgbm_models/")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline (TCN + LGBM)")
    parser.add_argument(
        "--start-from", 
        type=str, 
        choices=["layer1", "layer2a", "layer2b", "layer3", "layer4"], 
        default="layer1", 
        help="Skip earlier layers and load their models from disk for hot-starting."
    )
    args = parser.parse_args()
    
    if args.start_from == "layer1":
        run_layer1_tcn()
        run_layer1_mamba()
        run_lgbm_layers("layer2a")
    else:
        run_lgbm_layers(args.start_from)

if __name__ == "__main__":
    main()

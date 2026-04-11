import argparse
import os
import pickle
import sys
import torch
import lightgbm as lgb
import numpy as np

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
from core.trainers.layer1a_tcn import train_tcn, TCN_MIN_ATTENTION_SEQ_LEN, TCN_READOUT_TYPE
from core.trainers.layer1b_mamba import train_mamba, MAMBA_STATE_CLASSIFIER_FILE

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

def _experimental_mamba_enabled() -> bool:
    return os.environ.get("ENABLE_EXPERIMENTAL_MAMBA", "").strip().lower() in {"1", "true", "yes"}


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
        # tqdm (and similar) redraw the current line with '\r'; keep those out of layer*.log files
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
    "layer2a": "layer2a.log",
    "layer2b": "layer2b.log",
    "layer3": "layer3.log",
    "layer4": "layer4.log",
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

def run_layer1a_tcn():
    logger = setup_logger("layer1a")
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
                "readout_type": TCN_READOUT_TYPE,
                "min_attention_seq_len": TCN_MIN_ATTENTION_SEQ_LEN,
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

def run_layer1b_mamba():
    if not _experimental_mamba_enabled():
        raise RuntimeError(
            "Layer 1b Mamba is experimental and disabled by default. "
            "Set ENABLE_EXPERIMENTAL_MAMBA=1 to run it explicitly."
        )
    logger = setup_logger("layer1a")
    try:
        print("\n" + "=" * 70)
        print("  [1b] Experimental Mamba — Future Transition Signal")
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

            print(f"\n  Experimental Mamba model saved → {TCN_MODEL_DIR}/{MAMBA_STATE_CLASSIFIER_FILE}")
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
        obj = pickle.load(f)
        
        if isinstance(obj, dict):
            regime_cal = obj["calibrators"]
            thr_cp = obj.get("thr_cp", 0.95)
        else:
            # Fallback for old format
            regime_cal = obj
            thr_cp = float(os.environ.get("LAYER2A_THR_CP", "0.95"))
            print(
                f"\n  [WARN] Legacy state_calibrators.pkl format detected. "
                f"Using LAYER2A_THR_CP={thr_cp:.2f}. Please re-run layer2a to embed thr_cp.",
                flush=True
            )
    
    return regime_model, regime_cal, thr_cp

def load_layer2b_artifacts():
    with open(os.path.join(MODEL_DIR, "trade_quality_meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    regb = meta.get("regression_gate", {})
    step1_regression_bundle = {
        "thr_vec": np.array(regb.get("thr_vec", [])),
        "soft_opp_threshold": float(regb.get("soft_opp_threshold", 0.0)),
        "gating_mode": regb.get("gating_mode", "hard_route"),
    }
    for regime in regb.get("groups", []):
        mfe_file = regb["model_files"].get(f"{regime}_mfe", "")
        mae_file = regb["model_files"].get(f"{regime}_mae", "")
        if mfe_file and os.path.exists(os.path.join(MODEL_DIR, mfe_file)):
            step1_regression_bundle[f"{regime}_mfe"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, mfe_file))
        if mae_file and os.path.exists(os.path.join(MODEL_DIR, mae_file)):
            step1_regression_bundle[f"{regime}_mae"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, mae_file))

    model_files = meta.get("model_files", {})
    out = {
        "l2b_schema": int(meta.get("l2b_schema", 1)),
        "step1_regression": step1_regression_bundle,
        "thresholds": meta["hierarchy_thresholds"],
        "feature_cols": meta["feature_cols"],
        "gate_feature_cols": meta.get("gate_feature_cols"),
        "gate_architecture": meta.get("gate_architecture", "legacy_dual_binary"),
    }
    if "step1_has_trade" in model_files and "step2_direction" in model_files:
        out["step1_has_trade"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, model_files["step1_has_trade"]))
        out["step2_direction"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, model_files["step2_direction"]))
        return out

    long_name = model_files.get("step1_long", "trade_gate_long.txt")
    short_name = model_files.get("step1_short", "trade_gate_short.txt")
    long_path = os.path.join(MODEL_DIR, long_name)
    if not os.path.exists(long_path):
        raise FileNotFoundError(f"Missing Layer 2b model at {long_path}. Cannot skip Layer 2b.")
    out["step1_long"] = lgb.Booster(model_file=long_path)
    out["step1_short"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, short_name))
    return out

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
    parser = argparse.ArgumentParser(description="Unified Training Pipeline (TCN + LGBM; Mamba is experimental)")
    parser.add_argument(
        "--start-from", 
        type=str, 
        choices=["layer1", "layer1a", "layer1b", "layer2", "layer2a", "layer2b", "layer3", "layer4"],
        default="layer1",
        help="Skip earlier layers. layer2 is an alias for layer2a. layer1b is experimental and requires ENABLE_EXPERIMENTAL_MAMBA=1.",
    )
    args = parser.parse_args()
    start = "layer2a" if args.start_from == "layer2" else args.start_from

    if start == "layer1" or start == "layer1a":
        run_layer1a_tcn()
    if start == "layer1":
        print("\n  [1b] --- Skipping Layer 1b Mamba (experimental module; disabled by default) ---")
    elif start == "layer1b":
        run_layer1b_mamba()

    if start in ["layer1", "layer1a", "layer1b"]:
        run_lgbm_layers("layer2a")
    else:
        run_lgbm_layers(start)

if __name__ == "__main__":
    main()

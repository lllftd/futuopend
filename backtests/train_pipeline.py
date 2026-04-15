import argparse
import os
import pickle
import sys

import torch

from core.trainers.constants import (
    L1A_OUTPUT_CACHE_FILE,
    L1B_OUTPUT_CACHE_FILE,
    L1C_OUTPUT_CACHE_FILE,
    L2_OUTPUT_CACHE_FILE,
    MODEL_DIR,
    PREPARED_DATASET_CACHE_FILE,
)
from core.trainers.data_prep import prepare_dataset as prepare_lgbm_data
from core.trainers.layer1a_market import (
    infer_l1a_market_encoder,
    load_l1a_market_encoder,
    train_l1a_market_encoder,
)
from core.trainers.layer1b_descriptor import (
    infer_l1b_market_descriptor,
    load_l1b_market_descriptor,
    train_l1b_market_descriptor,
)
from core.trainers.layer2_decision import infer_l2_trade_decision, load_l2_trade_decision, train_l2_trade_decision
from core.trainers.l1c.train import train_l1c_direction
from core.trainers.layer3_exit import train_l3_exit_manager
from core.trainers.lgbm_utils import configure_compute_threads, _lgbm_n_jobs
from core.trainers.stack_v2_common import load_output_cache, save_output_cache


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
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()

# Fixed layer log filenames (overwrite each run) under logs/
_LAYER_LOG_FILES = {
    "layer1a": "layer1a.log",
    "layer1b": "layer1b.log",
    "layer1c": "layer1c.log",
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
    lg = Logger(log_file)
    print(f"\n>> Layer log (overwrite): {log_file}")
    return lg


def _prepared_dataset_cache_path() -> str:
    return os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE)


def _save_prepared_dataset_cache(df, feat_cols) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = _prepared_dataset_cache_path()
    with open(path, "wb") as f:
        pickle.dump({"df": df, "feat_cols": list(feat_cols)}, f)
    return path


def _load_prepared_dataset_cache():
    path = _prepared_dataset_cache_path()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "df" not in obj or "feat_cols" not in obj:
        raise TypeError(f"Prepared dataset cache {path} is malformed.")
    return obj["df"], list(obj["feat_cols"])


def _prepare_or_load_lgbm_dataset(symbols: list[str], *, prefer_cache: bool):
    cache_path = _prepared_dataset_cache_path()
    force_rebuild = os.environ.get("PREPARED_DATASET_CACHE_REBUILD", "").strip().lower() in {"1", "true", "yes"}
    if prefer_cache and not force_rebuild and os.path.exists(cache_path):
        print(f"\n[*] Loading prepared dataset cache from {cache_path} ...")
        return _load_prepared_dataset_cache()
    print(f"\n[*] Preparing LGBM dataset...")
    df, feat_cols = prepare_lgbm_data(symbols)
    saved = _save_prepared_dataset_cache(df, feat_cols)
    print(f"[*] Prepared dataset cache saved -> {saved}")
    return df, feat_cols


def _artifact_inferred_l1a_outputs(df):
    model, meta = load_l1a_market_encoder()
    outputs = infer_l1a_market_encoder(model, meta, df)
    save_output_cache(outputs, L1A_OUTPUT_CACHE_FILE)
    return outputs


def _artifact_inferred_l1b_outputs(df):
    models, meta = load_l1b_market_descriptor()
    outputs = infer_l1b_market_descriptor(models, meta, df)
    save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    return outputs


def _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs, l1c_outputs=None):
    models, meta = load_l2_trade_decision()
    outputs = infer_l2_trade_decision(models, meta, df, l1a_outputs, l1b_outputs, l1c_outputs)
    save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    return outputs


def _resolve_l1c_outputs_for_pipeline(df, feat_cols: list[str], sf: str):
    """Train L1c after L1a/L1b when running those stages; otherwise load cache if present."""
    import os

    skip = os.environ.get("L1C_SKIP_PIPELINE", "").strip().lower() in {"1", "true", "yes"}
    cache_path = os.path.join(MODEL_DIR, L1C_OUTPUT_CACHE_FILE)
    if skip:
        if os.path.exists(cache_path):
            print(f"\n[*] L1C_SKIP_PIPELINE: loading {L1C_OUTPUT_CACHE_FILE} ...")
            return load_output_cache(L1C_OUTPUT_CACHE_FILE)
        print("\n[*] L1C_SKIP_PIPELINE: no L1c cache; L2 will run without l1c_* features.")
        return None
    if sf in ("layer1a", "layer1b"):
        logger = setup_logger("layer1c")
        try:
            print("\n[1c] --- Training L1c (causal direction) ---")
            train_l1c_direction(df, feat_cols)
        finally:
            logger.close()
        return load_output_cache(L1C_OUTPUT_CACHE_FILE)
    if os.path.exists(cache_path):
        print(f"\n[*] start-from={sf}: loading L1c outputs from {L1C_OUTPUT_CACHE_FILE} ...")
        return load_output_cache(L1C_OUTPUT_CACHE_FILE)
    print("\n[*] No L1c output cache; training L2 without l1c_* features (train L1c separately if needed).")
    return None


def run_lgbm_layers(start_from: str = "layer1"):
    configure_compute_threads()
    n_th = torch.get_num_threads()
    print(f"\n  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}")

    print("=" * 70)
    print("  Dual-View Stack: L1a + L1b + L2 + L3")
    print("=" * 70)

    sf = start_from.strip().lower()
    if sf == "layer1":
        sf = "layer1a"

    if sf == "layer1c":
        df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=True)
        logger = setup_logger("layer1c")
        try:
            print("\n[1c] --- Training L1c (causal direction) only ---")
            train_l1c_direction(df, feat_cols)
        finally:
            logger.close()
        print("\n" + "=" * 70)
        print("  DONE — L1c artifacts saved in lgbm_models/")
        print("=" * 70)
        return

    df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=(sf != "layer1a"))

    if sf == "layer3":
        print("\n[*] start-from=layer3: load L1a + L2 output caches, train L3 only.")
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

    if sf == "layer1a":
        logger = setup_logger("layer1a")
        try:
            print("\n[1a] --- Training L1a (Sequence Market Encoder) ---")
            train_l1a_market_encoder(df, feat_cols)
        finally:
            logger.close()
        print("\n[*] Recomputing downstream-facing L1a outputs from frozen artifact ...")
        l1a_outputs = _artifact_inferred_l1a_outputs(df)
    else:
        print(f"\n[*] start-from={sf}: loading L1a outputs from {L1A_OUTPUT_CACHE_FILE} ...")
        l1a_outputs = load_output_cache(L1A_OUTPUT_CACHE_FILE)

    if sf in ("layer1a", "layer1b"):
        logger = setup_logger("layer1b")
        try:
            print("\n[1b] --- Training L1b (Tabular Market Descriptor) ---")
            train_l1b_market_descriptor(df, feat_cols)
        finally:
            logger.close()
        print("\n[*] Recomputing downstream-facing L1b outputs from frozen artifact ...")
        l1b_outputs = _artifact_inferred_l1b_outputs(df)
    else:
        print(f"\n[*] start-from={sf}: loading L1b outputs from {L1B_OUTPUT_CACHE_FILE} ...")
        l1b_outputs = load_output_cache(L1B_OUTPUT_CACHE_FILE)

    l1c_outputs = _resolve_l1c_outputs_for_pipeline(df, feat_cols, sf)

    logger = setup_logger("layer2")
    try:
        print("\n[2] --- Training L2 (Trade Decision) ---")
        train_l2_trade_decision(df, l1a_outputs, l1b_outputs, l1c_outputs)
    finally:
        logger.close()
    print("\n[*] Recomputing downstream-facing L2 outputs from frozen artifact ...")
    l2_outputs = _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs, l1c_outputs)

    logger = setup_logger("layer3")
    try:
        print("\n[3] --- Training L3 (Exit Manager) ---")
        train_l3_exit_manager(df, l1a_outputs, l2_outputs)
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
        choices=["layer1", "layer1a", "layer1b", "layer1c", "layer2", "layer3"],
        default="layer1",
        help="layer1/layer1a: L1a→L1b→L1c→L2→L3 (set L1C_SKIP_PIPELINE=1 to skip L1c train). "
        "layer1b: needs l1a_outputs.pkl, then L1b→L1c→L2→L3. layer1c: prepared cache + L1c only. "
        "layer2: needs l1a/l1b caches; loads l1c_outputs.pkl if present. "
        "layer3: needs l1a_outputs.pkl + l2_outputs.pkl, L3 only.",
    )
    args = parser.parse_args()
    run_lgbm_layers(start_from=args.start_from)


if __name__ == "__main__":
    main()

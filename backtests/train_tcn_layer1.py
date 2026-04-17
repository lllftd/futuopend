"""Train Layer-1 TCN (triple-barrier PAStateTCN) and write checkpoints required by data_prep / train_pipeline.

Produces under lgbm_models/:
  - tcn_meta.pkl
  - tcn_state_classifier_6c  (weights, no extension — matches tcn_constants.STATE_CLASSIFIER_FILE)

Run from repo root:
  PYTHONPATH=. python backtests/train_tcn_layer1.py

Optional env (see core/trainers/tcn_constants.py, l1a/tcn.py):
  TCN_MAX_EPOCHS, TCN_ES_PATIENCE, TCN_OOF_FOLDS, TCN_SKIP_OOF, TCN_DATASET_IN_RAM, etc.
"""
from __future__ import annotations

import os
import pickle
import sys

import torch

from core.trainers.l1a.tcn import (
    TCN_HEAD_NUM_CLASSES,
    TCN_MIN_ATTENTION_SEQ_LEN,
    TCN_READOUT_TYPE,
    train_tcn,
)
from core.trainers.lgbm_utils import configure_compute_threads
from core.trainers.tcn_constants import (
    MODEL_DIR,
    SEQ_LEN,
    SLIM_CHANNELS,
    STATE_CLASSIFIER_FILE,
    TCN_BOTTLENECK_DIM,
    TCN_KERNEL_SIZE,
)
from core.trainers.tcn_data_prep import prepare_data


def main() -> None:
    configure_compute_threads()
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  MODEL_DIR={MODEL_DIR}", flush=True)
    print(f"  torch device check: cuda={torch.cuda.is_available()}", flush=True)

    mm, y, train_idx, cal_idx, test_idx, norm_stats = prepare_data()
    feat_cols = norm_stats["feat_cols"]
    n_features = len(feat_cols)
    seq_len_used = int(norm_stats["seq_len_used"])
    ts = norm_stats["ts"]
    syms = norm_stats["syms"]

    final_model = train_tcn(mm, y, train_idx, cal_idx, test_idx, n_features, ts, syms)

    meta = {
        "feat_cols": list(feat_cols),
        "seq_len": seq_len_used,
        "input_size": int(n_features),
        "bottleneck_dim": int(TCN_BOTTLENECK_DIM),
        "num_regime_classes": int(TCN_HEAD_NUM_CLASSES),
        "num_channels": list(SLIM_CHANNELS),
        "kernel_size": int(TCN_KERNEL_SIZE),
        "readout_type": str(TCN_READOUT_TYPE),
        "min_attention_seq_len": int(TCN_MIN_ATTENTION_SEQ_LEN),
        "mean": norm_stats["mean"],
        "std": norm_stats["std"],
    }
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    weights_path = os.path.join(MODEL_DIR, STATE_CLASSIFIER_FILE)
    torch.save(final_model.state_dict(), weights_path)
    print(f"\n[*] Wrote {meta_path}", flush=True)
    print(f"[*] Wrote {weights_path}", flush=True)
    print("DONE — re-run train_pipeline / train_layer1a_only.", flush=True)

    mmap_path = norm_stats.get("memmap_path")
    if isinstance(mmap_path, str) and mmap_path and os.path.isfile(mmap_path):
        try:
            os.remove(mmap_path)
            print(f"[*] Removed temp memmap {mmap_path}", flush=True)
        except OSError as exc:
            print(f"[*] (warn) could not remove memmap: {exc}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

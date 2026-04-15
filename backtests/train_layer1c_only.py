"""Train L1c only; writes l1c_direction.pt, meta, and l1c_outputs.pkl.

Formal run (full epochs): do not set FAST_TRAIN.
Smoke: FAST_TRAIN=1 PYTHONPATH=. python3 backtests/train_layer1c_only.py
"""
from __future__ import annotations

import torch

from backtests.train_pipeline import _prepare_or_load_lgbm_dataset, setup_logger
from core.trainers.l1c.train import train_l1c_direction
from core.trainers.lgbm_utils import configure_compute_threads


def main() -> None:
    configure_compute_threads()
    logger = setup_logger("layer1c")
    try:
        print(f"Thread budget: PyTorch intra-op={torch.get_num_threads()}")
        df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=True)
        print("\n[1c] --- Training L1c (causal direction) only ---\n")
        train_l1c_direction(df, feat_cols)
    finally:
        logger.close()
    print("\nDONE — L1c only. Artifacts in lgbm_models/")


if __name__ == "__main__":
    main()

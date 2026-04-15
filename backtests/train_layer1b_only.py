"""Train L1b only (then refresh l1b_outputs.pkl). Run: PYTHONPATH=. python3 backtests/train_layer1b_only.py"""
from __future__ import annotations

import torch

from backtests.train_pipeline import (
    _artifact_inferred_l1b_outputs,
    _prepare_or_load_lgbm_dataset,
    setup_logger,
)
from core.trainers.l1b import train_l1b_market_descriptor
from core.trainers.lgbm_utils import configure_compute_threads


def main() -> None:
    configure_compute_threads()
    print(f"Thread budget: PyTorch intra-op={torch.get_num_threads()}")
    df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=True)
    logger = setup_logger("layer1b")
    try:
        print("\n[1b] --- Training L1b (Tabular Market Descriptor) only ---\n")
        train_l1b_market_descriptor(df, feat_cols)
    finally:
        logger.close()
    print("\n[*] Recomputing downstream-facing L1b outputs from frozen artifact ...")
    _artifact_inferred_l1b_outputs(df)
    print("\nDONE — L1b only. Artifacts in lgbm_models/")


if __name__ == "__main__":
    main()

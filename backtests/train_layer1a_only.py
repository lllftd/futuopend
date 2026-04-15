"""Train L1a only (then refresh l1a_outputs.pkl from checkpoint). Run: PYTHONPATH=. python3 backtests/train_layer1a_only.py"""
from __future__ import annotations

import torch

from backtests.train_pipeline import (
    _artifact_inferred_l1a_outputs,
    _prepare_or_load_lgbm_dataset,
    setup_logger,
)
from core.trainers.l1a import train_l1a_market_encoder
from core.trainers.lgbm_utils import configure_compute_threads


def main() -> None:
    configure_compute_threads()
    print(f"Thread budget: PyTorch intra-op={torch.get_num_threads()}")
    df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=False)
    logger = setup_logger("layer1a")
    try:
        print("\n[1a] --- Training L1a (Sequence Market Encoder) only ---\n")
        train_l1a_market_encoder(df, feat_cols)
    finally:
        logger.close()
    print("\n[*] Recomputing downstream-facing L1a outputs from frozen artifact ...")
    _artifact_inferred_l1a_outputs(df)
    print("\nDONE — L1a only. Artifacts in lgbm_models/")


if __name__ == "__main__":
    main()

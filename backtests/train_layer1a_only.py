"""Train L1a only (then refresh l1a_outputs.pkl from checkpoint). Run: PYTHONPATH=. python3 backtests/train_layer1a_only.py"""
from __future__ import annotations

import os

import torch

from backtests.train_pipeline import (
    _artifact_inferred_l1a_outputs,
    _prepare_or_load_lgbm_dataset,
    setup_logger,
)
from core.trainers.l1a import train_l1a_market_encoder
from core.trainers.lgbm_utils import configure_training_runtime


def main() -> None:
    configure_training_runtime()
    try:
        inter = torch.get_num_interop_threads()
    except RuntimeError:
        inter = -1
    print(
        f"Thread budget: PyTorch intra-op={torch.get_num_threads()} "
        f"inter-op={inter}  (set TORCH_CPU_THREADS)",
        flush=True,
    )
    use_prep = os.environ.get("LAYER1A_USE_PREPARED_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    df, feat_cols = _prepare_or_load_lgbm_dataset(
        ["QQQ", "SPY"],
        prefer_cache=use_prep,
        layer1a_frozen_prep_hint=use_prep,
    )
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

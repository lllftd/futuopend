"""Train L1b only (then refresh l1b_outputs.pkl). Run: PYTHONPATH=. python3 backtests/train_layer1b_only.py"""
from __future__ import annotations

import torch

from backtests.train_pipeline import (
    _artifact_inferred_l1b_outputs,
    _prepare_df_and_feat_for_l1b_train,
    _prepare_or_load_lgbm_dataset,
    setup_logger,
)
from core.training.common.constants import L1A_OUTPUT_CACHE_FILE
from core.training.l1b import train_l1b_market_descriptor
from core.training.l1b.l1a_bridge import l1b_l1a_feature_tier, l1b_l1a_inputs_enabled
from core.training.common.stack_v2_common import load_output_cache
from core.training.common.lgbm_utils import configure_training_runtime


def main() -> None:
    configure_training_runtime()
    print(f"Thread budget: PyTorch intra-op={torch.get_num_threads()}")
    df, feat_cols = _prepare_or_load_lgbm_dataset(["QQQ", "SPY"], prefer_cache=True)
    l1a_outputs = (
        load_output_cache(L1A_OUTPUT_CACHE_FILE)
        if (l1b_l1a_inputs_enabled() and l1b_l1a_feature_tier() != "none")
        else None
    )
    df, feat_cols = _prepare_df_and_feat_for_l1b_train(df, feat_cols, l1a_outputs)
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

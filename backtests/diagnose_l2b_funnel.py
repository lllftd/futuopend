"""
One-off funnel diagnostic for L2b hybrid-ratio labels.

From repo root:
  python -m backtests.diagnose_l2b_funnel
"""
from __future__ import annotations

from core.trainers.constants import TRAIN_END
from core.trainers.data_prep import prepare_dataset
from core.trainers.layer2b_quality import (
    _build_trade_quality_targets,
    _diagnose_l2b_label_funnel,
)


def main() -> None:
    print("Loading dataset via prepare_dataset(['QQQ', 'SPY']) …", flush=True)
    df, _feat = prepare_dataset(["QQQ", "SPY"])
    print(f"Rows: {len(df):,}", flush=True)
    print("Building trade_quality labels (hybrid-ratio directional gate) …", flush=True)
    y = _build_trade_quality_targets(df.copy())
    _diagnose_l2b_label_funnel(df, y, TRAIN_END)


if __name__ == "__main__":
    main()

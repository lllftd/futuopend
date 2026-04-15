from __future__ import annotations

from dataclasses import dataclass


@dataclass
class L1cConfig:
    seq_len: int = 60
    predict_horizon: int = 10
    symbols: tuple[str, ...] = ("QQQ", "SPY")

    input_dim: int | None = None
    embed_dim: int = 48
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 96

    attn_dropout: float = 0.25
    ff_dropout: float = 0.28
    embed_dropout: float = 0.25

    # Stochastic depth: skip an entire transformer block this fraction of forwards (train only).
    layer_drop: float = 0.08

    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 48
    patience: int = 3

    # Binary direction BCE: targets smoothed toward 0.5 (e.g. 0→0.05, 1→0.95).
    label_smoothing: float = 0.05

    # LR schedule: CosineAnnealingWarmRestarts (T_0, T_mult); set T0=0 to disable (plateau only).
    cosine_t0: int = 5
    cosine_t_mult: int = 2

    max_train_features: int = 40

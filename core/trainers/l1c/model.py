from __future__ import annotations

import torch
import torch.nn as nn

from core.trainers.l1c.config import L1cConfig


class TemporalPositionEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 120):
        super().__init__()
        self.learned_pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        decay = torch.linspace(-2, 0, max_len).exp()
        self.register_buffer("decay", decay.view(1, max_len, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        pe = self.learned_pe[:, :t, :]
        decay = self.decay[:, :t, :]
        return x * decay + pe


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class L1cDirectionModel(nn.Module):
    """Causal Transformer with a single logits head: P(future return > 0)."""

    def __init__(self, config: L1cConfig):
        super().__init__()
        self.config = config
        if config.input_dim is None:
            raise ValueError("L1cDirectionModel requires config.input_dim")
        d_in = int(config.input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.Dropout(config.embed_dropout),
        )
        self.pos_enc = TemporalPositionEncoding(config.embed_dim, max_len=config.seq_len + 8)
        self.layer_drop = float(config.layer_drop)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.ff_dim,
                    config.attn_dropout,
                    config.ff_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.direction_logits = nn.Sequential(
            nn.Linear(config.embed_dim, max(32, config.embed_dim // 2)),
            nn.GELU(),
            nn.Dropout(config.ff_dropout),
            nn.Linear(max(32, config.embed_dim // 2), 1),
        )
        t = int(config.seq_len)
        mask = torch.triu(torch.ones(t, t, dtype=torch.float32) * float("-inf"), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, t, _ = x.shape
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = self.causal_mask[:t, :t]
        for block in self.blocks:
            if self.training and self.layer_drop > 0.0 and torch.rand((), device=h.device) < self.layer_drop:
                continue
            h = block(h, attn_mask=mask)
        h = self.final_norm(h)
        h_last = h[:, -1, :]
        return self.direction_logits(h_last)

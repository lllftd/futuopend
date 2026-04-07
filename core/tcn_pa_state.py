"""
Shared PAStateTCN architecture for TCN training and inference.

Single source of truth for checkpoint compatibility with train_tcn_pa_state outputs.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def _default_bottleneck_dim() -> int:
    return max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))


class SpatialDropout1d(nn.Module):
    """Drops entire channels instead of individual elements."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full((x.size(0), x.size(1), 1), 1 - self.p, device=x.device))
        return x * mask / (1 - self.p)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.spatial_drop = SpatialDropout1d(dropout)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.spatial_drop(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.drop(out)
        res = self.downsample(x) if self.downsample is not None else x
        return self.act(out + res)


class PAStateTCN(nn.Module):
    """Causal TCN backbone + learned bottleneck z (for LGBM) + K-class regime head."""

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
        bottleneck_dim: int | None = None,
        num_classes: int = 6,
    ):
        super().__init__()
        bd = int(bottleneck_dim) if bottleneck_dim is not None else _default_bottleneck_dim()
        self.bottleneck_dim = bd
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        hidden_dim = num_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bd),
            nn.BatchNorm1d(bd),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(bd, num_classes)

    def forward(self, x):
        logits, _z = self.forward_with_embedding(x)
        return logits

    def forward_with_embedding(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        h = out[:, :, -1]
        z = self.bottleneck(h)
        regime_logits = self.classifier(z)
        return regime_logits, z

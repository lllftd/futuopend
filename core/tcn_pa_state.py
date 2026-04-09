"""
Shared PAStateTCN architecture for TCN training and inference.

Single source of truth for checkpoint compatibility with train_tcn_pa_state outputs.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def _default_bottleneck_dim() -> int:
    return max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma=0 退化为标准 weighted CE。
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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
        # Use WeightNorm instead of BatchNorm for causality and batch independence
        # Channel doubled for WaveNet gated activation.
        # weight_norm must wrap nn.Conv1d (has .weight); CausalConv1d only delegates to .conv.
        self.conv1 = CausalConv1d(in_ch, out_ch * 2, kernel_size, dilation)
        weight_norm(self.conv1.conv)
        self.conv2 = CausalConv1d(out_ch, out_ch * 2, kernel_size, dilation)
        weight_norm(self.conv2.conv)

        # Initialize gate bias to +1.0 so gates start open
        if self.conv1.conv.bias is not None:
            nn.init.constant_(self.conv1.conv.bias[out_ch:], 1.0)
        if self.conv2.conv.bias is not None:
            nn.init.constant_(self.conv2.conv.bias[out_ch:], 1.0)
        
        self.spatial_drop = SpatialDropout1d(dropout)
        self.drop = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        if self.downsample is not None:
            self.downsample = weight_norm(self.downsample)

    def _gated_activation(self, x):
        # x shape: (B, out_ch * 2, T)
        filter_out, gate_out = x.chunk(2, dim=1)
        return torch.tanh(filter_out) * torch.sigmoid(gate_out)

    def forward(self, x):
        out = self._gated_activation(self.conv1(x))
        out = self.spatial_drop(out)
        out = self._gated_activation(self.conv2(out))
        out = self.drop(out)
        res = self.downsample(x) if self.downsample is not None else x
        # Scale by sqrt(0.5) to prevent exploding activations in deeper layers 
        # (variance compensation for residual connections with weightnorm)
        return (out + res) * 0.70710678


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
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.noise_std = noise_std
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
        # Apply Gaussian noise injection for robust sequence learning
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        x = x.transpose(1, 2)
        out = self.tcn(x)
        h = out[:, :, -1]
        z = self.bottleneck(h)
        regime_logits = self.classifier(z)
        return regime_logits, z

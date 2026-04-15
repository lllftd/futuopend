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


def _default_tcn_readout_type() -> str:
    return os.environ.get("TCN_READOUT_TYPE", "last_timestep").strip().lower() or "last_timestep"


def _default_min_attention_seq_len() -> int:
    return max(1, int(os.environ.get("TCN_MIN_ATTENTION_SEQ_LEN", "4")))


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma=0 退化为标准 weighted CE。
    """
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        *,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
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


class ChannelSE1d(nn.Module):
    """Squeeze-and-Excitation over channel dim for (B, C, T)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // int(reduction), 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class L1AGatedTemporalBlock(nn.Module):
    """
    L1a backbone only: GLU (value * sigmoid(gate)), GroupNorm (groups=1), optional SE.
    Causal conv + residual scaling match TemporalBlock training stability.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        use_se: bool = True,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch * 2, kernel_size, dilation)
        weight_norm(self.conv1.conv)
        self.conv2 = CausalConv1d(out_ch, out_ch * 2, kernel_size, dilation)
        weight_norm(self.conv2.conv)
        if self.conv1.conv.bias is not None:
            nn.init.constant_(self.conv1.conv.bias[out_ch:], 1.0)
        if self.conv2.conv.bias is not None:
            nn.init.constant_(self.conv2.conv.bias[out_ch:], 1.0)
        self.gn1 = nn.GroupNorm(1, out_ch)
        self.gn2 = nn.GroupNorm(1, out_ch)
        self.spatial_drop = SpatialDropout1d(dropout)
        self.drop = nn.Dropout(dropout)
        self.se = ChannelSE1d(out_ch) if use_se else nn.Identity()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        if self.downsample is not None:
            self.downsample = weight_norm(self.downsample)

    @staticmethod
    def _glu(x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=1)
        return value * torch.sigmoid(gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x) if self.downsample is not None else x
        out = self._glu(self.conv1(x))
        out = self.gn1(out)
        out = self.spatial_drop(out)
        out = self._glu(self.conv2(out))
        out = self.gn2(out)
        out = self.drop(out)
        out = self.se(out)
        return (out + res) * 0.70710678


class TemporalAttentionReadout(nn.Module):
    """Lightweight attention pooling over causal TCN states."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _last_valid_indices(valid_mask: torch.Tensor) -> torch.Tensor:
        time_idx = torch.arange(valid_mask.size(1), device=valid_mask.device, dtype=torch.long)
        return (valid_mask.long() * time_idx.unsqueeze(0)).max(dim=1).values

    def forward(
        self,
        seq: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        min_seq_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # seq: (B, T, H)
        batch_size, seq_len, hidden_dim = seq.shape
        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), device=seq.device, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(device=seq.device, dtype=torch.bool)
            if valid_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"attention_mask shape {tuple(valid_mask.shape)} does not match sequence "
                    f"shape {(batch_size, seq_len)}"
                )

        valid_counts = valid_mask.sum(dim=1)
        empty_rows = valid_counts == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, -1] = True
            valid_counts = valid_mask.sum(dim=1)

        fallback_rows = valid_counts < max(1, int(min_seq_len))
        last_valid_idx = self._last_valid_indices(valid_mask)

        pooled = seq.new_zeros((batch_size, hidden_dim))
        weights = seq.new_zeros((batch_size, seq_len))

        if fallback_rows.any():
            pooled[fallback_rows] = seq[fallback_rows, last_valid_idx[fallback_rows]]
            weights[fallback_rows, last_valid_idx[fallback_rows]] = 1.0

        attn_rows = ~fallback_rows
        if attn_rows.any():
            seq_sub = seq[attn_rows]
            mask_sub = valid_mask[attn_rows]
            logits = self.score(seq_sub).squeeze(-1).float()
            logits = logits.masked_fill(~mask_sub, float("-inf"))
            logits = logits - logits.amax(dim=1, keepdim=True)
            attn_weights = torch.softmax(logits, dim=1).to(dtype=seq.dtype)
            pooled[attn_rows] = torch.sum(seq_sub * attn_weights.unsqueeze(-1), dim=1)
            weights[attn_rows] = attn_weights

        return pooled, weights


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
        readout_type: str | None = None,
        min_attention_seq_len: int | None = None,
    ):
        super().__init__()
        self.noise_std = noise_std
        bd = int(bottleneck_dim) if bottleneck_dim is not None else _default_bottleneck_dim()
        self.bottleneck_dim = bd
        self.readout_type = (readout_type or _default_tcn_readout_type()).strip().lower()
        self.min_attention_seq_len = (
            int(min_attention_seq_len)
            if min_attention_seq_len is not None
            else _default_min_attention_seq_len()
        )
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        hidden_dim = num_channels[-1]
        if self.readout_type == "attention":
            self.readout = TemporalAttentionReadout(hidden_dim, dropout=dropout)
        elif self.readout_type == "last_timestep":
            self.readout = None
        else:
            raise ValueError(f"Unsupported TCN readout_type: {self.readout_type}")
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bd),
            nn.BatchNorm1d(bd),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(bd, num_classes)

    def forward(self, x, *, attention_mask: torch.Tensor | None = None):
        logits, _z = self.forward_with_embedding(x, attention_mask=attention_mask)
        return logits

    def forward_with_embedding(
        self,
        x,
        *,
        return_attention: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        # Apply Gaussian noise injection for robust sequence learning
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        x = x.transpose(1, 2)
        out = self.tcn(x)
        attn_weights = None
        if self.readout_type == "attention":
            h, attn_weights = self.readout(
                out.transpose(1, 2),
                attention_mask=attention_mask,
                min_seq_len=self.min_attention_seq_len,
            )
        else:
            seq_mask = (
                attention_mask.to(device=out.device, dtype=torch.bool)
                if attention_mask is not None
                else None
            )
            last_valid_idx = None
            if seq_mask is None:
                h = out[:, :, -1]
            else:
                if seq_mask.shape != (out.size(0), out.size(2)):
                    raise ValueError(
                        f"attention_mask shape {tuple(seq_mask.shape)} does not match sequence "
                        f"shape {(out.size(0), out.size(2))}"
                    )
                valid_counts = seq_mask.sum(dim=1)
                empty_rows = valid_counts == 0
                if empty_rows.any():
                    seq_mask = seq_mask.clone()
                    seq_mask[empty_rows, -1] = True
                last_valid_idx = TemporalAttentionReadout._last_valid_indices(seq_mask)
                h = out.transpose(1, 2)[torch.arange(out.size(0), device=out.device), last_valid_idx]
            if return_attention:
                attn_weights = torch.zeros(
                    out.size(0),
                    out.size(2),
                    device=out.device,
                    dtype=out.dtype,
                )
                if last_valid_idx is None:
                    attn_weights[:, -1] = 1.0
                else:
                    attn_weights[torch.arange(out.size(0), device=out.device), last_valid_idx] = 1.0
        z = self.bottleneck(h)
        regime_logits = self.classifier(z)
        if return_attention:
            return regime_logits, z, attn_weights
        return regime_logits, z

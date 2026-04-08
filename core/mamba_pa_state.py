"""
Pure PyTorch implementation of Mamba-minimal architecture for PA sequence state extraction.
Designed to run on Apple Silicon (MPS) without Triton/CUDA dependencies.
"""
from __future__ import annotations

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _default_bottleneck_dim() -> int:
    return max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))


class MambaBlock(nn.Module):
    """
    A simplified, pure-PyTorch implementation of a Mamba (S4) block.
    Optimized for short sequences (seq_len < 100) where a sequential scan is fast enough.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, _ = x.shape
        
        xz = self.in_proj(x)
        x_hidden, z = xz.chunk(2, dim=-1)
        
        # 1D Convolution
        x_hidden = x_hidden.transpose(1, 2)
        x_hidden = self.conv1d(x_hidden)[:, :, :L]
        x_hidden = x_hidden.transpose(1, 2)
        x_hidden = F.silu(x_hidden)
        
        # SSM projections
        x_proj = self.x_proj(x_hidden)
        delta, B_proj, C_proj = torch.split(x_proj, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(delta)
        delta = self.dt_proj(delta) # (B, L, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        D = self.D.float()
        
        # Sequential associative scan
        y = torch.empty_like(x_hidden)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(L):
            dt_t = delta[:, t].unsqueeze(-1)
            A_t = dt_t * A
            B_t = dt_t * B_proj[:, t].unsqueeze(1)
            
            # Zero-order hold approximation
            dA = torch.exp(A_t)
            dB = B_t
            
            x_t = x_hidden[:, t].unsqueeze(-1)
            h = dA * h + dB * x_t
            
            y_t = (h * C_proj[:, t].unsqueeze(1)).sum(dim=-1)
            y[:, t] = y_t + D * x_hidden[:, t]
            
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class PAStateMamba(nn.Module):
    """
    Mamba backbone + learned bottleneck z (for LGBM) + classification head.
    Matches the PAStateTCN signature exactly.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_layers: int = 4,
        dropout: float = 0.25,
        bottleneck_dim: int | None = None,
        num_classes: int = 6,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.noise_std = noise_std
        bd = int(bottleneck_dim) if bottleneck_dim is not None else _default_bottleneck_dim()
        self.bottleneck_dim = bd
        
        self.projection = nn.Linear(input_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bd),
            nn.BatchNorm1d(bd),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(bd, num_classes)

    def forward(self, x):
        logits, _z = self.forward_with_embedding(x)
        return logits

    def forward_with_embedding(self, x):
        """
        x: (B, seq_len, input_size) or (B, input_size, seq_len)
        TCN outputs x as (B, seq_len, input_size) but layer1_tcn trainer expects x: (B, seq_len, feat)
        """
        # In TCN trainer, data is (B, L, C)
        # Apply Gaussian noise injection for robust sequence learning
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        x = self.projection(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x) + x # Residual connection
            
        # Take the final timestep for the sequence representation
        h = self.norm(x[:, -1, :])
        
        z = self.bottleneck(h)
        regime_logits = self.classifier(z)
        return regime_logits, z

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.trainers.l1c.config import L1cConfig


class L1cBinaryDirectionLoss(nn.Module):
    """Weighted BCE-with-logits on binary up/down; sample_weight scales per-row contribution."""

    def __init__(self, config: L1cConfig):
        super().__init__()
        self.label_smoothing = float(config.label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        y_up: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = logits.view(-1)
        y = y_up.view(-1).float()
        eps = self.label_smoothing * 0.5
        y_smooth = y * (1.0 - 2.0 * eps) + eps
        bce = F.binary_cross_entropy_with_logits(z, y_smooth, reduction="none")
        w = sample_weight.view(-1).float().clamp_min(1e-8)
        denom = w.sum().clamp_min(1e-8)
        loss = (bce * w).sum() / denom
        with torch.no_grad():
            p = torch.sigmoid(z)
            pred = (p >= 0.5).float()
            acc = ((pred == y).float() * w).sum() / denom
        parts = {
            "bce": loss.detach(),
            "weighted_acc": acc.detach(),
        }
        return loss, parts

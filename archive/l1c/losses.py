from __future__ import annotations

import torch
import torch.nn as nn

from archive.l1c.config import L1cConfig


class L1cRegressionLoss(nn.Module):
    """Weighted Huber on standardized forward return target."""

    def __init__(self, config: L1cConfig):
        super().__init__()
        self.delta = float(getattr(config, "huber_delta", 2.0))

    def forward(
        self,
        pred: torch.Tensor,
        y_target: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = pred.view(-1).float()
        y = y_target.view(-1).float()
        err = torch.abs(z - y)
        d = max(self.delta, 1e-6)
        huber = torch.where(err <= d, 0.5 * err * err, d * (err - 0.5 * d))
        w = sample_weight.view(-1).float().clamp_min(1e-8)
        denom = w.sum().clamp_min(1e-8)
        loss = (huber * w).sum() / denom
        with torch.no_grad():
            mae = (torch.abs(z - y) * w).sum() / denom
        parts = {
            "huber": loss.detach(),
            "weighted_mae": mae.detach(),
        }
        return loss, parts

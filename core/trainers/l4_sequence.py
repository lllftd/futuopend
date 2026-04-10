from __future__ import annotations

import torch
import torch.nn as nn


class L4ExitSequenceModel(nn.Module):
    """Lightweight GRU policy head for bar-by-bar exit decisions."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.exit_head = nn.Linear(self.hidden_size, 1)
        self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_proj = self.input_proj(x)
        out, hidden_out = self.gru(x_proj, hidden)
        out = self.dropout(out)
        exit_logits = self.exit_head(out).squeeze(-1)
        value_pred = self.value_head(out).squeeze(-1)
        return exit_logits, value_pred, hidden_out

    def forward_step(
        self,
        x_step: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exit_logits, value_pred, hidden_out = self.forward(x_step.unsqueeze(1), hidden)
        return exit_logits[:, 0], value_pred[:, 0], hidden_out

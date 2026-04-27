"""PyTorch unified L2 backbone + heads (replaces LightGBM when not L2_LEGACY_LGBM).

Position group is zero-filled for per-bar L2 training; L3 can extend later with real position_x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnifiedL2L3Config:
    group_dim: int = 64
    backbone_dim: int = 192
    n_backbone_layers: int = 4
    head_dim: int = 96
    head_dropout: float = 0.15
    position_dim: int = 8


def _residual_block(dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim),
        nn.LayerNorm(dim),
    )


class ResidualBackbone(nn.Module):
    def __init__(self, dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_residual_block(dim, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            y = layer(x)
            x = x + y
        return x


class ExitHead(nn.Module):
    """Holds/exit: close (binary) + urgency in (0,1) after sigmoid."""

    def __init__(self, backbone_dim: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(backbone_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.close_logit = nn.Linear(head_dim, 1)
        self.urgency_logit = nn.Linear(head_dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(z)
        return self.close_logit(h).squeeze(-1), self.urgency_logit(h).squeeze(-1)


class ValueHead(nn.Module):
    """Remaining path value: level + horizon + optional PnL quantile fan."""

    def __init__(self, backbone_dim: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(backbone_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.remaining_pnl = nn.Linear(head_dim, 1)
        self.remaining_bars = nn.Linear(head_dim, 1)
        self.pnl_quantiles = nn.Linear(head_dim, 5)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(z)
        rp = self.remaining_pnl(h).squeeze(-1)
        rb = F.softplus(self.remaining_bars(h).squeeze(-1))
        q = self.pnl_quantiles(h)
        return rp, rb, q


class UnifiedL2L3Net(nn.Module):
    """Market (incl. l1b_*) + L1a regime + zero position → shared backbone → L2 heads."""

    def __init__(
        self,
        n_market: int,
        n_regime: int,
        n_position: int,
        cfg: UnifiedL2L3Config,
        *,
        multi_horizon: bool,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.multi_horizon = bool(multi_horizon)
        self.n_market = int(n_market)
        self.n_regime = int(n_regime)
        self.n_position = int(n_position)

        self.market_enc = nn.Sequential(
            nn.Linear(n_market, cfg.group_dim),
            nn.LayerNorm(cfg.group_dim),
            nn.GELU(),
        )
        self.regime_enc = nn.Sequential(
            nn.Linear(n_regime, cfg.group_dim),
            nn.LayerNorm(cfg.group_dim),
            nn.GELU(),
        )
        self.position_enc = nn.Sequential(
            nn.Linear(n_position, cfg.group_dim),
            nn.LayerNorm(cfg.group_dim),
            nn.GELU(),
        )
        fuse_in = cfg.group_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.backbone_dim),
            nn.LayerNorm(cfg.backbone_dim),
            nn.GELU(),
        )
        self.backbone = ResidualBackbone(cfg.backbone_dim, cfg.n_backbone_layers, cfg.head_dropout)

        hd = cfg.head_dim
        self.gate_head = nn.Sequential(
            nn.Linear(cfg.backbone_dim, hd),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(hd, 1),
        )
        self.range_head = nn.Sequential(
            nn.Linear(cfg.backbone_dim, hd),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(hd, 1),
        )
        self.mfe_head = nn.Sequential(
            nn.Linear(cfg.backbone_dim, hd),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(hd, 1),
        )
        self.mae_head = nn.Sequential(
            nn.Linear(cfg.backbone_dim, hd),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(hd, 1),
        )
        self.range10_head: nn.Module | None
        self.range20_head: nn.Module | None
        self.ttp_head: nn.Module | None
        if self.multi_horizon:
            self.range10_head = nn.Sequential(
                nn.Linear(cfg.backbone_dim, hd),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(hd, 1),
            )
            self.range20_head = nn.Sequential(
                nn.Linear(cfg.backbone_dim, hd),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(hd, 1),
            )
            self.ttp_head = nn.Sequential(
                nn.Linear(cfg.backbone_dim, hd),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(hd, 1),
            )
        else:
            self.range10_head = None
            self.range20_head = None
            self.ttp_head = None
        self.exit_head = ExitHead(cfg.backbone_dim, cfg.head_dim, cfg.head_dropout)
        self.value_head = ValueHead(cfg.backbone_dim, cfg.head_dim, cfg.head_dropout)

    def forward(
        self,
        market_x: torch.Tensor,
        regime_x: torch.Tensor,
        position_x: torch.Tensor,
        *,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        m = self.market_enc(market_x)
        r = self.regime_enc(regime_x)
        p = self.position_enc(position_x)
        z = self.fusion(torch.cat([m, r, p], dim=-1))
        z = self.backbone(z)
        out: dict[str, torch.Tensor] = {
            "gate_logit": self.gate_head(z).squeeze(-1),
            "range": self.range_head(z).squeeze(-1),
            "mfe": self.mfe_head(z).squeeze(-1),
            "mae": self.mae_head(z).squeeze(-1),
        }
        if self.multi_horizon and self.range10_head is not None:
            out["range_10"] = self.range10_head(z).squeeze(-1)
            out["range_20"] = self.range20_head(z).squeeze(-1)  # type: ignore[union-attr]
            out["ttp90"] = self.ttp_head(z).squeeze(-1)  # type: ignore[union-attr]
        if return_all:
            c_logit, u_logit = self.exit_head(z)
            out["exit_close_logit"] = c_logit
            out["exit_urgency_logit"] = u_logit
            rp, rb, vq = self.value_head(z)
            out["value_remaining_pnl"] = rp
            out["value_remaining_bars"] = rb
            out["value_pnl_quantiles"] = vq
        return out

    @staticmethod
    def from_meta(meta: dict[str, Any]) -> UnifiedL2L3Net:
        ucfg = meta.get("l2_unified_config") or {}
        cfg = UnifiedL2L3Config(
            group_dim=int(ucfg.get("group_dim", 64)),
            backbone_dim=int(ucfg.get("backbone_dim", 192)),
            n_backbone_layers=int(ucfg.get("n_backbone_layers", 4)),
            head_dim=int(ucfg.get("head_dim", 96)),
            head_dropout=float(ucfg.get("head_dropout", 0.15)),
            position_dim=int(ucfg.get("position_dim", 8)),
        )
        net = UnifiedL2L3Net(
            n_market=int(ucfg["n_market"]),
            n_regime=int(ucfg["n_regime"]),
            n_position=int(ucfg["n_position"]),
            cfg=cfg,
            multi_horizon=bool(ucfg.get("multi_horizon", False)),
        )
        return net


def split_feature_indices(feature_cols: list[str]) -> tuple[list[int], list[int]]:
    """Regime = l1a_* only; market = all other columns (includes l1b_*, pa_*, etc.)."""
    regime_idx = [i for i, c in enumerate(feature_cols) if c.startswith("l1a_")]
    market_idx = [i for i in range(len(feature_cols)) if i not in set(regime_idx)]
    return market_idx, regime_idx

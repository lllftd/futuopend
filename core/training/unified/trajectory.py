"""GRU trajectory encoder forward pass + per-step feature builders for L3.

Training / joint fine-tune with exit+value heads was removed; L3 fits LightGBM on static rows only.
``L3TrajectoryEncoder`` + ``l3_single_trajectory_embedding`` remain for API compatibility if an
encoder checkpoint is ever loaded (current pipeline does not train or ship one).

Padding: valid timesteps are left-aligned in (B,T,D); GRU uses pack_padded_sequence so padding
does not affect recurrence. Attention masks padded positions to -inf before softmax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class L3TrajectoryConfig:
    max_seq_len: int = 32
    seq_feat_dim: int = 12
    gru_hidden: int = 32
    gru_layers: int = 2
    gru_dropout: float = 0.15
    embed_dim: int = 32
    # clip(live_mfe / mfe_norm_scale, 0, 1) in step features; fit from data (pred_mfe p99) at L3 build
    mfe_norm_scale: float = 5.0
    mae_norm_scale: float = 5.0


@dataclass
class L3TrajRollingState:
    """Per-trade trajectory buffer for live / OOS inference (matches training padding)."""

    max_seq_len: int = 32
    max_seq_ref: int = 32
    seq_feat_dim: int = 12
    mfe_norm_scale: float = 5.0
    mae_norm_scale: float = 5.0
    prev_unreal: float = 0.0
    peak_unreal: float = -1e9
    hist: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.prev_unreal = 0.0
        self.peak_unreal = -1e9
        self.hist.clear()

    def append_step(
        self,
        unreal: float,
        hold: int,
        ts: np.datetime64,
        close_prev: float,
        close_now: float,
        high_now: float,
        low_now: float,
        atr: float,
        vol_surprise: float,
        regime_div: float,
        live_mfe: float,
        live_mae: float,
    ) -> None:
        self.peak_unreal = max(self.peak_unreal, float(unreal))
        v = l3_traj_step_features(
            float(unreal),
            self.prev_unreal,
            self.peak_unreal,
            int(hold),
            ts,
            close_prev,
            close_now,
            high_now,
            low_now,
            atr,
            vol_surprise,
            regime_div,
            float(live_mfe),
            float(live_mae),
            max_seq_ref=self.max_seq_ref,
            mfe_scale=self.mfe_norm_scale,
            mae_scale=self.mae_norm_scale,
        )
        self.hist.append(v)
        self.prev_unreal = float(unreal)

    def padded_sequence(self) -> tuple[np.ndarray, int]:
        seq = np.zeros((self.max_seq_len, self.seq_feat_dim), dtype=np.float32)
        window = self.hist[-self.max_seq_len :]
        sl = len(window)
        if sl:
            seq[:sl] = np.stack(window, axis=0)
        return seq, sl

    def append_step_straddle(
        self,
        pnl_pct: float,
        hold: int,
        ts: np.datetime64,
        price_rel_prev: float,
        price_rel_now: float,
        underlying_abs_move: float,
        iv_now: float,
        vol_surprise: float,
        regime_div: float,
        vega: float,
        theta_abs: float,
    ) -> None:
        self.peak_unreal = max(self.peak_unreal, float(pnl_pct))
        v = l3_traj_step_features_straddle(
            float(pnl_pct),
            self.prev_unreal,
            self.peak_unreal,
            int(hold),
            ts,
            price_rel_prev,
            price_rel_now,
            underlying_abs_move,
            iv_now,
            vol_surprise,
            regime_div,
            float(vega),
            float(theta_abs),
            max_seq_ref=self.max_seq_ref,
            mfe_scale=self.mfe_norm_scale,
            mae_scale=self.mae_norm_scale,
        )
        self.hist.append(v)
        self.prev_unreal = float(pnl_pct)


def l3_traj_step_features(
    unreal: float,
    prev_unreal: float,
    peak_unreal: float,
    hold: int,
    ts: np.datetime64,
    close_prev: float,
    close_now: float,
    high_now: float,
    low_now: float,
    atr: float,
    vol_surprise: float,
    regime_div: float,
    live_mfe: float,
    live_mae: float,
    *,
    max_seq_ref: int = 32,
    mfe_scale: float = 5.0,
    mae_scale: float = 5.0,
) -> np.ndarray:
    """One bar of trajectory features (ATR-normalized where applicable)."""
    atr = max(float(atr), 1e-6)
    pnl_delta = float(unreal) - float(prev_unreal)
    pk = float(peak_unreal)
    if pk <= -1e8:
        pnl_vs_peak = 0.0
    else:
        denom = max(abs(pk), 1e-6)
        pnl_vs_peak = (pk - float(unreal)) / denom
    hold_norm = min(int(hold), max_seq_ref) / float(max_seq_ref)
    t = pd.Timestamp(ts)
    ang = 2.0 * np.pi * (t.hour * 60 + t.minute) / (24.0 * 60.0)
    sin_t = float(np.sin(ang))
    cos_t = float(np.cos(ang))
    bar_ret = (float(close_now) - float(close_prev)) / atr
    bar_range = (float(high_now) - float(low_now)) / atr
    ms = max(float(mfe_scale), 1e-6)
    mas = max(float(mae_scale), 1e-6)
    return np.asarray(
        [
            float(unreal),
            pnl_delta,
            float(pnl_vs_peak),
            hold_norm,
            sin_t,
            cos_t,
            float(bar_ret),
            float(bar_range),
            float(vol_surprise),
            float(regime_div),
            float(np.clip(live_mfe / ms, 0.0, 1.0)),
            float(np.clip(live_mae / mas, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def l3_traj_step_features_straddle(
    pnl_pct: float,
    prev_pnl_pct: float,
    peak_pnl_pct: float,
    hold: int,
    ts: np.datetime64,
    price_rel_prev: float,
    price_rel_now: float,
    underlying_abs_move: float,
    iv_now: float,
    vol_surprise: float,
    regime_div: float,
    vega: float,
    theta_abs: float,
    *,
    max_seq_ref: int = 32,
    mfe_scale: float = 5.0,
    mae_scale: float = 5.0,
) -> np.ndarray:
    """Alternative sequence encoding for straddle simulator mode.

    Keeps the same 12-dim contract as `l3_traj_step_features`.
    """
    pnl = float(pnl_pct)
    pnl_delta = pnl - float(prev_pnl_pct)
    pk = float(peak_pnl_pct)
    pnl_vs_peak = 0.0 if pk <= -1e8 else (pk - pnl) / max(abs(pk), 1e-6)
    hold_norm = min(int(hold), max_seq_ref) / float(max_seq_ref)
    t = pd.Timestamp(ts)
    ang = 2.0 * np.pi * (t.hour * 60 + t.minute) / (24.0 * 60.0)
    sin_t = float(np.sin(ang))
    cos_t = float(np.cos(ang))
    price_rel_prev = float(price_rel_prev)
    price_rel_now = float(price_rel_now)
    bar_ret = price_rel_now - price_rel_prev
    bar_range = float(underlying_abs_move)
    ms = max(float(mfe_scale), 1e-6)
    mas = max(float(mae_scale), 1e-6)
    return np.asarray(
        [
            pnl,
            pnl_delta,
            float(pnl_vs_peak),
            hold_norm,
            sin_t,
            cos_t,
            float(bar_ret),
            float(bar_range),
            float(vol_surprise),
            float(regime_div),
            float(np.clip(abs(vega) / ms, 0.0, 1.0)),
            float(np.clip(theta_abs / mas, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


class L3TrajectoryEncoder(nn.Module):
    """GRU + attention pooling -> fixed embedding."""

    def __init__(self, cfg: L3TrajectoryConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.seq_feat_dim)
        self.gru = nn.GRU(
            input_size=cfg.seq_feat_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            dropout=cfg.gru_dropout if cfg.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn_query = nn.Linear(cfg.gru_hidden, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.gru_dropout),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, sequences: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(sequences)
        B, T, _ = x.shape
        lens = seq_lens.clamp(min=1).long()
        sorted_lens, sort_idx = lens.sort(descending=True)
        _, unsort_idx = sort_idx.sort()
        x_sorted = x[sort_idx]
        packed = pack_padded_sequence(
            x_sorted,
            sorted_lens.detach().cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        gru_out, _ = self.gru(packed)
        unpacked, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=T)
        gru_output = unpacked[unsort_idx]

        scores = self.attn_query(gru_output).squeeze(-1)
        # Mask padding so attention weights are zero on padded timesteps (after softmax).
        mask = torch.arange(T, device=gru_output.device).unsqueeze(0) >= lens.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(mask, 0.0)
        ctx = torch.bmm(attn.unsqueeze(1), gru_output).squeeze(1)
        return self.proj(ctx)


@torch.no_grad()
def l3_single_trajectory_embedding(
    encoder: L3TrajectoryEncoder,
    seq: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Encode one padded trajectory (max_seq_len, D) -> (embed_dim,) float32."""
    encoder.eval()
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
    l = torch.tensor([max(1, int(seq_len))], dtype=torch.int64, device=device)
    return encoder(x, l).detach().cpu().numpy().astype(np.float32).ravel()

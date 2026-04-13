"""GRU trajectory encoder for L3 policy rows; embeddings are concatenated to LightGBM features.

Padding: valid timesteps are left-aligned in (B,T,D); GRU uses pack_padded_sequence so padding
does not affect recurrence. Attention masks padded positions to -inf before softmax.

Per-step inputs are ATR-normalized / bounded; LayerNorm on the last feature dim helps scale but
is not a fixed offline scaler — for strict train/test feature scaling, extend L3TrajectoryConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from core.trainers.constants import FAST_TRAIN_MODE


@dataclass(frozen=True)
class L3TrajectoryConfig:
    max_seq_len: int = 32
    seq_feat_dim: int = 12
    gru_hidden: int = 32
    gru_layers: int = 2
    gru_dropout: float = 0.15
    embed_dim: int = 32


@dataclass
class L3TrajRollingState:
    """Per-trade trajectory buffer for live / OOS inference (matches training padding)."""

    max_seq_len: int = 32
    max_seq_ref: int = 32
    seq_feat_dim: int = 12
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
            float(np.clip(live_mfe / 5.0, 0.0, 1.0)),
            float(np.clip(live_mae / 5.0, 0.0, 1.0)),
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


@torch.no_grad()
def l3_encode_trajectories(
    encoder: L3TrajectoryEncoder,
    seq: np.ndarray,
    lens: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """seq: (N, T, D), lens: (N,) -> (N, embed_dim)"""
    encoder.eval()
    out_list: list[np.ndarray] = []
    n = len(seq)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(seq[s:e].astype(np.float32)).to(device)
        lb = torch.from_numpy(lens[s:e].astype(np.int64)).to(device)
        emb = encoder(xb, lb).cpu().numpy()
        out_list.append(emb)
    return np.concatenate(out_list, axis=0)


def train_l3_trajectory_encoder(
    seq_tr: np.ndarray,
    len_tr: np.ndarray,
    y_exit_tr: np.ndarray,
    y_value_tr: np.ndarray,
    seq_va: np.ndarray,
    len_va: np.ndarray,
    y_exit_va: np.ndarray,
    y_value_va: np.ndarray,
    *,
    cfg: L3TrajectoryConfig,
    device: torch.device,
) -> L3TrajectoryEncoder:
    encoder = L3TrajectoryEncoder(cfg).to(device)
    exit_head = nn.Sequential(
        nn.Linear(cfg.embed_dim, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    ).to(device)
    value_head = nn.Sequential(
        nn.Linear(cfg.embed_dim, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    ).to(device)
    params = list(encoder.parameters()) + list(exit_head.parameters()) + list(value_head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    max_epochs = 18 if FAST_TRAIN_MODE else 50
    patience = 4 if FAST_TRAIN_MODE else 10
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=1.0)
    bs = 256 if len(seq_tr) >= 512 else max(32, len(seq_tr) // 4)

    tr_ds = TensorDataset(
        torch.from_numpy(seq_tr),
        torch.from_numpy(len_tr.astype(np.int64)),
        torch.from_numpy(y_exit_tr.astype(np.float32)),
        torch.from_numpy(y_value_tr.astype(np.float32)),
    )
    va_ds = TensorDataset(
        torch.from_numpy(seq_va),
        torch.from_numpy(len_va.astype(np.int64)),
        torch.from_numpy(y_exit_va.astype(np.float32)),
        torch.from_numpy(y_value_va.astype(np.float32)),
    )
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=bs * 2, shuffle=False)

    best_state = None
    best_loss = float("inf")
    bad = 0
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  [L3-GRU] trajectory encoder params={n_params:,}  max_epochs={max_epochs}", flush=True)

    for epoch in range(max_epochs):
        encoder.train()
        exit_head.train()
        value_head.train()
        tr_loss = 0.0
        nb = 0
        for xb, lb, ye, yv in tr_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            ye = ye.to(device)
            yv = yv.to(device)
            emb = encoder(xb, lb)
            le = bce(exit_head(emb).squeeze(-1), ye)
            lv = huber(value_head(emb).squeeze(-1), yv)
            loss = le + 0.5 * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += float(loss.item())
            nb += 1
        avg_tr = tr_loss / max(nb, 1)

        encoder.eval()
        exit_head.eval()
        value_head.eval()
        va_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, lb, ye, yv in va_loader:
                xb = xb.to(device)
                lb = lb.to(device)
                ye = ye.to(device)
                yv = yv.to(device)
                emb = encoder(xb, lb)
                le = bce(exit_head(emb).squeeze(-1), ye)
                lv = huber(value_head(emb).squeeze(-1), yv)
                va_loss += float((le + 0.5 * lv).item())
                nv += 1
        avg_va = va_loss / max(nv, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f" [L3-GRU] epoch {epoch + 1:>3}  train_loss={avg_tr:.4f}  val_loss={avg_va:.4f}", flush=True)

        if avg_va < best_loss - 1e-5:
            best_loss = avg_va
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  [L3-GRU] early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.eval()
    print(f"  [L3-GRU] best val proxy loss={best_loss:.4f}", flush=True)
    return encoder


def l3_trajectory_embed_importance_ratio(exit_model: Any, n_static: int, embed_dim: int) -> tuple[float, float]:
    """Return (static_share, embed_share) of total gain importance."""
    import lightgbm as lgb  # local import

    if not isinstance(exit_model, lgb.Booster):
        return 1.0, 0.0
    imp = np.asarray(exit_model.feature_importance(importance_type="gain"), dtype=np.float64)
    if imp.sum() <= 0:
        return 1.0, 0.0
    st = float(imp[:n_static].sum())
    em = float(imp[n_static : n_static + embed_dim].sum())
    tot = st + em
    return st / tot, em / tot

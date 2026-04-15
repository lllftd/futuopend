from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange, tqdm

from core.tcn_pa_state import FocalLoss, L1AGatedTemporalBlock, TemporalAttentionReadout
from core.trainers.constants import (
    FAST_TRAIN_MODE,
    L1A_META_FILE,
    L1A_MODEL_FILE,
    L1A_OUTPUT_CACHE_FILE,
    L1A_REGIME_COLS,
    L1A_SCHEMA_VERSION,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    REGIME_NOW_PROB_COLS,
)

from core.trainers.data_prep import _create_tcn_windows
from core.trainers.lgbm_utils import TQDM_FILE, _lgb_round_tqdm_enabled, _lgbm_n_jobs, _options_target_config
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.trainers.val_metrics_extra import brier_binary, brier_multiclass, ece_binary, ece_multiclass_maxprob, pearson_corr
from core.trainers.stack_v2_common import (
    build_stack_time_splits,
    compute_transition_event_labels,
    log_label_baseline,
    save_output_cache,
)
from core.trainers.tcn_constants import DEVICE, SEQ_LEN


def _bounded_scalar_cols() -> list[str]:
    return ["l1a_transition_risk"]


def _l1a_direction_cols() -> list[str]:
    return [
        "l1a_dir_bull_minus_bear",
        "l1a_dir_confidence",
        "l1a_dir_normalized",
        "l1a_bull_convergence",
        "l1a_bear_convergence",
        "l1a_dir_x_vol",
        "l1a_dir_x_stability",
    ]


def _l1a_embed_dim() -> int:
    return max(4, int(os.environ.get("L1A_EMBED_DIM", "8")))


def l1a_output_columns_with_embed_dim(embed_dim: int) -> list[str]:
    d = max(4, int(embed_dim))
    return (
        list(L1A_REGIME_COLS)
        + [
            "l1a_transition_risk",
            "l1a_vol_forecast",
            "l1a_vol_trend",
            "l1a_time_in_regime",
        ]
        + _l1a_direction_cols()
        + [f"l1a_market_embed_{idx}" for idx in range(d)]
        + ["l1a_is_warm"]
    )


def l1a_output_columns() -> list[str]:
    return l1a_output_columns_with_embed_dim(_l1a_embed_dim())


def _derive_l1a_direction_features(
    regime_probs: np.ndarray,
    transition_risk: np.ndarray,
    vol_forecast: np.ndarray,
) -> dict[str, np.ndarray]:
    regime = np.asarray(regime_probs, dtype=np.float32)
    if regime.ndim != 2 or regime.shape[1] != NUM_REGIME_CLASSES:
        raise ValueError(
            f"L1a direction features expect regime_probs shape (n, {NUM_REGIME_CLASSES}), got {regime.shape!r}."
        )
    transition = np.clip(np.asarray(transition_risk, dtype=np.float32).ravel(), 0.0, 1.0)
    vol = np.clip(np.asarray(vol_forecast, dtype=np.float32).ravel(), 0.0, 5.0)
    bull = regime[:, 0] + regime[:, 1]
    bear = regime[:, 2] + regime[:, 3]
    range_mass = regime[:, 4] + regime[:, 5]
    directional_mass = np.maximum(bull + bear, 1e-6)
    dir_signal = np.clip(bull - bear, -1.0, 1.0)
    dir_conf = np.clip(1.0 - range_mass, 0.0, 1.0)
    return {
        "l1a_dir_bull_minus_bear": dir_signal.astype(np.float32, copy=False),
        "l1a_dir_confidence": dir_conf.astype(np.float32, copy=False),
        "l1a_dir_normalized": np.clip(dir_signal / directional_mass, -1.0, 1.0).astype(np.float32, copy=False),
        "l1a_bull_convergence": np.clip(regime[:, 0] - regime[:, 1], -1.0, 1.0).astype(np.float32, copy=False),
        "l1a_bear_convergence": np.clip(regime[:, 2] - regime[:, 3], -1.0, 1.0).astype(np.float32, copy=False),
        "l1a_dir_x_vol": np.clip(dir_signal * vol, -5.0, 5.0).astype(np.float32, copy=False),
        "l1a_dir_x_stability": np.clip(dir_signal * (1.0 - transition), -1.0, 1.0).astype(np.float32, copy=False),
    }


def _l1a_readout_type() -> str:
    return (os.environ.get("L1A_READOUT_TYPE", "attention").strip().lower() or "attention")


def _l1a_min_attention_seq_len() -> int:
    return max(1, int(os.environ.get("L1A_MIN_ATTENTION_SEQ_LEN", os.environ.get("TCN_MIN_ATTENTION_SEQ_LEN", "4"))))


def _l1a_dataloader_workers(default: int) -> int:
    raw = os.environ.get("L1A_DATALOADER_WORKERS", "").strip()
    workers = int(raw) if raw else int(default)
    return max(0, min(workers, 4))


def _l1a_tcn_channels() -> list[int]:
    raw = os.environ.get("L1A_TCN_CHANNELS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [48, 48, 64]


def _l1a_tcn_dropout() -> float:
    return float(os.environ.get("L1A_TCN_DROPOUT", "0.3"))


def _l1a_readout_dropout() -> float:
    return float(os.environ.get("L1A_READOUT_DROPOUT", "0.25"))


def _l1a_head_dropout() -> float:
    return float(os.environ.get("L1A_HEAD_DROPOUT", "0.3"))


def _l1a_loss_weights() -> dict[str, float]:
    embed = float(os.environ.get("L1A_EMBED_RECON_WEIGHT", "0.22"))
    transition = float(os.environ.get("L1A_TRANSITION_WEIGHT", "0.10"))
    vol = float(os.environ.get("L1A_VOL_WEIGHT", "0.22"))
    regime = float(os.environ.get("L1A_REGIME_WEIGHT", "0.36"))
    vol_trend = float(os.environ.get("L1A_VOL_TREND_WEIGHT", "0.07"))
    time_ir = float(os.environ.get("L1A_TIME_IN_REGIME_WEIGHT", "0.07"))
    total = max(regime + vol + transition + embed + vol_trend + time_ir, 1e-8)
    return {
        "regime": regime / total,
        "vol": vol / total,
        "transition": transition / total,
        "embed_recon": embed / total,
        "vol_trend": vol_trend / total,
        "time_in_regime": time_ir / total,
    }


def _l1a_regime_loss(
    regime_train_labels: np.ndarray,
    *,
    device: torch.device,
) -> nn.Module:
    y = np.asarray(regime_train_labels, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=NUM_REGIME_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / counts
    weights = inv / max(float(np.mean(inv)), 1e-8)
    alpha = torch.tensor(weights, dtype=torch.float32, device=device)
    gamma = float(os.environ.get("L1A_REGIME_FOCAL_GAMMA", "1.5"))
    ls = float(os.environ.get("L1A_REGIME_LABEL_SMOOTHING", "0.1"))
    return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean", label_smoothing=ls)


class _BCEFocalWithLogits(nn.Module):
    """Binary focal loss on logits; targets in {0,1}."""

    def __init__(self, *, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float().view(-1)
        z = logits.view(-1)
        log_p = F.logsigmoid(z)
        log_1p = F.logsigmoid(-z)
        ce = -t * log_p - (1.0 - t) * log_1p
        prob = torch.exp(log_p)
        p_t = prob * t + (1.0 - prob) * (1.0 - t)
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)
        loss = alpha_t * (1.0 - p_t).clamp_min(1e-6).pow(self.gamma) * ce
        return loss.mean()


def _l1a_transition_loss(
    transition_train_labels: np.ndarray,
    *,
    device: torch.device,
) -> nn.Module:
    y = np.asarray(transition_train_labels, dtype=np.float32).ravel()
    pos = float(np.sum(y > 0.5))
    neg = float(len(y) - pos)
    pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0), dtype=torch.float32, device=device)
    use_focal = os.environ.get("L1A_TRANSITION_USE_FOCAL", "0").strip().lower() in {"1", "true", "yes"}
    if use_focal:
        gamma = float(os.environ.get("L1A_TRANSITION_FOCAL_GAMMA", "2.0"))
        alpha = float(os.environ.get("L1A_TRANSITION_FOCAL_ALPHA", "0.75"))
        return _BCEFocalWithLogits(gamma=gamma, alpha=alpha).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _future_mean_by_symbol(values: np.ndarray, symbols: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    horizon = max(int(horizon), 1)
    symbols = np.asarray(symbols)
    values = np.asarray(values, dtype=np.float64)
    for sym in pd.unique(symbols):
        idx = np.flatnonzero(symbols == sym)
        if idx.size == 0:
            continue
        vals = values[idx]
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        loc = np.arange(idx.size)
        start = loc + 1
        end = np.minimum(idx.size, loc + 1 + horizon)
        counts = end - start
        valid = counts > 0
        mean_vals = np.zeros(idx.size, dtype=np.float32)
        mean_vals[valid] = ((csum[end[valid]] - csum[start[valid]]) / counts[valid]).astype(np.float32)
        out[idx] = mean_vals
    return out


def _per_symbol_lag_diff(values: np.ndarray, symbols: np.ndarray, lag: int) -> np.ndarray:
    lag = max(int(lag), 1)
    out = np.zeros(len(values), dtype=np.float32)
    values = np.asarray(values, dtype=np.float64)
    symbols = np.asarray(symbols)
    clip_lo = float(os.environ.get("L1A_VOL_TREND_CLIP_LO", "-3"))
    clip_hi = float(os.environ.get("L1A_VOL_TREND_CLIP_HI", "3"))
    for sym in pd.unique(symbols):
        idx = np.flatnonzero(symbols == sym)
        if idx.size <= lag:
            continue
        v = values[idx]
        d = np.zeros(len(idx), dtype=np.float64)
        d[lag:] = v[lag:] - v[:-lag]
        out[idx] = np.clip(d, clip_lo, clip_hi).astype(np.float32)
    return out


def _time_in_regime_fraction(state: np.ndarray, symbols: np.ndarray, cap: int) -> np.ndarray:
    cap = max(int(cap), 1)
    out = np.zeros(len(state), dtype=np.float32)
    state = np.asarray(state, dtype=np.int64)
    symbols = np.asarray(symbols)
    for sym in pd.unique(symbols):
        idx = np.flatnonzero(symbols == sym)
        if idx.size == 0:
            continue
        s = state[idx]
        run = 0
        for j in range(len(idx)):
            if j == 0 or s[j] != s[j - 1]:
                run = 1
            else:
                run += 1
            out[idx[j]] = float(min(run, cap)) / float(cap)
    return out


def _select_l1a_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    preferred = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "lbl_atr",
        "pa_vol_rvol",
        "pa_vol_momentum",
        "pa_bull_pressure",
        "pa_bear_pressure",
        "pa_or_breakout_strength",
        "pa_struct_swing_range_atr",
        "pa_vol_zscore_20",
        "pa_bo_wick_imbalance",
        "pa_bo_close_extremity",
        "pa_lead_macd_hist_slope",
        "pa_lead_rsi_slope",
        "pa_bo_dist_vwap",
        "pa_ctx_setup_long",
        "pa_ctx_setup_short",
        "pa_ctx_follow_through_long",
        "pa_ctx_follow_through_short",
        "pa_ctx_range_pressure",
        "pa_ctx_structure_veto",
    ]
    extra = [
        c
        for c in feat_cols
        if c.startswith("pa_")
        and not c.startswith(("pa_hmm_", "pa_garch_", "pa_hsmm_", "pa_egarch_"))
        and c not in preferred
    ]
    cols = [c for c in preferred + extra[:12] if c in df.columns]
    time_key = pd.to_datetime(df["time_key"])
    minutes = (time_key.dt.hour * 60 + time_key.dt.minute).astype(np.float32)
    df["l1a_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)
    cols.append("l1a_session_progress")
    return cols


def _build_l1a_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cfg = _options_target_config()
    horizon = int(cfg["decision_horizon_bars"])
    safe_atr = np.where(pd.to_numeric(df["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, df["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
    high = pd.to_numeric(df["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    symbols = df["symbol"].to_numpy()
    state = pd.to_numeric(df["state_label"], errors="coerce").fillna(4).to_numpy(dtype=np.int64)

    transition_risk = compute_transition_event_labels(state, symbols, horizon=horizon)
    range_norm = np.clip((high - low) / safe_atr, 0.0, 5.0)
    vol_forecast = _future_mean_by_symbol(range_norm, symbols, horizon=horizon)
    lag = int(os.environ.get("L1A_VOL_TREND_LAG", "5"))
    vol_trend = _per_symbol_lag_diff(range_norm, symbols, lag=lag)
    cap = int(os.environ.get("L1A_TIME_IN_REGIME_CAP", "120"))
    time_in_regime = _time_in_regime_fraction(state, symbols, cap=cap)

    return {
        "regime": state,
        "transition_risk": transition_risk.astype(np.float32),
        "vol_forecast": vol_forecast.astype(np.float32),
        "vol_trend": vol_trend,
        "time_in_regime": time_in_regime,
    }


def _build_symbol_windows(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    windows_list: list[np.ndarray] = []
    end_indices: list[np.ndarray] = []
    for _, grp in df.groupby("symbol", sort=False):
        x = grp[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, end_idx = _create_tcn_windows(x, seq_len)
        if len(end_idx) == 0:
            continue
        windows_list.append(windows)
        end_indices.append(grp.index.to_numpy()[end_idx])
    if not windows_list:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(windows_list, axis=0), np.concatenate(end_indices, axis=0)


class TaskHead(nn.Module):
    """Two-layer MLP readout; use identity for logits (CE / BCEWithLogits)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: str = "identity",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.activation == "identity":
            return y
        if self.activation == "tanh":
            return torch.tanh(y)
        if self.activation == "sigmoid":
            return torch.sigmoid(y)
        raise ValueError(f"Unknown TaskHead activation: {self.activation}")


class EmbedHead(nn.Module):
    def __init__(self, input_dim: int = 128, embed_dim: int = 16):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return self.projector(shared_repr)


class L1AMarketTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int] | None = None,
        *,
        seq_len: int = SEQ_LEN,
        readout_type: str | None = None,
        min_attention_seq_len: int | None = None,
        tcn_dropout: float | None = None,
        readout_dropout: float | None = None,
        head_dropout: float | None = None,
        embed_dim: int | None = None,
    ):
        super().__init__()
        if channels is None:
            channels = _l1a_tcn_channels()
        td = float(tcn_dropout) if tcn_dropout is not None else _l1a_tcn_dropout()
        rd = float(readout_dropout) if readout_dropout is not None else _l1a_readout_dropout()
        hd_drop = float(head_dropout) if head_dropout is not None else _l1a_head_dropout()
        layers: list[nn.Module] = []
        use_se = os.environ.get("L1A_TCN_USE_SE", "1").strip().lower() in {"1", "true", "yes"}
        for idx, out_ch in enumerate(channels):
            in_ch = input_dim if idx == 0 else channels[idx - 1]
            layers.append(
                L1AGatedTemporalBlock(
                    in_ch, out_ch, kernel_size=3, dilation=2**idx, dropout=td, use_se=use_se
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.shared_dim = channels[-1]
        self.seq_len = seq_len
        self.readout_type = (readout_type or _l1a_readout_type()).strip().lower()
        self.min_attention_seq_len = (
            int(min_attention_seq_len) if min_attention_seq_len is not None else _l1a_min_attention_seq_len()
        )
        if self.readout_type == "attention":
            self.readout = TemporalAttentionReadout(self.shared_dim, dropout=rd)
        elif self.readout_type == "last_timestep":
            self.readout = None
        else:
            raise ValueError(f"Unsupported L1A readout_type: {self.readout_type}")
        hd = max(self.shared_dim // 2, 32)
        hd_small = max(hd // 2, 16)
        ed = int(embed_dim) if embed_dim is not None else _l1a_embed_dim()
        self.embed_dim = ed
        self.regime_head = TaskHead(self.shared_dim, hd, NUM_REGIME_CLASSES, activation="identity", dropout=hd_drop)
        self.transition_head = TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
        self.vol_head = TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
        self.vol_trend_head = TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
        self.time_in_regime_head = TaskHead(self.shared_dim, hd_small, 1, activation="sigmoid", dropout=hd_drop)
        self.embed_head = EmbedHead(self.shared_dim, ed)
        self.embed_decoder = nn.Sequential(
            nn.Linear(ed, 64),
            nn.GELU(),
            nn.Linear(64, self.shared_dim),
        )

    def shared_repr(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x.transpose(1, 2))
        if self.readout_type == "attention":
            pooled, _ = self.readout(h.transpose(1, 2), min_seq_len=self.min_attention_seq_len)
            return pooled
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_repr(x)
        embed = self.embed_head(shared)
        regime_logits = self.regime_head(shared)
        return {
            "regime_logits": regime_logits,
            "transition_logit": self.transition_head(shared).squeeze(-1),
            "vol_value": self.vol_head(shared).squeeze(-1),
            "vol_trend_value": self.vol_trend_head(shared).squeeze(-1),
            "time_in_regime_value": self.time_in_regime_head(shared).squeeze(-1),
            "market_embed": embed,
            "embed_recon": self.embed_decoder(embed),
            "shared_repr": shared,
        }


@dataclass
class L1ATrainingBundle:
    model: L1AMarketTCN
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _normalize_l1a_matrix(df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    mean = np.nanmean(X[train_mask], axis=0)
    std = np.nanstd(X[train_mask], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xn, mean.astype(np.float32), std.astype(np.float32)


def _train_epoch(
    model: L1AMarketTCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    regime_loss: nn.Module,
    transition_loss: nn.Module,
    loss_weights: dict[str, float],
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    mse = nn.MSELoss()
    it = loader
    if _lgb_round_tqdm_enabled():
        it = tqdm(
            loader,
            leave=False,
            desc="[L1a] train batches",
            file=TQDM_FILE,
            mininterval=0.25,
            unit="batch",
        )
    for xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir in it:
        xb = xb.to(device)
        y_regime = y_regime.to(device)
        y_transition = y_transition.to(device)
        y_vol = y_vol.to(device)
        y_vol_trend = y_vol_trend.to(device)
        y_time_ir = y_time_ir.to(device)
        out = model(xb)
        losses = {
            "regime": regime_loss(out["regime_logits"], y_regime),
            "vol": mse(out["vol_value"], y_vol),
            "vol_trend": mse(out["vol_trend_value"], y_vol_trend),
            "time_in_regime": mse(out["time_in_regime_value"], y_time_ir),
            "embed_recon": mse(out["embed_recon"], out["shared_repr"].detach()),
            "transition": transition_loss(out["transition_logit"], y_transition),
        }
        loss = (
            loss_weights["regime"] * losses["regime"]
            + loss_weights["vol"] * losses["vol"]
            + loss_weights["vol_trend"] * losses["vol_trend"]
            + loss_weights["time_in_regime"] * losses["time_in_regime"]
            + loss_weights["embed_recon"] * losses["embed_recon"]
            + loss_weights["transition"] * losses["transition"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(os.environ.get("L1A_MAX_GRAD_NORM", "1.0")))
        optimizer.step()
        total_loss += float(loss.item()) * len(xb)
        total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def _eval_epoch(
    model: L1AMarketTCN,
    loader: DataLoader,
    device: torch.device,
    *,
    regime_loss: nn.Module,
    transition_loss: nn.Module,
    loss_weights: dict[str, float],
) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    mse = nn.MSELoss()
    it = loader
    if _lgb_round_tqdm_enabled():
        it = tqdm(
            loader,
            leave=False,
            desc="[L1a] val batches",
            file=TQDM_FILE,
            mininterval=0.25,
            unit="batch",
        )
    with torch.no_grad():
        for xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir in it:
            xb = xb.to(device)
            y_regime = y_regime.to(device)
            y_transition = y_transition.to(device)
            y_vol = y_vol.to(device)
            y_vol_trend = y_vol_trend.to(device)
            y_time_ir = y_time_ir.to(device)
            out = model(xb)
            loss = (
                loss_weights["regime"] * regime_loss(out["regime_logits"], y_regime)
                + loss_weights["vol"] * mse(out["vol_value"], y_vol)
                + loss_weights["vol_trend"] * mse(out["vol_trend_value"], y_vol_trend)
                + loss_weights["time_in_regime"] * mse(out["time_in_regime_value"], y_time_ir)
                + loss_weights["embed_recon"] * mse(out["embed_recon"], out["shared_repr"].detach())
                + loss_weights["transition"] * transition_loss(out["transition_logit"], y_transition)
            )
            total_loss += float(loss.item()) * len(xb)
            total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def _l1a_transition_val_block(
    label: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    y_true = np.clip(np.asarray(y_true).ravel().astype(np.int32), 0, 1)
    y_score = np.clip(np.asarray(y_score).ravel().astype(np.float64), 1e-7, 1.0 - 1e-7)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    try:
        ll = float(log_loss(y_true, y_score))
    except ValueError:
        ll = float("nan")
    br = brier_binary(y_true.astype(np.float64), y_score)
    ece = ece_binary(y_true, y_score)
    pred = (y_score >= 0.5).astype(np.int32)
    print(
        f"  [L1a] val {label}:  n={len(y_true):,}  AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}",
        flush=True,
    )
    print(
        f"    positive_rate={float(np.mean(y_true)):.4f}  pred_mean={float(np.mean(y_score)):.4f}  "
        f"precision@0.5={float(np.mean(y_true[pred == 1])) if np.any(pred == 1) else 0.0:.4f}  "
        f"recall@0.5={float(np.mean(pred[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0:.4f}",
        flush=True,
    )
    if len(y_true) >= 20:
        order = np.argsort(y_score)
        top_n = max(1, int(0.10 * len(y_true)))
        top_mean = float(np.mean(y_true[order[-top_n:]]))
        base_mean = float(np.mean(y_true))
        lift = top_mean / max(base_mean, 1e-6)
        bot_mean = float(np.mean(y_true[order[:top_n]]))
        print(
            f"    top10% target_mean={top_mean:.4f}  bottom10% target_mean={bot_mean:.4f}  top10_lift={lift:.2f}x",
            flush=True,
        )
        try:
            dfq = pd.DataFrame({"pred": y_score, "target": y_true})
            dfq["bin"] = pd.qcut(dfq["pred"], 5, duplicates="drop")
            lift_tbl = dfq.groupby("bin", observed=True)["target"].agg(["mean", "count"])
            print(f"    target mean by pred quintile:\n{lift_tbl}", flush=True)
        except Exception as ex:
            print(f"    (skip transition quintile lift table: {ex})", flush=True)


def _log_l1a_val_metrics(model: L1AMarketTCN, val_dl: DataLoader, device: torch.device, *, label: str) -> None:
    """Validation report for regime calibration, transition event quality, vol regression, and embed stability."""
    model.eval()
    y_true_r: list[np.ndarray] = []
    y_pred_r: list[np.ndarray] = []
    y_prob_r: list[np.ndarray] = []
    vol_t: list[np.ndarray] = []
    vol_p: list[np.ndarray] = []
    vt_t: list[np.ndarray] = []
    vt_p: list[np.ndarray] = []
    tir_t: list[np.ndarray] = []
    tir_p: list[np.ndarray] = []
    tr_t, tr_s = [], []
    emb_mse: list[np.ndarray] = []
    with torch.no_grad():
        for xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir in val_dl:
            xb = xb.to(device)
            y_regime = y_regime.to(device)
            y_transition = y_transition.to(device)
            y_vol = y_vol.to(device)
            y_vol_trend = y_vol_trend.to(device)
            y_time_ir = y_time_ir.to(device)
            out = model(xb)
            y_true_r.append(y_regime.detach().cpu().numpy())
            regime_prob = torch.softmax(out["regime_logits"], dim=1)
            y_prob_r.append(regime_prob.detach().cpu().numpy())
            y_pred_r.append(torch.argmax(regime_prob, dim=1).detach().cpu().numpy())
            vol_t.append(y_vol.detach().cpu().numpy())
            vol_p.append(out["vol_value"].detach().cpu().numpy())
            vt_t.append(y_vol_trend.detach().cpu().numpy())
            vt_p.append(out["vol_trend_value"].detach().cpu().numpy())
            tir_t.append(y_time_ir.detach().cpu().numpy())
            tir_p.append(out["time_in_regime_value"].detach().cpu().numpy())
            tr_t.append(y_transition.detach().cpu().numpy())
            tr_s.append(torch.sigmoid(out["transition_logit"]).detach().cpu().numpy())
            emb_mse.append(
                F.mse_loss(out["embed_recon"], out["shared_repr"].detach(), reduction="none").mean(dim=1).detach().cpu().numpy()
            )
    yt = np.concatenate(y_true_r)
    yp = np.concatenate(y_pred_r)
    pr = np.clip(np.concatenate(y_prob_r), 1e-12, 1.0)
    pr = pr / pr.sum(axis=1, keepdims=True)
    labels = np.arange(NUM_REGIME_CLASSES)
    cm = confusion_matrix(yt, yp, labels=labels)
    acc = float(accuracy_score(yt, yp))
    f1_macro = float(f1_score(yt, yp, average="macro", zero_division=0))
    f1_weighted = float(f1_score(yt, yp, average="weighted", zero_division=0))
    try:
        kappa = float(cohen_kappa_score(yt, yp))
    except ValueError:
        kappa = float("nan")
    counts = np.bincount(yt.astype(int, copy=False), minlength=NUM_REGIME_CLASSES)
    try:
        regime_ll = float(log_loss(yt, pr, labels=list(range(NUM_REGIME_CLASSES))))
    except ValueError:
        regime_ll = float("nan")
    regime_br = brier_multiclass(yt, pr, NUM_REGIME_CLASSES)
    regime_ece = ece_multiclass_maxprob(yt, pr)
    vt = np.concatenate(vol_t)
    vp = np.concatenate(vol_p)
    mae_v = float(mean_absolute_error(vt, vp))
    rmse_v = float(np.sqrt(mean_squared_error(vt, vp)))
    r2_v = float(r2_score(vt, vp)) if len(np.unique(vt)) > 1 else float("nan")
    corr_v = pearson_corr(vt, vp)
    vtt = np.concatenate(vt_t)
    vtp = np.concatenate(vt_p)
    mae_vt = float(mean_absolute_error(vtt, vtp))
    corr_vt = pearson_corr(vtt, vtp)
    tirt = np.concatenate(tir_t)
    tirpr = np.concatenate(tir_p)
    mae_tir = float(mean_absolute_error(tirt, tirpr))
    corr_tir = pearson_corr(tirt, tirpr)
    emb_mean = float(np.mean(np.concatenate(emb_mse)))

    print(f"\n  [L1a] ========== val ({label}) effectiveness report ==========", flush=True)
    print(
        f"  [L1a] regime vs market_state  n={len(yt):,}  acc={acc:.4f}  macro-F1={f1_macro:.4f}  "
        f"weighted-F1={f1_weighted:.4f}  cohen_kappa={kappa:.4f}  log_loss={regime_ll:.4f}  "
        f"Brier={regime_br:.4f}  ECE={regime_ece:.4f}",
        flush=True,
    )
    print(f"  [L1a] true-class counts: {dict(zip(REGIME_NOW_PROB_COLS, counts.tolist()))}", flush=True)
    print("  [L1a] per-class precision/recall/F1/support:", flush=True)
    cr = classification_report(
        yt,
        yp,
        labels=list(range(NUM_REGIME_CLASSES)),
        target_names=list(REGIME_NOW_PROB_COLS),
        zero_division=0,
    )
    for line in cr.splitlines():
        print(f"    {line}", flush=True)
    w = max(10, max(len(n[:10]) for n in REGIME_NOW_PROB_COLS))
    head = " " * (w + 2) + "".join(f"{n[:10]:>{w}}" for n in REGIME_NOW_PROB_COLS)
    print("  [L1a] regime confusion matrix (row=true, col=pred):", flush=True)
    print(head, flush=True)
    for i, name in enumerate(REGIME_NOW_PROB_COLS):
        row_s = f"{name[:w]:<{w}}  " + "".join(f"{cm[i, j]:>{w}d}" for j in range(NUM_REGIME_CLASSES))
        print(f"  {row_s}", flush=True)

    print(
        f"  [L1a] vol head:  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}  corr(y,p)={corr_v:.4f}  "
        f"embed_recon_row_MSE={emb_mean:.6f}",
        flush=True,
    )
    print(
        f"  [L1a] vol_trend head:  MAE={mae_vt:.4f}  corr(y,p)={corr_vt:.4f}",
        flush=True,
    )
    print(
        f"  [L1a] time_in_regime head:  MAE={mae_tir:.4f}  corr(y,p)={corr_tir:.4f}",
        flush=True,
    )
    print(
        "  [L1a] l1a_dir_* at inference: regime-geometry only (no placeholder class probs / strength column).",
        flush=True,
    )

    _l1a_transition_val_block("transition_risk", np.concatenate(tr_t), np.concatenate(tr_s))
    print(f"  [L1a] ========== end val ({label}) report ==========\n", flush=True)


def materialize_l1a_outputs(
    model: L1AMarketTCN,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    device: torch.device,
    embed_dim: int | None = None,
) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    windows, end_idx = _build_symbol_windows(pd.concat([df[["symbol", "time_key"]], pd.DataFrame(X, columns=feature_cols)], axis=1), feature_cols, seq_len)
    ed = int(embed_dim) if embed_dim is not None else int(getattr(model, "embed_dim", _l1a_embed_dim()))
    out_cols = l1a_output_columns_with_embed_dim(ed)
    outputs = pd.DataFrame(
        {
            "symbol": df["symbol"].values,
            "time_key": pd.to_datetime(df["time_key"]),
        }
    )
    for col in out_cols:
        outputs[col] = 0.0
    outputs[L1A_REGIME_COLS] = 1.0 / float(NUM_REGIME_CLASSES)
    outputs[_bounded_scalar_cols()] = 0.0
    outputs["l1a_vol_forecast"] = float(np.nanmedian(pd.to_numeric(df["lbl_atr"], errors="coerce").fillna(1.0)))
    outputs["l1a_is_warm"] = 0.0
    if len(end_idx) == 0:
        return outputs

    ds = TensorDataset(torch.from_numpy(windows))
    infer_workers = _l1a_dataloader_workers(min(4, max(_lgbm_n_jobs(), 1)))
    dl_kwargs: dict[str, Any] = {
        "batch_size": 1024,
        "shuffle": False,
        "num_workers": infer_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    if infer_workers > 0:
        dl_kwargs["persistent_workers"] = True
    dl = DataLoader(ds, **dl_kwargs)
    dl_it = dl
    if _lgb_round_tqdm_enabled():
        dl_it = tqdm(dl, desc="[L1a] materialize outputs", file=TQDM_FILE, mininterval=0.3, unit="batch")
    regime_rows: list[np.ndarray] = []
    scalar_rows: dict[str, list[np.ndarray]] = {k: [] for k in ["transition", "vol", "vol_trend", "time_ir"]}
    embeds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl_it:
            xb = xb.to(device)
            out = model(xb)
            regime_rows.append(torch.softmax(out["regime_logits"], dim=1).cpu().numpy())
            scalar_rows["transition"].append(torch.sigmoid(out["transition_logit"]).cpu().numpy())
            scalar_rows["vol"].append(out["vol_value"].cpu().numpy())
            clip_lo = float(os.environ.get("L1A_VOL_TREND_CLIP_LO", "-3"))
            clip_hi = float(os.environ.get("L1A_VOL_TREND_CLIP_HI", "3"))
            scalar_rows["vol_trend"].append(
                np.clip(out["vol_trend_value"].detach().cpu().numpy(), clip_lo, clip_hi)
            )
            scalar_rows["time_ir"].append(out["time_in_regime_value"].detach().cpu().numpy())
            embeds.append(out["market_embed"].cpu().numpy())
    regime = np.concatenate(regime_rows, axis=0)
    outputs.loc[end_idx, L1A_REGIME_COLS] = regime
    outputs.loc[end_idx, "l1a_transition_risk"] = np.concatenate(scalar_rows["transition"], axis=0)
    outputs.loc[end_idx, "l1a_vol_forecast"] = np.clip(np.concatenate(scalar_rows["vol"], axis=0), 0.0, 5.0)
    outputs.loc[end_idx, "l1a_vol_trend"] = np.concatenate(scalar_rows["vol_trend"], axis=0)
    outputs.loc[end_idx, "l1a_time_in_regime"] = np.clip(
        np.concatenate(scalar_rows["time_ir"], axis=0), 0.0, 1.0
    )
    direction_features = _derive_l1a_direction_features(
        outputs.loc[end_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False),
        outputs.loc[end_idx, "l1a_transition_risk"].to_numpy(dtype=np.float32, copy=False),
        outputs.loc[end_idx, "l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False),
    )
    for col, values in direction_features.items():
        outputs.loc[end_idx, col] = values
    embed_mat = np.concatenate(embeds, axis=0)
    if embed_mat.shape[1] != ed:
        raise RuntimeError(
            f"L1a embed width mismatch: model expects {ed} market_embed cols but forward returned {embed_mat.shape[1]}."
        )
    embed_cols = [f"l1a_market_embed_{idx}" for idx in range(ed)]
    outputs.loc[end_idx, embed_cols] = embed_mat
    outputs.loc[end_idx, "l1a_is_warm"] = 1.0
    return outputs


def train_l1a_market_encoder(df: pd.DataFrame, feat_cols: list[str]) -> L1ATrainingBundle:
    work = df.copy(deep=False)
    feature_cols = _select_l1a_feature_cols(work, feat_cols)
    splits = build_stack_time_splits(work["time_key"])
    Xn, mean, std = _normalize_l1a_matrix(work, feature_cols, splits.train_mask)
    norm_df = pd.concat([work[["symbol", "time_key"]], pd.DataFrame(Xn, columns=feature_cols)], axis=1)
    windows, end_idx = _build_symbol_windows(norm_df, feature_cols, SEQ_LEN)
    if len(end_idx) == 0:
        raise RuntimeError("L1a: no valid sequence windows were created.")

    targets = _build_l1a_targets(work)
    window_train = splits.train_mask[end_idx]
    window_cal = splits.cal_mask[end_idx]
    window_val = splits.l2_val_mask[end_idx]
    if not window_val.any():
        raise RuntimeError("L1a: L2 validation window is empty for validation.")
    if not window_cal.any():
        raise RuntimeError("L1a: calibration window is empty for diagnostics.")

    X_t = torch.from_numpy(windows.astype(np.float32, copy=False))
    ds = TensorDataset(
        X_t,
        torch.from_numpy(targets["regime"][end_idx].astype(np.int64)),
        torch.from_numpy(targets["transition_risk"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["vol_forecast"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["vol_trend"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["time_in_regime"][end_idx].astype(np.float32)),
    )
    train_ds = TensorDataset(*[tensor[window_train] for tensor in ds.tensors])
    val_ds = TensorDataset(*[tensor[window_val] for tensor in ds.tensors])
    cal_ds = TensorDataset(*[tensor[window_cal] for tensor in ds.tensors])
    batch_size = 512 if FAST_TRAIN_MODE else 1024
    loader_workers = _l1a_dataloader_workers(min(4, max(_lgbm_n_jobs(), 1)))
    pin_memory = DEVICE.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_dl = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    cal_dl = DataLoader(cal_ds, shuffle=False, **loader_kwargs)

    embed_dim = _l1a_embed_dim()

    log_layer_banner("[L1a] Sequence Market Encoder (TCN)")
    log_time_key_arrays(
        "L1a",
        work.iloc[end_idx[window_train]]["time_key"],
        work.iloc[end_idx[window_val]]["time_key"],
        train_label="window train (end_idx in train split)",
        val_label="window val (end_idx in l2_val split)",
        extra_note="Primary L1a early stopping/reporting uses l2_val end-bars; full cal remains secondary diagnostic.",
    )
    w_tr = windows[window_train]
    log_numpy_x_stats("L1a", w_tr.reshape(w_tr.shape[0], -1), label="windows[train] (flattened seq×feat)")
    n_row = len(work)
    n_warm = int(len(end_idx))
    print(
        f"  [L1a] warm rows (full seq, materialize is_warm=1): {n_warm:,} ({100.0 * n_warm / max(n_row, 1):.2f}%)",
        flush=True,
    )
    print(
        f"  [L1a] cold rows (no full window / is_warm=0): {n_row - n_warm:,} ({100.0 * (n_row - n_warm) / max(n_row, 1):.2f}%)",
        flush=True,
    )
    out_cn = l1a_output_columns_with_embed_dim(embed_dim)
    print(f"  [L1a] output column count: {len(out_cn)} (expect {len(out_cn)})", flush=True)
    print(f"  [L1a] output columns: {out_cn}", flush=True)
    print(f"  [L1a] seq input: seq_len={SEQ_LEN}  input_feats={len(feature_cols)}", flush=True)
    print(f"  [L1a] artifact dir: {MODEL_DIR}", flush=True)
    print(
        f"  [L1a] will write: {artifact_path(L1A_MODEL_FILE)} | {artifact_path(L1A_META_FILE)} | {artifact_path(L1A_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    print(
        "  [L1a] note: forward uses this run's weights/data (not loading L1a from disk for features).",
        flush=True,
    )
    log_label_baseline("l1a_regime", targets["regime"][end_idx][window_train], task="cls")
    log_label_baseline("l1a_transition_risk", targets["transition_risk"][end_idx][window_train], task="cls")
    log_label_baseline("l1a_vol_forecast", targets["vol_forecast"][end_idx][window_train], task="reg")
    log_label_baseline("l1a_vol_trend", targets["vol_trend"][end_idx][window_train], task="reg")
    log_label_baseline("l1a_time_in_regime", targets["time_in_regime"][end_idx][window_train], task="reg")

    loss_weights = _l1a_loss_weights()
    regime_loss = _l1a_regime_loss(targets["regime"][end_idx][window_train], device=DEVICE)
    transition_loss = _l1a_transition_loss(targets["transition_risk"][end_idx][window_train], device=DEVICE)
    ch = _l1a_tcn_channels()
    td = _l1a_tcn_dropout()
    rd = _l1a_readout_dropout()
    hd_drop = _l1a_head_dropout()
    model = L1AMarketTCN(
        len(feature_cols),
        channels=ch,
        seq_len=SEQ_LEN,
        readout_type=_l1a_readout_type(),
        min_attention_seq_len=_l1a_min_attention_seq_len(),
        tcn_dropout=td,
        readout_dropout=rd,
        head_dropout=hd_drop,
        embed_dim=embed_dim,
    ).to(DEVICE)
    lr = float(os.environ.get("L1A_LR", "5e-4"))
    wd = float(os.environ.get("L1A_WEIGHT_DECAY", "1e-3"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    T0 = max(1, int(os.environ.get("L1A_COS_T0", "5")))
    Tm = max(1, int(os.environ.get("L1A_COS_T_MULT", "2")))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm)
    max_epochs = 8 if FAST_TRAIN_MODE else 24
    patience = 4 if FAST_TRAIN_MODE else 8
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    epoch_bar = trange(
        max_epochs,
        desc="[L1a] epochs",
        unit="ep",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    print(
        f"  [L1a] readout={model.readout_type}  min_attention_seq_len={model.min_attention_seq_len}  "
        f"tcn_channels={ch}  tcn_dropout={td}  "
        f"lr={lr}  weight_decay={wd}  cosine(T0={T0},T_mult={Tm})  loss_weights={loss_weights}",
        flush=True,
    )
    for epoch in epoch_bar:
        tr_loss = _train_epoch(
            model,
            train_dl,
            optimizer,
            DEVICE,
            regime_loss=regime_loss,
            transition_loss=transition_loss,
            loss_weights=loss_weights,
        )
        va_loss = _eval_epoch(
            model,
            val_dl,
            DEVICE,
            regime_loss=regime_loss,
            transition_loss=transition_loss,
            loss_weights=loss_weights,
        )
        scheduler.step()
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}", refresh=False)
        print(f"  [L1a] epoch={epoch + 1:02d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("L1a: training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    _log_l1a_val_metrics(model, val_dl, DEVICE, label="l2_val")
    if window_cal.sum() != window_val.sum():
        _log_l1a_val_metrics(model, cal_dl, DEVICE, label="cal_full")

    outputs = materialize_l1a_outputs(
        model, work, feature_cols, mean=mean, std=std, seq_len=SEQ_LEN, device=DEVICE, embed_dim=model.embed_dim
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, L1A_MODEL_FILE))
    meta = {
        "schema_version": L1A_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "readout_type": model.readout_type,
        "min_attention_seq_len": model.min_attention_seq_len,
        "mean": mean,
        "std": std,
        "output_cols": l1a_output_columns_with_embed_dim(embed_dim),
        "device": str(DEVICE),
        "model_file": L1A_MODEL_FILE,
        "output_cache_file": L1A_OUTPUT_CACHE_FILE,
        "transition_target_semantics": "probability of any regime change within decision_horizon_bars",
        "direction_target_semantics": (
            "l1a_dir_* (bull/bear geometry) from regime probabilities + transition/vol; no placeholder sign probs."
        ),
        "l1a_vol_trend_target": f"per-symbol range_norm[t]-range_norm[t-{os.environ.get('L1A_VOL_TREND_LAG', '5')}], clipped",
        "l1a_time_in_regime_target": f"min(run_length,cap)/cap from state_label runs; cap={os.environ.get('L1A_TIME_IN_REGIME_CAP', '120')}",
        "embed_dim": int(embed_dim),
        "l1a_direction_arch": "regime_derived_only",
        "loss_weights": loss_weights,
        "tcn_channels": list(ch),
        "tcn_dropout": float(td),
        "readout_dropout": float(rd),
        "head_dropout": float(hd_drop),
        "l1a_lr": float(lr),
        "l1a_weight_decay": float(wd),
        "l1a_cosine_T0": int(T0),
        "l1a_cosine_T_mult": int(Tm),
        "l1a_regime_label_smoothing": float(os.environ.get("L1A_REGIME_LABEL_SMOOTHING", "0.1")),
    }
    with open(os.path.join(MODEL_DIR, L1A_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1A_OUTPUT_CACHE_FILE)
    print(f"  [L1a] model saved -> {os.path.join(MODEL_DIR, L1A_MODEL_FILE)}", flush=True)
    print(f"  [L1a] meta saved  -> {os.path.join(MODEL_DIR, L1A_META_FILE)}", flush=True)
    print(f"  [L1a] cache saved -> {cache_path}", flush=True)
    return L1ATrainingBundle(model=model, meta=meta, outputs=outputs)


def load_l1a_market_encoder() -> tuple[L1AMarketTCN, dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L1A_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L1A_SCHEMA_VERSION:
        raise RuntimeError(
            f"L1a schema mismatch: artifact has {meta.get('schema_version')} but code expects {L1A_SCHEMA_VERSION}. "
            f"Retrain L1a so artifacts match schema {L1A_SCHEMA_VERSION}."
        )
    feature_cols = list(meta["feature_cols"])
    ch = meta.get("tcn_channels")
    if ch is None:
        ch = [64, 64, 128]
    else:
        ch = [int(x) for x in ch]
    td = float(meta.get("tcn_dropout", 0.15))
    rd = float(meta.get("readout_dropout", 0.10))
    hd_drop = float(meta.get("head_dropout", 0.1))
    embed_dim = int(meta.get("embed_dim", _l1a_embed_dim()))
    model = L1AMarketTCN(
        len(feature_cols),
        channels=ch,
        seq_len=int(meta.get("seq_len", SEQ_LEN)),
        readout_type=str(meta.get("readout_type", _l1a_readout_type())),
        min_attention_seq_len=int(meta.get("min_attention_seq_len", _l1a_min_attention_seq_len())),
        tcn_dropout=td,
        readout_dropout=rd,
        head_dropout=hd_drop,
        embed_dim=embed_dim,
    ).to(DEVICE)
    state = torch.load(os.path.join(MODEL_DIR, meta.get("model_file", L1A_MODEL_FILE)), map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "L1a checkpoint is incompatible with the current head contract. "
            "Retrain L1a so the saved model/meta match schema "
            f"{L1A_SCHEMA_VERSION}."
        ) from exc
    model.eval()
    return model, meta


def infer_l1a_market_encoder(model: L1AMarketTCN, meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy(deep=False)
    feature_cols = list(meta["feature_cols"])
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    seq_len = int(meta.get("seq_len", SEQ_LEN))
    return materialize_l1a_outputs(
        model, work, feature_cols, mean=mean, std=std, seq_len=seq_len, device=DEVICE, embed_dim=model.embed_dim
    )

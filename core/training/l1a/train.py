from __future__ import annotations

import inspect
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist
from typing import Any, Tuple

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
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import trange, tqdm

from core.models.tcn_pa_state import L1AGatedTemporalBlock, TemporalAttentionReadout
from core.training.prep.feature_registry import (
    L1A_EXTRA_EXCLUDE_PREFIXES,
    l1_ctx_stagger_enabled,
    l1a_extra_accepts_pa_ctx,
    l1a_preferred_columns,
)
from core.training.common.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1A_META_FILE,
    L1A_MODEL_FILE,
    L1A_OUTPUT_CACHE_FILE,
    L1A_REGIME_COLS,
    L1A_SCHEMA_VERSION,
    L1A_TRAIN_RESUME_FILE,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
    REGIME_NOW_PROB_COLS,
    l1a_straddle_edge_head_enabled,
    l1a_time_in_regime_head_enabled,
    l1a_vol_trend_head_enabled,
)

from core.training.prep.data_prep import _create_tcn_windows
from core.training.common.lgbm_utils import _lgbm_n_jobs, _options_target_config, _tqdm_stream
from core.training.logging.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.training.common.val_metrics_extra import brier_binary, brier_multiclass, ece_binary, ece_multiclass_maxprob, pearson_corr
from core.training.common.stack_v2_common import (
    build_stack_time_splits,
    compute_transition_event_labels,
    l1_expanding_oof_window_folds,
    l1a_resolve_transition_horizon_bars,
    l1_oof_folds_from_env,
    l1_oof_mode_from_env,
    l1a_expand_oof_val_windows_from_env,
    l2_val_start_time,
    log_label_baseline,
    save_output_cache,
    split_mask_for_tuning_and_report,
    time_blocked_fold_masks,
)
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry
from core.training.tcn.tcn_constants import DEVICE, SEQ_LEN as _TCN_SEQ_LEN_DEFAULT


def _l1a_seq_len() -> int:
    """Bars per sequence for L1a; set ``L1A_SEQ_LEN`` to slim without changing global ``TCN_SEQ_LEN``."""
    return max(20, int(os.environ.get("L1A_SEQ_LEN", os.environ.get("TCN_SEQ_LEN", str(_TCN_SEQ_LEN_DEFAULT)))))


@dataclass(frozen=True)
class L1aAmpSettings:
    """CUDA: opt-in via ``L1A_AMP=1``. macOS + MPS: float16 autocast always on (no env)."""

    enabled: bool
    device_type: str
    dtype: torch.dtype


def _l1a_is_macos() -> bool:
    return sys.platform == "darwin"


def _l1a_build_amp(device: torch.device) -> tuple[L1aAmpSettings, GradScaler | None]:
    if _l1a_is_macos() and device.type == "mps":
        return L1aAmpSettings(True, "mps", torch.float16), None
    raw_amp = os.environ.get("L1A_AMP", "").strip().lower()
    amp_enabled = raw_amp in {"1", "true", "yes"} or (raw_amp == "" and device.type == "cuda")
    if not amp_enabled:
        return L1aAmpSettings(False, device.type, torch.float32), None
    if device.type != "cuda":
        return L1aAmpSettings(False, device.type, torch.float32), None
    raw = (os.environ.get("L1A_AMP_DTYPE", "auto") or "auto").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        dt = torch.bfloat16
    elif raw in {"fp16", "float16"}:
        dt = torch.float16
    elif raw == "fp32":
        return L1aAmpSettings(False, device.type, torch.float32), None
    else:
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler: GradScaler | None = None if dt == torch.bfloat16 else GradScaler(device.type)
    return L1aAmpSettings(True, device.type, dt), scaler


def _l1a_autocast(amp: L1aAmpSettings):
    return autocast(amp.device_type, enabled=amp.enabled, dtype=amp.dtype)


def _l1a_maybe_compile(module: nn.Module) -> nn.Module:
    if os.environ.get("L1A_TORCH_COMPILE", "0").strip().lower() not in {"1", "true", "yes"}:
        return module
    if not hasattr(torch, "compile"):
        print("  [L1a][warn] L1A_TORCH_COMPILE=1 but torch.compile unavailable", flush=True)
        return module
    mode = (os.environ.get("L1A_TORCH_COMPILE_MODE", "default") or "default").strip()
    print(f"  [L1a] torch.compile(mode={mode!r}) — first step may be slow", flush=True)
    return torch.compile(module, mode=mode)  # type: ignore[no-any-return]


def _l1a_raw_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_orig_mod", module)


def _l1a_state_dict_for_save(module: nn.Module) -> dict[str, Any]:
    return _l1a_raw_module(module).state_dict()


def _l1a_nb(device: torch.device) -> bool:
    return device.type == "cuda"


def _l1a_progress_tqdm_enabled() -> bool:
    """Epoch / fold / materialize tqdm for L1a. Independent of LightGBM TTY rules; logs to ``_tqdm_stream()`` (stderr by default)."""
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("L1A_DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    raw = (os.environ.get("L1A_PROGRESS_TQDM", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _l1a_batch_progress_enabled() -> bool:
    """Train/val per-batch tqdm (nested under epoch bar). Default off for speed; set L1A_TQDM_BATCH=1 to enable."""
    if not _l1a_progress_tqdm_enabled():
        return False
    raw = (os.environ.get("L1A_TQDM_BATCH", "0") or "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _bounded_scalar_cols() -> list[str]:
    cols = ["l1a_transition_risk", "l1a_state_persistence"]
    if l1a_straddle_edge_head_enabled():
        cols.append("l1a_straddle_edge")
    return cols


def _l1a_embed_dim() -> int:
    return max(4, int(os.environ.get("L1A_EMBED_DIM", "8")))


def l1a_output_columns_with_embed_dim(embed_dim: int) -> list[str]:
    d = max(4, int(embed_dim))
    mid = [
        "l1a_transition_risk",
        "l1a_vol_forecast",
        "l1a_vol_trend",
        "l1a_time_in_regime",
        "l1a_state_persistence",
    ]
    if l1a_straddle_edge_head_enabled():
        mid.append("l1a_straddle_edge")
    return (
        list(L1A_REGIME_COLS)
        + mid
        + [f"l1a_market_embed_{idx}" for idx in range(d)]
        + ["l1a_is_warm"]
    )


def l1a_output_columns() -> list[str]:
    return l1a_output_columns_with_embed_dim(_l1a_embed_dim())


def _l1a_readout_type() -> str:
    return (os.environ.get("L1A_READOUT_TYPE", "attention").strip().lower() or "attention")


def _l1a_min_attention_seq_len() -> int:
    return max(1, int(os.environ.get("L1A_MIN_ATTENTION_SEQ_LEN", os.environ.get("TCN_MIN_ATTENTION_SEQ_LEN", "4"))))


def _l1a_dataloader_workers(default: int) -> int:
    raw = os.environ.get("L1A_DATALOADER_WORKERS", "").strip()
    cap = max(1, int(os.environ.get("L1A_DATALOADER_WORKERS_CAP", "8")))
    if raw:
        workers = int(raw)
    else:
        workers = int(default)
        if sys.platform == "darwin" and workers > 0:
            workers = 0
    return max(0, min(workers, cap))


def _l1a_configure_torch_backends(device: torch.device) -> None:
    """Matmul precision + cuDNN benchmark (fixed conv shapes) — code-level throughput defaults."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if device.type == "cuda":
        if os.environ.get("L1A_CUDNN_BENCHMARK", "1").strip().lower() in {"0", "false", "no"}:
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def _l1a_adamw_kwargs(device: torch.device) -> dict[str, Any]:
    """Fused (CUDA) or multi-tensor foreach AdamW when supported; disable via L1A_ADAMW_FUSED=0 / L1A_ADAMW_FOREACH=0."""
    kw: dict[str, Any] = {}
    try:
        params = inspect.signature(torch.optim.AdamW.__init__).parameters
    except (ValueError, TypeError):
        return kw
    fused_on = (
        device.type == "cuda"
        and "fused" in params
        and os.environ.get("L1A_ADAMW_FUSED", "1").strip().lower() not in {"0", "false", "no"}
    )
    if fused_on:
        kw["fused"] = True
    elif (
        "foreach" in params
        and os.environ.get("L1A_ADAMW_FOREACH", "1").strip().lower() not in {"0", "false", "no"}
    ):
        kw["foreach"] = True
    return kw


def _l1a_tcn_channels() -> list[int]:
    raw = os.environ.get("L1A_TCN_CHANNELS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    # Default 5 blocks → larger receptive field (dilations 1,2,4,8,16). Override: L1A_TCN_CHANNELS=64,64,64,128,128
    return [64, 64, 64, 128, 128]


def _l1a_tcn_kernel_size() -> int:
    """Causal TCN kernel per layer; k=5 increases receptive field vs k=3 at similar depth."""
    return max(2, int(os.environ.get("L1A_TCN_KERNEL_SIZE", "3")))


def _l1a_tcn_receptive_field_steps(*, n_layers: int, kernel_size: int) -> int:
    """Stack R_0=1, R_{i+1} = R_i + (k-1)*2^i (L1a uses dilation 2**idx per layer)."""
    k = int(kernel_size)
    rf = 1
    for i in range(int(n_layers)):
        rf += (k - 1) * (2**i)
    return int(rf)


def _l1a_tcn_dropout() -> float:
    return float(os.environ.get("L1A_TCN_DROPOUT", "0.3"))


def _l1a_readout_dropout() -> float:
    return float(os.environ.get("L1A_READOUT_DROPOUT", "0.25"))


def _l1a_head_dropout() -> float:
    return float(os.environ.get("L1A_HEAD_DROPOUT", "0.3"))


def _l1a_uw_val_metric() -> str:
    raw = (os.environ.get("L1A_UW_VAL_METRIC", "geom") or "geom").strip().lower()
    if raw in {"geom", "uw_total", "legacy"}:
        return raw
    return "geom"


def _l1a_uw_lr_ratio() -> float:
    return max(1e-6, float(os.environ.get("L1A_UW_LR_RATIO", "2.0")))


def _l1a_regime_aux_coef() -> float:
    """Scales the hierarchical aux term: 5→2 low/high-vol CE on ``base_regime_logits`` (not full 5-way focal)."""
    return float(os.environ.get("L1A_REGIME_AUX_COEF", os.environ.get("L1A_REGIME_AUX_WEIGHT", "0.20")))


def _l1a_transition_persist_coef() -> float:
    return float(
        os.environ.get(
            "L1A_TRANSITION_PERSIST_COEF",
            os.environ.get("L1A_PERSISTENCE_AUX_WEIGHT", "0.35"),
        )
    )


class L1aMultiTaskUncertaintyWeights(nn.Module):
    """Kendall et al. (2018) homoscedastic uncertainty weighting — one log-variance per L1a task."""

    # cls: CE/BCE-scale losses use L/(2σ²)+log σ  →  exp(-log_var)*L + 0.5*log_var with log_var=log(σ²)
    # reg: MSE-scale uses 1/(2σ²)*L + log σ  →  0.5*exp(-log_var)*L + 0.5*log_var
    TASK_TYPES: dict[str, str] = {
        "regime": "cls",
        "regime_aux": "cls",
        "transition": "cls",
        "vol": "reg",
        "vol_trend": "reg",
        "time_in_regime": "reg",
        "embed_recon": "reg",
        "straddle_edge": "reg",
    }

    @staticmethod
    def build_task_order(
        *,
        straddle_edge: bool,
        vol_trend_head: bool,
        time_in_regime_head: bool,
    ) -> tuple[str, ...]:
        t: list[str] = [
            "regime",
            "regime_aux",
            "transition",
            "vol",
        ]
        if vol_trend_head:
            t.append("vol_trend")
        if time_in_regime_head:
            t.append("time_in_regime")
        t.append("embed_recon")
        if straddle_edge:
            t.append("straddle_edge")
        return tuple(t)

    def __init__(
        self,
        *,
        device: torch.device,
        straddle_edge: bool,
        vol_trend_head: bool,
        time_in_regime_head: bool,
    ):
        super().__init__()
        self.task_order: tuple[str, ...] = self.build_task_order(
            straddle_edge=straddle_edge,
            vol_trend_head=vol_trend_head,
            time_in_regime_head=time_in_regime_head,
        )
        params: dict[str, nn.Parameter] = {}
        for name in self.task_order:
            params[name] = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
        self.log_vars = nn.ParameterDict(params)

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        if not losses:
            z = next(iter(self.log_vars.parameters()))
            return torch.zeros((), device=z.device, dtype=torch.float32), {}
        dev = next(iter(losses.values())).device
        total = torch.zeros((), device=dev, dtype=torch.float32)
        diag: dict[str, float] = {}
        for name in self.task_order:
            if name not in losses:
                continue
            log_var = self.log_vars[name].view(())
            raw = losses[name]
            prec = torch.exp(-log_var)
            kind = self.TASK_TYPES[name]
            if kind == "reg":
                weighted = 0.5 * prec * raw + 0.5 * log_var
            else:
                weighted = prec * raw + 0.5 * log_var
            total = total + weighted
            diag[f"uw_{name}_log_var"] = float(log_var.detach().item())
            diag[f"uw_{name}_precision"] = float(prec.detach().item())
            diag[f"uw_{name}_sigma2"] = float(torch.exp(log_var).detach().item())
            diag[f"uw_{name}_raw"] = float(raw.detach().item())
            diag[f"uw_{name}_weighted"] = float(weighted.detach().item())
        return total, diag


def _l1a_build_uw_module(device: torch.device) -> L1aMultiTaskUncertaintyWeights:
    return L1aMultiTaskUncertaintyWeights(
        device=device,
        straddle_edge=l1a_straddle_edge_head_enabled(),
        vol_trend_head=l1a_vol_trend_head_enabled(),
        time_in_regime_head=l1a_time_in_regime_head_enabled(),
    ).to(device)


def _l1a_forward_decoder_kw(
    model: L1AMarketTCN,
    y_vol_trend: torch.Tensor,
    y_time_ir: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """When auxiliary heads are disabled, feed decoder from targets (train/val) or statics (materialize)."""
    kw: dict[str, torch.Tensor] = {}
    if model.vol_trend_head is None:
        kw["decoder_vol_trend"] = y_vol_trend
    if model.time_in_regime_head is None:
        kw["decoder_time_in_regime"] = y_time_ir
    return kw


def _l1a_log_uw_auto_weights(uw: L1aMultiTaskUncertaintyWeights, *, prefix: str = "  [L1a]") -> None:
    """Log σ² and precision exp(-log_var) per task after an epoch."""
    for name in uw.task_order:
        lv = float(uw.log_vars[name].detach().item())
        s2 = float(math.exp(lv))
        prec = float(math.exp(-lv))
        print(f"{prefix} uw_task {name:16s}  σ²={s2:.4f}  prec={prec:.4f}", flush=True)


def _l1a_optimizer(
    model: L1AMarketTCN,
    uw: L1aMultiTaskUncertaintyWeights | None,
    lr: float,
    wd: float,
) -> torch.optim.AdamW:
    adam_kw = _l1a_adamw_kwargs(DEVICE)
    if uw is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, **adam_kw)
    ratio = _l1a_uw_lr_ratio()
    return torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(uw.parameters()), "lr": lr * ratio, "weight_decay": 0.0},
        ],
        **adam_kw,
    )


def _l1a_clip_grad_norm(
    model: L1AMarketTCN,
    uw: L1aMultiTaskUncertaintyWeights | None,
    max_norm: float,
) -> None:
    params: list[torch.nn.Parameter] = list(model.parameters())
    if uw is not None:
        params.extend(list(uw.parameters()))
    torch.nn.utils.clip_grad_norm_(params, max_norm)


def _l1a_regime_loss(
    regime_train_labels: np.ndarray,
    *,
    device: torch.device,
) -> nn.Module:
    _ = regime_train_labels, device  # API stable for call sites; no per-class weights
    ls = float(os.environ.get("L1A_REGIME_LABEL_SMOOTHING", "0.1"))
    return nn.CrossEntropyLoss(label_smoothing=ls)


def _l1a_coarse_vol_regime_aux_loss(regime_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Hierarchical aux on base head: 5-class logits → 2-way low-vol vs high-vol CE.

    Groups (vol lifecycle indices): {0,1} vol_compress + vol_breakout → coarse 0 (low);
    {2,3,4} vol_trending + vol_exhaust + vol_mean_revert → coarse 1 (high).
    Straddle-relevant split: long vs short straddle is mostly a low-vol / high-vol call.
    """
    coarse_target = (targets >= 2).long()
    coarse_logits = torch.stack(
        [
            regime_logits[:, 0:2].logsumexp(dim=1),
            regime_logits[:, 2:5].logsumexp(dim=1),
        ],
        dim=1,
    )
    return F.cross_entropy(coarse_logits, coarse_target)


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
    """BCE pos_weight = neg/pos from the **passed label slice** (always matches calibrated H / fold subset)."""
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


def _l1a_log_regime_class_distribution(y: np.ndarray, mask: np.ndarray | None, *, title: str) -> None:
    y = np.asarray(y, dtype=np.int64).ravel()
    m = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    if len(m) != len(y):
        raise ValueError("regime distribution: mask length mismatch")
    yy = y[m]
    n = int(yy.size)
    print(f"  [L1a] {title}  n={n:,}", flush=True)
    if n == 0:
        return
    for c in range(NUM_REGIME_CLASSES):
        cnt = int(np.sum(yy == c))
        print(
            f"    {c} {REGIME_NOW_PROB_COLS[c]:20s}  {cnt:>10,}  ({100.0 * cnt / n:5.2f}%)",
            flush=True,
        )


def _l1a_log_transition_h_stability_diag(
    work: pd.DataFrame,
    state: np.ndarray,
    symbols: np.ndarray,
    norm_mask: np.ndarray,
    forecast_h: int,
) -> None:
    """Optional: time-block splits on norm_mask, calibrate H on each train slice — report variance."""
    raw = (os.environ.get("L1A_TRANSITION_H_DIAG_FOLDS", "") or "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return
    k = max(2, int(raw))
    try:
        folds = time_blocked_fold_masks(work["time_key"], norm_mask, k, context="L1a transition H diag")
    except RuntimeError as ex:
        print(f"  [L1a] transition H diag skipped: {ex}", flush=True)
        return
    hs: list[int] = []
    for tr_m, _va_m in folds:
        h, _meta = l1a_resolve_transition_horizon_bars(
            state, symbols, tr_m, forecast_horizon=forecast_h
        )
        hs.append(int(h))
    arr = np.asarray(hs, dtype=np.float64)
    std = float(arr.std()) if arr.size else 0.0
    print(
        f"  [L1a] transition H stability diag: {k} time blocks on norm_mask (calibrate per train slice) → "
        f"H_list={hs}  mean={float(arr.mean()):.2f}  std={std:.2f}",
        flush=True,
    )


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
    """Per-symbol causal diff: out[t] = values[t] - values[t-lag] (only past/current bars; no leakage)."""
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


def _l1a_time_in_regime_transform() -> str:
    transform = (os.environ.get("L1A_TIME_IN_REGIME_TRANSFORM", "sqrt").strip().lower() or "sqrt")
    if transform not in {"none", "sqrt"}:
        transform = "sqrt"
    return transform


def _transform_time_in_regime_target(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if _l1a_time_in_regime_transform() == "sqrt":
        arr = np.sqrt(arr)
    return arr.astype(np.float32, copy=False)


def _inverse_time_in_regime_target(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if _l1a_time_in_regime_transform() == "sqrt":
        arr = np.square(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _l1a_rank_extra_by_temporal_strength(df: pd.DataFrame, extra: list[str]) -> list[str]:
    """Prefer PA extras with stronger bar-to-bar persistence (|lag-1 ACF|), max over symbols."""
    if not extra:
        return extra
    max_rows = max(1000, int(os.environ.get("L1A_EXTRA_RANK_MAX_ROWS", "80000")))
    scores: dict[str, float] = {c: 0.0 for c in extra}
    for _, grp in df.groupby("symbol", sort=False):
        g = grp.sort_values("time_key")
        if len(g) > max_rows:
            g = g.iloc[:max_rows]
        for c in extra:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            ac = v.autocorr(lag=1)
            ac_abs = abs(float(ac)) if pd.notna(ac) and np.isfinite(ac) else 0.0
            scores[c] = max(scores[c], ac_abs)
    return sorted(extra, key=lambda c: scores.get(c, 0.0), reverse=True)


def _select_l1a_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    preferred = list(l1a_preferred_columns())
    extra = [
        c
        for c in feat_cols
        if c.startswith("pa_")
        and not c.startswith(L1A_EXTRA_EXCLUDE_PREFIXES)
        and l1a_extra_accepts_pa_ctx(c)
        and c not in preferred
    ]
    rank_mode = (os.environ.get("L1A_EXTRA_RANK", "temporal") or "temporal").strip().lower()
    if rank_mode not in {"order", "list", "feat_cols"}:
        extra = _l1a_rank_extra_by_temporal_strength(df, extra)
    max_extra = max(0, int(os.environ.get("L1A_MAX_EXTRA_FEATURES", "20")))
    cols = [c for c in preferred + extra[:max_extra] if c in df.columns]
    time_key = pd.to_datetime(df["time_key"])
    minutes = (time_key.dt.hour * 60 + time_key.dt.minute).astype(np.float32)
    df["l1a_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)
    cols.append("l1a_session_progress")
    return cols


def _robust_sigma_from_values(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = float(mad / 0.67448975) if mad > 0 else 0.0
    return med, mad, sigma


def _l1a_clip_upper(
    values: np.ndarray,
    *,
    mode: str,
    quantile_q: float,
    alpha: float,
    floor: float,
    ceiling: float,
) -> tuple[float, dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    meta: dict[str, Any] = {
        "mode": str(mode),
        "fit_n": int(vals.size),
        "alpha": float(alpha),
        "quantile_q": float(quantile_q),
    }
    if vals.size == 0:
        clip = float(np.clip(ceiling, floor, ceiling))
        meta["fallback_reason"] = "no_finite_values"
        meta["clip"] = float(clip)
        return clip, meta
    if mode == "mad_z":
        med, mad, sigma = _robust_sigma_from_values(vals)
        if not np.isfinite(sigma) or sigma <= 1e-8:
            clip_q = float(np.quantile(vals, quantile_q))
            clip = float(np.clip(clip_q, floor, ceiling))
            meta.update(
                {
                    "statistical_principle": "quantile_clip_fallback_from_zero_mad",
                    "fallback_reason": "mad_or_sigma_zero",
                    "median": float(med),
                    "mad": float(mad),
                    "sigma_robust": float(sigma),
                    "clip": float(clip),
                }
            )
            return clip, meta
        z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
        clip = float(np.clip(med + z * max(sigma, 1e-8), floor, ceiling))
        meta.update(
            {
                "statistical_principle": "robust_median_mad_significance",
                "median": float(med),
                "mad": float(mad),
                "sigma_robust": float(sigma),
                "z_value": float(z),
                "clip": float(clip),
            }
        )
        return clip, meta
    clip_q = float(np.quantile(vals, quantile_q))
    clip = float(np.clip(clip_q, floor, ceiling))
    meta.update(
        {
            "statistical_principle": "quantile_clip",
            "clip": float(clip),
        }
    )
    return clip, meta


def _l1_oof_auto_cap_enabled() -> bool:
    return os.environ.get("L1_OOF_AUTO_CAP", "1").strip().lower() in {"1", "true", "yes"}


def _l1a_cap_oof_folds(requested: int, n_pool_windows: int) -> tuple[int, str]:
    """Cap K so each val fold has enough windows; avoids unstable median(best_epoch) on short spans."""
    if requested < 2:
        return requested, ""
    min_w = max(200, int(os.environ.get("L1_OOF_MIN_WINDOWS_PER_FOLD", "4000")))
    max_k = max(1, n_pool_windows // max(min_w, 1))
    max_k = min(max_k, requested, 128)
    if max_k < 2:
        return 1, (
            f"OOF fold cap: requested={requested} pool_windows={n_pool_windows:,} "
            f"but L1_OOF_MIN_WINDOWS_PER_FOLD={min_w} implies <2 folds → fallback to legacy (L1_OOF_FOLDS=1 path)."
        )
    if max_k < requested:
        per = n_pool_windows // max_k
        return max_k, (
            f"OOF folds capped {requested}→{max_k} (~{per:,} val windows/fold, "
            f"pool_windows={n_pool_windows:,}, L1_OOF_MIN_WINDOWS_PER_FOLD={min_w})."
        )
    return requested, ""


def _l1a_adaptive_range_clip_enabled() -> bool:
    return os.environ.get("L1A_ADAPTIVE_RANGE_CLIP", "1").strip().lower() in {"1", "true", "yes"}


def _l1a_adaptive_vol_forecast_clip_enabled() -> bool:
    return os.environ.get("L1A_ADAPTIVE_VOL_FORECAST_CLIP", "1").strip().lower() in {"1", "true", "yes"}


def _l1a_adaptive_range_norm_ceiling(fit_vals: np.ndarray) -> tuple[float, dict[str, Any]]:
    vals = np.asarray(fit_vals, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    ceiling_min = float(os.environ.get("L1A_RANGE_NORM_CEILING_MIN", "4.0"))
    ceiling_max = float(os.environ.get("L1A_RANGE_NORM_CEILING_MAX", "20.0"))
    clip_q = float(np.clip(float(os.environ.get("L1A_RANGE_NORM_CLIP_Q", "0.995")), 0.90, 1.0))
    meta: dict[str, Any] = {
        "fit_n": int(vals.size),
        "ceiling_min": float(ceiling_min),
        "ceiling_max": float(ceiling_max),
    }
    if vals.size == 0:
        c = float(np.clip(8.0, ceiling_min, ceiling_max))
        meta.update({"clip": c, "fallback_reason": "no_finite_values", "statistical_principle": "default"})
        return c, meta
    med, mad, sigma = _robust_sigma_from_values(vals)
    if not np.isfinite(sigma) or sigma <= 1e-8:
        c = float(np.quantile(vals, clip_q))
        meta["statistical_principle"] = "quantile_clip_fallback_from_zero_mad"
    else:
        c = float(med + 4.0 * max(sigma, 1e-8))
        meta.update(
            {
                "statistical_principle": "robust_median_mad_4sigma",
                "median": float(med),
                "mad": float(mad),
                "sigma_robust": float(sigma),
            }
        )
    c = float(np.clip(c, ceiling_min, ceiling_max))
    meta["clip"] = float(c)
    return c, meta


def _l1a_compute_vol_forecast_materialize_clip(vol_targets: np.ndarray) -> tuple[float, dict[str, Any]]:
    vals = np.asarray(vol_targets, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    hard_floor = float(os.environ.get("L1A_VOL_FORECAST_CLIP_FLOOR", "3.0"))
    hard_ceiling = float(os.environ.get("L1A_VOL_FORECAST_CLIP_CEILING", "15.0"))
    clip_q = float(np.clip(float(os.environ.get("L1A_VOL_MATERIALIZE_CLIP_Q", "0.995")), 0.90, 1.0))
    method = (os.environ.get("L1A_VOL_MATERIALIZE_CLIP_METHOD", "mad_z") or "mad_z").strip().lower()
    meta: dict[str, Any] = {
        "fit_n": int(vals.size),
        "method": str(method),
        "hard_floor": float(hard_floor),
        "hard_ceiling": float(hard_ceiling),
    }
    if vals.size == 0:
        u = float(np.clip(5.0, hard_floor, hard_ceiling))
        meta.update({"clip_upper": u, "fallback_reason": "no_finite_values"})
        return u, meta
    if method == "mad_z":
        med, mad, sigma = _robust_sigma_from_values(vals)
        if not np.isfinite(sigma) or sigma <= 1e-8:
            u = float(np.quantile(vals, clip_q))
            meta["statistical_principle"] = "quantile_fallback"
        else:
            u = float(med + 4.0 * max(sigma, 1e-8))
            meta.update({"median": float(med), "sigma_robust": float(sigma), "statistical_principle": "mad_4sigma"})
    else:
        u = float(np.quantile(vals, clip_q))
        meta["statistical_principle"] = "quantile"
    u = float(np.clip(u, hard_floor, hard_ceiling))
    meta["clip_upper"] = u
    return u, meta


def _l1a_targets_build_tqdm_enabled() -> bool:
    """Phased tqdm inside ``_build_l1a_targets`` (long on ~1M+ rows). Set ``L1A_TARGETS_BUILD_TQDM=1``."""
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("L1A_DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    raw = (os.environ.get("L1A_TARGETS_BUILD_TQDM", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _build_l1a_targets(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray | None = None,
    transition_horizon: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Build L1a supervision. ``vol_trend`` is causal: range_norm[t]-range_norm[t-lag] (no future bars).

    ``vol_forecast`` is forward-looking by design (mean of next ``decision_horizon_bars`` range_norm).
    """
    pbar = None
    if _l1a_targets_build_tqdm_enabled():
        pbar = tqdm(
            total=8,
            desc="[L1a] build targets",
            unit="step",
            file=_tqdm_stream(),
            mininterval=0.3,
            leave=True,
        )

    def _tick() -> None:
        if pbar is not None:
            pbar.update(1)

    try:
        cfg = _options_target_config()
        forecast_h = int(cfg["decision_horizon_bars"])
        safe_atr = np.where(pd.to_numeric(df["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, df["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
        high = pd.to_numeric(df["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        low = pd.to_numeric(df["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        symbols = df["symbol"].to_numpy()
        _tick()
        if "vol_regime_label" in df.columns:
            vcol = pd.to_numeric(df["vol_regime_label"], errors="coerce")
        else:
            from core.training.labels.vol_regime_labels import compute_vol_regime_labels

            vcol = compute_vol_regime_labels(df).astype(np.float64)
        state = (
            vcol.groupby(df["symbol"]).transform(lambda s: s.ffill().bfill()).fillna(4.0).to_numpy(dtype=np.float64)
        )
        state = np.clip(state, 0, NUM_REGIME_CLASSES - 1).astype(np.int64)
        _tick()

        if transition_horizon is not None:
            trans_h = max(1, int(transition_horizon))
            th_meta: dict[str, Any] = {"mode": "explicit", "horizon": trans_h, "forecast_horizon": forecast_h}
        else:
            trans_h, th_meta = l1a_resolve_transition_horizon_bars(
                state, symbols, fit_mask, forecast_horizon=forecast_h
            )
        transition_risk = compute_transition_event_labels(state, symbols, horizon=trans_h)
        _tick()
        range_raw = np.clip((high - low) / safe_atr, 0.0, np.inf)
        clip_q = float(np.clip(float(os.environ.get("L1A_RANGE_NORM_CLIP_Q", "0.995")), 0.90, 1.0))
        clip_mode = (os.environ.get("L1A_CLIP_MODE", "mad_z") or "mad_z").strip().lower()
        clip_alpha = float(np.clip(float(os.environ.get("L1A_CLIP_ALPHA", "0.01")), 1e-4, 0.20))
        fit = np.isfinite(range_raw)
        if fit_mask is not None:
            fit &= np.asarray(fit_mask, dtype=bool).ravel()
        fit_vals = range_raw[fit]
        if fit_vals.size == 0:
            fit_vals = range_raw[np.isfinite(range_raw)]
        if _l1a_adaptive_range_clip_enabled():
            range_clip, range_meta = _l1a_adaptive_range_norm_ceiling(fit_vals)
            range_meta = {
                **range_meta,
                "adaptive_range_clip": True,
                "legacy_clip_mode": clip_mode,
                "legacy_clip_alpha": float(clip_alpha),
            }
        else:
            range_clip, range_meta = _l1a_clip_upper(
                fit_vals,
                mode=clip_mode,
                quantile_q=clip_q,
                alpha=clip_alpha,
                floor=1.0,
                ceiling=8.0,
            )
            range_meta = {**range_meta, "adaptive_range_clip": False}
        range_norm = np.clip(range_raw, 0.0, range_clip)
        _tick()
        vol_forecast = _future_mean_by_symbol(range_norm, symbols, horizon=forecast_h)
        _tick()
        lag = int(os.environ.get("L1A_VOL_TREND_LAG", "5"))
        vol_trend = _per_symbol_lag_diff(range_norm, symbols, lag=lag)
        vt_q = float(np.clip(float(os.environ.get("L1A_VOL_TREND_CLIP_Q", "0.995")), 0.90, 1.0))
        vt_fit = np.isfinite(vol_trend)
        if fit_mask is not None:
            vt_fit &= np.asarray(fit_mask, dtype=bool).ravel()
        vt_abs = np.abs(vol_trend[vt_fit])
        if vt_abs.size == 0:
            vt_abs = np.abs(vol_trend[np.isfinite(vol_trend)])
        vt_cap, vt_meta = _l1a_clip_upper(
            vt_abs,
            mode=clip_mode,
            quantile_q=vt_q,
            alpha=clip_alpha,
            floor=0.5,
            ceiling=8.0,
        )
        vol_trend = np.clip(vol_trend, -vt_cap, vt_cap).astype(np.float32)
        _tick()
        cap = int(os.environ.get("L1A_TIME_IN_REGIME_CAP", "120"))
        time_in_regime = _time_in_regime_fraction(state, symbols, cap=cap)
        _tick()

        if "l1a_straddle_edge_target" in df.columns:
            straddle_edge = pd.to_numeric(df["l1a_straddle_edge_target"], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
        elif l1a_straddle_edge_head_enabled():
            from core.training.labels.straddle_edge_labels import compute_straddle_edge_labels

            straddle_edge = compute_straddle_edge_labels(df).to_numpy(dtype=np.float64)
        else:
            straddle_edge = np.zeros(len(df), dtype=np.float64)
        straddle_edge = np.clip(straddle_edge.astype(np.float32), -1.0, 1.0)
        _tick()

        tgt = {
            "regime": state,
            "transition_risk": transition_risk.astype(np.float32),
            "vol_forecast": vol_forecast.astype(np.float32),
            "vol_trend": vol_trend,
            "time_in_regime": time_in_regime,
            "straddle_edge": straddle_edge,
        }
        return (
            tgt,
            {
                "range_norm": range_meta,
                "vol_trend": vt_meta,
                "clip_mode": clip_mode,
                "clip_alpha": float(clip_alpha),
                "transition_horizon": th_meta,
            },
        )
    finally:
        if pbar is not None:
            pbar.close()


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


class StateStructureDecoder(nn.Module):
    """Refine regime logits with persistence/transition structure from shared state context."""

    def __init__(self, shared_dim: int, num_classes: int, *, hidden_dim: int, dropout: float):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(shared_dim + num_classes + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, num_classes)
        self.persistence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        shared_repr: torch.Tensor,
        base_regime_logits: torch.Tensor,
        transition_logit: torch.Tensor,
        vol_trend_value: torch.Tensor,
        time_in_regime_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition_prob = torch.sigmoid(transition_logit).unsqueeze(-1)
        time_ir = time_in_regime_value.unsqueeze(-1)
        vol_trend = torch.tanh(vol_trend_value).unsqueeze(-1)
        x = torch.cat([shared_repr, base_regime_logits, transition_prob, vol_trend, time_ir], dim=1)
        h = self.context(x)
        persistence_logit = self.persistence_head(h).squeeze(-1)
        persistence_gate = torch.sigmoid(persistence_logit).unsqueeze(-1)
        delta = self.delta_head(h)
        refined_logits = base_regime_logits + persistence_gate * delta
        return refined_logits, persistence_logit


class L1AMarketTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int] | None = None,
        *,
        seq_len: int | None = None,
        readout_type: str | None = None,
        min_attention_seq_len: int | None = None,
        tcn_kernel_size: int | None = None,
        tcn_dropout: float | None = None,
        readout_dropout: float | None = None,
        head_dropout: float | None = None,
        embed_dim: int | None = None,
        use_straddle_edge_head: bool | None = None,
        use_vol_trend_head: bool | None = None,
        use_time_in_regime_head: bool | None = None,
    ):
        super().__init__()
        if channels is None:
            channels = _l1a_tcn_channels()
        ks = int(tcn_kernel_size) if tcn_kernel_size is not None else _l1a_tcn_kernel_size()
        td = float(tcn_dropout) if tcn_dropout is not None else _l1a_tcn_dropout()
        rd = float(readout_dropout) if readout_dropout is not None else _l1a_readout_dropout()
        hd_drop = float(head_dropout) if head_dropout is not None else _l1a_head_dropout()
        layers: list[nn.Module] = []
        use_se = os.environ.get("L1A_TCN_USE_SE", "1").strip().lower() in {"1", "true", "yes"}
        for idx, out_ch in enumerate(channels):
            in_ch = input_dim if idx == 0 else channels[idx - 1]
            layers.append(
                L1AGatedTemporalBlock(
                    in_ch, out_ch, kernel_size=ks, dilation=2**idx, dropout=td, use_se=use_se
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.tcn_kernel_size = ks
        self.shared_dim = channels[-1]
        self.seq_len = int(seq_len) if seq_len is not None else _l1a_seq_len()
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
        self.base_regime_head = TaskHead(self.shared_dim, hd, NUM_REGIME_CLASSES, activation="identity", dropout=hd_drop)
        self.transition_head = TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
        self.vol_head = TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
        if use_vol_trend_head is None:
            use_vol_trend_head = l1a_vol_trend_head_enabled()
        if use_time_in_regime_head is None:
            use_time_in_regime_head = l1a_time_in_regime_head_enabled()
        self.vol_trend_head = (
            TaskHead(self.shared_dim, hd_small, 1, activation="identity", dropout=hd_drop)
            if use_vol_trend_head
            else None
        )
        self.time_in_regime_head = (
            TaskHead(self.shared_dim, hd_small, 1, activation="sigmoid", dropout=hd_drop)
            if use_time_in_regime_head
            else None
        )
        self.state_structure_decoder = StateStructureDecoder(
            self.shared_dim,
            NUM_REGIME_CLASSES,
            hidden_dim=hd,
            dropout=hd_drop,
        )
        self.embed_head = EmbedHead(self.shared_dim, ed)
        self.embed_decoder = nn.Sequential(
            nn.Linear(ed, 64),
            nn.GELU(),
            nn.Linear(64, self.shared_dim),
        )
        if use_straddle_edge_head is None:
            use_straddle_edge_head = l1a_straddle_edge_head_enabled()
        self.use_straddle_edge_head = bool(use_straddle_edge_head)
        if self.use_straddle_edge_head:
            self.straddle_edge_head = nn.Sequential(
                nn.Linear(self.shared_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.straddle_edge_head = None

    def shared_repr(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x.transpose(1, 2))
        if self.readout_type == "attention":
            pooled, _ = self.readout(h.transpose(1, 2), min_seq_len=self.min_attention_seq_len)
            return pooled
        return h[:, :, -1]

    def forward(
        self,
        x: torch.Tensor,
        *,
        decoder_vol_trend: torch.Tensor | None = None,
        decoder_time_in_regime: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.shared_repr(x)
        embed = self.embed_head(shared)
        base_regime_logits = self.base_regime_head(shared)
        transition_logit = self.transition_head(shared).squeeze(-1)
        vol_value = self.vol_head(shared).squeeze(-1)
        if self.vol_trend_head is not None:
            vol_trend_value = self.vol_trend_head(shared).squeeze(-1)
        elif decoder_vol_trend is not None:
            vol_trend_value = decoder_vol_trend
        else:
            vol_trend_value = torch.zeros(shared.shape[0], device=shared.device, dtype=shared.dtype)
        if self.time_in_regime_head is not None:
            time_in_regime_value = self.time_in_regime_head(shared).squeeze(-1)
        elif decoder_time_in_regime is not None:
            time_in_regime_value = decoder_time_in_regime
        else:
            time_in_regime_value = torch.zeros(shared.shape[0], device=shared.device, dtype=shared.dtype)
        regime_logits, state_persistence_logit = self.state_structure_decoder(
            shared,
            base_regime_logits,
            transition_logit,
            vol_trend_value,
            time_in_regime_value,
        )
        out: dict[str, torch.Tensor] = {
            "regime_logits": regime_logits,
            "base_regime_logits": base_regime_logits,
            "transition_logit": transition_logit,
            "vol_value": vol_value,
            "vol_trend_value": vol_trend_value,
            "time_in_regime_value": time_in_regime_value,
            "state_persistence_logit": state_persistence_logit,
            "market_embed": embed,
            "embed_recon": self.embed_decoder(embed),
            "shared_repr": shared,
        }
        if self.straddle_edge_head is not None:
            out["straddle_edge_value"] = self.straddle_edge_head(shared).squeeze(-1)
        return out


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
    amp: L1aAmpSettings,
    amp_scaler: GradScaler | None,
    uw_module: L1aMultiTaskUncertaintyWeights,
    regime_aux_coef: float | None = None,
    persist_coef: float | None = None,
) -> float:
    model.train()
    uw_module.train()
    total_loss = 0.0
    total_rows = 0
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    rac = float(_l1a_regime_aux_coef() if regime_aux_coef is None else regime_aux_coef)
    pcoef = float(_l1a_transition_persist_coef() if persist_coef is None else persist_coef)
    max_gn = float(os.environ.get("L1A_MAX_GRAD_NORM", "1.0"))
    nb = _l1a_nb(device)
    se_on = l1a_straddle_edge_head_enabled()
    vt_on = model.vol_trend_head is not None
    tir_on = model.time_in_regime_head is not None
    it = loader
    if _l1a_batch_progress_enabled():
        it = tqdm(
            loader,
            leave=False,
            desc="[L1a] train batches",
            file=_tqdm_stream(),
            mininterval=0.25,
            unit="batch",
            dynamic_ncols=True,
            disable=False,
        )
    for batch in it:
        if se_on:
            xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir, y_edge = batch
            y_edge = y_edge.to(device, non_blocking=nb)
        else:
            xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir = batch
        xb = xb.to(device, non_blocking=nb)
        y_regime = y_regime.to(device, non_blocking=nb)
        y_transition = y_transition.to(device, non_blocking=nb)
        y_vol = y_vol.to(device, non_blocking=nb)
        y_vol_trend = y_vol_trend.to(device, non_blocking=nb)
        y_time_ir = y_time_ir.to(device, non_blocking=nb)
        optimizer.zero_grad(set_to_none=True)
        with _l1a_autocast(amp):
            dec_kw = _l1a_forward_decoder_kw(model, y_vol_trend, y_time_ir)
            out = model(xb, **dec_kw)
            losses = {
                "regime": regime_loss(out["regime_logits"], y_regime),
                "regime_aux": _l1a_coarse_vol_regime_aux_loss(out["base_regime_logits"], y_regime),
                "vol": mse(out["vol_value"], y_vol),
                "embed_recon": mse(out["embed_recon"], out["shared_repr"].detach()),
                "transition": transition_loss(out["transition_logit"], y_transition),
                "state_persistence": bce(out["state_persistence_logit"], 1.0 - y_transition),
            }
            if vt_on:
                losses["vol_trend"] = mse(out["vol_trend_value"], y_vol_trend)
            if tir_on:
                losses["time_in_regime"] = mse(out["time_in_regime_value"], y_time_ir)
            if se_on:
                losses["straddle_edge"] = mse(out["straddle_edge_value"], y_edge)
            uw_in: dict[str, torch.Tensor] = {
                "regime": losses["regime"],
                "regime_aux": rac * losses["regime_aux"],
                "transition": losses["transition"] + pcoef * losses["state_persistence"],
                "vol": losses["vol"],
                "embed_recon": losses["embed_recon"],
            }
            if vt_on:
                uw_in["vol_trend"] = losses["vol_trend"]
            if tir_on:
                uw_in["time_in_regime"] = losses["time_in_regime"]
            if se_on:
                uw_in["straddle_edge"] = losses["straddle_edge"]
            loss, _diag = uw_module.weighted_loss(uw_in)
        if amp_scaler is not None:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            _l1a_clip_grad_norm(model, uw_module, max_gn)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            _l1a_clip_grad_norm(model, uw_module, max_gn)
            optimizer.step()
        total_loss += float(loss.detach().item()) * len(xb)
        total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def _eval_epoch(
    model: L1AMarketTCN,
    loader: DataLoader,
    device: torch.device,
    *,
    regime_loss: nn.Module,
    transition_loss: nn.Module,
    amp: L1aAmpSettings,
    uw_module: L1aMultiTaskUncertaintyWeights,
    val_metric: str = "geom",
    regime_aux_coef: float | None = None,
    persist_coef: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (metric_for_early_stop, diagnostics)."""
    model.eval()
    uw_module.eval()
    total_rows = 0
    sum_legacy = 0.0
    sum_regime = 0.0
    sum_transition = 0.0
    sum_vol = 0.0
    sum_vt = 0.0
    sum_tir = 0.0
    sum_emb = 0.0
    sum_uw = 0.0
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    rac = float(_l1a_regime_aux_coef() if regime_aux_coef is None else regime_aux_coef)
    pcoef = float(_l1a_transition_persist_coef() if persist_coef is None else persist_coef)
    nb = _l1a_nb(device)
    se_on = l1a_straddle_edge_head_enabled()
    vt_on = model.vol_trend_head is not None
    tir_on = model.time_in_regime_head is not None
    it = loader
    if _l1a_batch_progress_enabled():
        it = tqdm(
            loader,
            leave=False,
            desc="[L1a] val batches",
            file=_tqdm_stream(),
            mininterval=0.25,
            unit="batch",
            dynamic_ncols=True,
            disable=False,
        )
    with torch.no_grad():
        for batch in it:
            if se_on:
                xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir, y_edge = batch
                y_edge = y_edge.to(device, non_blocking=nb)
            else:
                xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir = batch
            xb = xb.to(device, non_blocking=nb)
            y_regime = y_regime.to(device, non_blocking=nb)
            y_transition = y_transition.to(device, non_blocking=nb)
            y_vol = y_vol.to(device, non_blocking=nb)
            y_vol_trend = y_vol_trend.to(device, non_blocking=nb)
            y_time_ir = y_time_ir.to(device, non_blocking=nb)
            with _l1a_autocast(amp):
                dec_kw = _l1a_forward_decoder_kw(model, y_vol_trend, y_time_ir)
                out = model(xb, **dec_kw)
                n = len(xb)
                l_reg = regime_loss(out["regime_logits"], y_regime)
                l_aux = _l1a_coarse_vol_regime_aux_loss(out["base_regime_logits"], y_regime)
                l_vol = mse(out["vol_value"], y_vol)
                l_vt = mse(out["vol_trend_value"], y_vol_trend) if vt_on else None
                l_tir = mse(out["time_in_regime_value"], y_time_ir) if tir_on else None
                l_emb = mse(out["embed_recon"], out["shared_repr"].detach())
                l_tr = transition_loss(out["transition_logit"], y_transition)
                l_pers = bce(out["state_persistence_logit"], 1.0 - y_transition)
                l_se = mse(out["straddle_edge_value"], y_edge) if se_on else None
                legacy = l_reg + rac * l_aux + l_vol + l_emb + l_tr + pcoef * l_pers
                if l_vt is not None:
                    legacy = legacy + l_vt
                if l_tir is not None:
                    legacy = legacy + l_tir
                if se_on and l_se is not None:
                    legacy = legacy + l_se
                sum_legacy += float(legacy.item()) * n
                sum_regime += float(l_reg.item()) * n
                sum_transition += float((l_tr + pcoef * l_pers).item()) * n
                sum_vol += float(l_vol.item()) * n
                if l_vt is not None:
                    sum_vt += float(l_vt.item()) * n
                if l_tir is not None:
                    sum_tir += float(l_tir.item()) * n
                sum_emb += float(l_emb.item()) * n
                v_uw: dict[str, torch.Tensor] = {
                    "regime": l_reg,
                    "regime_aux": rac * l_aux,
                    "transition": l_tr + pcoef * l_pers,
                    "vol": l_vol,
                    "embed_recon": l_emb,
                }
                if l_vt is not None:
                    v_uw["vol_trend"] = l_vt
                if l_tir is not None:
                    v_uw["time_in_regime"] = l_tir
                if se_on and l_se is not None:
                    v_uw["straddle_edge"] = l_se
                uwt, _ = uw_module.weighted_loss(v_uw)
                sum_uw += float(uwt.item()) * n
                total_rows += n
    denom = max(total_rows, 1)
    mean_legacy = sum_legacy / denom
    g_reg = max(sum_regime / denom, 1e-8)
    g_tr = max(sum_transition / denom, 1e-8)
    vol_group = sum_vol + sum_vt + sum_tir
    g_vol = max(vol_group / denom, 1e-8)
    g_emb = max(sum_emb / denom, 1e-8)
    geom = float(math.exp(0.25 * (math.log(g_reg) + math.log(g_tr) + math.log(g_vol) + math.log(g_emb))))
    mean_uw = sum_uw / denom
    vm = val_metric
    if vm == "uw_total":
        stop = mean_uw
    elif vm == "geom":
        stop = geom
    else:
        stop = mean_legacy
    diag = {
        "val_legacy_weighted": float(mean_legacy),
        "val_geom_raw_groups": float(geom),
        "val_uw_total": float(mean_uw),
        "val_mean_regime_ce": float(sum_regime / denom),
        "val_mean_transition_group": float(sum_transition / denom),
        "val_mean_volatility_sum": float((sum_vol + sum_vt + sum_tir) / denom),
        "val_mean_embed_mse": float(sum_emb / denom),
    }
    return float(stop), diag


def _l1a_oof_warmstart_expanding_enabled() -> bool:
    """Fold 2+ start from previous fold's ``state_dict`` (expanding OOF only; set ``L1A_OOF_WARMSTART=1``)."""
    return os.environ.get("L1A_OOF_WARMSTART", "").strip().lower() in {"1", "true", "yes"}


def _l1a_warmstart_oof_max_epochs(base_max: int) -> int:
    raw = os.environ.get("L1A_WARMSTART_OOF_MAX_EPOCHS", "").strip()
    if raw:
        return max(1, min(int(raw), int(base_max)))
    return max(1, min(24, max(8, int(base_max) // 2)))


def _l1a_warmstart_oof_lr_scale() -> float:
    return float(os.environ.get("L1A_WARMSTART_OOF_LR_SCALE", "0.5"))


def _l1a_early_stop_best_epoch(
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    n_feat: int,
    embed_dim: int,
    channels: list[int],
    tcn_kernel_size: int,
    tcn_dropout: float,
    readout_dropout: float,
    head_dropout: float,
    regime_loss: nn.Module,
    transition_loss: nn.Module,
    lr: float,
    wd: float,
    T0: int,
    Tm: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    desc: str,
    uw_val_metric: str,
    amp: L1aAmpSettings,
    init_state_dict: dict[str, torch.Tensor] | None = None,
) -> Tuple[int, dict[str, torch.Tensor]]:
    amp_scaler = GradScaler(amp.device_type) if (amp.enabled and amp.dtype == torch.float16) else None
    model = L1AMarketTCN(
        n_feat,
        channels=channels,
        seq_len=_l1a_seq_len(),
        readout_type=_l1a_readout_type(),
        min_attention_seq_len=_l1a_min_attention_seq_len(),
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        readout_dropout=readout_dropout,
        head_dropout=head_dropout,
        embed_dim=embed_dim,
        use_straddle_edge_head=l1a_straddle_edge_head_enabled(),
        use_vol_trend_head=l1a_vol_trend_head_enabled(),
        use_time_in_regime_head=l1a_time_in_regime_head_enabled(),
    ).to(DEVICE)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=True)
    uw = _l1a_build_uw_module(DEVICE)
    optimizer = _l1a_optimizer(model, uw, lr, wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm)
    best_val = float("inf")
    stale = 0
    best_ep = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_bar = trange(
        max_epochs,
        desc=desc,
        unit="ep",
        leave=True,
        file=_tqdm_stream(),
        mininterval=0.3,
        dynamic_ncols=True,
        disable=not _l1a_progress_tqdm_enabled(),
    )
    for epoch in epoch_bar:
        tr_loss = _train_epoch(
            model,
            train_dl,
            optimizer,
            DEVICE,
            regime_loss=regime_loss,
            transition_loss=transition_loss,
            amp=amp,
            amp_scaler=amp_scaler,
            uw_module=uw,
        )
        va_stop, va_diag = _eval_epoch(
            model,
            val_dl,
            DEVICE,
            regime_loss=regime_loss,
            transition_loss=transition_loss,
            amp=amp,
            uw_module=uw,
            val_metric=uw_val_metric,
        )
        scheduler.step()
        print(
            f"  [L1a] epoch={epoch + 1:02d} train_loss={tr_loss:.4f} val_stop={va_stop:.4f} "
            f"val_legacy_sum={va_diag.get('val_legacy_weighted', 0):.4f}",
            flush=True,
        )
        _l1a_log_uw_auto_weights(uw)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(
                train=f"{tr_loss:.4f}",
                val=f"{va_stop:.4f}",
                leg=f"{va_diag.get('val_legacy_weighted', 0):.4f}",
                refresh=False,
            )
        if va_stop < (best_val - min_delta):
            best_val = va_stop
            best_ep = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return max(1, int(best_ep)), best_state


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


def _log_l1a_val_metrics(
    model: L1AMarketTCN,
    val_dl: DataLoader,
    device: torch.device,
    *,
    label: str,
    amp: L1aAmpSettings,
) -> None:
    """Validation report for regime calibration, transition event quality, vol regression, and embed stability."""
    model.eval()
    y_true_r: list[np.ndarray] = []
    y_pred_r: list[np.ndarray] = []
    y_prob_r: list[np.ndarray] = []
    vol_t: list[np.ndarray] = []
    vol_p: list[np.ndarray] = []
    vt_t: list[np.ndarray] = []
    vt_p: list[np.ndarray] = []
    tir_t_fit: list[np.ndarray] = []
    tir_p_fit: list[np.ndarray] = []
    tr_t, tr_s = [], []
    persist_s: list[np.ndarray] = []
    emb_mse: list[np.ndarray] = []
    se_on = l1a_straddle_edge_head_enabled()
    vt_on = model.vol_trend_head is not None
    tir_on = model.time_in_regime_head is not None
    se_t: list[np.ndarray] = []
    se_p: list[np.ndarray] = []
    nb = _l1a_nb(device)
    with torch.no_grad():
        for batch in val_dl:
            if se_on:
                xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir, y_edge = batch
                y_edge = y_edge.to(device, non_blocking=nb)
            else:
                xb, y_regime, y_transition, y_vol, y_vol_trend, y_time_ir = batch
            xb = xb.to(device, non_blocking=nb)
            y_regime = y_regime.to(device, non_blocking=nb)
            y_transition = y_transition.to(device, non_blocking=nb)
            y_vol = y_vol.to(device, non_blocking=nb)
            y_vol_trend = y_vol_trend.to(device, non_blocking=nb)
            y_time_ir = y_time_ir.to(device, non_blocking=nb)
            with _l1a_autocast(amp):
                dec_kw = _l1a_forward_decoder_kw(model, y_vol_trend, y_time_ir)
                out = model(xb, **dec_kw)
            if se_on:
                se_t.append(y_edge.detach().cpu().numpy())
                se_p.append(out["straddle_edge_value"].detach().float().cpu().numpy())
            y_true_r.append(y_regime.detach().cpu().numpy())
            regime_prob = torch.softmax(out["regime_logits"].float(), dim=1)
            y_prob_r.append(regime_prob.detach().cpu().numpy())
            y_pred_r.append(torch.argmax(regime_prob, dim=1).detach().cpu().numpy())
            vol_t.append(y_vol.detach().cpu().numpy())
            vol_p.append(out["vol_value"].detach().float().cpu().numpy())
            if vt_on:
                vt_t.append(y_vol_trend.detach().cpu().numpy())
                vt_p.append(out["vol_trend_value"].detach().float().cpu().numpy())
            if tir_on:
                tir_t_fit.append(y_time_ir.detach().cpu().numpy())
                tir_p_fit.append(out["time_in_regime_value"].detach().float().cpu().numpy())
            tr_t.append(y_transition.detach().cpu().numpy())
            tr_s.append(torch.sigmoid(out["transition_logit"].float()).detach().cpu().numpy())
            persist_s.append(torch.sigmoid(out["state_persistence_logit"].float()).detach().cpu().numpy())
            emb_mse.append(
                F.mse_loss(out["embed_recon"].float(), out["shared_repr"].detach().float(), reduction="none")
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
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
    mae_vt = corr_vt = float("nan")
    if vt_on and vt_t:
        vtt = np.concatenate(vt_t)
        vtp = np.concatenate(vt_p)
        mae_vt = float(mean_absolute_error(vtt, vtp))
        corr_vt = pearson_corr(vtt, vtp)
    mae_tir = corr_tir = float("nan")
    if tir_on and tir_t_fit:
        tirt = _inverse_time_in_regime_target(np.concatenate(tir_t_fit))
        tirpr = _inverse_time_in_regime_target(np.concatenate(tir_p_fit))
        mae_tir = float(mean_absolute_error(tirt, tirpr))
        corr_tir = pearson_corr(tirt, tirpr)
    emb_mean = float(np.mean(np.concatenate(emb_mse)))
    persist_pred = np.concatenate(persist_s) if persist_s else np.array([], dtype=np.float32)
    stay_target = 1.0 - np.concatenate(tr_t)
    persist_mean = float(np.mean(persist_pred)) if persist_pred.size else float("nan")
    persist_std = float(np.std(persist_pred)) if persist_pred.size else float("nan")
    persist_corr = pearson_corr(stay_target, persist_pred)
    persist_q = (
        np.percentile(persist_pred, [10, 50, 90]).astype(np.float32)
        if persist_pred.size
        else np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    )
    persist_by_regime = []
    for idx, name in enumerate(REGIME_NOW_PROB_COLS):
        mask = yt == idx
        mean_val = float(np.mean(persist_pred[mask])) if mask.any() else float("nan")
        persist_by_regime.append(f"{name}={mean_val:.3f}")

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
    if vt_on:
        print(
            f"  [L1a] vol_trend head:  MAE={mae_vt:.4f}  corr(y,p)={corr_vt:.4f}",
            flush=True,
        )
    else:
        print(
            "  [L1a] vol_trend head: disabled (decoder + ``l1a_vol_trend`` use prep-derived targets)",
            flush=True,
        )
    if tir_on:
        print(
            f"  [L1a] time_in_regime head:  MAE={mae_tir:.4f}  corr(y,p)={corr_tir:.4f}",
            flush=True,
        )
    else:
        print(
            "  [L1a] time_in_regime head: disabled (decoder + ``l1a_time_in_regime`` use prep-derived targets)",
            flush=True,
        )
    if se_on and se_t:
        set_ = np.concatenate(se_t)
        sep = np.concatenate(se_p)
        mae_se = float(mean_absolute_error(set_, sep))
        corr_se = pearson_corr(set_, sep)
        print(
            f"  [L1a] straddle_edge head:  MAE={mae_se:.4f}  corr(y,p)={corr_se:.4f}",
            flush=True,
        )
    print(
        f"  [L1a] state persistence head:  mean(persist)={persist_mean:.4f}  std(persist)={persist_std:.4f}  "
        f"mean(stay_target)={float(np.mean(stay_target)):.4f}  corr(stay,persist)={persist_corr:.4f}",
        flush=True,
    )
    print(
        f"  [L1a] persistence quantiles:  p10={float(persist_q[0]):.4f}  p50={float(persist_q[1]):.4f}  p90={float(persist_q[2]):.4f}",
        flush=True,
    )
    print(
        f"  [L1a] persistence mean by true regime: {', '.join(persist_by_regime)}",
        flush=True,
    )
    print(
        f"  [L1a] time_in_regime target transform: {_l1a_time_in_regime_transform()} (metrics/output reported in raw [0,1] space)",
        flush=True,
    )

    _l1a_transition_val_block("transition_risk", np.concatenate(tr_t), np.concatenate(tr_s))
    print(f"  [L1a] ========== end val ({label}) report ==========\n", flush=True)


def _log_l1a_cached_metrics(
    outputs: pd.DataFrame,
    targets: dict[str, np.ndarray],
    end_idx: np.ndarray,
    window_mask: np.ndarray,
    *,
    label: str,
    requested_rows: int | None = None,
) -> None:
    """Metrics from cached/materialized L1a outputs (used for stitched OOF honesty checks)."""
    mask = np.asarray(window_mask, dtype=bool).ravel()
    if mask.size != end_idx.size:
        raise ValueError(f"L1a cached metrics {label}: window_mask size {mask.size} != n_windows {end_idx.size}")
    rows = end_idx[mask]
    n_req = int(requested_rows) if requested_rows is not None else int(mask.sum())
    if rows.size == 0:
        print(
            f"\n  [L1a] ========== val ({label}) effectiveness report ==========\n"
            f"  [L1a] skip: no evaluable rows (requested={n_req:,}, covered=0)\n"
            f"  [L1a] ========== end val ({label}) report ==========\n",
            flush=True,
        )
        return

    pr = outputs.loc[rows, L1A_REGIME_COLS].to_numpy(dtype=np.float64, copy=False)
    pr = np.nan_to_num(pr, nan=1.0 / float(NUM_REGIME_CLASSES), posinf=0.0, neginf=0.0)
    pr = np.clip(pr, 1e-12, np.inf)
    pr = pr / np.clip(pr.sum(axis=1, keepdims=True), 1e-12, np.inf)
    yt = np.asarray(targets["regime"][rows], dtype=np.int64).ravel()
    yp = np.argmax(pr, axis=1).astype(np.int64, copy=False)

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
    coverage_note = ""
    if rows.size != n_req:
        coverage_note = f"  covered_rows={rows.size:,}/{n_req:,} (subset with stitched OOF preds)"

    print(f"\n  [L1a] ========== val ({label}) effectiveness report ==========", flush=True)
    if coverage_note:
        print(f"  [L1a]{coverage_note}", flush=True)
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

    if "l1a_vol_forecast" in outputs.columns:
        vt = np.asarray(targets["vol_forecast"][rows], dtype=np.float64)
        vp = outputs.loc[rows, "l1a_vol_forecast"].to_numpy(dtype=np.float64, copy=False)
        vp = np.nan_to_num(vp, nan=float(np.nanmedian(vt) if vt.size else 0.0), posinf=0.0, neginf=0.0)
        mae_v = float(mean_absolute_error(vt, vp))
        rmse_v = float(np.sqrt(mean_squared_error(vt, vp)))
        r2_v = float(r2_score(vt, vp)) if len(np.unique(vt)) > 1 else float("nan")
        corr_v = pearson_corr(vt, vp)
        print(
            f"  [L1a] vol head:  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}  corr(y,p)={corr_v:.4f}",
            flush=True,
        )
    if "l1a_vol_trend" in outputs.columns:
        vtt = np.asarray(targets["vol_trend"][rows], dtype=np.float64)
        vtp = outputs.loc[rows, "l1a_vol_trend"].to_numpy(dtype=np.float64, copy=False)
        vtp = np.nan_to_num(vtp, nan=0.0, posinf=0.0, neginf=0.0)
        mae_vt = float(mean_absolute_error(vtt, vtp))
        corr_vt = pearson_corr(vtt, vtp)
        print(
            f"  [L1a] vol_trend head:  MAE={mae_vt:.4f}  corr(y,p)={corr_vt:.4f}",
            flush=True,
        )
    if "l1a_time_in_regime" in outputs.columns:
        tirt = np.asarray(targets["time_in_regime"][rows], dtype=np.float64)
        tirpr = outputs.loc[rows, "l1a_time_in_regime"].to_numpy(dtype=np.float64, copy=False)
        tirpr = np.clip(np.nan_to_num(tirpr, nan=float(np.nanmedian(tirt) if tirt.size else 0.0), posinf=1.0, neginf=0.0), 0.0, 1.0)
        mae_tir = float(mean_absolute_error(tirt, tirpr))
        corr_tir = pearson_corr(tirt, tirpr)
        print(
            f"  [L1a] time_in_regime head:  MAE={mae_tir:.4f}  corr(y,p)={corr_tir:.4f}",
            flush=True,
        )
        print(
            f"  [L1a] time_in_regime target transform: {_l1a_time_in_regime_transform()} (metrics/output reported in raw [0,1] space)",
            flush=True,
        )
    if "l1a_state_persistence" in outputs.columns:
        persist_pred = outputs.loc[rows, "l1a_state_persistence"].to_numpy(dtype=np.float64, copy=False)
        persist_pred = np.clip(np.nan_to_num(persist_pred, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        stay_target = 1.0 - np.asarray(targets["transition_risk"][rows], dtype=np.float64)
        persist_mean = float(np.mean(persist_pred)) if persist_pred.size else float("nan")
        persist_std = float(np.std(persist_pred)) if persist_pred.size else float("nan")
        persist_corr = pearson_corr(stay_target, persist_pred)
        persist_q = (
            np.percentile(persist_pred, [10, 50, 90]).astype(np.float32)
            if persist_pred.size
            else np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        )
        persist_by_regime = []
        for idx, name in enumerate(REGIME_NOW_PROB_COLS):
            rmask = yt == idx
            mean_val = float(np.mean(persist_pred[rmask])) if rmask.any() else float("nan")
            persist_by_regime.append(f"{name}={mean_val:.3f}")
        print(
            f"  [L1a] state persistence head:  mean(persist)={persist_mean:.4f}  std(persist)={persist_std:.4f}  "
            f"mean(stay_target)={float(np.mean(stay_target)):.4f}  corr(stay,persist)={persist_corr:.4f}",
            flush=True,
        )
        print(
            f"  [L1a] persistence quantiles:  p10={float(persist_q[0]):.4f}  p50={float(persist_q[1]):.4f}  p90={float(persist_q[2]):.4f}",
            flush=True,
        )
        print(
            f"  [L1a] persistence mean by true regime: {', '.join(persist_by_regime)}",
            flush=True,
        )
    if "l1a_transition_risk" in outputs.columns:
        tr_true = np.asarray(targets["transition_risk"][rows], dtype=np.float64)
        tr_score = outputs.loc[rows, "l1a_transition_risk"].to_numpy(dtype=np.float64, copy=False)
        tr_score = np.clip(np.nan_to_num(tr_score, nan=0.5, posinf=1.0, neginf=0.0), 1e-7, 1.0 - 1e-7)
        _l1a_transition_val_block("transition_risk", tr_true, tr_score)
    print(f"  [L1a] ========== end val ({label}) report ==========\n", flush=True)


def _l1a_cached_regime_probs_targets(
    outputs: pd.DataFrame,
    targets: dict[str, np.ndarray],
    end_idx: np.ndarray,
    window_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(window_mask, dtype=bool).ravel()
    rows = end_idx[mask]
    if rows.size == 0:
        return np.zeros((0, NUM_REGIME_CLASSES), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    probs = outputs.loc[rows, L1A_REGIME_COLS].to_numpy(dtype=np.float64, copy=False)
    probs = np.nan_to_num(probs, nan=1.0 / float(NUM_REGIME_CLASSES), posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 1e-12, np.inf)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, np.inf)
    labels = np.asarray(targets["regime"][rows], dtype=np.int64).ravel()
    return probs, labels


def _l1a_temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = float(np.clip(float(temperature), 1e-3, 1e3))
    log_p = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0))
    scaled = log_p / t
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    ex = np.exp(scaled)
    return ex / np.clip(np.sum(ex, axis=1, keepdims=True), 1e-12, np.inf)


def _l1a_multiclass_prob_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, np.inf)
    try:
        nll = float(log_loss(y, p, labels=list(range(NUM_REGIME_CLASSES))))
    except ValueError:
        nll = float("nan")
    return {
        "nll": nll,
        "brier": float(brier_multiclass(y, p, NUM_REGIME_CLASSES)),
        "ece": float(ece_multiclass_maxprob(y, p)),
    }


def _l1a_fit_temperature(
    probs_oof: np.ndarray,
    labels_oof: np.ndarray,
    *,
    t_lo: float = 0.5,
    t_hi: float = 3.0,
) -> float:
    probs = np.asarray(probs_oof, dtype=np.float64)
    labels = np.asarray(labels_oof, dtype=np.int64).ravel()
    if probs.shape[0] == 0:
        return 1.0

    def _objective(temp: float) -> float:
        met = _l1a_multiclass_prob_metrics(labels, _l1a_temperature_scale_probs(probs, temp))
        return float(met["nll"])

    lo = float(max(0.05, t_lo))
    hi = float(max(lo + 1e-3, t_hi))
    try:
        from scipy.optimize import minimize_scalar  # type: ignore

        res = minimize_scalar(_objective, bounds=(lo, hi), method="bounded")
        if bool(getattr(res, "success", False)) and np.isfinite(float(res.x)):
            return float(np.clip(float(res.x), lo, hi))
    except Exception:
        pass

    grid = np.linspace(lo, hi, 64, dtype=np.float64)
    vals = np.array([_objective(float(t)) for t in grid], dtype=np.float64)
    if np.all(~np.isfinite(vals)):
        return 1.0
    return float(grid[int(np.nanargmin(vals))])


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
    cold_vol_default: float | None = None,
    vol_forecast_clip_hi: float | None = None,
    amp: L1aAmpSettings | None = None,
    transition_horizon_bars: int | None = None,
) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    windows, end_idx = _build_symbol_windows(pd.concat([df[["symbol", "time_key"]], pd.DataFrame(X, columns=feature_cols)], axis=1), feature_cols, seq_len)
    ed = int(embed_dim) if embed_dim is not None else int(getattr(model, "embed_dim", _l1a_embed_dim()))
    edge_out = bool(getattr(model, "use_straddle_edge_head", False))
    mid_cols = [
        "l1a_transition_risk",
        "l1a_vol_forecast",
        "l1a_vol_trend",
        "l1a_time_in_regime",
        "l1a_state_persistence",
    ]
    if edge_out:
        mid_cols.append("l1a_straddle_edge")
    out_cols = list(L1A_REGIME_COLS) + mid_cols + [f"l1a_market_embed_{idx}" for idx in range(ed)] + ["l1a_is_warm"]
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
    atr_series = pd.to_numeric(df["lbl_atr"], errors="coerce")
    if cold_vol_default is not None and np.isfinite(float(cold_vol_default)):
        default_vol = float(cold_vol_default)
    else:
        n0 = min(max(1, int(seq_len)), int(len(atr_series)))
        default_vol = float(np.nanmedian(atr_series.iloc[:n0].to_numpy(dtype=np.float64)))
        if not np.isfinite(default_vol):
            default_vol = 1.0
    outputs["l1a_vol_forecast"] = default_vol
    outputs["l1a_is_warm"] = 0.0
    if len(end_idx) == 0:
        return outputs

    cfg_h = int(_options_target_config()["decision_horizon_bars"])
    th_arg = int(transition_horizon_bars) if transition_horizon_bars is not None else cfg_h
    prior_tgt, _ = _build_l1a_targets(df, fit_mask=None, transition_horizon=th_arg)
    static_vt = np.asarray(prior_tgt["vol_trend"], dtype=np.float32)
    static_tir_raw = np.asarray(prior_tgt["time_in_regime"], dtype=np.float64)
    static_tir_fit = _transform_time_in_regime_target(static_tir_raw).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(windows))
    infer_workers = _l1a_dataloader_workers(min(4, max(_lgbm_n_jobs(), 1)))
    infer_bs = max(32, int(os.environ.get("L1A_MATERIALIZE_BATCH_SIZE", "2048")))
    dl_kwargs: dict[str, Any] = {
        "batch_size": infer_bs,
        "shuffle": False,
        "num_workers": infer_workers,
        "pin_memory": device.type == "cuda",
    }
    if infer_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = max(2, int(os.environ.get("L1A_PREFETCH_FACTOR", "4")))
    dl = DataLoader(ds, **dl_kwargs)
    dl_it = dl
    if _l1a_progress_tqdm_enabled():
        dl_it = tqdm(
            dl,
            desc="[L1a] materialize outputs",
            file=_tqdm_stream(),
            mininterval=0.3,
            unit="batch",
            dynamic_ncols=True,
            disable=False,
        )
    regime_rows: list[np.ndarray] = []
    scalar_keys = ["transition", "vol", "vol_trend", "time_ir", "persistence"]
    if edge_out:
        scalar_keys.append("straddle_edge")
    scalar_rows: dict[str, list[np.ndarray]] = {k: [] for k in scalar_keys}
    embeds: list[np.ndarray] = []
    amp_m = amp if amp is not None else _l1a_build_amp(device)[0]
    nb = _l1a_nb(device)
    model.eval()
    w_off = 0
    vt_head_on = model.vol_trend_head is not None
    tir_head_on = model.time_in_regime_head is not None
    with torch.no_grad():
        for (xb,) in dl_it:
            xb = xb.to(device, non_blocking=nb)
            rows = end_idx[w_off : w_off + len(xb)]
            w_off += len(xb)
            dec_kw: dict[str, torch.Tensor] = {}
            if model.vol_trend_head is None:
                dec_kw["decoder_vol_trend"] = torch.from_numpy(static_vt[rows]).to(
                    device=device, dtype=torch.float32, non_blocking=nb
                )
            if model.time_in_regime_head is None:
                dec_kw["decoder_time_in_regime"] = torch.from_numpy(static_tir_fit[rows]).to(
                    device=device, dtype=torch.float32, non_blocking=nb
                )
            with _l1a_autocast(amp_m):
                out = model(xb, **dec_kw)
            regime_rows.append(torch.softmax(out["regime_logits"].float(), dim=1).detach().cpu().numpy())
            scalar_rows["transition"].append(torch.sigmoid(out["transition_logit"].float()).detach().cpu().numpy())
            scalar_rows["vol"].append(out["vol_value"].detach().float().cpu().numpy())
            clip_lo = float(os.environ.get("L1A_VOL_TREND_CLIP_LO", "-3"))
            clip_hi = float(os.environ.get("L1A_VOL_TREND_CLIP_HI", "3"))
            if vt_head_on:
                scalar_rows["vol_trend"].append(
                    np.clip(out["vol_trend_value"].detach().float().cpu().numpy(), clip_lo, clip_hi)
                )
            else:
                scalar_rows["vol_trend"].append(static_vt[rows])
            if tir_head_on:
                scalar_rows["time_ir"].append(out["time_in_regime_value"].detach().float().cpu().numpy())
            else:
                scalar_rows["time_ir"].append(static_tir_fit[rows])
            scalar_rows["persistence"].append(torch.sigmoid(out["state_persistence_logit"].float()).detach().cpu().numpy())
            if edge_out:
                scalar_rows["straddle_edge"].append(out["straddle_edge_value"].detach().float().cpu().numpy())
            embeds.append(out["market_embed"].detach().float().cpu().numpy())
    regime = np.concatenate(regime_rows, axis=0)
    outputs.loc[end_idx, L1A_REGIME_COLS] = regime
    outputs.loc[end_idx, "l1a_transition_risk"] = np.concatenate(scalar_rows["transition"], axis=0)
    v_hi = float(vol_forecast_clip_hi) if vol_forecast_clip_hi is not None and np.isfinite(float(vol_forecast_clip_hi)) else float(
        os.environ.get("L1A_VOL_FORECAST_MATERIALIZE_MAX", "5.0")
    )
    outputs.loc[end_idx, "l1a_vol_forecast"] = np.clip(np.concatenate(scalar_rows["vol"], axis=0), 0.0, v_hi)
    outputs.loc[end_idx, "l1a_vol_trend"] = np.concatenate(scalar_rows["vol_trend"], axis=0)
    if tir_head_on:
        tir_col = np.clip(
            _inverse_time_in_regime_target(np.concatenate(scalar_rows["time_ir"], axis=0)), 0.0, 1.0
        )
    else:
        tir_col = np.clip(static_tir_raw[end_idx].astype(np.float32), 0.0, 1.0)
    outputs.loc[end_idx, "l1a_time_in_regime"] = tir_col
    outputs.loc[end_idx, "l1a_state_persistence"] = np.clip(
        np.concatenate(scalar_rows["persistence"], axis=0), 0.0, 1.0
    )
    if edge_out:
        outputs.loc[end_idx, "l1a_straddle_edge"] = np.clip(
            np.concatenate(scalar_rows["straddle_edge"], axis=0), -1.0, 1.0
        )
    embed_mat = np.concatenate(embeds, axis=0)
    if embed_mat.shape[1] != ed:
        raise RuntimeError(
            f"L1a embed width mismatch: model expects {ed} market_embed cols but forward returned {embed_mat.shape[1]}."
        )
    embed_cols = [f"l1a_market_embed_{idx}" for idx in range(ed)]
    outputs.loc[end_idx, embed_cols] = embed_mat
    outputs.loc[end_idx, "l1a_is_warm"] = 1.0
    return outputs


def _l1a_overlay_expanding_oof_predictions(
    outputs: pd.DataFrame,
    fold_states: list[dict[str, torch.Tensor]],
    fold_va_masks: list[np.ndarray],
    *,
    windows: np.ndarray,
    end_idx: np.ndarray,
    prior_vol_trend: np.ndarray,
    prior_time_in_regime: np.ndarray,
    n_feat: int,
    embed_dim: int,
    channels: list[int],
    tcn_kernel_size: int,
    tcn_dropout: float,
    readout_dropout: float,
    head_dropout: float,
    amp: L1aAmpSettings,
) -> None:
    """Replace L1a outputs on each fold's val windows with that fold's OOF model forward (CPU state)."""
    nb = _l1a_nb(DEVICE)
    ed = int(embed_dim)
    edge_oof = l1a_straddle_edge_head_enabled()
    vt_oof = l1a_vol_trend_head_enabled()
    tir_oof = l1a_time_in_regime_head_enabled()
    static_vt = np.asarray(prior_vol_trend, dtype=np.float32)
    static_tir_raw = np.asarray(prior_time_in_regime, dtype=np.float64)
    static_tir_fit = _transform_time_in_regime_target(static_tir_raw).astype(np.float32)
    embed_cols = [f"l1a_market_embed_{idx}" for idx in range(ed)]
    infer_bs = max(32, int(os.environ.get("L1A_MATERIALIZE_BATCH_SIZE", "2048")))
    for state, w_va in zip(fold_states, fold_va_masks):
        if state is None or not np.any(w_va):
            continue
        model = L1AMarketTCN(
            n_feat,
            channels=channels,
            seq_len=_l1a_seq_len(),
            readout_type=_l1a_readout_type(),
            min_attention_seq_len=_l1a_min_attention_seq_len(),
            tcn_kernel_size=tcn_kernel_size,
            tcn_dropout=tcn_dropout,
            readout_dropout=readout_dropout,
            head_dropout=head_dropout,
            embed_dim=ed,
            use_straddle_edge_head=edge_oof,
            use_vol_trend_head=vt_oof,
            use_time_in_regime_head=tir_oof,
        ).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        w_sel = windows[w_va].astype(np.float32, copy=False)
        sub_end = end_idx[w_va]
        ds = TensorDataset(torch.from_numpy(w_sel))
        dl = DataLoader(ds, batch_size=infer_bs, shuffle=False, pin_memory=DEVICE.type == "cuda")
        regime_rows: list[np.ndarray] = []
        sk = ["transition", "vol", "vol_trend", "time_ir", "persistence"]
        if edge_oof:
            sk.append("straddle_edge")
        scalar_rows: dict[str, list[np.ndarray]] = {k: [] for k in sk}
        embeds: list[np.ndarray] = []
        vt_h = model.vol_trend_head is not None
        tir_h = model.time_in_regime_head is not None
        off = 0
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(DEVICE, non_blocking=nb)
                rows = sub_end[off : off + len(xb)]
                off += len(xb)
                dec_kw: dict[str, torch.Tensor] = {}
                if model.vol_trend_head is None:
                    dec_kw["decoder_vol_trend"] = torch.from_numpy(static_vt[rows]).to(
                        device=DEVICE, dtype=torch.float32, non_blocking=nb
                    )
                if model.time_in_regime_head is None:
                    dec_kw["decoder_time_in_regime"] = torch.from_numpy(static_tir_fit[rows]).to(
                        device=DEVICE, dtype=torch.float32, non_blocking=nb
                    )
                with _l1a_autocast(amp):
                    out = model(xb, **dec_kw)
                regime_rows.append(torch.softmax(out["regime_logits"].float(), dim=1).detach().cpu().numpy())
                scalar_rows["transition"].append(torch.sigmoid(out["transition_logit"].float()).detach().cpu().numpy())
                scalar_rows["vol"].append(out["vol_value"].detach().float().cpu().numpy())
                clip_lo = float(os.environ.get("L1A_VOL_TREND_CLIP_LO", "-3"))
                clip_hi = float(os.environ.get("L1A_VOL_TREND_CLIP_HI", "3"))
                if vt_h:
                    scalar_rows["vol_trend"].append(
                        np.clip(out["vol_trend_value"].detach().float().cpu().numpy(), clip_lo, clip_hi)
                    )
                else:
                    scalar_rows["vol_trend"].append(static_vt[rows])
                if tir_h:
                    scalar_rows["time_ir"].append(out["time_in_regime_value"].detach().float().cpu().numpy())
                else:
                    scalar_rows["time_ir"].append(static_tir_fit[rows])
                scalar_rows["persistence"].append(torch.sigmoid(out["state_persistence_logit"].float()).detach().cpu().numpy())
                if edge_oof:
                    scalar_rows["straddle_edge"].append(out["straddle_edge_value"].detach().float().cpu().numpy())
                embeds.append(out["market_embed"].detach().float().cpu().numpy())
        rows = end_idx[w_va]
        regime = np.concatenate(regime_rows, axis=0)
        outputs.loc[rows, L1A_REGIME_COLS] = regime
        outputs.loc[rows, "l1a_transition_risk"] = np.concatenate(scalar_rows["transition"], axis=0)
        v_hi = float(os.environ.get("L1A_VOL_FORECAST_MATERIALIZE_MAX", "5.0"))
        outputs.loc[rows, "l1a_vol_forecast"] = np.clip(np.concatenate(scalar_rows["vol"], axis=0), 0.0, v_hi)
        outputs.loc[rows, "l1a_vol_trend"] = np.concatenate(scalar_rows["vol_trend"], axis=0)
        if tir_h:
            tir_part = np.clip(
                _inverse_time_in_regime_target(np.concatenate(scalar_rows["time_ir"], axis=0)), 0.0, 1.0
            )
        else:
            tir_part = np.clip(static_tir_raw[rows].astype(np.float32), 0.0, 1.0)
        outputs.loc[rows, "l1a_time_in_regime"] = tir_part
        outputs.loc[rows, "l1a_state_persistence"] = np.clip(
            np.concatenate(scalar_rows["persistence"], axis=0), 0.0, 1.0
        )
        if edge_oof and "l1a_straddle_edge" in outputs.columns:
            outputs.loc[rows, "l1a_straddle_edge"] = np.clip(
                np.concatenate(scalar_rows["straddle_edge"], axis=0), -1.0, 1.0
            )
        embed_mat = np.concatenate(embeds, axis=0)
        outputs.loc[rows, embed_cols] = embed_mat
        outputs.loc[rows, "l1a_is_warm"] = 1.0



def _l1a_materialize_only_env() -> bool:
    return os.environ.get("L1A_MATERIALIZE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _l1a_resume_cpu_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _l1a_save_train_resume_checkpoint(
    *,
    model: L1AMarketTCN,
    uw_for_meta: L1aMultiTaskUncertaintyWeights | None,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: list[str],
    clip_meta: dict[str, Any],
    vol_forecast_clip_hi: float,
    vol_mat_clip_meta: dict[str, Any],
    lbl_atr_median_train: float,
    ch: list[int],
    tcn_ks: int,
    td: float,
    rd: float,
    hd_drop: float,
    embed_dim: int,
    n_l1_oof: int,
    n_l1_oof_req: int,
    expanding_oof_fold_states: list[dict[str, torch.Tensor]] | None,
    expanding_oof_va_masks: list[np.ndarray] | None,
    lr: float,
    wd: float,
    T0: int,
    t0_source: str,
    Tm: int,
    cos_auto: bool,
    batch_size: int,
    seq_steps_per_batch: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    rf_steps: int,
    oof_cap_message: str,
    l1_final_epochs_after_oof: int | None,
) -> None:
    fold_cpu = None
    if expanding_oof_fold_states is not None:
        fold_cpu = [_l1a_resume_cpu_state_dict(sd) for sd in expanding_oof_fold_states]
    masks = None
    if expanding_oof_va_masks is not None:
        masks = [np.asarray(m, dtype=bool).copy() for m in expanding_oof_va_masks]
    payload: dict[str, Any] = {
        "schema_version": L1A_SCHEMA_VERSION,
        "model_state": _l1a_resume_cpu_state_dict(_l1a_state_dict_for_save(model)),
        "uw_state": _l1a_resume_cpu_state_dict(uw_for_meta.state_dict()) if uw_for_meta is not None else None,
        "mean": np.asarray(mean),
        "std": np.asarray(std),
        "feature_cols": list(feature_cols),
        "clip_meta": clip_meta,
        "vol_forecast_clip_hi": float(vol_forecast_clip_hi),
        "vol_mat_clip_meta": dict(vol_mat_clip_meta),
        "lbl_atr_median_train": float(lbl_atr_median_train),
        "ch": list(ch),
        "tcn_ks": int(tcn_ks),
        "td": float(td),
        "rd": float(rd),
        "hd_drop": float(hd_drop),
        "embed_dim": int(embed_dim),
        "n_l1_oof": int(n_l1_oof),
        "n_l1_oof_req": int(n_l1_oof_req),
        "l1_oof_mode": l1_oof_mode_from_env(),
        "expanding_oof_fold_states": fold_cpu,
        "expanding_oof_va_masks": masks,
        "train_hyper": {
            "lr": lr,
            "wd": wd,
            "T0": T0,
            "t0_source": t0_source,
            "Tm": Tm,
            "cos_auto": cos_auto,
            "batch_size": batch_size,
            "seq_steps_per_batch": seq_steps_per_batch,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "rf_steps": rf_steps,
            "oof_cap_message": oof_cap_message,
            "l1_final_epochs_after_oof": l1_final_epochs_after_oof,
        },
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    rpath = os.path.join(MODEL_DIR, L1A_TRAIN_RESUME_FILE)
    with open(rpath, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"  [L1a] resume checkpoint saved -> {rpath}  "
        "(L1A_MATERIALIZE_ONLY=1 + same L1_OOF_* / prepared df to finish without retraining)",
        flush=True,
    )


def _l1a_run_materialize_through_save(
    *,
    model: L1AMarketTCN,
    uw_for_meta: L1aMultiTaskUncertaintyWeights | None,
    work: pd.DataFrame,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    targets: dict[str, Any],
    clip_meta: dict[str, Any],
    vol_forecast_clip_hi: float,
    vol_mat_clip_meta: dict[str, Any],
    lbl_atr_median_train: float,
    l1a_amp: L1aAmpSettings,
    expanding_oof_fold_states: list[dict[str, torch.Tensor]] | None,
    expanding_oof_va_masks: list[np.ndarray] | None,
    n_l1_oof: int,
    windows: np.ndarray,
    end_idx: np.ndarray,
    window_cal: np.ndarray,
    window_val_report: np.ndarray,
    ch: list[int],
    tcn_ks: int,
    td: float,
    rd: float,
    hd_drop: float,
    uw_val_metric: str,
    lr: float,
    wd: float,
    T0: int,
    t0_source: str,
    Tm: int,
    cos_auto: bool,
    batch_size: int,
    seq_steps_per_batch: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    rf_steps: int,
    embed_dim: int,
    n_l1_oof_req: int,
    oof_cap_message: str,
    l1_final_epochs_after_oof: int | None,
    state_label_missing_rate: float,
    norm_mask: np.ndarray,
    train_started_perf: float,
) -> L1ATrainingBundle:
    _trans_h_saved = int(
        (clip_meta.get("transition_horizon") or {}).get("horizon", int(_options_target_config()["decision_horizon_bars"]))
    )
    outputs = materialize_l1a_outputs(
        model,
        work,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=_l1a_seq_len(),
        device=DEVICE,
        embed_dim=model.embed_dim,
        cold_vol_default=lbl_atr_median_train,
        vol_forecast_clip_hi=vol_forecast_clip_hi,
        amp=l1a_amp,
        transition_horizon_bars=_trans_h_saved,
    )
    metrics_source: dict[str, str] = {
        "fit_report": "final_forward",
        "cal_full": "final_forward",
    }
    temperature_meta: dict[str, Any] = {
        "enabled": False,
        "source": "none",
        "temperature_T": 1.0,
        "metrics": {},
    }
    if (
        expanding_oof_fold_states is not None
        and expanding_oof_va_masks is not None
        and n_l1_oof >= 2
        and l1_oof_mode_from_env() == "expanding"
    ):
        _l1a_overlay_expanding_oof_predictions(
            outputs,
            expanding_oof_fold_states,
            expanding_oof_va_masks,
            windows=windows,
            end_idx=end_idx,
            prior_vol_trend=targets["vol_trend"],
            prior_time_in_regime=targets["time_in_regime"],
            n_feat=len(feature_cols),
            embed_dim=int(model.embed_dim),
            channels=ch,
            tcn_kernel_size=tcn_ks,
            tcn_dropout=td,
            readout_dropout=rd,
            head_dropout=hd_drop,
            amp=l1a_amp,
        )
        print(
            "  [L1a] stitched expanding OOF preds onto calibration window ends (honest L1a for L2 fit)",
            flush=True,
        )
        oof_covered = np.zeros(len(end_idx), dtype=bool)
        for w_va in expanding_oof_va_masks:
            oof_covered |= np.asarray(w_va, dtype=bool)
        _log_l1a_cached_metrics(
            outputs,
            targets,
            end_idx,
            window_val_report & oof_covered,
            label="oof_fit_report",
            requested_rows=int(window_val_report.sum()),
        )
        metrics_source["oof_fit_report"] = "oof_stitched_expanding"
        probs_fit, labels_fit = _l1a_cached_regime_probs_targets(
            outputs, targets, end_idx, window_val_report & oof_covered
        )
        probs_cal, labels_cal = _l1a_cached_regime_probs_targets(
            outputs, targets, end_idx, window_cal & oof_covered
        )
        if probs_fit.shape[0] >= 200 and probs_cal.shape[0] >= 200:
            t_opt = _l1a_fit_temperature(
                probs_fit,
                labels_fit,
                t_lo=float(np.clip(float(os.environ.get("L1A_TEMP_T_MIN", "0.5")), 0.05, 10.0)),
                t_hi=float(np.clip(float(os.environ.get("L1A_TEMP_T_MAX", "3.0")), 0.06, 10.0)),
            )
            fit_before = _l1a_multiclass_prob_metrics(labels_fit, probs_fit)
            fit_after = _l1a_multiclass_prob_metrics(labels_fit, _l1a_temperature_scale_probs(probs_fit, t_opt))
            cal_before = _l1a_multiclass_prob_metrics(labels_cal, probs_cal)
            cal_after = _l1a_multiclass_prob_metrics(labels_cal, _l1a_temperature_scale_probs(probs_cal, t_opt))
            print(
                f"  [L1a] temperature scaling (fit on oof_fit_report): T={t_opt:.4f}",
                flush=True,
            )
            print(
                f"    oof_fit_report: nll {fit_before['nll']:.4f} -> {fit_after['nll']:.4f}  "
                f"Brier {fit_before['brier']:.4f} -> {fit_after['brier']:.4f}  "
                f"ECE {fit_before['ece']:.4f} -> {fit_after['ece']:.4f}",
                flush=True,
            )
            print(
                f"    oof_cal_full:   nll {cal_before['nll']:.4f} -> {cal_after['nll']:.4f}  "
                f"Brier {cal_before['brier']:.4f} -> {cal_after['brier']:.4f}  "
                f"ECE {cal_before['ece']:.4f} -> {cal_after['ece']:.4f}",
                flush=True,
            )
            temperature_meta = {
                "enabled": True,
                "source": "oof_fit_report_expanding",
                "temperature_T": float(t_opt),
                "metrics": {
                    "oof_fit_nll_before": float(fit_before["nll"]),
                    "oof_fit_nll_after": float(fit_after["nll"]),
                    "oof_fit_brier_before": float(fit_before["brier"]),
                    "oof_fit_brier_after": float(fit_after["brier"]),
                    "oof_fit_ece_before": float(fit_before["ece"]),
                    "oof_fit_ece_after": float(fit_after["ece"]),
                    "oof_cal_nll_before": float(cal_before["nll"]),
                    "oof_cal_nll_after": float(cal_after["nll"]),
                    "oof_cal_brier_before": float(cal_before["brier"]),
                    "oof_cal_brier_after": float(cal_after["brier"]),
                    "oof_cal_ece_before": float(cal_before["ece"]),
                    "oof_cal_ece_after": float(cal_after["ece"]),
                },
            }
        if window_cal.sum() != window_val_report.sum() and os.environ.get("L1A_SKIP_CAL_FULL_METRICS", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            _log_l1a_cached_metrics(
                outputs,
                targets,
                end_idx,
                window_cal & oof_covered,
                label="oof_cal_full",
                requested_rows=int(window_cal.sum()),
            )
            metrics_source["oof_cal_full"] = "oof_stitched_expanding"
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(_l1a_state_dict_for_save(model), os.path.join(MODEL_DIR, L1A_MODEL_FILE))
    if uw_for_meta is not None:
        _to = uw_for_meta.task_order
        uw_meta = {
            "enabled": True,
            "val_metric": str(uw_val_metric),
            "regime_aux_coef": float(_l1a_regime_aux_coef()),
            "persistence_coef": float(_l1a_transition_persist_coef()),
            "uw_lr_ratio": float(_l1a_uw_lr_ratio()),
            "task_order": list(_to),
            "task_types": {n: L1aMultiTaskUncertaintyWeights.TASK_TYPES[n] for n in _to},
            "final_log_vars": {n: float(uw_for_meta.log_vars[n].detach().item()) for n in _to},
            "final_precision": {n: float(torch.exp(-uw_for_meta.log_vars[n]).detach().item()) for n in _to},
        }
    else:
        uw_meta = {"enabled": False}
    meta = {
        "schema_version": L1A_SCHEMA_VERSION,
        "l1a_ctx_stagger": l1_ctx_stagger_enabled(),
        "feature_cols": feature_cols,
        "seq_len": _l1a_seq_len(),
        "readout_type": model.readout_type,
        "min_attention_seq_len": model.min_attention_seq_len,
        "mean": mean,
        "std": std,
        "output_cols": l1a_output_columns_with_embed_dim(embed_dim),
        "device": str(DEVICE),
        "model_file": L1A_MODEL_FILE,
        "output_cache_file": L1A_OUTPUT_CACHE_FILE,
        "transition_target_semantics": (
            "probability of any vol-regime class change within l1a_transition_horizon_bars lookahead "
            "(calibrated on train/cal fit_mask to ~30–50% positives by default; vol_forecast target still uses OPTION_DECISION_HORIZON_BARS)"
        ),
        "l1a_transition_horizon_bars": _trans_h_saved,
        "l1a_transition_horizon_meta": dict(clip_meta.get("transition_horizon") or {}),
        "feature_contract_semantics": "L1a exports regime probabilities, volatility, time-in-regime, persistence, embeddings, and warm flag only; directional derived columns are intentionally excluded.",
        "state_structure_semantics": (
            "state_structure_decoder refines regime logits from shared state, transition risk, vol trend, and "
            "time-in-regime context (neural head outputs when trained, else prep-derived targets at train/infer); "
            "l1a_state_persistence estimates stay probability"
        ),
        "l1a_vol_trend_target": f"per-symbol range_norm[t]-range_norm[t-{os.environ.get('L1A_VOL_TREND_LAG', '5')}], clipped",
        "l1a_time_in_regime_target": (
            f"min(run_length,cap)/cap from vol_regime_label runs; cap={os.environ.get('L1A_TIME_IN_REGIME_CAP', '120')}  "
            f"transform={_l1a_time_in_regime_transform()}"
        ),
        "l1a_regime_definition": "vol5",
        "l1a_straddle_edge_head": bool(l1a_straddle_edge_head_enabled()),
        "l1a_vol_trend_head": bool(l1a_vol_trend_head_enabled()),
        "l1a_time_in_regime_head": bool(l1a_time_in_regime_head_enabled()),
        "l1a_regime_aux_loss": (
            "coarse_vol_5to2: CE on logsumexp groups of base_regime_logits — low-vol=classes 0-1, high-vol=2-4; "
            "Kendall UW task regime_aux uses (L1A_REGIME_AUX_COEF × aux CE) as its scalar loss"
        ),
        "embed_dim": int(embed_dim),
        "l1a_state_arch": "tcn_plus_state_structure_decoder",
        "l1a_multi_task_loss": "kendall_uncertainty_per_task",
        "tcn_channels": list(ch),
        "l1a_tcn_kernel_size": int(tcn_ks),
        "l1a_tcn_receptive_field_steps": int(rf_steps),
        "tcn_dropout": float(td),
        "readout_dropout": float(rd),
        "head_dropout": float(hd_drop),
        "l1a_lr": float(lr),
        "l1a_weight_decay": float(wd),
        "l1a_cosine_T0": int(T0),
        "l1a_cosine_T0_source": str(t0_source),
        "l1a_cosine_T_mult": int(Tm),
        "l1a_cos_auto": bool(cos_auto),
        "l1a_batch_size": int(batch_size),
        "l1a_amp_enabled": bool(l1a_amp.enabled),
        "l1a_amp_dtype": str(l1a_amp.dtype) if l1a_amp.enabled else "float32",
        "l1a_torch_compile": os.environ.get("L1A_TORCH_COMPILE", "0").strip().lower() in {"1", "true", "yes"},
        "metrics_source": metrics_source,
        "temperature_scaling": temperature_meta,
        "l1a_seq_steps_per_batch": int(seq_steps_per_batch),
        "l1a_max_epochs": int(max_epochs),
        "l1a_patience": int(patience),
        "l1a_early_stop_min_delta": float(min_delta),
        "l1a_regime_label_smoothing": float(os.environ.get("L1A_REGIME_LABEL_SMOOTHING", "0.1")),
        "l1a_regime_class_weights": "none",
        "l1a_clip_mode": str(clip_meta.get("clip_mode", "mad_z")),
        "l1a_clip_alpha": float(clip_meta.get("clip_alpha", 0.01)),
        "l1a_clip_stats": {
            "range_norm": dict(clip_meta.get("range_norm") or {}),
            "vol_trend": dict(clip_meta.get("vol_trend") or {}),
        },
        "lbl_atr_median_train": float(lbl_atr_median_train),
        "l1_oof_mode": l1_oof_mode_from_env(),
        "l1_oof_folds": int(n_l1_oof),
        "l1_oof_folds_requested": int(n_l1_oof_req),
        "l1_oof_auto_cap": bool(_l1_oof_auto_cap_enabled()),
        "l1_oof_min_windows_per_fold": max(200, int(os.environ.get("L1_OOF_MIN_WINDOWS_PER_FOLD", "4000"))),
        "l1_oof_cap_note": str(oof_cap_message),
        "l1_oof_enabled": bool(n_l1_oof >= 2),
        "l1_final_epochs_after_oof": l1_final_epochs_after_oof,
        "uncertainty_weighting": uw_meta,
        "state_label_missing_rate": float(state_label_missing_rate),
        "l1a_vol_forecast_clip_hi": float(vol_forecast_clip_hi),
        "l1a_vol_forecast_clip_meta": dict(vol_mat_clip_meta),
        "l1a_max_extra_features": max(0, int(os.environ.get("L1A_MAX_EXTRA_FEATURES", "20"))),
    }
    adaptive_min_samples = int(os.environ.get("ADAPTIVE_THRESHOLD_MIN_SAMPLES", "500"))
    fit_n = int(np.sum(np.asarray(norm_mask, dtype=bool)))
    meta = attach_threshold_registry(
        meta,
        "l1a",
        [
            threshold_entry(
                "l1a_range_norm_clip_upper",
                float((clip_meta.get("range_norm") or {}).get("clip", 8.0)),
                category="adaptive_candidate",
                role="range_norm upper clip derived from robust significance",
                adaptive_hint="median + z(alpha/2) * MAD/0.6745 (quantile fallback)",
                n_samples_used=int((clip_meta.get("range_norm") or {}).get("fit_n", fit_n)),
                min_reliable_samples=adaptive_min_samples,
                statistical_principle=str((clip_meta.get("range_norm") or {}).get("statistical_principle", "unknown")),
                alpha=float((clip_meta.get("range_norm") or {}).get("alpha", clip_meta.get("clip_alpha", 0.01))),
                method_selected=str((clip_meta.get("range_norm") or {}).get("mode", clip_meta.get("clip_mode", "unknown"))),
                fallback_reason=str((clip_meta.get("range_norm") or {}).get("fallback_reason", "")),
            ),
            threshold_entry(
                "l1a_vol_trend_clip_cap",
                float((clip_meta.get("vol_trend") or {}).get("clip", 8.0)),
                category="adaptive_candidate",
                role="vol-trend symmetric clip cap from robust significance",
                adaptive_hint="median + z(alpha/2) * MAD/0.6745 on |vol_trend|",
                n_samples_used=int((clip_meta.get("vol_trend") or {}).get("fit_n", fit_n)),
                min_reliable_samples=adaptive_min_samples,
                statistical_principle=str((clip_meta.get("vol_trend") or {}).get("statistical_principle", "unknown")),
                alpha=float((clip_meta.get("vol_trend") or {}).get("alpha", clip_meta.get("clip_alpha", 0.01))),
                method_selected=str((clip_meta.get("vol_trend") or {}).get("mode", clip_meta.get("clip_mode", "unknown"))),
                fallback_reason=str((clip_meta.get("vol_trend") or {}).get("fallback_reason", "")),
            ),
            threshold_entry(
                "L1A_CLIP_MODE",
                str(clip_meta.get("clip_mode", "mad_z")),
                category="adaptive_candidate",
                role="L1a clipping estimator mode",
                adaptive_hint="mad_z default with quantile fallback support",
                statistical_principle="estimator_selection",
                method_selected=str(clip_meta.get("clip_mode", "mad_z")),
            ),
            threshold_entry(
                "L1A_CLIP_ALPHA",
                float(clip_meta.get("clip_alpha", 0.01)),
                category="adaptive_candidate",
                role="type-I error control for robust clip",
                statistical_principle="two_sided_significance_level",
                alpha=float(clip_meta.get("clip_alpha", 0.01)),
            ),
            threshold_entry(
                "L1A_TIME_IN_REGIME_CAP",
                int(os.environ.get("L1A_TIME_IN_REGIME_CAP", "120")),
                category="data_guardrail",
                role="run-length normalization cap",
            ),
            threshold_entry(
                "L1A_PATIENCE",
                int(patience),
                category="data_guardrail",
                role="early-stop patience",
            ),
        ],
    )
    for w in meta.get("threshold_registry", {}).get("warnings", []):
        print(f"  [L1a][warn] {w}", flush=True)
    with open(os.path.join(MODEL_DIR, L1A_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1A_OUTPUT_CACHE_FILE)
    print(f"  [L1a] model saved -> {os.path.join(MODEL_DIR, L1A_MODEL_FILE)}", flush=True)
    print(f"  [L1a] meta saved  -> {os.path.join(MODEL_DIR, L1A_META_FILE)}", flush=True)
    print(f"  [L1a] cache saved -> {cache_path}", flush=True)
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L1a] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )
    return L1ATrainingBundle(model=model, meta=meta, outputs=outputs)



def _l1a_materialize_only_resume(df: pd.DataFrame, feat_cols: list[str]) -> L1ATrainingBundle:
    train_started_perf = time.perf_counter()
    train_started_at = datetime.now().astimezone()
    print(f"  [L1a] materialize-only started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    _l1a_configure_torch_backends(DEVICE)
    resume_path = os.path.join(MODEL_DIR, L1A_TRAIN_RESUME_FILE)
    if not os.path.isfile(resume_path):
        raise RuntimeError(
            f"L1A_MATERIALIZE_ONLY=1 requires {resume_path} (written after val metrics). "
            "Run full L1a once until that file appears, then re-run with L1A_MATERIALIZE_ONLY=1."
        )
    with open(resume_path, "rb") as f:
        r = pickle.load(f)
    if r.get("schema_version") != L1A_SCHEMA_VERSION:
        raise RuntimeError(
            f"L1a resume schema mismatch: checkpoint has {r.get('schema_version')} "
            f"but code expects {L1A_SCHEMA_VERSION}."
        )
    if int(r["n_l1_oof"]) >= 2:
        raise RuntimeError(
            "L1A_MATERIALIZE_ONLY does not yet support L1_OOF_FOLDS>=2 / stitched OOF; use L1_OOF_FOLDS=1."
        )
    if str(r.get("l1_oof_mode")) != str(l1_oof_mode_from_env()):
        raise RuntimeError(
            "L1A_MATERIALIZE_ONLY: L1_OOF_MODE must match the run that wrote the resume file."
        )
    if int(r["n_l1_oof_req"]) != int(l1_oof_folds_from_env()):
        raise RuntimeError(
            "L1A_MATERIALIZE_ONLY: L1_OOF_FOLDS must match the run that wrote the resume file."
        )
    work = df.copy(deep=False)
    feature_cols_cur = _select_l1a_feature_cols(work, feat_cols)
    if list(feature_cols_cur) != list(r["feature_cols"]):
        raise RuntimeError(
            "L1A_MATERIALIZE_ONLY: L1a feature column list does not match resume checkpoint."
        )
    feature_cols = list(feature_cols_cur)
    mean = np.asarray(r["mean"], dtype=np.float32)
    std = np.asarray(r["std"], dtype=np.float32)
    splits = build_stack_time_splits(work["time_key"])
    l1_fit_mask = np.asarray(splits.train_mask | splits.cal_mask, dtype=bool)
    n_l1_oof_req = l1_oof_folds_from_env()
    norm_mask = l1_fit_mask if n_l1_oof_req >= 2 else splits.train_mask
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    Xn = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm_df = pd.concat([work[["symbol", "time_key"]], pd.DataFrame(Xn, columns=feature_cols)], axis=1)
    windows, end_idx = _build_symbol_windows(norm_df, feature_cols, _l1a_seq_len())
    targets, clip_meta = _build_l1a_targets(work, fit_mask=norm_mask)
    window_cal = splits.cal_mask[end_idx]
    window_val = splits.l2_val_mask[end_idx]
    n_l1_oof = int(n_l1_oof_req)
    if not window_val.any():
        raise RuntimeError("L1a: L2 validation window is empty for validation.")
    if not window_cal.any():
        raise RuntimeError("L1a: calibration window is empty for diagnostics.")
    window_val_report = window_val
    raw_sl = pd.to_numeric(work["state_label"], errors="coerce")
    state_label_missing_rate = float(raw_sl.isna().mean()) if len(raw_sl) else 0.0
    ch = [int(x) for x in r["ch"]]
    tcn_ks = int(r["tcn_ks"])
    td, rd, hd_drop = float(r["td"]), float(r["rd"]), float(r["hd_drop"])
    embed_dim = int(r["embed_dim"])
    model = L1AMarketTCN(
        len(feature_cols),
        channels=ch,
        seq_len=_l1a_seq_len(),
        readout_type=_l1a_readout_type(),
        min_attention_seq_len=_l1a_min_attention_seq_len(),
        tcn_kernel_size=tcn_ks,
        tcn_dropout=td,
        readout_dropout=rd,
        head_dropout=hd_drop,
        embed_dim=embed_dim,
        use_straddle_edge_head=l1a_straddle_edge_head_enabled(),
        use_vol_trend_head=l1a_vol_trend_head_enabled(),
        use_time_in_regime_head=l1a_time_in_regime_head_enabled(),
    ).to(DEVICE)
    model = _l1a_maybe_compile(model)
    model.load_state_dict({k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in r["model_state"].items()})
    uw_for_meta = None
    if r.get("uw_state") is not None:
        uw_for_meta = _l1a_build_uw_module(DEVICE)
        uw_for_meta.load_state_dict({k: v.to(DEVICE) for k, v in r["uw_state"].items()})
    l1a_amp, _ = _l1a_build_amp(DEVICE)
    th = r["train_hyper"]
    vol_forecast_clip_hi = float(r["vol_forecast_clip_hi"])
    vol_mat_clip_meta = dict(r["vol_mat_clip_meta"])
    lbl_atr_median_train = float(r["lbl_atr_median_train"])
    expanding_oof_fold_states = None
    expanding_oof_va_masks = None
    if r.get("expanding_oof_fold_states") is not None or r.get("expanding_oof_va_masks") is not None:
        raise RuntimeError("L1A_MATERIALIZE_ONLY: expanding OOF resume not implemented.")
    uw_val_metric = _l1a_uw_val_metric()
    oof_cap_message = str(th["oof_cap_message"])
    print(
        "  [L1a] full-table materialize + artifact write (resume path; no training) ...",
        flush=True,
    )
    return _l1a_run_materialize_through_save(
        model=model,
        uw_for_meta=uw_for_meta,
        work=work,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        targets=targets,
        clip_meta=clip_meta,
        vol_forecast_clip_hi=vol_forecast_clip_hi,
        vol_mat_clip_meta=vol_mat_clip_meta,
        lbl_atr_median_train=lbl_atr_median_train,
        l1a_amp=l1a_amp,
        expanding_oof_fold_states=expanding_oof_fold_states,
        expanding_oof_va_masks=expanding_oof_va_masks,
        n_l1_oof=n_l1_oof,
        windows=windows,
        end_idx=end_idx,
        window_cal=window_cal,
        window_val_report=window_val_report,
        ch=ch,
        tcn_ks=tcn_ks,
        td=td,
        rd=rd,
        hd_drop=hd_drop,
        uw_val_metric=uw_val_metric,
        lr=float(th["lr"]),
        wd=float(th["wd"]),
        T0=int(th["T0"]),
        t0_source=str(th["t0_source"]),
        Tm=int(th["Tm"]),
        cos_auto=bool(th["cos_auto"]),
        batch_size=int(th["batch_size"]),
        seq_steps_per_batch=int(th["seq_steps_per_batch"]),
        max_epochs=int(th["max_epochs"]),
        patience=int(th["patience"]),
        min_delta=float(th["min_delta"]),
        rf_steps=int(th["rf_steps"]),
        embed_dim=embed_dim,
        n_l1_oof_req=int(r["n_l1_oof_req"]),
        oof_cap_message=oof_cap_message,
        l1_final_epochs_after_oof=th.get("l1_final_epochs_after_oof"),
        state_label_missing_rate=state_label_missing_rate,
        norm_mask=norm_mask,
        train_started_perf=train_started_perf,
    )


def train_l1a_market_encoder(df: pd.DataFrame, feat_cols: list[str]) -> L1ATrainingBundle:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L1a] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    if _l1a_materialize_only_env():
        return _l1a_materialize_only_resume(df, feat_cols)

    print(f"  [L1a] torch device={DEVICE} (set TORCH_DEVICE to override)", flush=True)
    _l1a_configure_torch_backends(DEVICE)
    if _l1a_progress_tqdm_enabled():
        print(
            f"  [L1a] tqdm: epoch/fold/materialize on (L1A_PROGRESS_TQDM=1); "
            f"per-batch={'on' if _l1a_batch_progress_enabled() else 'off'} (L1A_TQDM_BATCH, default off for speed); "
            "L1A_DISABLE_TQDM=1 silences all",
            flush=True,
        )
    work = df.copy(deep=False)
    feature_cols = _select_l1a_feature_cols(work, feat_cols)
    print(
        f"  [L1a] ctx_stagger={l1_ctx_stagger_enabled()}  n_features={len(feature_cols)}",
        flush=True,
    )
    print(
        f"  [L1a] trainable auxiliary heads: vol_trend={l1a_vol_trend_head_enabled()}  "
        f"time_in_regime={l1a_time_in_regime_head_enabled()}  "
        f"(when off, decoder sees prep targets; ``l1a_vol_trend`` / ``l1a_time_in_regime`` are prep-derived)",
        flush=True,
    )
    splits = build_stack_time_splits(work["time_key"])
    l1_fit_mask = np.asarray(splits.train_mask | splits.cal_mask, dtype=bool)
    n_l1_oof_req = l1_oof_folds_from_env()
    norm_mask = l1_fit_mask if n_l1_oof_req >= 2 else splits.train_mask
    l2_vs = l2_val_start_time()
    Xn, mean, std = _normalize_l1a_matrix(work, feature_cols, norm_mask)
    norm_df = pd.concat([work[["symbol", "time_key"]], pd.DataFrame(Xn, columns=feature_cols)], axis=1)
    windows, end_idx = _build_symbol_windows(norm_df, feature_cols, _l1a_seq_len())
    if len(end_idx) == 0:
        raise RuntimeError("L1a: no valid sequence windows were created.")

    targets, clip_meta = _build_l1a_targets(work, fit_mask=norm_mask)
    _tmeta = clip_meta.get("transition_horizon") or {}
    _tpr = _tmeta.get("pos_rate_fit")
    _tpr_s = f"{float(_tpr):.4f}" if _tpr is not None else "n/a"
    _fh = int(_options_target_config()["decision_horizon_bars"])
    print(
        f"  [L1a] transition_risk lookahead H={_tmeta.get('horizon')}  pos_rate(fit)={_tpr_s}  "
        f"mode={_tmeta.get('mode')!r}  (vol_forecast still uses decision_horizon={_fh}; "
        f"single global H for this run — all OOF folds share the same transition definition)",
        flush=True,
    )
    _tr_nm = np.asarray(targets["transition_risk"][norm_mask], dtype=np.float64)
    _pos = float(np.sum(_tr_nm > 0.5))
    _neg = float(np.size(_tr_nm) - _pos)
    _den = max(_pos + _neg, 1.0)
    _pw = max(_neg / max(_pos, 1.0), 1.0)
    print(
        f"  [L1a] transition_risk on norm_mask rows: pos={int(_pos):,} neg={int(_neg):,} "
        f"pos_rate={_pos / _den:.4f}  BCE pos_weight={_pw:.4f} (from these labels; each OOF fold loss uses that fold's train window slice)",
        flush=True,
    )
    _l1a_log_regime_class_distribution(
        targets["regime"],
        norm_mask,
        title="vol5 regime distribution (rows in norm_mask, used for clip stats / transition calibration)",
    )
    _l1a_log_transition_h_stability_diag(
        work,
        targets["regime"],
        work["symbol"].to_numpy(),
        norm_mask,
        _fh,
    )
    raw_sl = pd.to_numeric(work["state_label"], errors="coerce")
    state_label_missing_rate = float(raw_sl.isna().mean()) if len(raw_sl) else 0.0
    print(f"  [L1a] state_label missing rate (pre-imputation in targets): {state_label_missing_rate:.4%}", flush=True)
    if state_label_missing_rate > 0.10:
        print("  [L1a][warn] state_label missing > 10%; check upstream label generation quality.", flush=True)
    if _l1a_adaptive_vol_forecast_clip_enabled():
        vol_forecast_clip_hi, vol_mat_clip_meta = _l1a_compute_vol_forecast_materialize_clip(
            targets["vol_forecast"][norm_mask]
        )
    else:
        vol_forecast_clip_hi = float(os.environ.get("L1A_VOL_FORECAST_MATERIALIZE_MAX", "5.0"))
        vol_mat_clip_meta = {"clip_upper": vol_forecast_clip_hi, "fixed": True, "adaptive_disabled": True}
    print(
        f"  [L1a] vol_forecast materialize clip hi={vol_forecast_clip_hi:.4f} ({vol_mat_clip_meta.get('statistical_principle', 'fixed')})",
        flush=True,
    )
    window_train = splits.train_mask[end_idx]
    window_cal = splits.cal_mask[end_idx]
    window_val = splits.l2_val_mask[end_idx]
    window_pool = l1_fit_mask[end_idx]
    n_w = len(end_idx)
    n_l1_oof = int(n_l1_oof_req)
    oof_cap_message = ""
    if n_l1_oof >= 2 and _l1_oof_auto_cap_enabled() and l1_oof_mode_from_env() != "expanding":
        n_l1_oof, oof_cap_message = _l1a_cap_oof_folds(n_l1_oof, int(window_pool.sum()))
        if oof_cap_message:
            print(f"  [L1a] {oof_cap_message}", flush=True)
    if n_l1_oof >= 2:
        if l1_oof_mode_from_env() == "expanding":
            print(
                f"  [L1a] expanding calendar OOF: {n_l1_oof} folds on window ends (t < {CAL_END}); "
                f"L1_OOF_MODE=expanding",
                flush=True,
            )
        else:
            print(
                f"  [L1a] blocked time OOF: L1_OOF_FOLDS={n_l1_oof} (requested={n_l1_oof_req}) on train+cal window ends (t < {CAL_END}) "
                f"(set L1_OOF_FOLDS=1 for legacy train vs l2_val [{l2_vs}, {CAL_END}); L1_OOF_AUTO_CAP=0 to disable capping)",
                flush=True,
            )
        if not window_pool.any():
            raise RuntimeError("L1a OOF: no windows with end bar in train+cal.")
        tune_frac = float(os.environ.get("L1_TUNE_FRAC_WITHIN_FIT", "0.5"))
        val_tune_mask, val_report_mask = split_mask_for_tuning_and_report(
            work["time_key"], l1_fit_mask, tune_frac=tune_frac, min_rows_each=50
        )
        if not val_tune_mask.any() or not val_report_mask.any():
            raise RuntimeError("L1a OOF: failed to build tune/report masks inside train+cal.")
        window_val_report = val_report_mask[end_idx]
        if not window_val_report.any():
            raise RuntimeError("L1a OOF: no windows in fit_report slice.")
    else:
        if not window_val.any():
            raise RuntimeError("L1a: L2 validation window is empty for validation.")
        if not window_cal.any():
            raise RuntimeError("L1a: calibration window is empty for diagnostics.")
        window_val_report = window_val

    X_t = torch.from_numpy(windows.astype(np.float32, copy=False))
    time_ir_fit = _transform_time_in_regime_target(targets["time_in_regime"])
    ds_tensors: list[torch.Tensor] = [
        X_t,
        torch.from_numpy(targets["regime"][end_idx].astype(np.int64)),
        torch.from_numpy(targets["transition_risk"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["vol_forecast"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["vol_trend"][end_idx].astype(np.float32)),
        torch.from_numpy(time_ir_fit[end_idx].astype(np.float32)),
    ]
    if l1a_straddle_edge_head_enabled():
        ds_tensors.append(torch.from_numpy(targets["straddle_edge"][end_idx].astype(np.float32)))
    ds = TensorDataset(*ds_tensors)
    val_idx = np.flatnonzero(window_val_report).astype(np.int64, copy=False)
    cal_idx = np.flatnonzero(window_cal).astype(np.int64, copy=False)
    val_ds = Subset(ds, val_idx.tolist())
    cal_ds = Subset(ds, cal_idx.tolist())
    bs_raw = os.environ.get("L1A_BATCH_SIZE", "").strip()
    if bs_raw:
        batch_size = max(32, int(bs_raw))
    elif FAST_TRAIN_MODE:
        batch_size = 512
    elif DEVICE.type == "cuda":
        batch_size = 2048
    elif DEVICE.type == "mps":
        batch_size = 1536
    else:
        batch_size = 1024
    loader_workers = _l1a_dataloader_workers(min(4, max(_lgbm_n_jobs(), 1)))
    if os.name == "nt" and n_l1_oof >= 2:
        # Windows OOF with multi-worker DataLoader can fail with shared mapping error 1455.
        loader_workers = 0
        print("  [L1a] forcing DataLoader num_workers=0 on Windows OOF to avoid shared-memory mapping failures", flush=True)
    pin_memory = DEVICE.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, int(os.environ.get("L1A_PREFETCH_FACTOR", "4")))
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    cal_dl = DataLoader(cal_ds, shuffle=False, **loader_kwargs)
    n_feat_l1a = len(feature_cols)
    seq_steps_per_batch = int(batch_size) * int(_l1a_seq_len())
    print(
        f"  [L1a] batch_size={batch_size}  (~{seq_steps_per_batch:,} seq steps/batch = batch×seq_len, "
        f"×{n_feat_l1a} feats); smaller L1A_BATCH_SIZE → noisier gradients, larger → smoother",
        flush=True,
    )

    embed_dim = _l1a_embed_dim()
    log_w_tr = window_pool if n_l1_oof >= 2 else window_train

    log_layer_banner("[L1a] Sequence Market Encoder (TCN)")
    if n_l1_oof >= 2:
        log_time_key_arrays(
            "L1a",
            work.iloc[end_idx[log_w_tr]]["time_key"],
            work.iloc[end_idx[window_val_report]]["time_key"],
            train_label="window fit (end in train+cal)",
            val_label="window report (late slice in train+cal)",
            extra_note=f"OOF: {n_l1_oof} folds; late-time report slice for metrics.",
        )
    else:
        log_time_key_arrays(
            "L1a",
            work.iloc[end_idx[window_train]]["time_key"],
            work.iloc[end_idx[window_val]]["time_key"],
            train_label="window train (end_idx in train split)",
            val_label="window val (end_idx in l2_val split)",
            extra_note="Primary L1a early stopping/reporting uses l2_val end-bars; full cal remains secondary diagnostic.",
        )
    w_tr = windows[log_w_tr]
    log_numpy_x_stats(
        "L1a",
        w_tr.reshape(w_tr.shape[0], -1),
        label="windows[fit] (flattened seq×feat)" if n_l1_oof >= 2 else "windows[train] (flattened seq×feat)",
    )
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
    print(f"  [L1a] seq input: seq_len={_l1a_seq_len()}  input_feats={len(feature_cols)}", flush=True)
    print(f"  [L1a] artifact dir: {MODEL_DIR}", flush=True)
    print(
        f"  [L1a] will write: {artifact_path(L1A_MODEL_FILE)} | {artifact_path(L1A_META_FILE)} | {artifact_path(L1A_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    print(
        "  [L1a] note: forward uses this run's weights/data (not loading L1a from disk for features).",
        flush=True,
    )
    log_label_baseline("l1a_regime", targets["regime"][end_idx][log_w_tr], task="cls")
    log_label_baseline("l1a_transition_risk", targets["transition_risk"][end_idx][log_w_tr], task="cls")
    log_label_baseline("l1a_vol_forecast", targets["vol_forecast"][end_idx][log_w_tr], task="reg")
    log_label_baseline("l1a_vol_trend", targets["vol_trend"][end_idx][log_w_tr], task="reg")
    log_label_baseline("l1a_time_in_regime", targets["time_in_regime"][end_idx][log_w_tr], task="reg")
    if l1a_straddle_edge_head_enabled():
        log_label_baseline("l1a_straddle_edge", targets["straddle_edge"][end_idx][log_w_tr], task="reg")

    ch = _l1a_tcn_channels()
    td = _l1a_tcn_dropout()
    rd = _l1a_readout_dropout()
    hd_drop = _l1a_head_dropout()
    lr = float(os.environ.get("L1A_LR", "5e-4"))
    wd = float(os.environ.get("L1A_WEIGHT_DECAY", "1e-3"))
    Tm = max(1, int(os.environ.get("L1A_COS_T_MULT", "2")))
    max_epochs = max(4, int(os.environ.get("L1A_MAX_EPOCHS", "24")))
    t0_env = os.environ.get("L1A_COS_T0", "").strip()
    cos_auto = os.environ.get("L1A_COS_AUTO", "1").strip().lower() in {"1", "true", "yes"}
    if t0_env:
        T0 = max(1, int(t0_env))
        t0_source = "env"
    elif cos_auto:
        T0 = max(2, max_epochs // (1 + Tm))
        t0_source = f"auto(max_epochs//(1+T_mult)={max_epochs}//{1 + Tm})"
    else:
        T0 = 5
        t0_source = "default_fixed"
    span_first_two = T0 + T0 * Tm
    if span_first_two > max_epochs:
        print(
            f"  [L1a][warn] CosineAnnealingWarmRestarts: T0={T0} T_mult={Tm} → first two segments span {span_first_two} epochs "
            f"> max_epochs={max_epochs}; increase L1A_MAX_EPOCHS or lower L1A_COS_T0 (L1A_COS_AUTO=1 uses T0=max(2, max_epochs//(1+T_mult))).",
            flush=True,
        )
    elif max_epochs < T0 * (1 + Tm + Tm * Tm) and not FAST_TRAIN_MODE:
        print(
            f"  [L1a] cosine: T0={T0} T_mult={Tm} max_epochs={max_epochs} (T0 from {t0_source}); "
            f"third restart cycle would need ~{T0 + T0 * Tm + T0 * Tm * Tm} epochs to complete.",
            flush=True,
        )
    patience = max(2, int(os.environ.get("L1A_PATIENCE", "10")))
    min_delta = float(os.environ.get("L1A_EARLY_STOP_MIN_DELTA", "5e-4"))
    l1_final_epochs_after_oof: int | None = None
    expanding_oof_fold_states: list[dict[str, torch.Tensor]] | None = None
    expanding_oof_va_masks: list[np.ndarray] | None = None

    tcn_ks = _l1a_tcn_kernel_size()
    rf_steps = _l1a_tcn_receptive_field_steps(n_layers=len(ch), kernel_size=tcn_ks)
    print(
        f"  [L1a] TCN stack: channels={ch}  kernel={tcn_ks}  "
        f"dilations=1..2^{len(ch) - 1}  receptive_field~{rf_steps} steps / seq_len={_l1a_seq_len()} "
        f"({100.0 * rf_steps / max(_l1a_seq_len(), 1):.1f}% of sequence)",
        flush=True,
    )

    uw_val_metric = _l1a_uw_val_metric()
    uw_for_meta: L1aMultiTaskUncertaintyWeights | None = None
    print(
        f"  [L1a] multi-task loss: Kendall uncertainty (per-task log σ²); val early-stop metric={uw_val_metric!r}  "
        f"regime_aux_coef={_l1a_regime_aux_coef():g}  persist_coef={_l1a_transition_persist_coef():g}  "
        f"uw_lr_ratio={_l1a_uw_lr_ratio():g}",
        flush=True,
    )

    l1a_amp, l1a_train_scaler = _l1a_build_amp(DEVICE)
    if l1a_amp.enabled:
        if _l1a_is_macos() and l1a_amp.device_type == "mps":
            print(
                f"  [L1a] AMP: macOS MPS float16 autocast (fixed on this platform; GradScaler=n/a)",
                flush=True,
            )
        else:
            print(
                f"  [L1a] AMP: dtype={l1a_amp.dtype}  GradScaler={'on' if l1a_train_scaler is not None else 'off (bf16)'}  "
                f"(CUDA: L1A_AMP_DTYPE=auto|fp16|bf16)",
                flush=True,
            )
    else:
        print(
            "  [L1a] AMP off — Apple Silicon + MPS uses float16 autocast automatically; CUDA needs L1A_AMP=1",
            flush=True,
        )

    if n_l1_oof >= 2:
        if l1_oof_mode_from_env() == "expanding":
            t_end = work["time_key"].to_numpy()[end_idx]
            l1a_vw = l1a_expand_oof_val_windows_from_env()
            if l1a_vw is not None:
                print(
                    f"  [L1a] expanding OOF: L1A_EXPAND_OOF_VAL_WINDOWS → {len(l1a_vw)} folds "
                    f"(global L1_EXPAND_OOF_VAL_WINDOWS unchanged for L1b/L2).",
                    flush=True,
                )
            exp_folds = l1_expanding_oof_window_folds(t_end, val_windows=l1a_vw)
            fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for w_tr, w_va in exp_folds:
                w_tr_f = w_tr & window_pool
                w_va_f = w_va & window_pool
                if not w_tr_f.any() or not w_va_f.any():
                    raise RuntimeError(
                        "L1a expanding OOF: empty train or val windows "
                        f"(train={int(w_tr_f.sum())}, val={int(w_va_f.sum())})."
                    )
                fold_pairs.append((w_tr_f, w_va_f))
        else:
            w_pool_idx = np.flatnonzero(window_pool)
            tk_sub = work["time_key"].to_numpy()[end_idx[w_pool_idx]]
            fold_masks = time_blocked_fold_masks(tk_sub, np.ones(len(w_pool_idx), bool), n_l1_oof, context="L1a OOF")
            fold_pairs = []
            for tr_sub, va_sub in fold_masks:
                w_tr_f = np.zeros(n_w, dtype=bool)
                w_va_f = np.zeros(n_w, dtype=bool)
                w_tr_f[w_pool_idx[tr_sub]] = True
                w_va_f[w_pool_idx[va_sub]] = True
                fold_pairs.append((w_tr_f, w_va_f))
        n_oof_fold = len(fold_pairs)
        if (
            l1_oof_mode_from_env() == "expanding"
            and _l1a_oof_warmstart_expanding_enabled()
            and len(fold_pairs) >= 2
        ):
            wme = _l1a_warmstart_oof_max_epochs(max_epochs)
            print(
                f"  [L1a] OOF warmstart: ON  folds 2..K load previous fold weights; "
                f"max_epochs={wme} (cap vs base {max_epochs})  lr_scale={_l1a_warmstart_oof_lr_scale():g}  "
                f"(L1A_WARMSTART_OOF_MAX_EPOCHS / L1A_WARMSTART_OOF_LR_SCALE; L1A_OOF_WARMSTART=0 to disable)",
                flush=True,
            )
        best_eps: list[int] = []
        fold_states: list[dict[str, torch.Tensor]] = []
        prev_oof_state: dict[str, torch.Tensor] | None = None
        fold_loops = list(enumerate(fold_pairs))
        if _l1a_progress_tqdm_enabled():
            fold_loops = tqdm(
                fold_loops,
                total=len(fold_pairs),
                desc="[L1a] OOF folds",
                unit="fold",
                leave=True,
                file=_tqdm_stream(),
                mininterval=0.5,
                dynamic_ncols=True,
                disable=False,
            )
        for fk, (w_tr_f, w_va_f) in fold_loops:
            regime_loss_f = _l1a_regime_loss(targets["regime"][end_idx][w_tr_f], device=DEVICE)
            transition_loss_f = _l1a_transition_loss(targets["transition_risk"][end_idx][w_tr_f], device=DEVICE)
            tr_idx = np.flatnonzero(w_tr_f).astype(np.int64, copy=False)
            va_idx = np.flatnonzero(w_va_f).astype(np.int64, copy=False)
            train_ds_f = Subset(ds, tr_idx.tolist())
            val_ds_f = Subset(ds, va_idx.tolist())
            train_dl_f = DataLoader(train_ds_f, shuffle=True, **loader_kwargs)
            val_dl_f = DataLoader(val_ds_f, shuffle=False, **loader_kwargs)
            fold_max_epochs = int(max_epochs)
            fold_lr = float(lr)
            init_sd: dict[str, torch.Tensor] | None = None
            if (
                fk > 0
                and prev_oof_state is not None
                and l1_oof_mode_from_env() == "expanding"
                and _l1a_oof_warmstart_expanding_enabled()
            ):
                init_sd = prev_oof_state
                fold_max_epochs = _l1a_warmstart_oof_max_epochs(max_epochs)
                fold_lr = float(lr) * _l1a_warmstart_oof_lr_scale()
                print(
                    f"  [L1a] OOF fold {fk + 1}: warmstart  max_epochs={fold_max_epochs}  lr={fold_lr:.6g}",
                    flush=True,
                )
            be, st = _l1a_early_stop_best_epoch(
                train_dl_f,
                val_dl_f,
                n_feat=len(feature_cols),
                embed_dim=embed_dim,
                channels=ch,
                tcn_kernel_size=tcn_ks,
                tcn_dropout=td,
                readout_dropout=rd,
                head_dropout=hd_drop,
                regime_loss=regime_loss_f,
                transition_loss=transition_loss_f,
                lr=fold_lr,
                wd=wd,
                T0=T0,
                Tm=Tm,
                max_epochs=fold_max_epochs,
                patience=patience,
                min_delta=min_delta,
                desc=f"[L1a] oof {fk + 1}/{n_oof_fold}",
                uw_val_metric=uw_val_metric,
                amp=l1a_amp,
                init_state_dict=init_sd,
            )
            best_eps.append(be)
            fold_states.append(st)
            prev_oof_state = st
            print(f"  [L1a] OOF fold {fk + 1}/{n_oof_fold}: best_epoch={be}", flush=True)
        nr = int(np.clip(np.median(best_eps), 1, max_epochs))
        l1_final_epochs_after_oof = int(nr)
        print(f"  [L1a] OOF median best_epoch -> final_epochs={nr}", flush=True)
        if l1_oof_mode_from_env() == "expanding":
            expanding_oof_fold_states = fold_states
            expanding_oof_va_masks = [p[1] for p in fold_pairs]
        regime_loss = _l1a_regime_loss(targets["regime"][end_idx][window_pool], device=DEVICE)
        transition_loss = _l1a_transition_loss(targets["transition_risk"][end_idx][window_pool], device=DEVICE)
        train_idx = np.flatnonzero(window_pool).astype(np.int64, copy=False)
        train_ds = Subset(ds, train_idx.tolist())
        train_dl = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        model = L1AMarketTCN(
            len(feature_cols),
            channels=ch,
            seq_len=_l1a_seq_len(),
            readout_type=_l1a_readout_type(),
            min_attention_seq_len=_l1a_min_attention_seq_len(),
            tcn_kernel_size=tcn_ks,
            tcn_dropout=td,
            readout_dropout=rd,
            head_dropout=hd_drop,
            embed_dim=embed_dim,
            use_straddle_edge_head=l1a_straddle_edge_head_enabled(),
            use_vol_trend_head=l1a_vol_trend_head_enabled(),
            use_time_in_regime_head=l1a_time_in_regime_head_enabled(),
        ).to(DEVICE)
        model = _l1a_maybe_compile(model)
        uw_final = _l1a_build_uw_module(DEVICE)
        optimizer = _l1a_optimizer(model, uw_final, lr, wd)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm)
        print(
            f"  [L1a] readout={model.readout_type}  min_attention_seq_len={model.min_attention_seq_len}  "
            f"tcn_kernel={model.tcn_kernel_size}  tcn_channels={ch}  tcn_dropout={td}  "
            f"lr={lr}  weight_decay={wd}  cosine(T0={T0},T_mult={Tm})  "
            f"final_epochs={nr} (OOF median)  loss=Kendall_UW",
            flush=True,
        )
        epoch_bar = trange(
            nr,
            desc="[L1a] final fit",
            unit="ep",
            leave=True,
            file=_tqdm_stream(),
            mininterval=0.3,
            dynamic_ncols=True,
            disable=not _l1a_progress_tqdm_enabled(),
        )
        for epoch in epoch_bar:
            tr_loss = _train_epoch(
                model,
                train_dl,
                optimizer,
                DEVICE,
                regime_loss=regime_loss,
                transition_loss=transition_loss,
                amp=l1a_amp,
                amp_scaler=l1a_train_scaler,
                uw_module=uw_final,
            )
            scheduler.step()
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(train=f"{tr_loss:.4f}", refresh=False)
            print(f"  [L1a] epoch={epoch + 1:02d} train_loss={tr_loss:.4f}", flush=True)
            _l1a_log_uw_auto_weights(uw_final)
        uw_for_meta = uw_final
    else:
        regime_loss = _l1a_regime_loss(targets["regime"][end_idx][window_train], device=DEVICE)
        transition_loss = _l1a_transition_loss(targets["transition_risk"][end_idx][window_train], device=DEVICE)
        train_ds = TensorDataset(*[tensor[window_train] for tensor in ds.tensors])
        train_dl = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        model = L1AMarketTCN(
            len(feature_cols),
            channels=ch,
            seq_len=_l1a_seq_len(),
            readout_type=_l1a_readout_type(),
            min_attention_seq_len=_l1a_min_attention_seq_len(),
            tcn_kernel_size=tcn_ks,
            tcn_dropout=td,
            readout_dropout=rd,
            head_dropout=hd_drop,
            embed_dim=embed_dim,
            use_straddle_edge_head=l1a_straddle_edge_head_enabled(),
            use_vol_trend_head=l1a_vol_trend_head_enabled(),
            use_time_in_regime_head=l1a_time_in_regime_head_enabled(),
        ).to(DEVICE)
        model = _l1a_maybe_compile(model)
        uw = _l1a_build_uw_module(DEVICE)
        optimizer = _l1a_optimizer(model, uw, lr, wd)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm)
        best_state: dict[str, torch.Tensor] | None = None
        best_uw_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        stale = 0
        epoch_bar = trange(
            max_epochs,
            desc="[L1a] epochs",
            unit="ep",
            leave=True,
            file=_tqdm_stream(),
            mininterval=0.3,
            dynamic_ncols=True,
            disable=not _l1a_progress_tqdm_enabled(),
        )
        print(
            f"  [L1a] readout={model.readout_type}  min_attention_seq_len={model.min_attention_seq_len}  "
            f"tcn_kernel={model.tcn_kernel_size}  tcn_channels={ch}  tcn_dropout={td}  "
            f"lr={lr}  weight_decay={wd}  cosine(T0={T0},T_mult={Tm})  "
            f"max_epochs={max_epochs}  patience={patience}  min_delta={min_delta:g}  loss=Kendall_UW",
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
                amp=l1a_amp,
                amp_scaler=l1a_train_scaler,
                uw_module=uw,
            )
            va_stop, va_diag = _eval_epoch(
                model,
                val_dl,
                DEVICE,
                regime_loss=regime_loss,
                transition_loss=transition_loss,
                amp=l1a_amp,
                uw_module=uw,
                val_metric=uw_val_metric,
            )
            scheduler.step()
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(
                    train=f"{tr_loss:.4f}",
                    val=f"{va_stop:.4f}",
                    leg=f"{va_diag.get('val_legacy_weighted', 0):.4f}",
                    refresh=False,
                )
            print(
                f"  [L1a] epoch={epoch + 1:02d} train_loss={tr_loss:.4f} val_stop={va_stop:.4f} "
                f"val_legacy_sum={va_diag.get('val_legacy_weighted', 0):.4f}",
                flush=True,
            )
            _l1a_log_uw_auto_weights(uw)
            if va_stop < (best_val - min_delta):
                best_val = va_stop
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_uw_state = {k: v.detach().cpu().clone() for k, v in uw.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("L1a: training failed to produce a checkpoint.")
        model.load_state_dict(best_state)
        if best_uw_state is not None:
            uw.load_state_dict(best_uw_state)
        uw_for_meta = uw

    _log_l1a_val_metrics(
        model, val_dl, DEVICE, label="fit_report" if n_l1_oof >= 2 else "l2_val", amp=l1a_amp
    )
    if window_cal.sum() != window_val_report.sum() and os.environ.get("L1A_SKIP_CAL_FULL_METRICS", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        _log_l1a_val_metrics(model, cal_dl, DEVICE, label="cal_full", amp=l1a_amp)

    _tr_m = np.asarray(l1_fit_mask if n_l1_oof >= 2 else splits.train_mask, dtype=bool)
    _atr_tr = pd.to_numeric(work["lbl_atr"], errors="coerce").to_numpy(dtype=np.float64)
    _fin_tr = _tr_m & np.isfinite(_atr_tr)
    lbl_atr_median_train = float(np.median(_atr_tr[_fin_tr])) if np.any(_fin_tr) else 1.0

    _l1a_save_train_resume_checkpoint(
        model=model,
        uw_for_meta=uw_for_meta,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        clip_meta=clip_meta,
        vol_forecast_clip_hi=vol_forecast_clip_hi,
        vol_mat_clip_meta=vol_mat_clip_meta,
        lbl_atr_median_train=lbl_atr_median_train,
        ch=ch,
        tcn_ks=tcn_ks,
        td=td,
        rd=rd,
        hd_drop=hd_drop,
        embed_dim=embed_dim,
        n_l1_oof=n_l1_oof,
        n_l1_oof_req=n_l1_oof_req,
        expanding_oof_fold_states=expanding_oof_fold_states,
        expanding_oof_va_masks=expanding_oof_va_masks,
        lr=lr,
        wd=wd,
        T0=T0,
        t0_source=t0_source,
        Tm=Tm,
        cos_auto=cos_auto,
        batch_size=batch_size,
        seq_steps_per_batch=seq_steps_per_batch,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        rf_steps=rf_steps,
        oof_cap_message=oof_cap_message,
        l1_final_epochs_after_oof=l1_final_epochs_after_oof,
    )
    print(
        "  [L1a] full-table materialize + artifact write (long -- do not kill the process) ...",
        flush=True,
    )
    return _l1a_run_materialize_through_save(
        model=model,
        uw_for_meta=uw_for_meta,
        work=work,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        targets=targets,
        clip_meta=clip_meta,
        vol_forecast_clip_hi=vol_forecast_clip_hi,
        vol_mat_clip_meta=vol_mat_clip_meta,
        lbl_atr_median_train=lbl_atr_median_train,
        l1a_amp=l1a_amp,
        expanding_oof_fold_states=expanding_oof_fold_states,
        expanding_oof_va_masks=expanding_oof_va_masks,
        n_l1_oof=n_l1_oof,
        windows=windows,
        end_idx=end_idx,
        window_cal=window_cal,
        window_val_report=window_val_report,
        ch=ch,
        tcn_ks=tcn_ks,
        td=td,
        rd=rd,
        hd_drop=hd_drop,
        uw_val_metric=uw_val_metric,
        lr=lr,
        wd=wd,
        T0=T0,
        t0_source=t0_source,
        Tm=Tm,
        cos_auto=cos_auto,
        batch_size=batch_size,
        seq_steps_per_batch=seq_steps_per_batch,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        rf_steps=rf_steps,
        embed_dim=embed_dim,
        n_l1_oof_req=n_l1_oof_req,
        oof_cap_message=oof_cap_message,
        l1_final_epochs_after_oof=l1_final_epochs_after_oof,
        state_label_missing_rate=state_label_missing_rate,
        norm_mask=norm_mask,
        train_started_perf=train_started_perf,
    )



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
        ch = [64, 64, 64, 128, 128]
    else:
        ch = [int(x) for x in ch]
    td = float(meta.get("tcn_dropout", 0.15))
    rd = float(meta.get("readout_dropout", 0.10))
    hd_drop = float(meta.get("head_dropout", 0.1))
    embed_dim = int(meta.get("embed_dim", _l1a_embed_dim()))
    tcn_ks_load = int(meta.get("l1a_tcn_kernel_size", 3))
    model = L1AMarketTCN(
        len(feature_cols),
        channels=ch,
        seq_len=int(meta.get("seq_len", _l1a_seq_len())),
        readout_type=str(meta.get("readout_type", _l1a_readout_type())),
        min_attention_seq_len=int(meta.get("min_attention_seq_len", _l1a_min_attention_seq_len())),
        tcn_kernel_size=tcn_ks_load,
        tcn_dropout=td,
        readout_dropout=rd,
        head_dropout=hd_drop,
        embed_dim=embed_dim,
        use_straddle_edge_head=bool(meta.get("l1a_straddle_edge_head", False)),
        use_vol_trend_head=bool(meta.get("l1a_vol_trend_head", True)),
        use_time_in_regime_head=bool(meta.get("l1a_time_in_regime_head", True)),
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
    seq_len = int(meta.get("seq_len", _l1a_seq_len()))
    cold_raw = meta.get("lbl_atr_median_train")
    cold_vol = float(cold_raw) if cold_raw is not None and np.isfinite(float(cold_raw)) else None
    vraw = meta.get("l1a_vol_forecast_clip_hi")
    vclip = float(vraw) if vraw is not None and np.isfinite(float(vraw)) else None
    cfg = _options_target_config()
    trans_h = int(meta.get("l1a_transition_horizon_bars", cfg["decision_horizon_bars"]))
    return materialize_l1a_outputs(
        model,
        work,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        device=DEVICE,
        embed_dim=model.embed_dim,
        cold_vol_default=cold_vol,
        vol_forecast_clip_hi=vclip,
        transition_horizon_bars=trans_h,
    )

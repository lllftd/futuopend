from __future__ import annotations

import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from core.trainers.constants import (
    FAST_TRAIN_MODE,
    L1C_META_FILE,
    L1C_MODEL_FILE,
    L1C_OUTPUT_CACHE_FILE,
    L1C_SCHEMA_VERSION,
    MODEL_DIR,
)
from core.trainers.data_prep import _create_tcn_windows
from core.trainers.l1c.config import L1cConfig
from core.trainers.l1c.evaluate import evaluate_l1c, print_l1c_eval_report
from core.trainers.l1c.losses import L1cBinaryDirectionLoss
from core.trainers.l1c.model import L1cDirectionModel
from core.trainers.lgbm_utils import TQDM_FILE, _lgb_round_tqdm_enabled
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner
from core.trainers.stack_v2_common import build_stack_time_splits, log_label_baseline, save_output_cache
from core.trainers.threshold_registry import attach_threshold_registry, threshold_entry
from core.trainers.tcn_constants import DEVICE


def l1c_output_columns() -> list[str]:
    return [
        "l1c_direction_prob",
        "l1c_direction_score",
        "l1c_confidence",
        "l1c_direction_strength",
        "l1c_is_warm",
    ]


def _l1c_augment_batch(x: torch.Tensor, *, training: bool, seq_len: int) -> torch.Tensor:
    if not training:
        return x
    if os.environ.get("L1C_AUGMENT", "1").strip().lower() in {"0", "false", "no"}:
        return x
    out = x
    b, t, f = out.shape
    crop_prob = float(os.environ.get("L1C_AUG_CROP_PROB", "0.3"))
    if crop_prob > 0 and t >= 50 and torch.rand((), device=out.device) < crop_prob:
        start = int(torch.randint(0, max(1, t - 49), (1,), device=out.device).item())
        chunk = out[:, start : start + 50, :].transpose(1, 2)
        out = torch.nn.functional.interpolate(chunk, size=t, mode="linear", align_corners=False).transpose(1, 2)
    noise_std = float(os.environ.get("L1C_AUG_NOISE_STD", "0.01"))
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    feat_zero = float(os.environ.get("L1C_AUG_FEAT_ZERO_PROB", "0.10"))
    if feat_zero > 0 and feat_zero < 1:
        keep = (torch.rand(1, 1, f, device=out.device) > feat_zero).to(dtype=out.dtype)
        out = out * keep
    return out


def _sample_weights_from_abs_return(abs_ret: np.ndarray) -> np.ndarray:
    """Larger |return| gets larger weight (rank / N)."""
    n = int(abs_ret.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if os.environ.get("L1C_SAMPLE_WEIGHT_MODE", "").strip().lower() == "uniform":
        return np.ones(n, dtype=np.float32)
    abs_ret = np.asarray(abs_ret, dtype=np.float64).ravel()
    order = np.argsort(abs_ret, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return (ranks / float(max(n, 1))).astype(np.float32)


def _robust_sigma(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = float(mad / 0.67448975) if mad > 0 else 0.0
    return med, mad, sigma


def _l1c_forward_dict(model: L1cDirectionModel, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(x)
    if isinstance(out, dict):
        return out
    logits = out
    return {
        "direction_logits": logits,
        "direction_strength": logits.new_zeros(logits.shape),
    }


def _print_l1c_binary_label_stats(y_bin: np.ndarray, y_ret: np.ndarray, *, prefix: str) -> None:
    y_bin = np.asarray(y_bin, dtype=np.float64).ravel()
    y_ret = np.asarray(y_ret, dtype=np.float64).ravel()
    fin = np.isfinite(y_bin) & np.isfinite(y_ret)
    y_bin, y_ret = y_bin[fin], y_ret[fin]
    if y_bin.size == 0:
        print(f"  [L1c] {prefix}: no finite labels", flush=True)
        return
    up = float(np.mean(y_bin > 0.5))
    ar = np.abs(y_ret)
    print(
        f"  [L1c] {prefix}: P(up)={up:.2%}  |ret| mean={ar.mean():.6f} median={np.median(ar):.6f}",
        flush=True,
    )


def _first_epoch_diagnostic(
    model: L1cDirectionModel,
    train_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 24,
) -> None:
    model.eval()
    probs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for b_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].numpy().ravel()
            logits = _l1c_forward_dict(model, x)["direction_logits"]
            p = torch.sigmoid(logits.view(-1)).cpu().numpy()
            probs.append(p)
            trues.append(y)
            if b_idx + 1 >= max_batches:
                break
    model.train()
    pr = np.concatenate(probs)
    t = np.concatenate(trues)
    print("\n  " + "=" * 50, flush=True)
    print("  [L1c] FIRST EPOCH DIAGNOSTIC (subset of train batches)", flush=True)
    print("  " + "=" * 50, flush=True)
    print(
        f"  pred_prob — mean:{pr.mean():.4f} std:{pr.std():.4f} min:{pr.min():.4f} max:{pr.max():.4f}",
        flush=True,
    )
    print(f"  y_up — mean:{t.mean():.4f} (fraction up)", flush=True)
    if pr.std() < 0.01:
        print("  [L1c] WARNING: pred_prob std < 0.01 — outputs nearly constant", flush=True)
    acc = np.mean((pr >= 0.5) == (t > 0.5))
    confident = np.abs(pr - 0.5) > 0.2
    print(f"  batch subset accuracy vs0.5 threshold: {acc:.4f}", flush=True)
    print(
        f"  batch subset |p-0.5|>0.2 coverage: {float(np.mean(confident)):.2%}  "
        f"n={int(np.sum(confident)):,}",
        flush=True,
    )
    print("  " + "=" * 50 + "\n", flush=True)


def _select_l1c_feature_cols(df: pd.DataFrame, feat_cols: list[str], config: L1cConfig) -> list[str]:
    raw = os.environ.get("L1C_FEATURE_SUBSET", "").strip()
    if raw:
        names = [s.strip() for s in raw.split(",") if s.strip()]
        out = [c for c in names if c in df.columns]
        if not out:
            raise RuntimeError(
                f"L1c: L1C_FEATURE_SUBSET produced no valid columns. First names: {names[:12]!r}"
            )
        return out
    cand = [c for c in feat_cols if c in df.columns and c not in {"symbol", "time_key"}]
    cap = max(1, int(config.max_train_features))
    return cand[:cap]


def _normalize_l1c_matrix(
    df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    tr = X[train_mask]
    mean = tr.mean(axis=0).astype(np.float32)
    std = tr.std(axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)
    return Xn, mean, std


def _l1c_build_symbol_windows(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
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


def _build_l1c_windows_labels(
    df: pd.DataFrame,
    Xn: np.ndarray,
    *,
    feature_cols: list[str],
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_df = pd.DataFrame(Xn, columns=feature_cols, index=df.index)
    work_x = pd.concat([df[["symbol", "time_key"]], feat_df], axis=1)
    windows_list: list[np.ndarray] = []
    end_list: list[np.ndarray] = []
    ybin_list: list[np.ndarray] = []
    ret_list: list[np.ndarray] = []
    neutral_scale_list: list[np.ndarray] = []
    flat_band_stats: list[dict[str, float]] = []
    flat_mode = (os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()
    flat_alpha = float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20))
    flat_power = float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99))
    weight_mode = (os.environ.get("L1C_FLAT_WEIGHT_MODE", "snr_gaussian") or "snr_gaussian").strip().lower()
    for _, grp in work_x.groupby("symbol", sort=False):
        x = grp[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, end_local = _create_tcn_windows(x, seq_len)
        if len(end_local) == 0:
            continue
        sub = df.loc[grp.index]
        close = pd.to_numeric(sub["close"], errors="coerce").astype(np.float64).to_numpy()
        close = pd.Series(close).ffill().bfill().to_numpy(dtype=np.float64)
        valid = end_local + horizon < len(grp)
        if not valid.any():
            continue
        windows = windows[valid]
        end_local = end_local[valid]
        idx_global = grp.index.to_numpy()[end_local]
        el = end_local
        ret = (close[el + horizon] - close[el]) / (np.abs(close[el]) + 1e-9)
        abs_ret = np.abs(ret)
        flat_q = float(np.clip(float(os.environ.get("L1C_DIRECTION_FLAT_Q", "0.20")), 0.0, 0.45))
        med_ret, mad_ret, sigma_ret = _robust_sigma(ret)
        if flat_mode == "quantile":
            flat_band = float(np.quantile(abs_ret[np.isfinite(abs_ret)], flat_q)) if np.isfinite(abs_ret).any() else 0.0
            z_alpha = float("nan")
            z_beta = float("nan")
            principle = "quantile_flat_band"
        else:
            z_alpha = float(NormalDist().inv_cdf(1.0 - flat_alpha / 2.0))
            z_beta = float(NormalDist().inv_cdf(flat_power))
            flat_band = float((z_alpha + z_beta) * max(sigma_ret, 1e-8))
            principle = "minimum_detectable_effect"
        y_bin = (ret > flat_band).astype(np.float32)
        neutral_weight = float(np.clip(float(os.environ.get("L1C_FLAT_BAND_WEIGHT", "0.5")), 0.1, 1.0))
        if weight_mode == "fixed":
            neutral_scale = np.where(abs_ret <= flat_band, neutral_weight, 1.0).astype(np.float32)
        else:
            snr = abs_ret / max(sigma_ret, 1e-8)
            neutral_scale = (1.0 - np.exp(-0.5 * np.square(snr))).astype(np.float32)
        windows_list.append(windows)
        end_list.append(idx_global)
        ybin_list.append(y_bin.astype(np.float32))
        ret_list.append(ret.astype(np.float32))
        neutral_scale_list.append(neutral_scale)
        flat_band_stats.append(
            {
                "flat_band": float(flat_band),
                "sigma_ret": float(sigma_ret),
                "median_ret": float(med_ret),
                "mad_ret": float(mad_ret),
                "alpha": float(flat_alpha),
                "power": float(flat_power),
                "z_alpha": float(z_alpha) if np.isfinite(z_alpha) else float("nan"),
                "z_beta": float(z_beta) if np.isfinite(z_beta) else float("nan"),
                "principle": principle,
            }
        )
    if not windows_list:
        z = np.zeros(0, dtype=np.float32)
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            z,
            z,
            z,
            np.zeros(0, dtype=np.int64),
        )
    windows = np.concatenate(windows_list, axis=0)
    y_bin = np.concatenate(ybin_list, axis=0).astype(np.float32)
    y_ret = np.concatenate(ret_list, axis=0).astype(np.float32)
    sw = _sample_weights_from_abs_return(np.abs(y_ret.astype(np.float64)))
    if neutral_scale_list:
        sw = sw * np.concatenate(neutral_scale_list, axis=0).astype(np.float32)
    if flat_band_stats:
        band_values = np.asarray([x["flat_band"] for x in flat_band_stats], dtype=np.float64)
        sigma_values = np.asarray([x["sigma_ret"] for x in flat_band_stats], dtype=np.float64)
        print(
            f"  [L1c] flat-band mode={flat_mode}  principle={(flat_band_stats[0].get('principle') or '')}  "
            f"alpha={flat_alpha:.4f}  power={flat_power:.2f}  "
            f"flat_band median={float(np.nanmedian(band_values)):.6f} p90={float(np.nanquantile(band_values, 0.90)):.6f}  "
            f"sigma_ret median={float(np.nanmedian(sigma_values)):.6f}  weight_mode={weight_mode}",
            flush=True,
        )
    return windows, y_bin, sw, y_ret, np.concatenate(end_list, axis=0)


def materialize_l1c_outputs(
    model: L1cDirectionModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> pd.DataFrame:
    work = df.copy(deep=False)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm_df = pd.concat(
        [work[["symbol", "time_key"]], pd.DataFrame(X, columns=feature_cols, index=work.index)],
        axis=1,
    )
    windows, end_idx = _l1c_build_symbol_windows(norm_df, feature_cols, seq_len)
    outputs = pd.DataFrame(
        {
            "symbol": work["symbol"].values,
            "time_key": pd.to_datetime(work["time_key"]),
        }
    )
    for col in l1c_output_columns():
        outputs[col] = np.float32(0.0)
    if len(end_idx) == 0:
        return outputs

    ds = TensorDataset(torch.from_numpy(windows.astype(np.float32, copy=False)))
    dl = DataLoader(ds, batch_size=1024, shuffle=False)
    logit_chunks: list[np.ndarray] = []
    strength_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            out = _l1c_forward_dict(model, xb)
            logit_chunks.append(out["direction_logits"].cpu().numpy().astype(np.float32))
            strength_chunks.append(torch.relu(out["direction_strength"]).cpu().numpy().astype(np.float32))
    logits = np.concatenate(logit_chunks, axis=0).ravel()
    strength = np.concatenate(strength_chunks, axis=0).ravel() if strength_chunks else np.zeros_like(logits)
    prob = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    score = (2.0 * prob - 1.0).astype(np.float32)
    conf = np.abs(score).astype(np.float32)
    outputs.loc[end_idx, "l1c_direction_prob"] = prob
    outputs.loc[end_idx, "l1c_direction_score"] = score
    outputs.loc[end_idx, "l1c_confidence"] = conf
    outputs.loc[end_idx, "l1c_direction_strength"] = strength.astype(np.float32)
    outputs.loc[end_idx, "l1c_is_warm"] = np.float32(1.0)
    return outputs


def load_l1c_direction_model() -> tuple[L1cDirectionModel, dict[str, Any]]:
    path_meta = os.path.join(MODEL_DIR, L1C_META_FILE)
    with open(path_meta, "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L1C_SCHEMA_VERSION:
        raise RuntimeError(
            f"L1c schema mismatch: artifact has {meta.get('schema_version')} but code expects {L1C_SCHEMA_VERSION}."
        )
    cfg_dict = meta.get("l1c_config") or {}
    cfg = L1cConfig(**{k: v for k, v in cfg_dict.items() if k in L1cConfig.__dataclass_fields__})
    cfg.input_dim = int(meta["input_dim"])
    cfg.seq_len = int(meta.get("seq_len", cfg.seq_len))
    model = L1cDirectionModel(cfg).to(DEVICE)
    state = torch.load(
        os.path.join(MODEL_DIR, meta.get("model_file", L1C_MODEL_FILE)),
        map_location=DEVICE,
    )
    model.load_state_dict(state)
    model.eval()
    return model, meta


def infer_l1c_direction(model: L1cDirectionModel, meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(meta["feature_cols"])
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    seq_len = int(meta.get("seq_len", 60))
    return materialize_l1c_outputs(
        model,
        df,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        device=DEVICE,
    )


def _train_one_epoch(
    model: L1cDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: L1cBinaryDirectionLoss,
    device: torch.device,
    *,
    seq_len: int,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        xb = batch[0].to(device)
        yb = batch[1].to(device)
        wb = batch[2].to(device)
        yr = batch[3].to(device)
        xb = _l1c_augment_batch(xb, training=True, seq_len=seq_len)
        optimizer.zero_grad()
        out = _l1c_forward_dict(model, xb)
        logits = out["direction_logits"]
        loss, _parts = criterion(logits, yb, wb)
        strength_tgt = torch.asinh(torch.abs(yr.view(-1)) * 100.0)
        strength_pred = torch.relu(out["direction_strength"].view(-1))
        strength_loss = F.smooth_l1_loss(strength_pred, strength_tgt, reduction="none")
        strength_loss = (strength_loss * wb.view(-1)).sum() / torch.clamp(wb.sum(), min=1e-6)
        loss = loss + float(getattr(model.config, "strength_aux_weight", 0.10)) * strength_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(os.environ.get("L1C_MAX_GRAD_NORM", "1.0")))
        optimizer.step()
        total += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def _eval_loss(
    model: L1cDirectionModel,
    loader: DataLoader,
    criterion: L1cBinaryDirectionLoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)
            wb = batch[2].to(device)
            yr = batch[3].to(device)
            out = _l1c_forward_dict(model, xb)
            logits = out["direction_logits"]
            loss, _ = criterion(logits, yb, wb)
            strength_tgt = torch.asinh(torch.abs(yr.view(-1)) * 100.0)
            strength_pred = torch.relu(out["direction_strength"].view(-1))
            strength_loss = F.smooth_l1_loss(strength_pred, strength_tgt, reduction="none")
            strength_loss = (strength_loss * wb.view(-1)).sum() / torch.clamp(wb.sum(), min=1e-6)
            loss = loss + float(getattr(model.config, "strength_aux_weight", 0.10)) * strength_loss
            total += float(loss.cpu()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def train_l1c_direction(df: pd.DataFrame, feat_cols: list[str]) -> None:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L1c] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    config = L1cConfig()
    for _env, _attr in (
        ("L1C_LAYER_DROP", "layer_drop"),
        ("L1C_WEIGHT_DECAY", "weight_decay"),
        ("L1C_NUM_HEADS", "num_heads"),
        ("L1C_NUM_LAYERS", "num_layers"),
        ("L1C_EMBED_DIM", "embed_dim"),
        ("L1C_FF_DIM", "ff_dim"),
        ("L1C_CONV_KERNEL", "conv_kernel_size"),
        ("L1C_CONV_HIDDEN", "conv_hidden_dim"),
        ("L1C_CONV_DROPOUT", "conv_dropout"),
        ("L1C_ATTN_DROPOUT", "attn_dropout"),
        ("L1C_FF_DROPOUT", "ff_dropout"),
        ("L1C_EMBED_DROPOUT", "embed_dropout"),
        ("L1C_LR", "lr"),
        ("L1C_PATIENCE", "patience"),
        ("L1C_MAX_EPOCHS", "max_epochs"),
        ("L1C_EARLY_STOP_MIN_DELTA", "early_stop_min_delta"),
        ("L1C_LABEL_SMOOTHING", "label_smoothing"),
        ("L1C_STRENGTH_AUX_WEIGHT", "strength_aux_weight"),
        ("L1C_COS_T0", "cosine_t0"),
        ("L1C_COS_T_MULT", "cosine_t_mult"),
    ):
        raw = os.environ.get(_env, "").strip()
        if raw and _attr in L1cConfig.__dataclass_fields__:
            field_type = L1cConfig.__dataclass_fields__[_attr].type
            if _attr in (
                "num_heads",
                "num_layers",
                "embed_dim",
                "ff_dim",
                "conv_kernel_size",
                "conv_hidden_dim",
                "patience",
                "max_epochs",
                "cosine_t0",
                "cosine_t_mult",
            ):
                setattr(config, _attr, int(float(raw)))
            else:
                setattr(config, _attr, float(raw))
    if FAST_TRAIN_MODE:
        config.max_epochs = min(config.max_epochs, 8)
        config.patience = min(config.patience, 2)
        config.batch_size = min(config.batch_size, 256)

    work = df.copy(deep=False)
    feature_cols = _select_l1c_feature_cols(work, feat_cols, config)
    splits = build_stack_time_splits(work["time_key"])
    Xn, mean, std = _normalize_l1c_matrix(work, feature_cols, splits.train_mask)
    seq_len = int(config.seq_len)
    horizon = int(config.predict_horizon)
    windows, y_bin, sw, y_ret, end_idx = _build_l1c_windows_labels(
        work,
        Xn,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizon=horizon,
    )
    if len(end_idx) == 0:
        raise RuntimeError("L1c: no windows with valid future horizon.")
    window_train = splits.train_mask[end_idx]
    window_val = splits.l2_val_mask[end_idx]
    if not window_train.any():
        raise RuntimeError("L1c: no training windows (end bar in train split).")
    if not window_val.any():
        raise RuntimeError("L1c: no validation windows (end bar in l2_val).")

    log_layer_banner("[L1c] Direction (causal Transformer, binary BCE)")
    print(
        f"  [L1c] artifact dir: {MODEL_DIR}\n"
        f"  [L1c] will write: {artifact_path(L1C_MODEL_FILE)} | {artifact_path(L1C_META_FILE)} | "
        f"{artifact_path(L1C_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    print(
        f"  [L1c] seq_len={seq_len} horizon={horizon} feats={len(feature_cols)} "
        f"train_windows={int(window_train.sum())} val_windows={int(window_val.sum())}",
        flush=True,
    )
    log_label_baseline("l1c_up_binary", y_bin[window_train], task="cls")
    _print_l1c_binary_label_stats(y_bin[window_train], y_ret[window_train], prefix="train windows")
    print(
        "  [L1c] target: 1[future_ret>0]; loss: weighted BCE (logits); "
        "sample_weight=rank(|ret|)/N (set L1C_SAMPLE_WEIGHT_MODE=uniform to disable).",
        flush=True,
    )

    X_t = torch.from_numpy(windows.astype(np.float32, copy=False))
    ds = TensorDataset(
        X_t,
        torch.from_numpy(y_bin),
        torch.from_numpy(sw),
        torch.from_numpy(y_ret),
    )
    train_ds = TensorDataset(*[t[window_train] for t in ds.tensors])
    val_ds = TensorDataset(*[t[window_val] for t in ds.tensors])
    pin = DEVICE.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=pin)

    criterion = L1cBinaryDirectionLoss(config)
    config.input_dim = len(feature_cols)
    model = L1cDirectionModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scheduler: CosineAnnealingWarmRestarts | None = None
    if int(config.cosine_t0) > 0:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, int(config.cosine_t0)),
            T_mult=max(1, int(config.cosine_t_mult)),
        )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    max_epochs = int(config.max_epochs)
    patience = int(config.patience)
    min_delta = float(config.early_stop_min_delta)
    print(
        f"  [L1c] train_config: batch_size={config.batch_size}  lr={float(config.lr):g}  "
        f"weight_decay={float(config.weight_decay):g}  max_epochs={max_epochs}  patience={patience}  "
        f"min_delta={min_delta:g}  label_smoothing={float(config.label_smoothing):.4f}",
        flush=True,
    )
    epoch_bar = trange(
        max_epochs,
        desc="[L1c] epochs",
        unit="ep",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    for _ep in epoch_bar:
        tr_loss = _train_one_epoch(model, train_dl, optimizer, criterion, DEVICE, seq_len=seq_len)
        if _ep == 0:
            _first_epoch_diagnostic(model, train_dl, DEVICE)
        va_loss = _eval_loss(model, val_dl, criterion, DEVICE)
        if scheduler is not None:
            scheduler.step()
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}", refresh=False)
        print(f"  [L1c] epoch={_ep + 1:02d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}", flush=True)
        if va_loss < (best_val - min_delta):
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("L1c: training failed to produce a checkpoint.")
    model.load_state_dict(best_state)

    val_metrics = evaluate_l1c(model, val_dl, DEVICE)
    print_l1c_eval_report(val_metrics)

    outputs = materialize_l1c_outputs(
        model,
        work,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        device=DEVICE,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, L1C_MODEL_FILE))
    cfg_dump = asdict(config)
    meta = {
        "schema_version": L1C_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "predict_horizon": horizon,
        "mean": mean,
        "std": std,
        "input_dim": len(feature_cols),
        "l1c_config": cfg_dump,
        "output_cols": l1c_output_columns(),
        "device": str(DEVICE),
        "model_file": L1C_MODEL_FILE,
        "output_cache_file": L1C_OUTPUT_CACHE_FILE,
        "direction_target_semantics": (
            f"per-symbol 1[(close[t+H]-close[t])/close[t] > flat_band] at H={horizon}; "
            f"flat_mode={(os.environ.get('L1C_FLAT_MODE', 'mde') or 'mde').strip().lower()} "
            f"(MDE from robust sigma when mode=mde); weighted BCE on logits; "
            f"label_smoothing={float(config.label_smoothing):.4f}"
        ),
        "direction_aux_semantics": "dual-branch model with local CNN branch; auxiliary head predicts asinh(|future_ret|*100)",
        "early_stopping": {
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": float(min_delta),
        },
        "val_metrics": val_metrics,
    }
    meta = attach_threshold_registry(
        meta,
        "l1c",
        [
            threshold_entry(
                "L1C_FLAT_MODE",
                str((os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()),
                category="adaptive_candidate",
                role="flat-band estimator mode for binary direction labels",
                adaptive_hint="mde default; quantile fallback for compatibility",
                statistical_principle="estimator_selection",
                method_selected=str((os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()),
            ),
            threshold_entry(
                "L1C_FLAT_ALPHA",
                float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20)),
                category="adaptive_candidate",
                role="type-I error control for MDE flat band",
                statistical_principle="two_sided_significance_level",
                alpha=float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20)),
            ),
            threshold_entry(
                "L1C_FLAT_POWER",
                float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99)),
                category="adaptive_candidate",
                role="target power for MDE flat band",
                statistical_principle="minimum_detectable_effect",
                power=float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99)),
            ),
            threshold_entry(
                "L1C_FLAT_WEIGHT_MODE",
                str((os.environ.get("L1C_FLAT_WEIGHT_MODE", "snr_gaussian") or "snr_gaussian").strip().lower()),
                category="adaptive_candidate",
                role="flat-region sample weighting rule",
                statistical_principle="snr_soft_weighting",
                method_selected=str((os.environ.get("L1C_FLAT_WEIGHT_MODE", "snr_gaussian") or "snr_gaussian").strip().lower()),
            ),
            threshold_entry(
                "L1C_CONF_MODE",
                str((os.environ.get("L1C_CONF_MODE", "cost_based") or "cost_based").strip().lower()),
                category="adaptive_candidate",
                role="confidence zone derivation mode",
                statistical_principle="cost_aware_decision_boundary",
                method_selected=str((os.environ.get("L1C_CONF_MODE", "cost_based") or "cost_based").strip().lower()),
            ),
            threshold_entry(
                "L1C_TX_COST_RATE",
                float(max(0.0, float(os.environ.get("L1C_TX_COST_RATE", "0.0005")))),
                category="safety_constraint",
                role="transaction cost floor for confidence threshold",
                statistical_principle="observed_execution_cost",
                cost_input=float(max(0.0, float(os.environ.get("L1C_TX_COST_RATE", "0.0005")))),
            ),
            threshold_entry(
                "L1C_PATIENCE",
                int(patience),
                category="data_guardrail",
                role="early-stop patience",
            ),
        ],
    )
    with open(os.path.join(MODEL_DIR, L1C_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1C_OUTPUT_CACHE_FILE)
    print(f"  [L1c] model saved -> {os.path.join(MODEL_DIR, L1C_MODEL_FILE)}", flush=True)
    print(f"  [L1c] meta saved  -> {os.path.join(MODEL_DIR, L1C_META_FILE)}", flush=True)
    print(f"  [L1c] cache saved -> {cache_path}", flush=True)
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L1c] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )

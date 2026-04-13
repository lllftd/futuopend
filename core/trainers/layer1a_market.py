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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange, tqdm

from core.tcn_pa_state import TemporalBlock
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
from core.trainers.val_metrics_extra import pearson_corr
from core.trainers.stack_v2_common import (
    build_stack_time_splits,
    compute_transition_risk_labels,
    log_label_baseline,
    save_output_cache,
)
from core.trainers.tcn_constants import DEVICE, SEQ_LEN


def _bounded_scalar_cols() -> list[str]:
    return ["l1a_transition_risk"]


def l1a_output_columns() -> list[str]:
    return (
        list(L1A_REGIME_COLS)
        + [
            "l1a_transition_risk",
            "l1a_vol_forecast",
        ]
        + [f"l1a_market_embed_{idx}" for idx in range(16)]
        + ["l1a_is_warm"]
    )


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
    close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    symbols = df["symbol"].to_numpy()
    state = pd.to_numeric(df["state_label"], errors="coerce").fillna(4).to_numpy(dtype=np.int64)

    transition_risk = compute_transition_risk_labels(state, symbols, horizon=min(10, max(horizon // 2, 5)))
    vol_forecast = np.zeros(len(df), dtype=np.float32)

    for _, grp in df.groupby("symbol", sort=False):
        idx = grp.index.to_numpy()
        close_g = close[idx]
        high_g = high[idx]
        low_g = low[idx]
        atr_g = safe_atr[idx]
        for loc, row_idx in enumerate(idx):
            end = min(len(idx), loc + 1 + horizon)
            if end <= loc + 1:
                continue
            future_close = close_g[loc + 1 : end]
            future_high = high_g[loc + 1 : end]
            future_low = low_g[loc + 1 : end]
            atr = max(float(atr_g[loc]), 1e-3)
            future_range = float(np.mean((future_high - future_low) / atr))
            vol_forecast[row_idx] = np.clip(future_range, 0.0, 5.0)

    return {
        "regime": state,
        "transition_risk": transition_risk.astype(np.float32),
        "vol_forecast": vol_forecast.astype(np.float32),
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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.activation == "identity":
            return logits
        return logits


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
    def __init__(self, input_dim: int, channels: list[int] | None = None, *, seq_len: int = SEQ_LEN):
        super().__init__()
        if channels is None:
            channels = [64, 64, 128]
        layers: list[nn.Module] = []
        for idx, out_ch in enumerate(channels):
            in_ch = input_dim if idx == 0 else channels[idx - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2**idx, dropout=0.15))
        self.backbone = nn.Sequential(*layers)
        self.shared_dim = channels[-1]
        self.seq_len = seq_len
        self.regime_head = TaskHead(self.shared_dim, 48, NUM_REGIME_CLASSES, activation="identity")
        self.transition_head = TaskHead(self.shared_dim, 24, 1, activation="identity")
        self.vol_head = TaskHead(self.shared_dim, 32, 1, activation="identity")
        self.embed_head = EmbedHead(self.shared_dim, 16)
        self.embed_decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, self.shared_dim),
        )

    def shared_repr(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x.transpose(1, 2))
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_repr(x)
        embed = self.embed_head(shared)
        return {
            "regime_logits": self.regime_head(shared),
            "transition_logit": self.transition_head(shared).squeeze(-1),
            "vol_value": self.vol_head(shared).squeeze(-1),
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
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    mse = nn.MSELoss()
    smooth_l1 = nn.SmoothL1Loss(beta=0.10)
    ce = nn.CrossEntropyLoss()
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
    for xb, y_regime, y_transition, y_vol in it:
        xb = xb.to(device)
        y_regime = y_regime.to(device)
        y_transition = y_transition.to(device)
        y_vol = y_vol.to(device)
        out = model(xb)
        transition_score = torch.sigmoid(out["transition_logit"])
        losses = {
            "regime": ce(out["regime_logits"], y_regime),
            "vol": mse(out["vol_value"], y_vol),
            "embed_recon": mse(out["embed_recon"], out["shared_repr"].detach()),
            "transition": smooth_l1(transition_score, y_transition),
        }
        loss = (
            0.45 * losses["regime"]
            + 0.25 * losses["vol"]
            + 0.10 * losses["embed_recon"]
            + 0.20 * losses["transition"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(xb)
        total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def _eval_epoch(model: L1AMarketTCN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    mse = nn.MSELoss()
    smooth_l1 = nn.SmoothL1Loss(beta=0.10)
    ce = nn.CrossEntropyLoss()
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
        for xb, y_regime, y_transition, y_vol in it:
            xb = xb.to(device)
            y_regime = y_regime.to(device)
            y_transition = y_transition.to(device)
            y_vol = y_vol.to(device)
            out = model(xb)
            transition_score = torch.sigmoid(out["transition_logit"])
            loss = (
                0.45 * ce(out["regime_logits"], y_regime)
                + 0.25 * mse(out["vol_value"], y_vol)
                + 0.10 * mse(out["embed_recon"], out["shared_repr"].detach())
                + 0.20 * smooth_l1(transition_score, y_transition)
            )
            total_loss += float(loss.item()) * len(xb)
            total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def _l1a_transition_val_block(
    label: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    y_true = np.asarray(y_true).ravel().astype(np.float64)
    y_score = np.asarray(y_score).ravel().astype(np.float64)
    mae = float(mean_absolute_error(y_true, y_score))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_score)))
    r2 = float(r2_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    corr = pearson_corr(y_true, y_score)
    print(
        f"  [L1a] val {label}:  n={len(y_true):,}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  corr(y,p)={corr:.4f}",
        flush=True,
    )
    print(
        f"    target_mean={float(np.mean(y_true)):.4f}  pred_mean={float(np.mean(y_score)):.4f}  "
        f"target_p95={float(np.percentile(y_true, 95)):.4f}  pred_p95={float(np.percentile(y_score, 95)):.4f}",
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


def _log_l1a_val_metrics(model: L1AMarketTCN, val_dl: DataLoader, device: torch.device) -> None:
    """Calibration-window: regime CM + per-class report; binary-head AUC/CM; vol MAE/R2; embed recon MSE."""
    model.eval()
    y_true_r: list[np.ndarray] = []
    y_pred_r: list[np.ndarray] = []
    vol_t: list[np.ndarray] = []
    vol_p: list[np.ndarray] = []
    tr_t, tr_s = [], []
    emb_mse: list[np.ndarray] = []
    with torch.no_grad():
        for xb, y_regime, y_transition, y_vol in val_dl:
            xb = xb.to(device)
            y_regime = y_regime.to(device)
            y_transition = y_transition.to(device)
            y_vol = y_vol.to(device)
            out = model(xb)
            y_true_r.append(y_regime.detach().cpu().numpy())
            y_pred_r.append(torch.argmax(out["regime_logits"], dim=1).detach().cpu().numpy())
            vol_t.append(y_vol.detach().cpu().numpy())
            vol_p.append(out["vol_value"].detach().cpu().numpy())
            tr_t.append(y_transition.detach().cpu().numpy())
            tr_s.append(torch.sigmoid(out["transition_logit"]).detach().cpu().numpy())
            emb_mse.append(
                F.mse_loss(out["embed_recon"], out["shared_repr"].detach(), reduction="none").mean(dim=1).detach().cpu().numpy()
            )
    yt = np.concatenate(y_true_r)
    yp = np.concatenate(y_pred_r)
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
    vt = np.concatenate(vol_t)
    vp = np.concatenate(vol_p)
    mae_v = float(mean_absolute_error(vt, vp))
    rmse_v = float(np.sqrt(mean_squared_error(vt, vp)))
    r2_v = float(r2_score(vt, vp)) if len(np.unique(vt)) > 1 else float("nan")
    corr_v = pearson_corr(vt, vp)
    emb_mean = float(np.mean(np.concatenate(emb_mse)))

    print("\n  [L1a] ========== val (cal) effectiveness report ==========", flush=True)
    print(
        f"  [L1a] regime vs market_state  n={len(yt):,}  acc={acc:.4f}  macro-F1={f1_macro:.4f}  "
        f"weighted-F1={f1_weighted:.4f}  cohen_kappa={kappa:.4f}",
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

    _l1a_transition_val_block("transition_risk", np.concatenate(tr_t), np.concatenate(tr_s))
    print("  [L1a] ========== end val report ==========\n", flush=True)


def materialize_l1a_outputs(
    model: L1AMarketTCN,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    windows, end_idx = _build_symbol_windows(pd.concat([df[["symbol", "time_key"]], pd.DataFrame(X, columns=feature_cols)], axis=1), feature_cols, seq_len)
    outputs = pd.DataFrame(
        {
            "symbol": df["symbol"].values,
            "time_key": pd.to_datetime(df["time_key"]),
        }
    )
    for col in l1a_output_columns():
        outputs[col] = 0.0
    outputs[L1A_REGIME_COLS] = 1.0 / float(NUM_REGIME_CLASSES)
    outputs[_bounded_scalar_cols()] = 0.0
    outputs["l1a_vol_forecast"] = float(np.nanmedian(pd.to_numeric(df["lbl_atr"], errors="coerce").fillna(1.0)))
    outputs["l1a_is_warm"] = 0.0
    if len(end_idx) == 0:
        return outputs

    ds = TensorDataset(torch.from_numpy(windows))
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=min(4, _lgbm_n_jobs()))
    dl_it = dl
    if _lgb_round_tqdm_enabled():
        dl_it = tqdm(dl, desc="[L1a] materialize outputs", file=TQDM_FILE, mininterval=0.3, unit="batch")
    regime_rows: list[np.ndarray] = []
    scalar_rows: dict[str, list[np.ndarray]] = {k: [] for k in ["transition", "vol"]}
    embeds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl_it:
            xb = xb.to(device)
            out = model(xb)
            regime_rows.append(torch.softmax(out["regime_logits"], dim=1).cpu().numpy())
            scalar_rows["transition"].append(torch.sigmoid(out["transition_logit"]).cpu().numpy())
            scalar_rows["vol"].append(out["vol_value"].cpu().numpy())
            embeds.append(out["market_embed"].cpu().numpy())
    regime = np.concatenate(regime_rows, axis=0)
    outputs.loc[end_idx, L1A_REGIME_COLS] = regime
    outputs.loc[end_idx, "l1a_transition_risk"] = np.concatenate(scalar_rows["transition"], axis=0)
    outputs.loc[end_idx, "l1a_vol_forecast"] = np.clip(np.concatenate(scalar_rows["vol"], axis=0), 0.0, 5.0)
    embed_mat = np.concatenate(embeds, axis=0)
    for idx in range(embed_mat.shape[1]):
        outputs.loc[end_idx, f"l1a_market_embed_{idx}"] = embed_mat[:, idx]
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
    window_val = splits.cal_mask[end_idx]
    if not window_val.any():
        raise RuntimeError("L1a: calibration window is empty for validation.")

    X_t = torch.from_numpy(windows.astype(np.float32, copy=False))
    ds = TensorDataset(
        X_t,
        torch.from_numpy(targets["regime"][end_idx].astype(np.int64)),
        torch.from_numpy(targets["transition_risk"][end_idx].astype(np.float32)),
        torch.from_numpy(targets["vol_forecast"][end_idx].astype(np.float32)),
    )
    train_ds = TensorDataset(*[tensor[window_train] for tensor in ds.tensors])
    val_ds = TensorDataset(*[tensor[window_val] for tensor in ds.tensors])
    batch_size = 512 if FAST_TRAIN_MODE else 1024
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    log_layer_banner("[L1a] Sequence Market Encoder (TCN)")
    log_time_key_arrays(
        "L1a",
        work.iloc[end_idx[window_train]]["time_key"],
        work.iloc[end_idx[window_val]]["time_key"],
        train_label="window train (end_idx in train split)",
        val_label="window val (end_idx in cal split)",
        extra_note="Stacks: train_mask=time<TRAIN_END, cal=TRAIN_END..CAL_END on window end-bars.",
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
    out_cn = l1a_output_columns()
    print(f"  [L1a] output column count: {len(out_cn)} (expect 25)", flush=True)
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
    log_label_baseline("l1a_transition_risk", targets["transition_risk"][end_idx][window_train], task="reg")
    log_label_baseline("l1a_vol_forecast", targets["vol_forecast"][end_idx][window_train], task="reg")

    model = L1AMarketTCN(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
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
    for epoch in epoch_bar:
        tr_loss = _train_epoch(model, train_dl, optimizer, DEVICE)
        va_loss = _eval_epoch(model, val_dl, DEVICE)
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
    _log_l1a_val_metrics(model, val_dl, DEVICE)

    outputs = materialize_l1a_outputs(model, work, feature_cols, mean=mean, std=std, seq_len=SEQ_LEN, device=DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, L1A_MODEL_FILE))
    meta = {
        "schema_version": L1A_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "mean": mean,
        "std": std,
        "output_cols": l1a_output_columns(),
        "device": str(DEVICE),
        "model_file": L1A_MODEL_FILE,
        "output_cache_file": L1A_OUTPUT_CACHE_FILE,
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
    feature_cols = list(meta["feature_cols"])
    model = L1AMarketTCN(len(feature_cols), seq_len=int(meta.get("seq_len", SEQ_LEN))).to(DEVICE)
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
    feature_cols = list(meta["feature_cols"])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    seq_len = int(meta.get("seq_len", SEQ_LEN))
    return materialize_l1a_outputs(model, df, feature_cols, mean=mean, std=std, seq_len=seq_len, device=DEVICE)

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm

from core.pa_rules import add_pa_features
from core.tcn_pa_state import FocalLoss
from core.mamba_pa_state import PAStateMamba

from core.trainers.constants import TCN_REGIME_FUT_PROB_COLS
from core.trainers.tcn_constants import *
from core.trainers.tcn_utils import _tq, _tqdm_disabled

# Reusing the TCN labels for Layer 1 transition
MAMBA_HEAD_NUM_CLASSES = len(TCN_REGIME_FUT_PROB_COLS)
MAMBA_HEAD_TARGET_NAMES = ["same", "transition"]

MAMBA_STATE_CLASSIFIER_FILE = "mamba_state_classifier_6c.pt"


def _resolve_mamba_device() -> torch.device:
    device = DEVICE
    if device.type == "mps":
        use_mps = os.environ.get("MAMBA_USE_MPS", "").strip().lower()
        if use_mps not in {"1", "true", "yes"}:
            print("  [Warning] Mamba on MPS often leads to NaN. Falling back to CPU. (Set MAMBA_USE_MPS=1 to force MPS)")
            return torch.device("cpu")
    return device


def _train_mamba_model(
    X_seq_cpu: Union[np.memmap, torch.Tensor],
    y: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    n_features: int,
    *,
    desc_str: str,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    show_model_summary: bool = False,
    _retry_cpu_after_mps: bool = False,
):
    dev = device if device is not None else _resolve_mamba_device()
    me = TCN_MAX_EPOCHS if max_epochs is None else int(max_epochs)
    pat = TCN_ES_PATIENCE if patience is None else int(patience)

    print(
        f"\n  Training {desc_str}  (max_epochs={me}, early_stop_patience={pat}, device={dev})",
        flush=True,
    )

    if isinstance(X_seq_cpu, torch.Tensor):
        X_t = X_seq_cpu
    else:
        print("  Materializing memmap → tensor (slow on first use only)…", flush=True)
        t0 = time.perf_counter()
        X_t = torch.from_numpy(np.ascontiguousarray(np.array(X_seq_cpu)))
        print(
            f"  Loaded shape={tuple(X_t.shape)} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    base_ds = TensorDataset(X_t, y_t)
    train_ds = Subset(base_ds, train_idx)
    cal_ds = Subset(base_ds, cal_idx)

    nw = int(os.environ.get("TORCH_NUM_WORKERS", "4")) if sys.platform != "win32" else 0
    pin = (dev.type in ("cuda", "mps"))

    train_dl = DataLoader(
        train_ds, batch_size=4096, shuffle=True, drop_last=True, 
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0)
    )
    cal_dl = DataLoader(
        cal_ds, batch_size=4096, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0)
    )

    noise_std = float(os.environ.get("MAMBA_INPUT_NOISE", "0.02"))
    model = PAStateMamba(
        input_size=n_features,
        d_model=64,
        n_layers=4,
        dropout=TCN_DROPOUT,
        bottleneck_dim=TCN_BOTTLENECK_DIM,
        num_classes=MAMBA_HEAD_NUM_CLASSES,
        noise_std=noise_std,
    ).to(dev)

    if show_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}  (target ~70K)")
        print(f"  Architecture: d_model=64, n_layers=4, dropout={TCN_DROPOUT}")

    y_tr = y[train_idx]
    class_counts = np.bincount(y_tr, minlength=MAMBA_HEAD_NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    class_freq = class_counts / class_counts.sum()
    class_weights = (1.0 / np.maximum(class_freq * MAMBA_HEAD_NUM_CLASSES, 1e-6)) ** TCN_CE_WEIGHT_POWER
    class_weights /= class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(dev)

    focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0.0"))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.10"))
    
    if focal_gamma > 0.0:
        criterion_train = FocalLoss(alpha=class_weights_t, gamma=focal_gamma)
        if show_model_summary:
            print(f"  Loss: FocalLoss(gamma={focal_gamma}) + class_weights")
    else:
        criterion_train = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
        if show_model_summary:
            print(f"  Loss: CrossEntropyLoss(label_smoothing={label_smoothing}) + class_weights")

    criterion_eval = nn.CrossEntropyLoss(weight=class_weights_t)

    lr_env = os.environ.get("MAMBA_LR", "")
    if lr_env:
        lr = float(lr_env)
    else:
        # Mamba SSM + residuals diverge at TCN-like 8e-4; keep defaults conservative.
        lr = 1e-4 if dev.type == "mps" else 2e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

    best_cal_loss = float("inf")
    best_state = None
    patience_counter = 0
    log_epochs = _tqdm_disabled()

    batch_log = os.environ.get("MAMBA_TRAIN_BATCH_LOG", "").strip().lower()
    if batch_log in {"", "auto"}:
        log_train_batches = dev.type == "cpu"
    else:
        log_train_batches = batch_log in ("1", "true", "yes")
    n_train_batches = len(train_dl)
    batch_log_every = max(1, min(50, n_train_batches // 10 or 1))

    for epoch in _tq(range(me), desc=f"Epochs {desc_str}", unit="ep", leave=False, file=sys.stderr):
        model.train()
        train_loss = 0.0
        n_batches = 0
        if epoch == 0 and log_train_batches:
            print(
                f"    Epoch 1/{me}: {n_train_batches} train batches — "
                "first batch on CPU can take many minutes (SSM time loop); not stuck.",
                flush=True,
            )
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion_train(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            li = loss.item()
            if not np.isfinite(li):
                print(
                    f"  Non-finite loss at epoch {epoch + 1} batch {n_batches + 1}/{n_train_batches} — stopping. "
                    f"Try MAMBA_LR=1e-4 or MAMBA_INPUT_NOISE=0",
                    flush=True,
                )
                train_loss = float("nan")
                break
            train_loss += li
            if log_train_batches and (n_batches % batch_log_every == 0 or n_batches == 0):
                print(
                    f"    … {desc_str}  epoch {epoch + 1}/{me}  "
                    f"train batch {n_batches + 1}/{n_train_batches}",
                    flush=True,
                )
            n_batches += 1
        train_loss /= max(n_batches, 1)

        if not np.isfinite(train_loss):
            print(
                "  Non-finite train loss — training diverged. Try MAMBA_LR=1e-4, MAMBA_INPUT_NOISE=0, or smaller batch (patch).",
                flush=True,
            )
            break

        model.eval()
        cal_loss = 0.0
        cal_correct = cal_total = 0
        with torch.no_grad():
            for xb, yb in cal_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                cal_loss += criterion_eval(logits, yb).item() * len(yb)
                cal_correct += (logits.argmax(1) == yb).sum().item()
                cal_total += len(yb)
        cal_loss /= max(cal_total, 1)

        scheduler.step()

        if not np.isfinite(cal_loss):
            print(
                "  Non-finite calibration loss ( logits or weights hit NaN/Inf on this device ). "
                "Common on Apple MPS for this SSM. Try lower MAMBA_LR, or training will retry on CPU if enabled.",
                flush=True,
            )
            break

        if cal_loss < best_cal_loss:
            best_cal_loss = cal_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= pat:
                break

        if log_epochs:
            print(
                f"    epoch {epoch + 1}/{me}  train_loss={train_loss:.4f}  cal_loss={cal_loss:.4f}  "
                f"best_cal={best_cal_loss:.4f}  es={patience_counter}/{pat}",
                flush=True,
            )

    if best_state is None:
        fb = os.environ.get("MAMBA_CPU_FALLBACK", "1").strip().lower()
        cpu_fallback = fb not in ("0", "false", "no")
        if dev.type == "mps" and cpu_fallback and not _retry_cpu_after_mps:
            print(
                "  MPS run did not produce a valid checkpoint; retrying on CPU "
                "(disable: MAMBA_CPU_FALLBACK=0).",
                flush=True,
            )
            return _train_mamba_model(
                X_seq_cpu,
                y,
                train_idx,
                cal_idx,
                n_features,
                desc_str=f"{desc_str} [CPU after MPS non-finite loss]",
                device=torch.device("cpu"),
                max_epochs=max_epochs,
                patience=patience,
                show_model_summary=show_model_summary,
                _retry_cpu_after_mps=True,
            )
        raise RuntimeError(
            "Training failed (best_state is None): no finite train/cal loss before stop. "
            "Try MAMBA_LR=1e-4, MAMBA_INPUT_NOISE=0, omit MAMBA_USE_MPS if retrying CPU after MPS, "
            "or MAMBA_CPU_FALLBACK=1 after MPS NaN."
        )

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_mamba(
    X_mm: np.memmap,
    y: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    n_features: int,
    ts: np.ndarray,
    syms: np.ndarray,
):
    dev = _resolve_mamba_device()
    n_folds = TCN_OOF_FOLDS
    fold_size = len(train_idx) // n_folds

    print("\n" + "=" * 70, flush=True)
    print(
        f"  Training Mamba — {TCN_OOF_FOLDS}-fold OOF "
        f"(future transition signal, +15 bars)",
        flush=True,
    )
    print("=" * 70, flush=True)

    oof_embeds = np.zeros((len(train_idx), TCN_BOTTLENECK_DIM), dtype=np.float32)
    oof_regime_probs = np.zeros((len(train_idx), MAMBA_HEAD_NUM_CLASSES), dtype=np.float32)

    print("  Loading memmap into RAM once (shared by all folds)…", flush=True)
    t_load = time.perf_counter()
    X_tseq = torch.from_numpy(np.ascontiguousarray(np.array(X_mm)))
    print(
        f"  Dense tensor ready shape={tuple(X_tseq.shape)} in {time.perf_counter() - t_load:.1f}s. "
        f"CPU Mamba is slow per epoch; epoch bar stays at 0% until the first epoch finishes.",
        flush=True,
    )
    for fold in _tq(range(n_folds), desc="OOF CV folds", unit="fold", dynamic_ncols=True, file=sys.stderr):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_idx)

        val_fold_idx = train_idx[start_idx:end_idx]
        train_fold_idx = np.concatenate([train_idx[:start_idx], train_idx[end_idx:]])

        fold_model = _train_mamba_model(
            X_tseq,
            y,
            train_fold_idx,
            val_fold_idx,
            n_features,
            desc_str=f"Mamba OOF Fold {fold + 1}/{n_folds}",
            device=dev,
            show_model_summary=False,
        )
        fold_infer_dev = next(fold_model.parameters()).device

        val_ds = TensorDataset(X_tseq[val_fold_idx])
        val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False)

        fold_embs, fold_rp = [], []
        with torch.inference_mode():
            for (xb,) in val_dl:
                xb = xb.to(fold_infer_dev)
                r_log, emb = fold_model.forward_with_embedding(xb)
                fold_rp.append(torch.softmax(r_log, dim=1).cpu().numpy())
                fold_embs.append(emb.cpu().numpy())

        oof_embeds[start_idx:end_idx] = np.concatenate(fold_embs, axis=0)
        oof_regime_probs[start_idx:end_idx] = np.concatenate(fold_rp, axis=0)

    # Save OOF
    os.makedirs(MODEL_DIR, exist_ok=True)
    oof_cache = {
        "train_idx": train_idx,
        "ts": ts[train_idx],
        "syms": syms[train_idx],
        "embeds": oof_embeds,
        "regime_probs": oof_regime_probs,
    }
    with open(os.path.join(MODEL_DIR, "mamba_oof_cache.pkl"), "wb") as f:
        pickle.dump(oof_cache, f)
    print(f"\n  Saved OOF cache -> {MODEL_DIR}/mamba_oof_cache.pkl")

    # 2. Final Model Training (reuse same CPU tensor — no second memmap copy)
    final_model = _train_mamba_model(
        X_tseq,
        y,
        train_idx,
        cal_idx,
        n_features,
        desc_str="Final Mamba Model",
        device=dev,
        show_model_summary=True,
    )
    final_infer_dev = next(final_model.parameters()).device

    # Evaluate on test set
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    test_ds = TensorDataset(X_tseq, y_t)
    test_ds_sub = Subset(test_ds, test_idx)
    test_dl = DataLoader(test_ds_sub, batch_size=4096, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(final_infer_dev), yb.to(final_infer_dev)
            logits = final_model(xb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy — future transition (Mamba binary): {acc:.4f}")
    print("\n  Classification Report — future transition (+15 bars):")
    print(
        classification_report(
            y_true, y_pred,
            labels=list(range(MAMBA_HEAD_NUM_CLASSES)),
            target_names=MAMBA_HEAD_TARGET_NAMES,
            digits=4, zero_division=0,
        )
    )

    return final_model

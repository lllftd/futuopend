from __future__ import annotations

import gc
import os
import pickle
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

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


def _train_mamba_model(
    X_mm: np.memmap,
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
):
    dev = device if device is not None else DEVICE
    me = TCN_MAX_EPOCHS if max_epochs is None else int(max_epochs)
    pat = TCN_ES_PATIENCE if patience is None else int(patience)

    print(
        f"\n  Training {desc_str}  (max_epochs={me}, early_stop_patience={pat}, device={dev})",
        flush=True,
    )

    X_t = torch.from_numpy(np.array(X_mm))
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    base_ds = TensorDataset(X_t, y_t)
    train_ds = Subset(base_ds, train_idx)
    cal_ds = Subset(base_ds, cal_idx)

    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True, drop_last=True)
    cal_dl = DataLoader(cal_ds, batch_size=4096, shuffle=False)

    model = PAStateMamba(
        input_size=n_features,
        d_model=64,
        n_layers=4,
        dropout=TCN_DROPOUT,
        bottleneck_dim=TCN_BOTTLENECK_DIM,
        num_classes=MAMBA_HEAD_NUM_CLASSES,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

    best_cal_loss = float("inf")
    best_state = None
    patience_counter = 0
    log_epochs = _tqdm_disabled()

    for epoch in _tq(range(me), desc=f"Epochs {desc_str}", unit="ep", leave=False, file=sys.stderr):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion_train(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

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

        if np.isnan(cal_loss):
            print("  Model parameters became NaN! Training diverged.")
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
        raise RuntimeError("Training failed (best_state is None).")

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

    print("  Loading memmap into RAM for fast Mamba training…", flush=True)
    X_tseq = torch.from_numpy(np.array(X_mm))
    for fold in _tq(range(n_folds), desc="OOF CV folds", unit="fold", dynamic_ncols=True, file=sys.stderr):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_idx)

        val_fold_idx = train_idx[start_idx:end_idx]
        train_fold_idx = np.concatenate([train_idx[:start_idx], train_idx[end_idx:]])

        fold_model = _train_mamba_model(
            X_mm,
            y,
            train_fold_idx,
            val_fold_idx,
            n_features,
            desc_str=f"Mamba OOF Fold {fold + 1}/{n_folds}",
            show_model_summary=False,
        )

        val_ds = TensorDataset(X_tseq[val_fold_idx])
        val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False)

        fold_embs, fold_rp = [], []
        with torch.inference_mode():
            for (xb,) in val_dl:
                xb = xb.to(DEVICE)
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

    X_t = torch.from_numpy(np.array(X_mm))

    # 2. Final Model Training
    final_model = _train_mamba_model(
        X_mm,
        y,
        train_idx,
        cal_idx,
        n_features,
        desc_str="Final Mamba Model",
        show_model_summary=True,
    )

    # Evaluate on test set
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    test_ds = TensorDataset(X_t, y_t)
    test_ds_sub = Subset(test_ds, test_idx)
    test_dl = DataLoader(test_ds_sub, batch_size=4096, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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

from __future__ import annotations

import gc
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, TensorDataset

from core.models.tcn_pa_state import FocalLoss, PAStateTCN

from core.training.common.constants import TCN_REGIME_FUT_PROB_COLS
from core.training.common.lgbm_utils import configure_cuda_training_speedups
from core.training.tcn.tcn_constants import *
from core.training.tcn.tcn_utils import _tq, _tqdm_disabled

# Layer 1 label is Triple Barrier (Up, Down, TimeStop).
TCN_HEAD_NUM_CLASSES = len(TCN_REGIME_FUT_PROB_COLS)
TCN_HEAD_TARGET_NAMES = ["Bull_Hit", "Bear_Hit", "Chop_Timeout"]
TCN_READOUT_TYPE = os.environ.get("TCN_READOUT_TYPE", "attention").strip().lower() or "attention"
TCN_MIN_ATTENTION_SEQ_LEN = max(1, int(os.environ.get("TCN_MIN_ATTENTION_SEQ_LEN", "4")))


def _tcn_train_batch_size(dev: torch.device) -> int:
    raw = os.environ.get("TCN_TRAIN_BATCH_SIZE", "").strip()
    if raw:
        return max(8, int(raw))
    return 8192 if dev.type == "cuda" else 4096


def _tcn_dataloader_workers(dev: torch.device) -> int:
    raw = os.environ.get("TCN_DATALOADER_WORKERS", "").strip()
    if raw:
        return max(0, int(raw))
    if sys.platform == "win32":
        return min(8, max(2, (os.cpu_count() or 4) // 2)) if dev.type == "cuda" else 0
    return max(0, int(os.environ.get("TORCH_NUM_WORKERS", "4")))


def _materialize_tcn_tensor(X_mm: np.memmap) -> torch.Tensor:
    keep_in_ram = os.environ.get("TCN_DATASET_IN_RAM", "1").strip().lower() not in {"0", "false", "no"}
    if keep_in_ram:
        print("  Materializing TCN dataset in RAM once for all folds…", flush=True)
        x_np = np.array(X_mm, copy=True)
    else:
        print("  Using memmap-backed TCN dataset tensor…", flush=True)
        x_np = np.asarray(X_mm)
    return torch.from_numpy(x_np)


def _train_tcn_model(
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    y_np: np.ndarray,
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
    configure_cuda_training_speedups()

    base_ds = TensorDataset(X_t, y_t)
    train_ds = Subset(base_ds, train_idx)
    cal_ds = Subset(base_ds, cal_idx)

    nw = _tcn_dataloader_workers(dev)
    bs = _tcn_train_batch_size(dev)
    pin = dev.type in ("cuda", "mps")
    if show_model_summary:
        print(f"  TCN DataLoader: batch_size={bs}  num_workers={nw}  (TCN_TRAIN_BATCH_SIZE / TCN_DATALOADER_WORKERS)", flush=True)

    _dl_extras: dict = {}
    if nw > 0:
        _dl_extras["persistent_workers"] = True
        _dl_extras["prefetch_factor"] = max(2, int(os.environ.get("TCN_PREFETCH_FACTOR", "4")))
    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin,
        **_dl_extras,
    )
    cal_dl = DataLoader(
        cal_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        **_dl_extras,
    )

    model = PAStateTCN(
        input_size=n_features,
        num_channels=SLIM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
        bottleneck_dim=TCN_BOTTLENECK_DIM,
        num_classes=TCN_HEAD_NUM_CLASSES,
        readout_type=TCN_READOUT_TYPE,
        min_attention_seq_len=TCN_MIN_ATTENTION_SEQ_LEN,
    ).to(dev)

    if show_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}  (SLIM_CHANNELS={SLIM_CHANNELS})")
        print(
            f"  Architecture: channels={SLIM_CHANNELS}, kernel={TCN_KERNEL_SIZE}, "
            f"dropout={TCN_DROPOUT}, readout={TCN_READOUT_TYPE}, "
            f"min_attn_seq_len={TCN_MIN_ATTENTION_SEQ_LEN}"
        )

    y_tr = y_np[train_idx]
    class_counts = np.bincount(y_tr, minlength=TCN_HEAD_NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    class_freq = class_counts / class_counts.sum()
    class_weights = (1.0 / np.maximum(class_freq * TCN_HEAD_NUM_CLASSES, 1e-6)) ** TCN_CE_WEIGHT_POWER
    class_weights /= class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(dev)

    focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0.0"))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", "0.02"))
    
    if focal_gamma > 0.0:
        # Note: focal loss implementation doesn't currently use label smoothing directly in our custom class, 
        # but for clean baseline comparison, we ensure baseline CE uses it.
        criterion_train = FocalLoss(alpha=class_weights_t, gamma=focal_gamma)
        if show_model_summary:
            print(f"  Loss: FocalLoss(gamma={focal_gamma}) + class_weights")
    else:
        criterion_train = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
        if show_model_summary:
            print(f"  Loss: CrossEntropyLoss(label_smoothing={label_smoothing}) + class_weights")

    criterion_eval = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    # Changed from WarmRestarts to standard Cosine to prevent violent loss spikes triggering Early Stopping
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=me, eta_min=1e-5)

    use_cuda_amp = dev.type == "cuda" and os.environ.get("TCN_AMP", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if use_cuda_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler: GradScaler | None = None
    elif use_cuda_amp:
        amp_dtype = torch.float16
        scaler = GradScaler("cuda")
    else:
        amp_dtype = torch.float32
        scaler = None
    if show_model_summary and use_cuda_amp:
        print(f"  TCN AMP: dtype={amp_dtype}  (TCN_AMP=0 to disable)", flush=True)

    best_cal_loss = float("inf")
    best_state = None
    patience_counter = 0
    log_epochs = _tqdm_disabled()

    for epoch in _tq(range(me), desc=f"Epochs {desc_str}", unit="ep", leave=False, file=sys.stderr):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb = xb.to(dev, non_blocking=pin)
            yb = yb.to(dev, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            if use_cuda_amp:
                with autocast("cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = criterion_train(logits, yb)
            else:
                logits = model(xb)
                loss = criterion_train(logits, yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
                xb = xb.to(dev, non_blocking=pin)
                yb = yb.to(dev, non_blocking=pin)
                if use_cuda_amp:
                    with autocast("cuda", dtype=amp_dtype):
                        logits = model(xb)
                else:
                    logits = model(xb)
                logits_f = logits.float()
                cal_loss += criterion_eval(logits_f, yb).item() * len(yb)
                cal_correct += (logits_f.argmax(1) == yb).sum().item()
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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def train_tcn(
    X_mm: np.memmap,
    y: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    n_features: int,
    ts: np.ndarray,
    syms: np.ndarray,
):
    X_t = _materialize_tcn_tensor(X_mm)
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    skip_oof = _env_truthy("TCN_SKIP_OOF")
    oof_path = os.path.join(MODEL_DIR, "tcn_oof_cache.pkl")

    print("\n" + "=" * 70, flush=True)
    if skip_oof:
        print(
            "  Training TCN — OOF skipped (TCN_SKIP_OOF=1); final model + test eval only",
            flush=True,
        )
        if not os.path.isfile(oof_path):
            raise FileNotFoundError(
                f"TCN_SKIP_OOF=1 requires an existing OOF cache at {oof_path}. "
                "Run a full TCN train once without TCN_SKIP_OOF, then retry."
            )
        print(f"  Keeping on-disk OOF cache (not rewritten): {oof_path}", flush=True)
    else:
        print(
            f"  Training TCN — {TCN_OOF_FOLDS}-fold OOF "
            f"(future triple barrier hit, +15 bars, 1.0 ATR — matches tcn_data_prep.prepare_data)",
            flush=True,
        )
    print("=" * 70, flush=True)
    print(
        f"  Scheme A: OOF and final use the same max_epochs={TCN_MAX_EPOCHS}, "
        f"patience={TCN_ES_PATIENCE} (override: TCN_MAX_EPOCHS / TCN_ES_PATIENCE)",
        flush=True,
    )
    if not skip_oof:
        print(
            f"  OOF folds={TCN_OOF_FOLDS} (env TCN_OOF_FOLDS) — running sequentially in-process",
            flush=True,
        )

    if not skip_oof:
        n_folds = TCN_OOF_FOLDS
        fold_size = len(train_idx) // n_folds

        oof_embeds = np.zeros((len(train_idx), TCN_BOTTLENECK_DIM), dtype=np.float32)
        oof_regime_probs = np.zeros((len(train_idx), TCN_HEAD_NUM_CLASSES), dtype=np.float32)

        for fold in _tq(range(n_folds), desc="OOF CV folds", unit="fold", dynamic_ncols=True, file=sys.stderr):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_idx)

            val_fold_idx = train_idx[start_idx:end_idx]
            train_fold_idx = np.concatenate([train_idx[:start_idx], train_idx[end_idx:]])

            fold_model = _train_tcn_model(
                X_t,
                y_t,
                y,
                train_fold_idx,
                val_fold_idx,
                n_features,
                desc_str=f"TCN OOF Fold {fold + 1}/{n_folds}",
                show_model_summary=False,
            )

            val_ds = TensorDataset(X_t[val_fold_idx])
            val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False)

            fold_embs, fold_rp = [], []
            with torch.inference_mode():
                for (xb,) in val_dl:
                    xb = xb.to(DEVICE, non_blocking=(DEVICE.type in ("cuda", "mps")))
                    r_log, emb = fold_model.forward_with_embedding(xb)
                    fold_rp.append(torch.softmax(r_log, dim=1).cpu().numpy())
                    fold_embs.append(emb.cpu().numpy())

            oof_embeds[start_idx:end_idx] = np.concatenate(fold_embs, axis=0)
            oof_regime_probs[start_idx:end_idx] = np.concatenate(fold_rp, axis=0)

        os.makedirs(MODEL_DIR, exist_ok=True)
        oof_cache = {
            "train_idx": train_idx,
            "ts": ts[train_idx],
            "syms": syms[train_idx],
            "embeds": oof_embeds,
            "regime_probs": oof_regime_probs,
        }
        with open(oof_path, "wb") as f:
            pickle.dump(oof_cache, f)
        print(f"\n  Saved OOF cache -> {oof_path}")

        gc.collect()

    # 2. Final Model Training (same max_epochs / patience as OOF — Scheme A)
    final_model = _train_tcn_model(
        X_t,
        y_t,
        y,
        train_idx,
        cal_idx,
        n_features,
        desc_str="Final TCN Model",
        show_model_summary=True,
    )

    # Evaluate on test set
    test_ds = TensorDataset(X_t, y_t)
    test_ds_sub = Subset(test_ds, test_idx)
    test_dl = DataLoader(test_ds_sub, batch_size=4096, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type in ("cuda", "mps")))
            yb = yb.to(DEVICE, non_blocking=(DEVICE.type in ("cuda", "mps")))
            logits = final_model(xb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy — future triple barrier (3-class): {acc:.4f}")
    print("\n  Classification Report — future barrier (+15 bars, 1.0 ATR):")
    print(
        classification_report(
            y_true, y_pred,
            labels=list(range(TCN_HEAD_NUM_CLASSES)),
            target_names=TCN_HEAD_TARGET_NAMES,
            digits=4, zero_division=0,
        )
    )

    return final_model



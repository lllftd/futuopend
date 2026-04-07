"""
TCN (Temporal Convolutional Network) for future regime prediction.

Captures sequential patterns that LightGBM's per-bar view cannot see.
Uses same strict OOS splits as LightGBM training.

Architecture: causal dilated convolutions, single 6-class head (same label space as CSV market_state).
Target at end of each window: market_state shifted -15 bars within each symbol (future regime).
Input: sliding window of 1-min PA feature vectors (see SEQ_LEN).
Output: softmax regime futures + learned bottleneck embeddings (tcn_emb_*); LGBM consumes OOF-isolated TCN features as tcn_regime_fut_* + tcn_emb_0..K-1.

Training budget (Scheme A): TCN_MAX_EPOCHS + TCN_ES_PATIENCE apply to both OOF folds and final fit.
Parallel OOF: TCN_OOF_FOLDS (default 3) × TCN_OOF_WORKERS subprocesses per batch (see script printout).
  MPS / single-GPU: OOF children default to CPU to avoid many processes sharing one accelerator.

Logging:
    From repo root, ``scripts/run_train_tcn.sh`` overwrites ``train_tcn.log`` each run (sets ``FORCE_TQDM=1`` for tee).
    Manual tee: ``FORCE_TQDM=1 python3 -m backtests.train_tcn_pa_state 2>&1 | tee train_tcn.log``
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time as _time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN, FocalLoss

warnings.filterwarnings("ignore")


def _tqdm_disabled() -> bool:
    """Align with tqdm policy: DISABLE_TQDM=1 off; non-TTY off unless FORCE_TQDM=1."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return True
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return True
    return False


def _tq(it, **kwargs):
    """Iterator progress bar (epochs, etc.)."""
    return tqdm(it, disable=_tqdm_disabled(), **kwargs)


def _pbar(**kwargs):
    """Manual tqdm (e.g. subprocess folds completed). Caller should use ``with`` / ``.update()``."""
    return tqdm(disable=_tqdm_disabled(), **kwargs)


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lgbm_models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")

# ~1M seq × 60 × 78 × fp32 ≈ 19 GiB — keep on disk; stream batches from memmap.
TCN_MEMMAP_CHUNK = int(os.environ.get("TCN_MEMMAP_CHUNK", "1024"))
TCN_STATS_CHUNK = int(os.environ.get("TCN_STATS_CHUNK", "512"))

TRAIN_END = "2023-01-01"
CAL_END = "2023-07-01"
TEST_END = "2025-01-01"

SEQ_LEN = 60
# Stronger regularization / wider temporal receptive field for noisy 1-min inputs
TCN_KERNEL_SIZE = 5
TCN_DROPOUT = 0.375
SLIM_CHANNELS = [16, 16, 16]
# Learned bottleneck (replaces PCA of raw TCN channel vector). Override: TCN_BOTTLENECK_DIM=8.
TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))

# After inverse-frequency CE weights, shrink toward 1 (default sqrt) so 1-min minority
# classes are not wildly over-weighted. Set TCN_CE_WEIGHT_POWER=1.0 to restore legacy.
TCN_CE_WEIGHT_POWER = float(os.environ.get("TCN_CE_WEIGHT_POWER", "0.5"))

# Scheme A (recommended): same training budget for OOF folds and final model.
TCN_MAX_EPOCHS = max(1, int(os.environ.get("TCN_MAX_EPOCHS", "90")))
TCN_ES_PATIENCE = max(1, int(os.environ.get("TCN_ES_PATIENCE", "20")))

# OOF cross-validation fold count (train index only). Override: TCN_OOF_FOLDS=3.
TCN_OOF_FOLDS = max(2, int(os.environ.get("TCN_OOF_FOLDS", "3")))
# Concurrent OOF folds: each fold runs in a fresh Python subprocess (macOS-safe).
# Worker cap matches fold count. On MPS/single-GPU, children often default to CPU.
TCN_OOF_WORKERS = max(1, min(TCN_OOF_FOLDS, int(os.environ.get("TCN_OOF_WORKERS", "1"))))

TCN_OOF_SHARED_PKL = "tcn_oof_shared.pkl"
TCN_OOF_Y_NPY = "tcn_oof_y.npy"
TCN_OOF_TRAIN_IDX_NPY = "tcn_oof_train_idx.npy"

STATE_NAMES = {
    0: "bull_conv",
    1: "bull_div",
    2: "bear_conv",
    3: "bear_div",
    4: "range_conv",
    5: "range_div",
}
NUM_REGIME_CLASSES = 6
STATE_CLASSIFIER_FILE = "state_classifier_6c.txt"


def _pick_tcn_train_device() -> torch.device:
    """CUDA > MPS > CPU; set TORCH_DEVICE=cuda:0 / mps / cpu to force."""
    forced = os.environ.get("TORCH_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_tcn_train_device()


# ───────────────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────────────

def _load_and_compute_pa(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw.sort_values("time_key").reset_index(drop=True)
    atr_1m = compute_atr(raw, length=14)
    print(f"  [{symbol}] Computing PA features on {len(raw):,} bars…")
    t0 = _time.time()
    df = add_pa_features(raw, atr_1m, timeframe="5min")
    print(f"  [{symbol}] Done in {_time.time()-t0:.1f}s → {df.shape[1]} cols")
    return df


def _load_labels(symbol: str) -> pd.DataFrame:
    cols = ["time_key", "market_state", "signal", "outcome", "quality_bull_breakout", "quality_bear_breakout"]
    lbl = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}{LABELED_SUFFIX}.csv"), usecols=cols)
    lbl["time_key"] = pd.to_datetime(lbl["time_key"])
    return lbl


def _tcn_merge_pa_labels(symbol: str) -> pd.DataFrame:
    """Picklable worker for parallel symbol prep (see DATA_PREPARE_WORKERS)."""
    pa = _load_and_compute_pa(symbol)
    lbl = _load_labels(symbol)
    merged = pd.merge(pa, lbl, on="time_key", how="inner")
    merged["symbol"] = symbol
    print(f"  [{symbol}] Merged {len(merged):,} rows", flush=True)
    return merged


def _pa_feature_cols(df: pd.DataFrame) -> list[str]:
    """Select only continuous PA features for TCN (ignore discrete/boolean/targets)."""
    exclude = {"object", "category", "bool"}
    
    # Exclude non-continuous/non-boolean absolute targets
    exclude_substrings = [
        "count", "state", "signal", "outcome", 
        "max_favorable", "max_adverse", "exit_bar", "reward_risk",
        "stop_", "nearest_", "target", "retrace", "round_number",
        "double_top", "double_bottom", "wedge", "overshoot", "channel_reversal",
        "head_shoulders", "or_", "pa_struct_break", "pa_hmm_", "prev_day"
    ]
    
    continuous_cols = []
    bool_cols = []
    for c in df.columns:
        if not c.startswith("pa_"):
            continue
        if str(df[c].dtype) in exclude:
            continue
            
        if any(sub in c for sub in exclude_substrings):
            continue
            
        if c.startswith(("pa_is_", "pa_breakout_")) and df[c].nunique(dropna=True) <= 2:
            bool_cols.append(c)
        else:
            continuous_cols.append(c)

    for c in bool_cols:
        df[c] = df[c].astype(np.float32)
        
    return sorted(continuous_cols + bool_cols), sorted(bool_cols)


def _count_sequences(df_1m: pd.DataFrame, seq_len: int) -> int:
    """Count valid within-day windows without materializing feature tensors."""
    n = 0
    for _, grp in df_1m.groupby("symbol"):
        grp = grp.sort_values("time_key").reset_index(drop=True)
        if len(grp) < seq_len:
            continue
        timestamps = grp["time_key"].values
        dates = pd.to_datetime(timestamps).date
        for start in range(len(grp) - seq_len + 1):
            end = start + seq_len
            if dates[start] != dates[end - 1]:
                continue
            n += 1
    return n


def _fill_sequences_memmap(
    mm: np.memmap,
    df_1m: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write all sequences into mm; return (y, time_key, symbol) per row."""
    n = mm.shape[0]
    y_out = np.empty(n, dtype=np.int64)
    ts_out = np.empty(n, dtype="datetime64[ns]")
    sym_out = np.empty(n, dtype=object)
    row = 0
    for sym, grp in _tq(
        df_1m.groupby("symbol"),
        desc="TCN memmap sequences",
        unit="sym",
        total=int(df_1m["symbol"].nunique()),
    ):
        grp = grp.sort_values("time_key").reset_index(drop=True)
        if len(grp) < seq_len:
            continue
        feats = grp[feat_cols].values.astype(np.float32)
        labels = grp["state_label"].values.astype(np.int64)
        timestamps = grp["time_key"].values
        dates = pd.to_datetime(timestamps).date
        for start in range(len(grp) - seq_len + 1):
            end = start + seq_len
            if dates[start] != dates[end - 1]:
                continue
            mm[row] = feats[start:end]
            y_out[row] = labels[end - 1]
            ts_out[row] = pd.Timestamp(timestamps[end - 1]).asm8
            sym_out[row] = sym
            row += 1
    if row != n:
        raise RuntimeError(f"Sequence count mismatch: expected {n}, wrote {row}")
    mm.flush()
    return y_out, ts_out, sym_out


def _zscore_stats_train_memmap(
    mm: np.memmap,
    train_idx: np.ndarray,
    n_feat: int,
    bool_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sum_x = np.zeros(n_feat, dtype=np.float64)
    cnt_x = np.zeros(n_feat, dtype=np.float64)
    ch = TCN_STATS_CHUNK
    n_stat = (len(train_idx) + ch - 1) // ch
    for sl in _tq(
        range(0, len(train_idx), ch),
        desc="TCN z-score (mean)",
        total=n_stat,
        unit="chunk",
        leave=False,
    ):
        idx = train_idx[sl : sl + ch]
        block = np.asarray(mm[idx], dtype=np.float64)
        flat = block.reshape(-1, n_feat)
        fin = np.isfinite(flat)
        sum_x += np.sum(np.where(fin, flat, 0.0), axis=0)
        cnt_x += np.sum(fin, axis=0)
    mean = sum_x / np.maximum(cnt_x, 1.0)
    sum_sq = np.zeros(n_feat, dtype=np.float64)
    for sl in _tq(
        range(0, len(train_idx), ch),
        desc="TCN z-score (var)",
        total=n_stat,
        unit="chunk",
        leave=False,
    ):
        idx = train_idx[sl : sl + ch]
        block = np.asarray(mm[idx], dtype=np.float64)
        flat = block.reshape(-1, n_feat)
        fin = np.isfinite(flat)
        d = flat - mean
        sum_sq += np.sum(np.where(fin, d * d, 0.0), axis=0)
    std = np.sqrt(sum_sq / np.maximum(cnt_x - 1.0, 1.0))
    std = np.where(std < 1e-8, 1.0, std)
    
    if bool_indices:
        for idx in bool_indices:
            mean[idx] = 0.0
            std[idx] = 1.0
            
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_memmap_inplace(mm: np.memmap, mean: np.ndarray, std: np.ndarray) -> None:
    n = mm.shape[0]
    mean64 = mean.astype(np.float64)
    std64 = std.astype(np.float64)
    ch = TCN_MEMMAP_CHUNK
    n_norm = (n + ch - 1) // ch
    for sl in _tq(range(0, n, ch), desc="TCN normalize", total=n_norm, unit="chunk"):
        su = min(ch, n - sl)
        block = np.asarray(mm[sl : sl + su])
        b64 = block.astype(np.float64)
        b64 = (b64 - mean64) / std64
        b64 = np.nan_to_num(b64, nan=0.0)
        mm[sl : sl + su] = b64.astype(np.float32)
    mm.flush()


def prepare_data():
    symbols = ["QQQ", "SPY"]
    n_sym = len(symbols)
    _dw = os.environ.get("DATA_PREPARE_WORKERS", "").strip()
    default_w = min(n_sym, 2)
    prep_workers = int(_dw) if _dw else default_w
    prep_workers = max(1, min(prep_workers, n_sym))

    if prep_workers <= 1:
        parts = []
        for sym in _tq(symbols, desc="TCN prepare_data symbols", unit="sym"):
            parts.append(_tcn_merge_pa_labels(sym))
    else:
        print(
            f"  Parallel symbol load: {prep_workers} processes (DATA_PREPARE_WORKERS)",
            flush=True,
        )
        print(
            "  Note: each symbol ~3–5min PA features on full history; progress advances when any symbol finishes.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=prep_workers) as ex:
            futs = {ex.submit(_tcn_merge_pa_labels, sym): sym for sym in symbols}
            parts_by_sym: dict[str, pd.DataFrame] = {}
            for fut in _tq(
                as_completed(futs),
                total=n_sym,
                desc="TCN prepare_data symbols",
                unit="sym",
            ):
                sym = futs[fut]
                parts_by_sym[sym] = fut.result()
            parts = [parts_by_sym[s] for s in symbols]

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["symbol", "time_key"]).reset_index(drop=True)
    # Future regime (15 bars ahead), per symbol — same 6-class space as CSV market_state.
    ms = pd.to_numeric(df["market_state"], errors="coerce")
    df["state_label"] = (
        ms.groupby(df["symbol"]).transform(lambda s: s.shift(-15)).fillna(4).astype(int)
    )

    feat_cols, bool_cols = _pa_feature_cols(df)
    print(f"  PA features: {len(feat_cols) - len(bool_cols)} continuous + {len(bool_cols)} boolean = {len(feat_cols)} total")

    # Create sequences (with safe fallback if window too long for intraday bars)
    seq_len_used = SEQ_LEN
    tried: list[int] = [SEQ_LEN]
    n_feat = len(feat_cols)
    print(f"  Creating sequences (window={SEQ_LEN})…", flush=True)

    def _build_memmap_for_seq_len(seq_len: int) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, str]:
        n_seq = _count_sequences(df, seq_len)
        if n_seq == 0:
            return None, None, None, None, ""  # type: ignore[return-value]
        nbytes = n_seq * seq_len * n_feat * 4
        mmap_dir = os.environ.get("TCN_MEMMAP_DIR", DATA_DIR)
        os.makedirs(mmap_dir, exist_ok=True)
        mmap_path = os.path.join(mmap_dir, f".tcn_X_seq_{seq_len}_{os.getpid()}.dat")
        print(
            f"  Memmap → {mmap_path}  shape=({n_seq:,},{seq_len},{n_feat})  "
            f"~{nbytes / (1024**3):.2f} GiB on disk",
            flush=True,
        )
        mm = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(n_seq, seq_len, n_feat))
        print(f"  Filling sequences…", flush=True)
        y_a, ts_a, sym_a = _fill_sequences_memmap(mm, df, feat_cols, seq_len)
        return mm, y_a, ts_a, sym_a, mmap_path

    mm, y, ts, syms, mmap_path = _build_memmap_for_seq_len(seq_len_used)
    if mm is None or mm.shape[0] == 0:
        for alt in [30, 20]:
            if alt in tried:
                continue
            tried.append(alt)
            print(f"  No sequences with window={seq_len_used}. Retrying with window={alt}…", flush=True)
            mm, y, ts, syms, mmap_path = _build_memmap_for_seq_len(alt)
            if mm is not None and mm.shape[0] > 0:
                seq_len_used = alt
                break
    if mm is None or len(y) == 0:
        raise RuntimeError(
            "No valid TCN sequences generated. Check sequence length/day-boundary constraints."
        )

    print(f"  Total sequences: {len(y):,} (window_used={seq_len_used})", flush=True)

    ts_dt = pd.to_datetime(ts)
    train_mask = ts_dt < pd.Timestamp(TRAIN_END)
    cal_mask = (ts_dt >= pd.Timestamp(TRAIN_END)) & (ts_dt < pd.Timestamp(CAL_END))
    test_mask = (ts_dt >= pd.Timestamp(CAL_END)) & (ts_dt < pd.Timestamp(TEST_END))

    train_idx = np.flatnonzero(train_mask)
    cal_idx = np.flatnonzero(cal_mask)
    test_idx = np.flatnonzero(test_mask)

    print(
        f"  Train: {len(train_idx):,}  |  Cal: {len(cal_idx):,}  |  Test: {len(test_idx):,}",
        flush=True,
    )

    print("  Z-score stats (train rows, chunked)…", flush=True)
    bool_indices = {i for i, c in enumerate(feat_cols) if c in bool_cols}
    feat_mean, feat_std = _zscore_stats_train_memmap(mm, train_idx, n_feat, bool_indices=bool_indices)
    print("  Normalizing memmap in-place (chunked)…", flush=True)
    _normalize_memmap_inplace(mm, feat_mean, feat_std)
    print("  Normalization done.", flush=True)

    norm_stats = {
        "mean": feat_mean,
        "std": feat_std,
        "feat_cols": feat_cols,
        "seq_len_used": seq_len_used,
        "memmap_path": mmap_path,
        "ts": ts,
        "syms": syms,
    }
    return mm, y, train_idx, cal_idx, test_idx, norm_stats


# ───────────────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────────────

def _train_tcn_model(
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

    X_t = torch.from_numpy(X_mm)
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    base_ds = TensorDataset(X_t, y_t)
    train_ds = Subset(base_ds, train_idx)
    cal_ds = Subset(base_ds, cal_idx)

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
    cal_dl = DataLoader(cal_ds, batch_size=1024, shuffle=False)

    model = PAStateTCN(
        input_size=n_features,
        num_channels=SLIM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
        bottleneck_dim=TCN_BOTTLENECK_DIM,
        num_classes=NUM_REGIME_CLASSES,
    ).to(dev)

    if show_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}  (target ~70K)")
        print(f"  Architecture: channels={SLIM_CHANNELS}, kernel={TCN_KERNEL_SIZE}, dropout={TCN_DROPOUT}")

    y_tr = y[train_idx]
    class_counts = np.bincount(y_tr, minlength=NUM_REGIME_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    class_freq = class_counts / class_counts.sum()
    class_weights = (1.0 / np.maximum(class_freq * NUM_REGIME_CLASSES, 1e-6)) ** TCN_CE_WEIGHT_POWER
    class_weights /= class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(dev)

    focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0.0"))
    if focal_gamma > 0.0:
        criterion_train = FocalLoss(alpha=class_weights_t, gamma=focal_gamma)
        if show_model_summary:
            print(f"  Loss: FocalLoss(gamma={focal_gamma}) + class_weights")
    else:
        criterion_train = nn.CrossEntropyLoss(weight=class_weights_t)
        if show_model_summary:
            print("  Loss: CrossEntropyLoss(no_smoothing) + class_weights")

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


def _write_oof_shared(
    mmap_path: str,
    mm_shape: tuple[int, ...],
    y: np.ndarray,
    train_idx: np.ndarray,
    n_folds: int,
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    y_path = os.path.join(MODEL_DIR, TCN_OOF_Y_NPY)
    idx_path = os.path.join(MODEL_DIR, TCN_OOF_TRAIN_IDX_NPY)
    np.save(y_path, y)
    np.save(idx_path, train_idx)
    shared = {
        "mmap_path": mmap_path,
        "mm_shape": tuple(int(x) for x in mm_shape),
        "y_path": y_path,
        "train_idx_path": idx_path,
        "n_folds": int(n_folds),
        "n_features": int(mm_shape[2]),
        "max_epochs": TCN_MAX_EPOCHS,
        "patience": TCN_ES_PATIENCE,
    }
    with open(os.path.join(MODEL_DIR, TCN_OOF_SHARED_PKL), "wb") as f:
        pickle.dump(shared, f)


def run_oof_fold_child(fold_id: int) -> None:
    """One OOF fold in a fresh interpreter (env TCN_INTERNAL_OOF_FOLD + TCN_INTERNAL_OOF_DEVICE)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    shared_path = os.path.join(MODEL_DIR, TCN_OOF_SHARED_PKL)
    with open(shared_path, "rb") as f:
        shared = pickle.load(f)
    y = np.load(shared["y_path"])
    train_idx = np.load(shared["train_idx_path"])
    mmap_path = str(shared["mmap_path"])
    mm_shape = tuple(shared["mm_shape"])
    X_mm = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=mm_shape)
    n_folds = int(shared["n_folds"])
    n_features = int(shared["n_features"])
    fold_size = len(train_idx) // n_folds
    start_idx = fold_id * fold_size
    end_idx = (fold_id + 1) * fold_size if fold_id < n_folds - 1 else len(train_idx)
    val_fold_idx = train_idx[start_idx:end_idx]
    train_fold_idx = np.concatenate([train_idx[:start_idx], train_idx[end_idx:]])

    dev_str = os.environ.get("TCN_INTERNAL_OOF_DEVICE", "cpu").strip() or "cpu"
    dev = torch.device(dev_str)

    desc_str = f"TCN OOF Fold {fold_id + 1}/{n_folds} [subprocess]"
    model = _train_tcn_model(
        X_mm,
        y,
        train_fold_idx,
        val_fold_idx,
        n_features,
        desc_str=desc_str,
        device=dev,
        max_epochs=int(shared["max_epochs"]),
        patience=int(shared["patience"]),
        show_model_summary=False,
    )

    X_t = torch.from_numpy(X_mm)
    val_ds = TensorDataset(X_t[val_fold_idx])
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False)
    fold_embs, fold_rp = [], []
    with torch.inference_mode():
        for (xb,) in val_dl:
            xb = xb.to(dev)
            r_log, emb = model.forward_with_embedding(xb)
            fold_rp.append(torch.softmax(r_log, dim=1).cpu().numpy())
            fold_embs.append(emb.cpu().numpy())
    embs = np.concatenate(fold_embs, axis=0)
    r_prob = np.concatenate(fold_rp, axis=0)
    out_npz = os.path.join(MODEL_DIR, f"tcn_oof_part_{fold_id}.npz")
    np.savez_compressed(
        out_npz,
        start_idx=start_idx,
        end_idx=end_idx,
        embs=embs,
        regime_probs=r_prob,
    )
    print(f"  [OOF child] fold {fold_id} -> {out_npz}", flush=True)


def _oof_device_for_child_slot(
    batch_index: int,
    slot_in_batch: int,
    batch_len: int,
) -> str:
    """Pick device string for one OOF subprocess. Avoid piling many procs on one GPU."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if DEVICE.type == "mps":
        return "cpu"
    if DEVICE.type == "cuda" and n_gpus > 0:
        if batch_len <= n_gpus:
            return f"cuda:{(batch_index + slot_in_batch) % n_gpus}"
        return "cpu"
    return "cpu"


def _run_oof_subprocesses(repo_root: str, script_path: str, n_folds: int, max_parallel: int) -> None:
    fold_ids = list(range(n_folds))
    i = 0
    print(
        f"  OOF parallelism: TCN_OOF_WORKERS={max_parallel} "
        f"(subprocess / fold; parent device={DEVICE})",
        flush=True,
    )
    with _pbar(
        total=n_folds,
        desc="OOF CV folds",
        unit="fold",
        dynamic_ncols=True,
        file=sys.stderr,
    ) as fold_pbar:
        while i < n_folds:
            batch = fold_ids[i : i + max_parallel]
            procs: list[tuple[int, subprocess.Popen]] = []
            for slot, fold_id in enumerate(batch):
                dev_assign = _oof_device_for_child_slot(i, slot, len(batch))
                env = os.environ.copy()
                env["TCN_INTERNAL_OOF_FOLD"] = str(fold_id)
                env["TCN_INTERNAL_OOF_DEVICE"] = dev_assign
                env["TCN_OOF_WORKERS"] = "1"
                print(f"    spawn fold {fold_id} on {dev_assign}", flush=True)
                procs.append(
                    (
                        fold_id,
                        subprocess.Popen(
                            [sys.executable, script_path],
                            env=env,
                            cwd=repo_root,
                        ),
                    )
                )
            fut_to_fold: dict = {}
            with ThreadPoolExecutor(max_workers=len(procs)) as ex:
                for fold_id, proc in procs:
                    fut_to_fold[ex.submit(proc.wait)] = fold_id
                for fut in as_completed(fut_to_fold):
                    fold_done = fut_to_fold[fut]
                    rc = fut.result()
                    if rc != 0:
                        raise RuntimeError(
                            f"OOF fold {fold_done} subprocess failed: return code={rc}"
                        )
                    fold_pbar.update(1)
            i += max_parallel


def _merge_oof_parts(n_folds: int, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    emb_dim = TCN_BOTTLENECK_DIM
    oof_embeds = np.zeros((len(train_idx), emb_dim), dtype=np.float32)
    oof_regime_probs = np.zeros((len(train_idx), NUM_REGIME_CLASSES), dtype=np.float32)
    for fold_id in range(n_folds):
        part_path = os.path.join(MODEL_DIR, f"tcn_oof_part_{fold_id}.npz")
        part = np.load(part_path)
        s, e = int(part["start_idx"]), int(part["end_idx"])
        oof_embeds[s:e] = part["embs"]
        oof_regime_probs[s:e] = part["regime_probs"]
    return oof_embeds, oof_regime_probs


def _cleanup_oof_artifacts(n_folds: int) -> None:
    for fold_id in range(n_folds):
        p = os.path.join(MODEL_DIR, f"tcn_oof_part_{fold_id}.npz")
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
    for name in (TCN_OOF_SHARED_PKL, TCN_OOF_Y_NPY, TCN_OOF_TRAIN_IDX_NPY):
        p = os.path.join(MODEL_DIR, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

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
    n_folds = TCN_OOF_FOLDS
    fold_size = len(train_idx) // n_folds
    oof_workers = min(TCN_OOF_WORKERS, n_folds)

    print("\n" + "=" * 70, flush=True)
    print(
        f"  Training TCN — {TCN_OOF_FOLDS}-fold OOF "
        f"(future 6-class regime, +15 bars)",
        flush=True,
    )
    print("=" * 70, flush=True)
    print(
        f"  Scheme A: OOF and final use the same max_epochs={TCN_MAX_EPOCHS}, "
        f"patience={TCN_ES_PATIENCE} (override: TCN_MAX_EPOCHS / TCN_ES_PATIENCE)",
        flush=True,
    )
    print(
        f"  OOF folds={TCN_OOF_FOLDS} (env TCN_OOF_FOLDS), "
        f"parallel workers={oof_workers} (cap env TCN_OOF_WORKERS)",
        flush=True,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.abspath(__file__)
    _fn = getattr(X_mm, "filename", None)
    if _fn is not None:
        mmap_path_str = _fn.decode("utf-8") if isinstance(_fn, (bytes, bytearray)) else str(_fn)
    else:
        mmap_path_str = ""

    if oof_workers > 1:
        if not mmap_path_str or not os.path.isfile(mmap_path_str):
            raise RuntimeError(
                "Parallel OOF requires a file-backed memmap path (X_mm.filename). "
                "Cannot run TCN_OOF_WORKERS>1 on in-memory array."
            )
        _cleanup_oof_artifacts(n_folds)
        _write_oof_shared(mmap_path_str, tuple(X_mm.shape), y, train_idx, n_folds)
        _run_oof_subprocesses(repo_root, script_path, n_folds, oof_workers)
        oof_embeds, oof_regime_probs = _merge_oof_parts(n_folds, train_idx)
        _cleanup_oof_artifacts(n_folds)
    else:
        oof_embeds = np.zeros((len(train_idx), TCN_BOTTLENECK_DIM), dtype=np.float32)
        oof_regime_probs = np.zeros((len(train_idx), NUM_REGIME_CLASSES), dtype=np.float32)

        print("  Building torch view + starting in-process OOF (first line may take ~1–3 min on MPS)…", flush=True)
        X_tseq = torch.from_numpy(X_mm)
        for fold in _tq(range(n_folds), desc="OOF CV folds", unit="fold", dynamic_ncols=True, file=sys.stderr):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_idx)

            val_fold_idx = train_idx[start_idx:end_idx]
            train_fold_idx = np.concatenate([train_idx[:start_idx], train_idx[end_idx:]])

            fold_model = _train_tcn_model(
                X_mm,
                y,
                train_fold_idx,
                val_fold_idx,
                n_features,
                desc_str=f"TCN OOF Fold {fold + 1}/{n_folds}",
                show_model_summary=False,
            )

            val_ds = TensorDataset(X_tseq[val_fold_idx])
            val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False)

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
    with open(os.path.join(MODEL_DIR, "tcn_oof_cache.pkl"), "wb") as f:
        pickle.dump(oof_cache, f)
    print(f"\n  Saved OOF cache -> {MODEL_DIR}/tcn_oof_cache.pkl")

    X_t = torch.from_numpy(X_mm)

    # 2. Final Model Training (same max_epochs / patience as OOF — Scheme A)
    final_model = _train_tcn_model(
        X_mm,
        y,
        train_idx,
        cal_idx,
        n_features,
        desc_str="Final TCN Model",
        show_model_summary=True,
    )

    # Evaluate on test set
    y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))
    test_ds = TensorDataset(X_t, y_t)
    test_ds_sub = Subset(test_ds, test_idx)
    test_dl = DataLoader(test_ds_sub, batch_size=1024, shuffle=False)

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
    print(f"\n  Test Accuracy — future regime (6-class): {acc:.4f}")
    target_names = [STATE_NAMES[i] for i in range(NUM_REGIME_CLASSES)]
    print("\n  Classification Report — future market_state (+15 bars):")
    print(
        classification_report(
            y_true, y_pred,
            labels=list(range(NUM_REGIME_CLASSES)),
            target_names=target_names,
            digits=4, zero_division=0,
        )
    )

    return final_model


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    _oof_only = os.environ.get("TCN_INTERNAL_OOF_FOLD")
    if _oof_only is not None:
        run_oof_fold_child(int(_oof_only.strip()))
        return

    print("=" * 70)
    print("  TCN — Future Regime (6-class, +15 bars)")
    print("=" * 70)
    print(f"  Device: {DEVICE}  (override: TORCH_DEVICE=cuda:0 | mps | cpu)")

    mm: np.memmap | None = None
    mmap_path = ""
    try:
        mm, y, train_idx, cal_idx, test_idx, norm_stats = prepare_data()
        mmap_path = str(norm_stats.get("memmap_path", ""))
        n_features = int(mm.shape[2])
        print("  Starting optimization (batches stream from memmap)…", flush=True)

        tcn_model = train_tcn(
            mm, y, train_idx, cal_idx, test_idx, n_features,
            norm_stats["ts"], norm_stats["syms"]
        )

        # Save
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(tcn_model.state_dict(), os.path.join(MODEL_DIR, "tcn_state_classifier.pt"))

        meta = {
            "mean": norm_stats["mean"],
            "std": norm_stats["std"],
            "feat_cols": norm_stats["feat_cols"],
            "seq_len": int(norm_stats.get("seq_len_used", SEQ_LEN)),
            "input_size": n_features,
            "num_channels": list(SLIM_CHANNELS),
            "kernel_size": TCN_KERNEL_SIZE,
            "dropout": TCN_DROPOUT,
            "bottleneck_dim": TCN_BOTTLENECK_DIM,
            "is_dual_head": False,
            "tcn_head": "regime6_future_15_bottleneck",
            "num_regime_classes": NUM_REGIME_CLASSES,
            "state_names": STATE_NAMES,
        }
        with open(os.path.join(MODEL_DIR, "tcn_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"\n  TCN 6-class future-regime model saved → {MODEL_DIR}/tcn_state_classifier.pt")
        print(f"  Meta saved → {MODEL_DIR}/tcn_meta.pkl")
        print("\n" + "=" * 70)
        print("  DONE")
        print("=" * 70)
    finally:
        if mm is not None:
            try:
                mm._mmap.close()
            except Exception:
                pass
            del mm
        if (
            mmap_path
            and os.path.isfile(mmap_path)
            and not os.environ.get("TCN_KEEP_MEMMAP")
        ):
            try:
                os.remove(mmap_path)
                print(f"  Removed temp memmap {mmap_path}", flush=True)
            except OSError as exc:
                print(f"  Warning: could not remove memmap {mmap_path}: {exc}", flush=True)


if __name__ == "__main__":
    main()

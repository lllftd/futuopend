from __future__ import annotations

import gc
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from core.pa_rules import add_pa_features
from core.tcn_pa_state import FocalLoss, PAStateTCN

from core.trainers.tcn_constants import *
from core.trainers.tcn_utils import *

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
        if len(grp) < seq_len:
            continue
        grp = grp.sort_values("time_key").reset_index(drop=True)
        dates = pd.to_datetime(grp["time_key"].values).date
        
        start_dates = dates[:-seq_len + 1]
        end_dates = dates[seq_len - 1:]
        valid_mask = start_dates == end_dates
        n += valid_mask.sum()
    return n


def _fill_sequences_memmap(
    mm: np.memmap,
    df_1m: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write all sequences into mm; return (y, time_key, symbol) per row."""
    from numpy.lib.stride_tricks import sliding_window_view
    
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
        if len(grp) < seq_len:
            continue
        grp = grp.sort_values("time_key").reset_index(drop=True)
        
        feats = grp[feat_cols].values.astype(np.float32)
        labels = grp["state_label"].values.astype(np.int64)
        timestamps = grp["time_key"].values
        dates = pd.to_datetime(timestamps).date
        
        # Vectorized sliding window
        windows = sliding_window_view(feats, window_shape=(seq_len, feats.shape[1])).squeeze(axis=1)
        
        # Valid day boundary mask
        start_dates = dates[:-seq_len + 1]
        end_dates = dates[seq_len - 1:]
        valid_mask = start_dates == end_dates
        
        valid_windows = windows[valid_mask]
        n_valid = len(valid_windows)
        
        if n_valid > 0:
            mm[row : row + n_valid] = valid_windows
            y_out[row : row + n_valid] = labels[seq_len - 1:][valid_mask]
            ts_out[row : row + n_valid] = timestamps[seq_len - 1:][valid_mask].astype("datetime64[ns]")
            sym_out[row : row + n_valid] = sym
            row += n_valid
            
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
    # Future regime transition (15 bars ahead).
    ms = pd.to_numeric(df["market_state"], errors="coerce").fillna(4).astype(int)
    fut_ms = (
        ms.groupby(df["symbol"]).transform(lambda s: s.shift(-15)).fillna(4).astype(int)
    )
    df["state_label"] = (ms != fut_ms).astype(int)

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



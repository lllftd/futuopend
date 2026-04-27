from __future__ import annotations

import gc
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from core.models.tcn_pa_state import FocalLoss, PAStateTCN
from core.training.common.constants import pa_feature_cache_timeframe_key
from core.training.prep.pa_feature_cache import load_or_build_pa_features

from core.training.tcn.tcn_constants import *
from core.training.tcn.tcn_utils import _tq  # not included in `import *` (leading underscore)


def _prep_cpu_workers() -> int:
    """CPU pool size for TCN prep (barrier + memmap norm). Override: PREP_CPU_WORKERS."""
    raw = os.environ.get("PREP_CPU_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(2, min(8, max(1, (os.cpu_count() or 4) // 2)))


def _triple_barrier_labels_for_symbol(
    sym_times: np.ndarray,
    sym_highs: np.ndarray,
    sym_lows: np.ndarray,
    sym_upper: np.ndarray,
    sym_lower: np.ndarray,
    horizon: int,
    overnight_gap_ns: int,
) -> np.ndarray:
    """Per-symbol triple-barrier labels (length = sym_times). Top-level for ProcessPoolExecutor."""
    sym_len = len(sym_times)
    labels = np.full(sym_len, 2, dtype=np.int64)
    next_break = np.full(sym_len, sym_len, dtype=np.int32)
    if sym_len > 1:
        breaks_after = np.diff(sym_times) > overnight_gap_ns
        next_break_idx = sym_len
        for rel_idx in range(sym_len - 2, -1, -1):
            if breaks_after[rel_idx]:
                next_break_idx = rel_idx + 1
            next_break[rel_idx] = next_break_idx

    for rel_i in range(sym_len):
        valid_end = min(rel_i + horizon + 1, next_break[rel_i], sym_len)
        if valid_end <= rel_i + 1:
            continue

        upper_limit = sym_upper[rel_i]
        lower_limit = sym_lower[rel_i]
        for rel_j in range(rel_i + 1, valid_end):
            hit_up = sym_highs[rel_j] >= upper_limit
            hit_dn = sym_lows[rel_j] <= lower_limit
            if hit_up and hit_dn:
                labels[rel_i] = 2
                break
            if hit_up:
                labels[rel_i] = 0
                break
            if hit_dn:
                labels[rel_i] = 1
                break
    return labels


def _triple_barrier_pool_task(task: tuple) -> tuple[int, int, np.ndarray]:
    """Picklable (start, end, labels_part) for ProcessPoolExecutor."""
    (
        start_idx,
        end_idx,
        sym_times,
        sym_highs,
        sym_lows,
        sym_upper,
        sym_lower,
        horizon,
        overnight_gap_ns,
    ) = task
    part = _triple_barrier_labels_for_symbol(
        sym_times,
        sym_highs,
        sym_lows,
        sym_upper,
        sym_lower,
        int(horizon),
        int(overnight_gap_ns),
    )
    return int(start_idx), int(end_idx), part


def _norm_memmap_chunk_task(task: tuple) -> tuple[int, int, np.ndarray]:
    """Read one memmap chunk, normalize in worker, return (sl, su, block) for parent write."""
    mmap_path, shape, dtype_str, sl, su, mean, std = task
    dtype = np.dtype(dtype_str)
    mm = np.memmap(mmap_path, dtype=dtype, mode="r", shape=shape)
    # Keep normalization in float32 to avoid huge per-worker float64 allocations.
    mean32 = mean.astype(np.float32, copy=False)
    std32 = std.astype(np.float32, copy=False)
    block = np.array(mm[sl:su], dtype=np.float32, copy=True)
    block = (block - mean32) / std32
    np.nan_to_num(block, copy=False, nan=0.0)
    return sl, su, block


def _load_and_compute_pa(symbol: str) -> pd.DataFrame:
    return load_or_build_pa_features(symbol, DATA_DIR, timeframe=pa_feature_cache_timeframe_key())


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
        "head_shoulders", "pa_struct_break", "pa_hmm_", "prev_day"
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
        
        valid_mask = np.ones(len(dates) - seq_len + 1, dtype=bool)
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
        
        # To allow trading at the open, we MUST allow sequences to cross the overnight gap.
        # Otherwise, the first `seq_len` minutes of every day will have no predictions!
        # The PA features are mostly stationary, so crossing the day boundary is perfectly fine.
        valid_mask = np.ones(len(windows), dtype=bool)
        
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
    ch = TCN_MEMMAP_CHUNK
    n_norm = (n + ch - 1) // ch

    raw_nw = os.environ.get("TCN_MEMMAP_NORM_WORKERS", "").strip()
    if raw_nw:
        norm_workers = max(1, int(raw_nw))
    else:
        norm_workers = min(_prep_cpu_workers(), n_norm, 16)
    serial = os.environ.get("TCN_MEMMAP_NORM_SERIAL", "").strip().lower() in {"1", "true", "yes"}
    if serial or norm_workers <= 1 or n_norm <= 1:
        mean32 = mean.astype(np.float32, copy=False)
        std32 = std.astype(np.float32, copy=False)
        for sl in _tq(range(0, n, ch), desc="TCN normalize", total=n_norm, unit="chunk"):
            su = min(ch, n - sl)
            block = np.array(mm[sl : sl + su], dtype=np.float32, copy=True)
            block = (block - mean32) / std32
            np.nan_to_num(block, copy=False, nan=0.0)
            mm[sl : sl + su] = block
        mm.flush()
        return

    path = os.fspath(mm.filename)
    shape = mm.shape
    dtype_str = str(mm.dtype)
    print(
        f"  TCN memmap normalize: up to {min(norm_workers, n_norm)} worker(s), {n_norm} chunk(s) "
        f"(set TCN_MEMMAP_NORM_SERIAL=1 for single-process)",
        flush=True,
    )
    mm.flush()
    tasks: list[tuple] = []
    for sl in range(0, n, ch):
        su = min(sl + ch, n)
        tasks.append((path, shape, dtype_str, sl, su, mean, std))
    nw = min(norm_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(_norm_memmap_chunk_task, t) for t in tasks]
        for fut in _tq(as_completed(futs), total=len(futs), desc="TCN normalize", unit="chunk"):
            sl, su, block = fut.result()
            mm[sl:su] = block
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
            "  Note: each symbol PA feature build on full history can take several minutes; progress advances when any symbol finishes.",
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
    ms = pd.to_numeric(df["market_state"], errors="coerce").fillna(4).astype(int)
    
    # We use 1m resolution for Triple Barrier to be as accurate as possible without missing intra-bar spikes.
    # The horizon is in minutes. For intraday options (gamma scalping), 15 minutes is ideal.
    horizon = 15
    # To catch a gamma burst in 15 mins, we expect a fast 1.0 ATR_5m move (approx. 1 average 5m bar's range).
    atr_mult = 1.0
    
    # Pre-calculate upper and lower barrier prices for each bar
    closes = df["close"].values
    atrs = df["atr_5m"].values if "atr_5m" in df.columns else df["atr_1m"].values if "atr_1m" in df.columns else np.zeros_like(closes)
    # Fallback if ATR is somehow missing (though add_pa_features usually computes it)
    if atrs.sum() == 0:
        atrs = (df["high"] - df["low"]).ewm(span=14, min_periods=1).mean().fillna(0).values
    
    upper_barriers = closes + atr_mult * atrs
    lower_barriers = closes - atr_mult * atrs
    
    highs = df["high"].to_numpy(dtype=np.float32, copy=False)
    lows = df["low"].to_numpy(dtype=np.float32, copy=False)
    time_ns = df["time_key"].to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    n_rows = len(df)
    labels = np.full(n_rows, 2, dtype=np.int64) # Default to 2 (Time Stop / Chop)
    
    # We must ensure we don't look ahead across different symbols
    sym_boundaries = df.groupby("symbol").size().cumsum().values
    sym_starts = np.insert(sym_boundaries[:-1], 0, 0)
    
    print(f"  Applying Triple Barrier Labeling (Horizon={horizon}, ATR_mult={atr_mult})...", flush=True)
    overnight_gap_ns = int(4 * 60 * 60 * 1_000_000_000)
    n_sym_bar = len(sym_starts)
    raw_bw = os.environ.get("TCN_BARRIER_WORKERS", "").strip()
    if raw_bw:
        barrier_workers = max(1, int(raw_bw))
    else:
        barrier_workers = min(_prep_cpu_workers(), max(1, n_sym_bar))
    use_pool = (
        barrier_workers > 1
        and n_sym_bar > 1
        and os.environ.get("TCN_BARRIER_SERIAL", "").strip().lower() not in {"1", "true", "yes"}
    )
    if use_pool:
        print(
            f"  Triple-barrier: {min(barrier_workers, n_sym_bar)} process(es) over {n_sym_bar} symbol(s) "
            f"(TCN_BARRIER_WORKERS / PREP_CPU_WORKERS; TCN_BARRIER_SERIAL=1 to disable)",
            flush=True,
        )
        pool_tasks = []
        for start_idx, end_idx in zip(sym_starts, sym_boundaries):
            pool_tasks.append(
                (
                    start_idx,
                    end_idx,
                    time_ns[start_idx:end_idx],
                    highs[start_idx:end_idx],
                    lows[start_idx:end_idx],
                    upper_barriers[start_idx:end_idx],
                    lower_barriers[start_idx:end_idx],
                    horizon,
                    overnight_gap_ns,
                )
            )
        with ProcessPoolExecutor(max_workers=min(barrier_workers, n_sym_bar)) as ex:
            futs = [ex.submit(_triple_barrier_pool_task, t) for t in pool_tasks]
            for fut in futs:
                s, e, part = fut.result()
                labels[s:e] = part
    else:
        for start_idx, end_idx in zip(sym_starts, sym_boundaries):
            labels[start_idx:end_idx] = _triple_barrier_labels_for_symbol(
                time_ns[start_idx:end_idx],
                highs[start_idx:end_idx],
                lows[start_idx:end_idx],
                upper_barriers[start_idx:end_idx],
                lower_barriers[start_idx:end_idx],
                horizon,
                overnight_gap_ns,
            )

    df["state_label"] = labels

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
        mmap_path = os.path.join(mmap_dir, f"tcn_X_seq_{seq_len}_{os.getpid()}.dat")
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


def ensure_tcn_state_classifier_checkpoint() -> None:
    """If ``tcn_meta.pkl`` + ``STATE_CLASSIFIER_FILE`` are missing, train PA-state TCN and save.

    LGBM dataset prep (``prepare_dataset``) runs real TCN inference and requires these artifacts.
    Set ``TCN_SKIP_AUTO_TRAIN=1`` to disable and restore files manually.
    """
    if os.environ.get("TCN_SKIP_AUTO_TRAIN", "").strip().lower() in {"1", "true", "yes"}:
        return
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    model_path = os.path.join(MODEL_DIR, STATE_CLASSIFIER_FILE)
    if os.path.isfile(meta_path) and os.path.isfile(model_path):
        return

    from core.training.l1a.tcn import (  # local import: pulls torch training stack
        TCN_HEAD_NUM_CLASSES,
        TCN_MIN_ATTENTION_SEQ_LEN,
        TCN_READOUT_TYPE,
        train_tcn,
    )

    print(
        "\n[*] PA-state TCN checkpoint missing (tcn_meta.pkl + "
        f"{STATE_CLASSIFIER_FILE!r}). Training TCN via tcn_data_prep.prepare_data() …",
        flush=True,
    )
    mm, y, train_idx, cal_idx, test_idx, norm_stats = prepare_data()
    n_feat = len(norm_stats["feat_cols"])
    final_model = train_tcn(
        mm,
        y,
        train_idx,
        cal_idx,
        test_idx,
        n_feat,
        norm_stats["ts"],
        norm_stats["syms"],
    )

    meta_out = {
        "feat_cols": list(norm_stats["feat_cols"]),
        "seq_len": int(norm_stats["seq_len_used"]),
        "input_size": int(n_feat),
        "num_channels": list(SLIM_CHANNELS),
        "kernel_size": int(TCN_KERNEL_SIZE),
        "num_regime_classes": int(TCN_HEAD_NUM_CLASSES),
        "bottleneck_dim": int(TCN_BOTTLENECK_DIM),
        "readout_type": str(TCN_READOUT_TYPE),
        "min_attention_seq_len": int(TCN_MIN_ATTENTION_SEQ_LEN),
        "mean": norm_stats["mean"],
        "std": norm_stats["std"],
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    torch.save(sd, model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta_out, f)
    print(f"  Saved TCN state_dict -> {model_path}", flush=True)
    print(f"  Saved TCN meta -> {meta_path}", flush=True)

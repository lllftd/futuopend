from __future__ import annotations

import gc
import os
import pickle
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.mamba_pa_state import PAStateMamba
from core.tcn_pa_state import PAStateTCN, FocalLoss

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.tcn_constants import STATE_CLASSIFIER_FILE as TCN_STATE_DICT_BASENAME

def _load_and_compute_pa(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw.sort_values("time_key").reset_index(drop=True)

    atr_1m = compute_atr(raw, length=14)
    print(f"  [{symbol}] Computing PA features on {len(raw):,} 1-min bars …")
    t0 = _time.time()
    df_pa = add_pa_features(raw, atr_1m, timeframe="5min")
    print(f"  [{symbol}] PA features done in {_time.time()-t0:.1f}s  → {df_pa.shape[1]} cols")
    return df_pa


def _load_labels(symbol: str) -> pd.DataFrame:
    cols = [
        "time_key", "market_state",
        "signal", "outcome",
        "quality_bull_breakout", "quality_bear_breakout",
        "max_favorable", "max_adverse", "exit_bar",
        "atr",
    ]
    lbl = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}{LABELED_SUFFIX}.csv"), usecols=cols)
    lbl["time_key"] = pd.to_datetime(lbl["time_key"])
    lbl.rename(columns={"atr": "lbl_atr"}, inplace=True)
    return lbl


def _merge_pa_labels_for_symbol(symbol: str) -> pd.DataFrame:
    """Picklable worker: PA + labels merge (parallel per symbol via ProcessPoolExecutor)."""
    pa = _load_and_compute_pa(symbol)
    lbl = _load_labels(symbol)
    merged = pd.merge(pa, lbl, on="time_key", how="inner")
    merged["symbol"] = symbol
    print(f"  [{symbol}] Merged {len(merged):,} rows", flush=True)
    return merged


def _pa_feature_cols(df: pd.DataFrame) -> list[str]:
    """PA/OR columns suitable for LightGBM (numeric or bool only).

    Excludes object/category/string columns — e.g. causal PA may add string tags like channel names.
    """
    cols = []
    for c in df.columns:
        if not c.startswith(("pa_", "or_")):
            continue
        if _is_lgbm_string_tag_col(c):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            cols.append(c)
    return sorted(cols)


def _log_tcn_feature_health(df: pd.DataFrame, tcn_cols: list[str]) -> None:
    """Warn if TCN columns are constant / NaN — LightGBM gain will be ~0."""
    print("\n  TCN feature health (post-merge, full df):")
    for c in tcn_cols:
        if c not in df.columns:
            print(f"    {c}: MISSING")
            continue
        s = df[c]
        n_nan = int(s.isna().sum())
        std = float(s.std(skipna=True))
        vmin, vmax = float(s.min(skipna=True)), float(s.max(skipna=True))
        flag = ""
        if n_nan:
            flag = "  [NaN]"
        elif std < 1e-12:
            flag = "  [~constant → gain≈0 in LGBM]"
        print(f"    {c}: std={std:.6g}  min={vmin:.6g}  max={vmax:.6g}  nan={n_nan}{flag}")


def _create_tcn_windows(feat_1m: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(feat_1m)
    if n < seq_len:
        return np.zeros((0, seq_len, feat_1m.shape[1]), dtype=np.float32), np.zeros(0, dtype=int)
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    windows = feat_1m[idx].astype(np.float32)
    end_idx = np.arange(seq_len - 1, n, dtype=int)
    return windows, end_idx


def _compute_tcn_derived_features(df: pd.DataFrame, base_feat_cols: list[str]) -> pd.DataFrame:
    """
    Real TCN forward only — no uniform-prior placeholders.
    Requires ``tcn_meta.pkl`` + the same weight file basename as ``train_pipeline`` /
    ``tcn_constants.STATE_CLASSIFIER_FILE`` (not legacy ``tcn_transition_classifier.pt``).
    """
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    model_path = os.path.join(MODEL_DIR, TCN_STATE_DICT_BASENAME)

    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise RuntimeError(
            f"Missing TCN checkpoint under {MODEL_DIR}: need tcn_meta.pkl and {TCN_STATE_DICT_BASENAME!r} "
            "(from Layer 1 / backtests.train_pipeline). Retrain: ./scripts/run_train.sh layer1"
        )

    import pickle

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    device = _tcn_inference_device()
    tqdm.write(f"  TCN derived features: device={device}  (set TORCH_DEVICE=cpu|mps|cuda)")

    # Extra isolation: PyTorch Conv1d on CPU + multi-threaded BLAS has crashed with SIGSEGV on some Mac builds.
    _prev_intra = torch.get_num_threads()
    _prev_inter = torch.get_num_interop_threads()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def _restore_torch_threads() -> None:
        try:
            torch.set_num_threads(_prev_intra)
            torch.set_num_interop_threads(_prev_inter)
        except RuntimeError:
            pass

    try:
        tcn_feat_cols = meta.get("feat_cols", [])
        if not tcn_feat_cols:
            tcn_feat_cols = base_feat_cols
        missing = [c for c in tcn_feat_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing TCN input features: {missing[:10]}")
    
        seq_len = int(meta.get("seq_len", 30))
        input_size = int(meta["input_size"])
        bottleneck_dim = int(meta.get("bottleneck_dim", TCN_BOTTLENECK_DIM))
        if bottleneck_dim != TCN_BOTTLENECK_DIM:
            print(
                f"  Note: meta bottleneck_dim={bottleneck_dim} "
                f"(env TCN_BOTTLENECK_DIM={TCN_BOTTLENECK_DIM}); using meta.",
                flush=True,
            )
        n_tcn_classes = int(meta.get("num_regime_classes", 2))
        if n_tcn_classes != len(TCN_REGIME_FUT_PROB_COLS):
            raise ValueError(f"TCN output classes ({n_tcn_classes}) does not match TCN_REGIME_FUT_PROB_COLS ({len(TCN_REGIME_FUT_PROB_COLS)}).")
        
        model = PAStateTCN(
            input_size=input_size,
            num_channels=meta["num_channels"],
            kernel_size=meta["kernel_size"],
            dropout=0.0,
            bottleneck_dim=bottleneck_dim,
            num_classes=n_tcn_classes,
        )
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval()
        # Mac / CPU PyTorch Conv1d + large batches can SIGSEGV; keep CPU batches small unless overridden.
        _bs_default = "4096" if device.type in ("mps", "cuda") else "64"
        batch_size = max(8, int(os.environ.get("TCN_BATCH_SIZE", _bs_default)))
    
        feat_mean = np.asarray(meta["mean"], dtype=np.float32)
        feat_std = np.asarray(meta["std"], dtype=np.float32)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    
        work = df.copy()
    
        sym_outputs: list[pd.DataFrame] = []
    
        sym_groups = list(work.groupby("symbol"))
        for sym, grp in _tq(
            sym_groups,
            desc="  TCN→LGBM per-symbol",
            unit="sym",
            total=len(sym_groups),
            leave=True,
        ):
            g = grp.sort_values("time_key").reset_index(drop=True)

            x_raw = g[tcn_feat_cols].values.astype(np.float32)
            x_norm = np.nan_to_num((x_raw - feat_mean) / feat_std, nan=0.0).astype(np.float32)
            windows, end_idx = _create_tcn_windows(x_norm, seq_len)

            n_bars = len(g)
            if len(windows) == 0:
                raise RuntimeError(
                    f"No TCN windows for symbol {sym!r}: rows={n_bars} seq_len={seq_len}. "
                    "Need enough history per symbol."
                )

            regime_probs_arr = np.full((n_bars, n_tcn_classes), np.nan, dtype=np.float32)
            embeddings = np.full((n_bars, bottleneck_dim), np.nan, dtype=np.float32)

            all_reg_prob, all_emb = [], []
            bs = batch_size
            win_batches = range(0, len(windows), bs)
            n_win_b = (len(windows) + bs - 1) // bs
            with torch.inference_mode():
                for i in _tq(
                    win_batches,
                    desc=f"  {sym}",
                    total=n_win_b,
                    leave=False,
                    unit="batch",
                ):
                    chunk = np.ascontiguousarray(windows[i : i + bs])
                    xb = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
                    regime_logits, emb = model.forward_with_embedding(xb)
                    p_reg = torch.softmax(regime_logits, dim=1).cpu().numpy()
                    all_reg_prob.append(p_reg)
                    all_emb.append(emb.detach().cpu().numpy())
                    del xb, regime_logits, emb

            p_regs = np.concatenate(all_reg_prob, axis=0)
            embs = np.concatenate(all_emb, axis=0)

            regime_probs_arr[end_idx] = p_regs
            embeddings[end_idx] = embs

            regime_probs_arr = pd.DataFrame(regime_probs_arr).ffill().bfill().values.astype(np.float32)
            embeddings = pd.DataFrame(embeddings).ffill().bfill().values.astype(np.float32)

            if np.isnan(regime_probs_arr).any():
                raise RuntimeError(f"TCN regime probs still NaN for symbol {sym!r} after ffill/bfill.")
            if np.isnan(embeddings).any():
                raise RuntimeError(f"TCN embeddings still NaN for symbol {sym!r} after ffill/bfill.")

            sym_df = g[["symbol", "time_key"]].copy()
            eps = 1e-9
            sym_df["tcn_regime_fut_entropy"] = -np.sum(
                regime_probs_arr * np.log(np.maximum(regime_probs_arr, eps)), axis=1
            )
            for j, col in enumerate(TCN_REGIME_FUT_PROB_COLS):
                sym_df[col] = regime_probs_arr[:, j]
            for j in range(bottleneck_dim):
                sym_df[f"tcn_emb_{j}"] = embeddings[:, j]

            sym_outputs.append(sym_df)

        tcn_derived_list = _tcn_derived_feature_names(bottleneck_dim)
        tqdm.write(
            f"  TCN→LGBM: learned bottleneck z dim={bottleneck_dim} "
            f"(columns tcn_emb_0 … tcn_emb_{bottleneck_dim - 1})"
        )

        tcn_1m = pd.concat(sym_outputs, ignore_index=True)
        merged = work.merge(tcn_1m, on=["symbol", "time_key"], how="left")
        merged = merged.sort_values(["symbol", "time_key"])
        tcn_cols = [c for c in tcn_derived_list if c in merged.columns]
        for c in _tq(
            tcn_cols,
            desc="  TCN post-merge ffill",
            unit="col",
            leave=False,
        ):
            merged[c] = merged.groupby("symbol", group_keys=False)[c].transform(
                lambda s: s.ffill().bfill()
            )

        # --- INJECT OOF CACHE TO PREVENT DATA LEAKAGE ---
        oof_path = os.path.join(MODEL_DIR, "tcn_oof_cache.pkl")
        if os.path.exists(oof_path):
            tqdm.write(f"  Injecting TCN OOF predictions from {oof_path} into train split...")
            with open(oof_path, "rb") as f:
                oof = pickle.load(f)
            if "regime_probs" not in oof:
                raise RuntimeError(
                    f"{oof_path} is missing 'regime_probs' (transition). Retrain TCN: train_tcn_pa_state.py"
                )

            oof_df = pd.DataFrame({
                "symbol": oof["syms"],
                "time_key": pd.to_datetime(oof["ts"])
            })
            rp = np.asarray(oof["regime_probs"], dtype=np.float32)
            if rp.shape[1] != len(TCN_REGIME_FUT_PROB_COLS):
                raise RuntimeError(
                    f"{oof_path} regime_probs has width {rp.shape[1]} but TCN_REGIME_FUT_PROB_COLS "
                    f"expects {len(TCN_REGIME_FUT_PROB_COLS)} (e.g. old 6-class cache vs binary transition). "
                    f"Delete {oof_path} and re-train Layer 1 (TCN)."
                )
            for j, col in enumerate(TCN_REGIME_FUT_PROB_COLS):
                oof_df[col] = rp[:, j]

            eps = 1e-9
            oof_df["tcn_regime_fut_entropy"] = -np.sum(
                rp * np.log(np.maximum(rp, eps)), axis=1
            )
            oem = np.asarray(oof["embeds"], dtype=np.float32)
            if oem.shape[1] != bottleneck_dim:
                raise RuntimeError(
                    f"tcn_oof_cache.pkl embed width {oem.shape[1]} != bottleneck_dim {bottleneck_dim} "
                    f"from tcn_meta.pkl. Retrain TCN: backtests/train_tcn_pa_state.py"
                )
            for j in range(bottleneck_dim):
                oof_df[f"tcn_emb_{j}"] = oem[:, j]

            # Merge OOF into 'merged' (batched so tqdm can show progress; update() is opaque).
            merged.set_index(["symbol", "time_key"], inplace=True)
            oof_df.set_index(["symbol", "time_key"], inplace=True)

            update_cols = [c for c in tcn_derived_list if c in oof_df.columns]
            common_idx = merged.index.intersection(oof_df.index)
            n_common = len(common_idx)
            if n_common > 0:
                batch = max(256, int(os.environ.get("TCN_OOF_INJECT_BATCH", "65536")))
                idx_list = list(common_idx)
                n_batches = (n_common + batch - 1) // batch
                for b in _tq(
                    range(n_batches),
                    desc="  OOF inject (train rows)",
                    total=n_batches,
                    unit="batch",
                    leave=True,
                ):
                    lo = b * batch
                    hi = min(lo + batch, n_common)
                    sl = idx_list[lo:hi]
                    merged.loc[sl, update_cols] = oof_df.loc[sl, update_cols].values

            merged.reset_index(inplace=True)
        # ------------------------------------------------

        if merged[tcn_cols].isna().any().any():
            bad = [c for c in tcn_cols if merged[c].isna().any()]
            raise RuntimeError(
                f"TCN columns still NaN after per-symbol ffill/bfill (merge alignment?): {bad[:8]}"
            )
        return merged
    finally:
        _restore_torch_threads()



def _compute_mamba_derived_features(df: pd.DataFrame, base_feat_cols: list[str]) -> pd.DataFrame:
    """
    Real Mamba forward only — no uniform-prior placeholders.
    Requires ``mamba_meta.pkl`` + the same weight file basename as ``train_pipeline`` /
    ``tcn_constants.STATE_CLASSIFIER_FILE`` (not legacy ``tcn_transition_classifier.pt``).
    """
    meta_path = os.path.join(MODEL_DIR, "mamba_meta.pkl")
    model_path = os.path.join(MODEL_DIR, "mamba_state_classifier_6c.pt")

    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise RuntimeError(
            f"Missing Mamba checkpoint under {MODEL_DIR}: need mamba_meta.pkl and {os.path.basename(model_path)!r} "
            "(from Layer 1 / backtests.train_pipeline). Retrain: ./scripts/run_train.sh layer1"
        )

    import pickle

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    device = _tcn_inference_device()
    tqdm.write(f"  Mamba derived features: device={device}  (set TORCH_DEVICE=cpu|mps|cuda)")

    # Extra isolation: PyTorch Conv1d on CPU + multi-threaded BLAS has crashed with SIGSEGV on some Mac builds.
    _prev_intra = torch.get_num_threads()
    _prev_inter = torch.get_num_interop_threads()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def _restore_torch_threads() -> None:
        try:
            torch.set_num_threads(_prev_intra)
            torch.set_num_interop_threads(_prev_inter)
        except RuntimeError:
            pass

    try:
        mamba_feat_cols = meta.get("feat_cols", [])
        if not mamba_feat_cols:
            mamba_feat_cols = base_feat_cols
        missing = [c for c in mamba_feat_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing Mamba input features: {missing[:10]}")
    
        seq_len = int(meta.get("seq_len", 30))
        input_size = int(meta["input_size"])
        bottleneck_dim = int(meta.get("bottleneck_dim", int(meta.get('bottleneck_dim', 8))))
        if bottleneck_dim != int(meta.get('bottleneck_dim', 8)):
            print(
                f"  Note: meta bottleneck_dim={bottleneck_dim} "
                f"(env int(meta.get('bottleneck_dim', 8))={int(meta.get('bottleneck_dim', 8))}); using meta.",
                flush=True,
            )
        n_mamba_classes = int(meta.get("num_regime_classes", 2))
        if n_mamba_classes != len(MAMBA_REGIME_FUT_PROB_COLS):
            raise ValueError(f"Mamba output classes ({n_mamba_classes}) does not match MAMBA_REGIME_FUT_PROB_COLS ({len(MAMBA_REGIME_FUT_PROB_COLS)}).")
        
        model = PAStateMamba(
            input_size=input_size,
            d_model=int(meta.get("d_model", 64)),
            n_layers=int(meta.get("n_layers", 4)),
            dropout=0.0,
            bottleneck_dim=bottleneck_dim,
            num_classes=n_mamba_classes
        )
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval()
        # Mac / CPU PyTorch Conv1d + large batches can SIGSEGV; keep CPU batches small unless overridden.
        _bs_default = "4096" if device.type in ("mps", "cuda") else "64"
        batch_size = max(8, int(os.environ.get("MAMBA_BATCH_SIZE", _bs_default)))
    
        feat_mean = np.asarray(meta["mean"], dtype=np.float32)
        feat_std = np.asarray(meta["std"], dtype=np.float32)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    
        work = df.copy()
    
        sym_outputs: list[pd.DataFrame] = []
    
        sym_groups = list(work.groupby("symbol"))
        for sym, grp in _tq(
            sym_groups,
            desc="  Mamba→LGBM per-symbol",
            unit="sym",
            total=len(sym_groups),
            leave=True,
        ):
            g = grp.sort_values("time_key").reset_index(drop=True)

            x_raw = g[mamba_feat_cols].values.astype(np.float32)
            x_norm = np.nan_to_num((x_raw - feat_mean) / feat_std, nan=0.0).astype(np.float32)
            windows, end_idx = _create_tcn_windows(x_norm, seq_len)

            n_bars = len(g)
            if len(windows) == 0:
                raise RuntimeError(
                    f"No Mamba windows for symbol {sym!r}: rows={n_bars} seq_len={seq_len}. "
                    "Need enough history per symbol."
                )

            regime_probs_arr = np.full((n_bars, n_mamba_classes), np.nan, dtype=np.float32)
            embeddings = np.full((n_bars, bottleneck_dim), np.nan, dtype=np.float32)

            all_reg_prob, all_emb = [], []
            bs = batch_size
            win_batches = range(0, len(windows), bs)
            n_win_b = (len(windows) + bs - 1) // bs
            with torch.inference_mode():
                for i in _tq(
                    win_batches,
                    desc=f"  {sym}",
                    total=n_win_b,
                    leave=False,
                    unit="batch",
                ):
                    chunk = np.ascontiguousarray(windows[i : i + bs])
                    xb = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
                    regime_logits, emb = model.forward_with_embedding(xb)
                    p_reg = torch.softmax(regime_logits, dim=1).cpu().numpy()
                    all_reg_prob.append(p_reg)
                    all_emb.append(emb.detach().cpu().numpy())
                    del xb, regime_logits, emb

            p_regs = np.concatenate(all_reg_prob, axis=0)
            embs = np.concatenate(all_emb, axis=0)

            regime_probs_arr[end_idx] = p_regs
            embeddings[end_idx] = embs

            regime_probs_arr = pd.DataFrame(regime_probs_arr).ffill().bfill().values.astype(np.float32)
            embeddings = pd.DataFrame(embeddings).ffill().bfill().values.astype(np.float32)

            if np.isnan(regime_probs_arr).any():
                raise RuntimeError(f"Mamba regime probs still NaN for symbol {sym!r} after ffill/bfill.")
            if np.isnan(embeddings).any():
                raise RuntimeError(f"Mamba embeddings still NaN for symbol {sym!r} after ffill/bfill.")

            sym_df = g[["symbol", "time_key"]].copy()
            eps = 1e-9
            sym_df["mamba_regime_fut_entropy"] = -np.sum(
                regime_probs_arr * np.log(np.maximum(regime_probs_arr, eps)), axis=1
            )
            for j, col in enumerate(MAMBA_REGIME_FUT_PROB_COLS):
                sym_df[col] = regime_probs_arr[:, j]
            for j in range(bottleneck_dim):
                sym_df[f"mamba_emb_{j}"] = embeddings[:, j]

            sym_outputs.append(sym_df)

        mamba_derived_list = _mamba_derived_feature_names(bottleneck_dim)
        tqdm.write(
            f"  Mamba→LGBM: learned bottleneck z dim={bottleneck_dim} "
            f"(columns mamba_emb_0 … mamba_emb_{bottleneck_dim - 1})"
        )

        mamba_1m = pd.concat(sym_outputs, ignore_index=True)
        merged = work.merge(mamba_1m, on=["symbol", "time_key"], how="left")
        merged = merged.sort_values(["symbol", "time_key"])
        mamba_cols = [c for c in mamba_derived_list if c in merged.columns]
        for c in _tq(
            mamba_cols,
            desc="  Mamba post-merge ffill",
            unit="col",
            leave=False,
        ):
            merged[c] = merged.groupby("symbol", group_keys=False)[c].transform(
                lambda s: s.ffill().bfill()
            )

        # --- INJECT OOF CACHE TO PREVENT DATA LEAKAGE ---
        oof_path = os.path.join(MODEL_DIR, "mamba_oof_cache.pkl")
        if os.path.exists(oof_path):
            tqdm.write(f"  Injecting Mamba OOF predictions from {oof_path} into train split...")
            with open(oof_path, "rb") as f:
                oof = pickle.load(f)
            if "regime_probs" not in oof:
                raise RuntimeError(
                    f"{oof_path} is missing 'regime_probs' (transition). Retrain TCN: layer1_mamba.py"
                )

            oof_df = pd.DataFrame({
                "symbol": oof["syms"],
                "time_key": pd.to_datetime(oof["ts"])
            })
            rp = np.asarray(oof["regime_probs"], dtype=np.float32)
            if rp.shape[1] != len(MAMBA_REGIME_FUT_PROB_COLS):
                raise RuntimeError(
                    f"{oof_path} regime_probs has width {rp.shape[1]} but MAMBA_REGIME_FUT_PROB_COLS "
                    f"expects {len(MAMBA_REGIME_FUT_PROB_COLS)} (e.g. old 6-class cache vs binary transition). "
                    f"Delete {oof_path} and re-train Layer 1 (TCN)."
                )
            for j, col in enumerate(MAMBA_REGIME_FUT_PROB_COLS):
                oof_df[col] = rp[:, j]

            eps = 1e-9
            oof_df["mamba_regime_fut_entropy"] = -np.sum(
                rp * np.log(np.maximum(rp, eps)), axis=1
            )
            oem = np.asarray(oof["embeds"], dtype=np.float32)
            if oem.shape[1] != bottleneck_dim:
                raise RuntimeError(
                    f"mamba_oof_cache.pkl embed width {oem.shape[1]} != bottleneck_dim {bottleneck_dim} "
                    f"from mamba_meta.pkl. Retrain TCN: backtests/layer1_mamba.py"
                )
            for j in range(bottleneck_dim):
                oof_df[f"mamba_emb_{j}"] = oem[:, j]

            # Merge OOF into 'merged' (batched so tqdm can show progress; update() is opaque).
            merged.set_index(["symbol", "time_key"], inplace=True)
            oof_df.set_index(["symbol", "time_key"], inplace=True)

            update_cols = [c for c in mamba_derived_list if c in oof_df.columns]
            common_idx = merged.index.intersection(oof_df.index)
            n_common = len(common_idx)
            if n_common > 0:
                batch = max(256, int(os.environ.get("TCN_OOF_INJECT_BATCH", "65536")))
                idx_list = list(common_idx)
                n_batches = (n_common + batch - 1) // batch
                for b in _tq(
                    range(n_batches),
                    desc="  OOF inject (train rows)",
                    total=n_batches,
                    unit="batch",
                    leave=True,
                ):
                    lo = b * batch
                    hi = min(lo + batch, n_common)
                    sl = idx_list[lo:hi]
                    merged.loc[sl, update_cols] = oof_df.loc[sl, update_cols].values

            merged.reset_index(inplace=True)
        # ------------------------------------------------

        if merged[mamba_cols].isna().any().any():
            bad = [c for c in mamba_cols if merged[c].isna().any()]
            raise RuntimeError(
                f"Mamba columns still NaN after per-symbol ffill/bfill (merge alignment?): {bad[:8]}"
            )
        return merged
    finally:
        _restore_torch_threads()


def prepare_dataset(symbols: list[str] = ["QQQ", "SPY"]):
    n_sym = len(symbols)
    # Parallel CPU-bound PA+labels per symbol (safe). Set DATA_PREPARE_WORKERS=1 to disable.
    _dw = os.environ.get("DATA_PREPARE_WORKERS", "").strip()
    default_w = min(n_sym, 2)
    prep_workers = int(_dw) if _dw else default_w
    prep_workers = max(1, min(prep_workers, n_sym))

    if prep_workers <= 1:
        parts = []
        for sym in _tq(symbols, desc="LGBM prepare_dataset", unit="sym"):
            parts.append(_merge_pa_labels_for_symbol(sym))
    else:
        print(
            f"  Parallel symbol load: {prep_workers} processes "
            f"(DATA_PREPARE_WORKERS; MPS/LGBM untouched)",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=prep_workers) as ex:
            fut_to_sym = {
                ex.submit(_merge_pa_labels_for_symbol, sym): sym for sym in symbols
            }
            parts_by_sym: dict[str, pd.DataFrame] = {}
            for fut in _tq(
                as_completed(fut_to_sym),
                total=n_sym,
                desc="LGBM prepare_dataset",
                unit="sym",
            ):
                sym = fut_to_sym[fut]
                parts_by_sym[sym] = fut.result()
            parts = [parts_by_sym[sym] for sym in symbols]

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["symbol", "time_key"]).reset_index(drop=True)

    print("\n  === market_state distribution (from labels CSV, merged rows) ===")
    ms_num = pd.to_numeric(df["market_state"], errors="coerce")
    for lbl_id in range(NUM_REGIME_CLASSES):
        cnt = int((ms_num == lbl_id).sum())
        pct = cnt / len(df) * 100
        print(f"    {lbl_id} ({STATE_NAMES[lbl_id]:>13s}): {cnt:>9,}  ({pct:.1f}%)")
    n_ms_nan = int(ms_num.isna().sum())
    if n_ms_nan:
        print(f"    NaN market_state rows: {n_ms_nan:,}")

    # Layer 2a: y = causal HMM argmax on this bar (matches PA features). CSV market_state can lag
    # label_v2/HMM tie logic; pa_hmm_state uses the same softmax as pa_hmm_prob_* with range boost.
    hmm_state = pd.to_numeric(df["pa_hmm_state"], errors="coerce")
    df["state_label"] = hmm_state.fillna(ms_num).fillna(4).astype(int).clip(0, NUM_REGIME_CLASSES - 1)

    feat_cols = _pa_feature_cols(df)
    df = _compute_tcn_derived_features(df, feat_cols)
    feat_cols = _unique_cols(feat_cols + _tcn_derived_feature_names())

    # Add Mamba features if model exists
    if os.path.exists(os.path.join(MODEL_DIR, "mamba_state_classifier_6c.pt")):
        df = _compute_mamba_derived_features(df, feat_cols)
        feat_cols = _unique_cols(feat_cols + _mamba_derived_feature_names())

    base_feats, hmm_feats, garch_feats, tcn_feats, mamba_feats = _split_feature_groups(feat_cols)
    print(f"\n  Feature columns: {len(feat_cols)}")
    print(f"    Base PA/OR:  {len(base_feats)}")
    print(f"    HMM-style:   {len(hmm_feats)}")
    print(f"    GARCH-style: {len(garch_feats)}")
    print(f"    TCN-derived: {len(tcn_feats)}")
    print(f"    Mamba-derived: {len(mamba_feats)}")
    _log_tcn_feature_health(df, tcn_feats)
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['time_key'].min()} → {df['time_key'].max()}")
    print(f"\n  Temporal splits:")
    print(f"    Train:       < {TRAIN_END}  ({(df['time_key'] < TRAIN_END).sum():>9,} rows)")
    print(f"    Calibration: {TRAIN_END} → {CAL_END}  ({((df['time_key'] >= TRAIN_END) & (df['time_key'] < CAL_END)).sum():>9,} rows)")
    print(f"    Test:        {CAL_END} → {TEST_END}  ({((df['time_key'] >= CAL_END) & (df['time_key'] < TEST_END)).sum():>9,} rows)")
    print(f"    Holdout:     >= {TEST_END}  ({(df['time_key'] >= TEST_END).sum():>9,} rows)  (not used in this run)")

    print(f"\n  Regime supervision label distribution — state_label (= current market_state):")
    for lbl_id, name in STATE_NAMES.items():
        cnt = (df["state_label"] == lbl_id).sum()
        pct = cnt / len(df) * 100
        print(f"    {lbl_id} ({name:>13s}): {cnt:>9,}  ({pct:.1f}%)")
    return df, feat_cols


def compute_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Breakout / bar-context features computable for any bar (causal).
    Uses OHLCV from raw data + PA columns where available."""
    n = len(df)
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values.astype(float)

    # ATR: prefer lbl_atr (from labeled data), then fall back to prev_5m_atr
    if "lbl_atr" in df.columns:
        atr_vals = df["lbl_atr"].values
    elif "prev_5m_atr" in df.columns:
        atr_vals = df["prev_5m_atr"].values
    else:
        atr_vals = np.full(n, 0.25)
    safe_atr = np.where(atr_vals > 0.001, atr_vals, 0.001)

    body_abs = np.abs(close - open_)
    bar_range = high - low
    safe_range = np.where(bar_range > 1e-6, bar_range, 1e-6)

    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
    atr5 = pd.Series(bar_range).rolling(5, min_periods=1).mean().values
    atr20 = pd.Series(bar_range).rolling(20, min_periods=1).mean().values
    body_ma5 = pd.Series(body_abs).rolling(5, min_periods=1).mean().values

    # New range expansion/compression features
    tp = (high + low + close) / 3.0
    tp_ma20 = pd.Series(tp).rolling(20, min_periods=1).mean().values
    tp_std20 = pd.Series(tp).rolling(20, min_periods=1).std().fillna(0).values
    bo_bb_width = np.where(tp_ma20 > 1e-6, (4.0 * tp_std20) / tp_ma20, 0.0)
    
    atr_ma500 = pd.Series(safe_atr).rolling(500, min_periods=1).mean().values
    atr_std500 = pd.Series(safe_atr).rolling(500, min_periods=1).std().fillna(1e-6).values
    bo_atr_zscore = (safe_atr - atr_ma500) / np.maximum(atr_std500, 1e-6)

    # Consecutive bars in dominant direction
    bo_consec = np.zeros(n)
    for cand in ["pa_consec_bull", "consec_bull"]:
        if cand in df.columns:
            bear_col = cand.replace("bull", "bear")
            bo_consec = np.where(
                close >= open_,
                df[cand].fillna(0).values,
                -df[bear_col].fillna(0).values,
            )
            break

    # Inside bar (prior bar)
    bo_inside_prior = np.zeros(n)
    for cand in ["pa_is_inside_bar", "is_inside_bar"]:
        if cand in df.columns:
            ib = df[cand].fillna(False).values.astype(int)
            bo_inside_prior[1:] = ib[:-1]
            break

    bo_pressure_diff = np.zeros(n)
    if "pa_bull_pressure" in df.columns and "pa_bear_pressure" in df.columns:
        bo_pressure_diff = (
            df["pa_bull_pressure"].fillna(0).values
            - df["pa_bear_pressure"].fillna(0).values
        )

    bo_or_dist = np.zeros(n)
    if "or_mid" in df.columns:
        or_mid = df["or_mid"].ffill().values
        bo_or_dist = (close - or_mid) / safe_atr

    gap_signal = np.zeros(n)
    for cand in ["pa_gap_up", "gap_up"]:
        if cand in df.columns:
            down_col = cand.replace("up", "down")
            gap_signal = df[cand].fillna(0).values - df.get(down_col, pd.Series(0, index=df.index)).fillna(0).values
            break

    out = pd.DataFrame({
        "bo_body_atr": body_abs / safe_atr,
        "bo_range_atr": bar_range / safe_atr,
        "bo_vol_spike": vol / np.where(vol_ma20 > 0, vol_ma20, 1.0),
        "bo_close_extremity": np.where(
            close >= open_,
            (close - low) / safe_range,
            (high - close) / safe_range,
        ),
        "bo_wick_imbalance": ((close - low) - (high - close)) / safe_range,
        "bo_range_compress": atr5 / np.where(atr20 > 1e-6, atr20, 1e-6),
        "bo_body_growth": body_abs / np.where(body_ma5 > 1e-6, body_ma5, 1e-6),
        "bo_gap_signal": gap_signal,
        "bo_consec_dir": bo_consec,
        "bo_inside_prior": bo_inside_prior,
        "bo_pressure_diff": bo_pressure_diff,
        "bo_or_dist": bo_or_dist,
        "bo_bb_width": bo_bb_width,
        "bo_atr_zscore": bo_atr_zscore,
    }, index=df.index)
    return out



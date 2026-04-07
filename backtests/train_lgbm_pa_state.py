"""
LightGBM PA layered training pipeline.

Feature block:
    Base PA/OR + HMM-style regime + GARCH-style volatility.

Layer 2a — Regime (6-class supervised head, y = causal pa_hmm_state, no shift):
    Input:  PA + GARCH only (no TCN; no pa_hmm_* — avoids leaking the HMM-derived label)
    Output: regime_now probabilities — saved as state_classifier_6c.txt (+ state_calibrators.pkl)

Layer 2b — Hierarchical trade-quality stack (y from outcome KMeans + rules):
    Input:  PA + TCN + breakout + GARCH columns only (regime probs are NOT features here)
    Step 1: 6-regime MFE/MAE regression → opportunity → synthetic p_trade (L2a argmax routing on cal/test)
    Step 2: LONG vs SHORT (on tradable regime)
    Step 3: A vs B quality (on tradable regime)
    Regime probabilities are used for Step1 routing, reconstructing the 6-class label, and Layer 3.

Layer 3 — Execution Sizer (two-stage on L2b regression outputs):
    Input:  l2b opportunity + pred_mfe/mae + regime_now + tcn_regime_fut + GARCH + PA key feats
            (+ regime×opportunity interactions). Legacy ``execution_sizer_v1`` if meta has no l3_schema.
    Output: signed position in [-1, 1] via clip(p_gate × size_head); train on full cal rows (no flat downsampling).
    Env:     L3_FLAT_TAU (default 0.05), L3_FLAT_WEIGHT (default 0.35) for near-zero target downweighting on size head.

Temporal protocol (training-effect focus, no OOS backtest run here):
    Train:       2020-03 → 2022-12-31
    Calibration: 2023-01 → 2023-06-30
    Test:        2023-07 → 2024-12-31

Concurrency:
    DATA_PREPARE_WORKERS — processes for parallel PA+label load per symbol (default min(n_sym,2); set 1 for serial).
    LightGBM already parallelizes with n_jobs; TCN infer stays single-process per GPU.

Logging:
    From repo root, ``scripts/run_train_lgbm.sh`` overwrites ``train_lgbm.log`` each run (no append).
    Manual tee needs ``FORCE_TQDM=1`` so tqdm still shows (``2>&1 | tee`` makes stderr non-TTY).
"""
from __future__ import annotations

import os

# Must run before numpy/torch/lightgbm load BLAS/OpenMP — avoids macOS SIGSEGV in Conv1d on CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gc
import sys
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lgbm_models")
# Must match label_v2 default output: data/{SYM}_labeled_v2.csv (override: LABELED_SUFFIX=_labeled)
LABELED_SUFFIX = os.environ.get("LABELED_SUFFIX", "_labeled_v2")

NUM_REGIME_CLASSES = 6
STATE_CLASSIFIER_FILE = "state_classifier_6c.txt"

# Same 6-class space as CSV market_state (label_pdf_pa_data.label_market_state).
STATE_NAMES: dict[int, str] = {
    0: "bull_conv",
    1: "bull_div",
    2: "bear_conv",
    3: "bear_div",
    4: "range_conv",
    5: "range_div",
}

# Canonical regime names in L2a class-index order:
#   LightGBM multiclass label k,
#   raw predict column k, state_calibrators.pkl[k],
#   REGIME_NOW_PROB_COLS[k] / calibrated prob column k,
#   and np.argmax(those probs) → k  all use the same k  ↔  REGIMES_6[k].
REGIMES_6: tuple[str, ...] = tuple(STATE_NAMES[i] for i in range(NUM_REGIME_CLASSES))
# L2a prob columns / class indices where REGIMES_6[i] is a range_* state (conv + div).
RANGE_REGIME_INDICES: tuple[int, ...] = tuple(i for i, r in enumerate(REGIMES_6) if r.startswith("range"))
# Layer 3 v2: opp from routed L2b heads × per-regime calibrated mass (one col per REGIMES_6 class).
L2B_OPP_X_REGIME_COLS: list[str] = [f"l2b_opp_x_{REGIMES_6[k]}" for k in range(NUM_REGIME_CLASSES)]

REGIME_NOW_PROB_COLS = [f"regime_now_{STATE_NAMES[i]}" for i in range(NUM_REGIME_CLASSES)]
# Layer 2a calibrated probs + max-prob confidence for Layer 3 / diagnostics
REGIME_PROB_COLS = REGIME_NOW_PROB_COLS + ["regime_now_conf"]

TCN_REGIME_FUT_PROB_COLS = [f"tcn_regime_fut_{STATE_NAMES[i]}" for i in range(NUM_REGIME_CLASSES)]
# Default when tcn_meta.pkl is missing; otherwise use meta["bottleneck_dim"] (see _tcn_derived_feature_names).
TCN_BOTTLENECK_DIM = max(1, int(os.environ.get("TCN_BOTTLENECK_DIM", "8")))


def _tcn_bottleneck_dim_from_meta() -> int:
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    if not os.path.isfile(meta_path):
        return TCN_BOTTLENECK_DIM
    import pickle

    with open(meta_path, "rb") as f:
        m = pickle.load(f)
    return int(m.get("bottleneck_dim", TCN_BOTTLENECK_DIM))


def _tcn_derived_feature_names(bottleneck_dim: int | None = None) -> list[str]:
    bd = int(bottleneck_dim) if bottleneck_dim is not None else _tcn_bottleneck_dim_from_meta()
    emb = [f"tcn_emb_{i}" for i in range(bd)]
    return TCN_REGIME_FUT_PROB_COLS + ["tcn_regime_fut_entropy"] + emb


def configure_compute_threads() -> None:
    """
    Conservative defaults to reduce CPU oversubscription (PyTorch + LightGBM OpenMP + BLAS).

    Env:
      TORCH_CPU_THREADS — PyTorch intra-op threads (default: ceil(n_cpu/2), capped at 8).
      LGBM_N_JOBS — passed to LightGBM train() (default: same cap rule, max 16).
    """
    n_cpu = max(1, os.cpu_count() or 4)
    default_cap = max(1, min(8, (n_cpu + 1) // 2))
    n = int(os.environ.get("TORCH_CPU_THREADS", str(default_cap)))
    n = max(1, min(n, n_cpu))
    try:
        torch.set_num_threads(n)
        inter = max(1, min(2, max(1, n // 4)))
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass


def _lgbm_n_jobs() -> int:
    n_cpu = max(1, os.cpu_count() or 4)
    default = max(1, min(16, (n_cpu + 1) // 2))
    return int(os.environ.get("LGBM_N_JOBS", str(default)))


def _tcn_inference_device() -> torch.device:
    forced = os.environ.get("TORCH_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ───────────────────────────────────────────────────────────────────────
# 1.  Data preparation
# ───────────────────────────────────────────────────────────────────────

def _load_and_compute_pa(symbol: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    raw = pd.read_csv(os.path.join(data_dir, f"{symbol}.csv"))
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw.sort_values("time_key").reset_index(drop=True)

    atr_1m = compute_atr(raw, length=14)
    print(f"  [{symbol}] Computing PA features on {len(raw):,} 1-min bars …")
    t0 = _time.time()
    df_pa = add_pa_features(raw, atr_1m, timeframe="5min")
    print(f"  [{symbol}] PA features done in {_time.time()-t0:.1f}s  → {df_pa.shape[1]} cols")
    return df_pa


def _load_labels(symbol: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    cols = [
        "time_key", "market_state",
        "signal", "outcome",
        "quality_bull_breakout", "quality_bear_breakout",
        "max_favorable", "max_adverse", "exit_bar",
        "atr",
    ]
    lbl = pd.read_csv(os.path.join(data_dir, f"{symbol}{LABELED_SUFFIX}.csv"), usecols=cols)
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


# Categorical regime tags from pa_rules._market_regime_5m / HTF merge — must never enter float matrices.
_LGBM_EXCLUDE_PA_STRING_COLS = frozenset({
    "pa_env_state",
    "pa_env_order_type",
    "pa_env_entry_style",
    "pa_env_stop_style",
    "pa_env_take_profit_style",
    "pa_env_position_size",
})


def _is_lgbm_string_tag_col(name: str) -> bool:
    if name in _LGBM_EXCLUDE_PA_STRING_COLS:
        return True
    if name.startswith("pa_htf_") and name.endswith("_state"):
        return True
    return False


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


def _numeric_feature_cols_for_matrix(df: pd.DataFrame, names: list[str]) -> list[str]:
    """Keep only columns present in df and numeric/bool dtypes (defensive for float32 matrices)."""
    keep: list[str] = []
    dropped: list[str] = []
    for c in names:
        if c not in df.columns:
            dropped.append(f"{c}<missing>")
            continue
        if _is_lgbm_string_tag_col(c):
            dropped.append(c)
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            keep.append(c)
        else:
            dropped.append(f"{c}<{s.dtype}>")
    if dropped:
        preview = dropped[:24]
        more = " …" if len(dropped) > 24 else ""
        print(
            f"  [LGBM] Excluded {len(dropped)} non-numeric / tag columns from matrix: {preview}{more}",
            flush=True,
        )
    return keep


def _unique_cols(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _require_lgb_matrix_matches_names(
    X: np.ndarray, feature_names: list[str], context: str
) -> None:
    """LightGBM requires len(feature_name) == num_feature; fail fast with a clear error."""
    n_x = int(X.shape[1]) if X.ndim == 2 else 0
    n_n = len(feature_names)
    if n_x != n_n:
        raise ValueError(
            f"{context}: matrix has {n_x} columns but feature_name has {n_n} entries."
        )
    seen: set[str] = set()
    dup: list[str] = []
    for c in feature_names:
        if c in seen:
            dup.append(c)
        else:
            seen.add(c)
    if dup:
        raise ValueError(
            f"{context}: duplicate feature names: {sorted(set(dup))!r}"
        )


def _lgbm_booster_feature_names(model: lgb.Booster) -> list[str]:
    """Columns in training order; required for chunked predict to match Layer 2a."""
    names = [str(x) for x in model.feature_name()]
    if not names:
        raise ValueError("LightGBM booster returned no feature names.")
    if names[0].startswith("Column_"):
        raise ValueError(
            "Regime booster has generic Column_* names; retrain Layer 2a with named features."
        )
    return names


def _split_feature_groups(feat_cols: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    hmm = [c for c in feat_cols if c.startswith("pa_hmm_")]
    garch = [c for c in feat_cols if c.startswith("pa_garch_")]
    tcn = [c for c in feat_cols if c.startswith("tcn_")]
    base = [c for c in feat_cols if c not in set(hmm + garch + tcn)]
    return base, hmm, garch, tcn


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
    Run order: train_tcn_pa_state.py first so tcn_meta.pkl + tcn_state_classifier.pt exist.
    """
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    model_path = os.path.join(MODEL_DIR, "tcn_state_classifier.pt")

    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise RuntimeError(
            f"Missing TCN checkpoint under {MODEL_DIR}: need tcn_meta.pkl and tcn_state_classifier.pt. "
            "Train first: python3 backtests/train_tcn_pa_state.py"
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
        model = PAStateTCN(
            input_size=input_size,
            num_channels=meta["num_channels"],
            kernel_size=meta["kernel_size"],
            dropout=0.0,
            bottleneck_dim=bottleneck_dim,
            num_classes=NUM_REGIME_CLASSES,
        )
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval()
        # Mac / CPU PyTorch Conv1d + large batches can SIGSEGV; keep CPU batches small unless overridden.
        _bs_default = "2048" if device.type in ("mps", "cuda") else "64"
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

            regime_probs_arr = np.full((n_bars, NUM_REGIME_CLASSES), np.nan, dtype=np.float32)
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
                    f"{oof_path} is missing 'regime_probs' (6-class). Retrain TCN: train_tcn_pa_state.py"
                )

            oof_df = pd.DataFrame({
                "symbol": oof["syms"],
                "time_key": pd.to_datetime(oof["ts"])
            })
            rp = np.asarray(oof["regime_probs"], dtype=np.float32)
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



TRAIN_END = "2023-01-01"
CAL_END = "2023-07-01"
TEST_END = "2025-01-01"
FAST_TRAIN_MODE = True
RNG = np.random.default_rng(42)


def _tq(it, **kwargs):
    """Progress bar. DISABLE_TQDM=1 forces off; FORCE_TQDM=1 forces on even when logging to a file."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return it
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return it
    return tqdm(it, **kwargs)


def _lgb_log_eval_period() -> int:
    """Valid-set log interval for LightGBM; 0 = omit log_evaluation. Override: LGBM_LOG_EVAL_PERIOD."""
    raw = os.environ.get("LGBM_LOG_EVAL_PERIOD", "").strip()
    if raw:
        return max(0, int(raw))
    return 100 if sys.stderr.isatty() else 0


def _lgb_train_callbacks(early_stopping_rounds: int) -> list:
    p = _lgb_log_eval_period()
    out = []
    if p > 0:
        out.append(lgb.log_evaluation(p))
    out.append(lgb.early_stopping(early_stopping_rounds))
    return out


def _lgb_round_tqdm_enabled() -> bool:
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("LGBM_DISABLE_ROUND_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return False
    return True


def _lgb_train_callbacks_with_round_tqdm(
    early_stopping_rounds: int,
    num_boost_round: int,
    tqdm_desc: str,
) -> tuple[list, list]:
    """LightGBM callbacks plus per-boosting-round tqdm. Returns (callbacks, cleanup_fns)."""
    base = _lgb_train_callbacks(early_stopping_rounds)
    if not _lgb_round_tqdm_enabled() or num_boost_round <= 0:
        return base, []

    bar = tqdm(
        total=num_boost_round,
        desc=tqdm_desc,
        unit="round",
        leave=False,
        mininterval=0.2,
        file=sys.stderr,
    )

    def _round_cb(env) -> None:
        bar.update(1)
        er = getattr(env, "evaluation_result_list", None) or []
        if er:
            parts: list[str] = []
            for tup in er:
                if len(tup) >= 3:
                    parts.append(f"{tup[1]}={float(tup[2]):.4f}")
            if parts:
                bar.set_postfix_str(" ".join(parts[:3]), refresh=False)

    def _cleanup() -> None:
        bar.close()

    return [_round_cb] + base, [_cleanup]


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

    base_feats, hmm_feats, garch_feats, tcn_feats = _split_feature_groups(feat_cols)
    print(f"\n  Feature columns: {len(feat_cols)}")
    print(f"    Base PA/OR:  {len(base_feats)}")
    print(f"    HMM-style:   {len(hmm_feats)}")
    print(f"    GARCH-style: {len(garch_feats)}")
    print(f"    TCN-derived: {len(tcn_feats)}")
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


# ───────────────────────────────────────────────────────────────────────
# 2.  Layer 2a — 6-class regime head (y = current state_label; X = PA/HMM/GARCH only)
# ───────────────────────────────────────────────────────────────────────


def _compute_sample_weights(y: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Class-balanced weights with temporal recency boost."""
    n = len(y)
    class_frequencies = np.bincount(y, minlength=NUM_REGIME_CLASSES).astype(float)
    class_w = np.zeros(NUM_REGIME_CLASSES, dtype=float)
    for c in range(NUM_REGIME_CLASSES):
        if class_frequencies[c] <= 0:
            continue
        freq = class_frequencies[c] / n
        class_w[c] = 1.0 / max(freq * NUM_REGIME_CLASSES, 1e-6)
    active = class_w > 0
    if active.any():
        class_w[active] /= class_w[active].mean()
    else:
        class_w[:] = 1.0

    weights = np.array([class_w[label] for label in y], dtype=float)

    ts = pd.to_datetime(timestamps)
    days_from_end = (ts.max() - ts).total_seconds() / 86400
    max_days = days_from_end.max()
    recency = 0.7 + 0.3 * (1.0 - days_from_end / max(max_days, 1))
    weights *= recency.values

    return weights


def _optuna_search_params(
    X_train, y_train, w_train,
    X_cal, y_cal,
    feat_cols: list[str],
    base_params: dict,
    n_trials: int = 30,
) -> dict:
    """Restricted Optuna search for sensitive hyperparameters to prevent overfitting."""
    mode_name = "FAST" if FAST_TRAIN_MODE else "FULL"
    print(f"\n  Optuna search over {n_trials} trials ({mode_name} mode):")
    _require_lgb_matrix_matches_names(X_train, feat_cols, "Layer 2a Optuna search")

    # Limit search space for faster processing when FAST_TRAIN_MODE is enabled
    actual_trials = min(10, n_trials) if FAST_TRAIN_MODE else n_trials

    # We evaluate on Cal, so we want to be very careful not to overfit it
    # Use early stopping but fewer rounds to speed up the search
    optuna_rounds = 800 if FAST_TRAIN_MODE else 2000
    optuna_es = 40 if FAST_TRAIN_MODE else 60

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,
                             feature_name=feat_cols, free_raw_data=False)
    valid_data = lgb.Dataset(X_cal, label=y_cal,
                             feature_name=feat_cols, free_raw_data=False)

    def objective(trial):
        p = base_params.copy()
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.1, log=True)
        # Lock all other parameters to base_params to prevent small-data noise

        model = lgb.train(
            p, train_data, num_boost_round=optuna_rounds, valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(optuna_es, verbose=False),
            ],
        )
        if model.best_iteration < 10:
            raise optuna.TrialPruned()
        return model.best_score["valid_0"]["multi_logloss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  → Best Optuna params (n={actual_trials}): num_leaves={best['num_leaves']}  "
          f"learning_rate={best['learning_rate']:.4f}  logloss={study.best_value:.5f}")

    return {**base_params, **best}


def _regime_lgbm_feature_cols(feat_cols: list[str]) -> list[str]:
    """Layer-2a regime head: PA + GARCH only (no TCN, no HMM columns).

    state_label is derived from pa_hmm_state / HMM softmax; including pa_hmm_* in X is label leakage
    (trees memorize argmax). Exclude the full HMM block so L2a learns regime from causal price/vol features.
    """
    return [
        c
        for c in feat_cols
        if not c.startswith("tcn_") and not c.startswith("pa_hmm_")
    ]


def train_regime_classifier(df: pd.DataFrame, feat_cols: list[str]):
    print("\n" + "=" * 70)
    print("  LAYER 2a: 6-Class Regime Head (y = current market_state, no forward shift)")
    print(
        "  Predict: Bull/Bear/Range × Conv/Div | X = PA + GARCH only "
        "(no TCN; no pa_hmm_* — y matches HMM argmax, those cols would leak the label)"
    )
    print("=" * 70)

    raw_regime_feats = _regime_lgbm_feature_cols(feat_cols)
    pa_only_feats = _numeric_feature_cols_for_matrix(df, raw_regime_feats)
    if not pa_only_feats:
        raise ValueError("Layer 2a: no numeric PA features left after filtering string/tag cols.")

    X = df[pa_only_feats].values.astype(np.float32)
    y = df["state_label"].values.astype(int)
    t = df["time_key"].values

    train_mask = t < np.datetime64(TRAIN_END)
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    w_train = _compute_sample_weights(y_train, t[train_mask])

    print(f"  Dates — Train: < {TRAIN_END} | Cal: → {CAL_END} | Test: → {TEST_END}")

    base_params = {
        "objective": "multiclass",
        "num_class": NUM_REGIME_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.3,
        "lambda_l2": 2.0,
        "min_gain_to_split": 0.01,
        "path_smooth": 1.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
    }

    print(f"  Train: {len(y_train):,}  |  Cal: {len(y_cal):,}  |  Test: {len(y_test):,}")

    best_combo = _optuna_search_params(
        X_train, y_train, w_train, X_cal, y_cal, pa_only_feats, base_params, n_trials=30
    )
    params = {**base_params, **best_combo}

    print(f"\n  Training final model with best params (lr=0.01, colsample=0.7)…")
    _require_lgb_matrix_matches_names(X_train, pa_only_feats, "Layer 2a final train")
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,
                             feature_name=pa_only_feats, free_raw_data=False)
    valid_data = lgb.Dataset(X_cal, label=y_cal,
                             feature_name=pa_only_feats, free_raw_data=False)

    # Imbalanced 1-min labels: allow convergence (2000 rounds often still improving).
    state_rounds = 5000 if FAST_TRAIN_MODE else 6000
    state_es = 150
    model = lgb.train(
        params,
        train_data,
        num_boost_round=state_rounds,
        valid_sets=[valid_data],
        callbacks=_lgb_train_callbacks(state_es),
    )

    # ── Isotonic calibration per class on the calibration set ──
    # Index c == L2a class == STATE_NAMES[c] == REGIMES_6[c] (same axis as model.predict columns).
    raw_cal_probs = model.predict(X_cal)   # (n_cal, NUM_REGIME_CLASSES)
    calibrators = []
    for c in range(NUM_REGIME_CLASSES):
        y_binary = (y_cal == c).astype(int)
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(raw_cal_probs[:, c], y_binary)
        calibrators.append(iso)

    def predict_calibrated(X_in):
        raw = model.predict(X_in)
        cal = np.column_stack([calibrators[c].predict(raw[:, c]) for c in range(NUM_REGIME_CLASSES)])
        cal /= cal.sum(axis=1, keepdims=True)  # re-normalise to sum=1
        return cal

    # ── Evaluate on held-out test set ──
    probs = predict_calibrated(X_test)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, probs, labels=list(range(NUM_REGIME_CLASSES)))
    target_names = [STATE_NAMES[i] for i in range(NUM_REGIME_CLASSES)]

    print(f"\n  Accuracy (all): {acc:.4f}  |  Log-loss: {ll:.4f}")
    print(f"\n  Classification Report (all bars):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(NUM_REGIME_CLASSES)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_REGIME_CLASSES)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())

    # ── Confidence-filtered accuracy ──
    print(f"\n  Confidence-Filtered Accuracy:")
    print(f"  {'Threshold':>10s}  {'Acc':>7s}  {'Coverage':>9s}  {'Bars':>9s}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = confidence >= thr
        if mask.sum() > 0:
            acc_f = accuracy_score(y_test[mask], y_pred[mask])
            cov = mask.mean()
            print(f"  {thr:>10.2f}  {acc_f:>7.4f}  {cov:>9.1%}  {mask.sum():>9,}")

    # ── Per-class probability calibration ──
    print(f"\n  Per-Class Probability Calibration (10 bins):")
    for c in range(NUM_REGIME_CLASSES):
        y_bin = (y_test == c).astype(int)
        p_c = probs[:, c]
        frac_pos, mean_pred = calibration_curve(y_bin, p_c, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_bin, p_c)
        print(f"\n    {STATE_NAMES[c]} (Brier={brier:.4f}):")
        print(f"    {'Predicted':>10s}  {'Actual':>8s}")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    {mp:>10.3f}  {fp:>8.3f}")

    # ── Feature importance ──
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": pa_only_feats, "importance": importance}).sort_values("importance", ascending=False)
    print(f"\n  Top 25 features (gain):")
    print(imp_df.head(25).to_string(index=False))

    # ── Save ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, STATE_CLASSIFIER_FILE))
    import pickle
    with open(os.path.join(MODEL_DIR, "state_calibrators.pkl"), "wb") as f:
        pickle.dump(calibrators, f)
    print(f"\n  Model saved → {MODEL_DIR}/{STATE_CLASSIFIER_FILE}")
    print(f"  Calibrators saved → {MODEL_DIR}/state_calibrators.pkl")

    return model, calibrators, imp_df


# Backward-compatible name
train_state_classifier = train_regime_classifier


# ───────────────────────────────────────────────────────────────────────
# 3.  Layer 2b — Joint Discrete Trade-Quality Classification (6 classes)
# ───────────────────────────────────────────────────────────────────────

QUALITY_CLASS_NAMES = {
    0: "A_LONG",
    1: "B_LONG",
    2: "NEUTRAL",
    3: "CHOP",
    4: "B_SHORT",
    5: "A_SHORT",
}

QUALITY_CLASS_ORDER = [QUALITY_CLASS_NAMES[i] for i in range(6)]
TRADABLE_CLASS_IDS = {0, 1, 4, 5}

# Layer 3 — curated causal PA columns (first N present in frame are used, max ~15).
LAYER3_PA_KEY_FEATURES: tuple[str, ...] = (
    "pa_ma_slope_ema20",
    "pa_ma_compress",
    "pa_sr_position",
    "pa_vol_rvol",
    "pa_vol_trend",
    "pa_vol_vwap_dist",
    "pa_lead_macd_hist_slope",
    "pa_lead_bias_5",
    "pa_lead_rsi_14",
    "pa_struct_score",
    "pa_pressure_score",
    "pa_trend_alignment",
    "pa_range_compression",
    "pa_breakout_strength",
    "pa_pullback_depth",
)

EXECUTION_SIZER_GATE_FILE = "execution_sizer_gate.txt"
EXECUTION_SIZER_SIZE_FILE = "execution_sizer_size.txt"
EXECUTION_SIZER_LEGACY_V1_FILE = "execution_sizer_v1.txt"


def _mfe_mae_atr_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same scaling as trade-quality KMeans targets (MFE/MAE in ATR units)."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    return mfe.astype(np.float64), mae.astype(np.float64)


def _opp_regression_sample_weights(mfe_tgt: np.ndarray, regime_name: str) -> np.ndarray:
    w = np.ones(len(mfe_tgt), dtype=np.float64)
    w[mfe_tgt > 1.0] = 5.0
    w[mfe_tgt > 2.0] = 15.0
    if regime_name.startswith("range"):
        w[mfe_tgt < 0.2] = 0.3
    return w


def _l2b_reg_objective_params() -> dict:
    obj = os.environ.get("L2B_REG_OBJECTIVE", "regression").strip().lower()
    if obj == "huber":
        return {"objective": "huber", "alpha": 0.9}
    return {"objective": "regression", "metric": "mae"}


def _train_regime_opp_regression_models(
    X: np.ndarray,
    state_label: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    train_mask: np.ndarray,
    cal_mask: np.ndarray,
    all_bo_feats: list[str],
    regime_cal_probs_cal: np.ndarray,
    y_trade_cal: np.ndarray,
) -> tuple[dict[str, dict[str, lgb.Booster]], dict[str, float], np.ndarray]:
    """Train up to 6× (MFE + MAE) regressors (one pair per REGIMES_6 name); tune thresholds on cal.

    **Training rows** for regime ``REGIMES_6[k]``: ground-truth ``state_label == k`` (supervision).

    **Cal / inference routing**: pick head with ``argmax(regime_now probabilities) == k``; then
    ``score = pred_mfe / (pred_mae + 0.1)`` on those rows only; F1 grid-search yields ``thr_vec[k]``.

    Six regimes ⇒ fewer routed cal rows per bucket than old 3-way groups — default min row
    count is lower (``L2B_OPP_CAL_MIN_ROWS``).
    """
    reg_rounds = 800 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ROUNDS", "2000"))
    reg_es = 80 if FAST_TRAIN_MODE else int(os.environ.get("L2B_REG_ES", "150"))
    min_prec = float(os.environ.get("L2B_OPP_MIN_PRECISION", "0.30"))
    min_cal_for_opp_thr = int(os.environ.get("L2B_OPP_CAL_MIN_ROWS", "150"))
    base_reg = {
        "boosting_type": "gbdt",
        "learning_rate": float(os.environ.get("L2B_REG_LR", "0.01")),
        "num_leaves": int(os.environ.get("L2B_REG_NUM_LEAVES", "63")),
        "max_depth": int(os.environ.get("L2B_REG_MAX_DEPTH", "7")),
        "min_child_samples": int(os.environ.get("L2B_REG_MIN_CHILD", "100")),
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        **_l2b_reg_objective_params(),
    }
    cb = _lgb_train_callbacks(reg_es)
    models: dict[str, dict[str, lgb.Booster]] = {}

    X_tr = X[train_mask]
    X_ca = X[cal_mask]
    st_tr = state_label[train_mask]
    st_ca = state_label[cal_mask]
    mfe_tr = mfe[train_mask]
    mae_tr = mae[train_mask]
    mfe_ca = mfe[cal_mask]
    mae_ca = mae[cal_mask]

    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        mtr = st_tr == argmax_idx
        mca = st_ca == argmax_idx
        ntr, nca = int(mtr.sum()), int(mca.sum())
        print(f"  [L2b regression] {predicted_regime}: train_rows={ntr:,}  cal_rows(early_stop)={nca:,}")
        if ntr < 2000:
            print(f"    [warn] {predicted_regime}: too few train rows — skipping pair (needs ≥2000).")
            continue
        idx_sub = np.where(mtr)[0]
        X_g = X_tr[idx_sub]
        y_mfe_g = mfe_tr[idx_sub]
        y_mae_g = mae_tr[idx_sub]
        w_base = _opp_regression_sample_weights(y_mfe_g, predicted_regime)

        if nca >= 200:
            idx_va = np.where(mca)[0]
            X_va = X_ca[idx_va]
            y_mfe_va = mfe_ca[idx_va]
            y_mae_va = mae_ca[idx_va]
        else:
            tail = min(5000, ntr)
            X_va = X_g[-tail:]
            y_mfe_va = y_mfe_g[-tail:]
            y_mae_va = y_mae_g[-tail:]

        d_mfe_tr = lgb.Dataset(X_g, label=y_mfe_g, weight=w_base, feature_name=all_bo_feats, free_raw_data=False)
        d_mfe_va = lgb.Dataset(X_va, label=y_mfe_va, feature_name=all_bo_feats, free_raw_data=False)
        print(f"    [L2b regression] {predicted_regime}: train MFE head …", flush=True)
        m_mfe = lgb.train(
            base_reg, d_mfe_tr, num_boost_round=reg_rounds, valid_sets=[d_mfe_va], callbacks=cb,
        )
        w_mae = _opp_regression_sample_weights(y_mfe_g, predicted_regime)
        d_mae_tr = lgb.Dataset(X_g, label=y_mae_g, weight=w_mae, feature_name=all_bo_feats, free_raw_data=False)
        d_mae_va = lgb.Dataset(X_va, label=y_mae_va, feature_name=all_bo_feats, free_raw_data=False)
        print(f"    [L2b regression] {predicted_regime}: train MAE head …", flush=True)
        m_mae = lgb.train(
            base_reg, d_mae_tr, num_boost_round=reg_rounds, valid_sets=[d_mae_va], callbacks=cb,
        )
        models[predicted_regime] = {"mfe": m_mfe, "mae": m_mae}

    if not models:
        raise RuntimeError("Regime opportunity regression: no group had enough data.")

    # L2a routing: argmax on cal probs → class index k == predicted_regime REGIMES_6[k]
    st_cal_pred = np.argmax(regime_cal_probs_cal, axis=1)
    gix_cal = st_cal_pred.astype(np.int64, copy=False)
    n_cal = len(y_trade_cal)
    opp_cal = np.zeros(n_cal, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = gix_cal == argmax_idx
        if not m.any():
            continue
        mfe_p = models[predicted_regime]["mfe"].predict(X_ca[m])
        mae_p = models[predicted_regime]["mae"].predict(X_ca[m])
        mfe_p = np.clip(mfe_p, 0.0, None)
        mae_p = np.clip(mae_p, 0.01, None)
        opp_cal[m] = np.log1p(mfe_p) - np.log1p(mae_p)

    opp_nonzero = opp_cal[opp_cal != 0]
    if len(opp_nonzero) > 0:
        print(f"  [L2b regression] opportunity(log-ratio) dist (cal): "
              f"min={np.min(opp_nonzero):.3f} 25%={np.percentile(opp_nonzero, 25):.3f} "
              f"median={np.median(opp_nonzero):.3f} 75%={np.percentile(opp_nonzero, 75):.3f} "
              f"max={np.max(opp_nonzero):.3f}")

    opp_thr: dict[str, float] = {}
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        m = gix_cal == argmax_idx
        n_routed = int(m.sum())
        if predicted_regime not in models or n_routed < min_cal_for_opp_thr:
            opp_thr[predicted_regime] = float(os.environ.get(f"L2B_OPP_THR_{predicted_regime.upper()}", "0.0"))
            why = "no model" if predicted_regime not in models else f"routed cal n={n_routed} < {min_cal_for_opp_thr}"
            print(
                f"  [L2b regression] {predicted_regime}: fallback opp_thr={opp_thr[predicted_regime]:.2f} ({why})",
            )
            continue
        y_true = y_trade_cal[m]
        o = opp_cal[m]
        best_f1, best_thr, best_row = 0.0, 0.0, None
        for thr in np.arange(-1.0, 3.0, 0.1):
            pred = (o >= thr).astype(int)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            f1v = f1_score(y_true, pred, zero_division=0)
            if prec >= min_prec and f1v > best_f1:
                best_f1, best_thr = f1v, float(thr)
                best_row = (prec, rec, f1v)
        if best_row is None:
            best_thr = float(os.environ.get(f"L2B_OPP_THR_{predicted_regime.upper()}", "0.0"))
            print(
                f"  [L2b regression] {predicted_regime}: no thr met prec>={min_prec}; fallback thr={best_thr:.2f}",
            )
        else:
            prec, rec, f1v = best_row
            print(
                f"  [L2b regression] {predicted_regime}: opp_thr={best_thr:.2f}  "
                f"F1={f1v:.4f}  prec={prec:.3f}  rec={rec:.3f}  (cal, routed by L2a)",
            )
        opp_thr[predicted_regime] = best_thr

    thr_vec = np.array([opp_thr.get(g, 1.0) for g in REGIMES_6], dtype=np.float64)
    return models, opp_thr, thr_vec


def _l2b_nested_opp_models(regb: dict) -> dict[str, dict[str, lgb.Booster]]:
    """``step1_regression`` bundle: flat ``{regime}_mfe`` / ``{regime}_mae`` → nested routing dict."""
    out: dict[str, dict[str, lgb.Booster]] = {}
    for regime in REGIMES_6:
        km, ka = f"{regime}_mfe", f"{regime}_mae"
        if km in regb and ka in regb:
            out[regime] = {"mfe": regb[km], "mae": regb[ka]}
    return out


def _compute_opportunity_triplet(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Routed dual-head: pred_mfe, pred_mae, opportunity = mfe/(mae+0.1)."""
    # st[i] = L2a argmax index k → use models[REGIMES_6[k]] (same k as REGIME_NOW_PROB_COLS[k]).
    st = np.argmax(regime_probs, axis=1).astype(np.int64, copy=False)
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = st == argmax_idx
        if not m.any():
            continue
        mf = models[predicted_regime]["mfe"].predict(X[m])
        ma = models[predicted_regime]["mae"].predict(X[m])
        mf = np.clip(mf, 0.0, None)
        ma = np.clip(ma, 0.01, None)
        opp[m] = np.log1p(mf) - np.log1p(ma)
        mfe_p[m] = mf
        mae_p[m] = ma
    return opp, mfe_p, mae_p


def _compute_opportunity_scores(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> np.ndarray:
    o, _, _ = _compute_opportunity_triplet(X, regime_probs, models)
    return o


def _l2b_triplet_from_trade_prob(p_trade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary Step1 fallback: synthetic MFE/MAE/opp from p_trade (no label leakage)."""
    pt = np.clip(p_trade.astype(np.float64), 0.0, 1.0)
    mf = np.clip(2.0 * pt, 0.0, 5.0)
    ma = np.clip(0.5 + 0.5 * (1.0 - pt), 0.01, 4.0)
    opp = np.log1p(mf) - np.log1p(ma)
    return opp, mf, ma


def _layer3_fill_l2b_triplet_arrays(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_trade: np.ndarray,
    opp_out: np.ndarray,
    mfe_out: np.ndarray,
    mae_out: np.ndarray,
    chunk: int,
) -> None:
    """Fill L2b regression outputs for Layer 3 (chunked). Uses Step1 regression if available."""
    regb = trade_quality_models.get("step1_regression")
    n = len(work)
    regime_mat = work[list(REGIME_NOW_PROB_COLS)].to_numpy(dtype=np.float32, copy=False)
    if regb:
        models = _l2b_nested_opp_models(regb)
        n_chunk = (n + chunk - 1) // chunk
        for i in _tq(range(0, n, chunk), desc="Layer3 L2b triplet (reg)", total=n_chunk, unit="chunk"):
            j = min(i + chunk, n)
            x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
            rp = regime_mat[i:j]
            o, mf, ma = _compute_opportunity_triplet(x_b, rp, models)
            opp_out[i:j] = o.astype(np.float32)
            mfe_out[i:j] = mf.astype(np.float32)
            mae_out[i:j] = ma.astype(np.float32)
        return
    o, mf, ma = _l2b_triplet_from_trade_prob(p_trade)
    opp_out[:] = o.astype(np.float32)
    mfe_out[:] = mf.astype(np.float32)
    mae_out[:] = ma.astype(np.float32)


def _opp_to_synthetic_p_trade(opp: np.ndarray, thr_row: np.ndarray, kappa: float = 4.0) -> np.ndarray:
    """Map opportunity score to (0,1) so Layer3 / reconstruct can keep thr_trade=0.5."""
    z = np.clip(kappa * (opp - thr_row), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _layer3_fill_p_trade_from_regression(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_a: np.ndarray,
    chunk: int,
) -> None:
    s2 = trade_quality_models["step2"]
    s3 = trade_quality_models["step3"]
    regb = trade_quality_models["step1_regression"]
    models = _l2b_nested_opp_models(regb)
    thr_vec = regb["thr_vec"]
    regime_mat = work[list(REGIME_NOW_PROB_COLS)].to_numpy(dtype=np.float32, copy=False)
    n = len(work)
    n_chunk = (n + chunk - 1) // chunk
    for i in _tq(range(0, n, chunk), desc="Layer3 trade stack (reg gate)", total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_b = work[layer2_feats].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
        rp = regime_mat[i:j]
        opp = _compute_opportunity_scores(x_b, rp, models)
        gix = np.argmax(rp, axis=1).astype(np.int64, copy=False)  # per-row L2a argmax → REGIMES_6[gix]
        thr_row = thr_vec[gix]
        p_trade[i:j] = _opp_to_synthetic_p_trade(opp, thr_row)
        p_long[i:j] = s2.predict(x_b)
        p_a[i:j] = s3.predict(x_b)
        del x_b, rp, opp


def compute_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """12 breakout-specific features computable for any bar (causal).
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
    }, index=df.index)
    return out


BO_FEAT_COLS = [
    "bo_body_atr", "bo_range_atr", "bo_vol_spike", "bo_close_extremity",
    "bo_wick_imbalance", "bo_range_compress", "bo_body_growth",
    "bo_gap_signal", "bo_consec_dir", "bo_inside_prior",
    "bo_pressure_diff", "bo_or_dist",
]

# Calibrated "regime now" probabilities (Layer 2a). Not in Layer 2b X; used for reconstruct + Layer 3.
# (REGIME_PROB_COLS defined with REGIME_NOW_PROB_COLS at module top.)


def _print_quality_label_outcome_stats(df: pd.DataFrame, y6: np.ndarray) -> None:
    """Mean MFE/ATR, MAE/ATR, RR by KMeans-derived quality class (A/B sanity check)."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae_arr = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    rr = mfe / np.maximum(mae_arr, 0.1)
    print("  Outcome stats by quality label (MFE/ATR, MAE/ATR, RR):")
    for c in range(6):
        sel = y6 == c
        if not sel.any():
            continue
        print(
            f"    {QUALITY_CLASS_NAMES[c]:>8s}: n={int(sel.sum()):>9,}  "
            f"mfe={mfe[sel].mean():.3f}  mae={mae_arr[sel].mean():.3f}  rr={rr[sel].mean():.3f}"
        )
    for a, b, name in [(0, 1, "A_LONG vs B_LONG"), (5, 4, "A_SHORT vs B_SHORT"), (2, 3, "NEUTRAL vs CHOP")]:
        ma, mb = y6 == a, y6 == b
        if ma.sum() and mb.sum():
            dmfe = mfe[ma].mean() - mfe[mb].mean()
            drr = rr[ma].mean() - rr[mb].mean()
            print(f"    Δ({name}): Δmfe={dmfe:+.3f}  Δrr={drr:+.3f}")


def _build_trade_quality_targets(df: pd.DataFrame) -> np.ndarray:
    """
    Build 6-class joint discrete labels using Unsupervised Clustering (K-Means)
    on outcome metrics (MFE/ATR, MAE/ATR, RR, log1p(Hold_Time)) for breakouts.
    This replaces the rule-based approach with data-driven statistical groupings.
    """
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    rr = mfe / np.maximum(mae, 0.1)
    hold_time = np.maximum(df["exit_bar"].fillna(0).values.astype(float), 0.0)
    # Right-skewed hold lengths: log1p before Z-score so K-Means isn't dominated by rare long holds
    log_hold_time = np.log1p(hold_time)

    qbull = df["quality_bull_breakout"].fillna(0).values.astype(int)
    qbear = df["quality_bear_breakout"].fillna(0).values.astype(int)
    state = df["state_label"].fillna(2).values.astype(int)

    y = np.full(len(df), 2, dtype=int)  # default NEUTRAL

    both_breakout = (qbull == 1) & (qbear == 1)
    y[both_breakout] = 3  # CHOP

    long_mask = (qbull == 1) & (qbear == 0)
    short_mask = (qbear == 1) & (qbull == 0)
    print(
        f"  [L2b labels] KMeans setup — long={int(long_mask.sum()):,} short={int(short_mask.sum()):,} rows …",
        flush=True,
    )

    def _cluster_and_assign(mask: np.ndarray, is_long: bool):
        indices = np.where(mask)[0]
        if len(indices) < 3:
            # Fallback if too few samples
            y[mask] = 1 if is_long else 4
            return

        # Features: [MFE, MAE, RR, log1p(Hold_Time)] then Z-score below
        X_cluster = np.column_stack([
            mfe[indices],
            mae[indices],
            rr[indices],
            log_hold_time[indices],
        ])

        # Z-score normalization for clustering
        X_mean = X_cluster.mean(axis=0)
        X_std = X_cluster.std(axis=0) + 1e-6
        X_scaled = (X_cluster - X_mean) / X_std

        # K-Means clustering into 3 classes (A, B, CHOP)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Interpret clusters based on heuristic score: MFE + RR - MAE
        cluster_scores = []
        for c in range(3):
            c_mask = (clusters == c)
            if not c_mask.any():
                cluster_scores.append(-1000)
                continue
            c_mfe = mfe[indices][c_mask].mean()
            c_mae = mae[indices][c_mask].mean()
            c_rr = rr[indices][c_mask].mean()
            score = c_mfe + c_rr - c_mae
            cluster_scores.append(score)

        # Sort clusters by score descending
        ranked_clusters = np.argsort(cluster_scores)[::-1]
        
        # Best score -> A grade
        # Middle score -> B grade
        # Lowest score -> CHOP
        grade_a_cluster = ranked_clusters[0]
        grade_b_cluster = ranked_clusters[1]
        chop_cluster = ranked_clusters[2]

        label_a = 0 if is_long else 5
        label_b = 1 if is_long else 4
        label_chop = 3

        for c, label in [(grade_a_cluster, label_a), 
                         (grade_b_cluster, label_b), 
                         (chop_cluster, label_chop)]:
            c_idx = indices[clusters == c]
            y[c_idx] = label

    # Apply clustering separately for longs and shorts
    if long_mask.any():
        print(
            f"  [L2b labels] KMeans long breakouts: n={int(long_mask.sum()):,} …",
            flush=True,
        )
        _cluster_and_assign(long_mask, is_long=True)
    if short_mask.any():
        print(
            f"  [L2b labels] KMeans short breakouts: n={int(short_mask.sum()):,} …",
            flush=True,
        )
        _cluster_and_assign(short_mask, is_long=False)

    # Non-breakout bars in trend regimes may still offer weak directional quality.
    no_breakout = (qbull == 0) & (qbear == 0)
    trend_long_weak = no_breakout & (state == 0) & (mfe >= 0.35) & (rr >= 0.40)
    trend_short_weak = no_breakout & (state == 1) & (mfe >= 0.35) & (rr >= 0.40)
    y[trend_long_weak] = 1
    y[trend_short_weak] = 4
    range_state = np.isin(state, (4, 5))
    y[no_breakout & range_state] = 3
    y[no_breakout & ~range_state & ~(trend_long_weak | trend_short_weak)] = 2
    return y


def _binary_weights(y: np.ndarray, timestamps: np.ndarray, pos_boost: float = 1.0) -> np.ndarray:
    """Balanced weights for binary tasks with recency adjustment."""
    ts = pd.to_datetime(timestamps)
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(n - pos, 1)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    w = np.where(y == 1, w_pos * pos_boost, w_neg).astype(float)

    days_from_end = (ts.max() - ts).total_seconds() / 86400
    max_days = max(days_from_end.max(), 1.0)
    recency = 0.85 + 0.15 * (1.0 - days_from_end / max_days)
    w *= recency.values
    return w


def _reconstruct_quality_classes(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_a: np.ndarray,
    p_range_mass: np.ndarray,
    thr_trade: float = 0.55,
    thr_long: float = 0.50,
    thr_a: float = 0.50,
) -> np.ndarray:
    """Map hierarchical binary outputs back into 6-class trade-quality space.

    ``p_range_mass`` must be the summed L2a probability for range regimes only
    (indices ``RANGE_REGIME_INDICES``, i.e. ``range_conv`` + ``range_div`` in ``REGIMES_6``),
    not a 3-way bull/bear/range bucket and not the sum of all six probs.
    """
    pred = np.full(len(p_trade), 2, dtype=int)  # NEUTRAL default
    skip = p_trade < thr_trade
    pred[skip & (p_range_mass >= 0.50)] = 3  # CHOP when range_* mass is high
    pred[skip & (p_range_mass < 0.50)] = 2   # NEUTRAL when bull/bear mass dominates

    trade = ~skip
    is_long = p_long >= thr_long
    is_a = p_a >= thr_a
    pred[trade & is_long & is_a] = 0
    pred[trade & is_long & ~is_a] = 1
    pred[trade & ~is_long & ~is_a] = 4
    pred[trade & ~is_long & is_a] = 5
    return pred


def train_trade_quality_classifier(
    df: pd.DataFrame,
    feat_cols: list[str],
    regime_model: lgb.Booster,
    regime_calibrators: list,
):
    print("\n" + "=" * 70)
    print("  LAYER 2b: Hierarchical Trade-Quality Stack (regression Step1)")
    print("  y = trade_quality (KMeans outcomes); X excludes regime probabilities")
    print("  Step1 6-regime opp gate  |  Step2 LONG/SHORT  |  Step3 A/B")
    print("=" * 70)

    print(
        f"  [L2b prep] Copying dataframe ({len(df):,} rows) and building labels …",
        flush=True,
    )
    work = df.copy()
    work["trade_quality_label"] = _build_trade_quality_targets(work)

    print("  Label distribution (full):")
    for c in range(6):
        cnt = (work["trade_quality_label"] == c).sum()
        pct = 100.0 * cnt / max(len(work), 1)
        print(f"    {QUALITY_CLASS_NAMES[c]:>8s}: {cnt:>9,}  ({pct:>5.2f}%)")
    _print_quality_label_outcome_stats(work, work["trade_quality_label"].values)

    # Breakout-context features are computed for all bars, so NEUTRAL/CHOP are covered too.
    print("  [L2b prep] Breakout / bar-context features …", flush=True)
    if "symbol" in work.columns:
        bo_parts: list[pd.DataFrame] = []
        grps = list(work.groupby("symbol", sort=False))
        for _, g in _tq(
            grps,
            desc="  L2b BO feats",
            total=len(grps),
            unit="sym",
            leave=False,
        ):
            bo_parts.append(compute_breakout_features(g))
        bo_feats = pd.concat(bo_parts)
        bo_feats = bo_feats.loc[work.index]
    else:
        bo_feats = compute_breakout_features(work)
    for c in _tq(BO_FEAT_COLS, desc="  L2b BO → cols", leave=False, unit="col"):
        work[c] = bo_feats[c].values

    print(
        "  [L2b prep] Regime head predict + per-class calibration (all bars) …",
        flush=True,
    )
    regime_X_cols = _numeric_feature_cols_for_matrix(
        work, _regime_lgbm_feature_cols(feat_cols)
    )
    if not regime_X_cols:
        raise ValueError("Layer 2b prep: no numeric columns for regime head predict.")
    X_state = work[regime_X_cols].values.astype(np.float32)
    n_l2b = len(work)
    l2b_chunk = _layer3_chunk_rows()
    n_l2b_b = (n_l2b + l2b_chunk - 1) // l2b_chunk
    raw_regime = np.empty((n_l2b, NUM_REGIME_CLASSES), dtype=np.float64)
    for i in _tq(
        range(0, n_l2b, l2b_chunk),
        desc="  L2b regime raw",
        total=n_l2b_b,
        unit="chunk",
        leave=False,
    ):
        j = min(i + l2b_chunk, n_l2b)
        raw_regime[i:j] = regime_model.predict(X_state[i:j])
    cal_regime = np.empty((n_l2b, NUM_REGIME_CLASSES), dtype=np.float64)
    for i in _tq(
        range(0, n_l2b, l2b_chunk),
        desc="  L2b regime cal",
        total=n_l2b_b,
        unit="chunk",
        leave=False,
    ):
        j = min(i + l2b_chunk, n_l2b)
        row = raw_regime[i:j]
        cal_blk = np.column_stack([
            regime_calibrators[c].predict(row[:, c]) for c in range(NUM_REGIME_CLASSES)
        ])
        cal_blk = np.maximum(cal_blk, 1e-12)
        cal_blk /= cal_blk.sum(axis=1, keepdims=True)
        cal_regime[i:j] = cal_blk

    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        work[col] = cal_regime[:, j]
    work["regime_now_conf"] = cal_regime.max(axis=1)

    print("  [L2b prep] Resolving feature groups (PA / HMM / GARCH / TCN) …", flush=True)
    garch_cols = [
        c for c in work.columns
        if "garch" in c.lower() and str(work[c].dtype) not in {"object", "category"}
    ]
    hmm_cols = [
        c for c in work.columns
        if c.startswith("pa_hmm_") and str(work[c].dtype) not in {"object", "category"}
    ]

    all_bo_feats_raw = _unique_cols(feat_cols + BO_FEAT_COLS + sorted(garch_cols))
    all_bo_feats = _numeric_feature_cols_for_matrix(work, all_bo_feats_raw)
    if not all_bo_feats:
        raise ValueError("Layer 2b: no numeric features left after filtering.")
    pa_base, pa_hmm, pa_garch, pa_tcn = _split_feature_groups(feat_cols)
    print(f"  Feature set (deduped):")
    print(f"    Base PA/OR: {len(pa_base)}")
    print(f"    HMM-style:  {len(pa_hmm)}")
    print(f"    GARCH-style:{len(pa_garch)}")
    print(f"    TCN-derived:{len(pa_tcn)}")
    print(f"    Breakout:   {len(BO_FEAT_COLS)}")
    print(f"    Regime probs: not in X (Layer 2a outputs; used for reconstruct + Layer 3 only)")
    print(f"    TOTAL L2b:  {len(all_bo_feats)}")
    if hmm_cols:
        print(f"  HMM features included: {', '.join(sorted(hmm_cols)[:6])}"
              + (" ..." if len(hmm_cols) > 6 else ""))
    if garch_cols:
        print(f"  GARCH features included: {', '.join(garch_cols[:6])}"
              + (" ..." if len(garch_cols) > 6 else ""))

    X = work[all_bo_feats].values.astype(np.float32)
    _require_lgb_matrix_matches_names(X, all_bo_feats, "Layer 2b (trade-quality stack)")
    y6 = work["trade_quality_label"].values.astype(int)
    t = work["time_key"].values

    train_mask = t < np.datetime64(TRAIN_END)
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, y_train6 = X[train_mask], y6[train_mask]
    X_cal, y_cal6 = X[cal_mask], y6[cal_mask]
    X_test, y_test6 = X[test_mask], y6[test_mask]

    y_trade_train = np.isin(y_train6, list(TRADABLE_CLASS_IDS)).astype(int)
    y_trade_cal = np.isin(y_cal6, list(TRADABLE_CLASS_IDS)).astype(int)
    y_trade_test = np.isin(y_test6, list(TRADABLE_CLASS_IDS)).astype(int)

    tradable_train = y_trade_train == 1
    tradable_cal = y_trade_cal == 1
    tradable_test = y_trade_test == 1

    y_dir_train = np.isin(y_train6[tradable_train], [0, 1]).astype(int)   # 1=LONG, 0=SHORT
    y_dir_cal = np.isin(y_cal6[tradable_cal], [0, 1]).astype(int)
    y_dir_test = np.isin(y_test6[tradable_test], [0, 1]).astype(int)

    y_ab_train = np.isin(y_train6[tradable_train], [0, 5]).astype(int)     # 1=A, 0=B
    y_ab_cal = np.isin(y_cal6[tradable_cal], [0, 5]).astype(int)
    y_ab_test = np.isin(y_test6[tradable_test], [0, 5]).astype(int)

    print(f"  Dates — Train: < {TRAIN_END} | Cal: → {CAL_END} | Test: → {TEST_END}")
    print(f"  Train: {len(y_train6):,}  |  Cal: {len(y_cal6):,}  |  Test: {len(y_test6):,}")
    print(f"  Step1 TRADE rate (train/test): {y_trade_train.mean():.2%} / {y_trade_test.mean():.2%}")

    common_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": 7,
        "learning_rate": 0.02,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 60,
        "lambda_l1": 0.2,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
    }
    rounds = 1800 if FAST_TRAIN_MODE else 4000
    es = 90 if FAST_TRAIN_MODE else 140

    X2_train, X2_cal, X2_test = X_train[tradable_train], X_cal[tradable_cal], X_test[tradable_test]
    t2_train, t2_cal = t[train_mask][tradable_train], t[cal_mask][tradable_cal]
    w2_train = _binary_weights(y_dir_train, t2_train, pos_boost=1.0)
    w2_cal = _binary_weights(y_dir_cal, t2_cal, pos_boost=1.0)
    d2_train = lgb.Dataset(X2_train, label=y_dir_train, weight=w2_train, feature_name=all_bo_feats, free_raw_data=False)
    d2_cal = lgb.Dataset(X2_cal, label=y_dir_cal, weight=w2_cal, feature_name=all_bo_feats, free_raw_data=False)
    w3_train = _binary_weights(y_ab_train, t2_train, pos_boost=1.3)
    w3_cal = _binary_weights(y_ab_cal, t2_cal, pos_boost=1.1)
    d3_train = lgb.Dataset(X2_train, label=y_ab_train, weight=w3_train, feature_name=all_bo_feats, free_raw_data=False)
    d3_cal = lgb.Dataset(X2_cal, label=y_ab_cal, weight=w3_cal, feature_name=all_bo_feats, free_raw_data=False)

    reg_models: dict[str, dict[str, lgb.Booster]] = {}
    thr_vec = np.ones(len(REGIMES_6), dtype=np.float64)

    print(
        "  [L2b train] Step1 = 6-regime dual regression (MFE & MAE @ ATR → opportunity) …",
        flush=True,
    )
    mfe_full, mae_full = _mfe_mae_atr_arrays(work)
    st_full = work["state_label"].values.astype(int)
    rp_full = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)
    reg_models, _, thr_vec = _train_regime_opp_regression_models(
        X,
        st_full,
        mfe_full,
        mae_full,
        train_mask,
        cal_mask,
        all_bo_feats,
        rp_full[cal_mask],
        y_trade_cal,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    for regime in REGIMES_6:
        if regime not in reg_models:
            continue
        reg_models[regime]["mfe"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mfe_{regime}.txt"))
        reg_models[regime]["mae"].save_model(os.path.join(MODEL_DIR, f"l2b_opp_mae_{regime}.txt"))
    step1_regression_bundle = {"thr_vec": thr_vec}
    for regime, pair in reg_models.items():
        step1_regression_bundle[f"{regime}_mfe"] = pair["mfe"]
        step1_regression_bundle[f"{regime}_mae"] = pair["mae"]

    print("  [L2b train] Step2 LONG/SHORT LightGBM …", flush=True)
    c2, clean2 = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step2 LONG/SHORT")
    try:
        step2_model = lgb.train(
            common_params, d2_train, num_boost_round=rounds, valid_sets=[d2_cal], callbacks=c2,
        )
    finally:
        for fn in clean2:
            fn()
    print("  [L2b train] Step3 A/B LightGBM …", flush=True)
    c3, clean3 = _lgb_train_callbacks_with_round_tqdm(es, rounds, "L2b Step3 A/B")
    try:
        step3_model = lgb.train(
            common_params, d3_train, num_boost_round=rounds, valid_sets=[d3_cal], callbacks=c3,
        )
    finally:
        for fn in clean3:
            fn()

    rp_cal = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[cal_mask]
    opp_cal = _compute_opportunity_scores(X_cal, rp_cal, reg_models)
    gix_c = np.argmax(rp_cal, axis=1).astype(np.int64, copy=False)  # L2a class index; thr_vec[k]=REGIMES_6[k]
    p_trade_cal = _opp_to_synthetic_p_trade(opp_cal, thr_vec[gix_c])
    thr_trade_cal = 0.5
    pr_cal = (p_trade_cal >= thr_trade_cal).astype(int)
    f1_cal_at_thr = f1_score(y_trade_cal, pr_cal, zero_division=0)
    n_trade_pred_cal = int(pr_cal.sum())
    rec_cal_thr = recall_score(y_trade_cal, pr_cal, zero_division=0)
    prec_cal_thr = precision_score(y_trade_cal, pr_cal, zero_division=0)
    thr_rule_note = "regression gate: synth p_trade, fixed thr=0.5"
    print(
        f"  Step1 cal: synthetic p_trade @ {thr_trade_cal:.2f}  "
        f"F1={f1_cal_at_thr:.4f}  recall={rec_cal_thr:.3f}  precision={prec_cal_thr:.4f}  "
        f"n_trade={n_trade_pred_cal:,}/{len(y_trade_cal):,}  ({thr_rule_note})"
    )

    # Evaluate cascade
    rp_test = work[REGIME_NOW_PROB_COLS].values.astype(np.float32)[test_mask]
    opp_te = _compute_opportunity_scores(X_test, rp_test, reg_models)
    gix_t = np.argmax(rp_test, axis=1).astype(np.int64, copy=False)  # L2a class index → REGIMES_6[gix_t]
    p_trade = _opp_to_synthetic_p_trade(opp_te, thr_vec[gix_t])
    p_long = np.full(len(X_test), 0.5, dtype=float)
    p_a = np.full(len(X_test), 0.5, dtype=float)
    if len(X2_test) > 0:
        p_long[tradable_test] = step2_model.predict(X2_test)
        p_a[tradable_test] = step3_model.predict(X2_test)
    rp_rows = work.loc[test_mask, REGIME_NOW_PROB_COLS].to_numpy(dtype=np.float64, copy=False)
    p_range_mass = rp_rows[:, RANGE_REGIME_INDICES].sum(axis=1)
    y_pred6 = _reconstruct_quality_classes(
        p_trade, p_long, p_a, p_range_mass, thr_trade=thr_trade_cal
    )

    print("\n  Step metrics (test):")
    step1_pred = (p_trade >= thr_trade_cal).astype(int)
    s1_rec = recall_score(y_trade_test, step1_pred, zero_division=0)
    s1_prec = precision_score(y_trade_test, step1_pred, zero_division=0)
    print(
        f"    Step1 TRADE/SKIP  acc={accuracy_score(y_trade_test, step1_pred):.4f} "
        f"f1={f1_score(y_trade_test, step1_pred, zero_division=0):.4f}  "
        f"recall={s1_rec:.3f}  precision={s1_prec:.4f}  (thr={thr_trade_cal:.3f})"
    )
    if len(y_dir_test) > 0:
        step2_pred = (step2_model.predict(X2_test) >= 0.5).astype(int)
        print(f"    Step2 LONG/SHORT acc={accuracy_score(y_dir_test, step2_pred):.4f} "
              f"f1={f1_score(y_dir_test, step2_pred, zero_division=0):.4f}")
    if len(y_ab_test) > 0:
        step3_pred = (step3_model.predict(X2_test) >= 0.5).astype(int)
        print(f"    Step3 A/B        acc={accuracy_score(y_ab_test, step3_pred):.4f} "
              f"f1={f1_score(y_ab_test, step3_pred, zero_division=0):.4f}")

    acc = accuracy_score(y_test6, y_pred6)
    macro_f1 = f1_score(y_test6, y_pred6, average="macro")
    print("\n  Reconstructed 6-class metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test6, y_pred6, target_names=QUALITY_CLASS_ORDER, digits=4))
    cm = confusion_matrix(y_test6, y_pred6, labels=list(range(6)))
    print("  Confusion Matrix:")
    print(pd.DataFrame(cm, index=QUALITY_CLASS_ORDER, columns=QUALITY_CLASS_ORDER).to_string())

    _g = next(g for g in REGIMES_6 if g in reg_models)
    importance = reg_models[_g]["mfe"].feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": all_bo_feats, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    print("\n  Top 25 Step1 features (gain):")
    print(imp_df.head(25).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    step2_model.save_model(os.path.join(MODEL_DIR, "trade_dir_step2.txt"))
    step3_model.save_model(os.path.join(MODEL_DIR, "trade_grade_step3.txt"))
    import pickle

    meta = {
        "type": "trade_quality_hier_regression_gate",
        "class_names": QUALITY_CLASS_NAMES,
        "feature_cols": all_bo_feats,
        "pa_base_feat_cols": pa_base,
        "pa_hmm_feat_cols": pa_hmm,
        "pa_garch_feat_cols": pa_garch,
        "tcn_feat_cols": pa_tcn,
        "bo_feat_cols": BO_FEAT_COLS,
        "regime_prob_cols_layer3": REGIME_PROB_COLS,
        "garch_cols": garch_cols,
        "hierarchy_thresholds": {"trade": float(thr_trade_cal), "long": 0.50, "grade_a": 0.50},
        "step1_calibration": {
            "best_trade_threshold": float(thr_trade_cal),
            "best_f1_cal": float(f1_cal_at_thr),
            "recall_cal": float(rec_cal_thr),
            "precision_cal": float(prec_cal_thr),
            "n_trade_pred_cal": int(n_trade_pred_cal),
            "threshold_selection_rule": thr_rule_note,
        },
        "model_files": {
            "step2_direction": "trade_dir_step2.txt",
            "step3_grade": "trade_grade_step3.txt",
        },
        "position_scale_map": {
            "A_LONG": 1.50,
            "B_LONG": 1.00,
            "NEUTRAL": 0.00,
            "CHOP": 0.00,
            "B_SHORT": -1.00,
            "A_SHORT": -1.50,
        },
        "regression_gate": {
            "groups": list(REGIMES_6),
            "thr_vec": thr_vec.tolist(),
            "model_files": {
                fk: fn
                for regime in REGIMES_6
                if regime in reg_models
                for fk, fn in (
                    (f"{regime}_mfe", f"l2b_opp_mfe_{regime}.txt"),
                    (f"{regime}_mae", f"l2b_opp_mae_{regime}.txt"),
                )
            },
        },
    }

    with open(os.path.join(MODEL_DIR, "trade_quality_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(
        f"\n  Models saved → l2b_opp_mfe/mae_<regime>.txt (trained regimes), "
        f"trade_dir_step2.txt, trade_grade_step3.txt",
    )
    print(f"  Meta saved  → {MODEL_DIR}/trade_quality_meta.pkl")

    model_bundle = {
        "step1_regression": step1_regression_bundle,
        "step2": step2_model,
        "step3": step3_model,
        "thresholds": meta["hierarchy_thresholds"],
        "feature_cols": all_bo_feats,
    }
    return model_bundle, meta, imp_df


# ───────────────────────────────────────────────────────────────────────
# 4.  Layer 3 — Execution Sizer (state + quality + garch_vol -> position)
# ───────────────────────────────────────────────────────────────────────


def _layer3_chunk_rows() -> int:
    """Row chunk size for Layer-3 batched LGBM predict (RAM). Override with LAYER3_CHUNK."""
    return max(4096, int(os.environ.get("LAYER3_CHUNK", "65536")))


def _layer3_fill_regime_calibrated(
    regime_model: lgb.Booster,
    regime_calibrators: list,
    work: pd.DataFrame,
    out: np.ndarray,
    chunk: int,
) -> None:
    n = len(work)
    n_cls = NUM_REGIME_CLASSES
    n_chunk = (n + chunk - 1) // chunk
    regime_cols = _lgbm_booster_feature_names(regime_model)
    for i in _tq(range(0, n, chunk), desc="Layer3 regime→cal", total=n_chunk, unit="chunk"):
        j = min(i + chunk, n)
        x_s = work[regime_cols].iloc[i:j].to_numpy(dtype=np.float32, copy=False)
        raw = regime_model.predict(x_s)
        row = np.empty((j - i, n_cls), dtype=np.float64)
        for c in range(n_cls):
            row[:, c] = regime_calibrators[c].predict(raw[:, c])
        row = np.maximum(row, 1e-12)
        row /= row.sum(axis=1, keepdims=True)
        out[i:j] = row.astype(np.float32, copy=False)
        del x_s, raw, row


def _layer3_attach_regime_probs_to_work(work: pd.DataFrame, cal_regime: np.ndarray) -> None:
    """Persist Layer-2a calibrated probs on ``work`` (L2b regression gate reads ``REGIME_NOW_PROB_COLS``)."""
    for j, col in enumerate(REGIME_NOW_PROB_COLS):
        work[col] = cal_regime[:, j]
    work["regime_now_conf"] = cal_regime.max(axis=1)


def _layer3_fill_trade_stack_probs(
    trade_quality_models: dict,
    work: pd.DataFrame,
    layer2_feats: list[str],
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_a: np.ndarray,
    chunk: int,
) -> None:
    if not trade_quality_models.get("step1_regression"):
        raise RuntimeError("Layer 2b Step1 is regression-only; missing step1_regression in model bundle.")
    _layer3_fill_p_trade_from_regression(
        trade_quality_models, work, layer2_feats, p_trade, p_long, p_a, chunk,
    )


def train_execution_sizer(
    df: pd.DataFrame,
    feat_cols: list[str],
    regime_model: lgb.Booster,
    regime_calibrators: list,
    trade_quality_models: dict,
):
    print("\n" + "=" * 70)
    print("  LAYER 3: Execution Sizer v2 (L2b triplet × regime × TCN × GARCH × PA + gate×size)")
    print("=" * 70)

    l3_flat_tau = float(os.environ.get("L3_FLAT_TAU", "0.05"))
    l3_flat_w = float(os.environ.get("L3_FLAT_WEIGHT", "0.35"))

    chunk = _layer3_chunk_rows()
    print(f"  Memory: chunked predicts (LAYER3_CHUNK={chunk}); shallow df, no full feature matrices")

    work = df.copy(deep=False)
    bo_frame = compute_breakout_features(work)
    for c in BO_FEAT_COLS:
        work[c] = bo_frame[c].values
    del bo_frame

    n = len(work)
    cal_regime = np.empty((n, NUM_REGIME_CLASSES), dtype=np.float32)
    _layer3_fill_regime_calibrated(
        regime_model, regime_calibrators, work, cal_regime, chunk,
    )
    _layer3_attach_regime_probs_to_work(work, cal_regime)

    garch_cols = sorted([
        c for c in work.columns
        if c.startswith("pa_garch_") and str(work[c].dtype) not in {"object", "category"}
    ])
    layer2_feats = trade_quality_models["feature_cols"]
    thr = trade_quality_models["thresholds"]

    p_trade = np.empty(n, dtype=np.float32)
    p_long = np.empty(n, dtype=np.float32)
    p_a = np.empty(n, dtype=np.float32)
    _layer3_fill_trade_stack_probs(
        trade_quality_models, work, layer2_feats, p_trade, p_long, p_a, chunk,
    )

    l2b_opp = np.empty(n, dtype=np.float32)
    l2b_mfe = np.empty(n, dtype=np.float32)
    l2b_mae = np.empty(n, dtype=np.float32)
    _layer3_fill_l2b_triplet_arrays(
        trade_quality_models, work, layer2_feats, p_trade, l2b_opp, l2b_mfe, l2b_mae, chunk,
    )

    p_range_mass = cal_regime[:, RANGE_REGIME_INDICES].sum(axis=1)
    y_cls_est = _reconstruct_quality_classes(
        p_trade=p_trade,
        p_long=p_long,
        p_a=p_a,
        p_range_mass=p_range_mass,
        thr_trade=thr["trade"],
        thr_long=thr["long"],
        thr_a=thr["grade_a"],
    )

    y_cls = _build_trade_quality_targets(work)
    class_size = np.array([1.5, 1.0, 0.0, 0.0, -1.0, -1.5], dtype=float)
    base_size = 0.7 * class_size[y_cls] + 0.3 * class_size[y_cls_est]

    safe_atr = np.where(work["lbl_atr"].values > 1e-3, work["lbl_atr"].values, 1e-3)
    edge = (work["max_favorable"].values - work["max_adverse"].values) / safe_atr
    edge = np.clip(edge, -3.0, 3.0)
    edge_scale = np.clip(0.90 + 0.30 * edge, 0.0, 1.60)
    y_target = np.clip(base_size * edge_scale, -1.0, 1.0)

    tcn_prob_cols = [c for c in TCN_REGIME_FUT_PROB_COLS if c in work.columns]
    pa_key_cols = [c for c in LAYER3_PA_KEY_FEATURES if c in work.columns][:15]

    # Routed scalar opp (this bar's argmax regime head) × each regime's probability — 6 L3 interaction cols.
    inter_blk = (
        l2b_opp.astype(np.float64)[:, None] * cal_regime.astype(np.float64)
    ).astype(np.float32, copy=False)

    triplet_blk = np.hstack([
        l2b_opp.reshape(-1, 1),
        l2b_mfe.reshape(-1, 1),
        l2b_mae.reshape(-1, 1),
    ]).astype(np.float32, copy=False)
    sc_conf = cal_regime.max(axis=1, keepdims=True).astype(np.float32, copy=False)
    regime_blk = np.hstack([cal_regime, sc_conf]).astype(np.float32, copy=False)

    tcn_mat = work[tcn_prob_cols].to_numpy(dtype=np.float32, copy=False) if tcn_prob_cols else np.empty((n, 0), np.float32)
    pa_mat = work[pa_key_cols].to_numpy(dtype=np.float32, copy=False) if pa_key_cols else np.empty((n, 0), np.float32)
    if garch_cols:
        g_mat = work[garch_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        g_mat = np.empty((n, 0), dtype=np.float32)

    X = np.hstack([triplet_blk, regime_blk, tcn_mat, g_mat, pa_mat, inter_blk])
    exec_feat_cols = (
        ["l2b_opportunity_score", "l2b_pred_mfe", "l2b_pred_mae"]
        + REGIME_NOW_PROB_COLS
        + ["regime_now_conf"]
        + tcn_prob_cols
        + garch_cols
        + pa_key_cols
        + L2B_OPP_X_REGIME_COLS
    )
    _require_lgb_matrix_matches_names(X, exec_feat_cols, "Layer 3 (execution sizer v2)")

    del triplet_blk, regime_blk, tcn_mat, pa_mat, g_mat, inter_blk, cal_regime
    del p_trade, p_long, p_a, sc_conf, work
    del l2b_opp, l2b_mfe, l2b_mae
    gc.collect()

    t = df["time_key"].values
    cal_mask = (t >= np.datetime64(TRAIN_END)) & (t < np.datetime64(CAL_END))
    test_mask = (t >= np.datetime64(CAL_END)) & (t < np.datetime64(TEST_END))

    X_train, y_train = X[cal_mask], y_target[cal_mask]
    X_test, y_test = X[test_mask], y_target[test_mask]

    y_gate_train = (np.abs(y_train) >= l3_flat_tau).astype(np.int32)
    pos_ct = int(y_gate_train.sum())
    neg_ct = int(len(y_gate_train) - pos_ct)
    spw = float(neg_ct / max(pos_ct, 1)) if pos_ct else 1.0

    w_size = np.where(np.abs(y_train) < l3_flat_tau, l3_flat_w, 1.0).astype(np.float64)
    y_gate_test = (np.abs(y_test) >= l3_flat_tau).astype(np.int32)

    print(
        f"  Features: {len(exec_feat_cols)} "
        f"(L2b triplet=3, regime_now={len(REGIME_NOW_PROB_COLS)}+conf, "
        f"tcn_fut={len(tcn_prob_cols)}, garch={len(garch_cols)}, pa_key={len(pa_key_cols)}, "
        f"opp×regime=3)",
    )
    print(
        f"  Train (cal, full rows): {len(y_train):,}  |  Valid/Test: {len(y_test):,}  "
        f"| flat weight={l3_flat_w}  τ={l3_flat_tau}",
    )
    print(
        f"  Active (|y|≥τ) — train: {(np.abs(y_train) >= l3_flat_tau).mean():.1%} | "
        f"test: {(np.abs(y_test) >= l3_flat_tau).mean():.1%}",
    )

    rounds = 1600 if FAST_TRAIN_MODE else 4000
    es_cb = _lgb_train_callbacks(90 if FAST_TRAIN_MODE else 120)

    gate_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 120,
        "lambda_l1": 0.15,
        "lambda_l2": 1.5,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
        "scale_pos_weight": spw,
    }
    d_gate_tr = lgb.Dataset(X_train, label=y_gate_train, feature_name=exec_feat_cols, free_raw_data=True)
    d_gate_va = lgb.Dataset(X_test, label=y_gate_test, feature_name=exec_feat_cols, free_raw_data=True)
    model_gate = lgb.train(
        gate_params,
        d_gate_tr,
        num_boost_round=rounds,
        valid_sets=[d_gate_va],
        callbacks=es_cb,
    )

    size_params = {
        "objective": "huber",
        "alpha": 0.9,
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": 43,
        "n_jobs": _lgbm_n_jobs(),
    }
    d_sz_tr = lgb.Dataset(
        X_train, label=y_train, weight=w_size, feature_name=exec_feat_cols, free_raw_data=True,
    )
    d_sz_va = lgb.Dataset(X_test, label=y_test, feature_name=exec_feat_cols, free_raw_data=True)
    model_size = lgb.train(
        size_params,
        d_sz_tr,
        num_boost_round=rounds,
        valid_sets=[d_sz_va],
        callbacks=es_cb,
    )

    pred_g = model_gate.predict(X_test)
    pred_s = model_size.predict(X_test)
    pred = np.clip(pred_g * pred_s, -1.0, 1.0)

    mse = float(np.mean((pred - y_test) ** 2))
    nz = np.abs(y_test) >= l3_flat_tau
    sign_hit = float((np.sign(pred[nz]) == np.sign(y_test[nz])).mean()) if nz.sum() > 0 else float("nan")
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if len(pred) > 2 else float("nan")
    try:
        gate_auc = float(roc_auc_score(y_gate_test, pred_g)) if len(np.unique(y_gate_test)) > 1 else float("nan")
    except ValueError:
        gate_auc = float("nan")

    print("\n  Test metrics (combined = p_gate × size, clipped):")
    print(f"    MSE:         {mse:.5f}")
    print(f"    Gate AUC:    {gate_auc:.4f}  (|y|≥τ)")
    print(f"    Corr(y,p):   {corr:.4f}")
    print(f"    Sign hit:    {sign_hit:.4f}  (|target|≥τ)")
    print(f"    Mean |pos|:  {np.mean(np.abs(pred)):.3f}")

    imp_size = model_size.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": exec_feat_cols,
        "importance": imp_size,
    }).sort_values("importance", ascending=False)
    print("\n  Top 20 Layer-3 size-head features (gain):")
    print(imp_df.head(20).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_gate.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_GATE_FILE))
    model_size.save_model(os.path.join(MODEL_DIR, EXECUTION_SIZER_SIZE_FILE))
    import pickle

    meta = {
        "l3_schema": 2,
        "type": "execution_sizer_two_stage",
        "feature_cols": exec_feat_cols,
        "position_clip": [-1.0, 1.0],
        "combine_rule": "clip(p_gate * pred_size, -1, 1)",
        "target_definition": "clip(class_blend * edge_scale, -1, 1); same tier blend as v1",
        "flat_tau": l3_flat_tau,
        "flat_sample_weight": l3_flat_w,
        "gate_metric": "auc",
        "size_objective": "huber",
        "uses_garch": bool(garch_cols),
        "garch_cols": garch_cols,
        "pa_key_cols": pa_key_cols,
        "tcn_prob_cols": tcn_prob_cols,
        "model_files": {
            "gate": EXECUTION_SIZER_GATE_FILE,
            "size": EXECUTION_SIZER_SIZE_FILE,
        },
    }
    with open(os.path.join(MODEL_DIR, "execution_sizer_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    bundle = {"gate": model_gate, "size": model_size, "meta": meta}
    print(
        f"\n  Models saved → {MODEL_DIR}/{EXECUTION_SIZER_GATE_FILE}, "
        f"{EXECUTION_SIZER_SIZE_FILE}",
    )
    print(f"  Meta saved  → {MODEL_DIR}/execution_sizer_meta.pkl")
    return bundle, meta, imp_df


# ───────────────────────────────────────────────────────────────────────
# 5.  Main
# ───────────────────────────────────────────────────────────────────────

def main():
    configure_compute_threads()
    n_th = torch.get_num_threads()
    print(
        f"  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}  "
        f"(override with TORCH_CPU_THREADS / LGBM_N_JOBS)"
    )

    print("=" * 70)
    print("  TCN(derived 8 feats) + LightGBM Layer 2 (regime + trade) + Layer 3")
    print("=" * 70)

    df, feat_cols = prepare_dataset(["QQQ", "SPY"])

    regime_model, regime_cal, regime_imp = train_regime_classifier(df, feat_cols)
    print(
        "  [progress] Layer 2a done → Layer 2b (prep + hierarchical trade models) …",
        flush=True,
    )
    tq_model, tq_meta, tq_imp = train_trade_quality_classifier(
        df, feat_cols, regime_model, regime_cal,
    )
    _, exec_meta, exec_imp = train_execution_sizer(
        df, feat_cols, regime_model, regime_cal, tq_model,
    )

    print("\n" + "=" * 70)
    print("  DONE — Models saved:")
    print(f"    Layer 2a regime head: {MODEL_DIR}/{STATE_CLASSIFIER_FILE}")
    print(f"    Layer 2b trade stack: l2b_opp_mfe/mae_*.txt + trade_dir/step3 (+ meta)")
    print(
        f"    Layer 3 sizer:        {MODEL_DIR}/{EXECUTION_SIZER_GATE_FILE} + "
        f"{EXECUTION_SIZER_SIZE_FILE}",
    )
    print(f"    Regime calibrators:   {MODEL_DIR}/state_calibrators.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()

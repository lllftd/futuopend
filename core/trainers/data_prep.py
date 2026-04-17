from __future__ import annotations

import gc
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm

from core.mamba_pa_state import PAStateMamba
from core.tcn_pa_state import PAStateTCN, FocalLoss
from core.trainers.pa_feature_cache import load_or_build_pa_features
from core.trainers.pa_state_controls import ensure_pa_state_features

from core.trainers.constants import *
from core.trainers.lgbm_utils import *
from core.trainers.tcn_constants import STATE_CLASSIFIER_FILE as TCN_STATE_DICT_BASENAME

def _load_and_compute_pa(symbol: str) -> pd.DataFrame:
    return load_or_build_pa_features(symbol, DATA_DIR, timeframe="5min")


def _load_labels(symbol: str) -> pd.DataFrame:
    required_cols = [
        "time_key", "market_state",
        "signal", "outcome",
        "quality_bull_breakout", "quality_bear_breakout",
        "max_favorable", "max_adverse", "exit_bar",
        "atr",
    ]
    optional_cols = [
        "decision_mfe_atr", "decision_mae_atr", "decision_peak_bar",
        "decision_theta_decay", "decision_net_edge_atr",
        "optimal_tp_atr", "optimal_sl_atr", "optimal_exit_bar", "optimal_net_edge_atr",
    ]
    path = os.path.join(DATA_DIR, f"{symbol}{LABELED_SUFFIX}.csv")
    keep_cols = set(required_cols + optional_cols)
    lbl = pd.read_csv(path, usecols=lambda c: c in keep_cols)
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


def _add_tcn_summary_features(df: pd.DataFrame, *, validate: bool) -> pd.DataFrame:
    """Derive stable scalar TCN features used by L2b/L3 from the raw TCN barrier probabilities."""
    required = list(TCN_REGIME_FUT_PROB_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing raw TCN barrier columns for derived features: {missing}")

    chop = pd.to_numeric(df["tcn_barrier_chop"], errors="coerce")
    if validate:
        desc = chop.describe(percentiles=[0.05, 0.50, 0.95])
        print(
            "  TCN barrier_chop describe: "
            f"count={float(desc.get('count', 0.0)):.0f}  mean={float(desc.get('mean', float('nan'))):.4f}  "
            f"std={float(desc.get('std', float('nan'))):.4f}  min={float(desc.get('min', float('nan'))):.4f}  "
            f"p50={float(desc.get('50%', float('nan'))):.4f}  p95={float(desc.get('95%', float('nan'))):.4f}  "
            f"max={float(desc.get('max', float('nan'))):.4f}",
            flush=True,
        )
    finite_chop = chop.dropna()
    if finite_chop.empty:
        raise RuntimeError("tcn_barrier_chop is entirely NaN; cannot derive tcn_transition_prob.")
    cmin = float(finite_chop.min())
    cmax = float(finite_chop.max())
    if cmin < -1e-4 or cmax > 1.0001:
        raise RuntimeError(
            f"tcn_barrier_chop looks unnormalized (min={cmin:.4f}, max={cmax:.4f}); "
            "cannot safely derive tcn_transition_prob = 1 - tcn_barrier_chop."
        )

    df[TCN_TRANSITION_PROB_COL] = np.clip(1.0 - chop.to_numpy(dtype=np.float32, copy=False), 0.0, 1.0)
    up = pd.to_numeric(df["tcn_barrier_hit_up"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    dn = pd.to_numeric(df["tcn_barrier_hit_dn"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    df[TCN_BARRIER_DIR_DIFF_COL] = (up - dn).astype(np.float32, copy=False)

    if validate:
        transition = pd.to_numeric(df[TCN_TRANSITION_PROB_COL], errors="coerce")
        valid_ratio = float(transition.notna().mean())
        std = float(transition.std(skipna=True))
        if valid_ratio <= 0.95:
            raise RuntimeError(
                f"tcn_transition_prob has {(1.0 - valid_ratio):.1%} NaN values after derivation."
            )
        if not np.isfinite(std) or std <= 0.01:
            raise RuntimeError(
                f"tcn_transition_prob std={std:.4f}, looks too close to a constant column."
            )
    return df


def _create_tcn_windows(feat_1m: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(feat_1m)
    if n < seq_len:
        return np.zeros((0, seq_len, feat_1m.shape[1]), dtype=np.float32), np.zeros(0, dtype=int)
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    windows = feat_1m[idx].astype(np.float32)
    end_idx = np.arange(seq_len - 1, n, dtype=int)
    return windows, end_idx


def _nan_safe_ffill_2d(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    filled = np.asarray(arr, dtype=np.float32).copy()
    mask = np.isnan(filled)
    if not mask.any():
        return filled

    row_idx = np.arange(filled.shape[0], dtype=np.int32)[:, None]

    last_valid = np.where(~mask, row_idx, -1)
    np.maximum.accumulate(last_valid, axis=0, out=last_valid)
    cols = np.broadcast_to(np.arange(filled.shape[1]), filled.shape)
    valid_last = last_valid >= 0
    filled[valid_last] = filled[last_valid[valid_last], cols[valid_last]]
    return filled


def _finalize_sequence_feature_block(
    regime_probs_arr: np.ndarray,
    embeddings: np.ndarray,
    warm_mask: np.ndarray,
    *,
    cold_regime_defaults: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime_probs_arr = _nan_safe_ffill_2d(regime_probs_arr)
    embeddings = _nan_safe_ffill_2d(embeddings)
    warm = np.asarray(warm_mask, dtype=bool).ravel()
    cold = ~warm
    if cold.any():
        regime_probs_arr[cold] = np.asarray(cold_regime_defaults, dtype=np.float32)
        embeddings[cold] = 0.0
    return regime_probs_arr.astype(np.float32), embeddings.astype(np.float32), warm.astype(np.float32)


def ensure_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in BO_FEAT_COLS if c not in df.columns]
    if not missing_cols:
        return df

    if "symbol" in df.columns:
        parts: list[pd.DataFrame] = []
        for _, grp in df.groupby("symbol", sort=False):
            parts.append(compute_breakout_features(grp))
        bo_feats = pd.concat(parts).loc[df.index]
    else:
        bo_feats = compute_breakout_features(df)

    for col in missing_cols:
        df[col] = bo_feats[col].to_numpy(dtype=np.float32, copy=False)
    return df


def _ctx_first_numeric(df: pd.DataFrame, names: list[str], default: float = 0.0) -> np.ndarray:
    for name in names:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce").fillna(default)
            return series.to_numpy(dtype=np.float32, copy=False)
    return np.full(len(df), default, dtype=np.float32)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def ensure_structure_context_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in PA_CTX_FEATURES if c not in df.columns]
    if not missing_cols:
        return ensure_pa_state_features(df)

    ensure_breakout_features(df)

    pressure_diff = _ctx_first_numeric(df, ["bo_pressure_diff"])
    consec_dir = _ctx_first_numeric(df, ["bo_consec_dir"])
    gap_signal = _ctx_first_numeric(df, ["bo_gap_signal"])
    or_dist = _ctx_first_numeric(df, ["bo_or_dist"])
    close_ext = _ctx_first_numeric(df, ["bo_close_extremity"], default=0.5)
    body_growth = _ctx_first_numeric(df, ["bo_body_growth"])
    range_atr = _ctx_first_numeric(df, ["bo_range_atr"])
    inside_prior = _ctx_first_numeric(df, ["bo_inside_prior", "pa_is_inside_bar"])
    range_compress = _ctx_first_numeric(df, ["bo_range_compress"], default=1.0)
    bb_width = _ctx_first_numeric(df, ["bo_bb_width"])

    trend_dir = _ctx_first_numeric(df, ["pa_env_trend_dir"])
    trend_score = _ctx_first_numeric(df, ["pa_env_trend_score_ratio", "pa_trend_ratio"])
    range_score = _ctx_first_numeric(df, ["pa_env_range_score_ratio", "pa_regime_range"])
    breakout_fail = _ctx_first_numeric(df, ["pa_breakout_likely_fail"])
    trend_weakened = _ctx_first_numeric(df, ["pa_trend_weakened"])
    resume_prob = _ctx_first_numeric(df, ["pa_trend_resume_prob"], default=0.75)
    opening_reversal = _ctx_first_numeric(df, ["pa_opening_reversal"])
    gap_open = _ctx_first_numeric(df, ["pa_gap_open_flag"])
    h2 = _ctx_first_numeric(df, ["pa_is_h2_setup"])
    l2 = _ctx_first_numeric(df, ["pa_is_l2_setup"])
    pullback_stage = _ctx_first_numeric(df, ["pa_pullback_stage"])

    bull_pressure = _clip01(0.5 + 0.5 * np.tanh(pressure_diff / 2.0))
    bear_pressure = _clip01(0.5 - 0.5 * np.tanh(pressure_diff / 2.0))
    consec_long = _clip01(0.5 + 0.5 * np.tanh(consec_dir / 3.0))
    consec_short = _clip01(0.5 - 0.5 * np.tanh(consec_dir / 3.0))
    gap_long = _clip01(0.5 + 0.5 * np.tanh(gap_signal))
    gap_short = _clip01(0.5 - 0.5 * np.tanh(gap_signal))
    or_above = _clip01(0.5 + 0.5 * np.tanh(or_dist))
    or_below = _clip01(0.5 - 0.5 * np.tanh(or_dist))
    bull_close = _clip01(close_ext)
    bear_close = _clip01(1.0 - close_ext)
    body_push = _clip01(body_growth / 2.5)
    range_push = _clip01(range_atr / 3.0)
    tight_range = _clip01(1.0 - np.clip(range_compress / 1.5, 0.0, 1.0))
    bb_tight = _clip01(1.0 - np.tanh(np.maximum(bb_width, 0.0) * 2.0))
    trend_long_dir = _clip01(trend_dir)
    trend_short_dir = _clip01(-trend_dir)
    trend_score = _clip01(trend_score)
    range_score = _clip01(range_score)
    breakout_fail = _clip01(np.maximum(breakout_fail, range_score * 0.75))
    trend_weakened = _clip01(trend_weakened)
    resume_score = _clip01((resume_prob - 0.5) / 0.45)
    h2 = _clip01(h2)
    l2 = _clip01(l2)
    pullback_long = _clip01(np.clip(pullback_stage, 0.0, 4.0) / 4.0)
    pullback_short = _clip01(np.clip(-pullback_stage, 0.0, 4.0) / 4.0)
    open_ctx = _clip01(0.6 * opening_reversal + 0.4 * gap_open)

    trend_setup_long = _clip01(
        0.26 * trend_long_dir
        + 0.20 * trend_score
        + 0.18 * bull_pressure
        + 0.12 * consec_long
        + 0.10 * body_push
        + 0.08 * range_push
        + 0.08 * bull_close
        + 0.06 * gap_long
        - 0.24 * range_score
        - 0.20 * breakout_fail
    )
    trend_setup_short = _clip01(
        0.26 * trend_short_dir
        + 0.20 * trend_score
        + 0.18 * bear_pressure
        + 0.12 * consec_short
        + 0.10 * body_push
        + 0.08 * range_push
        + 0.08 * bear_close
        + 0.06 * gap_short
        - 0.24 * range_score
        - 0.20 * breakout_fail
    )
    pullback_setup_long = _clip01(
        0.30 * trend_long_dir
        + 0.24 * h2
        + 0.18 * pullback_long
        + 0.16 * resume_score
        + 0.08 * or_below
        + 0.08 * bull_pressure
        - 0.20 * range_score
        - 0.12 * trend_weakened
    )
    pullback_setup_short = _clip01(
        0.30 * trend_short_dir
        + 0.24 * l2
        + 0.18 * pullback_short
        + 0.16 * resume_score
        + 0.08 * or_above
        + 0.08 * bear_pressure
        - 0.20 * range_score
        - 0.12 * trend_weakened
    )
    range_setup_long = _clip01(
        0.38 * range_score
        + 0.18 * tight_range
        + 0.14 * bb_tight
        + 0.12 * inside_prior
        + 0.10 * or_below
        + 0.08 * bear_close
        + 0.06 * open_ctx
        - 0.12 * trend_score
    )
    range_setup_short = _clip01(
        0.38 * range_score
        + 0.18 * tight_range
        + 0.14 * bb_tight
        + 0.12 * inside_prior
        + 0.10 * or_above
        + 0.08 * bull_close
        + 0.06 * open_ctx
        - 0.12 * trend_score
    )
    failed_breakout_long = _clip01(
        0.34 * breakout_fail
        + 0.18 * range_score
        + 0.16 * bull_pressure
        + 0.12 * bull_close
        + 0.10 * inside_prior
        + 0.10 * open_ctx
        - 0.12 * trend_short_dir
    )
    failed_breakout_short = _clip01(
        0.34 * breakout_fail
        + 0.18 * range_score
        + 0.16 * bear_pressure
        + 0.12 * bear_close
        + 0.10 * inside_prior
        + 0.10 * open_ctx
        - 0.12 * trend_long_dir
    )
    follow_long = _clip01(
        0.28 * bull_pressure
        + 0.18 * consec_long
        + 0.16 * bull_close
        + 0.14 * body_push
        + 0.10 * range_push
        + 0.10 * gap_long
        - 0.20 * inside_prior
        - 0.20 * breakout_fail
    )
    follow_short = _clip01(
        0.28 * bear_pressure
        + 0.18 * consec_short
        + 0.16 * bear_close
        + 0.14 * body_push
        + 0.10 * range_push
        + 0.10 * gap_short
        - 0.20 * inside_prior
        - 0.20 * breakout_fail
    )

    setup_long = np.maximum.reduce(
        [trend_setup_long, pullback_setup_long, range_setup_long, failed_breakout_long]
    ).astype(np.float32, copy=False)
    setup_short = np.maximum.reduce(
        [trend_setup_short, pullback_setup_short, range_setup_short, failed_breakout_short]
    ).astype(np.float32, copy=False)

    range_pressure = _clip01(
        0.50 * range_score
        + 0.18 * tight_range
        + 0.12 * bb_tight
        + 0.10 * inside_prior
        + 0.10 * (1.0 - np.abs(pressure_diff) / (np.abs(pressure_diff) + 1.0))
    )
    structure_veto = _clip01(
        0.42 * range_pressure
        + 0.28 * breakout_fail
        + 0.18 * trend_weakened
        + 0.12 * (1.0 - np.maximum(follow_long, follow_short))
    )
    premise_break_long = _clip01(
        0.42 * structure_veto
        + 0.22 * setup_short
        + 0.20 * np.maximum(follow_short - follow_long, 0.0)
        + 0.16 * range_pressure
    )
    premise_break_short = _clip01(
        0.42 * structure_veto
        + 0.22 * setup_long
        + 0.20 * np.maximum(follow_long - follow_short, 0.0)
        + 0.16 * range_pressure
    )

    ctx_values = {
        "pa_ctx_setup_trend_long": trend_setup_long,
        "pa_ctx_setup_trend_short": trend_setup_short,
        "pa_ctx_setup_pullback_long": pullback_setup_long,
        "pa_ctx_setup_pullback_short": pullback_setup_short,
        "pa_ctx_setup_range_long": range_setup_long,
        "pa_ctx_setup_range_short": range_setup_short,
        "pa_ctx_setup_failed_breakout_long": failed_breakout_long,
        "pa_ctx_setup_failed_breakout_short": failed_breakout_short,
        "pa_ctx_setup_long": setup_long,
        "pa_ctx_setup_short": setup_short,
        "pa_ctx_follow_through_long": follow_long,
        "pa_ctx_follow_through_short": follow_short,
        "pa_ctx_range_pressure": range_pressure,
        "pa_ctx_structure_veto": structure_veto,
        "pa_ctx_premise_break_long": premise_break_long,
        "pa_ctx_premise_break_short": premise_break_short,
    }
    for col in missing_cols:
        df[col] = ctx_values[col]
    return ensure_pa_state_features(df)


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
            f"Missing TCN checkpoint under {MODEL_DIR}: need tcn_meta.pkl and {TCN_STATE_DICT_BASENAME!r}. "
            "Train the PAStateTCN stack first: PYTHONPATH=. python backtests/train_tcn_layer1.py "
            "(train_pipeline layer1/layer1a does not create these files)."
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
            readout_type=str(meta.get("readout_type", "last_timestep")),
            min_attention_seq_len=int(meta.get("min_attention_seq_len", 4)),
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
    
        work = df.copy(deep=False)
    
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
            warm_mask = np.zeros(len(g), dtype=np.float32)
            warm_mask[end_idx] = 1.0
            regime_probs_arr, embeddings, warm_mask = _finalize_sequence_feature_block(
                regime_probs_arr,
                embeddings,
                warm_mask,
                cold_regime_defaults=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )

            if np.isnan(regime_probs_arr).any():
                raise RuntimeError(f"TCN regime probs still NaN for symbol {sym!r} after cold-start finalize.")
            if np.isnan(embeddings).any():
                raise RuntimeError(f"TCN embeddings still NaN for symbol {sym!r} after cold-start finalize.")

            sym_df = g[["symbol", "time_key"]].copy()
            eps = 1e-9
            sym_df["tcn_regime_fut_entropy"] = -np.sum(
                regime_probs_arr * np.log(np.maximum(regime_probs_arr, eps)), axis=1
            )
            for j, col in enumerate(TCN_REGIME_FUT_PROB_COLS):
                sym_df[col] = regime_probs_arr[:, j]
            sym_df["tcn_is_warm"] = warm_mask
            _add_tcn_summary_features(sym_df, validate=False)
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
        if tcn_cols:
            merged[tcn_cols] = (
                merged.groupby("symbol", sort=False)[tcn_cols]
                .ffill()
                .astype(np.float32, copy=False)
            )

        # --- INJECT OOF CACHE TO PREVENT DATA LEAKAGE ---
        oof_path = os.path.join(MODEL_DIR, "tcn_oof_cache.pkl")
        if os.path.exists(oof_path):
            tqdm.write(f"  Injecting TCN OOF predictions from {oof_path} into train split...")
            with open(oof_path, "rb") as f:
                oof = pickle.load(f)
            if "regime_probs" not in oof:
                raise RuntimeError(
                    f"{oof_path} is missing 'regime_probs' (transition). "
                    "Re-run: PYTHONPATH=. python backtests/train_tcn_layer1.py"
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
                    f"Delete {oof_path} and re-run: PYTHONPATH=. python backtests/train_tcn_layer1.py"
                )
            for j, col in enumerate(TCN_REGIME_FUT_PROB_COLS):
                oof_df[col] = rp[:, j]
            oof_df["tcn_is_warm"] = np.ones(len(oof_df), dtype=np.float32)

            eps = 1e-9
            oof_df["tcn_regime_fut_entropy"] = -np.sum(
                rp * np.log(np.maximum(rp, eps)), axis=1
            )
            oem = np.asarray(oof["embeds"], dtype=np.float32)
            if oem.shape[1] != bottleneck_dim:
                raise RuntimeError(
                    f"tcn_oof_cache.pkl embed width {oem.shape[1]} != bottleneck_dim {bottleneck_dim} "
                    f"from tcn_meta.pkl. Re-run: PYTHONPATH=. python backtests/train_tcn_layer1.py"
                )
            for j in range(bottleneck_dim):
                oof_df[f"tcn_emb_{j}"] = oem[:, j]
            _add_tcn_summary_features(oof_df, validate=False)

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
        _add_tcn_summary_features(merged, validate=True)
        return merged
    finally:
        _restore_torch_threads()



def _compute_mamba_derived_features(df: pd.DataFrame, base_feat_cols: list[str]) -> pd.DataFrame:
    """
    Real Mamba forward only — no uniform-prior placeholders.
    Requires ``mamba_meta.pkl`` and ``mamba_state_classifier_6c.pt`` from the current
    Layer 1 training flow.
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
    
        work = df.copy(deep=False)
    
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
            warm_mask = np.zeros(len(g), dtype=np.float32)
            warm_mask[end_idx] = 1.0
            regime_probs_arr, embeddings, warm_mask = _finalize_sequence_feature_block(
                regime_probs_arr,
                embeddings,
                warm_mask,
                cold_regime_defaults=np.array([1.0, 0.0], dtype=np.float32),
            )

            if np.isnan(regime_probs_arr).any():
                raise RuntimeError(f"Mamba regime probs still NaN for symbol {sym!r} after cold-start finalize.")
            if np.isnan(embeddings).any():
                raise RuntimeError(f"Mamba embeddings still NaN for symbol {sym!r} after cold-start finalize.")

            sym_df = g[["symbol", "time_key"]].copy()
            eps = 1e-9
            sym_df["mamba_regime_fut_entropy"] = -np.sum(
                regime_probs_arr * np.log(np.maximum(regime_probs_arr, eps)), axis=1
            )
            for j, col in enumerate(MAMBA_REGIME_FUT_PROB_COLS):
                sym_df[col] = regime_probs_arr[:, j]
            sym_df["mamba_is_warm"] = warm_mask
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
        if mamba_cols:
            merged[mamba_cols] = (
                merged.groupby("symbol", sort=False)[mamba_cols]
                .ffill()
                .astype(np.float32, copy=False)
            )

        # --- INJECT OOF CACHE TO PREVENT DATA LEAKAGE ---
        oof_path = os.path.join(MODEL_DIR, "mamba_oof_cache.pkl")
        if os.path.exists(oof_path):
            tqdm.write(f"  Injecting Mamba OOF predictions from {oof_path} into train split...")
            with open(oof_path, "rb") as f:
                oof = pickle.load(f)
            if "regime_probs" not in oof:
                raise RuntimeError(
                    f"{oof_path} is missing 'regime_probs' (transition). "
                    "Re-run: PYTHONPATH=. python backtests/train_tcn_layer1.py"
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
                    f"Delete {oof_path} and re-train Layer 1."
                )
            for j, col in enumerate(MAMBA_REGIME_FUT_PROB_COLS):
                oof_df[col] = rp[:, j]
            oof_df["mamba_is_warm"] = np.ones(len(oof_df), dtype=np.float32)

            eps = 1e-9
            oof_df["mamba_regime_fut_entropy"] = -np.sum(
                rp * np.log(np.maximum(rp, eps)), axis=1
            )
            oem = np.asarray(oof["embeds"], dtype=np.float32)
            if oem.shape[1] != bottleneck_dim:
                raise RuntimeError(
                    f"mamba_oof_cache.pkl embed width {oem.shape[1]} != bottleneck_dim {bottleneck_dim} "
                    f"from mamba_meta.pkl. Retrain Layer 1 via ./scripts/run_train.sh layer1"
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
    df["state_label"] = (
        hmm_state.fillna(ms_num).fillna(UNKNOWN_REGIME_CLASS_ID).astype(int).clip(0, NUM_REGIME_CLASSES - 1)
    )

    feat_cols = _pa_feature_cols(df)
    df = _compute_tcn_derived_features(df, feat_cols)
    feat_cols = _unique_cols(feat_cols + _tcn_derived_feature_names())

    # Mamba is kept behind an explicit opt-in because it is experimental and not part of the default stack.
    enable_experimental_mamba = os.environ.get("ENABLE_EXPERIMENTAL_MAMBA", "").strip().lower() in ("1", "true", "yes")
    disable_mamba = os.environ.get("DISABLE_MAMBA_FEATURES", "").strip().lower() in ("1", "true", "yes")
    mamba_ckpt = os.path.join(MODEL_DIR, "mamba_state_classifier_6c.pt")
    if not enable_experimental_mamba:
        print("  Mamba-derived features skipped (experimental module; set ENABLE_EXPERIMENTAL_MAMBA=1 to enable).", flush=True)
    elif disable_mamba:
        print("  Mamba-derived features skipped (DISABLE_MAMBA_FEATURES=1).", flush=True)
    elif os.path.exists(mamba_ckpt):
        df = _compute_mamba_derived_features(df, feat_cols)
        feat_cols = _unique_cols(feat_cols + _mamba_derived_feature_names())
    else:
        print("  Experimental Mamba requested but no checkpoint found; continuing with TCN (+PA) features only.", flush=True)

    df = ensure_breakout_features(df)
    df = ensure_structure_context_features(df)
    feat_cols = _unique_cols(feat_cols + list(PA_CTX_FEATURES))

    base_feats, hmm_feats, garch_feats, hsmm_feats, egarch_feats, tcn_feats, mamba_feats = _split_feature_groups(feat_cols)
    print(f"\n  Feature columns: {len(feat_cols)}")
    print(f"    Base PA/OR:  {len(base_feats)}")
    print(f"    HMM-style:   {len(hmm_feats)}")
    print(f"    GARCH-style: {len(garch_feats)}")
    print(f"    HSMM-style:  {len(hsmm_feats)}")
    print(f"    EGARCH-style:{len(egarch_feats)}")
    print(f"    TCN-derived: {len(tcn_feats)}")
    print(f"    Mamba-derived (experimental): {len(mamba_feats)}")
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



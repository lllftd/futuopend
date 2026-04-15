from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from core.trainers.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1B_DQ_PRED_FILE,
    L1B_EDGE_PRED_FILE,
    L1B_META_FILE,
    L1B_OUTPUT_CACHE_FILE,
    L1B_SCHEMA_VERSION,
    MODEL_DIR,
    TRAIN_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _mfe_mae_atr_arrays,
    _numeric_feature_cols_for_matrix,
    _options_target_config,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.trainers.val_metrics_extra import brier_binary
from core.trainers.stack_v2_common import (
    build_stack_time_splits,
    compute_cross_asset_context,
    diagnose_l1b_leakage,
    log_label_baseline,
    save_output_cache,
)


L1B_CLUSTER_COLS = [
    "l1b_cluster_prob_0",
    "l1b_cluster_prob_1",
    "l1b_cluster_prob_2",
    "l1b_cluster_prob_3",
    "l1b_cluster_prob_4",
]
L1B_LATENT_HEADS = [
    "l1b_latent_0",
    "l1b_latent_1",
    "l1b_latent_2",
    "l1b_latent_3",
    "l1b_novelty_score",
    "l1b_regime_change_score",
]
L1B_LATENT_EMBED_COLS = [col for col in L1B_LATENT_HEADS if col.startswith("l1b_latent_")]
L1B_MODEL_HEADS: list[str] = []
L1B_DIRECT_SEMANTIC_COLS = [
    "l1b_breakout_quality",
    "l1b_mean_reversion_setup",
    "l1b_trend_strength",
    "l1b_range_reversal_setup",
    "l1b_failed_breakout_setup",
    "l1b_setup_alignment",
    "l1b_follow_through_score",
    "l1b_liquidity_score",
]
L1B_SUPERVISED_REGRESSOR_COLS = ("l1b_edge_pred", "l1b_dq_pred")

# Extra PA/BO columns appended only for L1b supervised heads (not exported to L2). Source col must exist on frame.
L1B_ATOMIC_SUPERVISED_SOURCES: tuple[tuple[str, str], ...] = (
    ("l1b_atom_bo_range_compress", "bo_range_compress"),
    ("l1b_atom_bo_body_growth", "bo_body_growth"),
    ("l1b_atom_bo_bb_width", "bo_bb_width"),
    ("l1b_atom_bo_gap_signal", "bo_gap_signal"),
    ("l1b_atom_bo_consec_dir", "bo_consec_dir"),
    ("l1b_atom_pa_ctx_pullback_long", "pa_ctx_setup_pullback_long"),
    ("l1b_atom_pa_ctx_pullback_short", "pa_ctx_setup_pullback_short"),
    ("l1b_atom_pa_ctx_range_long", "pa_ctx_setup_range_long"),
    ("l1b_atom_pa_ctx_range_short", "pa_ctx_setup_range_short"),
    ("l1b_atom_pa_hsmm_switch", "pa_hsmm_switch_hazard"),
)
L1B_UNSUPERVISED_COLS = list(L1B_CLUSTER_COLS) + list(L1B_LATENT_HEADS)
L1B_BINARY_HEADS: tuple[str, ...] = ()
L1B_DIRECT_CONTEXT_COLS = [
    "l1b_sector_relative_strength",
    "l1b_correlation_regime",
    "l1b_market_breadth",
]
# Direct condition semantics are exported to L2; directional rule heads are removed from the contract.
L1B_RULE_DIRECT_COLS = list(L1B_DIRECT_SEMANTIC_COLS)
L1B_EXPORT_DETERMINISTIC_COLS: tuple[str, ...] = tuple(L1B_DIRECT_SEMANTIC_COLS)
L1B_OUTPUT_COLS = (
    list(L1B_UNSUPERVISED_COLS)
    + list(L1B_SUPERVISED_REGRESSOR_COLS)
    + list(L1B_EXPORT_DETERMINISTIC_COLS)
)


@dataclass
class L1BTrainingBundle:
    models: dict[str, lgb.Booster]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _select_l1b_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """Numeric PA / BO / orthogonal-stat columns only; does not include ``decision_*`` labels or raw OHLCV."""
    keep = []
    orthogonal_stat_cols = [
        "pa_hmm_state",
        "pa_hmm_transition_pressure",
        "pa_garch_vol",
        "pa_garch_shock",
        "pa_garch_vol_of_vol",
        "pa_hsmm_duration_norm",
        "pa_hsmm_remaining_duration",
        "pa_hsmm_switch_hazard",
        "pa_hsmm_duration_percentile",
        "pa_egarch_leverage_effect",
        "pa_egarch_downside_vol_ratio",
        "pa_egarch_vol_asymmetry",
        "pa_egarch_std_residual",
    ]
    base_pref = [
        "bo_body_atr",
        "bo_range_atr",
        "bo_vol_spike",
        "bo_close_extremity",
        "bo_wick_imbalance",
        "bo_range_compress",
        "bo_body_growth",
        "bo_gap_signal",
        "bo_consec_dir",
        "bo_inside_prior",
        "bo_pressure_diff",
        "bo_or_dist",
        "bo_bb_width",
        "bo_atr_zscore",
        "pa_ctx_setup_long",
        "pa_ctx_setup_short",
        "pa_ctx_setup_trend_long",
        "pa_ctx_setup_trend_short",
        "pa_ctx_setup_pullback_long",
        "pa_ctx_setup_pullback_short",
        "pa_ctx_setup_range_long",
        "pa_ctx_setup_range_short",
        "pa_ctx_setup_failed_breakout_long",
        "pa_ctx_setup_failed_breakout_short",
        "pa_ctx_follow_through_long",
        "pa_ctx_follow_through_short",
        "pa_ctx_range_pressure",
        "pa_ctx_structure_veto",
        "pa_ctx_premise_break_long",
        "pa_ctx_premise_break_short",
        "pa_vol_rvol",
        "pa_vol_momentum",
        "pa_bo_wick_imbalance",
        "pa_bo_close_extremity",
        "pa_lead_macd_hist_slope",
        "pa_lead_rsi_slope",
        "pa_bo_dist_vwap",
        "pa_struct_swing_range_atr",
        "pa_vol_exhaustion_climax",
        "pa_vol_zscore_20",
        "pa_vol_evr_ratio",
        "pa_vol_absorption_bull",
        "pa_vol_absorption_bear",
    ]
    keep.extend([c for c in base_pref if c in df.columns])
    keep.extend([c for c in orthogonal_stat_cols if c in feat_cols and c in df.columns])
    _optional_structure = [
        "pa_struct_score",
        "pa_struct_break_up",
        "pa_struct_break_down",
        "pa_struct_leg_count",
        "pa_structure_clarity",
        "pa_struct_hh_count",
        "pa_struct_hl_count",
        "pa_struct_lh_count",
        "pa_struct_ll_count",
    ]
    keep.extend([c for c in _optional_structure if c in df.columns and c not in keep])
    _allow = os.environ.get("L1B_EXTRA_FEATURE_ALLOWLIST", "").strip()
    if _allow:
        feat_set = set(feat_cols)
        for name in [s.strip() for s in _allow.split(",") if s.strip()]:
            if name in df.columns and name not in keep and name in feat_set:
                keep.append(name)
    ts = pd.to_datetime(df["time_key"])
    minutes = (ts.dt.hour * 60 + ts.dt.minute).astype(np.float32)
    df["l1b_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)
    keep.append("l1b_session_progress")
    return _numeric_feature_cols_for_matrix(df, keep)


def _l1b_attach_atomic_supervised_columns(df: pd.DataFrame, base_feature_cols: list[str]) -> list[str]:
    """PA/BO atomics for edge/dq boosters only. Skips sources already in ``base_feature_cols`` (no duplicates)."""
    base_set = set(base_feature_cols)
    added: list[str] = []
    for out_name, src in L1B_ATOMIC_SUPERVISED_SOURCES:
        if src in base_set or src not in df.columns:
            continue
        df[out_name] = pd.to_numeric(df[src], errors="coerce").fillna(0.0).astype(np.float32)
        added.append(out_name)
    if len(added) < 4 and {"bo_range_atr", "bo_body_atr"}.issubset(df.columns):
        if "l1b_atom_range_over_body" not in df.columns and "l1b_atom_range_over_body" not in base_set:
            body = np.abs(pd.to_numeric(df["bo_body_atr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
            rng = pd.to_numeric(df["bo_range_atr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            df["l1b_atom_range_over_body"] = (rng / np.maximum(body, 1e-6)).clip(0.0, 50.0).astype(np.float32)
            added.append("l1b_atom_range_over_body")
    return added


def _l1b_cross_context_reliable(df: pd.DataFrame) -> bool:
    n_symbols = int(pd.Series(df["symbol"]).nunique(dropna=True))
    n_rows = len(df)
    return n_symbols >= 2 and n_rows >= 200


def _col_f32(df: pd.DataFrame, name: str) -> np.ndarray:
    """Read a numeric column as float32; missing column -> zeros (``df.get(..., 0.0)`` is a scalar and breaks ``fillna``)."""
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _clip01(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _robust_clipped_zscore(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    clip_z: float = 3.0,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).ravel()
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[fit]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = max(1.4826 * mad, 0.25 * float(np.std(finite)), 1e-3)
    z = (arr - med) / scale
    return np.clip(z, -float(clip_z), float(clip_z)).astype(np.float32)


def _train_only_percentile_map(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).ravel()
    bad = ~np.isfinite(arr)
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = np.sort(arr[fit])
    if finite.size == 0:
        finite = np.sort(arr[np.isfinite(arr)])
    if finite.size == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    left = np.searchsorted(finite, arr, side="left").astype(np.float32)
    right = np.searchsorted(finite, arr, side="right").astype(np.float32)
    pct = 0.5 * (left + right) / max(float(finite.size), 1.0)
    pct[bad] = 0.5
    return np.clip(pct, 0.0, 1.0).astype(np.float32)


def _rank_normalized_target(
    x: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    tail_boost: float = 0.0,
) -> np.ndarray:
    z = _robust_clipped_zscore(x, fit_mask=fit_mask, clip_z=3.0)
    pct = _train_only_percentile_map(z, fit_mask=fit_mask)
    if tail_boost > 0.0:
        upper = np.clip((pct - 0.80) / 0.20, 0.0, 1.0)
        pct = np.clip((1.0 - float(tail_boost)) * pct + float(tail_boost) * upper, 0.0, 1.0)
    return pct.astype(np.float32)


def _upper_tail_sample_weight(
    y: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    start_pct: float = 0.80,
    alpha: float = 2.0,
) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32).ravel()
    fit = np.isfinite(arr)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[fit]
    if finite.size == 0:
        return np.ones_like(arr, dtype=np.float32)
    cutoff = float(np.quantile(finite, float(np.clip(start_pct, 0.5, 0.99))))
    denom = max(1.0 - cutoff, 1e-3)
    safe = np.where(np.isfinite(arr), arr, cutoff).astype(np.float32, copy=False)
    tail = np.clip((safe - cutoff) / denom, 0.0, 1.0)
    weights = 1.0 + float(alpha) * tail
    return weights.astype(np.float32)


def _env_float_clipped(key: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(key, "").strip()
    val = default if not raw else float(raw)
    return float(np.clip(val, lo, hi))


def _fit_train_quantile_range(values: np.ndarray, fit_mask: np.ndarray, *, q_low: float = 0.05, q_high: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32).ravel()
    fit = np.asarray(fit_mask, dtype=bool).ravel() & np.isfinite(arr)
    finite = arr[fit]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def _scale_by_train_quantiles(values: np.ndarray, fit_mask: np.ndarray, *, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    lo, hi = _fit_train_quantile_range(values, fit_mask, q_low=q_low, q_high=q_high)
    arr = np.asarray(values, dtype=np.float32)
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _build_l1b_candidate_scores(df: pd.DataFrame, direct_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    trend = np.maximum(_col_f32(df, "pa_ctx_setup_trend_long"), _col_f32(df, "pa_ctx_setup_trend_short"))
    pullback = np.maximum(_col_f32(df, "pa_ctx_setup_pullback_long"), _col_f32(df, "pa_ctx_setup_pullback_short"))
    follow = np.maximum(_col_f32(df, "pa_ctx_follow_through_long"), _col_f32(df, "pa_ctx_follow_through_short"))
    failed = np.maximum(_col_f32(df, "pa_ctx_setup_failed_breakout_long"), _col_f32(df, "pa_ctx_setup_failed_breakout_short"))
    premise_break = np.maximum(_col_f32(df, "pa_ctx_premise_break_long"), _col_f32(df, "pa_ctx_premise_break_short"))
    structure_veto = _col_f32(df, "pa_ctx_structure_veto")
    range_pressure = _col_f32(df, "pa_ctx_range_pressure")
    inside = _clip01(_col_f32(df, "bo_inside_prior"))
    wick = _clip01(_col_f32(df, "bo_wick_imbalance"))
    close_ext = _clip01(_col_f32(df, "bo_close_extremity"))
    vol_spike = _clip01(np.clip(_col_f32(df, "bo_vol_spike"), 0.0, 2.0) / 2.0)
    atr_z = _clip01(np.clip(_col_f32(df, "bo_atr_zscore"), 0.0, 3.0) / 3.0)
    garch_shock = _clip01(np.clip(_col_f32(df, "pa_garch_shock"), 0.0, 2.0) / 2.0)
    downside = _clip01((np.clip(_col_f32(df, "pa_egarch_downside_vol_ratio"), 0.8, 2.0) - 0.8) / 1.2)
    std_resid = _clip01(np.clip(np.abs(_col_f32(df, "pa_egarch_std_residual")), 0.0, 3.0) / 3.0)
    vol_rvol = _clip01((np.clip(_col_f32(df, "pa_vol_rvol"), 0.5, 2.5) - 0.5) / 2.0)

    pullback_score = _clip01(
        0.34 * trend
        + 0.24 * pullback
        + 0.18 * follow
        + 0.10 * direct_outputs["l1b_trend_strength"]
        + 0.08 * inside
        + 0.06 * (1.0 - 0.5 * structure_veto)
    )
    failure_score = _clip01(
        0.28 * failed
        + 0.24 * premise_break
        + 0.18 * structure_veto
        + 0.12 * wick
        + 0.10 * garch_shock
        + 0.08 * atr_z
    )
    shock_score = _clip01(
        0.24 * garch_shock
        + 0.18 * downside
        + 0.18 * std_resid
        + 0.15 * vol_spike
        + 0.13 * atr_z
        + 0.12 * vol_rvol
    )
    return {
        "l1b_pullback_setup": pullback_score.astype(np.float32, copy=False),
        "l1b_failure_risk": failure_score.astype(np.float32, copy=False),
        "l1b_shock_risk": shock_score.astype(np.float32, copy=False),
    }


def _build_l1b_semantic_binary_targets(
    raw_targets: dict[str, np.ndarray],
    candidate_scores: dict[str, np.ndarray],
    *,
    train_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    cfg = {
        "l1b_pullback_setup": {"cand_thr": 0.30, "pos_frac": 0.35},
        "l1b_failure_risk": {"cand_thr": 0.25, "pos_frac": 0.30},
        "l1b_shock_risk": {"cand_thr": 0.20, "pos_frac": 0.20},
    }
    labels: dict[str, np.ndarray] = {}
    candidate_masks: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, float]] = {}
    train = np.asarray(train_mask, dtype=bool).ravel()
    for name, raw in raw_targets.items():
        cand_score = np.asarray(candidate_scores[name], dtype=np.float32).ravel()
        cand_thr = float(cfg[name]["cand_thr"])
        pos_frac = float(cfg[name]["pos_frac"])
        cand_mask = cand_score >= cand_thr
        raw_arr = np.asarray(raw, dtype=np.float32).ravel()
        fit = cand_mask & train & np.isfinite(raw_arr)
        fit_idx = np.flatnonzero(fit)
        fit_vals = raw_arr[fit]
        if fit_vals.size < 50:
            labels[name] = np.zeros(len(cand_score), dtype=np.float32)
            candidate_masks[name] = cand_mask.astype(bool)
            meta[name] = {
                "candidate_threshold": cand_thr,
                "positive_fraction": pos_frac,
                "raw_positive_threshold": float("nan"),
                "candidate_train_coverage": float(np.mean(cand_mask[train])) if train.any() else 0.0,
                "candidate_train_rows": float(fit_vals.size),
                "positive_rate_in_candidate_train": 0.0,
            }
            continue

        pos_count = int(round(fit_vals.size * pos_frac))
        pos_count = int(np.clip(pos_count, 1, max(1, fit_vals.size - 1)))
        # Break large weak-label ties deterministically so we keep both classes inside the candidate set.
        order = np.lexsort((fit_idx.astype(np.int64), cand_score[fit], raw_arr[fit]))
        pos_idx = fit_idx[order[-pos_count:]]
        y = np.zeros(len(cand_score), dtype=np.float32)
        y[pos_idx] = 1.0
        y[~cand_mask] = 0.0

        raw_pos_cutoff = float(raw_arr[pos_idx].min()) if pos_idx.size else float("nan")
        labels[name] = y
        candidate_masks[name] = cand_mask.astype(bool)
        meta[name] = {
            "candidate_threshold": cand_thr,
            "positive_fraction": pos_frac,
            "raw_positive_threshold": raw_pos_cutoff,
            "candidate_train_coverage": float(np.mean(cand_mask[train])) if train.any() else 0.0,
            "candidate_train_rows": float(fit_vals.size),
            "positive_rate_in_candidate_train": float(np.mean(y[fit])) if fit.any() else 0.0,
        }
    return labels, candidate_masks, meta


def _fit_l1b_latent_block(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    class _L1BTabularAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    latent_cols = list(feature_cols)
    X = df[latent_cols].to_numpy(dtype=np.float32, copy=False)
    train = np.asarray(train_mask, dtype=bool).ravel()
    X_train = X[train]
    mean = np.mean(X_train, axis=0).astype(np.float32)
    scale = np.std(X_train, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    Xz = ((X - mean) / scale).astype(np.float32)
    Xz_train = Xz[train]
    n_comp = min(len(L1B_LATENT_EMBED_COLS), max(2, Xz_train.shape[1]))
    hidden_dim = max(32, min(128, 8 * n_comp))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = _L1BTabularAutoencoder(Xz_train.shape[1], n_comp, hidden_dim).to(device)
    ds = TensorDataset(torch.from_numpy(Xz_train))
    dl = DataLoader(ds, batch_size=min(512, max(64, len(Xz_train))), shuffle=True)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 4 if FAST_TRAIN_MODE else 12
    ae.train()
    for _ in range(max(1, epochs)):
        for (xb,) in dl:
            xb = xb.to(device)
            noisy = xb + 0.05 * torch.randn_like(xb)
            z_b, recon_b = ae(noisy)
            loss = F.smooth_l1_loss(recon_b, xb) + 0.02 * torch.mean(z_b.pow(2))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
    ae.eval()
    emb_parts: list[np.ndarray] = []
    recon_parts: list[np.ndarray] = []
    with torch.no_grad():
        full_x = torch.from_numpy(Xz)
        for start in range(0, len(Xz), 2048):
            z_b, recon_b = ae(full_x[start : start + 2048].to(device))
            emb_parts.append(z_b.cpu().numpy().astype(np.float32))
            recon_parts.append(recon_b.cpu().numpy().astype(np.float32))
    emb = np.concatenate(emb_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
    resid = np.mean((Xz - recon) ** 2, axis=1).astype(np.float32)
    novelty_lo, novelty_hi = _fit_train_quantile_range(resid, train_mask, q_low=0.05, q_high=0.95)
    novelty = np.clip((resid - novelty_lo) / max(novelty_hi - novelty_lo, 1e-6), 0.0, 1.0).astype(np.float32)
    latent_var = np.var(emb[train], axis=0).astype(np.float32) if np.any(train) else np.ones(n_comp, dtype=np.float32)
    explained = (latent_var / max(float(np.sum(latent_var)), 1e-8)).astype(np.float32)
    out = pd.DataFrame(index=df.index)
    for i in range(n_comp):
        out[L1B_LATENT_EMBED_COLS[i]] = emb[:, i].astype(np.float32, copy=False)
    for i in range(n_comp, len(L1B_LATENT_EMBED_COLS)):
        out[L1B_LATENT_EMBED_COLS[i]] = np.zeros(len(df), dtype=np.float32)
    out["l1b_novelty_score"] = novelty.astype(np.float32, copy=False)
    meta = {
        "feature_cols": latent_cols,
        "mean": mean,
        "scale": scale,
        "autoencoder_state_dict": {k: v.detach().cpu() for k, v in ae.state_dict().items()},
        "latent_dim": int(n_comp),
        "hidden_dim": int(hidden_dim),
        "explained_variance_ratio": explained,
        "novelty_lo": float(novelty_lo),
        "novelty_hi": float(novelty_hi),
    }
    return out, meta


def _apply_l1b_latent_block(df: pd.DataFrame, latent_meta: dict[str, Any]) -> pd.DataFrame:
    feature_cols = list(latent_meta.get("feature_cols") or [])
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    mean = np.asarray(latent_meta.get("mean"), dtype=np.float32)
    scale = np.asarray(latent_meta.get("scale"), dtype=np.float32)
    Xz = ((X - mean) / np.where(scale > 1e-6, scale, 1.0)).astype(np.float32)
    latent_dim = int(latent_meta.get("latent_dim", len(L1B_LATENT_EMBED_COLS)))
    hidden_dim = int(latent_meta.get("hidden_dim", max(32, min(128, 8 * latent_dim))))

    class _L1BTabularAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    ae = _L1BTabularAutoencoder(Xz.shape[1], latent_dim, hidden_dim)
    ae.load_state_dict(latent_meta.get("autoencoder_state_dict") or {})
    ae.eval()
    with torch.no_grad():
        emb_t, recon_t = ae(torch.from_numpy(Xz))
    emb = emb_t.numpy().astype(np.float32)
    recon = recon_t.numpy().astype(np.float32)
    resid = np.mean((Xz - recon) ** 2, axis=1).astype(np.float32)
    novelty_lo = float(latent_meta.get("novelty_lo", 0.0))
    novelty_hi = float(latent_meta.get("novelty_hi", 1.0))
    novelty = np.clip((resid - novelty_lo) / max(novelty_hi - novelty_lo, 1e-6), 0.0, 1.0).astype(np.float32)
    out = pd.DataFrame(index=work.index)
    for i in range(emb.shape[1]):
        out[L1B_LATENT_EMBED_COLS[i]] = emb[:, i].astype(np.float32, copy=False)
    for i in range(emb.shape[1], len(L1B_LATENT_EMBED_COLS)):
        out[L1B_LATENT_EMBED_COLS[i]] = np.zeros(len(work), dtype=np.float32)
    out["l1b_novelty_score"] = novelty.astype(np.float32, copy=False)
    return out


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    expv = np.exp(shifted)
    denom = np.sum(expv, axis=1, keepdims=True)
    return (expv / np.maximum(denom, 1e-8)).astype(np.float32)


def _fit_l1b_kmeans(latent_train: np.ndarray, *, n_clusters: int, n_iter: int = 25) -> np.ndarray:
    X = np.asarray(latent_train, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((n_clusters, max(X.shape[1] if X.ndim == 2 else 1, 1)), dtype=np.float32)
    n_clusters = int(np.clip(n_clusters, 1, X.shape[0]))
    order = np.argsort(X[:, 0], kind="mergesort")
    seeds = np.linspace(0, len(order) - 1, num=n_clusters, dtype=int)
    centroids = X[order[seeds]].astype(np.float32, copy=True)
    for _ in range(int(max(n_iter, 1))):
        diff = X[:, None, :] - centroids[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_centroids = np.array(centroids, copy=True)
        for k in range(n_clusters):
            members = X[assign == k]
            if members.size:
                new_centroids[k] = np.mean(members, axis=0).astype(np.float32)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids.astype(np.float32)


def _cluster_probs_from_latent(latent: np.ndarray, centroids: np.ndarray, *, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    diff = latent[:, None, :] - centroids[None, :, :]
    dist2 = np.sum(diff * diff, axis=2).astype(np.float32)
    logits = -dist2 / max(float(temperature) ** 2, 1e-6)
    probs = _softmax_rows(logits)
    return probs.astype(np.float32), dist2.astype(np.float32)


def _symbol_time_order(df: pd.DataFrame) -> np.ndarray:
    symbols = pd.Series(df["symbol"]).astype(str).to_numpy()
    times = pd.to_datetime(df["time_key"]).astype("int64").to_numpy()
    return np.lexsort((times, symbols))


def _cluster_regime_change_score(df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float32)
    symbols = pd.Series(df["symbol"]).astype(str).to_numpy()
    order = _symbol_time_order(df)
    prev_idx = -1
    prev_sym = None
    for idx in order:
        sym = symbols[idx]
        if prev_sym == sym and prev_idx >= 0:
            out[idx] = 0.5 * float(np.sum(np.abs(probs[idx] - probs[prev_idx])))
        else:
            out[idx] = 0.0
        prev_idx = int(idx)
        prev_sym = sym
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _cluster_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent.astype(np.float32)


def _log_l1b_unsupervised_diagnostics(
    outputs: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    cluster_cols: list[str],
    latent_cols: list[str],
) -> None:
    train = np.asarray(train_mask, dtype=bool).ravel()
    val = np.asarray(val_mask, dtype=bool).ravel()
    cluster_train = outputs.loc[train, cluster_cols].to_numpy(dtype=np.float32, copy=False)
    cluster_val = outputs.loc[val, cluster_cols].to_numpy(dtype=np.float32, copy=False)
    if cluster_train.size:
        train_share = np.mean(cluster_train, axis=0)
        val_share = np.mean(cluster_val, axis=0) if cluster_val.size else np.zeros_like(train_share)
        drift = float(np.mean(np.abs(train_share - val_share)))
        train_ent = float(np.mean(_cluster_entropy(cluster_train)))
        val_ent = float(np.mean(_cluster_entropy(cluster_val))) if cluster_val.size else float("nan")
        print("  [L1b] unsupervised diagnostic — clusters", flush=True)
        print(
            f"    train_share={np.round(train_share, 4).tolist()}  val_share={np.round(val_share, 4).tolist()}  "
            f"train_entropy={train_ent:.4f}  val_entropy={val_ent:.4f}  train_val_drift={drift:.4f}",
            flush=True,
        )
    if latent_cols:
        latent_train = outputs.loc[train, latent_cols].to_numpy(dtype=np.float32, copy=False)
        latent_std = np.std(latent_train, axis=0) if latent_train.size else np.zeros(len(latent_cols), dtype=np.float32)
        print(
            f"  [L1b] unsupervised diagnostic — latent train_std={np.round(latent_std, 4).tolist()}",
            flush=True,
        )
    for extra_col in ["l1b_novelty_score", "l1b_regime_change_score"]:
        if extra_col in outputs.columns:
            train_vals = outputs.loc[train, extra_col].to_numpy(dtype=np.float32, copy=False)
            val_vals = outputs.loc[val, extra_col].to_numpy(dtype=np.float32, copy=False)
            train_pcts = np.percentile(train_vals, [5, 25, 50, 75, 95]).tolist() if train_vals.size else [0.0] * 5
            val_pcts = np.percentile(val_vals, [5, 25, 50, 75, 95]).tolist() if val_vals.size else [0.0] * 5
            print(
                f"  [L1b] unsupervised diagnostic — {extra_col}: "
                f"train_pcts={np.round(train_pcts, 4).tolist()}  val_pcts={np.round(val_pcts, 4).tolist()}",
                flush=True,
            )


def _fit_l1b_unsupervised_block(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    latent_dim = min(4, max(2, len(feature_cols)))
    n_clusters = 5
    latent_outputs, latent_meta = _fit_l1b_latent_block(df, feature_cols, train_mask=train_mask)
    latent_cols = [f"l1b_latent_{i}" for i in range(latent_dim)]
    latent_train = latent_outputs.loc[np.asarray(train_mask, dtype=bool), latent_cols].to_numpy(dtype=np.float32, copy=False)
    centroids = _fit_l1b_kmeans(latent_train, n_clusters=n_clusters)
    train_probs, train_dist2 = _cluster_probs_from_latent(latent_train, centroids, temperature=1.0)
    nearest_train = np.sqrt(np.min(train_dist2, axis=1)) if train_dist2.size else np.array([1.0], dtype=np.float32)
    cluster_temperature = float(max(np.median(nearest_train), 0.35))
    all_latent = latent_outputs[latent_cols].to_numpy(dtype=np.float32, copy=False)
    cluster_probs, _ = _cluster_probs_from_latent(all_latent, centroids, temperature=cluster_temperature)
    novelty = latent_outputs["l1b_novelty_score"].to_numpy(dtype=np.float32, copy=False)
    regime_change_raw = _cluster_regime_change_score(df, cluster_probs)
    rc_lo, rc_hi = _fit_train_quantile_range(regime_change_raw, train_mask, q_low=0.05, q_high=0.95)
    regime_change = np.clip((regime_change_raw - rc_lo) / max(rc_hi - rc_lo, 1e-6), 0.0, 1.0).astype(np.float32)

    out = pd.DataFrame(index=df.index)
    for idx, col in enumerate(L1B_CLUSTER_COLS):
        out[col] = cluster_probs[:, idx].astype(np.float32, copy=False)
    for col in latent_cols:
        out[col] = latent_outputs[col].to_numpy(dtype=np.float32, copy=False)
    out["l1b_novelty_score"] = novelty.astype(np.float32, copy=False)
    out["l1b_regime_change_score"] = regime_change.astype(np.float32, copy=False)

    meta = {
        "feature_cols": list(feature_cols),
        "latent_dim": latent_dim,
        "cluster_cols": list(L1B_CLUSTER_COLS),
        "latent_cols": latent_cols,
        "cluster_centroids": centroids.astype(np.float32),
        "cluster_temperature": cluster_temperature,
        "latent_head_meta": latent_meta,
        "regime_change_lo": float(rc_lo),
        "regime_change_hi": float(rc_hi),
    }
    return out, meta


def _apply_l1b_unsupervised_block(df: pd.DataFrame, unsup_meta: dict[str, Any]) -> pd.DataFrame:
    latent_meta = dict(unsup_meta.get("latent_head_meta") or {})
    latent_outputs = _apply_l1b_latent_block(df, latent_meta)
    latent_cols = list(unsup_meta.get("latent_cols") or [])
    centroids = np.asarray(unsup_meta.get("cluster_centroids"), dtype=np.float32)
    cluster_temperature = float(unsup_meta.get("cluster_temperature", 1.0))
    all_latent = latent_outputs[latent_cols].to_numpy(dtype=np.float32, copy=False)
    cluster_probs, _ = _cluster_probs_from_latent(all_latent, centroids, temperature=cluster_temperature)
    regime_change_raw = _cluster_regime_change_score(df, cluster_probs)
    rc_lo = float(unsup_meta.get("regime_change_lo", 0.0))
    rc_hi = float(unsup_meta.get("regime_change_hi", 1.0))
    regime_change = np.clip((regime_change_raw - rc_lo) / max(rc_hi - rc_lo, 1e-6), 0.0, 1.0).astype(np.float32)

    out = pd.DataFrame(index=df.index)
    for idx, col in enumerate(L1B_CLUSTER_COLS):
        out[col] = cluster_probs[:, idx].astype(np.float32, copy=False)
    for col in latent_cols:
        out[col] = latent_outputs[col].to_numpy(dtype=np.float32, copy=False)
    out["l1b_novelty_score"] = latent_outputs["l1b_novelty_score"].to_numpy(dtype=np.float32, copy=False)
    out["l1b_regime_change_score"] = regime_change.astype(np.float32, copy=False)
    return out


def _candidate_mask_from_scores(head: str, candidate_scores: dict[str, np.ndarray], candidate_meta: dict[str, dict[str, float]]) -> np.ndarray:
    score = np.asarray(candidate_scores.get(head, 0.0), dtype=np.float32).ravel()
    head_meta = candidate_meta.get(head) or {}
    cand_thr = float(head_meta.get("candidate_threshold", 0.5))
    return (score >= cand_thr).astype(bool)


def _l1b_head_feature_cols(head: str, feature_cols: list[str]) -> tuple[list[str], bool]:
    preferred = {
        "l1b_breakout_quality": [
            "bo_body_atr",
            "bo_range_atr",
            "bo_vol_spike",
            "bo_close_extremity",
            "bo_wick_imbalance",
            "bo_body_growth",
            "bo_gap_signal",
            "bo_consec_dir",
            "bo_or_dist",
            "bo_atr_zscore",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "pa_vol_rvol",
            "pa_hsmm_switch_hazard",
            "pa_egarch_std_residual",
        ],
        "l1b_mean_reversion_setup": [
            "pa_ctx_range_pressure",
            "bo_inside_prior",
            "bo_bb_width",
            "bo_or_dist",
            "pa_struct_swing_range_atr",
            "pa_vol_zscore_20",
            "pa_hsmm_duration_norm",
            "pa_hsmm_switch_hazard",
        ],
        "l1b_trend_strength": [
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "bo_consec_dir",
            "bo_body_growth",
            "bo_atr_zscore",
            "pa_vol_momentum",
            "pa_lead_macd_hist_slope",
            "pa_hmm_transition_pressure",
            "pa_hsmm_switch_hazard",
        ],
        "l1b_pullback_setup": [
            "pa_ctx_setup_trend_long",
            "pa_ctx_setup_trend_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "bo_or_dist",
            "bo_inside_prior",
            "pa_struct_swing_range_atr",
            "pa_hsmm_remaining_duration",
            "pa_ctx_range_pressure",
            "bo_bb_width",
            "pa_hmm_transition_pressure",
            "pa_egarch_std_residual",
        ],
        "l1b_range_reversal_setup": [
            "pa_ctx_range_pressure",
            "bo_inside_prior",
            "bo_bb_width",
            "bo_or_dist",
            "bo_close_extremity",
            "pa_vol_absorption_bull",
            "pa_vol_absorption_bear",
            "pa_hsmm_duration_percentile",
        ],
        "l1b_failed_breakout_setup": [
            "bo_wick_imbalance",
            "bo_close_extremity",
            "bo_gap_signal",
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_garch_shock",
            "pa_egarch_vol_asymmetry",
            "pa_hmm_transition_pressure",
            "pa_ctx_structure_veto",
        ],
        "l1b_setup_alignment": [
            "pa_ctx_setup_long",
            "pa_ctx_setup_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_ctx_structure_veto",
            "pa_hsmm_switch_hazard",
            "pa_hsmm_remaining_duration",
            "pa_egarch_std_residual",
        ],
        "l1b_follow_through_score": [
            "bo_consec_dir",
            "bo_body_growth",
            "bo_close_extremity",
            "pa_vol_rvol",
            "pa_vol_momentum",
            "pa_lead_macd_hist_slope",
            "pa_hsmm_switch_hazard",
            "pa_egarch_std_residual",
            "bo_gap_signal",
            "bo_body_atr",
        ],
        "l1b_failure_risk": [
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
            "pa_ctx_structure_veto",
            "pa_ctx_range_pressure",
            "bo_wick_imbalance",
            "pa_garch_shock",
            "pa_egarch_downside_vol_ratio",
            "pa_hsmm_switch_hazard",
            "pa_hmm_transition_pressure",
            "bo_atr_zscore",
        ],
        "l1b_shock_risk": [
            "pa_garch_shock",
            "pa_garch_vol",
            "pa_garch_vol_of_vol",
            "pa_egarch_leverage_effect",
            "pa_egarch_downside_vol_ratio",
            "pa_egarch_vol_asymmetry",
            "pa_egarch_std_residual",
            "pa_hsmm_switch_hazard",
            "pa_hsmm_duration_norm",
            "bo_atr_zscore",
            "bo_vol_spike",
            "pa_vol_zscore_20",
            "pa_hmm_transition_pressure",
            "pa_vol_evr_ratio",
            "bo_range_atr",
            "bo_body_atr",
            "pa_vol_rvol",
            "pa_vol_momentum",
            "pa_ctx_range_pressure",
            "pa_ctx_structure_veto",
            "pa_ctx_premise_break_long",
            "pa_ctx_premise_break_short",
        ],
        "l1b_liquidity_score": [
            "pa_vol_rvol",
            "bo_wick_imbalance",
            "bo_body_atr",
            "bo_range_atr",
            "bo_close_extremity",
            "pa_vol_absorption_bull",
            "pa_vol_absorption_bear",
            "pa_vol_evr_ratio",
            "pa_vol_exhaustion_climax",
            "pa_egarch_std_residual",
        ],
    }.get(head)
    if preferred is None:
        return list(feature_cols), False
    chosen = [c for c in preferred if c in feature_cols]
    min_features = min(8, max(5, len(preferred) // 2))
    if len(chosen) >= min_features:
        return chosen, False
    safe_fallback = [
        c
        for c in feature_cols
        if c
        not in {
            "pa_ctx_setup_long",
            "pa_ctx_setup_short",
            "pa_ctx_setup_trend_long",
            "pa_ctx_setup_trend_short",
            "pa_ctx_setup_pullback_long",
            "pa_ctx_setup_pullback_short",
            "pa_ctx_setup_range_long",
            "pa_ctx_setup_range_short",
            "pa_ctx_setup_failed_breakout_long",
            "pa_ctx_setup_failed_breakout_short",
            "pa_ctx_follow_through_long",
            "pa_ctx_follow_through_short",
        }
    ]
    augmented = list(dict.fromkeys(chosen + safe_fallback[: max(0, min_features - len(chosen) + 4)]))
    return augmented, True


def _build_l1b_direct_semantic_outputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Heuristic PA/BO condition composites exported as L1b condition heads."""
    setup_long = _col_f32(df, "pa_ctx_setup_long")
    setup_short = _col_f32(df, "pa_ctx_setup_short")
    trend_long = _col_f32(df, "pa_ctx_setup_trend_long")
    trend_short = _col_f32(df, "pa_ctx_setup_trend_short")
    pullback_long = _col_f32(df, "pa_ctx_setup_pullback_long")
    pullback_short = _col_f32(df, "pa_ctx_setup_pullback_short")
    range_long = _col_f32(df, "pa_ctx_setup_range_long")
    range_short = _col_f32(df, "pa_ctx_setup_range_short")
    failed_breakout_long = _col_f32(df, "pa_ctx_setup_failed_breakout_long")
    failed_breakout_short = _col_f32(df, "pa_ctx_setup_failed_breakout_short")
    follow_long = _col_f32(df, "pa_ctx_follow_through_long")
    follow_short = _col_f32(df, "pa_ctx_follow_through_short")
    structure_veto = _col_f32(df, "pa_ctx_structure_veto")
    premise_break_long = _col_f32(df, "pa_ctx_premise_break_long")
    premise_break_short = _col_f32(df, "pa_ctx_premise_break_short")
    vol_rvol = _col_f32(df, "pa_vol_rvol")
    wick_imbalance = _col_f32(df, "bo_wick_imbalance")

    breakout_quality = _clip01(
        0.22 * np.clip(_col_f32(df, "bo_body_atr"), 0.0, 2.0) / 2.0
        + 0.18 * np.clip(_col_f32(df, "bo_range_atr"), 0.0, 2.5) / 2.5
        + 0.18 * np.clip(_col_f32(df, "bo_vol_spike"), 0.0, 2.0) / 2.0
        + 0.14 * np.clip(_col_f32(df, "bo_close_extremity"), 0.0, 1.0)
        + 0.12 * np.maximum(follow_long, follow_short)
        + 0.10 * np.maximum(setup_long, setup_short)
        - 0.10 * structure_veto
        - 0.08 * np.maximum(premise_break_long, premise_break_short)
    )
    mean_reversion_setup = _clip01(np.maximum(range_long, range_short))
    trend_strength = _clip01(np.maximum(trend_long, trend_short))
    absorb_mx = np.maximum(_col_f32(df, "pa_vol_absorption_bull"), _col_f32(df, "pa_vol_absorption_bear"))
    close_ext = _clip01(_col_f32(df, "bo_close_extremity"))
    wick_abs = _clip01(np.abs(wick_imbalance))
    failed_mx = np.maximum(failed_breakout_long, failed_breakout_short)
    range_pressure = _clip01(_col_f32(df, "pa_ctx_range_pressure"))
    reversal_micro = _clip01(
        0.28 * absorb_mx
        + 0.22 * close_ext
        + 0.20 * failed_mx
        + 0.18 * wick_abs
        + 0.12 * range_pressure
    )
    range_reversal_setup = _clip01(
        np.maximum(range_long, range_short) * (0.15 + 0.45 * failed_mx + 0.40 * reversal_micro)
    )
    failed_breakout_setup = _clip01(np.maximum(failed_breakout_long, failed_breakout_short))

    long_continuation = np.maximum(trend_long, pullback_long) * follow_long * (1.0 - premise_break_long)
    short_continuation = np.maximum(trend_short, pullback_short) * follow_short * (1.0 - premise_break_short)
    long_reversal = np.maximum(range_long, failed_breakout_long) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_long)
    short_reversal = np.maximum(range_short, failed_breakout_short) * (1.0 - 0.5 * structure_veto) * (1.0 - premise_break_short)
    setup_alignment = _clip01(np.maximum.reduce([long_continuation, short_continuation, long_reversal, short_reversal]))
    follow_through_score = _clip01(np.maximum(follow_long, follow_short))
    liquidity_score = _clip01(0.5 + 0.15 * vol_rvol - 0.10 * wick_imbalance)
    return {
        "l1b_breakout_quality": breakout_quality.astype(np.float32, copy=False),
        "l1b_mean_reversion_setup": mean_reversion_setup.astype(np.float32, copy=False),
        "l1b_trend_strength": trend_strength.astype(np.float32, copy=False),
        "l1b_range_reversal_setup": range_reversal_setup.astype(np.float32, copy=False),
        "l1b_failed_breakout_setup": failed_breakout_setup.astype(np.float32, copy=False),
        "l1b_setup_alignment": setup_alignment.astype(np.float32, copy=False),
        "l1b_follow_through_score": follow_through_score.astype(np.float32, copy=False),
        "l1b_liquidity_score": liquidity_score.astype(np.float32, copy=False),
    }


def _build_l1b_targets(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    direct_outputs = _build_l1b_direct_semantic_outputs(df)
    edge = np.clip(_decision_edge_atr_array(df), -4.0, 4.0).astype(np.float32)
    mfe, mae = _mfe_mae_atr_arrays(df)
    mfe = np.clip(mfe, 0.0, 5.0).astype(np.float32)
    mae = np.clip(mae, 0.0, 4.0).astype(np.float32)
    cfg = _options_target_config()
    horizon = float(max(int(cfg["decision_horizon_bars"]), 1))
    if "decision_peak_bar" in df.columns:
        peak_bar = pd.to_numeric(df["decision_peak_bar"], errors="coerce").fillna(horizon).to_numpy(dtype=np.float32)
    else:
        peak_bar = np.full(len(df), horizon, dtype=np.float32)
    peak_bar = np.clip(peak_bar, 1.0, horizon)
    peak_frac = np.clip(peak_bar / horizon, 0.0, 1.0).astype(np.float32)
    late_peak_bonus = np.clip((peak_frac - 0.25) / 0.75, 0.0, 1.0).astype(np.float32)
    early_peak_bonus = np.clip(1.0 - peak_frac, 0.0, 1.0).astype(np.float32)
    rr = mfe / np.maximum(mae, 0.10)
    pullback_raw = (
        0.52 * np.clip(edge, -1.0, 3.0)
        + 0.18 * np.clip(rr - 1.0, -1.0, 2.0)
        + 0.15 * late_peak_bonus
        - 0.20 * np.clip(mae, 0.0, 3.0)
        + 0.08 * np.clip(mfe, 0.0, 4.0)
    ).astype(np.float32)
    failure_raw = (
        0.54 * np.clip(-edge, -1.0, 3.0)
        + 0.24 * np.clip(mae, 0.0, 3.0)
        + 0.14 * np.clip(1.25 - rr, -1.0, 1.5)
        + 0.08 * early_peak_bonus
        + 0.06 * np.clip(np.abs(edge), 0.0, 3.0)
    ).astype(np.float32)
    shock_raw = (
        0.32 * np.clip(mfe + mae, 0.0, 6.0)
        + 0.24 * np.clip(np.abs(edge), 0.0, 4.0)
        + 0.18 * np.clip(mae, 0.0, 4.0)
        + 0.15 * early_peak_bonus
        + 0.11 * np.clip(np.abs(mfe - mae), 0.0, 4.0)
    ).astype(np.float32)
    raw_targets = {
        "l1b_pullback_setup": pullback_raw,
        "l1b_failure_risk": failure_raw,
        "l1b_shock_risk": shock_raw,
    }
    fit = np.asarray(fit_mask if fit_mask is not None else np.ones(len(df), dtype=bool), dtype=bool)
    candidate_scores = _build_l1b_candidate_scores(df, direct_outputs)
    targets, candidate_masks, target_meta = _build_l1b_semantic_binary_targets(
        raw_targets,
        candidate_scores,
        train_mask=fit,
    )
    cross = compute_cross_asset_context(df)
    cross.index = df.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    return raw_targets, targets, candidate_scores, candidate_masks, target_meta, direct_outputs, cross


def _compute_l1b_deterministic_outputs(
    direct_outputs: dict[str, np.ndarray],
    cross: pd.DataFrame,
    *,
    cross_context_reliable: bool,
) -> pd.DataFrame:
    """Columns merged into the L1b cache / L2 merge: slim deterministic export only."""
    del cross_context_reliable
    n_rows = len(cross.index)
    out = pd.DataFrame(index=cross.index)
    for col in L1B_EXPORT_DETERMINISTIC_COLS:
        values = np.asarray(direct_outputs.get(col, np.zeros(n_rows, dtype=np.float32)), dtype=np.float32).ravel()
        out[col] = values
    return out


def _l1b_rule_diagnostics_frame(
    direct_outputs: dict[str, np.ndarray],
    cross: pd.DataFrame,
    *,
    cross_context_reliable: bool,
) -> pd.DataFrame:
    """Direct condition heads + cross-asset context for train/val correlation logging."""
    n_rows = len(cross.index)
    out = pd.DataFrame(index=cross.index)
    for col in L1B_RULE_DIRECT_COLS:
        values = np.asarray(direct_outputs.get(col, np.zeros(n_rows, dtype=np.float32)), dtype=np.float32).ravel()
        out[col] = values
    for col in L1B_DIRECT_CONTEXT_COLS:
        values = pd.to_numeric(cross[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if not cross_context_reliable:
            values = np.zeros(n_rows, dtype=np.float32)
        out[col] = values
    return out


def _corr1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _log_l1b_semantic_head_diagnostics(
    work: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> None:
    """Train/val distribution and correlation with realized decision-window edge (ATR). Heads are rule-based [0,1] or [-1,1]."""
    fwd = _decision_edge_atr_array(work).astype(np.float64)
    label_ok = _l1b_edge_label_valid(work)
    train = np.asarray(train_mask, dtype=bool).ravel()
    val = np.asarray(val_mask, dtype=bool).ravel()
    print(
        "  [L1b] semantic head diagnostics (deterministic PA/BO rules; corr vs raw decision edge, finite labels only)",
        flush=True,
    )

    def _one_split(head: str, hv: np.ndarray, split: np.ndarray, name: str) -> None:
        hv = np.asarray(hv, dtype=np.float64).ravel()
        dist_m = split & np.isfinite(hv)
        corr_m = split & label_ok & np.isfinite(hv) & np.isfinite(fwd)
        n_d = int(np.sum(dist_m))
        n_c = int(np.sum(corr_m))
        if n_d < 1:
            print(f"    [{name}] {head}: (no rows)", flush=True)
            return
        xs = hv[dist_m]
        p5, p50, p95 = np.percentile(xs, [5.0, 50.0, 95.0]).tolist()
        cor = _corr1d(hv[corr_m], fwd[corr_m]) if n_c >= 30 else float("nan")
        cor_s = f"{cor:.4f}" if np.isfinite(cor) else "nan"
        print(
            f"    [{name}] {head}: n_dist={n_d:,}  mean={float(np.mean(xs)):.4f}  std={float(np.std(xs)):.4f}  "
            f"p5/p50/p95={p5:.4f}/{p50:.4f}/{p95:.4f}  corr(edge)_n={n_c:,}  corr={cor_s}",
            flush=True,
        )

    for group_name, cols in (("direct condition [0,1]", list(L1B_DIRECT_SEMANTIC_COLS)),):
        print(f"  [L1b] — {group_name} —", flush=True)
        for col in cols:
            if col not in outputs.columns:
                print(f"    [train] {col}: (missing column)", flush=True)
                continue
            hv = outputs[col].to_numpy(dtype=np.float64, copy=False)
            _one_split(col, hv, train, "train")
            _one_split(col, hv, val, "val")


def _l1b_val_report(head: str, y_t: np.ndarray, y_p: np.ndarray, *, binary: bool | None = None) -> None:
    """Validation metrics on calibration split (train=pre-CAL_END, val=cal window)."""
    y_t = np.asarray(y_t, dtype=np.float64).ravel()
    y_p = np.asarray(y_p, dtype=np.float64).ravel()
    n = len(y_t)
    print(f"\n  [L1b] val — {head}  (n={n:,})", flush=True)
    if n < 5:
        print("    (skip: too few val rows)", flush=True)
        return

    if binary is None:
        binary = head in L1B_BINARY_HEADS

    if binary:
        yi = np.clip(np.round(y_t), 0, 1).astype(np.int32)
        y_pc = np.clip(y_p, 1e-7, 1.0 - 1e-7)
        try:
            auc = float(roc_auc_score(yi, y_p))
        except ValueError:
            auc = float("nan")
        try:
            ll = float(log_loss(yi, y_pc))
        except ValueError:
            ll = float("nan")
        yhat = (y_p >= 0.5).astype(np.int32)
        cm = confusion_matrix(yi, yhat, labels=[0, 1])
        br = brier_binary(yi.astype(np.float64), y_p)
        print(
            f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  acc@0.5={accuracy_score(yi, yhat):.4f}  "
            f"precision={precision_score(yi, yhat, zero_division=0):.4f}  "
            f"recall={recall_score(yi, yhat, zero_division=0):.4f}  "
            f"F1={f1_score(yi, yhat, zero_division=0):.4f}",
            flush=True,
        )
        print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)
        return

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else float("nan")
    cor = _corr1d(y_t, y_p)
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  corr(y,pred)={cor:.4f}", flush=True)


def _log_l1b_target_diagnostics(
    head: str,
    raw_target: np.ndarray,
    shaped_target: np.ndarray,
    *,
    train_mask: np.ndarray,
) -> None:
    def _summarize(arr: np.ndarray, *, decimals: int = 6) -> tuple[int, int, float, float, list[tuple[float, float]]]:
        vals = np.asarray(arr, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0, 0, 0.0, 0.0, []
        rounded = np.round(vals, decimals=decimals)
        uniq_exact = int(np.unique(vals).size)
        uniq_round = int(np.unique(rounded).size)
        uniq_vals, counts = np.unique(rounded, return_counts=True)
        order = np.argsort(counts)[::-1]
        counts = counts[order]
        uniq_vals = uniq_vals[order]
        top1_share = float(counts[0] / vals.size) if counts.size else 0.0
        top3_share = float(counts[:3].sum() / vals.size) if counts.size else 0.0
        top_vals = [
            (float(uniq_vals[i]), float(counts[i] / vals.size))
            for i in range(min(3, counts.size))
        ]
        return uniq_exact, uniq_round, top1_share, top3_share, top_vals

    tm = np.asarray(train_mask, dtype=bool).ravel()
    raw = np.asarray(raw_target, dtype=np.float32).ravel()[tm]
    shaped = np.asarray(shaped_target, dtype=np.float32).ravel()[tm]
    raw_unique, raw_round_unique, raw_top1, raw_top3, raw_top_vals = _summarize(raw)
    shp_unique, shp_round_unique, shp_top1, shp_top3, shp_top_vals = _summarize(shaped)
    print(f"  [L1b] target diagnostic — {head} [train]", flush=True)
    print(
        f"    raw:    unique={raw_unique:,}  rounded6_unique={raw_round_unique:,}  "
        f"top1_share={raw_top1:.3f}  top3_share={raw_top3:.3f}",
        flush=True,
    )
    if raw_top_vals:
        print(
            f"    raw top rounded values: {[(round(v, 6), round(s, 3)) for v, s in raw_top_vals]}",
            flush=True,
        )
    print(
        f"    shaped: unique={shp_unique:,}  rounded6_unique={shp_round_unique:,}  "
        f"top1_share={shp_top1:.3f}  top3_share={shp_top3:.3f}",
        flush=True,
    )
    if shp_top_vals:
        print(
            f"    shaped top rounded values: {[(round(v, 6), round(s, 3)) for v, s in shp_top_vals]}",
            flush=True,
        )
    if shp_round_unique <= 8 or shp_top3 >= 0.80:
        print(
            "    WARNING: shaped target is highly concentrated; consider classification/ordinal labels or less aggressive shaping.",
            flush=True,
        )


def _log_l1b_candidate_diagnostics(
    head: str,
    candidate_score: np.ndarray,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    *,
    train_mask: np.ndarray,
) -> None:
    train = np.asarray(train_mask, dtype=bool).ravel()
    cand = np.asarray(candidate_mask, dtype=bool).ravel()
    score = np.asarray(candidate_score, dtype=np.float32).ravel()
    y = np.asarray(labels, dtype=np.float32).ravel()
    cand_train = cand & train
    coverage = float(np.mean(cand_train)) if train.any() else 0.0
    pos_rate = float(np.mean(y[cand_train])) if cand_train.any() else 0.0
    score_pcts = np.percentile(score[train], [5, 25, 50, 75, 95]).tolist() if train.any() else [0.0] * 5
    print(f"  [L1b] candidate diagnostic — {head} [train]", flush=True)
    print(
        f"    coverage={coverage:.3f}  candidate_rows={int(np.sum(cand_train)):,}  "
        f"positive_rate_in_candidate={pos_rate:.3f}  score_pcts={np.round(score_pcts, 4).tolist()}",
        flush=True,
    )


def _l1b_edge_dq_lgb_params() -> dict[str, Any]:
    fast = FAST_TRAIN_MODE
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.08 if fast else 0.05,
        "num_leaves": 31 if fast else 63,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 25 if fast else 40,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": _lgbm_n_jobs(),
    }


def _l1b_edge_dq_train_config() -> tuple[int, int]:
    rounds = int(os.environ.get("L1B_EDGE_DQ_BOOST_ROUNDS", str(250 if FAST_TRAIN_MODE else 800)))
    es = int(os.environ.get("L1B_EDGE_DQ_ES_ROUNDS", str(45 if FAST_TRAIN_MODE else 110)))
    return max(50, rounds), max(20, es)


def _l1b_edge_label_valid(work: pd.DataFrame) -> np.ndarray:
    """Rows with a realizable edge label (raw label finite). Excludes tail/invalid rows where label_v2 left NaN (``fillna(0)`` in helpers would otherwise fake a target)."""
    if "decision_net_edge_atr" in work.columns:
        v = pd.to_numeric(work["decision_net_edge_atr"], errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(v)
    if {"decision_mfe_atr", "decision_mae_atr"}.issubset(work.columns):
        mfe = pd.to_numeric(work["decision_mfe_atr"], errors="coerce").to_numpy(dtype=np.float64)
        mae = pd.to_numeric(work["decision_mae_atr"], errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(mfe) & np.isfinite(mae)
    return np.ones(len(work), dtype=bool)


def _l1b_dq_label_valid(work: pd.DataFrame) -> np.ndarray:
    if {"decision_mfe_atr", "decision_mae_atr"}.issubset(work.columns):
        mfe = pd.to_numeric(work["decision_mfe_atr"], errors="coerce").to_numpy(dtype=np.float64)
        mae = pd.to_numeric(work["decision_mae_atr"], errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(mfe) & np.isfinite(mae)
    return np.ones(len(work), dtype=bool)


def _l1b_fit_lgb_regressor(
    label: str,
    y: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    out_path: str,
    label_row_ok: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    train = np.asarray(train_mask, dtype=bool).ravel()
    val = np.asarray(val_mask, dtype=bool).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if label_row_ok is None:
        label_row_ok = np.ones(len(y), dtype=bool)
    else:
        label_row_ok = np.asarray(label_row_ok, dtype=bool).ravel()
    row_ok = np.all(np.isfinite(X), axis=1) & label_row_ok & np.isfinite(y)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
        if sample_weight.shape[0] != len(y):
            raise ValueError(f"L1b {label}: sample_weight length mismatch.")
    fit_tr = train & row_ok
    fit_val = val & row_ok
    rounds, es_rounds = _l1b_edge_dq_train_config()
    params = _l1b_edge_dq_lgb_params()
    if int(np.sum(fit_tr)) < 80 or int(np.sum(fit_val)) < 20:
        raise RuntimeError(
            f"L1b {label}: insufficient rows for edge/dq regressor "
            f"(train={int(np.sum(fit_tr))}, val={int(np.sum(fit_val))})."
        )
    dtrain = lgb.Dataset(
        X[fit_tr],
        label=y[fit_tr],
        weight=sample_weight[fit_tr],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X[fit_val],
        label=y[fit_val],
        weight=sample_weight[fit_val],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, f"[L1b] {label}", first_metric_only=True)
    try:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()
    booster.save_model(out_path)
    pred_tr = booster.predict(X[fit_tr]).astype(np.float64)
    y_tr = y[fit_tr]
    mae_tr = float(mean_absolute_error(y_tr, pred_tr))
    std_tr = float(np.std(y_tr)) if y_tr.size > 1 else 0.0
    pred_v = booster.predict(X[fit_val]).astype(np.float64)
    y_v = y[fit_val]
    mae_v = float(mean_absolute_error(y_v, pred_v))
    std_v = float(np.std(y_v)) if y_v.size > 1 else 0.0
    cor_v = _corr1d(y_v, pred_v)
    gap = mae_v - mae_tr
    print(
        f"  [L1b] {label} booster: train_MAE={mae_tr:.4f}  val_MAE={mae_v:.4f}  gap={gap:+.4f}  "
        f"val_corr={cor_v:.4f}  val_n={int(np.sum(fit_val)):,}  val_target_std={std_v:.4f}  "
        f"train_n={int(np.sum(fit_tr)):,}  train_target_std={std_tr:.4f}  -> {out_path}",
        flush=True,
    )
    return booster


def _fit_l1b_edge_dq_boosters(
    work: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    direct_outputs: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, lgb.Booster], dict[str, str], dict[str, np.ndarray], dict[str, Any]]:
    edge_signed = np.clip(_decision_edge_atr_array(work), -5.0, 5.0).astype(np.float64)
    edge_tgt = np.abs(edge_signed).astype(np.float64)
    mfe, mae = _mfe_mae_atr_arrays(work)
    mode = (os.environ.get("L1B_DQ_TARGET_MODE", "path_balance") or "path_balance").strip().lower()
    mfe_c = np.clip(np.asarray(mfe, dtype=np.float64), 0.0, 8.0)
    mae_c = np.clip(np.asarray(mae, dtype=np.float64), 0.0, 8.0)
    if mode in {"legacy", "mfe_minus_mae"}:
        dq_tgt = np.clip(mfe_c - mae_c, -5.0, 5.0).astype(np.float64)
        dq_desc = "clip(mfe-mae,±5) ATR [legacy]"
        dq_clip_lo, dq_clip_hi = -5.0, 5.0
    else:
        dq_tgt = (mfe_c - mae_c) / np.maximum(mfe_c + mae_c, 1e-6)
        dq_tgt = np.clip(dq_tgt, -1.0, 1.0).astype(np.float64)
        dq_desc = "clip((mfe-mae)/(mfe+mae),±1) path-balance (orthogonal to edge construction)"
        dq_clip_lo, dq_clip_hi = -1.0, 1.0
    X = work[feature_cols].to_numpy(dtype=np.float64, copy=False)
    edge_label_ok = _l1b_edge_label_valid(work)
    dq_label_ok = _l1b_dq_label_valid(work)
    tm = np.asarray(train_mask, dtype=bool).ravel()
    ok_both = tm & edge_label_ok & dq_label_ok & np.isfinite(edge_tgt) & np.isfinite(dq_tgt)
    if int(np.sum(ok_both)) > 50:
        c_ed = _corr1d(edge_tgt[ok_both], dq_tgt[ok_both])
        print(
            f"  [L1b] edge vs dq_target train corr (finite rows n={int(np.sum(ok_both)):,})={c_ed:.4f}  dq_mode={mode!r}",
            flush=True,
        )
    if int(np.sum(tm & edge_label_ok & np.isfinite(edge_signed))) > 50:
        c_sign = _corr1d(edge_tgt[tm & edge_label_ok], edge_signed[tm & edge_label_ok])
        print(
            f"  [L1b] edge opportunity target corr(|signed_edge|, signed_edge)={c_sign:.4f}  "
            "target_semantics='absolute edge magnitude only'",
            flush=True,
        )
    log_label_baseline("l1b_edge_target", edge_tgt[tm & edge_label_ok], task="reg")
    log_label_baseline("l1b_dq_target", dq_tgt[tm & dq_label_ok], task="reg")
    models: dict[str, lgb.Booster] = {}
    model_files: dict[str, str] = {}
    edge_path = os.path.join(MODEL_DIR, L1B_EDGE_PRED_FILE)
    dq_path = os.path.join(MODEL_DIR, L1B_DQ_PRED_FILE)
    models["l1b_edge_pred"] = _l1b_fit_lgb_regressor(
        "l1b_edge_pred",
        edge_tgt,
        X,
        feature_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        out_path=edge_path,
        label_row_ok=edge_label_ok,
    )
    models["l1b_dq_pred"] = _l1b_fit_lgb_regressor(
        "l1b_dq_pred",
        dq_tgt,
        X,
        feature_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        out_path=dq_path,
        label_row_ok=dq_label_ok,
    )
    model_files["l1b_edge_pred"] = L1B_EDGE_PRED_FILE
    model_files["l1b_dq_pred"] = L1B_DQ_PRED_FILE
    preds = {
        "l1b_edge_pred": np.clip(models["l1b_edge_pred"].predict(X), 0.0, 5.0).astype(np.float32),
        "l1b_dq_pred": np.clip(
            models["l1b_dq_pred"].predict(X).astype(np.float64),
            dq_clip_lo,
            dq_clip_hi,
        ).astype(np.float32),
    }
    block_meta = {
        "targets": {
            "l1b_edge_pred": "clip(abs(decision_net_edge_atr or mfe*theta - penalty*mae), 0..5) ATR opportunity magnitude",
            "l1b_dq_pred": dq_desc,
        },
        "edge_target_semantics": "absolute edge magnitude only; no directional sign",
        "dq_target_mode": mode,
        "dq_pred_clip": [dq_clip_lo, dq_clip_hi],
        "boost_rounds_env": "L1B_EDGE_DQ_BOOST_ROUNDS",
        "early_stopping_env": "L1B_EDGE_DQ_ES_ROUNDS",
    }
    if direct_outputs:
        routing_raw = {
            "trend": np.maximum(
                np.asarray(direct_outputs.get("l1b_trend_strength", 0.0), dtype=np.float32),
                np.asarray(direct_outputs.get("l1b_setup_alignment", 0.0), dtype=np.float32),
            ),
            "breakout": np.maximum(
                np.asarray(direct_outputs.get("l1b_breakout_quality", 0.0), dtype=np.float32),
                np.asarray(direct_outputs.get("l1b_follow_through_score", 0.0), dtype=np.float32),
            ),
            "mean_reversion": np.maximum(
                np.asarray(direct_outputs.get("l1b_mean_reversion_setup", 0.0), dtype=np.float32),
                np.asarray(direct_outputs.get("l1b_range_reversal_setup", 0.0), dtype=np.float32),
            ),
        }
        routing_mat = np.column_stack([np.clip(v, 0.0, 1.0) for v in routing_raw.values()]).astype(np.float64)
        routing_mat = routing_mat + 1e-3
        routing_mat = routing_mat / np.maximum(routing_mat.sum(axis=1, keepdims=True), 1e-6)
        expert_names = list(routing_raw.keys())
        block_meta["expert_names"] = expert_names
        block_meta["routing_semantics"] = "rule-driven setup-family routing over trend/breakout/mean_reversion experts"
        for target_name, target_arr, clip_lo, clip_hi in (
            ("l1b_edge_pred", edge_tgt, 0.0, 5.0),
            ("l1b_dq_pred", dq_tgt, dq_clip_lo, dq_clip_hi),
        ):
            blended = preds[target_name].astype(np.float64)
            for j, expert_name in enumerate(expert_names):
                sample_weight = 0.10 + routing_mat[:, j]
                exp_path = os.path.join(MODEL_DIR, f"{target_name}_{expert_name}_expert.txt")
                expert_model = _l1b_fit_lgb_regressor(
                    f"{target_name}_{expert_name}_expert",
                    target_arr,
                    X,
                    feature_cols,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    out_path=exp_path,
                    label_row_ok=(edge_label_ok if target_name == "l1b_edge_pred" else dq_label_ok),
                    sample_weight=sample_weight.astype(np.float32),
                )
                expert_key = f"{target_name}__expert__{expert_name}"
                models[expert_key] = expert_model
                model_files[expert_key] = os.path.basename(exp_path)
                expert_pred = expert_model.predict(X).astype(np.float64)
                expert_pred = np.clip(expert_pred, clip_lo, clip_hi)
                blended = 0.35 * blended + 0.65 * (
                    routing_mat[:, j] * expert_pred + (1.0 - routing_mat[:, j]) * blended
                )
            preds[target_name] = np.clip(blended, clip_lo, clip_hi).astype(np.float32)
    return models, model_files, preds, block_meta


def train_l1b_market_descriptor(df: pd.DataFrame, feat_cols: list[str]) -> L1BTrainingBundle:
    work = df.copy()
    feature_cols = _select_l1b_feature_cols(work, feat_cols)
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(work["time_key"])
    train_mask = splits.train_mask
    val_mask = splits.l2_val_mask
    direct_outputs = _build_l1b_direct_semantic_outputs(work)
    cross = compute_cross_asset_context(work)
    cross.index = work.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    cross_context_reliable = _l1b_cross_context_reliable(work)

    log_layer_banner("[L1b] Tabular market descriptor")
    log_time_key_split(
        "L1b",
        work["time_key"],
        train_mask,
        val_mask,
        train_label="train (t < TRAIN_END)",
        val_label="val (l2_val)",
        extra_note=(
            f"Primary L1b early stopping/reporting uses l2_val end-bars inside [{TRAIN_END}, {CAL_END}); "
            f"full cal remains a secondary diagnostic."
        ),
    )
    log_numpy_x_stats("L1b", X[train_mask], label="X[train]")
    print(f"  [L1b] target/output schema L1B_OUTPUT_COLS count={len(L1B_OUTPUT_COLS)}: {L1B_OUTPUT_COLS}", flush=True)
    atomic_supervised = _l1b_attach_atomic_supervised_columns(work, feature_cols)
    supervised_feature_cols = list(feature_cols) + atomic_supervised
    print(f"  [L1b] input feature count (unsupervised autoencoder/KMeans): {len(feature_cols)}", flush=True)
    print(
        f" supervised augment: +{len(atomic_supervised)} atom cols (internal only; not in L1B_OUTPUT_COLS)  "
        f"examples={atomic_supervised[:5]}{'...' if len(atomic_supervised) > 5 else ''}",
        flush=True,
    )
    print(f"  [L1b] artifact dir: {MODEL_DIR}", flush=True)
    print(f"  [L1b] will write meta/cache: {artifact_path(L1B_META_FILE)} | {artifact_path(L1B_OUTPUT_CACHE_FILE)}", flush=True)
    print(
        "  [L1b] note: L1b heads use in-memory X from this run (not OOF row-by-row unless you change data prep).",
        flush=True,
    )
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    outputs = pd.concat(
        [
            outputs,
            _compute_l1b_deterministic_outputs(
                direct_outputs,
                cross,
                cross_context_reliable=cross_context_reliable,
            ),
        ],
        axis=1,
    )
    rule_diag = _l1b_rule_diagnostics_frame(
        direct_outputs, cross, cross_context_reliable=cross_context_reliable
    )
    print(
        "  [L1b] note: 8 direct condition + 3 cross-asset ctx columns are logged as diagnostics; "
        f"cache/L2 export is {len(L1B_OUTPUT_COLS)} cols (unsup + edge/dq + direct condition heads).",
        flush=True,
    )
    _log_l1b_semantic_head_diagnostics(work, rule_diag, train_mask=train_mask, val_mask=val_mask)
    unsup_outputs, unsup_meta = _fit_l1b_unsupervised_block(work, feature_cols, train_mask=train_mask)
    outputs = pd.concat([outputs, unsup_outputs.reset_index(drop=True)], axis=1)
    print(f"  [L1b] cluster heads: {L1B_CLUSTER_COLS}", flush=True)
    print(f"  [L1b] latent heads: {L1B_LATENT_HEADS}", flush=True)
    print(f"  [L1b] direct condition heads: {L1B_DIRECT_SEMANTIC_COLS}", flush=True)
    print(f"  [L1b] rule diagnostic context heads: {L1B_DIRECT_CONTEXT_COLS}", flush=True)
    print(
        f"  [L1b] latent explained_variance_ratio="
        f"{np.round(np.asarray(unsup_meta['latent_head_meta']['explained_variance_ratio'], dtype=np.float32), 4).tolist()}  "
        f"cluster_temperature={float(unsup_meta['cluster_temperature']):.4f}",
        flush=True,
    )
    if not cross_context_reliable:
        print(
            "  [L1b] cross-asset context unreliable for diagnostics (need ≥2 symbols & 200 rows); "
            "rule_diag context columns zeroed (not exported to L1b cache).",
            flush=True,
        )
    _log_l1b_unsupervised_diagnostics(
        outputs,
        train_mask=train_mask,
        val_mask=val_mask,
        cluster_cols=list(L1B_CLUSTER_COLS),
        latent_cols=list(L1B_LATENT_EMBED_COLS),
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    l1b_supervised_models, edge_dq_model_files, edge_dq_preds, edge_dq_block_meta = _fit_l1b_edge_dq_boosters(
        work,
        supervised_feature_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        direct_outputs=direct_outputs,
    )
    for k, arr in edge_dq_preds.items():
        outputs[k] = arr
    for col in L1B_OUTPUT_COLS:
        if col not in outputs.columns:
            outputs[col] = 0.0
    _out_keep = ["symbol", "time_key"] + [c for c in L1B_OUTPUT_COLS if c in outputs.columns]
    outputs = outputs[_out_keep]

    meta = {
        "schema_version": L1B_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "supervised_feature_cols": supervised_feature_cols,
        "supervised_atomic_cols": atomic_supervised,
        "output_cols": L1B_OUTPUT_COLS,
        "model_output_cols": list(L1B_SUPERVISED_REGRESSOR_COLS),
        "cluster_output_cols": list(L1B_CLUSTER_COLS),
        "latent_output_cols": list(L1B_LATENT_HEADS),
        "unsupervised_output_cols": list(L1B_UNSUPERVISED_COLS),
        "direct_output_cols": sorted(L1B_EXPORT_DETERMINISTIC_COLS),
        "deterministic_output_cols": list(L1B_EXPORT_DETERMINISTIC_COLS),
        "rule_diagnostic_cols": sorted(L1B_RULE_DIRECT_COLS + L1B_DIRECT_CONTEXT_COLS),
        "deprecated_output_cols": ["l1b_pullback_setup", "l1b_failure_risk", "l1b_shock_risk"],
        "constant_output_values": {},
        "model_files": dict(edge_dq_model_files),
        "head_feature_cols": {h: list(supervised_feature_cols) for h in L1B_SUPERVISED_REGRESSOR_COLS},
        "cross_context_reliable": cross_context_reliable,
        "weak_supervision_semantics": (
            "Binary pullback/failure/shock targets still use candidate scores from full internal rule dict; "
            "exported L1b cache is unsupervised + l1b_edge_pred/l1b_dq_pred + direct condition heads only."
        ),
        "unsupervised_semantics": (
            "deterministic direct descriptors plus denoising-autoencoder latent embeddings, soft cluster posteriors, "
            "novelty from nonlinear reconstruction error, and regime-change from cluster posterior turnover"
        ),
        "latent_head_meta": unsup_meta["latent_head_meta"],
        "unsupervised_block_meta": unsup_meta,
        "supervised_edge_dq_block_meta": edge_dq_block_meta,
        "supervised_edge_dq_semantics": (
            "Independent LightGBM regressors plus rule-routed setup experts: l1b_edge_pred ~ clipped absolute edge magnitude (0..5 ATR opportunity size only); "
            "l1b_dq_pred ~ path-balance (mfe−mae)/(mfe+mae) in ±1 by default (env L1B_DQ_TARGET_MODE=legacy for mfe−mae). "
            "Supervised inputs = tabular base + l1b_atom_* columns (not passed to L2)."
        ),
        "output_cache_file": L1B_OUTPUT_CACHE_FILE,
    }
    with open(os.path.join(MODEL_DIR, L1B_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    print(f"  [L1b] meta saved  -> {os.path.join(MODEL_DIR, L1B_META_FILE)}", flush=True)
    print(f"  [L1b] cache saved -> {cache_path}", flush=True)
    return L1BTrainingBundle(models=l1b_supervised_models, meta=meta, outputs=outputs)


def load_l1b_market_descriptor() -> tuple[dict[str, lgb.Booster], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L1B_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L1B_SCHEMA_VERSION:
        raise RuntimeError(
            "L1b checkpoint is incompatible with the current head contract. "
            f"Retrain L1b so artifacts match schema {L1B_SCHEMA_VERSION}."
        )
    models = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in (meta.get("model_files") or {}).items()
    }
    return models, meta


def infer_l1b_market_descriptor(models: dict[str, lgb.Booster], meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    feature_cols = list(meta["feature_cols"])
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    _l1b_attach_atomic_supervised_columns(work, feature_cols)
    sup_cols = list(meta.get("supervised_feature_cols") or feature_cols)
    dq_meta = meta.get("supervised_edge_dq_block_meta") or {}
    dq_lo, dq_hi = dq_meta.get("dq_pred_clip", [-5.0, 5.0])
    dq_lo, dq_hi = float(dq_lo), float(dq_hi)
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    direct_outputs = _build_l1b_direct_semantic_outputs(work)
    cross = compute_cross_asset_context(work)
    cross.index = work.index
    cross = cross.rename(
        columns={
            "sector_relative_strength": "l1b_sector_relative_strength",
            "correlation_regime": "l1b_correlation_regime",
            "market_breadth": "l1b_market_breadth",
        }
    )
    outputs = pd.concat(
        [
            outputs,
            _compute_l1b_deterministic_outputs(
                direct_outputs,
                cross,
                cross_context_reliable=bool(meta.get("cross_context_reliable", True)),
            ),
        ],
        axis=1,
    )
    unsup_meta = meta.get("unsupervised_block_meta") or {}
    if unsup_meta:
        outputs = pd.concat([outputs, _apply_l1b_unsupervised_block(work, unsup_meta).reset_index(drop=True)], axis=1)
    for col in sup_cols:
        if col not in work.columns:
            work[col] = 0.0
    X_inf = work[sup_cols].to_numpy(dtype=np.float64, copy=False)
    routing_raw = {
        "trend": np.maximum(
            np.asarray(direct_outputs.get("l1b_trend_strength", 0.0), dtype=np.float32),
            np.asarray(direct_outputs.get("l1b_setup_alignment", 0.0), dtype=np.float32),
        ),
        "breakout": np.maximum(
            np.asarray(direct_outputs.get("l1b_breakout_quality", 0.0), dtype=np.float32),
            np.asarray(direct_outputs.get("l1b_follow_through_score", 0.0), dtype=np.float32),
        ),
        "mean_reversion": np.maximum(
            np.asarray(direct_outputs.get("l1b_mean_reversion_setup", 0.0), dtype=np.float32),
            np.asarray(direct_outputs.get("l1b_range_reversal_setup", 0.0), dtype=np.float32),
        ),
    }
    routing_names = list((dq_meta.get("expert_names") or list(routing_raw.keys())))
    routing_mat = np.column_stack([np.clip(routing_raw.get(name, 0.0), 0.0, 1.0) for name in routing_names]).astype(np.float64)
    routing_mat = routing_mat + 1e-3
    routing_mat = routing_mat / np.maximum(routing_mat.sum(axis=1, keepdims=True), 1e-6)
    for head in L1B_SUPERVISED_REGRESSOR_COLS:
        mdl = models.get(head)
        if mdl is not None:
            pred = mdl.predict(X_inf).astype(np.float64)
            for j, expert_name in enumerate(routing_names):
                expert_key = f"{head}__expert__{expert_name}"
                expert_model = models.get(expert_key)
                if expert_model is None:
                    continue
                expert_pred = expert_model.predict(X_inf).astype(np.float64)
                pred = 0.35 * pred + 0.65 * (routing_mat[:, j] * expert_pred + (1.0 - routing_mat[:, j]) * pred)
            if head == "l1b_dq_pred":
                pred = np.clip(pred, dq_lo, dq_hi)
            else:
                pred = np.clip(pred, 0.0, 5.0)
            outputs[head] = pred.astype(np.float32)
    out_cols = list(meta.get("output_cols", L1B_OUTPUT_COLS))
    for col in out_cols:
        if col not in outputs.columns:
            outputs[col] = 0.0
    _keep = ["symbol", "time_key"] + out_cols
    return outputs[_keep]

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import NormalDist
from typing import Any, Callable

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

from core.training.common.constants import (
    BO_FEAT_COLS,
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
from core.training.prep.feature_registry import l1_ctx_stagger_enabled, l1b_base_pref_columns
from core.training.common.lgbm_utils import (
    _tqdm_stream,
    _decision_edge_atr_array,
    _decision_forward_range_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _lgbm_n_jobs,
    _mfe_mae_atr_arrays,
    _numeric_feature_cols_for_matrix,
)
from core.training.logging.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_split
from core.training.common.val_metrics_extra import brier_binary
from core.training.l1b.l1a_bridge import (
    l1b_apply_honest_l1a_fit_mask,
    l1b_baseline_align_to_l1a_pool_enabled,
    l1b_l1a_feature_tier,
    l1b_l1a_inputs_enabled,
    l1b_should_use_shifted_expand_oof_windows,
    l1b_use_honest_l1a_fit_pool,
)
from core.training.common.stack_v2_common import (
    build_stack_time_splits,
    l1_expanding_oof_row_folds,
    l1_oof_folds_from_env,
    l1_oof_mode_from_env,
    l1b_expand_oof_val_windows,
    l2_val_start_time,
    log_label_baseline,
    save_output_cache,
    split_mask_for_tuning_and_report,
    time_blocked_fold_masks,
)
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry


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
L1B_EXPORT_UNSUPERVISED_COLS = list(L1B_CLUSTER_COLS) + [
    "l1b_novelty_score",
    "l1b_regime_change_score",
]
# Cache contract: cluster/latent summaries + staged edge/dq LGBM heads only (no rule-based direct columns).
L1B_OUTPUT_COLS = (
    list(L1B_EXPORT_UNSUPERVISED_COLS)
    + list(L1B_SUPERVISED_REGRESSOR_COLS)
    + ["l1b_edge_candidate_tau"]
)

# HMM / GARCH / HSMM / EGARCH columns used by L1b (full set). ``compact`` mode keeps one informative column per family.
L1B_ORTHO_STAT_COLS_FULL: tuple[str, ...] = (
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
)
L1B_ORTHO_STAT_COLS_COMPACT: tuple[str, ...] = (
    "pa_hmm_transition_pressure",
    "pa_garch_vol",
    "pa_hsmm_switch_hazard",
    "pa_egarch_leverage_effect",
    "pa_egarch_std_residual",
)
L1B_HANDCRAFTED_TAB_COLS: tuple[str, ...] = (
    "l1b_bo_composite",
    "l1b_bo_active",
    "l1b_bo_range_x_garch_vol",
    "l1b_bo_pressure_x_ctx_long",
    "l1b_hmm_state_x_garch_vol",
    "l1b_session_x_bo_range",
)

L1B_EDGE_DQ_PRED_COLS: tuple[str, ...] = ("l1b_edge_pred", "l1b_dq_pred", "l1b_edge_candidate_tau")


@dataclass
class L1BTrainingBundle:
    models: dict[str, lgb.Booster]
    meta: dict[str, Any]
    outputs: pd.DataFrame


def _l1b_resolve_orthogonal_stat_cols(feat_cols: list[str], df: pd.DataFrame) -> tuple[list[str], str]:
    """Orthogonal HMM/GARCH/HSMM/EGARCH inputs: ``full`` or default ``compact`` (lower collinearity)."""
    mode = (os.environ.get("L1B_ORTHO_MODE", "compact") or "compact").strip().lower()
    template = L1B_ORTHO_STAT_COLS_FULL if mode == "full" else L1B_ORTHO_STAT_COLS_COMPACT
    out = [c for c in template if c in feat_cols and c in df.columns]
    return out, mode


def _add_l1b_bo_composite(df: pd.DataFrame) -> None:
    """Single scalar summarizing ``bo_*`` (tanh-squashed weighted mix). Disabled: L1B_BO_COMPOSITE=0."""
    w_raw = os.environ.get("L1B_BO_COMPOSITE_WEIGHTS", "").strip()
    weights: dict[str, float] = {}
    if w_raw:
        for tok in w_raw.split(","):
            tok = tok.strip()
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            weights[k.strip()] = float(v.strip())
    if not weights:
        weights = {k: 1.0 for k in BO_FEAT_COLS}
    acc = np.zeros(len(df), dtype=np.float64)
    ws = 0.0
    for k, w in weights.items():
        if k not in df.columns:
            continue
        x = pd.to_numeric(df[k], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        acc += float(w) * np.tanh(np.clip(x, -5.0, 5.0))
        ws += abs(float(w))
    if ws <= 0:
        df["l1b_bo_composite"] = np.float32(0.0)
    else:
        df["l1b_bo_composite"] = np.clip(acc / ws, -3.0, 3.0).astype(np.float32)


def _ensure_l1b_session_progress_column(df: pd.DataFrame) -> None:
    if "time_key" not in df.columns:
        return
    ts = pd.to_datetime(df["time_key"])
    minutes = (ts.dt.hour * 60 + ts.dt.minute).astype(np.float32)
    df["l1b_session_progress"] = (minutes / (24.0 * 60.0)).astype(np.float32)


def _add_l1b_handcrafted_tabular_features(df: pd.DataFrame, *, wanted: set[str] | None = None) -> list[str]:
    """Explicit LGBM-friendly interactions + BO activity flag. ``wanted`` limits compute (infer); None = all."""
    added: list[str] = []

    def _want(name: str) -> bool:
        return wanted is None or name in wanted

    bo_comp_on = os.environ.get("L1B_BO_COMPOSITE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if bo_comp_on and _want("l1b_bo_composite"):
        _add_l1b_bo_composite(df)
        added.append("l1b_bo_composite")

    if _want("l1b_bo_active") and "bo_range_atr" in df.columns:
        eps = float(os.environ.get("L1B_BO_ACTIVE_EPS", "0.02"))
        r = np.abs(pd.to_numeric(df["bo_range_atr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        df["l1b_bo_active"] = (r > eps).astype(np.float32)
        added.append("l1b_bo_active")

    if _want("l1b_bo_range_x_garch_vol") and "bo_range_atr" in df.columns and "pa_garch_vol" in df.columns:
        a = pd.to_numeric(df["bo_range_atr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = pd.to_numeric(df["pa_garch_vol"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        df["l1b_bo_range_x_garch_vol"] = np.clip(a * b, -1e4, 1e4).astype(np.float32)
        added.append("l1b_bo_range_x_garch_vol")

    if _want("l1b_bo_pressure_x_ctx_long") and "bo_pressure_diff" in df.columns and "pa_ctx_setup_long" in df.columns:
        a = pd.to_numeric(df["bo_pressure_diff"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = pd.to_numeric(df["pa_ctx_setup_long"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        df["l1b_bo_pressure_x_ctx_long"] = np.clip(a * b, -1e4, 1e4).astype(np.float32)
        added.append("l1b_bo_pressure_x_ctx_long")

    if _want("l1b_hmm_state_x_garch_vol") and "pa_hmm_state" in df.columns and "pa_garch_vol" in df.columns:
        a = pd.to_numeric(df["pa_hmm_state"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = pd.to_numeric(df["pa_garch_vol"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        df["l1b_hmm_state_x_garch_vol"] = np.clip(a * b, -1e4, 1e4).astype(np.float32)
        added.append("l1b_hmm_state_x_garch_vol")

    if (
        _want("l1b_session_x_bo_range")
        and "l1b_session_progress" in df.columns
        and "bo_range_atr" in df.columns
    ):
        s = pd.to_numeric(df["l1b_session_progress"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = pd.to_numeric(df["bo_range_atr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        df["l1b_session_x_bo_range"] = np.clip(s * b, -1e4, 1e4).astype(np.float32)
        added.append("l1b_session_x_bo_range")

    return added


def _materialize_l1b_tabular_inputs_for_infer(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Recompute session + handcrafted columns expected by ``feature_cols`` (checkpoint contract)."""
    names = set(feature_cols)
    if "l1b_bo_composite" in names:
        _add_l1b_bo_composite(df)
    if "l1b_session_progress" in names or names & set(L1B_HANDCRAFTED_TAB_COLS):
        _ensure_l1b_session_progress_column(df)
    craft = names & set(L1B_HANDCRAFTED_TAB_COLS)
    if craft:
        _add_l1b_handcrafted_tabular_features(df, wanted=craft)


def _select_l1b_feature_cols(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """Numeric PA / BO / orthogonal-stat columns only; does not include ``decision_*`` labels or raw OHLCV."""
    keep = []
    orthogonal_stat_cols, ortho_mode = _l1b_resolve_orthogonal_stat_cols(feat_cols, df)
    keep.extend([c for c in l1b_base_pref_columns() if c in df.columns])
    keep.extend(orthogonal_stat_cols)
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
        allow_names = [s.strip() for s in _allow.split(",") if s.strip()]
        unknown_pool = [n for n in allow_names if n not in feat_set]
        unknown_df = [n for n in allow_names if n not in df.columns]
        strict = os.environ.get("L1B_EXTRA_ALLOWLIST_STRICT", "").strip().lower() in {"1", "true", "yes"}
        if unknown_pool:
            msg = f"L1B_EXTRA_FEATURE_ALLOWLIST names not in feat_cols pool: {unknown_pool[:12]!r}"
            if strict:
                raise ValueError(msg)
            print(f"  [L1b][warn] {msg}", flush=True)
        if unknown_df:
            msg = f"L1B_EXTRA_FEATURE_ALLOWLIST names missing on frame: {unknown_df[:12]!r}"
            if strict:
                raise ValueError(msg)
            print(f"  [L1b][warn] {msg}", flush=True)
        max_allow = max(0, int(os.environ.get("L1B_EXTRA_ALLOWLIST_MAX", "32")))
        allow_names = [n for n in allow_names if n in feat_set and n in df.columns]
        if max_allow and len(allow_names) > max_allow:
            print(
                f"  [L1b][warn] allowlist truncated {len(allow_names)} → {max_allow} (L1B_EXTRA_ALLOWLIST_MAX)",
                flush=True,
            )
            allow_names = allow_names[:max_allow]
        for name in allow_names:
            if name not in keep:
                keep.append(name)
    _ensure_l1b_session_progress_column(df)
    bo_comp_on = os.environ.get("L1B_BO_COMPOSITE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if bo_comp_on:
        _add_l1b_bo_composite(df)
    handcrafted_on = os.environ.get("L1B_HANDCRAFTED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if handcrafted_on:
        for c in _add_l1b_handcrafted_tabular_features(df, wanted=None):
            if c not in keep:
                keep.append(c)
    elif bo_comp_on and "l1b_bo_composite" in df.columns and "l1b_bo_composite" not in keep:
        keep.append("l1b_bo_composite")
    if "l1b_session_progress" not in keep:
        keep.append("l1b_session_progress")
    print(
        f"  [L1b] ctx_stagger={l1_ctx_stagger_enabled()}  ortho_mode={ortho_mode}  "
        f"ortho_cols={len(orthogonal_stat_cols)}  handcrafted={handcrafted_on}",
        flush=True,
    )
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


def _l1b_supervised_target_mode() -> str:
    """``forward_range`` (default): predict ``decision_forward_range_atr`` for L2 straddle alignment. ``staged_edge``: legacy |edge| from directional edge label."""
    raw = (os.environ.get("L1B_SUPERVISED_TARGET_MODE", "forward_range") or "forward_range").strip().lower()
    if raw in {"legacy", "staged_edge", "edge_abs", "directional_edge"}:
        return "staged_edge"
    if raw in {"forward_range", "range", "vol", "straddle", "realized_range"}:
        return "forward_range"
    raise ValueError(
        f"L1B_SUPERVISED_TARGET_MODE={raw!r} invalid; use forward_range | staged_edge"
    )


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


def _fit_l1b_isolation_forest_novelty_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train IsolationForest on tabular features; map ``-score_samples`` to ``[0,1]`` novelty (high = anomalous)."""
    from sklearn.ensemble import IsolationForest

    train = np.asarray(train_mask, dtype=bool).ravel()
    n = len(df)
    cols: list[np.ndarray] = []
    for c in feature_cols:
        if c not in df.columns:
            cols.append(np.zeros(n, dtype=np.float64))
        else:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    X = np.column_stack(cols) if cols else np.zeros((n, 1), dtype=np.float64)
    X_train = X[train]
    max_fit = min(131072, max(1, int(X_train.shape[0])))
    if X_train.shape[0] > max_fit:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], size=max_fit, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train
    if X_fit.shape[0] < 80:
        return np.zeros(n, dtype=np.float32), {"iforest_disabled": True, "reason": "insufficient_train_rows"}
    n_est = 64 if FAST_TRAIN_MODE else 200
    ms = int(np.clip(min(256, max(16, X_fit.shape[0])), 16, 256))
    iso = IsolationForest(
        n_estimators=n_est,
        max_samples=ms,
        contamination="auto",
        random_state=42,
        n_jobs=max(1, min(4, _lgbm_n_jobs())),
    )
    iso.fit(X_fit)
    scores = iso.score_samples(X).astype(np.float64)
    anom = -scores
    lo, hi = _fit_train_quantile_range(anom.astype(np.float32), train_mask, q_low=0.05, q_high=0.95)
    nov = np.clip((anom - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    meta_if: dict[str, Any] = {
        "iforest_sklearn": iso,
        "iforest_feature_cols": list(feature_cols),
        "iforest_anom_lo": float(lo),
        "iforest_anom_hi": float(hi),
        "iforest_disabled": False,
    }
    return nov, meta_if


def _apply_l1b_isolation_forest_novelty_infer(df: pd.DataFrame, if_meta: dict[str, Any]) -> np.ndarray | None:
    if if_meta.get("iforest_disabled") or if_meta.get("iforest_sklearn") is None:
        return None
    iso = if_meta["iforest_sklearn"]
    lo = float(if_meta.get("iforest_anom_lo", 0.0))
    hi = float(if_meta.get("iforest_anom_hi", 1.0))
    cols = list(if_meta.get("iforest_feature_cols") or [])
    n = len(df)
    parts: list[np.ndarray] = []
    for c in cols:
        if c not in df.columns:
            parts.append(np.zeros(n, dtype=np.float64))
        else:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    X = np.column_stack(parts) if parts else np.zeros((n, 1), dtype=np.float64)
    anom = (-iso.score_samples(X).astype(np.float64)).astype(np.float64)
    return np.clip((anom - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _apply_l1b_latent_block(
    df: pd.DataFrame,
    latent_meta: dict[str, Any],
    *,
    on_substage: Callable[[str], None] | None = None,
) -> pd.DataFrame:
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
    # macOS: large matmul + default PyTorch/BLAS thread pools can hang or thrash; smaller chunks + 1 thread.
    _chunk_default = "512" if sys.platform == "darwin" else "2048"
    chunk = max(256, int(os.environ.get("L1B_LATENT_INFER_CHUNK", _chunk_default)))
    emb_parts: list[np.ndarray] = []
    recon_parts: list[np.ndarray] = []
    chunk_starts = list(range(0, len(Xz), chunk))
    use_nested_tqdm = (
        on_substage is None and _lgb_round_tqdm_enabled() and len(chunk_starts) > 1
    )
    chunk_it: Any = chunk_starts
    if use_nested_tqdm:
        chunk_it = tqdm(
            chunk_starts,
            desc="[L1b] latent AE",
            unit="batch",
            leave=False,
            mininterval=0.2,
            file=_tqdm_stream(),
            dynamic_ncols=True,
        )
    _prev_torch_threads = int(torch.get_num_threads())
    _infer_threads = max(1, int(os.environ.get("L1B_TORCH_INFER_THREADS", "1")))
    try:
        torch.set_num_threads(_infer_threads)
        with torch.inference_mode():
            for j, start in enumerate(chunk_it):
                if on_substage is not None and len(chunk_starts) > 1:
                    on_substage(f"L1b · AE {j + 1}/{len(chunk_starts)}")
                xb = torch.from_numpy(np.ascontiguousarray(Xz[start : start + chunk]))
                z_b, recon_b = ae(xb)
                emb_parts.append(z_b.numpy().astype(np.float32, copy=False))
                recon_parts.append(recon_b.numpy().astype(np.float32, copy=False))
                try:
                    _tqdm_stream().flush()
                except Exception:
                    pass
    finally:
        torch.set_num_threads(_prev_torch_threads)
    emb = np.concatenate(emb_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
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
    ae_nov = latent_outputs["l1b_novelty_score"].to_numpy(dtype=np.float32, copy=False)
    novelty_mode = (os.environ.get("L1B_NOVELTY_MODE", "blend") or "blend").strip().lower()
    iforest_nov: np.ndarray | None = None
    iforest_meta: dict[str, Any] = {"iforest_disabled": True}
    if novelty_mode in {"iforest", "blend", "ae_if"}:
        iforest_nov, iforest_meta = _fit_l1b_isolation_forest_novelty_train(df, feature_cols, train_mask=train_mask)
    w_ae = _env_float_clipped("L1B_NOVELTY_BLEND_W_AE", 0.5, lo=0.0, hi=1.0)
    w_if = _env_float_clipped("L1B_NOVELTY_BLEND_W_IF", 0.5, lo=0.0, hi=1.0)
    if novelty_mode in {"ae", "autoencoder"}:
        novelty = ae_nov
    elif novelty_mode in {"iforest", "iso"}:
        novelty = iforest_nov if iforest_nov is not None else ae_nov
    else:
        if iforest_nov is None or iforest_meta.get("iforest_disabled"):
            novelty = ae_nov
        else:
            s = w_ae + w_if
            novelty = ((w_ae * ae_nov + w_if * iforest_nov) / max(s, 1e-6)).astype(np.float32)
    latent_outputs["l1b_novelty_score"] = novelty
    latent_cols = [f"l1b_latent_{i}" for i in range(latent_dim)]
    latent_train = latent_outputs.loc[np.asarray(train_mask, dtype=bool), latent_cols].to_numpy(dtype=np.float32, copy=False)
    centroids = _fit_l1b_kmeans(latent_train, n_clusters=n_clusters)
    train_probs, train_dist2 = _cluster_probs_from_latent(latent_train, centroids, temperature=1.0)
    nearest_train = np.sqrt(np.min(train_dist2, axis=1)) if train_dist2.size else np.array([1.0], dtype=np.float32)
    cluster_temperature = float(max(np.median(nearest_train), 0.35))
    all_latent = latent_outputs[latent_cols].to_numpy(dtype=np.float32, copy=False)
    cluster_probs, _ = _cluster_probs_from_latent(all_latent, centroids, temperature=cluster_temperature)
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
        "novelty_mode": novelty_mode,
        "novelty_blend_w_ae": float(w_ae),
        "novelty_blend_w_if": float(w_if),
        "iforest_meta": iforest_meta,
    }
    return out, meta


def _apply_l1b_unsupervised_block(
    df: pd.DataFrame,
    unsup_meta: dict[str, Any],
    *,
    on_substage: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    latent_meta = dict(unsup_meta.get("latent_head_meta") or {})
    latent_outputs = _apply_l1b_latent_block(df, latent_meta, on_substage=on_substage)
    novelty_mode = str(unsup_meta.get("novelty_mode") or "blend").strip().lower()
    ae_nov = latent_outputs["l1b_novelty_score"].to_numpy(dtype=np.float32, copy=False)
    ifm = dict(unsup_meta.get("iforest_meta") or {})
    if_n = _apply_l1b_isolation_forest_novelty_infer(df, ifm)
    w_ae = float(unsup_meta.get("novelty_blend_w_ae", 0.5))
    w_if = float(unsup_meta.get("novelty_blend_w_if", 0.5))
    if novelty_mode in {"ae", "autoencoder"}:
        pass
    elif novelty_mode in {"iforest", "iso"}:
        if if_n is not None:
            latent_outputs["l1b_novelty_score"] = if_n
    else:
        if if_n is not None and not ifm.get("iforest_disabled"):
            s = w_ae + w_if
            latent_outputs["l1b_novelty_score"] = (
                (w_ae * ae_nov + w_if * if_n) / max(s, 1e-6)
            ).astype(np.float32)
    latent_cols = list(unsup_meta.get("latent_cols") or [])
    centroids = np.asarray(unsup_meta.get("cluster_centroids"), dtype=np.float32)
    cluster_temperature = float(unsup_meta.get("cluster_temperature", 1.0))
    if on_substage is not None:
        on_substage("L1b · cluster")
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


def _l1b_oof_stitch_unsupervised_outputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    fold_masks: list[tuple[np.ndarray, np.ndarray]],
    fit_pool: np.ndarray,
) -> dict[str, np.ndarray]:
    cols = list(L1B_UNSUPERVISED_COLS)
    stitched = {col: np.full(len(df), np.nan, dtype=np.float32) for col in cols}
    pool = np.asarray(fit_pool, dtype=bool).ravel()
    for fk, (tr_m, va_m) in enumerate(fold_masks):
        fit_tr = np.asarray(tr_m, dtype=bool).ravel() & pool
        fit_va = np.asarray(va_m, dtype=bool).ravel() & pool
        if int(np.sum(fit_tr)) < 80 or int(np.sum(fit_va)) < 20:
            raise RuntimeError(
                f"L1b unsup OOF fold {fk + 1}: insufficient rows "
                f"(train={int(np.sum(fit_tr))}, val={int(np.sum(fit_va))})."
            )
        fold_out, _ = _fit_l1b_unsupervised_block(df, feature_cols, train_mask=fit_tr)
        for col in cols:
            arr = fold_out[col].to_numpy(dtype=np.float32, copy=False)
            stitched[col][fit_va] = arr[fit_va]
    return stitched


def _corr1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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


def _l1b_edge_candidate_tau() -> float:
    return float(np.clip(float(os.environ.get("L1B_EDGE_CANDIDATE_TAU", "0.05")), 0.0, 1.0))


def _l1b_robust_sigma(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = float(mad / 0.67448975) if mad > 0 else 0.0
    return med, mad, sigma


def _l1b_excess_kurtosis(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return float("nan")
    centered = vals - float(np.mean(vals))
    v2 = float(np.mean(centered ** 2))
    if not np.isfinite(v2) or v2 <= 1e-12:
        return float("nan")
    v4 = float(np.mean(centered ** 4))
    return float(v4 / (v2 * v2) - 3.0)


def _l1b_signflip_bootstrap_tau(values: np.ndarray, *, alpha: float, rounds: int) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    rng = np.random.default_rng(int(os.environ.get("L1B_EDGE_TAU_BOOTSTRAP_SEED", "42")))
    stats = np.empty(max(100, rounds), dtype=np.float64)
    abs_vals = np.abs(vals)
    for i in range(stats.size):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=vals.size, replace=True)
        shuffled = abs_vals * signs
        _, _, sigma = _l1b_robust_sigma(shuffled)
        stats[i] = max(0.0, float(np.mean(np.abs(shuffled)) + sigma))
    q = float(np.quantile(stats[np.isfinite(stats)], max(0.50, 1.0 - alpha))) if np.isfinite(stats).any() else 0.0
    return float(max(0.0, q))


def _l1b_formula_edge_tau(values: np.ndarray) -> tuple[float, dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    mode = (os.environ.get("L1B_EDGE_TAU_MODE", "hybrid") or "hybrid").strip().lower()
    alpha = float(np.clip(float(os.environ.get("L1B_EDGE_TAU_ALPHA", "0.05")), 1e-4, 0.20))
    min_n = int(max(50, round(float(os.environ.get("L1B_EDGE_TAU_MIN_N", "200")))))
    rounds = int(max(100, round(float(os.environ.get("L1B_EDGE_TAU_BOOTSTRAP_ROUNDS", "600")))))
    kurt_thr = float(max(0.0, float(os.environ.get("L1B_EDGE_TAU_HEAVY_TAIL_KURT", "8.0"))))
    p_thr = float(np.clip(float(os.environ.get("L1B_EDGE_TAU_KURTOSIS_P", "0.05")), 1e-4, 0.20))
    meta: dict[str, Any] = {
        "edge_candidate_tau_mode": mode,
        "edge_candidate_tau_alpha": float(alpha),
        "edge_candidate_tau_fit_samples": int(vals.size),
        "edge_candidate_tau_bootstrap_rounds": int(rounds),
    }
    if vals.size == 0:
        meta.update({"edge_candidate_tau_method": "fixed", "edge_candidate_tau_fallback_reason": "no_finite_values"})
        return _l1b_edge_candidate_tau(), meta
    _, mad, sigma = _l1b_robust_sigma(vals)
    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    mad_tau = float(np.clip(z * max(sigma, 1e-8), 1e-6, 5.0))
    ex_kurt = _l1b_excess_kurtosis(vals)
    jb_stat = float(vals.size / 24.0 * (ex_kurt ** 2)) if np.isfinite(ex_kurt) else float("nan")
    jb_p = float(np.exp(-0.5 * jb_stat)) if np.isfinite(jb_stat) else float("nan")
    heavy_tail = bool((np.isfinite(ex_kurt) and ex_kurt > kurt_thr) or (np.isfinite(jb_p) and jb_p < p_thr))
    low_n = vals.size < min_n
    use_bootstrap = mode == "bootstrap" or (mode == "hybrid" and (low_n or heavy_tail))
    if mode == "mad_z":
        use_bootstrap = False
    tau = _l1b_signflip_bootstrap_tau(vals, alpha=alpha, rounds=rounds) if use_bootstrap else mad_tau
    if mode == "hybrid" and use_bootstrap:
        reason = "low_n" if low_n else "heavy_tail"
    elif mode == "bootstrap":
        reason = "forced_bootstrap"
    elif mode == "mad_z":
        reason = "forced_mad_z"
    else:
        reason = "mad_z_ok"
    meta.update(
        {
            "edge_candidate_tau_method": "bootstrap" if use_bootstrap else "mad_z",
            "edge_candidate_tau": float(np.clip(tau, 1e-6, 5.0)),
            "edge_candidate_tau_fallback_reason": reason,
            "edge_candidate_tau_sigma_robust": float(sigma),
            "edge_candidate_tau_mad": float(mad),
            "edge_candidate_tau_z": float(z),
            "edge_candidate_tau_excess_kurtosis": float(ex_kurt) if np.isfinite(ex_kurt) else float("nan"),
            "edge_candidate_tau_kurtosis_pvalue": float(jb_p) if np.isfinite(jb_p) else float("nan"),
            "edge_candidate_tau_statistical_principle": "two_sided_significance_with_robust_scale",
        }
    )
    return float(np.clip(tau, 1e-6, 5.0)), meta


def _l1b_edge_candidate_lgb_params() -> dict[str, Any]:
    fast = FAST_TRAIN_MODE
    return {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
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
        "is_unbalance": True,
    }


def _l1b_edge_dq_train_config() -> tuple[int, int]:
    rounds = int(os.environ.get("L1B_EDGE_DQ_BOOST_ROUNDS", str(200 if FAST_TRAIN_MODE else 500)))
    es = int(os.environ.get("L1B_EDGE_DQ_ES_ROUNDS", str(35 if FAST_TRAIN_MODE else 70)))
    return max(50, rounds), max(20, es)


def _l1b_oof_median_rounds_binary(
    label: str,
    y: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    fold_masks: list[tuple[np.ndarray, np.ndarray]],
    fit_pool: np.ndarray,
    label_row_ok: np.ndarray,
    sample_weight: np.ndarray,
) -> int:
    pool = np.asarray(fit_pool, dtype=bool).ravel()
    y = np.asarray(y, dtype=np.int32).ravel()
    label_row_ok = np.asarray(label_row_ok, dtype=bool).ravel()
    sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
    rounds, es_rounds = _l1b_edge_dq_train_config()
    params = _l1b_edge_candidate_lgb_params()
    best_iters: list[int] = []
    for fk, (tr_m, va_m) in enumerate(fold_masks):
        tr_m = np.asarray(tr_m, dtype=bool).ravel()
        va_m = np.asarray(va_m, dtype=bool).ravel()
        row_ok = np.all(np.isfinite(X), axis=1) & label_row_ok & np.isfinite(y.astype(np.float64))
        fit_tr = tr_m & pool & row_ok
        fit_va = va_m & pool & row_ok
        if int(np.sum(fit_tr)) < 80 or int(np.sum(fit_va)) < 20:
            raise RuntimeError(
                f"L1b {label} OOF fold {fk + 1}: insufficient rows "
                f"(train={int(np.sum(fit_tr))}, val={int(np.sum(fit_va))})."
            )
        dtrain = lgb.Dataset(
            X[fit_tr],
            label=y[fit_tr],
            weight=sample_weight[fit_tr],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X[fit_va],
            label=y[fit_va],
            weight=sample_weight[fit_va],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(
            es_rounds, rounds, f"[L1b] {label} oof {fk + 1}/{len(fold_masks)}", first_metric_only=True
        )
        try:
            booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], callbacks=cbs)
        finally:
            for fn in cl:
                fn()
        bi = booster.best_iteration
        best_iters.append(max(1, int(bi) if bi is not None else rounds))
    return int(np.clip(np.median(best_iters), 1, rounds))


def _l1b_oof_median_rounds_regressor(
    label: str,
    y: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    fold_masks: list[tuple[np.ndarray, np.ndarray]],
    fit_pool: np.ndarray,
    label_row_ok: np.ndarray,
    sample_weight: np.ndarray,
) -> int:
    pool = np.asarray(fit_pool, dtype=bool).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    label_row_ok = np.asarray(label_row_ok, dtype=bool).ravel()
    sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
    rounds, es_rounds = _l1b_edge_dq_train_config()
    params = _l1b_edge_dq_lgb_params()
    best_iters: list[int] = []
    for fk, (tr_m, va_m) in enumerate(fold_masks):
        tr_m = np.asarray(tr_m, dtype=bool).ravel()
        va_m = np.asarray(va_m, dtype=bool).ravel()
        row_ok = np.all(np.isfinite(X), axis=1) & label_row_ok & np.isfinite(y)
        fit_tr = tr_m & pool & row_ok
        fit_va = va_m & pool & row_ok
        if int(np.sum(fit_tr)) < 80 or int(np.sum(fit_va)) < 20:
            raise RuntimeError(
                f"L1b {label} OOF fold {fk + 1}: insufficient rows "
                f"(train={int(np.sum(fit_tr))}, val={int(np.sum(fit_va))})."
            )
        dtrain = lgb.Dataset(
            X[fit_tr],
            label=y[fit_tr],
            weight=sample_weight[fit_tr],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X[fit_va],
            label=y[fit_va],
            weight=sample_weight[fit_va],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(
            es_rounds, rounds, f"[L1b] {label} oof {fk + 1}/{len(fold_masks)}", first_metric_only=True
        )
        try:
            booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval], callbacks=cbs)
        finally:
            for fn in cl:
                fn()
        bi = booster.best_iteration
        best_iters.append(max(1, int(bi) if bi is not None else rounds))
    return int(np.clip(np.median(best_iters), 1, rounds))


def _l1b_expanding_stack_edge_dq_preds(
    X: np.ndarray,
    feature_cols: list[str],
    fold_masks: list[tuple[np.ndarray, np.ndarray]],
    fit_pool: np.ndarray,
    edge_candidate_tgt: np.ndarray,
    edge_tgt: np.ndarray,
    dq_tgt: np.ndarray,
    edge_candidate_row_ok: np.ndarray,
    edge_quality_row_ok: np.ndarray,
    dq_label_ok: np.ndarray,
    sample_weight: np.ndarray,
    nr_cand: int,
    nr_qual: int,
    nr_dq: int,
    *,
    sample_weight_qual: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold edge candidate prob, edge quality, dq on each fold's val rows (expanding calendar)."""
    n = int(X.shape[0])
    pool = np.asarray(fit_pool, dtype=bool).ravel()
    y_c = np.asarray(edge_candidate_tgt, dtype=np.int32).ravel()
    y_e = np.asarray(edge_tgt, dtype=np.float64).ravel()
    y_d = np.asarray(dq_tgt, dtype=np.float64).ravel()
    sw = np.asarray(sample_weight, dtype=np.float32).ravel()
    sw_q = sw if sample_weight_qual is None else np.asarray(sample_weight_qual, dtype=np.float32).ravel()
    cand_acc = np.full(n, np.nan, dtype=np.float64)
    qual_acc = np.full(n, np.nan, dtype=np.float64)
    dq_acc = np.full(n, np.nan, dtype=np.float64)
    for fk, (tr_m, va_m) in enumerate(fold_masks):
        tr_m = np.asarray(tr_m, dtype=bool).ravel()
        va_m = np.asarray(va_m, dtype=bool).ravel()
        row_c = np.all(np.isfinite(X), axis=1) & np.asarray(edge_candidate_row_ok, dtype=bool) & np.isfinite(y_c.astype(np.float64))
        fit_tr_c = tr_m & pool & row_c
        fit_va_c = va_m & pool & row_c
        if int(np.sum(fit_tr_c)) < 80 or int(np.sum(fit_va_c)) < 20:
            raise RuntimeError(
                f"L1b expanding stack fold {fk + 1}: insufficient candidate rows "
                f"(train={int(np.sum(fit_tr_c))}, val={int(np.sum(fit_va_c))})."
            )
        params_c = _l1b_edge_candidate_lgb_params()
        dtr = lgb.Dataset(
            X[fit_tr_c], label=y_c[fit_tr_c], weight=sw[fit_tr_c], feature_name=feature_cols, free_raw_data=False
        )
        b_c = lgb.train(params_c, dtr, num_boost_round=max(1, int(nr_cand)))
        cand_acc[fit_va_c] = np.clip(b_c.predict(X[fit_va_c], num_iteration=int(nr_cand)).astype(np.float64), 0.0, 1.0)

        row_q = np.all(np.isfinite(X), axis=1) & np.asarray(edge_quality_row_ok, dtype=bool) & np.isfinite(y_e)
        fit_tr_q = tr_m & pool & row_q
        fit_va_q = va_m & pool & row_q
        if int(np.sum(fit_tr_q)) < 80 or int(np.sum(fit_va_q)) < 20:
            raise RuntimeError(
                f"L1b expanding stack fold {fk + 1}: insufficient quality rows "
                f"(train={int(np.sum(fit_tr_q))}, val={int(np.sum(fit_va_q))})."
            )
        params_q = _l1b_edge_dq_lgb_params()
        dtq = lgb.Dataset(
            X[fit_tr_q], label=y_e[fit_tr_q], weight=sw_q[fit_tr_q], feature_name=feature_cols, free_raw_data=False
        )
        b_q = lgb.train(params_q, dtq, num_boost_round=max(1, int(nr_qual)))
        qual_acc[fit_va_q] = b_q.predict(X[fit_va_q], num_iteration=int(nr_qual)).astype(np.float64)

        row_d = np.all(np.isfinite(X), axis=1) & np.asarray(dq_label_ok, dtype=bool) & np.isfinite(y_d)
        fit_tr_d = tr_m & pool & row_d
        fit_va_d = va_m & pool & row_d
        if int(np.sum(fit_tr_d)) < 80 or int(np.sum(fit_va_d)) < 20:
            raise RuntimeError(
                f"L1b expanding stack fold {fk + 1}: insufficient dq rows "
                f"(train={int(np.sum(fit_tr_d))}, val={int(np.sum(fit_va_d))})."
            )
        dtd = lgb.Dataset(
            X[fit_tr_d], label=y_d[fit_tr_d], weight=sw[fit_tr_d], feature_name=feature_cols, free_raw_data=False
        )
        b_d = lgb.train(params_q, dtd, num_boost_round=max(1, int(nr_dq)))
        dq_acc[fit_va_d] = b_d.predict(X[fit_va_d], num_iteration=int(nr_dq)).astype(np.float64)
    return cand_acc, qual_acc, dq_acc


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
    num_boost_round_fixed: int | None = None,
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
    if num_boost_round_fixed is not None:
        nr = max(1, int(num_boost_round_fixed))
        if int(np.sum(fit_tr)) < 80:
            raise RuntimeError(
                f"L1b {label}: insufficient rows for fixed-round fit (train={int(np.sum(fit_tr))})."
            )
        dtrain = lgb.Dataset(
            X[fit_tr],
            label=y[fit_tr],
            weight=sample_weight[fit_tr],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        booster = lgb.train(params, dtrain, num_boost_round=nr)
        booster.save_model(out_path)
        pred_tr = booster.predict(X[fit_tr]).astype(np.float64)
        y_tr = y[fit_tr]
        mae_tr = float(mean_absolute_error(y_tr, pred_tr))
        std_tr = float(np.std(y_tr)) if y_tr.size > 1 else 0.0
        mae_v = cor_v = float("nan")
        std_v = 0.0
        if int(np.sum(fit_val)) >= 20:
            pred_v = booster.predict(X[fit_val]).astype(np.float64)
            y_v = y[fit_val]
            mae_v = float(mean_absolute_error(y_v, pred_v))
            std_v = float(np.std(y_v)) if y_v.size > 1 else 0.0
            cor_v = _corr1d(y_v, pred_v)
        print(
            f"  [L1b] {label} booster (OOF final rounds={nr}): train_MAE={mae_tr:.4f}  "
            f"report_MAE={mae_v:.4f}  report_corr={cor_v:.4f}  "
            f"report_n={int(np.sum(fit_val)):,}  train_n={int(np.sum(fit_tr)):,}  -> {out_path}",
            flush=True,
        )
        return booster
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
    pos_mask_v = y_v > 1e-8
    cond_corr_v = _corr1d(y_v[pos_mask_v], pred_v[pos_mask_v]) if int(np.sum(pos_mask_v)) >= 30 else float("nan")
    gap = mae_v - mae_tr
    print(
        f"  [L1b] {label} booster: train_MAE={mae_tr:.4f}  val_MAE={mae_v:.4f}  gap={gap:+.4f}  "
        f"val_corr={cor_v:.4f}  val_n={int(np.sum(fit_val)):,}  val_target_std={std_v:.4f}  "
        f"train_n={int(np.sum(fit_tr)):,}  train_target_std={std_tr:.4f}  -> {out_path}",
        flush=True,
    )
    if label == "l1b_edge_pred":
        cond_corr_s = f"{cond_corr_v:.4f}" if np.isfinite(cond_corr_v) else "nan"
        print(
            f"  [L1b] {label} conditional val: true_edge>0 rows={int(np.sum(pos_mask_v)):,}  corr={cond_corr_s}",
            flush=True,
        )
    return booster


def _l1b_fit_lgb_binary_classifier(
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
    num_boost_round_fixed: int | None = None,
) -> lgb.Booster:
    train = np.asarray(train_mask, dtype=bool).ravel()
    val = np.asarray(val_mask, dtype=bool).ravel()
    y = np.asarray(y, dtype=np.int32).ravel()
    if label_row_ok is None:
        label_row_ok = np.ones(len(y), dtype=bool)
    else:
        label_row_ok = np.asarray(label_row_ok, dtype=bool).ravel()
    row_ok = np.all(np.isfinite(X), axis=1) & label_row_ok & np.isfinite(y.astype(np.float64))
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
        if sample_weight.shape[0] != len(y):
            raise ValueError(f"L1b {label}: sample_weight length mismatch.")
    fit_tr = train & row_ok
    fit_val = val & row_ok
    rounds, es_rounds = _l1b_edge_dq_train_config()
    params = _l1b_edge_candidate_lgb_params()
    if num_boost_round_fixed is not None:
        nr = max(1, int(num_boost_round_fixed))
        if int(np.sum(fit_tr)) < 80:
            raise RuntimeError(
                f"L1b {label}: insufficient rows for fixed-round fit (train={int(np.sum(fit_tr))})."
            )
        dtrain = lgb.Dataset(
            X[fit_tr],
            label=y[fit_tr],
            weight=sample_weight[fit_tr],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        booster = lgb.train(params, dtrain, num_boost_round=nr)
        booster.save_model(out_path)
        if int(np.sum(fit_val)) >= 20:
            pred_v = np.clip(booster.predict(X[fit_val]).astype(np.float64), 1e-7, 1.0 - 1e-7)
            y_v = y[fit_val].astype(np.int32)
            try:
                auc = float(roc_auc_score(y_v, pred_v))
            except ValueError:
                auc = float("nan")
            print(
                f"  [L1b] {label} booster (OOF final rounds={nr}): report_AUC={auc:.4f}  "
                f"report_n={int(np.sum(fit_val)):,}  train_n={int(np.sum(fit_tr)):,}  -> {out_path}",
                flush=True,
            )
        else:
            print(
                f"  [L1b] {label} booster (OOF final rounds={nr}): train_n={int(np.sum(fit_tr)):,}  -> {out_path}",
                flush=True,
            )
        return booster
    if int(np.sum(fit_tr)) < 80 or int(np.sum(fit_val)) < 20:
        raise RuntimeError(
            f"L1b {label}: insufficient rows for binary classifier "
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
    pred_v = np.clip(booster.predict(X[fit_val]).astype(np.float64), 1e-7, 1.0 - 1e-7)
    y_v = y[fit_val].astype(np.int32)
    try:
        auc = float(roc_auc_score(y_v, pred_v))
    except ValueError:
        auc = float("nan")
    try:
        ll = float(log_loss(y_v, pred_v))
    except ValueError:
        ll = float("nan")
    yhat = (pred_v >= 0.5).astype(np.int32)
    br = brier_binary(y_v.astype(np.float64), pred_v)
    print(
        f"  [L1b] {label} booster: val_AUC={auc:.4f}  val_log_loss={ll:.4f}  Brier={br:.4f}  "
        f"val_acc@0.5={accuracy_score(y_v, yhat):.4f}  "
        f"precision={precision_score(y_v, yhat, zero_division=0):.4f}  "
        f"recall={recall_score(y_v, yhat, zero_division=0):.4f}  "
        f"F1={f1_score(y_v, yhat, zero_division=0):.4f}  val_n={int(np.sum(fit_val)):,}  "
        f"train_n={int(np.sum(fit_tr)):,}  -> {out_path}",
        flush=True,
    )
    return booster


def _fit_l1b_edge_dq_boosters(
    work: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    n_oof_folds: int = 1,
    l1_fit_mask: np.ndarray | None = None,
) -> tuple[dict[str, lgb.Booster], dict[str, str], dict[str, np.ndarray], dict[str, Any]]:
    supervised_tgt_mode = _l1b_supervised_target_mode()
    print(f"  [L1b] supervised target mode: {supervised_tgt_mode}", flush=True)
    edge_pred_hi = 5.0
    edge_signed = np.zeros(len(work), dtype=np.float64)
    if supervised_tgt_mode == "forward_range":
        raw_fr = _decision_forward_range_atr_array(work)
        edge_pred_hi = float(os.environ.get("L1B_FORWARD_RANGE_CLIP_HI", "15"))
        edge_tgt = np.clip(raw_fr, 0.0, edge_pred_hi).astype(np.float64)
        edge_label_ok = np.isfinite(raw_fr) & np.isfinite(edge_tgt)
    else:
        edge_signed = np.clip(_decision_edge_atr_array(work), -5.0, 5.0).astype(np.float64)
        edge_tgt = np.abs(edge_signed).astype(np.float64)
        edge_label_ok = _l1b_edge_label_valid(work)
        edge_pred_hi = 5.0
    tm = np.asarray(train_mask, dtype=bool).ravel()
    fit = tm & np.isfinite(edge_tgt)
    fit_n = int(np.sum(fit))
    legacy_mode = (os.environ.get("L1B_EDGE_CANDIDATE_TAU_MODE", "") or "").strip().lower()
    tau_meta: dict[str, Any]
    if legacy_mode == "quantile":
        edge_candidate_tau = _l1b_edge_candidate_tau()
        q = float(np.clip(float(os.environ.get("L1B_EDGE_CANDIDATE_TAU_Q", "0.70")), 0.50, 0.95))
        if np.any(fit):
            edge_candidate_tau = float(np.quantile(edge_tgt[fit], q))
            tau_hi = float(edge_pred_hi) if supervised_tgt_mode == "forward_range" else 1.0
            edge_candidate_tau = float(np.clip(edge_candidate_tau, 0.01, tau_hi))
        tau_meta = {
            "edge_candidate_tau_mode": "quantile",
            "edge_candidate_tau_method": "quantile",
            "edge_candidate_tau_alpha": float("nan"),
            "edge_candidate_tau_fit_samples": fit_n,
            "edge_candidate_tau_bootstrap_rounds": 0,
            "edge_candidate_tau_fallback_reason": "legacy_quantile_mode",
            "edge_candidate_tau_sigma_robust": float("nan"),
            "edge_candidate_tau_mad": float("nan"),
            "edge_candidate_tau_z": float("nan"),
            "edge_candidate_tau_excess_kurtosis": float("nan"),
            "edge_candidate_tau_kurtosis_pvalue": float("nan"),
            "edge_candidate_tau_statistical_principle": "quantile_clip",
        }
        print(
            f"  [L1b] edge candidate tau: mode=quantile  q={q:.3f}  tau={edge_candidate_tau:.4f}",
            flush=True,
        )
    else:
        edge_candidate_tau, tau_meta = _l1b_formula_edge_tau(edge_tgt[fit])
        if supervised_tgt_mode == "forward_range":
            edge_candidate_tau = float(np.clip(edge_candidate_tau, 0.05, float(edge_pred_hi)))
        print(
            f"  [L1b] edge candidate tau: mode={tau_meta.get('edge_candidate_tau_mode')}  "
            f"method={tau_meta.get('edge_candidate_tau_method')}  tau={edge_candidate_tau:.4f}  "
            f"fit_n={fit_n:,}  fallback={tau_meta.get('edge_candidate_tau_fallback_reason')}",
            flush=True,
        )
    edge_candidate_tgt = (edge_tgt > edge_candidate_tau).astype(np.int32)
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
    n_oof = max(1, int(n_oof_folds))
    use_oof = n_oof >= 2 and l1_fit_mask is not None
    fit_pool = np.asarray(l1_fit_mask, dtype=bool).ravel() if use_oof else tm
    if use_oof and not np.array_equal(fit_pool, tm):
        raise RuntimeError("L1b OOF: train_mask must match l1_fit_mask.")
    ok_both = tm & edge_label_ok & dq_label_ok & np.isfinite(edge_tgt) & np.isfinite(dq_tgt)
    if int(np.sum(ok_both)) > 50:
        c_ed = _corr1d(edge_tgt[ok_both], dq_tgt[ok_both])
        print(
            f"  [L1b] primary_supervision vs dq_target train corr (finite rows n={int(np.sum(ok_both)):,})={c_ed:.4f}  dq_mode={mode!r}",
            flush=True,
        )
    if supervised_tgt_mode == "staged_edge" and int(np.sum(tm & edge_label_ok & np.isfinite(edge_signed))) > 50:
        c_sign = _corr1d(edge_tgt[tm & edge_label_ok], edge_signed[tm & edge_label_ok])
        print(
            f"  [L1b] edge opportunity target corr(|signed_edge|, signed_edge)={c_sign:.4f}  "
            "target_semantics='absolute edge magnitude only'",
            flush=True,
        )
    log_label_baseline("l1b_edge_candidate", edge_candidate_tgt[tm & edge_label_ok], task="cls")
    log_label_baseline(
        "l1b_forward_range_target" if supervised_tgt_mode == "forward_range" else "l1b_edge_target",
        edge_tgt[tm & edge_label_ok],
        task="reg",
    )
    log_label_baseline("l1b_dq_target", dq_tgt[tm & dq_label_ok], task="reg")
    models: dict[str, lgb.Booster] = {}
    model_files: dict[str, str] = {}
    edge_candidate_row_ok = edge_label_ok & np.isfinite(edge_tgt)
    edge_quality_row_ok = edge_label_ok & np.isfinite(edge_tgt) & (edge_tgt > edge_candidate_tau)
    edge_candidate_path = os.path.join(MODEL_DIR, "l1b_edge_candidate.txt")
    edge_quality_path = os.path.join(MODEL_DIR, L1B_EDGE_PRED_FILE)
    dq_path = os.path.join(MODEL_DIR, L1B_DQ_PRED_FILE)
    sw_uni = np.ones(len(work), dtype=np.float32)
    tw = float(os.environ.get("L1B_FORWARD_RANGE_TAIL_WEIGHT", "1.0"))
    if supervised_tgt_mode == "forward_range" and tw > 0.0:
        sw_qual = _upper_tail_sample_weight(edge_tgt, fit_mask=tm, start_pct=0.72, alpha=float(tw)).astype(np.float32)
    else:
        sw_qual = sw_uni
    if use_oof:
        if l1_oof_mode_from_env() == "expanding":
            val_windows = l1b_expand_oof_val_windows() if l1b_should_use_shifted_expand_oof_windows() else None
            fold_masks = l1_expanding_oof_row_folds(work["time_key"], fit_pool, val_windows=val_windows)
            note = "  L1B_EXPAND_OOF_VAL_WINDOWS" if val_windows is not None else ""
            print(
                f"  [L1b] expanding calendar OOF: {len(fold_masks)} folds (L1_OOF_MODE=expanding){note}",
                flush=True,
            )
        else:
            fold_masks = time_blocked_fold_masks(work["time_key"], fit_pool, n_oof, context="L1b OOF")
        nr_cand = _l1b_oof_median_rounds_binary(
            "l1b_edge_candidate",
            edge_candidate_tgt,
            X,
            feature_cols,
            fold_masks,
            fit_pool,
            edge_candidate_row_ok,
            sw_uni,
        )
        nr_qual = _l1b_oof_median_rounds_regressor(
            "l1b_edge_quality",
            edge_tgt,
            X,
            feature_cols,
            fold_masks,
            fit_pool,
            edge_quality_row_ok,
            sw_qual,
        )
        nr_dq = _l1b_oof_median_rounds_regressor(
            "l1b_dq_pred",
            dq_tgt,
            X,
            feature_cols,
            fold_masks,
            fit_pool,
            dq_label_ok,
            sw_uni,
        )
        print(
            f"  [L1b] OOF median best_iteration → edge_candidate={nr_cand}  edge_quality={nr_qual}  dq={nr_dq}",
            flush=True,
        )
        models["l1b_edge_candidate_model"] = _l1b_fit_lgb_binary_classifier(
            "l1b_edge_candidate",
            edge_candidate_tgt,
            X,
            feature_cols,
            train_mask=fit_pool,
            val_mask=val_mask,
            out_path=edge_candidate_path,
            label_row_ok=edge_candidate_row_ok,
            num_boost_round_fixed=nr_cand,
        )
        models["l1b_edge_quality_model"] = _l1b_fit_lgb_regressor(
            "l1b_edge_quality",
            edge_tgt,
            X,
            feature_cols,
            train_mask=fit_pool,
            val_mask=val_mask,
            out_path=edge_quality_path,
            label_row_ok=edge_quality_row_ok,
            sample_weight=sw_qual,
            num_boost_round_fixed=nr_qual,
        )
        models["l1b_dq_pred"] = _l1b_fit_lgb_regressor(
            "l1b_dq_pred",
            dq_tgt,
            X,
            feature_cols,
            train_mask=fit_pool,
            val_mask=val_mask,
            out_path=dq_path,
            label_row_ok=dq_label_ok,
            num_boost_round_fixed=nr_dq,
        )
    else:
        models["l1b_edge_candidate_model"] = _l1b_fit_lgb_binary_classifier(
            "l1b_edge_candidate",
            edge_candidate_tgt,
            X,
            feature_cols,
            train_mask=train_mask,
            val_mask=val_mask,
            out_path=edge_candidate_path,
            label_row_ok=edge_candidate_row_ok,
        )
        models["l1b_edge_quality_model"] = _l1b_fit_lgb_regressor(
            "l1b_edge_quality",
            edge_tgt,
            X,
            feature_cols,
            train_mask=train_mask,
            val_mask=val_mask,
            out_path=edge_quality_path,
            label_row_ok=edge_quality_row_ok,
            sample_weight=sw_qual,
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
    model_files["l1b_edge_candidate_model"] = "l1b_edge_candidate.txt"
    model_files["l1b_edge_quality_model"] = L1B_EDGE_PRED_FILE
    model_files["l1b_dq_pred"] = L1B_DQ_PRED_FILE
    edge_candidate_prob = np.clip(models["l1b_edge_candidate_model"].predict(X).astype(np.float64), 0.0, 1.0)
    edge_quality_pred = np.clip(
        models["l1b_edge_quality_model"].predict(X).astype(np.float64),
        edge_candidate_tau,
        float(edge_pred_hi),
    )
    dq_pred_arr = np.clip(
        models["l1b_dq_pred"].predict(X).astype(np.float64),
        dq_clip_lo,
        dq_clip_hi,
    )
    if use_oof and l1_oof_mode_from_env() == "expanding":
        sc, sq, sd = _l1b_expanding_stack_edge_dq_preds(
            X,
            feature_cols,
            fold_masks,
            fit_pool,
            edge_candidate_tgt,
            edge_tgt,
            dq_tgt,
            edge_candidate_row_ok,
            edge_quality_row_ok,
            dq_label_ok,
            sw_uni,
            nr_cand,
            nr_qual,
            nr_dq,
            sample_weight_qual=sw_qual,
        )
        m_ce = np.isfinite(sc) & np.isfinite(sq)
        edge_candidate_prob[m_ce] = np.clip(sc[m_ce], 0.0, 1.0)
        edge_quality_pred[m_ce] = np.clip(sq[m_ce], float(edge_candidate_tau), float(edge_pred_hi))
        m_dq = np.isfinite(sd)
        dq_pred_arr[m_dq] = np.clip(sd[m_dq], dq_clip_lo, dq_clip_hi)
        print("  [L1b] stitched expanding OOF edge/dq preds on cal rows (honest L1b for L2)", flush=True)
    edge_tradeability = np.clip(edge_candidate_prob * edge_quality_pred, 0.0, float(edge_pred_hi))
    preds = {
        "l1b_edge_pred": edge_tradeability.astype(np.float32),
        "l1b_dq_pred": dq_pred_arr.astype(np.float32),
        "l1b_edge_candidate_tau": np.full(len(work), float(edge_candidate_tau), dtype=np.float32),
    }
    vm_edge = np.asarray(val_mask, dtype=bool).ravel() & edge_label_ok & np.isfinite(edge_tgt)
    vm_pos = vm_edge & (edge_tgt > edge_candidate_tau)
    if int(np.sum(vm_edge)) >= 30:
        combined_corr = _corr1d(edge_tgt[vm_edge], edge_tradeability[vm_edge])
        print(
            f"  [L1b] l1b_edge_pred staged val: all_rows={int(np.sum(vm_edge)):,}  "
            f"corr={combined_corr:.4f}  candidate_tau={edge_candidate_tau:.4f}",
            flush=True,
        )
    if int(np.sum(vm_pos)) >= 30:
        quality_corr = _corr1d(edge_tgt[vm_pos], edge_quality_pred[vm_pos])
        combined_pos_corr = _corr1d(edge_tgt[vm_pos], edge_tradeability[vm_pos])
        candidate_recall = float(np.mean(edge_candidate_prob[vm_pos] >= 0.5))
        print(
            f"  [L1b] l1b_edge_pred staged val (true_edge>tau): rows={int(np.sum(vm_pos)):,}  "
            f"quality_corr={quality_corr:.4f}  blended_corr={combined_pos_corr:.4f}  "
            f"candidate_recall@0.5={candidate_recall:.4f}",
            flush=True,
        )
    t_hi = float(edge_pred_hi)
    if supervised_tgt_mode == "forward_range":
        targets_meta = {
            "l1b_edge_candidate_model": (
                f"1[decision_forward_range_atr > {edge_candidate_tau:g}] high forward-range candidate"
            ),
            "l1b_edge_quality_model": (
                f"regress clipped forward_range_atr in [0,{t_hi:g}] on rows with range > {edge_candidate_tau:g}"
            ),
            "l1b_edge_pred": "P(candidate) × quality_pred — straddle-aligned tradeability (no directional return)",
            "l1b_dq_pred": dq_desc,
        }
        edge_tgt_sem = (
            "decision_forward_range_atr (label_v2), aligned with L2 range/straddle; "
            "l1b_edge_pred = staged tradeability; dq = path-balance auxiliary"
        )
    else:
        targets_meta = {
            "l1b_edge_candidate_model": (
                f"1[clip(abs(edge),0..5) > {edge_candidate_tau:g}] opportunity candidate classifier"
            ),
            "l1b_edge_quality_model": (
                f"clip(abs(edge),0..5) ATR opportunity magnitude on candidate rows edge>{edge_candidate_tau:g}"
            ),
            "l1b_edge_pred": (
                f"P(edge>{edge_candidate_tau:g}) * E[abs(edge)|candidate] staged tradeability score"
            ),
            "l1b_dq_pred": dq_desc,
        }
        edge_tgt_sem = (
            "staged tradeability score = opportunity probability times positive-edge magnitude; no directional sign"
        )
    block_meta = {
        "supervised_target_mode": supervised_tgt_mode,
        "targets": targets_meta,
        "edge_target_semantics": edge_tgt_sem,
        "edge_model_type": "staged_tradeability",
        "edge_candidate_tau": float(edge_candidate_tau),
        "edge_candidate_tau_mode": str(tau_meta.get("edge_candidate_tau_mode", "hybrid")),
        "edge_candidate_tau_method": str(tau_meta.get("edge_candidate_tau_method", "mad_z")),
        "edge_candidate_tau_alpha": float(tau_meta.get("edge_candidate_tau_alpha", float("nan"))),
        "edge_candidate_tau_fit_samples": int(tau_meta.get("edge_candidate_tau_fit_samples", fit_n)),
        "edge_candidate_tau_bootstrap_rounds": int(tau_meta.get("edge_candidate_tau_bootstrap_rounds", 0)),
        "edge_candidate_tau_fallback_reason": str(tau_meta.get("edge_candidate_tau_fallback_reason", "")),
        "edge_candidate_tau_sigma_robust": float(tau_meta.get("edge_candidate_tau_sigma_robust", float("nan"))),
        "edge_candidate_tau_mad": float(tau_meta.get("edge_candidate_tau_mad", float("nan"))),
        "edge_candidate_tau_z": float(tau_meta.get("edge_candidate_tau_z", float("nan"))),
        "edge_candidate_tau_excess_kurtosis": float(tau_meta.get("edge_candidate_tau_excess_kurtosis", float("nan"))),
        "edge_candidate_tau_kurtosis_pvalue": float(tau_meta.get("edge_candidate_tau_kurtosis_pvalue", float("nan"))),
        "edge_candidate_tau_statistical_principle": str(
            tau_meta.get("edge_candidate_tau_statistical_principle", "two_sided_significance_with_robust_scale")
        ),
        "edge_pred_clip": [0.0, t_hi],
        "dq_target_mode": mode,
        "dq_pred_clip": [dq_clip_lo, dq_clip_hi],
        "boost_rounds_env": "L1B_EDGE_DQ_BOOST_ROUNDS",
        "early_stopping_env": "L1B_EDGE_DQ_ES_ROUNDS",
        "experts_enabled": False,
        "routing_semantics": "disabled; base edge/dq regressors only",
        "l1_oof_mode": l1_oof_mode_from_env(),
        "l1_oof_folds": int(n_oof),
        "l1_oof_enabled": bool(use_oof),
    }
    return models, model_files, preds, block_meta


def _l1b_supervised_gain_by_feature(
    models: dict[str, lgb.Booster],
    supervised_feature_cols: list[str],
) -> dict[str, float]:
    acc = {c: 0.0 for c in supervised_feature_cols}
    for mdl in models.values():
        try:
            names = [str(x) for x in mdl.feature_name()]
            g = mdl.feature_importance(importance_type="gain").astype(np.float64)
        except Exception:
            continue
        for i, n in enumerate(names):
            if n in acc and i < len(g):
                acc[n] += float(g[i])
    return acc


def _l1b_prune_base_feature_cols(
    base_cols: list[str],
    gain_by_name: dict[str, float],
    *,
    min_frac_of_total: float,
    min_keep: int,
) -> tuple[list[str], list[str]]:
    """Drop low-gain base columns using cumulative gain threshold + minimum keep."""
    base_gain = np.array([max(0.0, gain_by_name.get(c, 0.0)) for c in base_cols], dtype=np.float64)
    total = float(np.sum(base_gain))
    if total <= 0 or len(base_cols) <= min_keep:
        return list(base_cols), []
    order = sorted(range(len(base_cols)), key=lambda i: base_gain[i], reverse=True)
    topk = set(order[: min(min_keep, len(base_cols))])
    thresh = total * float(min_frac_of_total)
    keep_idx: set[int] = set()
    for i in order:
        if i in topk or base_gain[i] >= thresh:
            keep_idx.add(i)
    kept = [base_cols[i] for i in range(len(base_cols)) if i in keep_idx]
    kset = set(kept)
    dropped = [c for c in base_cols if c not in kset]
    return kept, dropped


def _l1b_outputs_drop_for_refit(outputs: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in L1B_UNSUPERVISED_COLS if c in outputs.columns]
    drop.extend([c for c in L1B_EDGE_DQ_PRED_COLS if c in outputs.columns])
    return outputs.drop(columns=drop, errors="ignore")


def train_l1b_market_descriptor(df: pd.DataFrame, feat_cols: list[str]) -> L1BTrainingBundle:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L1b] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    work = df.copy()
    feature_cols = _select_l1b_feature_cols(work, feat_cols)
    X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    splits = build_stack_time_splits(work["time_key"])
    l1_fit_mask = np.asarray(splits.train_mask | splits.cal_mask, dtype=bool)
    if l1b_use_honest_l1a_fit_pool():
        l1_fit_mask = l1b_apply_honest_l1a_fit_mask(work, l1_fit_mask)
        if l1b_l1a_inputs_enabled() and l1b_l1a_feature_tier() != "none":
            print(
                f"  [L1b] L1a features: honest fit pool t>={TRAIN_END}  rows={int(np.sum(l1_fit_mask)):,}",
                flush=True,
            )
        else:
            print(
                f"  [L1b] baseline aligned to L1a pool (L1B_BASELINE_ALIGN_TO_L1A_POOL=1): "
                f"t>={TRAIN_END}  rows={int(np.sum(l1_fit_mask)):,}",
                flush=True,
            )
    n_l1_oof = l1_oof_folds_from_env()
    l2_vs = l2_val_start_time()
    if n_l1_oof >= 2:
        train_mask = l1_fit_mask
        val_mask = l1_fit_mask
        print(
            f"  [L1b] blocked time OOF: L1_OOF_FOLDS={n_l1_oof} on train+cal (t < {CAL_END}) "
            f"(set L1_OOF_FOLDS=1 for legacy train vs l2_val [{l2_vs}, {CAL_END}))",
            flush=True,
        )
        tune_frac = float(os.environ.get("L1_TUNE_FRAC_WITHIN_FIT", "0.5"))
        val_tune_mask, val_report_mask = split_mask_for_tuning_and_report(
            work["time_key"], l1_fit_mask, tune_frac=tune_frac, min_rows_each=50
        )
        if not val_tune_mask.any() or not val_report_mask.any():
            raise RuntimeError("L1b OOF: failed to build non-empty tune/report masks inside train+cal.")
    else:
        train_mask = splits.train_mask
        val_mask = splits.l2_val_mask
        val_tune_mask = val_report_mask = val_mask
    unsup_mask = l1_fit_mask if n_l1_oof >= 2 else splits.train_mask

    oof_fold_display = int(n_l1_oof)
    if (
        n_l1_oof >= 2
        and l1_oof_mode_from_env() == "expanding"
        and l1b_should_use_shifted_expand_oof_windows()
    ):
        oof_fold_display = len(l1b_expand_oof_val_windows())

    log_layer_banner("[L1b] Tabular market descriptor")
    if n_l1_oof >= 2:
        log_time_key_split(
            "L1b",
            work["time_key"],
            train_mask,
            val_mask,
            train_label=f"train+cal (t < {CAL_END})",
            val_label=f"train+cal (t < {CAL_END})",
            extra_note=(
                f"OOF: {oof_fold_display} folds (L1_OOF_MODE={l1_oof_mode_from_env()}); "
                f"L1_OOF_FOLDS env={n_l1_oof} applies to blocked mode; "
                f"expanding uses len(val_windows). Tune/report masks slice the fit pool."
            ),
        )
        log_time_key_split(
            "L1b(tune/report)",
            work["time_key"],
            val_tune_mask,
            val_report_mask,
            train_label="fit_tune (early slice)",
            val_label="fit_report (late slice)",
            extra_note="Unsupervised/LGBM diagnostics vs held-out late time within train+cal.",
        )
    else:
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
    log_numpy_x_stats("L1b", X[unsup_mask], label="X[fit_pool]" if n_l1_oof >= 2 else "X[train]")
    print(f"  [L1b] target/output schema L1B_OUTPUT_COLS count={len(L1B_OUTPUT_COLS)}: {L1B_OUTPUT_COLS}", flush=True)
    print(f"  [L1b] artifact dir: {MODEL_DIR}", flush=True)
    print(f"  [L1b] will write meta/cache: {artifact_path(L1B_META_FILE)} | {artifact_path(L1B_OUTPUT_CACHE_FILE)}", flush=True)
    outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
    print(
        f"  [L1b] note: cache/L2 export is {len(L1B_OUTPUT_COLS)} cols "
        f"(unsupervised + l1b_edge_pred/l1b_dq_pred/l1b_edge_candidate_tau; no rule-based direct heads).",
        flush=True,
    )

    prune_on = os.environ.get("L1B_AUTO_PRUNE", "").strip().lower() in {"1", "true", "yes"}
    prune_refit = os.environ.get("L1B_AUTO_PRUNE_REFIT", "").strip().lower() in {"1", "true", "yes"}
    prune_min_frac = float(os.environ.get("L1B_PRUNE_MIN_GAIN_FRAC", "0.008"))
    prune_min_keep = max(8, int(os.environ.get("L1B_PRUNE_MIN_FEATURES", "28")))
    pruned_meta: dict[str, Any] = {}
    prune_pass = 0
    atomic_supervised: list[str] = []
    supervised_feature_cols: list[str] = []
    unsup_meta: dict[str, Any] = {}
    unsup_oof_stitched = False
    unsup_oof_rows = 0
    l1b_supervised_models: dict[str, lgb.Booster] = {}
    edge_dq_model_files: dict[str, str] = {}
    edge_dq_preds: dict[str, np.ndarray] = {}
    edge_dq_block_meta: dict[str, Any] = {}

    while True:
        atomic_supervised = _l1b_attach_atomic_supervised_columns(work, feature_cols)
        supervised_feature_cols = list(feature_cols) + atomic_supervised
        if prune_pass == 0:
            print(f"  [L1b] input feature count (unsupervised autoencoder/KMeans): {len(feature_cols)}", flush=True)
            print(
                f" supervised augment: +{len(atomic_supervised)} atom cols (internal only; not in L1B_OUTPUT_COLS)  "
                f"examples={atomic_supervised[:5]}{'...' if len(atomic_supervised) > 5 else ''}",
                flush=True,
            )
        elif prune_pass > 0:
            print(
                f"  [L1b] prune refit pass {prune_pass}: base_feats={len(feature_cols)} "
                f"(L1B_AUTO_PRUNE + L1B_AUTO_PRUNE_REFIT)",
                flush=True,
            )

        unsup_outputs, unsup_meta = _fit_l1b_unsupervised_block(work, feature_cols, train_mask=unsup_mask)
        if n_l1_oof >= 2:
            if l1_oof_mode_from_env() == "expanding":
                val_windows = l1b_expand_oof_val_windows() if l1b_should_use_shifted_expand_oof_windows() else None
                unsup_fold_masks = l1_expanding_oof_row_folds(work["time_key"], l1_fit_mask, val_windows=val_windows)
            else:
                unsup_fold_masks = time_blocked_fold_masks(work["time_key"], l1_fit_mask, n_l1_oof, context="L1b unsup OOF")
            stitched_unsup = _l1b_oof_stitch_unsupervised_outputs(work, feature_cols, unsup_fold_masks, l1_fit_mask)
            unsup_oof_stitched = True
            unsup_oof_rows = 0
            for col, arr in stitched_unsup.items():
                mask = np.isfinite(arr)
                if np.any(mask):
                    unsup_outputs.loc[mask, col] = arr[mask]
                    unsup_oof_rows = max(unsup_oof_rows, int(np.sum(mask)))
            print(
                f"  [L1b] stitched {'expanding' if l1_oof_mode_from_env() == 'expanding' else 'blocked'} "
                f"OOF unsupervised outputs on cal rows (rows={unsup_oof_rows:,})",
                flush=True,
            )
        if prune_pass > 0:
            outputs = _l1b_outputs_drop_for_refit(outputs)
        outputs = pd.concat([outputs, unsup_outputs.reset_index(drop=True)], axis=1)

        l1b_supervised_models, edge_dq_model_files, edge_dq_preds, edge_dq_block_meta = _fit_l1b_edge_dq_boosters(
            work,
            supervised_feature_cols,
            train_mask=unsup_mask,
            val_mask=val_report_mask,
            n_oof_folds=n_l1_oof,
            l1_fit_mask=l1_fit_mask if n_l1_oof >= 2 else None,
        )
        for k, arr in edge_dq_preds.items():
            outputs[k] = arr

        if (
            prune_on
            and prune_refit
            and prune_pass == 0
            and len(feature_cols) > prune_min_keep + 2
        ):
            g_all = _l1b_supervised_gain_by_feature(l1b_supervised_models, supervised_feature_cols)
            g_base = {c: float(g_all.get(c, 0.0)) for c in feature_cols}
            new_base, dropped = _l1b_prune_base_feature_cols(
                feature_cols,
                g_base,
                min_frac_of_total=prune_min_frac,
                min_keep=prune_min_keep,
            )
            if len(new_base) < len(feature_cols):
                pruned_meta = {
                    "dropped": dropped,
                    "n_before": len(feature_cols),
                    "n_after": len(new_base),
                    "min_frac": prune_min_frac,
                    "min_keep": prune_min_keep,
                }
                print(
                    f"  [L1b] auto-prune: base {len(feature_cols)} -> {len(new_base)}  "
                    f"dropped_sample={dropped[:10]!r}",
                    flush=True,
                )
                feature_cols = new_base
                prune_pass += 1
                continue
        break

    print(f"  [L1b] cluster heads: {L1B_CLUSTER_COLS}", flush=True)
    print(f"  [L1b] latent heads (internal diagnostics): {L1B_LATENT_HEADS}", flush=True)
    print(
        f"  [L1b] latent explained_variance_ratio="
        f"{np.round(np.asarray(unsup_meta['latent_head_meta']['explained_variance_ratio'], dtype=np.float32), 4).tolist()}  "
        f"cluster_temperature={float(unsup_meta['cluster_temperature']):.4f}",
        flush=True,
    )
    _log_l1b_unsupervised_diagnostics(
        outputs,
        train_mask=unsup_mask,
        val_mask=val_report_mask,
        cluster_cols=list(L1B_CLUSTER_COLS),
        latent_cols=list(L1B_LATENT_EMBED_COLS),
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    for col in L1B_OUTPUT_COLS:
        if col not in outputs.columns:
            outputs[col] = 0.0
    _out_keep = ["symbol", "time_key"] + [c for c in L1B_OUTPUT_COLS if c in outputs.columns]
    outputs = outputs[_out_keep]

    meta = {
        "schema_version": L1B_SCHEMA_VERSION,
        "l1b_l1a_inputs": l1b_l1a_inputs_enabled(),
        "l1b_l1a_feature_tier": l1b_l1a_feature_tier(),
        "l1b_l1a_feature_col_names": [c for c in feature_cols if str(c).startswith("l1a_")],
        "l1b_baseline_align_to_l1a_pool": bool(l1b_baseline_align_to_l1a_pool_enabled()),
        "l1b_expand_oof_shifted": bool(l1b_should_use_shifted_expand_oof_windows()),
        "l1b_expanding_oof_fold_count": (
            len(l1b_expand_oof_val_windows())
            if l1b_should_use_shifted_expand_oof_windows() and l1_oof_mode_from_env() == "expanding"
            else None
        ),
        "l1_oof_folds": int(n_l1_oof),
        "l1_oof_enabled": bool(n_l1_oof >= 2),
        "l1b_unsupervised_oof_stitched": bool(unsup_oof_stitched),
        "l1b_unsupervised_oof_rows": int(unsup_oof_rows),
        "l1b_ortho_mode": (os.environ.get("L1B_ORTHO_MODE", "compact") or "compact").strip().lower(),
        "l1b_handcrafted_enabled": os.environ.get("L1B_HANDCRAFTED", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        "l1b_ctx_stagger": l1_ctx_stagger_enabled(),
        "l1b_bo_composite_enabled": os.environ.get("L1B_BO_COMPOSITE", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        "l1b_auto_prune": prune_on,
        "l1b_auto_prune_refit": prune_refit,
        "l1b_prune_meta": pruned_meta,
        "feature_cols": feature_cols,
        "supervised_feature_cols": supervised_feature_cols,
        "supervised_atomic_cols": atomic_supervised,
        "output_cols": L1B_OUTPUT_COLS,
        "model_output_cols": list(L1B_SUPERVISED_REGRESSOR_COLS),
        "cluster_output_cols": list(L1B_CLUSTER_COLS),
        "latent_output_cols": list(L1B_LATENT_EMBED_COLS),
        "internal_unsupervised_cols": list(L1B_UNSUPERVISED_COLS),
        "unsupervised_output_cols": list(L1B_EXPORT_UNSUPERVISED_COLS),
        "deprecated_output_cols": [
            "l1b_pullback_setup",
            "l1b_failure_risk",
            "l1b_shock_risk",
            "l1b_latent_0",
            "l1b_latent_1",
            "l1b_latent_2",
            "l1b_latent_3",
            "l1b_follow_through_score",
            "l1b_liquidity_score",
            "l1b_breakout_quality",
            "l1b_mean_reversion_setup",
            "l1b_trend_strength",
            "l1b_range_reversal_setup",
            "l1b_failed_breakout_setup",
            "l1b_setup_alignment",
        ],
        "constant_output_values": {},
        "model_files": dict(edge_dq_model_files),
        "head_feature_cols": {
            "l1b_edge_candidate_model": list(supervised_feature_cols),
            "l1b_edge_quality_model": list(supervised_feature_cols),
            "l1b_dq_pred": list(supervised_feature_cols),
            "l1b_edge_pred": list(supervised_feature_cols),
        },
        "weak_supervision_semantics": (
            "Exported L1b cache: unsupervised (clusters + AE/IF novelty + regime_change) + "
            "l1b_edge_pred/l1b_dq_pred/l1b_edge_candidate_tau from LGBM boosters; schema 1.24.0+."
        ),
        "unsupervised_semantics": (
            "exported unsupervised contract = soft cluster posteriors + novelty + regime-change; "
            "latent embeddings remain internal diagnostics and are not exported to L2 by default"
        ),
        "latent_head_meta": unsup_meta["latent_head_meta"],
        "unsupervised_block_meta": unsup_meta,
        "supervised_edge_dq_block_meta": edge_dq_block_meta,
        "supervised_edge_dq_semantics": (
            "Default L1B_SUPERVISED_TARGET_MODE=forward_range: l1b_edge_pred = P(high-range)×E[range|candidate] using "
            "decision_forward_range_atr (L2-aligned). AE+IsolationForest novelty (L1B_NOVELTY_MODE=blend). "
            "Legacy: set L1B_SUPERVISED_TARGET_MODE=staged_edge. DQ = path-balance unless L1B_DQ_TARGET_MODE=legacy."
        ),
        "output_cache_file": L1B_OUTPUT_CACHE_FILE,
    }
    edge_meta = meta.get("supervised_edge_dq_block_meta") or {}
    adaptive_min_samples = int(os.environ.get("ADAPTIVE_THRESHOLD_MIN_SAMPLES", "500"))
    meta = attach_threshold_registry(
        meta,
        "l1b",
        [
            threshold_entry(
                "L1B_EDGE_CANDIDATE_TAU",
                float(edge_meta.get("edge_candidate_tau", _l1b_edge_candidate_tau())),
                category="adaptive_candidate",
                role="edge candidate gating tau",
                adaptive_hint="MAD+z default; bootstrap fallback on low-n/heavy-tail",
                n_samples_used=int(edge_meta.get("edge_candidate_tau_fit_samples", 0)),
                min_reliable_samples=adaptive_min_samples,
                statistical_principle=str(edge_meta.get("edge_candidate_tau_statistical_principle", "")),
                alpha=float(edge_meta.get("edge_candidate_tau_alpha", float("nan"))),
                method_selected=str(edge_meta.get("edge_candidate_tau_method", "")),
                fallback_reason=str(edge_meta.get("edge_candidate_tau_fallback_reason", "")),
            ),
            threshold_entry(
                "L1B_EDGE_CANDIDATE_TAU_MODE",
                str(edge_meta.get("edge_candidate_tau_mode", "hybrid")),
                category="adaptive_candidate",
                role="tau estimation mode",
                statistical_principle="estimator_selection",
                method_selected=str(edge_meta.get("edge_candidate_tau_mode", "hybrid")),
            ),
            threshold_entry(
                "L1B_EDGE_CANDIDATE_TAU_METHOD",
                str(edge_meta.get("edge_candidate_tau_method", "mad_z")),
                category="adaptive_candidate",
                role="selected tau estimator for this run",
                method_selected=str(edge_meta.get("edge_candidate_tau_method", "mad_z")),
            ),
            threshold_entry(
                "L1B_EDGE_TAU_ALPHA",
                float(edge_meta.get("edge_candidate_tau_alpha", float(np.clip(float(os.environ.get("L1B_EDGE_TAU_ALPHA", "0.05")), 1e-4, 0.20)))),
                category="adaptive_candidate",
                role="significance level for tau formula",
                statistical_principle="two_sided_significance_level",
                alpha=float(edge_meta.get("edge_candidate_tau_alpha", float(np.clip(float(os.environ.get("L1B_EDGE_TAU_ALPHA", "0.05")), 1e-4, 0.20)))),
            ),
            threshold_entry(
                "L1B_EDGE_DQ_ES_ROUNDS",
                int(os.environ.get("L1B_EDGE_DQ_ES_ROUNDS", str(35 if FAST_TRAIN_MODE else 70))),
                category="data_guardrail",
                role="lgb early stopping rounds",
            ),
        ],
    )
    for w in meta.get("threshold_registry", {}).get("warnings", []):
        print(f"  [L1b][warn] {w}", flush=True)
    with open(os.path.join(MODEL_DIR, L1B_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    print(f"  [L1b] meta saved  -> {os.path.join(MODEL_DIR, L1B_META_FILE)}", flush=True)
    print(f"  [L1b] cache saved -> {cache_path}", flush=True)
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L1b] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )
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


def infer_l1b_market_descriptor(
    models: dict[str, lgb.Booster],
    meta: dict[str, Any],
    df: pd.DataFrame,
    *,
    infer_stage_pbar: Any | None = None,
) -> pd.DataFrame:
    """infer_stage_pbar: optional tqdm from the caller (e.g. ``[QQQ] infer``); L1b substages update its postfix."""
    # Shallow copy: we only append missing feature / atomic columns (same pattern as L1a infer).
    work = df.copy(deep=False)
    feature_cols = list(meta["feature_cols"])
    _materialize_l1b_tabular_inputs_for_infer(work, feature_cols)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    _l1b_attach_atomic_supervised_columns(work, feature_cols)
    sup_cols = list(meta.get("supervised_feature_cols") or feature_cols)
    dq_meta = meta.get("supervised_edge_dq_block_meta") or {}
    dq_lo, dq_hi = dq_meta.get("dq_pred_clip", [-5.0, 5.0])
    dq_lo, dq_hi = float(dq_lo), float(dq_hi)
    edge_clip = dq_meta.get("edge_pred_clip", [0.0, 5.0])
    edge_lo, edge_hi = float(edge_clip[0]), float(edge_clip[1])
    unsup_meta = meta.get("unsupervised_block_meta") or {}
    infer_steps = 2 if unsup_meta else 1
    own_pbar = None
    if infer_stage_pbar is None and _lgb_round_tqdm_enabled():
        own_pbar = tqdm(
            total=infer_steps,
            desc="[L1b] infer",
            unit="step",
            leave=False,
            mininterval=0.2,
            file=_tqdm_stream(),
            dynamic_ncols=True,
        )

    def _sub(s: str) -> None:
        if infer_stage_pbar is not None:
            infer_stage_pbar.set_postfix_str(s)
            try:
                infer_stage_pbar.refresh()
            except Exception:
                pass
        elif own_pbar is not None:
            own_pbar.set_postfix_str(s)
        try:
            _tqdm_stream().flush()
        except Exception:
            pass

    try:
        outputs = pd.DataFrame({"symbol": work["symbol"].values, "time_key": pd.to_datetime(work["time_key"])})
        if unsup_meta:
            _sub("L1b · unsup")
            outputs = pd.concat(
                [
                    outputs,
                    _apply_l1b_unsupervised_block(work, unsup_meta, on_substage=_sub).reset_index(drop=True),
                ],
                axis=1,
            )
            if own_pbar is not None:
                own_pbar.update(1)
        _sub("L1b · lgbm")
        for col in sup_cols:
            if col not in work.columns:
                work[col] = 0.0
        X_inf = work[sup_cols].to_numpy(dtype=np.float64, copy=False)
        edge_model_type = str(dq_meta.get("edge_model_type", "") or "")
        tau = float(dq_meta.get("edge_candidate_tau", _l1b_edge_candidate_tau()))
        if edge_model_type == "staged_tradeability":
            cand_mdl = models.get("l1b_edge_candidate_model")
            qual_mdl = models.get("l1b_edge_quality_model")
            if cand_mdl is not None and qual_mdl is not None:
                cand_p = np.clip(cand_mdl.predict(X_inf).astype(np.float64), 0.0, 1.0)
                qual_pred = np.clip(qual_mdl.predict(X_inf).astype(np.float64), tau, edge_hi)
                outputs["l1b_edge_pred"] = np.clip(cand_p * qual_pred, edge_lo, edge_hi).astype(np.float32)
        else:
            mdl = models.get("l1b_edge_pred")
            if mdl is not None:
                pred = np.clip(mdl.predict(X_inf).astype(np.float64), edge_lo, edge_hi)
                outputs["l1b_edge_pred"] = pred.astype(np.float32)
        outputs["l1b_edge_candidate_tau"] = np.full(len(outputs), float(tau), dtype=np.float32)
        dq_mdl = models.get("l1b_dq_pred")
        if dq_mdl is not None:
            dq_pred = np.clip(dq_mdl.predict(X_inf).astype(np.float64), dq_lo, dq_hi)
            outputs["l1b_dq_pred"] = dq_pred.astype(np.float32)
        out_cols = list(meta.get("output_cols", L1B_OUTPUT_COLS))
        for col in out_cols:
            if col not in outputs.columns:
                outputs[col] = 0.0
        _keep = ["symbol", "time_key"] + out_cols
        if own_pbar is not None:
            own_pbar.update(1)
        return outputs[_keep]
    finally:
        if own_pbar is not None:
            own_pbar.close()

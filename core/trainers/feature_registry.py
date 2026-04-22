"""Single source of truth for layer-wise feature policy (pool vs selectors).

``prepare_dataset`` builds ``feat_cols``; L1a/L1b consume subsets (L1c archived). This module
documents coverage and supplies shared column lists so prefix rules are not scattered.
"""

from __future__ import annotations

import os
from typing import Any, FrozenSet

from core.trainers.constants import BO_FEAT_COLS, PA_CTX_FEATURES, PA_STRADDLE_FEATURES

# --- L1a / L1b pa_ctx stagger (default ON): L1a = short-horizon / composite ctx; L1b = structural ctx ---
L1A_PREFERRED_CORE: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "lbl_atr",
    "pa_vol_rvol",
    "pa_vol_momentum",
    "pa_bull_pressure",
    "pa_bear_pressure",
    "pa_or_breakout_strength",
    "pa_struct_swing_range_atr",
    "pa_vol_zscore_20",
    "pa_bo_wick_imbalance",
    "pa_bo_close_extremity",
    "pa_lead_macd_hist_slope",
    "pa_lead_rsi_slope",
    "pa_bo_dist_vwap",
)

# Sequence-friendly ctx (straddle stack: range pressure only; directional ctx dropped from LGBM).
L1A_CTX_DYNAMIC_COLUMNS: tuple[str, ...] = ("pa_ctx_range_pressure",)

# Legacy L1a ctx block (includes structure_veto) when stagger is disabled.
L1A_CTX_LEGACY_COLUMNS: tuple[str, ...] = L1A_CTX_DYNAMIC_COLUMNS + ("pa_ctx_structure_veto",)

L1A_CTX_DYNAMIC_SET: FrozenSet[str] = frozenset(L1A_CTX_DYNAMIC_COLUMNS)

# Granular / structural ctx for L1b only under stagger (trend legs, failed BO, premise, veto).
L1B_CTX_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "pa_ctx_setup_trend_long",
    "pa_ctx_setup_trend_short",
    "pa_ctx_setup_pullback_long",
    "pa_ctx_setup_pullback_short",
    "pa_ctx_setup_range_long",
    "pa_ctx_setup_range_short",
    "pa_ctx_setup_failed_breakout_long",
    "pa_ctx_setup_failed_breakout_short",
    "pa_ctx_structure_veto",
    "pa_ctx_premise_break_long",
    "pa_ctx_premise_break_short",
)

L1B_BASE_PREF_TAIL: tuple[str, ...] = (
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
)

# Full L1b base (all pa_ctx) — used when ``L1_CTX_STAGGER=0``.
L1B_BASE_PREF_LEGACY_FULL: tuple[str, ...] = tuple(BO_FEAT_COLS) + tuple(PA_CTX_FEATURES) + L1B_BASE_PREF_TAIL

# Staggered L1b: bo + structural ctx + tail (no L1a dynamic ctx).
L1B_BASE_PREF_STAGGERED: tuple[str, ...] = tuple(BO_FEAT_COLS) + L1B_CTX_STRUCTURAL_COLUMNS + L1B_BASE_PREF_TAIL


def l1_ctx_stagger_enabled() -> bool:
    return os.environ.get("L1_CTX_STAGGER", "1").strip().lower() not in {"0", "false", "no", "off"}


def l1a_preferred_columns() -> tuple[str, ...]:
    straddle = tuple(PA_STRADDLE_FEATURES)
    if l1_ctx_stagger_enabled():
        return L1A_PREFERRED_CORE + L1A_CTX_DYNAMIC_COLUMNS + straddle
    return L1A_PREFERRED_CORE + L1A_CTX_LEGACY_COLUMNS + straddle


def l1b_base_pref_columns() -> tuple[str, ...]:
    if l1_ctx_stagger_enabled():
        return L1B_BASE_PREF_STAGGERED
    return L1B_BASE_PREF_LEGACY_FULL


# Extras: any ``pa_*`` in ``feat_cols`` except these prefixes (orthogonal / tabular stats).
L1A_EXTRA_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "pa_hmm_",
    "pa_garch_",
    "pa_hsmm_",
    "pa_egarch_",
)


def l1a_extra_accepts_pa_ctx(name: str) -> bool:
    """Under stagger, L1a extras may only add dynamic ctx (not structural L1b ctx)."""
    if not name.startswith("pa_ctx_"):
        return True
    if not l1_ctx_stagger_enabled():
        return True
    return name in L1A_CTX_DYNAMIC_SET


# --- Prepared dataset: which downstream layers need which expensive blocks ---
_DEFAULT_PREP_TARGETS: FrozenSet[str] = frozenset({"l1a", "l1b", "l2"})


def parse_prep_layer_targets() -> FrozenSet[str]:
    raw = (os.environ.get("PREPARED_DATASET_LAYER_TARGETS", "") or "").strip()
    if not raw:
        return _DEFAULT_PREP_TARGETS
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return parts if parts else _DEFAULT_PREP_TARGETS


def prep_needs_tcn_derivatives(targets: FrozenSet[str]) -> bool:
    """TCN forward + ``tcn_*`` columns: opt in via ``PREPARED_DATASET_LAYER_TARGETS`` including ``l1c`` (archived) or ``tcn``."""
    return "l1c" in targets or "tcn" in targets


def prep_needs_mamba(targets: FrozenSet[str]) -> bool:
    """Mamba block is experimental; only run if opted in *and* targets include ``mamba`` or legacy ``l1c``."""
    return "mamba" in targets or "l1c" in targets


LAYER_FEATURE_COVERAGE: dict[str, dict[str, Any]] = {
    "feat_cols_pool": {
        "numeric_pa_or": "From ``_pa_feature_cols`` after PA+labels merge (excludes string tag cols).",
        "bo": f"Always merged into pool after ``ensure_breakout_features``: {list(BO_FEAT_COLS)}",
        "pa_ctx": f"Merged after ``ensure_structure_context_features``: {list(PA_CTX_FEATURES)}",
        "pa_straddle": f"Merged into LGBM pool (causal 1m): {list(PA_STRADDLE_FEATURES)}",
        "tcn": "Optional; when targets include ``tcn`` or legacy ``l1c`` (see prep_needs_tcn_derivatives).",
        "mamba": "Optional; ``ENABLE_EXPERIMENTAL_MAMBA`` + checkpoint + targets include ``mamba`` or legacy ``l1c``.",
    },
    "l1a": {
        "selector": "_select_l1a_feature_cols",
        "preferred": "l1a_preferred_columns() — range_pressure + straddle vol features (+ legacy ctx_veto when stagger off).",
        "extra_policy": f"Ranked ``pa_*`` from feat_cols excluding prefixes {L1A_EXTRA_EXCLUDE_PREFIXES}; "
        "under stagger, non-dynamic ``pa_ctx_*`` excluded from extras.",
    },
    "l1b": {
        "selector": "_select_l1b_feature_cols",
        "base_pref": "l1b_base_pref_columns() — structural ctx only when stagger; full ctx when L1_CTX_STAGGER=0",
    },
    "l1c": {
        "selector": "(archived) archive/l1c/",
        "pool": "Layer removed from main pipeline; see archive/train_layer1c_only.py.",
    },
    "l2": {
        "selector": "_build_l2_frame",
        "raw_residual": "Uses pa_ctx / pa_state / bo_* slices from df (not full feat_cols list).",
    },
}


def describe_feature_coverage() -> str:
    lines = ["Layer feature coverage (registry):"]
    for layer, body in LAYER_FEATURE_COVERAGE.items():
        lines.append(f"  [{layer}]")
        for k, v in body.items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)

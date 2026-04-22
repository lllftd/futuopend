"""Merge L1a outputs into L1b training/infer frames and configure L1b L1a feature tiers (B0/B1/B2)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.trainers.constants import L1A_REGIME_COLS, TRAIN_END, l1a_straddle_edge_head_enabled


def l1b_l1a_inputs_enabled() -> bool:
    return os.environ.get("L1B_USE_L1A_FEATURES", "").strip().lower() in {"1", "true", "yes"}


def l1b_l1a_feature_tier() -> str:
    """none | scalar | full — baseline (B0) / scalar L1a (B1) / +embeddings (B2)."""
    raw = (os.environ.get("L1B_L1A_FEATURE_TIER", "") or "").strip().lower()
    if raw in ("none", "0", "baseline", "off"):
        return "none"
    if raw in ("1", "scalar", "b1"):
        return "scalar"
    if raw in ("2", "full", "b2"):
        return "full"
    if l1b_l1a_inputs_enabled():
        return "scalar"
    return "none"


def _l1a_embed_dim() -> int:
    return max(4, int(os.environ.get("L1A_EMBED_DIM", "8")))


def l1b_l1a_scalar_feature_names() -> list[str]:
    """5× regime probs + five L1a scalar heads; + l1a_straddle_edge when edge head is enabled (same env as L1a)."""
    out = list(L1A_REGIME_COLS) + [
        "l1a_transition_risk",
        "l1a_vol_forecast",
        "l1a_vol_trend",
        "l1a_time_in_regime",
        "l1a_state_persistence",
    ]
    if l1a_straddle_edge_head_enabled():
        out.append("l1a_straddle_edge")
    return out


def l1b_l1a_full_feature_names(embed_dim: int | None = None) -> list[str]:
    d = _l1a_embed_dim() if embed_dim is None else max(4, int(embed_dim))
    return l1b_l1a_scalar_feature_names() + [f"l1a_market_embed_{i}" for i in range(d)]


def l1b_l1a_feature_names_for_tier(tier: str | None = None) -> list[str]:
    t = tier or l1b_l1a_feature_tier()
    if t == "none":
        return []
    if t == "scalar":
        return l1b_l1a_scalar_feature_names()
    return l1b_l1a_full_feature_names()


def configure_l1b_l1a_allowlist_from_tier() -> None:
    """Set L1B_EXTRA_FEATURE_ALLOWLIST from tier when enabled and allowlist not already set."""
    if not l1b_l1a_inputs_enabled():
        return
    if l1b_l1a_feature_tier() == "none":
        return
    existing = os.environ.get("L1B_EXTRA_FEATURE_ALLOWLIST", "").strip()
    if existing:
        return
    names = l1b_l1a_feature_names_for_tier()
    if not names:
        return
    os.environ["L1B_EXTRA_FEATURE_ALLOWLIST"] = ",".join(names)


def attach_l1a_outputs_to_df(
    df: pd.DataFrame,
    l1a_df: pd.DataFrame,
    *,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Left-merge L1a columns into ``df`` (drops prior ``l1a_*`` on ``df``).

    If ``cols`` is set, merge only those names (must exist on ``l1a_df``); otherwise all ``l1a_*`` on ``l1a_df``.
    Use ``cols`` for tier B1/B2 so the frame matches training/infer without unused L1a columns.
    """
    key_cols = ["symbol", "time_key"]
    for c in key_cols:
        if c not in df.columns or c not in l1a_df.columns:
            raise KeyError(f"L1a merge requires columns {key_cols} on both frames; missing {c!r}")
    if cols is None:
        l1a_cols = [c for c in l1a_df.columns if c.startswith("l1a_")]
    else:
        l1a_cols = [c for c in cols if c.startswith("l1a_")]
    if not l1a_cols:
        raise ValueError("No L1a columns to merge (cols empty or l1a_df missing requested names).")
    missing = [c for c in l1a_cols if c not in l1a_df.columns]
    if missing:
        raise KeyError(f"l1a_df missing columns required for merge: {missing[:12]!r}")
    sub = l1a_df[key_cols + l1a_cols].copy()
    sub["time_key"] = pd.to_datetime(sub["time_key"])
    out = df.copy()
    out["time_key"] = pd.to_datetime(out["time_key"])
    drop = [c for c in out.columns if c.startswith("l1a_")]
    if drop:
        out = out.drop(columns=drop, errors="ignore")
    return out.merge(sub, on=key_cols, how="left")


def l1a_feature_cols_from_l1b_meta(meta: dict) -> list[str]:
    """``l1a_*`` names from checkpoint ``feature_cols`` (for infer-time merge)."""
    fc = meta.get("feature_cols") or []
    return [str(c) for c in fc if str(c).startswith("l1a_")]


def extend_feat_cols_with_l1a(feat_cols: list[str], df: pd.DataFrame) -> list[str]:
    """Append tier-selected ``l1a_*`` names that exist on ``df``."""
    if not l1b_l1a_inputs_enabled() or l1b_l1a_feature_tier() == "none":
        return feat_cols
    want = set(l1b_l1a_feature_names_for_tier())
    fc: list[str] = list(feat_cols)
    seen = set(fc)
    for c in sorted(want):
        if c in df.columns and c not in seen:
            fc.append(c)
            seen.add(c)
    return fc


def l1b_baseline_align_to_l1a_pool_enabled() -> bool:
    """Strict ablation: baseline without ``l1a_*`` still uses the same fit pool + OOF grid as L1a runs.

    When ``L1B_BASELINE_ALIGN_TO_L1A_POOL=1`` and L1a columns are off, apply ``t >= TRAIN_END`` (honest pool)
    and ``L1B_EXPAND_OOF_VAL_WINDOWS`` (typically 3 expanding folds). Redundant when L1a features are on.
    """
    if not os.environ.get("L1B_BASELINE_ALIGN_TO_L1A_POOL", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if l1b_l1a_inputs_enabled() and l1b_l1a_feature_tier() != "none":
        return False
    return True


def l1b_should_use_shifted_expand_oof_windows() -> bool:
    """Use ``L1B_EXPAND_OOF_VAL_WINDOWS`` for L1b expanding OOF (see ``constants``)."""
    if l1b_l1a_inputs_enabled() and l1b_l1a_feature_tier() != "none":
        return True
    return l1b_baseline_align_to_l1a_pool_enabled()


def l1b_use_honest_l1a_fit_pool() -> bool:
    """Honest fit pool (``t >= TRAIN_END``): L1a tiers, or baseline when align flag set."""
    if l1b_l1a_inputs_enabled() and l1b_l1a_feature_tier() != "none":
        return True
    return l1b_baseline_align_to_l1a_pool_enabled()


def l1b_apply_honest_l1a_fit_mask(work: pd.DataFrame, l1_fit_mask: np.ndarray) -> np.ndarray:
    """Restrict fit pool to rows with honest stitched L1a OOF (``t >= TRAIN_END``)."""
    m = np.asarray(l1_fit_mask, dtype=bool).ravel()
    ts = pd.to_datetime(work["time_key"])
    return m & (ts >= np.datetime64(TRAIN_END))


def meta_expects_l1a_features(meta: dict) -> bool:
    fc = meta.get("feature_cols") or []
    return any(str(c).startswith("l1a_") for c in fc)

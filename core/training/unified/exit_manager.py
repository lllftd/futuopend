"""Load L2 unified net + L3 exit wrapper for inference (exit scores from Phase 2)."""

from __future__ import annotations

import os
import pickle
from typing import Any

import torch

from core.training.common.constants import L2_META_FILE, L2_UNIFIED_MODEL_FILE, L3_META_FILE, L3_SCHEMA_VERSION, MODEL_DIR

from core.training.unified.exit_wrapper import UnifiedExitWrapper
from core.training.unified.model import UnifiedL2L3Net
from core.training.unified.trajectory import L3TrajectoryConfig


def load_l3_exit_manager() -> tuple[dict[str, Any], dict[str, Any]]:
    p_l2 = os.path.join(MODEL_DIR, L2_META_FILE)
    if not os.path.isfile(p_l2):
        raise FileNotFoundError(f"L2 meta missing: {p_l2}")
    with open(p_l2, "rb") as f:
        l2_meta: dict[str, Any] = pickle.load(f)
    if str(l2_meta.get("backend", "")) != "pytorch_unified":
        raise RuntimeError("L2 meta must be backend=pytorch_unified (unified L2+L3).")
    if not l2_meta.get("has_exit_value_heads"):
        raise RuntimeError(
            "L2 meta missing has_exit_value_heads. Retrain L2 with L2_UNIFIED_PHASE2=1."
        )
    p_l3 = os.path.join(MODEL_DIR, L3_META_FILE)
    if not os.path.isfile(p_l3):
        raise FileNotFoundError(f"L3 meta missing: {p_l3}")
    with open(p_l3, "rb") as f:
        meta: dict[str, Any] = pickle.load(f)
    msv = meta.get("schema_version")
    if msv != L3_SCHEMA_VERSION:
        raise RuntimeError(
            f"L3 schema mismatch: artifact {msv!r} != {L3_SCHEMA_VERSION!r}. Regenerate: train_pipeline layer3."
        )
    be = str(meta.get("backend", ""))
    if be != "l2_unified_exit":
        raise RuntimeError(
            f"L3 meta backend {be!r} != l2_unified_exit. Regenerate: train_pipeline layer3."
        )
    ufile = str(l2_meta.get("unified_model_file", L2_UNIFIED_MODEL_FILE))
    path_u = os.path.join(MODEL_DIR, ufile)
    if not os.path.isfile(path_u):
        raise FileNotFoundError(
            f"Unified L2/L3 weights missing: {path_u} (L2 meta unified_model_file={ufile!r})."
        )
    state = torch.load(path_u, map_location="cpu", weights_only=False)
    net = UnifiedL2L3Net.from_meta(l2_meta)
    net.load_state_dict(state, strict=True)
    net.eval()
    l3fc = [str(c) for c in (meta.get("l3_feature_cols") or meta.get("feature_cols") or ())]
    fcols2 = [str(c) for c in (l2_meta.get("feature_cols") or ())]
    ucfg = l2_meta.get("l2_unified_config") or {}
    midx = ucfg.get("market_idx", []) or []
    ridx = ucfg.get("regime_idx", []) or []
    mnames = [fcols2[i] for i in midx] if fcols2 and midx else []
    rnames = [fcols2[i] for i in ridx] if fcols2 and ridx else []
    npos = int(ucfg.get("n_position", ucfg.get("position_dim", 8)))
    if not l3fc:
        raise RuntimeError("L3 meta missing feature_cols.")
    train_stats = l2_meta.get("l2_unified_train_input_stats")
    vtn = l2_meta.get("l2_unified_value_target_norm")
    wx = UnifiedExitWrapper(
        net,
        l3fc,
        mnames,
        rnames,
        n_pos=npos,
        device="cpu",
        train_input_stats=train_stats if isinstance(train_stats, dict) else None,
        value_target_norm=vtn if isinstance(vtn, dict) else None,
    )
    return {"exit": wx, "value": None}, meta


def load_l3_trajectory_encoder_for_infer(meta: dict[str, Any]) -> tuple[Any, L3TrajectoryConfig | None]:
    """Trajectory GRU removed; kept for call-site compatibility."""
    _ = meta
    return None, None

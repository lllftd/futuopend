"""L3 policy feature registry (meta) — exit weights live in L2 unified checkpoint after Phase 2."""

from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from core.training.common.constants import L3_META_FILE, L3_SCHEMA_VERSION, MODEL_DIR
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry

from core.training.unified.trajectory import L3TrajectoryConfig
from core.training.unified.config import defaults as L3DEF


def _write_l3_policy_exit_metadata(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
) -> "L3TrainingBundle":
    from core.training.unified.policy_data import (  # noqa: WPS433
        L3TrainingBundle,
        _build_l3_policy_dataset,
        _l3_value_target_mode,
    )

    print(
        "  [L3] writing policy feature registry (backend=l2_unified_exit) — no separate exit MLP.",
        flush=True,
    )
    traj_cfg = L3TrajectoryConfig()
    max_hold = int(L3DEF.max_hold_bars())
    _X, _y_exit, _y_value, _t_state, feature_cols, _rows_entry, _, _, _rows_from_model, dataset_policy = _build_l3_policy_dataset(
        df,
        l1a_outputs,
        l2_outputs,
        max_hold=max_hold,
        traj_cfg=traj_cfg,
        build_traj=False,
    )
    if _X.shape[0] == 0:
        raise RuntimeError("L3: policy dataset is empty (cannot write feature registry).")
    value_disabled = True
    meta: dict[str, Any] = {
        "schema_version": L3_SCHEMA_VERSION,
        "backend": "l2_unified_exit",
        "feature_cols": list(feature_cols),
        "l3_feature_cols": list(feature_cols),
        "l3_value_disabled": value_disabled,
        "l3_value_target_mode": str(_l3_value_target_mode()) if not value_disabled else "disabled",
        "dataset_policy": dict(dataset_policy),
        "model_files": {},
        "l3_exit_weights_in_l2": True,
    }
    meta = attach_threshold_registry(
        meta,
        "l3",
        [threshold_entry("l2_unified_exit", 1.0, category="build", role="policy feature registry", adaptive_hint="")],
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, L3_META_FILE)
    with open(out_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  [L3] policy metadata -> {out_path}", flush=True)
    _fcols = list(feature_cols)
    print(
        f"  [L3] policy registry: n_l3_feature_cols={len(_fcols)}  meta -> {out_path!r}",
        flush=True,
    )
    return L3TrainingBundle(models={}, meta=meta)


def train_l3_exit_manager(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
) -> "L3TrainingBundle":
    """Build policy rows once, persist ``l3_feature_cols`` + ``dataset_policy`` to ``L3_META_FILE`` (no separate exit MLP).

    Exit weights live in the L2 ``UnifiedL2L3Net`` when ``L2_UNIFIED_PHASE2=1``.  ``train_pipeline`` runs this
    automatically after L2 (unless ``L2_AUTO_L3_META=0``); use ``--start-from layer3`` to refresh the registry only.
    """
    return _write_l3_policy_exit_metadata(df, l1a_outputs, l2_outputs)

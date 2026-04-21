from __future__ import annotations

import os

from core.trainers.lgbm_utils import _tq, _tqdm_stream


def _tqdm_disabled() -> bool:
    """Match ``_tq`` / ``FORCE_TQDM`` policy for epoch logging in TCN train."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return True
    if not _tqdm_stream().isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return True
    return False


__all__ = ["_tq", "_tqdm_disabled"]

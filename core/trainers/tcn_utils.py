from __future__ import annotations

import os
import sys
from tqdm.auto import tqdm

def _tqdm_disabled() -> bool:
    """Align with tqdm policy: DISABLE_TQDM=1 off; non-TTY off unless FORCE_TQDM=1."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return True
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return True
    return False


def _tq(it, **kwargs):
    """Iterator progress bar (epochs, etc.)."""
    return tqdm(it, disable=_tqdm_disabled(), **kwargs)



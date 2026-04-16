"""Small env readers for L3 defaults (read at call time when using functions, or at import for module constants)."""

from __future__ import annotations

import os


def env_int(key: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    raw = os.environ.get(key, "").strip()
    v = default if not raw else int(raw)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return int(v)


def env_float(key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.environ.get(key, "").strip()
    v = default if not raw else float(raw)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return float(v)


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return default if not v else v

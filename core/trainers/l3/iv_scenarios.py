from __future__ import annotations

import os

import numpy as np


def scenario_count() -> int:
    return max(1, int(os.environ.get("L3_STRADDLE_SCENARIO_COUNT", "10")))


def dte_grid_days() -> list[int]:
    raw = (os.environ.get("L3_STRADDLE_DTE_GRID", "7,14,21,30") or "").strip()
    vals: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if x.isdigit():
            vals.append(max(1, int(x)))
    return vals or [7, 14, 21, 30]


def generate_iv_scenarios(
    base_iv: float,
    n_minutes: int,
    *,
    rng: np.random.Generator | None = None,
    n_scenarios: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate training IV paths around a base IV level."""
    rng = rng or np.random.default_rng(7)
    n = max(1, int(n_scenarios or scenario_count()))
    base = float(np.clip(base_iv, 0.05, 5.0))
    mins = max(1, int(n_minutes))
    out: dict[str, np.ndarray] = {}
    counts = {
        "stable": max(1, int(round(n * 0.30))),
        "crush": max(1, int(round(n * 0.25))),
        "expansion": max(1, int(round(n * 0.25))),
        "rise_fall": max(1, n - int(round(n * 0.30)) - int(round(n * 0.25)) - int(round(n * 0.25))),
    }

    for i in range(counts["stable"]):
        noise = rng.normal(0.0, base * 0.015, mins).cumsum()
        path = np.maximum(base + noise, base * 0.50)
        out[f"stable_{i}"] = path.astype(np.float32)

    for i in range(counts["crush"]):
        crush_minute = int(rng.integers(0, max(2, min(60, mins))))
        path = np.full(mins, base, dtype=np.float64)
        crush_mag = float(rng.uniform(0.15, 0.40))
        path[crush_minute:] = base * (1.0 - crush_mag)
        path += rng.normal(0.0, base * 0.01, mins)
        out[f"crush_{i}"] = np.maximum(path, 0.05).astype(np.float32)

    for i in range(counts["expansion"]):
        start = int(rng.integers(0, max(2, mins // 2)))
        path = np.full(mins, base, dtype=np.float64)
        expansion = float(rng.uniform(0.20, 0.60))
        ramp = np.linspace(0.0, expansion, max(1, mins - start))
        path[start:] = base * (1.0 + ramp[: len(path[start:])])
        out[f"expansion_{i}"] = np.maximum(path, 0.05).astype(np.float32)

    for i in range(counts["rise_fall"]):
        peak = int(rng.integers(max(1, mins // 3), max(2, 2 * mins // 3)))
        path = np.zeros(mins, dtype=np.float64)
        rise = np.linspace(0.0, 0.30, max(1, peak))
        fall = np.linspace(0.30, -0.10, max(1, mins - peak))
        path[:peak] = rise[:peak]
        path[peak:] = fall[: len(path[peak:])]
        out[f"rise_fall_{i}"] = np.maximum(base * (1.0 + path), 0.05).astype(np.float32)
    return out

from __future__ import annotations

from typing import Any


def threshold_entry(
    name: str,
    value: Any,
    *,
    category: str,
    role: str,
    source: str = "env_or_code_default",
    adaptive_hint: str = "",
    n_samples_used: int | None = None,
    min_reliable_samples: int | None = None,
    statistical_principle: str = "",
    alpha: float | None = None,
    power: float | None = None,
    cost_input: float | None = None,
    method_selected: str = "",
    fallback_reason: str = "",
) -> dict[str, Any]:
    out = {
        "name": str(name),
        "value": value,
        "category": str(category),  # adaptive_candidate | safety_constraint | data_guardrail
        "role": str(role),
        "source": str(source),
        "adaptive_hint": str(adaptive_hint),
    }
    if n_samples_used is not None:
        out["n_samples_used"] = int(n_samples_used)
    if min_reliable_samples is not None:
        out["min_reliable_samples"] = int(min_reliable_samples)
    if statistical_principle:
        out["statistical_principle"] = str(statistical_principle)
    if alpha is not None:
        out["alpha"] = float(alpha)
    if power is not None:
        out["power"] = float(power)
    if cost_input is not None:
        out["cost_input"] = float(cost_input)
    if method_selected:
        out["method_selected"] = str(method_selected)
    if fallback_reason:
        out["fallback_reason"] = str(fallback_reason)
    return out


def attach_threshold_registry(meta: dict[str, Any], layer: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    m = dict(meta)
    m["threshold_registry"] = {
        "layer": str(layer),
        "entries": list(entries),
    }
    counts = {"adaptive_candidate": 0, "safety_constraint": 0, "data_guardrail": 0}
    for e in entries:
        k = str(e.get("category", ""))
        if k in counts:
            counts[k] += 1
    warnings: list[str] = []
    for e in entries:
        n = e.get("n_samples_used")
        lo = e.get("min_reliable_samples")
        if isinstance(n, int) and isinstance(lo, int) and n < lo:
            warnings.append(f"{e.get('name')} low sample support: n={n} < {lo}")
        fr = str(e.get("fallback_reason", "") or "").strip()
        if fr:
            warnings.append(f"{e.get('name')} fallback activated: {fr}")
    m["threshold_registry"]["category_counts"] = counts
    m["threshold_registry"]["warnings"] = warnings
    return m

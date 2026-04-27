"""OOS append log under ``logs/``. Unified L2 readable summary + optional ``print_unified_stack_meta`` (for callers that still want JSON; L2 train no longer prints large JSON to log)."""

from __future__ import annotations

import json
import math
import os
from typing import Any

# Repo root: core/training/logging/metrics_file_log.py -> ../../..
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO_LOGS_DIR = os.path.join(_REPO_ROOT, "logs")
# Single file in train_pipeline for L2+auto L3: logs/l2l3_unified.log
L2L3_UNIFIED_LOG = "l2l3_unified.log"
OOS_UNIFIED_LOG = "oos_unified.log"
# ``print_unified_stack_meta`` only: avoid multi‑MB log lines (ndarray tolist, long index lists, etc.)
_MAX_LOG_JSON_LIST = 48
_MAX_LOG_ARRAY_ELEMS = 64


def _ensure_logs_dir() -> str:
    os.makedirs(REPO_LOGS_DIR, exist_ok=True)
    return REPO_LOGS_DIR


def json_log_safe(x: Any) -> Any:  # noqa: ANN401
    """Recursively make ``x`` JSON-serializable for **stdout / l2l3_unified.log** only.

    Large ``ndarray`` / long lists are **summarized** so logs stay readable; full data remains in
    ``*.pkl`` artifacts, not in this JSON block.
    """
    import numpy as np

    if x is None or isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(x, np.ndarray):
        if int(x.size) > _MAX_LOG_ARRAY_ELEMS:
            try:
                flat = x.ravel()
                s = f"shape={list(x.shape)} dtype={x.dtype!s}"
                if flat.size and np.issubdtype(flat.dtype, np.number):
                    q = np.isfinite(flat.astype(np.float64, copy=False))
                    if bool(q.any()):
                        fv = flat[q]
                        s += f" finite_min={float(fv.min()):.6g} finite_max={float(fv.max()):.6g} finite_mean={float(fv.mean()):.6g}"
                return {"_omitted_from_log": True, "summary": s}
            except Exception:  # noqa: BLE001
                return {
                    "_omitted_from_log": True,
                    "summary": f"shape={list(x.shape)} dtype={x.dtype!s} (no stats)",
                }
        return json_log_safe(x.tolist())
    mod = getattr(type(x), "__module__", "")
    if mod.startswith("torch") and hasattr(x, "numel") and hasattr(x, "shape"):
        try:
            n = int(x.numel())  # type: ignore[union-attr]
            if n > _MAX_LOG_ARRAY_ELEMS:
                return {
                    "_omitted_from_log": True,
                    "summary": f"torch.Tensor shape={list(x.shape)}",
                }
            return json_log_safe(x.detach().cpu().numpy())  # type: ignore[attr-defined, union-attr]
        except Exception:  # noqa: BLE001
            return str(x)[:2000]
    if isinstance(x, dict):
        return {str(k): json_log_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out: list[Any] = [json_log_safe(v) for v in x]
        if len(out) > _MAX_LOG_JSON_LIST:
            n = len(out) - _MAX_LOG_JSON_LIST
            return out[:_MAX_LOG_JSON_LIST] + [f"<{n} more list elements omitted from log>"]
        return out
    if isinstance(x, (bytes, bytearray, memoryview)):
        return str(x)[:200]
    if isinstance(x, (np.number,)) and not isinstance(x, (np.void,)):
        return json_log_safe(x.item())  # type: ignore[no-untyped-call]
    if hasattr(x, "tolist") and hasattr(x, "shape") and not isinstance(x, (dict, list, str)):
        try:
            return json_log_safe(x.tolist())  # type: ignore[no-untyped-call]
        except Exception:  # noqa: BLE001
            return str(x)[:2000]
    return str(x)[:2000]


# Backwards alias (same behavior as ``json_log_safe``; kept if external code used ``json_safe``)
json_safe = json_log_safe


def print_unified_stack_meta(title: str, payload: dict[str, Any]) -> None:
    """Print a JSON block to **stdout** (captured in ``logs/l2l3_unified.log`` when train_pipeline ``setup_logger("unified")`` is active)."""
    try:
        body = json.dumps(json_log_safe(payload), ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"  [unified][warn] could not JSON meta {title!r}: {e}", flush=True)
        return
    print(f"\n=== {title} (JSON) ===\n{body}\n=== end {title} ===\n", flush=True)


def print_l2_unified_meta_readable(title: str, payload: dict[str, Any]) -> None:
    """Short human-readable L2 meta block (key numbers only); full meta is in ``l2_decision_meta.pkl``."""
    dc = payload.get("dim_check") or {}
    hw = payload.get("head_weight_norms") or {}
    cal = payload.get("calibration_summary") or {}
    vtn = payload.get("l2_unified_value_target_norm")
    gprob = payload.get("phase2_gate_entry_mean_prob") or {}
    g_gate = (cal.get("gate") or {}) if isinstance(cal, dict) else {}
    lines: list[str] = [
        f"\n--- {title} ---",
        f"  features: n_cols={dc.get('n_feature_cols')}  market+regime+pos={dc.get('sum_check')}",
    ]
    if isinstance(hw, dict) and hw:
        top = []
        for k, v in list(hw.items())[:6]:
            if isinstance(v, (int, float)):
                top.append(f"{k}={v:.4g}")
            else:
                top.append(f"{k}={v!r}")
        lines.append("  head_weight_norms: " + ("; ".join(top) if top else "(empty)"))
    if g_gate:
        lines.append(
            f"  cal gate: pred_mean={g_gate.get('pred_mean')}  auroc={g_gate.get('auroc')}"
        )
    if isinstance(vtn, dict) and vtn:
        pm, ps = vtn.get("remaining_pnl_mean"), vtn.get("remaining_pnl_std")
        bm, bs = vtn.get("remaining_bars_mean"), vtn.get("remaining_bars_std")
        lines.append(
            f"  value_target_norm: enabled={vtn.get('enabled')}  "
            f"pnl μ/σ={pm}/{ps}  bars μ/σ={bm}/{bs}"
        )
    if isinstance(gprob, dict) and gprob.get("before") is not None:
        lines.append(
            f"  phase2 gate (fixed batch): before={gprob.get('before'):.4f}  "
            f"after={gprob.get('after'):.4f}  n={gprob.get('batch_rows')}"
        )
    lines.append(f"--- end {title} ---\n")
    print("\n".join(lines), flush=True)


def append_oos_unified_log(line: str, *, log_filename: str = OOS_UNIFIED_LOG) -> str:
    """Append one line to ``logs/oos_unified.log`` (OOS exit/drift/distribution)."""
    s = line if line.endswith("\n") else line + "\n"
    p = os.path.join(_ensure_logs_dir(), log_filename)
    with open(p, "a", encoding="utf-8") as f:
        f.write(s)
    return p

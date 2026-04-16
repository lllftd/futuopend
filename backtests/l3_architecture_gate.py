from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def architecture_gate_from_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    if isinstance(metrics, list):
        metrics_by_label = {str(row.get("label")): row for row in metrics}
    else:
        metrics_by_label = dict(metrics)
    combined = metrics_by_label.get("COMBINED", {})
    qqq = metrics_by_label.get("QQQ", {})
    spy = metrics_by_label.get("SPY", {})

    val_auc = _safe_float(payload.get("model_metrics", {}).get("val_exit_auc"))
    holdout_auc = _safe_float(payload.get("model_metrics", {}).get("holdout_exit_auc"))
    value_r2 = _safe_float(payload.get("model_metrics", {}).get("val_value_r2"))
    policy_exit_share = _safe_float(payload.get("diagnostics", {}).get("combined_policy_exit_share"))

    combined_sharpe = _safe_float(combined.get("sharpe_trade_annualized"))
    combined_pf = _safe_float(combined.get("profit_factor"))
    combined_mret = _safe_float(combined.get("mean_return_per_trade_frac"))
    sym_ret_gap = abs(_safe_float(qqq.get("mean_return_per_trade_frac"), 0.0) - _safe_float(spy.get("mean_return_per_trade_frac"), 0.0))

    thresholds = {
        "min_holdout_exit_auc": 0.70,
        "min_combined_sharpe": 0.70,
        "min_combined_profit_factor": 1.05,
        "min_combined_mean_return_per_trade": 2.0e-5,
        "min_policy_exit_share": 0.20,
        "max_symbol_mean_return_gap": 1.2e-4,
        "min_value_r2_for_value_driven_policy": 0.03,
    }
    checks = {
        "holdout_auc_ok": holdout_auc >= thresholds["min_holdout_exit_auc"],
        "combined_sharpe_ok": combined_sharpe >= thresholds["min_combined_sharpe"],
        "combined_pf_ok": combined_pf >= thresholds["min_combined_profit_factor"],
        "combined_mret_ok": combined_mret >= thresholds["min_combined_mean_return_per_trade"],
        "policy_exit_share_ok": policy_exit_share >= thresholds["min_policy_exit_share"],
        "symbol_gap_ok": sym_ret_gap <= thresholds["max_symbol_mean_return_gap"],
        "value_signal_ok": value_r2 >= thresholds["min_value_r2_for_value_driven_policy"],
    }
    score = sum(1 for v in checks.values() if bool(v))
    # Escalate architecture only when most settings/policy checks fail after alignment work.
    escalate = score <= 3
    reasons = [k for k, v in checks.items() if not bool(v)]

    return {
        "escalate_architecture": bool(escalate),
        "passed_checks": int(score),
        "total_checks": int(len(checks)),
        "failed_reasons": reasons,
        "checks": checks,
        "thresholds": thresholds,
        "snapshot": {
            "val_exit_auc": val_auc,
            "holdout_exit_auc": holdout_auc,
            "val_value_r2": value_r2,
            "combined_sharpe": combined_sharpe,
            "combined_profit_factor": combined_pf,
            "combined_mean_return_per_trade": combined_mret,
            "combined_policy_exit_share": policy_exit_share,
            "symbol_mean_return_gap": sym_ret_gap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Layer3 architecture escalation gate from metrics JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to run summary JSON (single run or matrix-best row).")
    parser.add_argument("--output", type=str, default="", help="Optional output JSON path for gate decision.")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(f"Input JSON not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    gate = architecture_gate_from_metrics(payload)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

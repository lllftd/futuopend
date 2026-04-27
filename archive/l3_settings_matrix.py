from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive.l3_architecture_gate import architecture_gate_from_metrics


RESULTS_ROOT = REPO_ROOT / "results" / "l3_matrix"
# Unified stack: L2+L3 are tee'd to one file; legacy layer3.log may be absent.
L3_LOG_PATH = REPO_ROOT / "logs" / "l2l3_unified.log"


MATRIX_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "vol_balanced_baseline",
        "env": {
            "L3_EXIT_EARLY_PATIENCE_STRENGTH_OVERRIDE": "0.10",
            "L3_EXIT_LATE_PRESSURE_STRENGTH_OVERRIDE": "0.18",
            "L3_EXIT_HYST_ENTER_OVERRIDE": "0.55",
            "L3_EXIT_HYST_LEAVE_OVERRIDE": "0.35",
        },
    },
    {
        "name": "longvol_selective",
        "env": {
            "L3_EXIT_EARLY_PATIENCE_STRENGTH_OVERRIDE": "0.18",
            "L3_EXIT_LATE_PRESSURE_STRENGTH_OVERRIDE": "0.18",
            "L3_EXIT_HYST_ENTER_OVERRIDE": "0.56",
            "L3_EXIT_HYST_LEAVE_OVERRIDE": "0.36",
        },
    },
    {
        "name": "shortvol_selective",
        "env": {
            "L3_EXIT_EARLY_PATIENCE_STRENGTH_OVERRIDE": "0.10",
            "L3_EXIT_LATE_PRESSURE_STRENGTH_OVERRIDE": "0.30",
            "L3_EXIT_HYST_ENTER_OVERRIDE": "0.55",
            "L3_EXIT_HYST_LEAVE_OVERRIDE": "0.35",
        },
    },
    {
        "name": "tight_short_stop",
        "env": {
            "L3_EXIT_EARLY_PATIENCE_STRENGTH_OVERRIDE": "0.10",
            "L3_EXIT_LATE_PRESSURE_STRENGTH_OVERRIDE": "0.20",
            "L3_EXIT_HYST_ENTER_OVERRIDE": "0.58",
            "L3_EXIT_HYST_LEAVE_OVERRIDE": "0.30",
        },
    },
]


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _parse_layer3_log_metrics(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    m_val = re.search(r"\[L3\] val — exit \(extended\)\s*\n\s*AUC=([0-9.\-]+)", txt)
    if m_val:
        out["val_exit_auc"] = float(m_val.group(1))
    m_holdout = re.search(r"\[L3\] holdout — exit AUC=([0-9.\-]+)", txt)
    if m_holdout:
        out["holdout_exit_auc"] = float(m_holdout.group(1))
    m_val_value = re.search(r"\[L3\] val — value \(extended\)\s*\n\s*MAE=.*?R2=([0-9.\-]+)", txt)
    if m_val_value:
        out["val_value_r2"] = float(m_val_value.group(1))
    return out


def _load_oos_summary(run_dir: Path) -> dict[str, Any]:
    """OOS summary: single oos_summary.txt (JSON body) from oos_backtest; fall back to legacy oos_summary.json."""
    p_txt = run_dir / "oos_summary.txt"
    p_json = run_dir / "oos_summary.json"
    if p_txt.exists() and p_txt.stat().st_size > 0:
        return json.loads(p_txt.read_text(encoding="utf-8"))
    if p_json.exists() and p_json.stat().st_size > 0:
        return json.loads(p_json.read_text(encoding="utf-8"))
    return {}


def _combined_metric_map(oos_summary: dict[str, Any]) -> dict[str, Any]:
    rows = oos_summary.get("metrics") or []
    by_label = {str(r.get("label")): r for r in rows}
    exit_rows = oos_summary.get("exit_path_diagnostics") or []
    by_exit = {str(r.get("label")): r for r in exit_rows}
    combined = dict(by_label.get("COMBINED", {}))
    combined["combined_policy_exit_share"] = float(by_exit.get("COMBINED", {}).get("policy_exit_share", float("nan")))
    return {
        "QQQ": by_label.get("QQQ", {}),
        "SPY": by_label.get("SPY", {}),
        "COMBINED": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layer3 settings matrix and summarize model+economic metrics.")
    parser.add_argument("--execute", action="store_true", help="Actually execute train/backtest commands.")
    parser.add_argument("--retrain-layer3", action="store_true", help="Retrain Layer3 before each run (slower).")
    parser.add_argument("--oos-start", default="2023-07-01", help="Backtest start for matrix runs.")
    parser.add_argument("--oos-end", default="2025-01-01", help="Backtest end for matrix runs.")
    parser.add_argument("--max-runs", type=int, default=4, help="Cap number of matrix configs to run.")
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    n_runs = max(1, min(int(args.max_runs), len(MATRIX_CONFIGS)))

    for cfg in MATRIX_CONFIGS[:n_runs]:
        name = str(cfg["name"])
        run_dir = RESULTS_ROOT / name
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
        env["OOS_START"] = str(args.oos_start)
        env["OOS_END"] = str(args.oos_end)
        env["OOS_RESULTS_DIR"] = str(run_dir)
        env["OOS_DISABLE_TQDM"] = "1"
        env["DISABLE_TQDM"] = "1"

        print(f"\n[matrix] {name}")
        if args.execute:
            if args.retrain_layer3:
                _run(["bash", "scripts/training/run_train.sh", "layer3"], env=env, cwd=REPO_ROOT)
            _run([sys.executable, "run_in_sample_backtest.py"], env=env, cwd=REPO_ROOT)
        else:
            print("  dry-run commands:")
            if args.retrain_layer3:
                print("   ", "bash scripts/training/run_train.sh layer3")
            print("   ", f"OOS_START={args.oos_start} OOS_END={args.oos_end} OOS_RESULTS_DIR={run_dir} python run_in_sample_backtest.py")

        oos_summary = _load_oos_summary(run_dir)
        model_metrics = _parse_layer3_log_metrics(L3_LOG_PATH)
        metric_map = _combined_metric_map(oos_summary)
        combined = metric_map["COMBINED"]
        row = {
            "run_name": name,
            "executed": bool(args.execute),
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "total_return_pct": float(combined.get("total_return_pct", float("nan"))),
            "max_drawdown_pct": float(combined.get("max_drawdown_pct", float("nan"))),
            "profit_factor": float(combined.get("profit_factor", float("nan"))),
            "sharpe_trade_annualized": float(combined.get("sharpe_trade_annualized", float("nan"))),
            "mean_return_per_trade_frac": float(combined.get("mean_return_per_trade_frac", float("nan"))),
            "avg_holding_bars": float(combined.get("avg_holding_bars", float("nan"))),
            "policy_exit_share": float(combined.get("combined_policy_exit_share", float("nan"))),
            "val_exit_auc": float(model_metrics.get("val_exit_auc", float("nan"))),
            "holdout_exit_auc": float(model_metrics.get("holdout_exit_auc", float("nan"))),
            "val_value_r2": float(model_metrics.get("val_value_r2", float("nan"))),
            "QQQ_mean_return_per_trade": float(metric_map["QQQ"].get("mean_return_per_trade_frac", float("nan"))),
            "SPY_mean_return_per_trade": float(metric_map["SPY"].get("mean_return_per_trade_frac", float("nan"))),
            "settings": cfg.get("env", {}),
        }
        rows.append(row)

    summary_json = RESULTS_ROOT / "matrix_summary.json"
    summary_csv = RESULTS_ROOT / "matrix_summary.csv"
    payload = {"runs": rows}
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "run_name",
                "executed",
                "oos_start",
                "oos_end",
                "total_return_pct",
                "max_drawdown_pct",
                "profit_factor",
                "sharpe_trade_annualized",
                "mean_return_per_trade_frac",
                "avg_holding_bars",
                "policy_exit_share",
                "val_exit_auc",
                "holdout_exit_auc",
                "val_value_r2",
                "QQQ_mean_return_per_trade",
                "SPY_mean_return_per_trade",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})

    best = None
    if rows:
        # Simple economic-first score.
        best = max(
            rows,
            key=lambda r: (
                float(r.get("sharpe_trade_annualized", float("-inf"))),
                float(r.get("profit_factor", float("-inf"))),
                float(r.get("total_return_pct", float("-inf"))),
            ),
        )
        best_payload = {
            "metrics": [
                {"label": "QQQ", "mean_return_per_trade_frac": best.get("QQQ_mean_return_per_trade", float("nan"))},
                {"label": "SPY", "mean_return_per_trade_frac": best.get("SPY_mean_return_per_trade", float("nan"))},
                {
                    "label": "COMBINED",
                    "sharpe_trade_annualized": best.get("sharpe_trade_annualized", float("nan")),
                    "profit_factor": best.get("profit_factor", float("nan")),
                    "mean_return_per_trade_frac": best.get("mean_return_per_trade_frac", float("nan")),
                },
            ],
            "model_metrics": {
                "val_exit_auc": best.get("val_exit_auc", float("nan")),
                "holdout_exit_auc": best.get("holdout_exit_auc", float("nan")),
                "val_value_r2": best.get("val_value_r2", float("nan")),
            },
            "diagnostics": {"combined_policy_exit_share": best.get("policy_exit_share", float("nan"))},
        }
        gate = architecture_gate_from_metrics(best_payload)
    else:
        gate = {"escalate_architecture": False, "reason": "no_rows"}
    gate_path = RESULTS_ROOT / "architecture_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[matrix] summary -> {summary_json}")
    print(f"[matrix] table   -> {summary_csv}")
    print(f"[matrix] gate    -> {gate_path}")
    if best is not None:
        print(
            f"[matrix] best={best['run_name']}  sharpe={best.get('sharpe_trade_annualized', float('nan')):.3f}  "
            f"pf={best.get('profit_factor', float('nan')):.3f}  ret={best.get('total_return_pct', float('nan')):.2f}%",
        )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

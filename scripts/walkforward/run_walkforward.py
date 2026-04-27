#!/usr/bin/env python3
"""
Walk-forward OOS: rolling test windows on **fixed** stack artifacts (default).

## Answers to design questions (this repo, as of ``core/training/common/constants.py``)

1. **Effective calendar / data range**
   - Stack split: ``TRAIN_END=2021-01-01``, ``CAL_END=2024-07-01``, ``TEST_END=2025-01-01``.
   - Default full OOS uses ``OOS_START=TEST_END`` and ``OOS_END`` far forward (``backtests/oos_backtest``), so
     holdout bars are ``time_key >= TEST_END`` until you narrow ``OOS_*``.
   - Your bar cache must actually cover each fold’s ``[test_start, test_end)``; otherwise that fold will have
     few or zero trades. Edit ``DEFAULT_FOLDS`` or pass ``--folds-json`` to match your data.
   - **Leakage (fixed stack):** with ``TRAIN_END / CAL_END / TEST_END`` from ``constants``, any fold whose
     ``test_end`` is **before** ``TEST_END`` (2025-01-01) still uses bars from **calibration or test-train** windows
     that the **models have already seen** at fit/cal time — not a clean *post-artifact* OOS. Default folds
     in this file are only **``time >= TEST_END``** (no CAL overlap). For CAL-era stress tests, add folds explicitly
     and label them in your own notes.

2. **Retrain L1/L2/L3 every fold?**
   - **Not in this script by default.** Training reads fixed dates from ``constants``; there is no supported
     one-shot env to set IS end per fold. **True walk-forward retrain** means either editing ``constants`` +
     running ``backtests.train_pipeline`` per fold, or extending the pipeline (out of scope here).
   - This runner implements **the fast, common baseline**: *same* ``lgbm_models/*`` artifacts, **only** the
     backtest window changes. That still answers “is the *signal* stable across eras?” (model stays fixed).

3. **Do train scripts take custom IS ranges?**
   - Not as env flags on ``train_pipeline``; dates live in ``core/training/common/constants.py`` and imports.
   - Use this script for **OOS window sweeps**; use manual constant edits (or a future trainer flag) for
     per-fold retrain.

## Usage (from repo root)

  PYTHONPATH=. python3 scripts/walkforward/run_walkforward.py
  PYTHONPATH=. python3 scripts/walkforward/run_walkforward.py --dry-run
  PYTHONPATH=. python3 scripts/walkforward/run_walkforward.py --out results/walkforward_myrun
  # Same 4 post-``TEST_END`` folds, but only L1a R0/R1/R3 (block R2, R4 at entry; same as full-sample r0r1r3):
  PYTHONPATH=. python3 scripts/walkforward/run_walkforward.py --out results/walkforward_true_oos_4f_r0r1r3 --oos-allow-l1a-regimes 0,1,3
  # Or explicit block:
  PYTHONPATH=. python3 scripts/walkforward/run_walkforward.py --oos-block-l1a-regimes 2,4

Outputs per fold under ``<out>/fold_01 .. fold_N/`` (``oos_summary.txt``, ``trades_*.csv``, charts) and
``<out>/walkforward_report.json`` + ``walkforward_report.txt``. Stitches ``trades_ALL.csv`` with ``walkforward_fold`` into ``<out>/stitched_trades_ALL.csv``.
After a successful stitch, ``scripts/diagnostics/walkforward_stitched_diagnostics.py`` writes ``stitched_cumulative_by_exit.png``,
``fold01_daily_pnl.png``, and ``walkforward_stitched_diagnostics.txt`` (unless ``--no-stitched-diagnostics``).
Unless ``--no-regime-charts``, each ``fold_NN/trades_ALL.csv`` is split by **5-day** windows into
``fold_NN/regime_charts/{sym}_week_....png`` (**not** for the stitched file).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Default folds: true post-``TEST_END`` OOS only (see ``core/training.constants.TEST_END`` = 2025-01-01). --- #
# Windows are half-open [test_start, test_end) in ``oos_backtest``; use the **first calendar day of the next month**
# as test_end to include all bars in the last month (pd.Timestamp("2025-04-30") would exclude most of 4/30).
# Last end aligns with QQQ/SPY data through 2026-04-16 (update when you refresh CSVs).
DEFAULT_FOLDS: tuple[dict[str, str], ...] = (
    {
        "is_start": "",
        "is_end": "",
        "test_start": "2025-01-01",
        "test_end": "2025-05-01",
    },
    {
        "is_start": "",
        "is_end": "",
        "test_start": "2025-05-01",
        "test_end": "2025-09-01",
    },
    {
        "is_start": "",
        "is_end": "",
        "test_start": "2025-09-01",
        "test_end": "2026-01-01",
    },
    {
        "is_start": "",
        "is_end": "",
        "test_start": "2026-01-01",
        "test_end": "2026-04-17",
    },
)


@dataclass
class FoldSpec:
    test_start: str
    test_end: str
    is_start: str = ""
    is_end: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_folds(path: Path | None) -> tuple[FoldSpec, ...]:
    if path is None:
        return tuple(
            FoldSpec(
                test_start=f["test_start"],
                test_end=f["test_end"],
                is_start=f.get("is_start", ""),
                is_end=f.get("is_end", ""),
            )
            for f in DEFAULT_FOLDS
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("folds JSON must be a list of objects with test_start, test_end")
    out: list[FoldSpec] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "test_start" not in item or "test_end" not in item:
            raise ValueError(f"folds[{i}] needs test_start and test_end")
        out.append(
            FoldSpec(
                test_start=str(item["test_start"]),
                test_end=str(item["test_end"]),
                is_start=str(item.get("is_start", "")),
                is_end=str(item.get("is_end", "")),
            )
        )
    return tuple(out)


def _parse_combined_metrics(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    try:
        raw = summary_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return None
    for row in payload.get("metrics") or []:
        if row.get("label") == "COMBINED":
            return dict(row)
    return None


def _run_one_fold(
    *,
    repo: Path,
    fold_idx: int,
    fold: FoldSpec,
    out_dir: Path,
    dry_run: bool,
    oos_allow_l1a_regimes: str | None = None,
    oos_block_l1a_regimes: str | None = None,
) -> int:
    tag = f"fold_{fold_idx:02d}"
    fold_dir = out_dir / tag
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["OOS_START"] = fold.test_start
    env["OOS_END"] = fold.test_end
    env["OOS_RESULTS_DIR"] = str(fold_dir)
    if oos_allow_l1a_regimes is not None:
        env["OOS_ALLOW_ENTRY_L1A_REGIMES"] = oos_allow_l1a_regimes.strip()
        env.pop("OOS_BLOCK_ENTRY_L1A_REGIMES", None)
    elif oos_block_l1a_regimes is not None:
        env["OOS_BLOCK_ENTRY_L1A_REGIMES"] = oos_block_l1a_regimes.strip()
        env.pop("OOS_ALLOW_ENTRY_L1A_REGIMES", None)
    cmd = [sys.executable, "-u", str(repo / "backtests" / "oos_backtest.py")]
    _rf = ""
    if oos_allow_l1a_regimes is not None:
        _rf = f"  OOS_ALLOW_ENTRY_L1A_REGIMES={oos_allow_l1a_regimes.strip()!r}"
    elif oos_block_l1a_regimes is not None:
        _rf = f"  OOS_BLOCK_ENTRY_L1A_REGIMES={oos_block_l1a_regimes.strip()!r}"
    print(f"\n=== {tag}  OOS_START={fold.test_start}  OOS_END={fold.test_end}{_rf}  -> {fold_dir}", flush=True)
    if dry_run:
        print(f"    (dry-run) would run: OOS_START=... OOS_END=... OOS_RESULTS_DIR=... {cmd}", flush=True)
        return 0
    fold_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=str(repo), env=env)
    return int(r.returncode)


def _stitch_trades(fold_dirs: list[Path], stitched_path: Path) -> int:
    import numpy as np
    import pandas as pd

    parts: list[pd.DataFrame] = []
    for i, d in enumerate(fold_dirs, start=1):
        p = d / "trades_ALL.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        df.insert(0, "walkforward_fold", int(i))
        parts.append(df)
    if not parts:
        return 0
    all_df = pd.concat(parts, ignore_index=True)
    if "exit_time" in all_df.columns:
        all_df = all_df.sort_values("exit_time", kind="mergesort").reset_index(drop=True)
    all_df.to_csv(stitched_path, index=False)
    r = all_df["return"].to_numpy(dtype=np.float64) if "return" in all_df.columns else np.array([])
    cum = float(np.cumsum(r)[-1]) if r.size else 0.0
    mdd = 0.0
    if r.size:
        c = np.cumsum(r)
        peak = np.maximum.accumulate(c)
        dd = c - peak
        mdd = float(np.min(dd))
    print(f"\n[stitch] {len(all_df)} rows -> {stitched_path}  sum_ret={cum:.6f}  max_dd_frac={mdd:.6f}", flush=True)
    return len(all_df)


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward OOS (fixed models, rolling OOS window by default).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: results/walkforward_<timestamp>)",
    )
    ap.add_argument("--folds-json", type=Path, default=None, help="JSON list of {test_start, test_end, is_* optional}")
    ap.add_argument("--dry-run", action="store_true", help="Print folds and commands only")
    ap.add_argument("--no-stitch", action="store_true", help="Do not write stitched_trades_ALL.csv")
    ap.add_argument(
        "--no-regime-charts",
        action="store_true",
        help="Skip K-line + regime shading PNGs (see scripts/walkforward/walkforward_regime_chart.py)",
    )
    ap.add_argument(
        "--no-stitched-diagnostics",
        action="store_true",
        help="Skip stitched_cumulative_by_exit.png and fold01 daily PnL from walkforward_stitched_diagnostics.py",
    )
    g_reg = ap.add_mutually_exclusive_group()
    g_reg.add_argument(
        "--oos-allow-l1a-regimes",
        type=str,
        default=None,
        metavar="CSV",
        help="Per-fold OOS: only these L1a argmax ids may enter, e.g. 0,1,3 (sets OOS_ALLOW_ENTRY_L1A_REGIMES; overrides shell).",
    )
    g_reg.add_argument(
        "--oos-block-l1a-regimes",
        type=str,
        default=None,
        metavar="CSV",
        help="Per-fold OOS: block these L1a ids at entry, e.g. 2,4 (sets OOS_BLOCK_ENTRY_L1A_REGIMES; overrides shell).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    folds = _load_folds(args.folds_json)
    out = args.out
    if out is None:
        out = repo / "results" / f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "mode": "oos_only_fixed_artifacts",
        "repo": str(repo),
        "output_root": str(out),
        "folds": [asdict(f) for f in folds],
    }
    if args.oos_allow_l1a_regimes:
        meta["oos_regime_entry"] = {
            "OOS_ALLOW_ENTRY_L1A_REGIMES": str(args.oos_allow_l1a_regimes).strip(),
        }
    elif args.oos_block_l1a_regimes:
        meta["oos_regime_entry"] = {
            "OOS_BLOCK_ENTRY_L1A_REGIMES": str(args.oos_block_l1a_regimes).strip(),
        }
    (out / "walkforward_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Walk-forward: {len(folds)} fold(s) -> {out}", flush=True)

    codes: list[int] = []
    fold_out_dirs: list[Path] = []
    for i, fold in enumerate(folds, start=1):
        c = _run_one_fold(
            repo=repo,
            fold_idx=i,
            fold=fold,
            out_dir=out,
            dry_run=bool(args.dry_run),
            oos_allow_l1a_regimes=str(args.oos_allow_l1a_regimes).strip() if args.oos_allow_l1a_regimes else None,
            oos_block_l1a_regimes=str(args.oos_block_l1a_regimes).strip() if args.oos_block_l1a_regimes else None,
        )
        codes.append(c)
        fold_out_dirs.append(out / f"fold_{i:02d}")

    report: dict[str, Any] = {
        "folds": [],
        "exit_codes": codes,
    }
    for i, (fold, ddir) in enumerate(zip(folds, fold_out_dirs), start=1):
        sm = ddir / "oos_summary.txt"
        comb = _parse_combined_metrics(sm)
        report["folds"].append(
            {
                "fold": i,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "is_start": fold.is_start,
                "is_end": fold.is_end,
                "oos_summary": str(sm),
                "combined": comb,
            }
        )
        if comb:
            print(
                f"  [fold {i:02d}]  n={comb.get('n_trades')}  "
                f"Sharpe~{comb.get('sharpe_trade_annualized', float('nan')):.3f}  "
                f"PF={comb.get('profit_factor', float('nan')):.3f}  "
                f"max_dd%={comb.get('max_drawdown_pct', float('nan')):.2f}  "
                f"win%={100.0 * float(comb.get('win_rate', 0.0)):.2f}",
                flush=True,
            )

    rep_json = out / "walkforward_report.json"
    rep_txt = out / "walkforward_report.txt"
    rep_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "Walk-forward OOS (fixed models)",
        f"output: {out}",
        "",
    ]
    for f in report["folds"]:
        c = f.get("combined") or {}
        lines.append(
            f"fold {f.get('fold')}  {f.get('test_start')} .. {f.get('test_end')}  |  "
            f"n_trades={c.get('n_trades')}  sharpe~={c.get('sharpe_trade_annualized')}  "
            f"max_dd%={c.get('max_drawdown_pct')}  pf={c.get('profit_factor')}"
        )
    rep_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {rep_json}  {rep_txt}", flush=True)

    n_stitched = 0
    stitched_path = out / "stitched_trades_ALL.csv"
    if not args.no_stitch and not args.dry_run:
        n_stitched = _stitch_trades(
            fold_out_dirs,
            stitched_path,
        )

    if (
        not args.no_stitched_diagnostics
        and not args.dry_run
        and n_stitched > 0
        and stitched_path.is_file()
    ):
        rc = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "diagnostics" / "walkforward_stitched_diagnostics.py"),
                "--stitched",
                str(stitched_path),
                "--out-dir",
                str(out),
            ],
            cwd=str(repo),
            env={**os.environ, "PYTHONPATH": str(repo)},
        )
        if rc.returncode != 0:
            print(f"\n[walkforward] walkforward_stitched_diagnostics exit {rc.returncode}", flush=True)

    if not args.no_regime_charts and not args.dry_run:
        for i, ddir in enumerate(fold_out_dirs):
            if i >= len(codes) or int(codes[i]) != 0:
                continue
            tp = ddir / "trades_ALL.csv"
            if not tp.is_file():
                continue
            fol = folds[i]
            suffix = f"fold_{i + 1:02d} [{fol.test_start}, {fol.test_end})"
            rc = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "walkforward" / "walkforward_regime_chart.py"),
                    "--trades",
                    str(tp),
                    "--out-dir",
                    str(ddir),
                    "--title-suffix",
                    suffix,
                ],
                cwd=str(repo),
                env={**os.environ, "PYTHONPATH": str(repo)},
            )
            if rc.returncode != 0:
                print(f"\n[walkforward] walkforward_regime_chart fold {i + 1} exit {rc.returncode}", flush=True)

    bad = [i for i, c in enumerate(codes, start=1) if c != 0]
    if bad:
        print(f"\n[walkforward] non-zero exit codes for folds: {bad}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

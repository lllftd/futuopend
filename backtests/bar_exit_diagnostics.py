"""
Post-entry mark-to-market diagnostics: track up to N bars after fill (1-min closes),
including continuation after the backtest closes the trade. Independent of L3 max_hold.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np


def default_bar_exit_track_bars() -> int:
    raw = (os.environ.get("OOS_BAR_EXIT_TRACK_BARS") or "").strip()
    if raw:
        return max(1, int(raw))
    return 30


class BarExitDiagnostics:
    def __init__(self, max_track_bars: int | None = None):
        self.max_track = int(max_track_bars) if max_track_bars is not None else default_bar_exit_track_bars()
        self.trades: list[dict[str, Any]] = []

    def record_trade(
        self,
        trade_id: int,
        entry_price: float,
        side: int = 1,
        entry_regime: int = -1,
        actual_exit_bar: int = -1,
        *,
        entry_bar_idx: int | None = None,
        all_close_prices: np.ndarray | Sequence[float] | None = None,
        bar_prices: Sequence[float] | None = None,
    ) -> None:
        """
        bar 1 = first 1-min close at/after entry fill row (index entry_bar_idx+1 in OOS).

        Provide either:
          - bar_prices: post-entry closes (length <= max_track), or
          - entry_bar_idx + all_close_prices: slice [entry_bar_idx+1 : entry_bar_idx+1+max_track).
        """
        ep = float(entry_price)
        if ep <= 0.0 or not np.isfinite(ep):
            return

        if bar_prices is not None:
            raw = [float(p) for p in bar_prices[: self.max_track]]
        elif entry_bar_idx is not None and all_close_prices is not None:
            ac = np.asarray(all_close_prices, dtype=np.float64)
            start = int(entry_bar_idx) + 1
            end = min(start + self.max_track, int(ac.shape[0]))
            if start >= end:
                return
            raw = [float(x) for x in ac[start:end]]
        else:
            return

        if not raw:
            return

        si = int(side)
        returns = [si * (p - ep) / ep for p in raw]
        arr = np.asarray(returns, dtype=np.float64)
        n_bars = int(arr.shape[0])
        best_i = int(np.argmax(arr))
        worst_i = int(np.argmin(arr))
        peak_ret = float(np.max(arr))
        ab = int(actual_exit_bar)
        idx = min(max(ab, 1), n_bars) - 1
        actual_ret = float(arr[idx])
        regret = float(peak_ret - actual_ret)

        self.trades.append(
            {
                "trade_id": int(trade_id),
                "returns": returns,
                "n_bars": n_bars,
                "best_bar": best_i + 1,
                "worst_bar": worst_i + 1,
                "peak_ret": peak_ret,
                "actual_exit_bar": ab,
                "actual_ret": actual_ret,
                "regret": regret,
                "entry_regime": int(entry_regime),
            }
        )

    def report(self, label: str = "") -> dict[str, Any]:
        if not self.trades:
            return {}

        n = len(self.trades)
        best_bars = np.array([t["best_bar"] for t in self.trades], dtype=np.int64)
        peak_rets = np.array([t["peak_ret"] for t in self.trades], dtype=np.float64)
        actual_rets = np.array([t["actual_ret"] for t in self.trades], dtype=np.float64)
        regrets = np.array([t["regret"] for t in self.trades], dtype=np.float64)
        regimes = np.array([t["entry_regime"] for t in self.trades], dtype=np.int64)
        n_bars_arr = np.array([t["n_bars"] for t in self.trades], dtype=np.int64)

        max_bar = int(min(self.max_track, int(np.max(n_bars_arr)))) if n else 0
        sep = "═" * 82
        print(f"\n{sep}")
        print(f"  Post-Exit Continuation Analysis ({label}, n={n})")
        print(f"  Tracking up to {self.max_track} bars post-entry (diagnostic only; L3 max_hold may be smaller)")
        print(f"{sep}")

        print("\n  ┌─ KEY QUESTION: Does the trend continue after bar 2? ─┐")

        report_bars = [1, 2, 3, 5, 10, 15, 20, 30]
        print(
            f"\n  {'bar':>4}  {'mean_ret':>10}  {'med_ret':>10}  "
            f"{'win_rate':>9}  {'is_best%':>9}  {'n_valid':>7}"
        )
        print(f"  {'─' * 60}")

        bar_table: list[dict[str, Any]] = []
        for b in report_bars:
            if b > max_bar:
                break
            rets_at_b: list[float] = []
            is_best_count = 0
            n_valid = 0
            for t in self.trades:
                if b <= int(t["n_bars"]):
                    r_b = float(t["returns"][b - 1])
                    rets_at_b.append(r_b)
                    n_valid += 1
                    if int(t["best_bar"]) == b:
                        is_best_count += 1

            if not rets_at_b:
                continue

            arr_b = np.asarray(rets_at_b, dtype=np.float64)
            mean_b = float(np.mean(arr_b))
            med_b = float(np.median(arr_b))
            win_b = float(np.mean(arr_b > 0))
            is_best_pct = float(is_best_count / max(len(rets_at_b), 1))

            marker = ""
            if b == 2:
                marker = " ← typical L3 deadline region (meta max_hold often 2)"
            elif b > 2 and mean_b > float(np.mean(actual_rets)):
                marker = " ← mean MTM > mean MTM at actual exit bar"

            print(
                f"  {b:>4}  {mean_b:>+10.4%}  {med_b:>+10.4%}  "
                f"{win_b:>9.1%}  {is_best_pct:>9.1%}  {len(rets_at_b):>7}{marker}"
            )
            bar_table.append(
                {
                    "bar": b,
                    "mean_ret": mean_b,
                    "median_ret": med_b,
                    "win_rate": win_b,
                    "is_best_pct": is_best_pct,
                    "n_valid": int(len(rets_at_b)),
                }
            )

        mtm_bar2: list[float] = []
        for t in self.trades:
            if int(t["n_bars"]) >= 2:
                mtm_bar2.append(float(t["returns"][1]))
        mean_mtm_bar2 = float(np.mean(mtm_bar2)) if mtm_bar2 else float("nan")

        print("\n  Oracle Analysis (MTM at bar closes; actual = close of each trade's exit bar, capped by track length):")
        print(f"    mean MTM at bar 2 close (all trades w/ data): {mean_mtm_bar2:>+.4%}")
        print(f"    mean MTM at actual exit bar:                 {float(np.mean(actual_rets)):>+.4%}")
        print(f"    oracle_exit@best (per-trade peak MTM):       {float(np.mean(peak_rets)):>+.4%}")
        print(f"    regret (peak - actual exit MTM):             {float(np.mean(regrets)):>+.4%}  "
              f"median={float(np.median(regrets)):>+.4%}  "
              f"p90={float(np.percentile(regrets, 90)):>+.4%}")

        print("\n  Best exit bar distribution (which minute is optimal?):")
        cdf_points = [1, 2, 3, 5, 10, 15, 20, 30]
        cdf_rows: list[dict[str, Any]] = []
        for bp in cdf_points:
            if bp > max_bar:
                break
            cdf = float(np.mean(best_bars <= bp))
            bar_str = f"  bar<={bp:>2}: {cdf:>6.1%} of trades"
            if bp == 2:
                bar_str += f"  ← only {cdf:.1%} have best exit within bar 2"
            print(f"    {bar_str}")
            cdf_rows.append({"bar_cap": bp, "cdf_best_bar_le": cdf})

        leftover = float(np.mean(best_bars > 2))
        print(f"\n    >>> {leftover:.1%} of trades have optimal exit AFTER bar 2 <<<")
        if leftover > 0.5:
            print("    >>> STRONG evidence: short max_hold may be truncating upside (diagnostic) <<<")

        print("\n  Regime breakdown (entry L1a regime):")
        print(f"  {'rgm':>3}  {'n':>5}  {'actual_exit':>11}  {'oracle_peak':>11}  "
              f"{'regret':>10}  {'med_best':>8}  {'best>2':>8}")
        print(f"  {'─' * 68}")
        regime_rows: list[dict[str, Any]] = []
        for r in sorted(set(regimes.tolist())):
            mask = regimes == r
            r_actual = actual_rets[mask]
            r_peak = peak_rets[mask]
            r_regret = regrets[mask]
            r_best = best_bars[mask]
            row = {
                "regime": int(r),
                "n": int(np.sum(mask)),
                "mean_actual_exit_mtm": float(np.mean(r_actual)),
                "mean_oracle_peak_mtm": float(np.mean(r_peak)),
                "mean_regret": float(np.mean(r_regret)),
                "median_best_bar": float(np.median(r_best)),
                "share_best_after_bar2": float(np.mean(r_best > 2)),
            }
            regime_rows.append(row)
            print(
                f"  {int(r):>3}  {int(np.sum(mask)):>5}  "
                f"{float(np.mean(r_actual)):>+11.4%}  "
                f"{float(np.mean(r_peak)):>+11.4%}  "
                f"{float(np.mean(r_regret)):>+10.4%}  "
                f"{float(np.median(r_best)):>8.1f}  "
                f"{float(np.mean(r_best > 2)):>8.1%}"
            )

        print(f"\n{sep}")

        mean_regret = float(np.mean(regrets))
        pct_best_after_2 = float(np.mean(best_bars > 2))
        rec_lines: list[str] = []
        print("\n  RECOMMENDATION:")
        if pct_best_after_2 > 0.6 and mean_regret > 0.001:
            rec_lines = [
                "max_hold=2 (or very short) may be severely limiting MTM at tracked horizon",
                f"{pct_best_after_2:.0%} of trades peak after bar 2 (within {self.max_track}-bar window)",
                "Consider raising l3_target_horizon_bars (and labels) and retraining L3; confirm with execution costs.",
            ]
            for ln in rec_lines:
                print(f"    {ln}")
        elif pct_best_after_2 > 0.4:
            rec_lines = [
                "Short max_hold may be moderately limiting; review bar 3–10 mean MTM vs bar 2.",
                "Consider intermediate horizon (e.g. 10–15 bars) if training pipeline allows.",
            ]
            for ln in rec_lines:
                print(f"    {ln}")
        else:
            rec_lines = [
                "Within this window, many optima fall by bar 2; short hold may be reasonable (diagnostic only).",
                "If live returns still disappoint, focus on entry (L1/L2) or costs.",
            ]
            for ln in rec_lines:
                print(f"    {ln}")
        print()

        return {
            "label": label,
            "n_trades": n,
            "max_track_bars": self.max_track,
            "max_bar_observed": max_bar,
            "bar_table": bar_table,
            "mean_mtm_bar2_close": _finite_or_none(mean_mtm_bar2),
            "mean_actual_exit_mtm": float(np.mean(actual_rets)),
            "mean_oracle_peak_mtm": float(np.mean(peak_rets)),
            "mean_regret": mean_regret,
            "median_regret": float(np.median(regrets)),
            "p90_regret": float(np.percentile(regrets, 90)),
            "share_best_exit_after_bar2": pct_best_after_2,
            "best_bar_cdf": cdf_rows,
            "regime_breakdown": regime_rows,
            "recommendation_lines": rec_lines,
        }


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None

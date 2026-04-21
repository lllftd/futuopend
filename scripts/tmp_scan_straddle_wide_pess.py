import os
import pickle
import numpy as np
import pandas as pd

from core.trainers.constants import MODEL_DIR, PREPARED_DATASET_CACHE_FILE, L2_META_FILE, L2_OUTPUT_CACHE_FILE, TEST_END
from core.trainers.stack_v2_common import load_output_cache


def load_oos_with_gate() -> pd.DataFrame:
    prep_path = os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE)
    with open(prep_path, "rb") as f:
        obj = pickle.load(f)
    df = obj["df"].copy()
    df["time_key"] = pd.to_datetime(df["time_key"])

    with open(os.path.join(MODEL_DIR, L2_META_FILE), "rb") as f:
        l2_meta = pickle.load(f)
    l2_out = load_output_cache(L2_OUTPUT_CACHE_FILE).copy()
    l2_out["time_key"] = pd.to_datetime(l2_out["time_key"])

    merged = df.merge(
        l2_out[["symbol", "time_key", "l2_decision_neutral"]],
        on=["symbol", "time_key"],
        how="inner",
    )
    merged = merged[merged["time_key"] >= pd.Timestamp(TEST_END)].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No OOS rows after TEST_END filter.")

    if "lbl_atr" in merged.columns:
        atr = pd.to_numeric(merged["lbl_atr"], errors="coerce")
    elif "atr_5m" in merged.columns:
        atr = pd.to_numeric(merged["atr_5m"], errors="coerce")
    elif "atr_1m" in merged.columns:
        atr = pd.to_numeric(merged["atr_1m"], errors="coerce")
    else:
        atr = (
            pd.to_numeric(merged["high"], errors="coerce")
            - pd.to_numeric(merged["low"], errors="coerce")
        ).rolling(14, min_periods=1).mean()

    merged["atr"] = np.clip(np.nan_to_num(atr.to_numpy(dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0), 1e-4, np.inf)
    for c in ("open", "high", "low", "close"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce").ffill().bfill()

    trade_thr = float((l2_meta.get("two_stage_policy") or l2_meta).get("trade_threshold", l2_meta.get("trade_threshold", 0.5)))
    merged["gate_pred"] = 1.0 - pd.to_numeric(merged["l2_decision_neutral"], errors="coerce").fillna(1.0)
    merged["gate_on"] = merged["gate_pred"] >= trade_thr
    return merged


def straddle_wide_pess(df: pd.DataFrame, gate_mask: np.ndarray, sl_atr: float, trail_act: float, trail_dist: float, max_hold: int = 60) -> pd.DataFrame:
    results = []
    trade_end_idx = -1
    n = len(df)

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)

    for i in np.flatnonzero(gate_mask):
        if i <= trade_end_idx or i + 1 >= n:
            continue

        entry = close[i]
        a = atr[i]
        long_sl = entry - sl_atr * a
        short_sl = entry + sl_atr * a
        long_peak = entry
        short_trough = entry
        long_alive = True
        short_alive = True
        long_pnl = None
        short_pnl = None
        hold_bars = 0
        ambig = 0
        exit_bar = min(i + max_hold, n - 1)

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            hold_bars = j - i
            bj_open = open_[j]
            bj_high = high[j]
            bj_low = low[j]

            long_sl_hit = long_alive and (bj_low <= long_sl)
            short_sl_hit = short_alive and (bj_high >= short_sl)
            if long_sl_hit and short_sl_hit and long_alive and short_alive:
                ambig = 1
                if bj_open <= entry:
                    long_pnl = (long_sl - entry) / a
                    long_alive = False
                else:
                    short_pnl = (entry - short_sl) / a
                    short_alive = False

            if long_alive and (bj_low <= long_sl):
                long_pnl = (long_sl - entry) / a
                long_alive = False
            if short_alive and (bj_high >= short_sl):
                short_pnl = (entry - short_sl) / a
                short_alive = False

            if long_alive:
                long_peak = max(long_peak, bj_high)
                if (long_peak - entry) / a >= trail_act:
                    long_sl = max(long_sl, long_peak - trail_dist * a)
                if bj_low <= long_sl:
                    long_pnl = (long_sl - entry) / a
                    long_alive = False

            if short_alive:
                short_trough = min(short_trough, bj_low)
                if (entry - short_trough) / a >= trail_act:
                    short_sl = min(short_sl, short_trough + trail_dist * a)
                if bj_high >= short_sl:
                    short_pnl = (entry - short_sl) / a
                    short_alive = False

            if (not long_alive) and (not short_alive):
                exit_bar = j
                break

        if long_pnl is None:
            long_pnl = (close[exit_bar] - entry) / a
        if short_pnl is None:
            short_pnl = (entry - close[exit_bar]) / a

        trade_end_idx = exit_bar
        total = 0.5 * (long_pnl + short_pnl)
        results.append(
            {
                "entry_idx": int(i),
                "exit_idx": int(exit_bar),
                "total_pnl": float(total),
                "long_pnl": float(long_pnl),
                "short_pnl": float(short_pnl),
                "hold_bars": int(hold_bars),
                "ambig": int(ambig),
            }
        )

    return pd.DataFrame(results)


def run_scan(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_symbol = [g.sort_values("time_key").reset_index(drop=True) for _, g in df_all.groupby("symbol", sort=True)]

    for sl in [0.5, 0.7, 1.0, 1.5]:
        for ta in [0.5, 0.8, 1.0, 1.5, 2.0]:
            for td in [0.2, 0.3, 0.5]:
                if td >= ta or ta < sl:
                    continue

                parts = []
                for g in by_symbol:
                    gate_mask = g["gate_on"].to_numpy(dtype=bool)
                    res = straddle_wide_pess(g, gate_mask, sl_atr=sl, trail_act=ta, trail_dist=td, max_hold=60)
                    if not res.empty:
                        parts.append(res)
                if not parts:
                    continue

                res_all = pd.concat(parts, ignore_index=True)
                n = len(res_all)
                if n < 200:
                    continue

                ev = float(res_all["total_pnl"].mean())
                wr = float((res_all["total_pnl"] > 0).mean())
                avg_h = float(res_all["hold_bars"].mean())
                amb = float(res_all["ambig"].mean())
                std = float(res_all["total_pnl"].std())
                ev_cost = ev - 0.04
                sharpe = float(ev / std * np.sqrt(252.0)) if std > 1e-12 else np.nan

                rows.append(
                    {
                        "sl": sl,
                        "ta": ta,
                        "td": td,
                        "n": n,
                        "ev": ev,
                        "wr": wr,
                        "std": std,
                        "sharpe": sharpe,
                        "avg_h": avg_h,
                        "ambig_pct": amb,
                        "ev_cost": ev_cost,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["ev_cost", "ev", "sharpe"], ascending=False).reset_index(drop=True)


def main():
    df = load_oos_with_gate()
    print("=== OOS Scope ===")
    print(f"rows={len(df):,} symbols={df['symbol'].nunique()} gate_on={int(df['gate_on'].sum()):,} ({float(df['gate_on'].mean()):.2%})")

    scan = run_scan(df)
    if scan.empty:
        print("No valid parameter combo found.")
        return

    print("\n=== Wide-Stop Pessimistic Scan Top 25 (sorted by EV-cost) ===")
    print(scan.head(25).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    out_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "l2_straddle_wide_pess_scan.csv")
    scan.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

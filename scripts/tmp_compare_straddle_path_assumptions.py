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
        atr = (pd.to_numeric(merged["high"], errors="coerce") - pd.to_numeric(merged["low"], errors="coerce")).rolling(
            14, min_periods=1
        ).mean()

    merged["atr"] = np.clip(np.nan_to_num(atr.to_numpy(dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0), 1e-4, np.inf)
    for c in ("open", "high", "low", "close"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce").ffill().bfill()

    trade_thr = float((l2_meta.get("two_stage_policy") or l2_meta).get("trade_threshold", l2_meta.get("trade_threshold", 0.5)))
    merged["gate_pred"] = 1.0 - pd.to_numeric(merged["l2_decision_neutral"], errors="coerce").fillna(1.0)
    merged["gate_on"] = merged["gate_pred"] >= trade_thr
    return merged


def straddle_no_overlap(df: pd.DataFrame, gate_mask: np.ndarray, sl_atr=0.3, trail_act=0.3, trail_dist=0.2, max_hold=30):
    results = []
    trade_end_idx = -1
    for i in np.flatnonzero(gate_mask):
        if i <= trade_end_idx or i + 1 >= len(df):
            continue
        price = float(df.iloc[i]["close"])
        atr = float(df.iloc[i]["atr"])
        long_entry = price
        long_sl = price - sl_atr * atr
        long_peak = price
        long_pnl = None
        short_entry = price
        short_sl = price + sl_atr * atr
        short_trough = price
        short_pnl = None
        exit_bar = i
        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            bar = df.iloc[j]
            if long_pnl is None:
                if bar["high"] > long_peak:
                    long_peak = float(bar["high"])
                if long_peak - long_entry >= trail_act * atr:
                    trailing_sl = long_peak - trail_dist * atr
                    long_sl = max(long_sl, trailing_sl)
                if bar["low"] <= long_sl:
                    long_pnl = (long_sl - long_entry) / atr
            if short_pnl is None:
                if bar["low"] < short_trough:
                    short_trough = float(bar["low"])
                if short_entry - short_trough >= trail_act * atr:
                    trailing_sl = short_trough + trail_dist * atr
                    short_sl = min(short_sl, trailing_sl)
                if bar["high"] >= short_sl:
                    short_pnl = (short_entry - short_sl) / atr
            if long_pnl is not None and short_pnl is not None:
                exit_bar = j
                break
        if long_pnl is None:
            k = min(i + max_hold, len(df) - 1)
            last = float(df.iloc[k]["close"])
            long_pnl = (last - long_entry) / atr
            exit_bar = k
        if short_pnl is None:
            k = min(i + max_hold, len(df) - 1)
            last = float(df.iloc[k]["close"])
            short_pnl = (short_entry - last) / atr
            exit_bar = k
        trade_end_idx = exit_bar
        results.append({"entry_idx": i, "exit_idx": exit_bar, "total_pnl_atr": (long_pnl + short_pnl) * 0.5})
    return pd.DataFrame(results)


def straddle_pessimistic(df: pd.DataFrame, gate_mask: np.ndarray, sl_atr=0.3, trail_act=0.3, trail_dist=0.2, max_hold=30):
    results = []
    trade_end_idx = -1
    for i in np.flatnonzero(gate_mask):
        if i <= trade_end_idx or i + 1 >= len(df):
            continue
        price = float(df.iloc[i]["close"])
        atr = float(df.iloc[i]["atr"])
        done = False
        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            bar = df.iloc[j]
            bar_range_atr = (float(bar["high"]) - float(bar["low"])) / atr
            if bar_range_atr >= 2.0 * sl_atr:
                if float(bar["open"]) >= price:
                    favorable_move = (float(bar["high"]) - price) / atr
                    winner_pnl = (favorable_move - trail_dist) if favorable_move >= trail_act else (float(bar["close"]) - price) / atr
                else:
                    favorable_move = (price - float(bar["low"])) / atr
                    winner_pnl = (favorable_move - trail_dist) if favorable_move >= trail_act else (price - float(bar["close"])) / atr
                loser_pnl = -sl_atr
                total_pnl = (winner_pnl + loser_pnl) * 0.5
                results.append({"entry_idx": i, "exit_idx": j, "total_pnl_atr": total_pnl})
                trade_end_idx = j
                done = True
                break
        if not done:
            k = min(i + max_hold, len(df) - 1)
            move = (float(df.iloc[k]["close"]) - price) / atr
            total_pnl = (abs(move) - sl_atr) * 0.5
            results.append({"entry_idx": i, "exit_idx": k, "total_pnl_atr": total_pnl})
            trade_end_idx = k
    return pd.DataFrame(results)


def straddle_bar2_only(df: pd.DataFrame, gate_mask: np.ndarray, sl_atr=0.3):
    results = []
    trade_end_idx = -1
    for i in np.flatnonzero(gate_mask):
        if i <= trade_end_idx or i + 1 >= len(df):
            continue
        price = float(df.iloc[i]["close"])
        atr = float(df.iloc[i]["atr"])
        nxt = df.iloc[i + 1]
        bar_range_atr = (float(nxt["high"]) - float(nxt["low"])) / atr
        move = (float(nxt["close"]) - price) / atr
        if abs(move) < 0.01:
            total_pnl = 0.0
        else:
            if bar_range_atr < sl_atr:
                total_pnl = 0.0
            else:
                winner_pnl = abs(move)
                loser_pnl = -sl_atr
                total_pnl = (winner_pnl + loser_pnl) * 0.5
        results.append({"entry_idx": i, "exit_idx": i + 1, "total_pnl_atr": total_pnl, "move_atr": move})
        trade_end_idx = i + 1
    return pd.DataFrame(results)


def report_df(name: str, res: pd.DataFrame) -> dict:
    if res.empty:
        return {"variant": name, "n": 0, "ev": np.nan, "wr": np.nan, "std": np.nan}
    return {
        "variant": name,
        "n": int(len(res)),
        "ev": float(res["total_pnl_atr"].mean()),
        "wr": float((res["total_pnl_atr"] > 0).mean()),
        "std": float(res["total_pnl_atr"].std()),
    }


def main():
    merged = load_oos_with_gate()
    print("=== OOS Scope ===")
    print(f"rows={len(merged):,} symbols={merged['symbol'].nunique()} gate_on={int(merged['gate_on'].sum()):,} ({float(merged['gate_on'].mean()):.2%})")

    summary_rows = []
    all_results = []
    for sym, g in merged.groupby("symbol", sort=True):
        g = g.sort_values("time_key").reset_index(drop=True)
        gate_mask = g["gate_on"].to_numpy(dtype=bool)
        res_orig = straddle_no_overlap(g, gate_mask, sl_atr=0.3, trail_act=0.3, trail_dist=0.2, max_hold=30)
        res_pess = straddle_pessimistic(g, gate_mask, sl_atr=0.3, trail_act=0.3, trail_dist=0.2, max_hold=30)
        res_cons = straddle_bar2_only(g, gate_mask, sl_atr=0.3)
        for nm, rs in (("orig", res_orig), ("pess", res_pess), ("cons", res_cons)):
            if not rs.empty:
                rs["symbol"] = sym
                rs["variant"] = nm
                all_results.append(rs)
        summary_rows.extend(
            [
                report_df(f"{sym}:orig", res_orig),
                report_df(f"{sym}:pess", res_pess),
                report_df(f"{sym}:cons", res_cons),
            ]
        )

    all_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if all_df.empty:
        raise RuntimeError("No results produced.")

    total_summary = []
    for nm in ("orig", "pess", "cons"):
        sub = all_df[all_df["variant"] == nm]
        total_summary.append(report_df(f"all:{nm}", sub))

    summary = pd.DataFrame(summary_rows + total_summary)
    print("\n=== Path Assumption Compare ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    out_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "l2_straddle_path_compare_summary.csv"), index=False)
    all_df.to_csv(os.path.join(out_dir, "l2_straddle_path_compare_trades.csv"), index=False)
    print(f"\nSaved reports to: {out_dir}")


if __name__ == "__main__":
    main()

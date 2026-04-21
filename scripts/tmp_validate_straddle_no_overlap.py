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


def straddle_no_overlap(
    df: pd.DataFrame,
    gate_mask: np.ndarray,
    sl_atr: float = 0.3,
    trail_act: float = 0.3,
    trail_dist: float = 0.2,
    max_hold: int = 30,
) -> pd.DataFrame:
    out: list[dict] = []
    trade_end_idx = -1

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["time_key"]).to_numpy()

    for i in np.flatnonzero(gate_mask):
        if i <= trade_end_idx:
            continue
        if i + 1 >= len(df):
            continue

        price = close[i]
        a = atr[i]

        long_entry = price
        long_sl = price - sl_atr * a
        long_peak = price
        long_pnl = None

        short_entry = price
        short_sl = price + sl_atr * a
        short_trough = price
        short_pnl = None

        exit_bar = i
        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            if long_pnl is None:
                if high[j] > long_peak:
                    long_peak = high[j]
                if long_peak - long_entry >= trail_act * a:
                    long_sl = max(long_sl, long_peak - trail_dist * a)
                if low[j] <= long_sl:
                    long_pnl = (long_sl - long_entry) / a

            if short_pnl is None:
                if low[j] < short_trough:
                    short_trough = low[j]
                if short_entry - short_trough >= trail_act * a:
                    short_sl = min(short_sl, short_trough + trail_dist * a)
                if high[j] >= short_sl:
                    short_pnl = (short_entry - short_sl) / a

            if long_pnl is not None and short_pnl is not None:
                exit_bar = j
                break

        if long_pnl is None:
            k = min(i + max_hold, len(df) - 1)
            long_pnl = (close[k] - long_entry) / a
            exit_bar = k
        if short_pnl is None:
            k = min(i + max_hold, len(df) - 1)
            short_pnl = (short_entry - close[k]) / a
            exit_bar = k

        trade_end_idx = exit_bar
        total_pnl = (long_pnl + short_pnl) * 0.5
        out.append(
            {
                "entry_idx": int(i),
                "exit_idx": int(exit_bar),
                "total_pnl_atr": float(total_pnl),
                "long_pnl": float(long_pnl),
                "short_pnl": float(short_pnl),
                "holding_bars": int(exit_bar - i),
                "entry_time": pd.Timestamp(ts[i]),
                "exit_time": pd.Timestamp(ts[exit_bar]),
            }
        )
    return pd.DataFrame(out)


def count_same_bar_ambiguity(df: pd.DataFrame, gate_mask: np.ndarray, sl_atr: float = 0.3, trail_act: float = 0.3) -> tuple[float, int, int]:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)
    ambiguous = 0
    total = 0
    n = len(df)

    for i in np.flatnonzero(gate_mask):
        if i + 1 >= n:
            continue
        j = i + 1
        range_atr = (high[j] - low[j]) / max(atr[i], 1e-9)
        if range_atr >= (sl_atr + trail_act):
            ambiguous += 1
        total += 1

    rate = (ambiguous / total) if total > 0 else np.nan
    return rate, ambiguous, total


def main() -> None:
    merged = load_oos_with_gate()
    print("=== OOS Scope ===")
    print(f"rows={len(merged):,} symbols={merged['symbol'].nunique()} gate_on={int(merged['gate_on'].sum()):,} ({float(merged['gate_on'].mean()):.2%})")

    all_trades: list[pd.DataFrame] = []
    for sym, g in merged.groupby("symbol", sort=True):
        g = g.sort_values("time_key").reset_index(drop=True)
        trades = straddle_no_overlap(g, g["gate_on"].to_numpy(dtype=bool), sl_atr=0.3, trail_act=0.3, trail_dist=0.2, max_hold=30)
        if not trades.empty:
            trades["symbol"] = sym
            gate_prob = g["gate_pred"].to_numpy(dtype=np.float64)
            trades["gate_prob"] = gate_prob[trades["entry_idx"].to_numpy(dtype=np.int64)]
            all_trades.append(trades)

    if not all_trades:
        raise RuntimeError("No trades generated in straddle_no_overlap.")
    res_clean = pd.concat(all_trades, ignore_index=True)

    print("\n=== 验证1 去重叠真实交易 ===")
    print(f"n_trades: {len(res_clean):,}")
    print(f"EV: {res_clean['total_pnl_atr'].mean():.6f} ATR")
    print(f"WR: {(res_clean['total_pnl_atr'] > 0).mean():.2%}")
    print(f"std: {res_clean['total_pnl_atr'].std():.6f}")
    print(f"avg_hold: {res_clean['holding_bars'].mean():.2f} bars")
    print(f"median_hold: {res_clean['holding_bars'].median():.0f} bars")

    res_clean["date"] = pd.to_datetime(res_clean["entry_time"]).dt.date
    daily = res_clean.groupby("date")["total_pnl_atr"].sum()
    all_dates = pd.date_range(pd.Timestamp(daily.index.min()), pd.Timestamp(daily.index.max()), freq="B")
    daily = daily.reindex(all_dates.date, fill_value=0.0)

    daily_mean = float(daily.mean())
    daily_std = float(daily.std())
    daily_sharpe = (daily_mean / daily_std * np.sqrt(252.0)) if daily_std > 1e-12 else np.nan
    dd = (daily.cumsum() - daily.cumsum().cummax()).min()

    print("\n=== 验证2 日级真实Sharpe ===")
    print(f"daily_sharpe: {daily_sharpe:.6f}")
    print(f"daily_mean: {daily_mean:.6f} ATR")
    print(f"daily_std: {daily_std:.6f} ATR")
    print(f"daily_WR: {(daily > 0).mean():.2%}")
    print(f"max_drawdown: {dd:.6f} ATR")
    print(f"total_PnL: {daily.sum():.6f} ATR over {len(all_dates)} business days")

    monthly = res_clean.groupby(pd.to_datetime(res_clean["entry_time"]).dt.to_period("M")).agg(
        n=("total_pnl_atr", "count"),
        ev=("total_pnl_atr", "mean"),
        total=("total_pnl_atr", "sum"),
        wr=("total_pnl_atr", lambda x: float((x > 0).mean())),
    )
    print("\n=== 月度分解 ===")
    print(monthly.to_string())

    cost_per_trade = 0.04
    res_clean["pnl_after_cost"] = res_clean["total_pnl_atr"] - cost_per_trade
    print("\n=== 验证3 扣成本 ===")
    print(f"EV_after_cost: {res_clean['pnl_after_cost'].mean():.6f} ATR")
    print(f"WR_after_cost: {(res_clean['pnl_after_cost'] > 0).mean():.2%}")

    print("\n=== 验证4 gate_prob分层 ===")
    qs = res_clean["gate_prob"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_numpy(dtype=np.float64)
    for k in range(4):
        lo = qs[k]
        hi = qs[k + 1]
        if k < 3:
            m = (res_clean["gate_prob"] >= lo) & (res_clean["gate_prob"] < hi)
        else:
            m = (res_clean["gate_prob"] >= lo) & (res_clean["gate_prob"] <= hi)
        sub = res_clean[m]
        print(
            f"Q{k}: n={len(sub):,}  gate_prob_range=[{lo:.6f}, {hi:.6f}]  "
            f"EV={sub['total_pnl_atr'].mean():.6f}  WR={(sub['total_pnl_atr'] > 0).mean():.2%}"
        )

    amb_total = 0
    amb_hit = 0
    for _, g in merged.groupby("symbol", sort=True):
        g = g.sort_values("time_key").reset_index(drop=True)
        rate, hit, total = count_same_bar_ambiguity(g, g["gate_on"].to_numpy(dtype=bool), sl_atr=0.3, trail_act=0.3)
        _ = rate
        amb_hit += hit
        amb_total += total
    amb_rate = (amb_hit / amb_total) if amb_total > 0 else np.nan
    print("\n=== 验证5 同bar双触发歧义率 ===")
    print(f"ambiguity_rate: {amb_rate:.2%} ({amb_hit}/{amb_total})")

    out_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(out_dir, exist_ok=True)
    res_clean.to_csv(os.path.join(out_dir, "l2_straddle_no_overlap_trades.csv"), index=False)
    daily.reset_index().rename(columns={"index": "date", 0: "daily_pnl_atr", "total_pnl_atr": "daily_pnl_atr"}).to_csv(
        os.path.join(out_dir, "l2_straddle_no_overlap_daily.csv"),
        index=False,
    )
    monthly.to_csv(os.path.join(out_dir, "l2_straddle_no_overlap_monthly.csv"))
    print(f"\nSaved reports to: {out_dir}")


if __name__ == "__main__":
    main()

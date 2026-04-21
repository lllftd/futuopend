import os
import pickle
import numpy as np
import pandas as pd

from core.trainers.constants import MODEL_DIR, PREPARED_DATASET_CACHE_FILE, L2_META_FILE, L2_OUTPUT_CACHE_FILE, TEST_END
from core.trainers.stack_v2_common import load_output_cache


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
    l2_out[["symbol", "time_key", "l2_decision_neutral", "l2_decision_long", "l2_decision_short"]],
    on=["symbol", "time_key"],
    how="inner",
)
merged = merged[merged["time_key"] >= pd.Timestamp(TEST_END)].reset_index(drop=True)
if merged.empty:
    raise RuntimeError("No rows after TEST_END filter; cannot run scan.")

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

print("=== Data Scope ===")
print(f"window: [{pd.Timestamp(TEST_END)}, {merged['time_key'].max()}]")
print(f"rows={len(merged):,} symbols={merged['symbol'].nunique()} trade_threshold={trade_thr:.4f}")
print(f"gate_on={int(merged['gate_on'].sum()):,} ({float(merged['gate_on'].mean()):.2%})")


def _future_true_dir(close_arr, i, horizon):
    j = min(i + horizon, len(close_arr) - 1)
    return 1 if close_arr[j] > close_arr[i] else 0


def bracket_direction_accuracy_symbol(g, offset_atr, max_hold=20, true_horizon=10):
    open_ = g["open"].to_numpy(dtype=np.float64)
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    close = g["close"].to_numpy(dtype=np.float64)
    atr = g["atr"].to_numpy(dtype=np.float64)
    gate = g["gate_on"].to_numpy(dtype=bool)
    n = len(g)
    correct = 0
    total = 0
    no_trigger = 0
    for i in np.flatnonzero(gate):
        if i + 1 >= n:
            continue
        p = close[i]
        off = offset_atr * atr[i]
        buy = p + off
        sell = p - off
        triggered = None
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            hb = high[j] >= buy
            hs = low[j] <= sell
            if hb and hs:
                triggered = 1 if open_[j] >= p else 0
                break
            if hb:
                triggered = 1
                break
            if hs:
                triggered = 0
                break
        if triggered is None:
            no_trigger += 1
            continue
        correct += int(triggered == _future_true_dir(close, i, true_horizon))
        total += 1
    acc = (correct / total) if total > 0 else np.nan
    return total, acc, no_trigger


def bracket_pnl_symbol(g, offset, tp, sl, max_hold=20):
    open_ = g["open"].to_numpy(dtype=np.float64)
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    close = g["close"].to_numpy(dtype=np.float64)
    atr = g["atr"].to_numpy(dtype=np.float64)
    gate = g["gate_on"].to_numpy(dtype=bool)
    n = len(g)
    pnl = []
    exits = []
    for i in np.flatnonzero(gate):
        if i + 1 >= n:
            continue
        p = close[i]
        a = atr[i]
        buy = p + offset * a
        sell = p - offset * a
        direction = None
        entry = None
        done = False
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if direction is None:
                hb = high[j] >= buy
                hs = low[j] <= sell
                if hb and hs:
                    direction = 1 if open_[j] >= p else -1
                    entry = buy if direction == 1 else sell
                elif hb:
                    direction = 1
                    entry = buy
                elif hs:
                    direction = -1
                    entry = sell
                else:
                    continue
            if direction == 1:
                tp_hit = high[j] >= entry + tp * a
                sl_hit = low[j] <= entry - sl * a
            else:
                tp_hit = low[j] <= entry - tp * a
                sl_hit = high[j] >= entry + sl * a
            if tp_hit and sl_hit:
                pnl.append(-(sl + offset))
                exits.append("sl")
                done = True
                break
            if tp_hit:
                pnl.append(tp - offset)
                exits.append("tp")
                done = True
                break
            if sl_hit:
                pnl.append(-(sl + offset))
                exits.append("sl")
                done = True
                break
        if not done:
            if direction is None:
                pnl.append(0.0)
                exits.append("no_trigger")
            else:
                lc = close[min(i + max_hold, n - 1)]
                pp = (lc - entry) / a if direction == 1 else (entry - lc) / a
                pnl.append(pp - offset)
                exits.append("timeout")
    return np.asarray(pnl, dtype=np.float64), np.asarray(exits, dtype=object)


def straddle_symbol(g, sl_atr=0.5, trail_activation=0.5, trail_distance=0.3, max_hold=30):
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    close = g["close"].to_numpy(dtype=np.float64)
    gate = g["gate_on"].to_numpy(dtype=bool)
    atr = g["atr"].to_numpy(dtype=np.float64)
    n = len(g)
    out = []
    for i in np.flatnonzero(gate):
        if i + 1 >= n:
            continue
        p = close[i]
        a = atr[i]
        long_sl = p - sl_atr * a
        short_sl = p + sl_atr * a
        long_peak = p
        short_trough = p
        long_pnl = None
        short_pnl = None
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if long_pnl is None:
                long_peak = max(long_peak, high[j])
                if long_peak - p >= trail_activation * a:
                    long_sl = max(long_sl, long_peak - trail_distance * a)
                if low[j] <= long_sl:
                    long_pnl = (long_sl - p) / a
            if short_pnl is None:
                short_trough = min(short_trough, low[j])
                if p - short_trough >= trail_activation * a:
                    short_sl = min(short_sl, short_trough + trail_distance * a)
                if high[j] >= short_sl:
                    short_pnl = (p - short_sl) / a
            if long_pnl is not None and short_pnl is not None:
                break
        lc = close[min(i + max_hold, n - 1)]
        if long_pnl is None:
            long_pnl = (lc - p) / a
        if short_pnl is None:
            short_pnl = (p - lc) / a
        out.append(0.5 * (long_pnl + short_pnl))
    return np.asarray(out, dtype=np.float64)


def summarize_pnl(p):
    if p.size == 0:
        return dict(n=0, ev=np.nan, wr=np.nan, sharpe=np.nan)
    ev = float(np.mean(p))
    sd = float(np.std(p))
    wr = float(np.mean(p > 0))
    sharpe = float(ev / sd * np.sqrt(252.0)) if sd > 1e-12 else np.nan
    return dict(n=int(p.size), ev=ev, wr=wr, sharpe=sharpe)


by_symbol = [g.sort_values("time_key").reset_index(drop=True) for _, g in merged.groupby("symbol", sort=True)]

acc_rows = []
for off in [0.05, 0.10, 0.15, 0.20, 0.30]:
    trig = 0
    corr_sum = 0.0
    gate_n = 0
    nt_sum = 0
    for g in by_symbol:
        t, a, nt = bracket_direction_accuracy_symbol(g, off, max_hold=20, true_horizon=10)
        gate_count = int(g["gate_on"].sum())
        gate_n += gate_count
        nt_sum += nt
        if t > 0 and np.isfinite(a):
            trig += t
            corr_sum += a * t
    acc_rows.append(
        dict(
            offset=off,
            acc=(corr_sum / trig) if trig > 0 else np.nan,
            n_triggered=trig,
            gate_n=gate_n,
            trigger_rate=(trig / gate_n) if gate_n > 0 else np.nan,
            no_trigger=nt_sum,
        )
    )
acc_df = pd.DataFrame(acc_rows).sort_values("acc", ascending=False)

br_rows = []
for off in [0.05, 0.10, 0.15, 0.20]:
    for tp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for sl in [0.15, 0.22, 0.30, 0.50]:
            pnl_all = []
            ex_all = []
            for g in by_symbol:
                p, ex = bracket_pnl_symbol(g, off, tp, sl, max_hold=20)
                if p.size:
                    pnl_all.append(p)
                    ex_all.append(ex)
            if not pnl_all:
                continue
            p = np.concatenate(pnl_all)
            ex = np.concatenate(ex_all)
            st = summarize_pnl(p)
            if st["n"] < 100:
                continue
            br_rows.append(
                dict(
                    offset=off,
                    tp=tp,
                    sl=sl,
                    n=st["n"],
                    ev=st["ev"],
                    wr=st["wr"],
                    sharpe=st["sharpe"],
                    tp_rate=float(np.mean(ex == "tp")),
                    sl_rate=float(np.mean(ex == "sl")),
                    timeout_rate=float(np.mean(ex == "timeout")),
                    no_trigger_rate=float(np.mean(ex == "no_trigger")),
                )
            )
bracket_df = pd.DataFrame(br_rows).sort_values(["ev", "sharpe"], ascending=False)

st_rows = []
for sl in [0.3, 0.5, 0.7, 1.0]:
    for ta in [0.3, 0.5, 0.8, 1.0]:
        for td in [0.2, 0.3, 0.5]:
            p_all = []
            for g in by_symbol:
                p = straddle_symbol(g, sl_atr=sl, trail_activation=ta, trail_distance=td, max_hold=30)
                if p.size:
                    p_all.append(p)
            if not p_all:
                continue
            p = np.concatenate(p_all)
            st = summarize_pnl(p)
            if st["n"] < 100:
                continue
            st_rows.append(dict(sl=sl, trail_act=ta, trail_dist=td, n=st["n"], ev=st["ev"], wr=st["wr"], sharpe=st["sharpe"]))
straddle_df = pd.DataFrame(st_rows).sort_values(["ev", "sharpe"], ascending=False)

best_br = bracket_df.iloc[0].to_dict() if len(bracket_df) else {}
best_st = straddle_df.iloc[0].to_dict() if len(straddle_df) else {}
compare = pd.DataFrame(
    [
        {
            "strategy": "bracket_best",
            "params": f"off={best_br.get('offset', np.nan)},tp={best_br.get('tp', np.nan)},sl={best_br.get('sl', np.nan)}",
            "n": int(best_br.get("n", 0)) if best_br else 0,
            "ev_atr": best_br.get("ev", np.nan),
            "wr": best_br.get("wr", np.nan),
            "sharpe": best_br.get("sharpe", np.nan),
        },
        {
            "strategy": "straddle_best",
            "params": f"sl={best_st.get('sl', np.nan)},ta={best_st.get('trail_act', np.nan)},td={best_st.get('trail_dist', np.nan)}",
            "n": int(best_st.get("n", 0)) if best_st else 0,
            "ev_atr": best_st.get("ev", np.nan),
            "wr": best_st.get("wr", np.nan),
            "sharpe": best_st.get("sharpe", np.nan),
        },
    ]
)

print("\n=== Bracket Direction Accuracy (by offset) ===")
print(acc_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n=== Bracket Scan Top20 (by EV) ===")
print(bracket_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n=== Straddle Scan Top20 (by EV) ===")
print(straddle_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n=== Strategy Comparison ===")
print(compare.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

out_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(out_dir, exist_ok=True)
acc_df.to_csv(os.path.join(out_dir, "l2_bracket_direction_accuracy.csv"), index=False)
bracket_df.to_csv(os.path.join(out_dir, "l2_bracket_scan.csv"), index=False)
straddle_df.to_csv(os.path.join(out_dir, "l2_straddle_scan.csv"), index=False)
compare.to_csv(os.path.join(out_dir, "l2_bracket_vs_straddle_compare.csv"), index=False)
print(f"\nSaved CSVs to: {out_dir}")

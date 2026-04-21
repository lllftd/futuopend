import os
import pickle
import numpy as np
import pandas as pd

from core.trainers.constants import MODEL_DIR, PREPARED_DATASET_CACHE_FILE, L1A_OUTPUT_CACHE_FILE, L1B_OUTPUT_CACHE_FILE, L1C_OUTPUT_CACHE_FILE
from core.trainers.stack_v2_common import load_output_cache, build_stack_time_splits
from core.trainers.l2.train import _build_l2_frame, _decision_edge_atr_array, _conditional_tau_from_state, _l2_build_two_stage_labels


def _atr_series(df: pd.DataFrame) -> np.ndarray:
    for c in ("lbl_atr", "atr_5m", "atr_1m"):
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 1.0)
            return np.clip(x, 1e-4, np.inf)
    x = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")).to_numpy(dtype=np.float64)
    x = pd.Series(x).ewm(span=14, min_periods=1).mean().to_numpy(dtype=np.float64)
    return np.clip(np.nan_to_num(x, nan=1.0), 1e-4, np.inf)


def first_trigger_direction(sub: pd.DataFrame, i: int, offset_atr: float, max_hold: int):
    price = float(sub.iloc[i]["close"])
    atr = float(sub.iloc[i]["atr"])
    buy_level = price + offset_atr * atr
    sell_level = price - offset_atr * atr
    end = min(i + 1 + max_hold, len(sub))
    for j in range(i + 1, end):
        bar = sub.iloc[j]
        hit_buy = float(bar["high"]) >= buy_level
        hit_sell = float(bar["low"]) <= sell_level
        if hit_buy and hit_sell:
            return ("long" if float(bar["open"]) >= price else "short"), j, (buy_level if float(bar["open"]) >= price else sell_level)
        if hit_buy:
            return "long", j, buy_level
        if hit_sell:
            return "short", j, sell_level
    return None, None, None


def bracket_direction_accuracy(sub: pd.DataFrame, gate_mask: np.ndarray, offsets, max_hold=20, horizon=10):
    out = []
    idxs = np.flatnonzero(gate_mask)
    for off in offsets:
        correct = 0
        total = 0
        no_trigger = 0
        for i in idxs:
            d, _j, _e = first_trigger_direction(sub, int(i), float(off), int(max_hold))
            if d is None:
                no_trigger += 1
                continue
            fut_i = min(int(i) + int(horizon), len(sub) - 1)
            true_dir = "long" if float(sub.iloc[fut_i]["close"]) > float(sub.iloc[i]["close"]) else "short"
            correct += int(d == true_dir)
            total += 1
        acc = (correct / total) if total > 0 else np.nan
        out.append({"offset": off, "acc": acc, "n_triggered": total, "n_no_trigger": no_trigger})
    return pd.DataFrame(out)


def bracket_pnl(sub: pd.DataFrame, gate_mask: np.ndarray, offset, tp, sl, max_hold=20):
    rows = []
    for i in np.flatnonzero(gate_mask):
        i = int(i)
        price = float(sub.iloc[i]["close"])
        atr = float(sub.iloc[i]["atr"])
        d, j0, entry_price = first_trigger_direction(sub, i, float(offset), int(max_hold))
        if d is None:
            rows.append({"pnl_atr": 0.0, "exit": "no_trigger", "dir": "none"})
            continue
        end = min(i + 1 + int(max_hold), len(sub))
        closed = False
        for j in range(int(j0), end):
            bar = sub.iloc[j]
            if d == "long":
                tp_hit = float(bar["high"]) >= entry_price + float(tp) * atr
                sl_hit = float(bar["low"]) <= entry_price - float(sl) * atr
            else:
                tp_hit = float(bar["low"]) <= entry_price - float(tp) * atr
                sl_hit = float(bar["high"]) >= entry_price + float(sl) * atr
            if tp_hit and sl_hit:
                rows.append({"pnl_atr": -(float(sl) + float(offset)), "exit": "sl", "dir": d})
                closed = True
                break
            if tp_hit:
                rows.append({"pnl_atr": float(tp) - float(offset), "exit": "tp", "dir": d})
                closed = True
                break
            if sl_hit:
                rows.append({"pnl_atr": -(float(sl) + float(offset)), "exit": "sl", "dir": d})
                closed = True
                break
        if not closed:
            last = float(sub.iloc[end - 1]["close"]) if end - 1 >= 0 else float(sub.iloc[-1]["close"])
            pnl = (last - entry_price) / atr
            if d == "short":
                pnl = -pnl
            rows.append({"pnl_atr": float(pnl - float(offset)), "exit": "timeout", "dir": d})
    return pd.DataFrame(rows)


def straddle_equity(sub: pd.DataFrame, gate_mask: np.ndarray, sl_atr=0.5, trail_activation=0.5, trail_distance=0.3, max_hold=30):
    rows = []
    for i in np.flatnonzero(gate_mask):
        i = int(i)
        price = float(sub.iloc[i]["close"])
        atr = float(sub.iloc[i]["atr"])
        long_entry = price
        short_entry = price
        long_sl = price - float(sl_atr) * atr
        short_sl = price + float(sl_atr) * atr
        long_peak = price
        short_trough = price
        long_pnl = None
        short_pnl = None
        end = min(i + 1 + int(max_hold), len(sub))
        for j in range(i + 1, end):
            bar = sub.iloc[j]
            hi = float(bar["high"])
            lo = float(bar["low"])
            if long_pnl is None:
                if hi > long_peak:
                    long_peak = hi
                if long_peak - long_entry >= float(trail_activation) * atr:
                    long_sl = max(long_sl, long_peak - float(trail_distance) * atr)
                if lo <= long_sl:
                    long_pnl = (long_sl - long_entry) / atr
            if short_pnl is None:
                if lo < short_trough:
                    short_trough = lo
                if short_entry - short_trough >= float(trail_activation) * atr:
                    short_sl = min(short_sl, short_trough + float(trail_distance) * atr)
                if hi >= short_sl:
                    short_pnl = (short_entry - short_sl) / atr
            if long_pnl is not None and short_pnl is not None:
                break
        last_close = float(sub.iloc[end - 1]["close"]) if end - 1 >= 0 else float(sub.iloc[-1]["close"])
        if long_pnl is None:
            long_pnl = (last_close - long_entry) / atr
        if short_pnl is None:
            short_pnl = (short_entry - last_close) / atr
        total = 0.5 * (float(long_pnl) + float(short_pnl))
        rows.append({"total_pnl_atr": total, "long_pnl": float(long_pnl), "short_pnl": float(short_pnl)})
    return pd.DataFrame(rows)


def safe_sharpe(x: pd.Series) -> float:
    arr = x.to_numpy(dtype=np.float64)
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return float("nan")
    return float(np.mean(arr) / sd * np.sqrt(252.0))


prep_path = os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE)
with open(prep_path, "rb") as f:
    prep = pickle.load(f)
df = prep["df"].copy()
df["time_key"] = pd.to_datetime(df["time_key"])

a = load_output_cache(L1A_OUTPUT_CACHE_FILE)
b = load_output_cache(L1B_OUTPUT_CACHE_FILE)
c = load_output_cache(L1C_OUTPUT_CACHE_FILE)
l2 = load_output_cache("l2_outputs.pkl")

# Rebuild labels for active reference
frame_l2, _ = _build_l2_frame(df, a, b, c)
splits = build_stack_time_splits(df["time_key"])
train_mask = np.asarray(splits.l2_train_mask, dtype=bool)
edge = _decision_edge_atr_array(df)
_, _, _, tau_row = _conditional_tau_from_state(frame_l2, edge, train_mask)
y_decision = np.full(len(df), 1, dtype=np.int64)
y_decision[edge > tau_row] = 0
y_decision[edge < -tau_row] = 2
y_trade, y_dir = _l2_build_two_stage_labels(y_decision)

base = df[["symbol", "time_key", "open", "high", "low", "close"]].copy()
base["atr"] = _atr_series(df)
base["y_trade"] = y_trade
base["y_dir"] = y_dir
base = base.merge(l2[["symbol", "time_key", "l2_decision_neutral"]], on=["symbol", "time_key"], how="left")
base["gate_pred"] = 1.0 - pd.to_numeric(base["l2_decision_neutral"], errors="coerce").fillna(1.0)

trade_threshold = 0.4087
print(f"trade_threshold={trade_threshold}")

acc_rows = []
br_rows = []
st_rows = []

for sym, sub in base.sort_values(["symbol", "time_key"]).groupby("symbol", sort=False):
    sub = sub.reset_index(drop=True)
    gate_mask = sub["gate_pred"].to_numpy(dtype=np.float64) >= trade_threshold
    n_gate = int(np.sum(gate_mask))
    if n_gate < 50:
        continue

    acc_df = bracket_direction_accuracy(sub, gate_mask, offsets=[0.05, 0.10, 0.15, 0.20, 0.30], max_hold=20, horizon=10)
    acc_df["symbol"] = sym
    acc_df["n_gate"] = n_gate
    acc_rows.append(acc_df)

    for off in [0.05, 0.10, 0.15, 0.20]:
        for tp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            for sl in [0.15, 0.22, 0.30, 0.50]:
                res = bracket_pnl(sub, gate_mask, off, tp, sl, max_hold=20)
                if len(res) < 50:
                    continue
                br_rows.append({
                    "symbol": sym,
                    "offset": off,
                    "tp": tp,
                    "sl": sl,
                    "n": int(len(res)),
                    "ev": float(res["pnl_atr"].mean()),
                    "wr": float((res["pnl_atr"] > 0).mean()),
                    "sharpe": safe_sharpe(res["pnl_atr"]),
                    "tp_rate": float((res["exit"] == "tp").mean()),
                    "sl_rate": float((res["exit"] == "sl").mean()),
                    "timeout_rate": float((res["exit"] == "timeout").mean()),
                    "no_trigger_rate": float((res["exit"] == "no_trigger").mean()),
                })

    for sl in [0.3, 0.5, 0.7, 1.0]:
        for ta in [0.3, 0.5, 0.8, 1.0]:
            for td in [0.2, 0.3, 0.5]:
                res = straddle_equity(sub, gate_mask, sl_atr=sl, trail_activation=ta, trail_distance=td, max_hold=30)
                if len(res) < 50:
                    continue
                st_rows.append({
                    "symbol": sym,
                    "sl": sl,
                    "trail_act": ta,
                    "trail_dist": td,
                    "n": int(len(res)),
                    "ev": float(res["total_pnl_atr"].mean()),
                    "wr": float((res["total_pnl_atr"] > 0).mean()),
                    "sharpe": safe_sharpe(res["total_pnl_atr"]),
                })

acc_all = pd.concat(acc_rows, ignore_index=True) if acc_rows else pd.DataFrame()
br_all = pd.DataFrame(br_rows)
st_all = pd.DataFrame(st_rows)

print("\n=== Bracket Direction Accuracy (aggregate by offset) ===")
if not acc_all.empty:
    g = acc_all.groupby("offset", as_index=False).agg(
        n_gate=("n_gate", "sum"),
        n_triggered=("n_triggered", "sum"),
        n_no_trigger=("n_no_trigger", "sum"),
        acc=("acc", "mean"),
    )
    print(g.to_string(index=False))
else:
    print("(empty)")

print("\n=== Bracket PnL Top 20 (all symbols) ===")
if not br_all.empty:
    top = br_all.groupby(["offset", "tp", "sl"], as_index=False).agg(
        n=("n", "sum"),
        ev=("ev", "mean"),
        wr=("wr", "mean"),
        sharpe=("sharpe", "mean"),
        tp_rate=("tp_rate", "mean"),
        sl_rate=("sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        no_trigger_rate=("no_trigger_rate", "mean"),
    ).sort_values("ev", ascending=False)
    print(top.head(20).to_string(index=False))
else:
    print("(empty)")

print("\n=== Straddle Top 20 (all symbols) ===")
if not st_all.empty:
    top_s = st_all.groupby(["sl", "trail_act", "trail_dist"], as_index=False).agg(
        n=("n", "sum"),
        ev=("ev", "mean"),
        wr=("wr", "mean"),
        sharpe=("sharpe", "mean"),
    ).sort_values("ev", ascending=False)
    print(top_s.head(20).to_string(index=False))
else:
    print("(empty)")

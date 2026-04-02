import pandas as pd
import numpy as np
from core.utils import load_price_data
from core.pa_rules import add_all_pa_features
from core.indicators import atr, add_pseudo_cvd, cvd_divergence_features
from datetime import time
import warnings
import itertools

warnings.filterwarnings("ignore")

symbol = "SPY"
print(f"Loading data for {symbol}...")
raw = load_price_data(symbol)
raw["time_key"] = pd.to_datetime(raw["time_key"])
raw = raw[(raw["time_key"] >= "2024-01-01") & (raw["time_key"] < "2026-01-01")].reset_index(drop=True)

# 1. Resample to 5m to calculate PA features
raw_idx = raw.set_index("time_key")
df_5m = raw_idx[["open", "high", "low", "close", "volume"]].resample("5min").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
).dropna(subset=["open"]).reset_index()

# 2. Compute ATR and EMA on 5m
df_5m["atr_raw"] = atr(df_5m, length=14)
df_5m["ema_20"] = df_5m["close"].ewm(span=20, adjust=False).mean()

# 3. Compute PA features directly on 5m
print("Computing 5m PA features...")
df_pa = add_all_pa_features(df_5m, df_5m["atr_raw"], timeframe="1min")

# Add custom filter metrics for grid search
rng_5m = np.maximum(df_5m["high"] - df_5m["low"], 1e-12)
body_5m = np.abs(df_5m["close"] - df_5m["open"])
df_5m["body_ratio"] = body_5m / rng_5m
df_5m["body_mult"] = (body_5m / body_5m.rolling(20, min_periods=5).mean()).fillna(0)

ema_slope_5m = df_5m["ema_20"].diff()
ema_slope_abs_ma40_5m = ema_slope_5m.abs().rolling(40, min_periods=10).mean().replace(0, np.nan)
df_5m["ema_slope_ratio"] = (ema_slope_5m / ema_slope_abs_ma40_5m).fillna(0)

df_pa["prev_5m_body_ratio"] = df_5m["body_ratio"]
df_pa["prev_5m_body_mult"] = df_5m["body_mult"]
df_pa["prev_5m_ema_slope_ratio"] = df_5m["ema_slope_ratio"]

# 4. Prepare for 1m execution merge
df_pa["signal_time"] = df_pa["time_key"] + pd.Timedelta(minutes=5)
df_pa = df_pa.rename(columns={
    "high": "prev_5m_high",
    "low": "prev_5m_low",
    "close": "prev_5m_close",
    "open": "prev_5m_open",
    "atr_raw": "prev_5m_atr",
    "ema_20": "prev_5m_ema20"
})
df_pa = df_pa.drop(columns=["time_key", "volume"])

# Merge with 1m data using merge_asof
print("Merging 5m features with 1m execution data...")
df_1m = pd.merge_asof(
    raw.sort_values("time_key"),
    df_pa.rename(columns={"signal_time": "time_key"}).sort_values("time_key"),
    on="time_key",
    direction="backward",
    tolerance=pd.Timedelta(minutes=4)
)

# Filter out rows before the first 5m bar is fully formed
df_1m = df_1m.dropna(subset=["pa_env_state"]).reset_index(drop=True)

# Add 1m CVD / Order Flow Proxy Features
print("Computing 1m CVD features...")
cvd_df = add_pseudo_cvd(df_1m, method="clv_body_volume")[["cvd_pressure", "cvd_session"]]
df_1m = df_1m.join(cvd_df)

div_df = cvd_divergence_features(df_1m, cvd_column="cvd_session", lookback=10, slope_lookback=3)
df_1m = df_1m.join(div_df)

print(f"Data ready. Total 1-min bars: {len(df_1m)}")

times = df_1m["time_key"]
bar_times = times.dt.time
close_p = df_1m["close"].to_numpy()
open_p = df_1m["open"].to_numpy()
high_p = df_1m["high"].to_numpy()
low_p = df_1m["low"].to_numpy()

# 5m PA Features
state = df_1m["pa_env_state"].astype(str).to_numpy()
h1 = (df_1m["pa_h_count"] == 1).fillna(False).to_numpy()
h2 = df_1m["pa_is_h2_setup"].fillna(False).to_numpy()
l1 = (df_1m["pa_l_count"] == 1).fillna(False).to_numpy()
l2 = df_1m["pa_is_l2_setup"].fillna(False).to_numpy()
mag_bull = df_1m["pa_mag20_bull"].fillna(False).to_numpy()
mag_bear = df_1m["pa_mag20_bear"].fillna(False).to_numpy()

wedge_up = df_1m["pa_wedge_third_push_up"].fillna(False).to_numpy()
wedge_down = df_1m["pa_wedge_third_push_down"].fillna(False).to_numpy()
overshoot_up = df_1m["pa_channel_overshoot_revert_up"].fillna(False).to_numpy()
overshoot_down = df_1m["pa_channel_overshoot_revert_down"].fillna(False).to_numpy()

bo_up = df_1m["pa_breakout_success_up"].fillna(False).to_numpy()
bo_down = df_1m["pa_breakout_success_down"].fillna(False).to_numpy()

pa_stop_long = df_1m["pa_stop_long"].to_numpy()
pa_stop_short = df_1m["pa_stop_short"].to_numpy()
pa_mm_up = df_1m["pa_mm_target_up"].to_numpy()
pa_mm_down = df_1m["pa_mm_target_down"].to_numpy()

prev_5m_high = df_1m["prev_5m_high"].to_numpy()
prev_5m_low = df_1m["prev_5m_low"].to_numpy()
prev_5m_atr = df_1m["prev_5m_atr"].to_numpy()
prev_5m_ema20 = df_1m["prev_5m_ema20"].to_numpy()
prev_5m_body_ratio = df_1m["prev_5m_body_ratio"].to_numpy()
prev_5m_body_mult = df_1m["prev_5m_body_mult"].to_numpy()
prev_5m_ema_slope_ratio = df_1m["prev_5m_ema_slope_ratio"].to_numpy()

# 1m CVD Arrays
cvd_pressure = df_1m["cvd_pressure"].to_numpy()
cvd_classic_long_ok = df_1m["cvd_classic_long_ok"].to_numpy()    # False -> bearish divergence
cvd_classic_short_ok = df_1m["cvd_classic_short_ok"].to_numpy()  # False -> bullish divergence

def run_simulation(sl_atr_min, tp_atr_mult, body_mult_thresh, slope_thresh, use_cvd_filter=True, time_limit=30):
    position = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    entry_time = None
    entry_idx = -1

    setup_long = False
    setup_long_price = np.nan
    setup_long_sl = np.nan
    setup_long_tp = np.nan

    setup_short = False
    setup_short_price = np.nan
    setup_short_sl = np.nan
    setup_short_tp = np.nan

    trade_records = []
    trades_today = 0
    
    for i in range(1, len(df_1m)):
        curr_time = bar_times.iloc[i]
        is_eod = (i == len(df_1m) - 1) or (times.iloc[i].date() != times.iloc[i+1].date()) or (curr_time >= time(15, 50))
        is_new_5m = times.iloc[i].minute % 5 == 0
        
        if i > 0 and times.iloc[i].date() != times.iloc[i-1].date():
            trades_today = 0
            setup_long = False
            setup_short = False
            
        if is_eod or curr_time >= time(15, 30):
            setup_long = False
            setup_short = False
            
        # --- Look for New Setups ---
        # 避免垃圾时间：避开中午 11:30 到 13:30 的无序震荡期
        is_garbage_time = time(11, 30) <= curr_time < time(13, 30)
        if is_new_5m and position == 0 and not is_eod and curr_time < time(15, 30) and not is_garbage_time:
            setup_long = False
            setup_short = False
            
            # 1. 强通道判定：EMA斜率 > slope_thresh
            is_strong_bull_trend = prev_5m_ema_slope_ratio[i] > slope_thresh
            is_strong_bear_trend = prev_5m_ema_slope_ratio[i] < -slope_thresh
            
            # 2. 动力棒判定：实体比例 > 60%，实体大小 > body_mult_thresh * 20周期平均实体，且顺着EMA方向
            custom_mag_bull = (prev_5m_low[i] > prev_5m_ema20[i]) and \
                              (prev_5m_body_ratio[i] > 0.6) and \
                              (prev_5m_body_mult[i] > body_mult_thresh) and \
                              (close_p[i-1] > open_p[i-1])
            
            custom_mag_bear = (prev_5m_high[i] < prev_5m_ema20[i]) and \
                              (prev_5m_body_ratio[i] > 0.6) and \
                              (prev_5m_body_mult[i] > body_mult_thresh) and \
                              (close_p[i-1] < open_p[i-1])
            
            # --- 均值回归 (Fade) ---
            # 强多头爆出巨阳(抢筹高潮)，或者通道超调/楔形衰竭，尝试摸顶做空
            fade_short = (is_strong_bull_trend and custom_mag_bull) or overshoot_up[i] or wedge_up[i]
            # 强空头爆出巨阴(恐慌抛售)，或者通道超调/楔形衰竭，尝试抄底做多
            fade_long = (is_strong_bear_trend and custom_mag_bear) or overshoot_down[i] or wedge_down[i]
            
            # --- 顺势突破 (Trend / Breakout) ---
            # 强通道中的自然回调(H1/H2) 或者 交易区间的成功突破
            trend_long = (is_strong_bull_trend and (h1[i] or h2[i])) or bo_up[i]
            trend_short = (is_strong_bear_trend and (l1[i] or l2[i])) or bo_down[i]
            
            if fade_long or trend_long:
                setup_long = True
            if fade_short or trend_short:
                setup_short = True
                
            # 如果同时触发（虽然罕见），则放弃开仓
            if setup_long and setup_short:
                setup_long = False
                setup_short = False
                    
            if setup_long:
                setup_long_price = prev_5m_high[i] + 0.01
                
                # SL: 优先使用结构止损防打掉，但至少要距离 sl_atr_min 个 ATR
                sl_candidate = pa_stop_long[i]
                if np.isfinite(sl_candidate) and sl_candidate < setup_long_price:
                    setup_long_sl = sl_candidate - 0.1 * prev_5m_atr[i]
                    if (setup_long_price - setup_long_sl) < sl_atr_min * prev_5m_atr[i]:
                        setup_long_sl = setup_long_price - sl_atr_min * prev_5m_atr[i]
                else:
                    setup_long_sl = setup_long_price - sl_atr_min * prev_5m_atr[i]
                    
                # TP: 不再看 Risk，直接设定固定 ATR 倍数的爆发目标
                setup_long_tp = setup_long_price + tp_atr_mult * prev_5m_atr[i]
                    
            if setup_short:
                setup_short_price = prev_5m_low[i] - 0.01
                
                sl_candidate = pa_stop_short[i]
                if np.isfinite(sl_candidate) and sl_candidate > setup_short_price:
                    setup_short_sl = sl_candidate + 0.1 * prev_5m_atr[i]
                    if (setup_short_sl - setup_short_price) < sl_atr_min * prev_5m_atr[i]:
                        setup_short_sl = setup_short_price + sl_atr_min * prev_5m_atr[i]
                else:
                    setup_short_sl = setup_short_price + sl_atr_min * prev_5m_atr[i]
                    
                setup_short_tp = setup_short_price - tp_atr_mult * prev_5m_atr[i]

        # --- Check Active Setups for Entry ---
        if position == 0 and not is_eod:
            if setup_long and high_p[i] > setup_long_price:
                # Add CVD Divergence filter for Long
                cvd_ok = True
                if use_cvd_filter:
                    # For long, we check if there's a recent bullish divergence (price lower low, but CVD not lower low)
                    recent_bull_div = not all(cvd_classic_short_ok[max(0, i-3):i+1])
                    cvd_ok = recent_bull_div or cvd_pressure[i] > 0
                
                if cvd_ok:
                    position = 1
                    entry_price = max(open_p[i], setup_long_price)
                    entry_time = times.iloc[i]
                    entry_idx = i
                    stop_loss = setup_long_sl
                    take_profit = setup_long_tp
                    trades_today += 1
                setup_long = False
                setup_short = False
                
            elif setup_short and low_p[i] < setup_short_price:
                # Add CVD Divergence filter for Short
                cvd_ok = True
                if use_cvd_filter:
                    recent_bear_div = not all(cvd_classic_long_ok[max(0, i-3):i+1])
                    cvd_ok = recent_bear_div or cvd_pressure[i] < 0
                    
                if cvd_ok:
                    position = -1
                    entry_price = min(open_p[i], setup_short_price)
                    entry_time = times.iloc[i]
                    entry_idx = i
                    stop_loss = setup_short_sl
                    take_profit = setup_short_tp
                    trades_today += 1
                setup_long = False
                setup_short = False

        # --- Check Exits ---
        if position != 0:
            exit_reason = None
            exit_price = np.nan
            
            if position == 1:
                if low_p[i] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                elif high_p[i] >= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                elif is_eod:
                    exit_price = close_p[i]
                    exit_reason = "eod"
                elif i - entry_idx >= time_limit: # TIME STOP
                    exit_price = close_p[i]
                    exit_reason = "time_stop"
                else:
                    # Trailing Stop: lock profit using EMA20 or 50% MFE if we reached 1 ATR
                    mfe = high_p[i] - entry_price
                    if mfe > 1.0 * prev_5m_atr[i]: # activate trail sooner for options
                        new_sl_ema = prev_5m_ema20[i] - 0.2 * prev_5m_atr[i]
                        new_sl_mfe = entry_price + mfe * 0.5
                        new_sl = max(new_sl_ema, new_sl_mfe)
                        if new_sl > stop_loss:
                            stop_loss = new_sl
                            
            elif position == -1:
                if high_p[i] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                elif low_p[i] <= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                elif is_eod:
                    exit_price = close_p[i]
                    exit_reason = "eod"
                elif i - entry_idx >= time_limit: # TIME STOP
                    exit_price = close_p[i]
                    exit_reason = "time_stop"
                else:
                    mfe = entry_price - low_p[i]
                    if mfe > 1.0 * prev_5m_atr[i]:
                        new_sl_ema = prev_5m_ema20[i] + 0.2 * prev_5m_atr[i]
                        new_sl_mfe = entry_price - mfe * 0.5
                        new_sl = min(new_sl_ema, new_sl_mfe)
                        if new_sl < stop_loss:
                            stop_loss = new_sl
                            
            if exit_reason:
                trade_return = position * (exit_price / entry_price - 1.0)
                holding_bars = i - entry_idx
                trade_records.append({
                    "return": trade_return,
                    "holding_bars": holding_bars,
                    "reason": exit_reason
                })
                position = 0
                
    log = pd.DataFrame(trade_records)
    if log.empty:
        return 0, 0, 0, 0, 0, log
        
    round_trip_cost = 0.0002 # 2 bps, corresponds to $0.10 SPY move (covers $0.05 option profit at Delta 0.5)
    log["net_pnl"] = log["return"] - round_trip_cost
    
    wins = log[log["net_pnl"] > 0]["net_pnl"]
    losses = log[log["net_pnl"] < 0]["net_pnl"]
    
    win_rate = len(wins) / len(log) if len(log) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    net_ret = log["net_pnl"].sum()
    gross_ret = log["return"].sum()
    
    return gross_ret, net_ret, win_rate, pf, len(log), log

# --- GRID SEARCH (Fine-tuning 30m + CVD) ---
sl_atr_options = [0.3, 0.5, 0.75]
tp_atr_options = [1.5, 2.0, 3.0]
body_mult_options = [1.5, 2.0]
slope_thresh_options = [1.0, 1.5]

print("\nFine-Tuning 30m + CVD Options Strategy...")
print(f"{'SL_ATR':<6} | {'TP_ATR':<6} | {'BodyX':<5} | {'Slope':<5} | {'Trades':<6} | {'Gross%':<8} | {'Net(0.02%)':<10} | {'WinRate':<8} | {'PF':<6}")
print("-" * 85)

best_net = -np.inf
best_params = None
best_log = None

for sl_atr, tp_atr, body_m, slope_t in itertools.product(sl_atr_options, tp_atr_options, body_mult_options, slope_thresh_options):
    gross, net, wr, pf, n_trades, log = run_simulation(
        sl_atr_min=sl_atr, 
        tp_atr_mult=tp_atr, 
        body_mult_thresh=body_m, 
        slope_thresh=slope_t,
        use_cvd_filter=True,
        time_limit=30
    )
    print(f"{sl_atr:<6} | {tp_atr:<6} | {body_m:<5} | {slope_t:<5} | {n_trades:<6} | {gross*100:>6.2f}% | {net*100:>8.2f}% | {wr*100:>6.2f}% | {pf:>5.2f}")
    
    if net > best_net:
        best_net = net
        best_params = (sl_atr, tp_atr, body_m, slope_t)
        best_log = log

print("-" * 85)
print(f"\nBEST PARAMS: SL_ATR = {best_params[0]}, TP_ATR = {best_params[1]}, Body Mult = {best_params[2]}, Slope = {best_params[3]}")
print(f"Best Net Return (0.02% cost): {best_net*100:.2f}%")

print("\nBest Model Exit Reasons:")
if best_log is not None and not best_log.empty:
    print(best_log["reason"].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))
    print(f"Avg Holding Time: {best_log['holding_bars'].mean():.1f} mins")

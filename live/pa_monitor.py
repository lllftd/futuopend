from __future__ import annotations

import csv
import numpy as np
import pandas as pd
from datetime import time, timedelta
from typing import Optional

from core.pa_rules import add_all_pa_features
from core.indicators import atr, add_pseudo_cvd, cvd_divergence_features
from live.monitor import SymbolMonitor, PositionState, TradeRecord, _C, _log_info, _log_signal

class PAOptionsMonitor(SymbolMonitor):
    def __init__(self, *args, **kwargs):
        # Override default tag if not specified
        if "strategy_tag" not in kwargs:
            kwargs["strategy_tag"] = "PA_OPT"
        super().__init__(*args, **kwargs)
        
        # Override the minute signal path so it doesn't clash with V3
        self.minute_signal_path = self.daily_output_dir / f"{self.symbol.lower()}_pa_minute_signals.csv"
        self._ensure_minute_signal_header()

        # Strategy parameters
        self.pa_sl_atr = 0.75
        self.pa_tp_atr = 3.0
        self.pa_body_mult_thresh = 2.0
        self.pa_slope_thresh = 1.0
        self.pa_time_limit_mins = 30
        self.pa_use_cvd_filter = True
        
        self.last_reported_5m_state = None
        self.last_reported_structure = None

    def _minute_signal_columns(self) -> list[str]:
        return [
            "time_key", "symbol", "open", "high", "low", "close", "volume",
            "cvd_sim", "cvd_real",
            "prev_5m_env_state", "prev_5m_ema_slope_ratio", 
            "prev_5m_body_mult", "cvd_pressure",
            "cvd_classic_long_ok", "cvd_classic_short_ok",
            "zone_fade_long", "zone_fade_short",
            "entry_event", "entry_side", "entry_price", "entry_sl", "entry_tp", "entry_deadline",
            "exit_event", "exit_side", "exit_reason", "exit_price", "exit_trade_return",
            "position_side_after_bar", "position_entry_price", "position_stop_price", "position_take_profit"
        ]

    def _rebuild_features(self) -> None:
        if self.bars.empty:
            self.featured = pd.DataFrame()
            return
            
        df_1m = self.bars.copy()
        df_1m["time_key"] = pd.to_datetime(df_1m["time_key"])
        
        # Build 5m features
        raw_idx = df_1m.set_index("time_key")
        df_5m = raw_idx[["open", "high", "low", "close", "volume"]].resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"]).reset_index()
        
        if df_5m.empty:
            self.featured = pd.DataFrame()
            return
            
        df_5m["atr_raw"] = atr(df_5m, length=14)
        df_5m["ema_20"] = df_5m["close"].ewm(span=20, adjust=False).mean()
        
        rng_5m = np.maximum(df_5m["high"] - df_5m["low"], 1e-12)
        body_5m = np.abs(df_5m["close"] - df_5m["open"])
        df_5m["body_ratio"] = body_5m / rng_5m
        df_5m["body_mult"] = (body_5m / body_5m.rolling(20, min_periods=5).mean()).fillna(0)
        
        ema_slope_5m = df_5m["ema_20"].diff()
        ema_slope_abs_ma40_5m = ema_slope_5m.abs().rolling(40, min_periods=10).mean().replace(0, np.nan)
        df_5m["ema_slope_ratio"] = (ema_slope_5m / ema_slope_abs_ma40_5m).fillna(0)
        
        df_pa = add_all_pa_features(df_5m, df_5m["atr_raw"], timeframe="1min")
        
        df_pa["prev_5m_body_ratio"] = df_5m["body_ratio"]
        df_pa["prev_5m_body_mult"] = df_5m["body_mult"]
        df_pa["prev_5m_ema_slope_ratio"] = df_5m["ema_slope_ratio"]
        df_pa["signal_time"] = df_pa["time_key"] + pd.Timedelta(minutes=5)
        
        df_pa = df_pa.rename(columns={
            "high": "prev_5m_high", "low": "prev_5m_low",
            "close": "prev_5m_close", "open": "prev_5m_open",
            "atr_raw": "prev_5m_atr", "ema_20": "prev_5m_ema20"
        }).drop(columns=["time_key", "volume"])
        
        # Merge 5m features to 1m
        featured = pd.merge_asof(
            df_1m.sort_values("time_key"),
            df_pa.rename(columns={"signal_time": "time_key"}).sort_values("time_key"),
            on="time_key", direction="backward", tolerance=pd.Timedelta(minutes=4)
        )
        
        featured = featured.dropna(subset=["pa_env_state"]).reset_index(drop=True)
        if featured.empty:
            self.featured = pd.DataFrame()
            return
            
        # CVD logic
        real_cvd_series = self._build_real_cvd_series()
        cvd_sim_df = add_pseudo_cvd(featured, method="clv_body_volume")[["cvd_pressure", "cvd_session"]]
        featured = featured.join(cvd_sim_df)
        
        # Replace simulated cvd with real cvd for divergence if real is available
        featured["cvd_real_session"] = real_cvd_series.values[-len(featured):] if len(real_cvd_series) >= len(featured) else featured["cvd_session"]
        
        div_df = cvd_divergence_features(featured, cvd_column="cvd_real_session", lookback=10, slope_lookback=3)
        self.featured = featured.join(div_df)

    def _build_signal_snapshot(self, bar_time: pd.Timestamp) -> dict[str, object]:
        if self.featured.empty: return {}
        idx = self.featured.index[self.featured["time_key"] == bar_time]
        if len(idx) == 0: return {}
        row = self.featured.loc[idx[0]]
        i = idx[0]
        prev = self.featured.loc[i-1] if i > 0 else row
        
        minute_key = bar_time.floor("min")
        cvd_real = float(self.real_cvd_minute_value.get(minute_key, self.real_cvd_value))
        
        # Evaluate structure
        structures = []
        if row.get("prev_5m_ema_slope_ratio", 0) > self.pa_slope_thresh: structures.append("StrongBullTrend")
        if row.get("prev_5m_ema_slope_ratio", 0) < -self.pa_slope_thresh: structures.append("StrongBearTrend")
        
        custom_mag_bull = (row.get("prev_5m_low", 0) > row.get("prev_5m_ema20", np.inf)) and \
                          (row.get("prev_5m_body_ratio", 0) > 0.6) and \
                          (row.get("prev_5m_body_mult", 0) > self.pa_body_mult_thresh) and \
                          (prev.get("close", 0) > prev.get("open", 0))
        if custom_mag_bull: structures.append("MagBull")
        
        custom_mag_bear = (row.get("prev_5m_high", 0) < row.get("prev_5m_ema20", -np.inf)) and \
                          (row.get("prev_5m_body_ratio", 0) > 0.6) and \
                          (row.get("prev_5m_body_mult", 0) > self.pa_body_mult_thresh) and \
                          (prev.get("close", 0) < prev.get("open", 0))
        if custom_mag_bear: structures.append("MagBear")
        
        if row.get("pa_channel_overshoot_revert_up"): structures.append("OvershootUp")
        if row.get("pa_channel_overshoot_revert_down"): structures.append("OvershootDown")
        if row.get("pa_wedge_third_push_up"): structures.append("WedgeUp")
        if row.get("pa_wedge_third_push_down"): structures.append("WedgeDown")
        if row.get("pa_h_count") == 1: structures.append("H1")
        if row.get("pa_is_h2_setup"): structures.append("H2")
        if row.get("pa_l_count") == 1: structures.append("L1")
        if row.get("pa_is_l2_setup"): structures.append("L2")
        if row.get("pa_breakout_success_up"): structures.append("BreakoutUp")
        if row.get("pa_breakout_success_down"): structures.append("BreakoutDown")
        
        structure_str = ", ".join(structures) if structures else "None"
        
        return {
            "time_key": bar_time,
            "close": float(row.get("close", np.nan)),
            "cvd_sim": float(row.get("cvd_session", np.nan)),
            "cvd_real": cvd_real,
            "prev_5m_env_state": str(row.get("pa_env_state", "")),
            "prev_5m_ema_slope_ratio": float(row.get("prev_5m_ema_slope_ratio", np.nan)),
            "prev_5m_body_mult": float(row.get("prev_5m_body_mult", np.nan)),
            "cvd_pressure": float(row.get("cvd_pressure", np.nan)),
            "cvd_classic_long_ok": bool(row.get("cvd_classic_long_ok", False)),
            "cvd_classic_short_ok": bool(row.get("cvd_classic_short_ok", False)),
            "structure": structure_str,
            "row_idx": idx[0]
        }

    def _log_minute_status(self, bar_time: pd.Timestamp, snapshot: dict[str, object]) -> None:
        if not snapshot: return
        close_px = snapshot["close"]
        state = snapshot["prev_5m_env_state"]
        cvd_real = snapshot["cvd_real"]
        structure = snapshot["structure"]
        
        pos_str = "flat"
        if self.position:
            side = "long" if self.position.direction == 1 else "short"
            pos_str = f"{side}@{self.position.entry_price:.2f} tp={self.position.take_profit:.2f} sl={self.position.stop_price:.2f} ts={pd.Timestamp(self.position.time_stop_deadline).strftime('%H:%M')}"
            
        _log_info(f"[BAR] {self.symbol} {bar_time.strftime('%Y-%m-%d %H:%M')} close={close_px:.2f} | 5mState={state} | Struct={structure} | CVD={cvd_real:.0f} | pos={pos_str}")

        changed = False
        if self.last_reported_5m_state is not None and self.last_reported_5m_state != state:
            changed = True
        if self.last_reported_structure is not None and self.last_reported_structure != structure:
            changed = True

        if self.notifier and self.notifier.enabled:
            # 1. Send every 5 minutes
            if bar_time.minute % 5 == 0:
                msg = f"[{self.strategy_tag} STATUS] {self.symbol} {bar_time.strftime('%H:%M')} | State: {state} | Struct: {structure}"
                self.notifier.send_text(msg)
            # 2. Or send if there's a breakout/change intra-5m
            elif changed:
                msg = f"[{self.strategy_tag} CHANGE] {self.symbol} {bar_time.strftime('%H:%M')} | New State: {state} | New Struct: {structure}"
                self.notifier.send_text(msg)

        self.last_reported_5m_state = state
        self.last_reported_structure = structure

    def _check_entry(self, bar_time: pd.Timestamp, snapshot: dict[str, object], bar_event: dict[str, object]) -> None:
        if self.featured.empty: return
        if "row_idx" not in snapshot: return
        i = snapshot["row_idx"]
        
        # Extract necessary variables directly from self.featured
        curr_time = bar_time.time()
        # Avoid garbage time
        if time(11, 30) <= curr_time < time(13, 30):
            return
            
        # We need the current and previous bar to check triggers
        if i < 1: return
        
        row = self.featured.loc[i]
        prev = self.featured.loc[i-1]
        
        # 1. 5m Context Zone evaluation
        is_strong_bull_trend = row["prev_5m_ema_slope_ratio"] > self.pa_slope_thresh
        is_strong_bear_trend = row["prev_5m_ema_slope_ratio"] < -self.pa_slope_thresh
        
        custom_mag_bull = (row["prev_5m_low"] > row["prev_5m_ema20"]) and \
                          (row["prev_5m_body_ratio"] > 0.6) and \
                          (row["prev_5m_body_mult"] > self.pa_body_mult_thresh) and \
                          (prev["close"] > prev["open"])
                          
        custom_mag_bear = (row["prev_5m_high"] < row["prev_5m_ema20"]) and \
                          (row["prev_5m_body_ratio"] > 0.6) and \
                          (row["prev_5m_body_mult"] > self.pa_body_mult_thresh) and \
                          (prev["close"] < prev["open"])
                          
        fade_short = (is_strong_bull_trend and custom_mag_bull) or row["pa_channel_overshoot_revert_up"] or row["pa_wedge_third_push_up"]
        fade_long = (is_strong_bear_trend and custom_mag_bear) or row["pa_channel_overshoot_revert_down"] or row["pa_wedge_third_push_down"]
        
        trend_long = (is_strong_bull_trend and (row["pa_h_count"] == 1 or row["pa_is_h2_setup"])) or row["pa_breakout_success_up"]
        trend_short = (is_strong_bear_trend and (row["pa_l_count"] == 1 or row["pa_is_l2_setup"])) or row["pa_breakout_success_down"]
        
        zone_fade_long = fade_long or trend_long
        zone_fade_short = fade_short or trend_short
        if zone_fade_long and zone_fade_short:
            return
            
        snapshot["zone_fade_long"] = zone_fade_long
        snapshot["zone_fade_short"] = zone_fade_short
        
        setup_long_price = row["prev_5m_high"] + 0.01
        setup_short_price = row["prev_5m_low"] - 0.01
        
        # Calculate stops
        setup_long_sl = row["prev_5m_high"] + 0.01 - self.pa_sl_atr * row["prev_5m_atr"]
        sl_cand = row["pa_stop_long"]
        if np.isfinite(sl_cand) and sl_cand < setup_long_price:
            sl_cand = sl_cand - 0.1 * row["prev_5m_atr"]
            if (setup_long_price - sl_cand) >= self.pa_sl_atr * row["prev_5m_atr"]:
                setup_long_sl = setup_long_price - self.pa_sl_atr * row["prev_5m_atr"]
            else:
                setup_long_sl = sl_cand
                
        setup_short_sl = row["prev_5m_low"] - 0.01 + self.pa_sl_atr * row["prev_5m_atr"]
        sl_cand = row["pa_stop_short"]
        if np.isfinite(sl_cand) and sl_cand > setup_short_price:
            sl_cand = sl_cand + 0.1 * row["prev_5m_atr"]
            if (sl_cand - setup_short_price) >= self.pa_sl_atr * row["prev_5m_atr"]:
                setup_short_sl = setup_short_price + self.pa_sl_atr * row["prev_5m_atr"]
            else:
                setup_short_sl = sl_cand
                
        setup_long_tp = setup_long_price + self.pa_tp_atr * row["prev_5m_atr"]
        setup_short_tp = setup_short_price - self.pa_tp_atr * row["prev_5m_atr"]

        direction = 0
        entry_price = np.nan
        stop_price = np.nan
        take_profit = np.nan
        
        if zone_fade_long and row["high"] > setup_long_price:
            # CVD Filter
            cvd_ok = True
            if self.pa_use_cvd_filter:
                recent_bull_div = not all(self.featured.loc[max(0, i-3):i, "cvd_classic_short_ok"])
                cvd_ok = recent_bull_div or row["cvd_pressure"] > 0
            if cvd_ok:
                direction = 1
                entry_price = max(row["open"], setup_long_price)
                stop_price = setup_long_sl
                take_profit = setup_long_tp
                
        elif zone_fade_short and row["low"] < setup_short_price:
            # CVD Filter
            cvd_ok = True
            if self.pa_use_cvd_filter:
                recent_bear_div = not all(self.featured.loc[max(0, i-3):i, "cvd_classic_long_ok"])
                cvd_ok = recent_bear_div or row["cvd_pressure"] < 0
            if cvd_ok:
                direction = -1
                entry_price = min(row["open"], setup_short_price)
                stop_price = setup_short_sl
                take_profit = setup_short_tp

        if direction != 0:
            deadline = bar_time + timedelta(minutes=self.pa_time_limit_mins)
            self.position = PositionState(
                symbol=self.symbol, direction=direction,
                entry_price=entry_price, entry_time=bar_time,
                stop_price=stop_price, take_profit=take_profit,
                time_stop_deadline=deadline,
            )
            side = "long" if direction == 1 else "short"
            bar_event.update({
                "entry_event": True, "entry_side": side, "entry_price": entry_price,
                "entry_sl": stop_price, "entry_tp": take_profit, "entry_deadline": str(deadline)
            })
            
            side_str = "LONG" if direction == 1 else "SHORT"
            color = _C.GREEN if direction == 1 else _C.RED
            _log_signal(f"{color}{_C.BOLD}[{self.strategy_tag} ENTRY] {self.symbol} {side_str}{_C.RESET} @ {entry_price:.2f} | TP {take_profit:.2f} | SL {stop_price:.2f} | TimeStop {deadline.strftime('%H:%M')}")
            
            if self.notifier and self.notifier.enabled:
                self.notifier.send_text(f"[{self.strategy_tag} ENTRY] {self.symbol} {side_str} | price={entry_price:.2f} | tp={take_profit:.2f} sl={stop_price:.2f} | time_stop={deadline.strftime('%Y-%m-%d %H:%M:%S')}")

    def _check_exit(self, bar: pd.Series, bar_time: pd.Timestamp, snapshot: dict[str, object], bar_event: dict[str, object]) -> None:
        if not self.position: return
        pos = self.position
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        
        idx = self.featured.index[self.featured["time_key"] == bar_time]
        if len(idx) == 0: return
        row = self.featured.loc[idx[0]]
        
        held_minutes = max((bar_time - pos.entry_time).total_seconds() / 60, 0)
        exit_price, exit_reason = None, None
        
        if pos.direction == 1:
            if low <= pos.stop_price: exit_price, exit_reason = pos.stop_price, "stop_loss"
            elif high >= pos.take_profit: exit_price, exit_reason = pos.take_profit, "take_profit"
            elif bar_time >= pos.time_stop_deadline: exit_price, exit_reason = close, "time_stop"
            else:
                mfe = high - pos.entry_price
                if mfe > 1.0 * row["prev_5m_atr"]:
                    new_sl_ema = row["prev_5m_ema20"] - 0.2 * row["prev_5m_atr"]
                    new_sl_mfe = pos.entry_price + mfe * 0.5
                    new_sl = max(new_sl_ema, new_sl_mfe)
                    if new_sl > pos.stop_price: pos.stop_price = new_sl
        else:
            if high >= pos.stop_price: exit_price, exit_reason = pos.stop_price, "stop_loss"
            elif low <= pos.take_profit: exit_price, exit_reason = pos.take_profit, "take_profit"
            elif bar_time >= pos.time_stop_deadline: exit_price, exit_reason = close, "time_stop"
            else:
                mfe = pos.entry_price - low
                if mfe > 1.0 * row["prev_5m_atr"]:
                    new_sl_ema = row["prev_5m_ema20"] + 0.2 * row["prev_5m_atr"]
                    new_sl_mfe = pos.entry_price - mfe * 0.5
                    new_sl = min(new_sl_ema, new_sl_mfe)
                    if new_sl < pos.stop_price: pos.stop_price = new_sl
                    
        if exit_reason:
            self._close_position(exit_price, bar_time, held_minutes, exit_reason, snapshot, bar_event)

    def _append_minute_signal_row(self, bar, bar_time, snapshot, bar_event) -> None:
        if not snapshot: return
        pos = self.position
        pos_side = "flat" if not pos else ("long" if pos.direction == 1 else "short")
        pos_entry = np.nan if not pos else pos.entry_price
        pos_stop = np.nan if not pos else pos.stop_price
        pos_tp = np.nan if not pos else pos.take_profit
        
        with open(self.minute_signal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                bar_time.strftime("%Y-%m-%d %H:%M:%S"), self.symbol,
                bar.get("open", np.nan), bar.get("high", np.nan), bar.get("low", np.nan), bar.get("close", np.nan), bar.get("volume", np.nan),
                snapshot.get("cvd_sim", np.nan), snapshot.get("cvd_real", np.nan),
                snapshot.get("prev_5m_env_state", ""), snapshot.get("prev_5m_ema_slope_ratio", np.nan),
                snapshot.get("prev_5m_body_mult", np.nan), snapshot.get("cvd_pressure", np.nan),
                snapshot.get("cvd_classic_long_ok", False), snapshot.get("cvd_classic_short_ok", False),
                snapshot.get("zone_fade_long", False), snapshot.get("zone_fade_short", False),
                bar_event.get("entry_event", False), bar_event.get("entry_side", ""), bar_event.get("entry_price", np.nan),
                bar_event.get("entry_sl", np.nan), bar_event.get("entry_tp", np.nan), bar_event.get("entry_deadline", ""),
                bar_event.get("exit_event", False), bar_event.get("exit_side", ""), bar_event.get("exit_reason", ""),
                bar_event.get("exit_price", np.nan), bar_event.get("exit_trade_return", np.nan),
                pos_side, pos_entry, pos_stop, pos_tp
            ])
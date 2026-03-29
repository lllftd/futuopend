"""
Real-time signal monitoring + performance deviation tracking for v3 strategy.

Usage:
    # Live mode (requires Futu OpenD running)
    python monitor.py

    # Custom Futu host/port
    python monitor.py --host 127.0.0.1 --port 11111
"""

from __future__ import annotations

import argparse
import csv
import time as time_mod
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import V3_BENCHMARKS, V3_BEST_PARAMS
from optimize_ce_zlsma_kama_rule import (
    RuleParams,
    apply_ce_features,
    build_base_features,
    build_entry_signals,
    build_exit_prices,
    session_entry_allowed,
)

try:
    from futu import KLType, OpenQuoteContext, RET_OK, SubType
except ImportError:
    OpenQuoteContext = None
    KLType = None
    RET_OK = None
    SubType = None

# Keep runtime output concise for long-running monitor.
warnings.filterwarnings(
    "ignore",
    message=".*Downcasting object dtype arrays on \\.fillna.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Console colors (ANSI — works on Windows 10+ and all modern terminals)
# ---------------------------------------------------------------------------

class _C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


MONITOR_DIR = Path("results") / "monitor"
CHARTS_DIR = MONITOR_DIR / "charts"
MARKET_DIR = MONITOR_DIR / "market"
SIGNALS_DIR = MONITOR_DIR / "signals"
WARMUP_DAYS = 5
POLL_INTERVAL_SECONDS = 30
MAX_BARS_KEPT = 5000
PERFORMANCE_REVIEW_EVERY_N_TRADES = 20
MONITOR_SYMBOLS = ("SPY", "QQQ")
FUTU_SYMBOLS = {symbol: f"US.{symbol}" for symbol in MONITOR_SYMBOLS}


def _date_tag(d: date) -> str:
    return d.strftime("%Y%m%d")


def _day_folder_name(d: date) -> str:
    return d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    symbol: str
    direction: int          # 1 = long, -1 = short
    entry_price: float
    entry_time: pd.Timestamp
    stop_price: float
    take_profit: float
    time_stop_deadline: pd.Timestamp
    remaining_weight: float = 1.0
    partial_exit_triggered: bool = False
    max_progress: float = 0.0


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    trade_return: float
    holding_minutes: float
    exit_reason: str
    strategy: str


class PerformanceTracker:
    def __init__(self, symbol: str, benchmark: dict[str, float]):
        self.symbol = symbol
        self.benchmark = benchmark
        self.trades: list[TradeRecord] = []
        self.equity = 1.0
        self.peak_equity = 1.0

    def record_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        self.equity *= 1.0 + trade.trade_return
        self.peak_equity = max(self.peak_equity, self.equity)

    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0

    def rolling_win_rate(self, n: int = 50) -> float:
        recent = self.trades[-n:]
        if not recent:
            return float("nan")
        return sum(1 for t in recent if t.trade_return > 0) / len(recent)

    def rolling_avg_return(self, n: int = 50) -> float:
        recent = self.trades[-n:]
        if not recent:
            return float("nan")
        return sum(t.trade_return for t in recent) / len(recent)

    def check_deviation(self) -> list[str]:
        warnings: list[str] = []
        if len(self.trades) < 10:
            return warnings

        wr = self.rolling_win_rate()
        bench_wr = self.benchmark.get("win_rate", 0)
        if not np.isnan(wr) and bench_wr > 0 and wr < bench_wr - 0.10:
            warnings.append(
                f"Win rate {wr:.1%} is {(bench_wr - wr):.1%} below backtest ({bench_wr:.1%})"
            )

        dd = self.current_drawdown()
        bench_dd = self.benchmark.get("max_drawdown", 0)
        if bench_dd < 0 and dd < bench_dd * 1.5:
            warnings.append(
                f"Drawdown {dd:.2%} exceeds 1.5x backtest max ({bench_dd:.2%})"
            )

        avg_hold = np.mean([t.holding_minutes for t in self.trades[-50:]])
        bench_hold = self.benchmark.get("avg_holding_minutes", 0)
        if bench_hold > 0 and avg_hold > bench_hold * 2.0:
            warnings.append(
                f"Avg hold {avg_hold:.1f}min is 2x+ backtest ({bench_hold:.1f}min)"
            )

        return warnings

    def summary_line(self) -> str:
        n = len(self.trades)
        if n == 0:
            return f"{self.symbol}: no trades yet"
        wr = self.rolling_win_rate()
        dd = self.current_drawdown()
        total = self.equity - 1.0
        return (
            f"{self.symbol}: {n} trades | "
            f"equity {self.equity:.4f} ({total:+.2%}) | "
            f"win {wr:.1%} | dd {dd:.2%}"
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "trade_return": t.trade_return,
                "holding_minutes": t.holding_minutes,
                "exit_reason": t.exit_reason,
                "strategy": t.strategy,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Symbol monitor — per-symbol state
# ---------------------------------------------------------------------------

class SymbolMonitor:
    def __init__(
        self,
        symbol: str,
        params: RuleParams,
        benchmark: dict[str, float],
        run_date: date,
    ):
        self.symbol = symbol
        self.params = params
        self.benchmark = benchmark
        self.bars: pd.DataFrame = pd.DataFrame()
        self.featured: pd.DataFrame = pd.DataFrame()
        self.position: Optional[PositionState] = None
        self.last_processed_time: Optional[pd.Timestamp] = None
        self.run_date = run_date
        self.daily_output_dir = MONITOR_DIR / "daily" / _day_folder_name(run_date)
        self.daily_output_dir.mkdir(parents=True, exist_ok=True)
        self.minute_signal_path = self.daily_output_dir / f"{symbol.lower()}_minute_signals.csv"
        self.tracker: Optional[PerformanceTracker] = PerformanceTracker(symbol, benchmark)
        self._ticker_last_seq: Optional[int] = None
        self._ticker_last_time: Optional[pd.Timestamp] = None
        self.real_cvd_value: float = 0.0
        self.real_cvd_minute_value: dict[pd.Timestamp, float] = {}
        self._ensure_minute_signal_header()

    def _minute_signal_columns(self) -> list[str]:
        return [
            "time_key",
            "symbol",
            "ce_length",
            "ce_multiplier",
            "trend_mode",
            "session_filter",
            "exit_model",
            "tp_atr_multiple",
            "tp_pct",
            "sl_pct",
            "time_stop_minutes",
            "force_time_stop",
            "zlsma_length",
            "kama_er_length",
            "kama_fast_length",
            "kama_slow_length",
            "pa_require_signal_bar",
            "pa_use_pa_stops",
            "pa_regime_filter",
            "pa_position_sizing_mode",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cvd_sim",
            "cvd_real",
            "session_allowed",
            "atr_valid",
            "long_entry_signal",
            "short_entry_signal",
            "long_exit_cond",
            "short_exit_cond",
            "entry_event",
            "entry_side",
            "entry_reason",
            "entry_price",
            "entry_tp",
            "entry_sl",
            "entry_deadline",
            "entry_sig_long_entry",
            "entry_sig_short_entry",
            "entry_sig_long_exit",
            "entry_sig_short_exit",
            "exit_event",
            "exit_side",
            "exit_reason",
            "exit_price",
            "exit_entry_time",
            "exit_entry_price",
            "exit_trade_return",
            "exit_holding_minutes",
            "exit_sig_long_entry",
            "exit_sig_short_entry",
            "exit_sig_long_exit",
            "exit_sig_short_exit",
            "position_side_after_bar",
            "position_entry_price",
            "position_stop_price",
            "position_take_profit",
            "position_deadline",
        ]

    def _ensure_minute_signal_header(self) -> None:
        columns = self._minute_signal_columns()
        if self.minute_signal_path.exists():
            try:
                with open(self.minute_signal_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                existing = first_line.split(",") if first_line else []
                if existing == columns:
                    return
            except OSError:
                pass
        with open(self.minute_signal_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def warm_up(self, historical_bars: pd.DataFrame) -> None:
        self.bars = historical_bars.copy()
        self._rebuild_features()
        if not self.bars.empty:
            self.last_processed_time = self.bars["time_key"].iloc[-1]
        _log_info(f"{self.symbol} warmed up with {len(self.bars)} bars, "
                  f"features: {len(self.featured)} rows")

    def _rebuild_features(self) -> None:
        if self.bars.empty:
            return
        base = build_base_features(
            self.bars,
            zlsma_length=self.params.zlsma_length,
            zlsma_offset=self.params.zlsma_offset,
            kama_er_length=self.params.kama_er_length,
            kama_fast_length=self.params.kama_fast_length,
            kama_slow_length=self.params.kama_slow_length,
            atr_percentile_lookback=self.params.atr_percentile_lookback,
            pseudo_cvd_method=self.params.pseudo_cvd_method,
            cvd_lookback=self.params.cvd_lookback,
            cvd_slope_lookback=self.params.cvd_slope_lookback,
        )
        self.featured = apply_ce_features(base, self.params.ce_length, self.params.ce_multiplier)

    def ingest_ticker(self, ticker_df: pd.DataFrame) -> None:
        if ticker_df is None or ticker_df.empty:
            return

        df = ticker_df.copy()
        time_col = "time" if "time" in df.columns else "time_key" if "time_key" in df.columns else None
        if time_col is None:
            return

        if "ticker_direction" not in df.columns or "volume" not in df.columns:
            return

        ts = pd.to_datetime(df[time_col], errors="coerce")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        if ts.dt.date.nunique() <= 1 and ts.dt.year.max() < 2000:
            today_prefix = datetime.now().strftime("%Y-%m-%d")
            ts = pd.to_datetime(today_prefix + " " + df[time_col].astype(str), errors="coerce")
        df["_ts"] = ts
        df = df.dropna(subset=["_ts"]).copy()
        if df.empty:
            return

        # Deduplicate on sequence when available.
        if "sequence" in df.columns:
            seq = pd.to_numeric(df["sequence"], errors="coerce")
            df["_seq"] = seq
            if self._ticker_last_seq is not None:
                df = df.loc[df["_seq"] > self._ticker_last_seq].copy()
            if not df.empty:
                self._ticker_last_seq = int(df["_seq"].max())
        elif self._ticker_last_time is not None:
            df = df.loc[df["_ts"] > self._ticker_last_time].copy()

        if df.empty:
            return
        self._ticker_last_time = df["_ts"].max()

        direction = df["ticker_direction"].astype(str).str.upper()
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        sign = np.where(direction.str.contains("BUY"), 1.0, np.where(direction.str.contains("SELL"), -1.0, 0.0))
        df["_signed_volume"] = sign * vol
        df["_minute"] = df["_ts"].dt.floor("min")

        grouped = df.groupby("_minute")["_signed_volume"].sum().sort_index()
        for minute_ts, delta in grouped.items():
            self.real_cvd_value += float(delta)
            self.real_cvd_minute_value[pd.Timestamp(minute_ts)] = self.real_cvd_value

    def process_new_bars(self, new_bars: pd.DataFrame, ticker_df: Optional[pd.DataFrame] = None) -> None:
        if new_bars.empty:
            return

        if ticker_df is not None and not ticker_df.empty:
            self.ingest_ticker(ticker_df)

        self.bars = pd.concat([self.bars, new_bars]).drop_duplicates(
            subset=["time_key"], keep="last",
        ).sort_values("time_key").reset_index(drop=True)

        if len(self.bars) > MAX_BARS_KEPT:
            self.bars = self.bars.tail(MAX_BARS_KEPT).reset_index(drop=True)

        self._rebuild_features()

        for _, bar in new_bars.iterrows():
            bar_time = pd.Timestamp(bar["time_key"])
            if self.last_processed_time is not None and bar_time <= self.last_processed_time:
                continue
            self._on_bar(bar, bar_time)
            self.last_processed_time = bar_time

    def _on_bar(self, bar: pd.Series, bar_time: pd.Timestamp) -> None:
        snapshot = self._build_signal_snapshot(bar_time)
        bar_event: dict[str, object] = {}

        is_session_start = bar_time.time() <= time(9, 31)
        is_session_end = bar_time.time() >= time(15, 59)

        if is_session_start and self.position is not None:
            self._force_close(bar, bar_time, "new_session_reset", snapshot, bar_event)

        if self.position is not None:
            self._check_exit(bar, bar_time, snapshot, bar_event)

        if self.position is None and not is_session_end:
            self._check_entry(bar_time, snapshot, bar_event)

        if is_session_end and self.position is not None:
            self._force_close(bar, bar_time, "end_of_day", snapshot, bar_event)

        self._append_minute_signal_row(bar, bar_time, snapshot, bar_event)

    def _build_signal_snapshot(self, bar_time: pd.Timestamp) -> dict[str, object]:
        if self.featured.empty:
            return {}

        idx = self.featured.index[self.featured["time_key"] == bar_time]
        if len(idx) == 0:
            return {}
        row_idx = idx[0]
        pos_in_df = self.featured.index.get_loc(row_idx)
        row = self.featured.loc[row_idx]

        long_sig, short_sig = build_entry_signals(self.featured, self.params)
        allowed = session_entry_allowed(self.featured["time_key"], self.params.session_filter)

        atr_val = row.get("ce_atr", np.nan)
        atr_valid = bool(np.isfinite(atr_val) and atr_val > 0)
        session_allowed = bool(allowed.iloc[pos_in_df])
        long_entry = bool(long_sig.iloc[pos_in_df]) and atr_valid and session_allowed
        short_entry = bool(short_sig.iloc[pos_in_df]) and atr_valid and session_allowed

        ce_buy = bool(row.get("ce_buy_signal", False))
        ce_sell = bool(row.get("ce_sell_signal", False))
        minute_key = pd.Timestamp(bar_time).floor("min")
        cvd_sim = float(row.get("cvd_session", np.nan))
        cvd_real = float(self.real_cvd_minute_value.get(minute_key, self.real_cvd_value))

        return {
            "time_key": bar_time,
            "symbol": self.symbol,
            "close": float(row.get("close", np.nan)),
            "cvd_sim": cvd_sim,
            "cvd_real": cvd_real,
            "session_allowed": session_allowed,
            "atr_valid": atr_valid,
            "long_entry_signal": long_entry,
            "short_entry_signal": short_entry,
            "long_exit_cond": ce_sell,
            "short_exit_cond": ce_buy,
        }

    def _append_minute_signal_row(
        self,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        snapshot: dict[str, object],
        bar_event: dict[str, object],
    ) -> None:
        if not snapshot:
            return
        pos = self.position
        if pos is None:
            pos_side = "flat"
            pos_entry = np.nan
            pos_stop = np.nan
            pos_tp = np.nan
            pos_deadline = ""
        else:
            pos_side = "long" if pos.direction == 1 else "short"
            pos_entry = float(pos.entry_price)
            pos_stop = float(pos.stop_price)
            pos_tp = float(pos.take_profit)
            pos_deadline = pd.Timestamp(pos.time_stop_deadline).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.minute_signal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    pd.Timestamp(bar_time).strftime("%Y-%m-%d %H:%M:%S"),
                    self.symbol,
                    int(self.params.ce_length),
                    float(self.params.ce_multiplier),
                    str(self.params.trend_mode),
                    str(self.params.session_filter),
                    str(self.params.exit_model),
                    float(self.params.tp_atr_multiple),
                    float(self.params.tp_pct),
                    float(self.params.sl_pct),
                    int(self.params.time_stop_minutes),
                    int(bool(self.params.force_time_stop)),
                    int(self.params.zlsma_length),
                    int(self.params.kama_er_length),
                    int(self.params.kama_fast_length),
                    int(self.params.kama_slow_length),
                    int(bool(self.params.pa_require_signal_bar)),
                    int(bool(self.params.pa_use_pa_stops)),
                    int(bool(self.params.pa_regime_filter)),
                    str(self.params.pa_position_sizing_mode),
                    float(bar.get("open", np.nan)),
                    float(bar.get("high", np.nan)),
                    float(bar.get("low", np.nan)),
                    float(bar.get("close", np.nan)),
                    float(bar.get("volume", np.nan)),
                    float(snapshot.get("cvd_sim", np.nan)),
                    float(snapshot.get("cvd_real", np.nan)),
                    int(bool(snapshot.get("session_allowed", False))),
                    int(bool(snapshot.get("atr_valid", False))),
                    int(bool(snapshot.get("long_entry_signal", False))),
                    int(bool(snapshot.get("short_entry_signal", False))),
                    int(bool(snapshot.get("long_exit_cond", False))),
                    int(bool(snapshot.get("short_exit_cond", False))),
                    int(bool(bar_event.get("entry_event", False))),
                    str(bar_event.get("entry_side", "")),
                    str(bar_event.get("entry_reason", "")),
                    float(bar_event.get("entry_price", np.nan)),
                    float(bar_event.get("entry_tp", np.nan)),
                    float(bar_event.get("entry_sl", np.nan)),
                    str(bar_event.get("entry_deadline", "")),
                    int(bool(bar_event.get("entry_sig_long_entry", False))),
                    int(bool(bar_event.get("entry_sig_short_entry", False))),
                    int(bool(bar_event.get("entry_sig_long_exit", False))),
                    int(bool(bar_event.get("entry_sig_short_exit", False))),
                    int(bool(bar_event.get("exit_event", False))),
                    str(bar_event.get("exit_side", "")),
                    str(bar_event.get("exit_reason", "")),
                    float(bar_event.get("exit_price", np.nan)),
                    str(bar_event.get("exit_entry_time", "")),
                    float(bar_event.get("exit_entry_price", np.nan)),
                    float(bar_event.get("exit_trade_return", np.nan)),
                    float(bar_event.get("exit_holding_minutes", np.nan)),
                    int(bool(bar_event.get("exit_sig_long_entry", False))),
                    int(bool(bar_event.get("exit_sig_short_entry", False))),
                    int(bool(bar_event.get("exit_sig_long_exit", False))),
                    int(bool(bar_event.get("exit_sig_short_exit", False))),
                    pos_side,
                    pos_entry,
                    pos_stop,
                    pos_tp,
                    pos_deadline,
                ]
            )

    def _check_entry(self, bar_time: pd.Timestamp, snapshot: dict[str, object], bar_event: dict[str, object]) -> None:
        if self.featured.empty:
            return

        idx = self.featured.index[self.featured["time_key"] == bar_time]
        if len(idx) == 0:
            return
        row_idx = idx[0]
        pos_in_df = self.featured.index.get_loc(row_idx)

        long_sig, short_sig = build_entry_signals(self.featured, self.params)
        allowed = session_entry_allowed(self.featured["time_key"], self.params.session_filter)

        if not allowed.iloc[pos_in_df]:
            return

        atr_val = self.featured.loc[row_idx, "ce_atr"]
        if not np.isfinite(atr_val) or atr_val <= 0:
            return

        direction = 0
        if long_sig.iloc[pos_in_df]:
            direction = 1
        elif short_sig.iloc[pos_in_df]:
            direction = -1

        if direction == 0:
            return

        entry_price = float(self.featured.loc[row_idx, "close"])
        stop_price, take_profit = build_exit_prices(entry_price, atr_val, direction, self.params)

        remaining_weight = 1.0
        if self.params.pa_position_sizing_mode == "risk_based":
            default_sl = self.params.sl_atr_multiple * atr_val if self.params.sl_atr_multiple > 0 else atr_val
            actual_sl = abs(entry_price - stop_price)
            if default_sl > 0 and actual_sl > 2.0 * default_sl:
                remaining_weight = 0.5

        deadline = bar_time + timedelta(minutes=self.params.time_stop_minutes)

        self.position = PositionState(
            symbol=self.symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=bar_time,
            stop_price=stop_price,
            take_profit=take_profit,
            time_stop_deadline=deadline,
            remaining_weight=remaining_weight,
        )
        side = "long" if direction == 1 else "short"
        bar_event.update(
            {
                "entry_event": True,
                "entry_side": side,
                "entry_reason": "entry_signal",
                "entry_price": entry_price,
                "entry_tp": float(take_profit),
                "entry_sl": float(stop_price),
                "entry_deadline": deadline.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_sig_long_entry": bool(snapshot.get("long_entry_signal", False)),
                "entry_sig_short_entry": bool(snapshot.get("short_entry_signal", False)),
                "entry_sig_long_exit": bool(snapshot.get("long_exit_cond", False)),
                "entry_sig_short_exit": bool(snapshot.get("short_exit_cond", False)),
            }
        )

        side = "LONG" if direction == 1 else "SHORT"
        color = _C.GREEN if direction == 1 else _C.RED
        _log_signal(
            f"{color}{_C.BOLD}[ENTRY] {self.symbol} {side}{_C.RESET} "
            f"@ {entry_price:.2f} | "
            f"TP {take_profit:.2f} | SL {stop_price:.2f} | "
            f"TimeStop {deadline.strftime('%H:%M')}"
        )

    def _check_exit(
        self,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        snapshot: dict[str, object],
        bar_event: dict[str, object],
    ) -> None:
        pos = self.position
        if pos is None:
            return

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        held_minutes = max((bar_time - pos.entry_time).total_seconds() / 60, 0)
        target_dist = abs(pos.take_profit - pos.entry_price)
        if target_dist > 0:
            if pos.direction == 1:
                progress = max((high - pos.entry_price) / target_dist, 0)
            else:
                progress = max((pos.entry_price - low) / target_dist, 0)
            pos.max_progress = max(pos.max_progress, progress)

        exit_price = None
        exit_reason = None

        if pos.direction == 1:
            if low <= pos.stop_price:
                exit_price, exit_reason = pos.stop_price, "stop_loss"
            elif high >= pos.take_profit:
                exit_price, exit_reason = pos.take_profit, "take_profit"
            elif bar_time >= pos.time_stop_deadline and (
                self.params.force_time_stop
                or pos.max_progress < self.params.time_progress_threshold
            ):
                exit_price, exit_reason = close, "time_stop"
        else:
            if high >= pos.stop_price:
                exit_price, exit_reason = pos.stop_price, "stop_loss"
            elif low <= pos.take_profit:
                exit_price, exit_reason = pos.take_profit, "take_profit"
            elif bar_time >= pos.time_stop_deadline and (
                self.params.force_time_stop
                or pos.max_progress < self.params.time_progress_threshold
            ):
                exit_price, exit_reason = close, "time_stop"

        if exit_reason is not None:
            self._close_position(exit_price, bar_time, held_minutes, exit_reason, snapshot, bar_event)

    def _force_close(
        self,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        reason: str,
        snapshot: dict[str, object],
        bar_event: dict[str, object],
    ) -> None:
        if self.position is None:
            return
        close = float(bar["close"])
        held = max((bar_time - self.position.entry_time).total_seconds() / 60, 0)
        self._close_position(close, bar_time, held, reason, snapshot, bar_event)

    def _close_position(
        self, exit_price: float, exit_time: pd.Timestamp,
        held_minutes: float, reason: str,
        snapshot: dict[str, object],
        bar_event: dict[str, object],
    ) -> None:
        pos = self.position
        trade_return = pos.remaining_weight * pos.direction * (exit_price / pos.entry_price - 1.0)
        side_str = "long" if pos.direction == 1 else "short"

        record = TradeRecord(
            symbol=self.symbol,
            side=side_str,
            entry_time=str(pos.entry_time),
            exit_time=str(exit_time),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            trade_return=trade_return,
            holding_minutes=held_minutes,
            exit_reason=reason,
            strategy=self.params.strategy_name,
        )
        if self.tracker is not None:
            self.tracker.record_trade(record)
        bar_event.update(
            {
                "exit_event": True,
                "exit_side": side_str,
                "exit_reason": reason,
                "exit_price": float(exit_price),
                "exit_entry_time": str(pos.entry_time),
                "exit_entry_price": float(pos.entry_price),
                "exit_trade_return": float(trade_return),
                "exit_holding_minutes": float(held_minutes),
                "exit_sig_long_entry": bool(snapshot.get("long_entry_signal", False)),
                "exit_sig_short_entry": bool(snapshot.get("short_entry_signal", False)),
                "exit_sig_long_exit": bool(snapshot.get("long_exit_cond", False)),
                "exit_sig_short_exit": bool(snapshot.get("short_exit_cond", False)),
            }
        )

        pnl_color = _C.GREEN if trade_return > 0 else _C.RED
        _log_signal(
            f"{_C.YELLOW}{_C.BOLD}[EXIT]{_C.RESET} {self.symbol} {side_str} "
            f"@ {exit_price:.2f} | "
            f"{pnl_color}{trade_return:+.3%}{_C.RESET} | "
            f"{held_minutes:.0f}min | {reason}"
        )

        if self.tracker is not None and len(self.tracker.trades) % PERFORMANCE_REVIEW_EVERY_N_TRADES == 0:
            self._print_performance_review()

        self.position = None

    def _print_performance_review(self) -> None:
        _log_info(f"\n{'='*60}")
        _log_info(f"  Performance Review: {self.symbol} ({len(self.tracker.trades)} trades)")
        _log_info(f"  {self.tracker.summary_line()}")

        warnings = self.tracker.check_deviation()
        if warnings:
            for w in warnings:
                _log_warn(f"  DEVIATION: {w}")
        else:
            _log_info(f"  {_C.GREEN}Metrics within expected range{_C.RESET}")
        _log_info(f"{'='*60}\n")

    def end_of_day_summary(self, trading_day: Optional[date] = None) -> None:
        if trading_day is None:
            trading_day = datetime.now().date()
        _log_info(f"\n{'-'*50}")
        _log_info(f"  Daily Summary: {self.symbol}")
        _log_info(f"  Minute+Signal CSV: {self.minute_signal_path.as_posix()}")
        self._save_intraday_review_chart(self.daily_output_dir, trading_day)
        if self.tracker is not None:
            _log_info(f"  {self.tracker.summary_line()}")
            warnings = self.tracker.check_deviation()
            for w in warnings:
                _log_warn(f"  {w}")
        _log_info(f"{'-'*50}")

    def _save_equity_curve_chart(self, chart_dir: Path) -> None:
        trades = self.tracker.to_frame()
        if trades.empty:
            return

        vals = trades["trade_return"].astype(float).to_numpy()
        equity = np.cumprod(1.0 + vals)
        x = np.arange(1, len(equity) + 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, equity, color="#1f77b4", linewidth=1.4)
        ax.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
        ax.set_title(f"{self.symbol} Cumulative Equity (Run To Date)")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Equity")
        ax.grid(True, linestyle="--", alpha=0.25)

        chart_dir.mkdir(parents=True, exist_ok=True)
        out = chart_dir / f"{self.symbol.lower()}_equity_curve.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log_info(f"  Chart: {out.as_posix()}")

    def _save_daily_scatter_chart(self, chart_dir: Path, trading_day: date) -> None:
        trades = self.tracker.to_frame()
        if trades.empty:
            return
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
        daily = trades.loc[trades["exit_time"].dt.date == trading_day].copy()
        if daily.empty:
            return

        daily["trade_return_pct"] = pd.to_numeric(daily["trade_return"], errors="coerce") * 100.0
        daily["holding_minutes"] = pd.to_numeric(daily["holding_minutes"], errors="coerce")
        daily = daily.dropna(subset=["trade_return_pct", "holding_minutes"])
        if daily.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        long_mask = daily["side"].eq("long")
        short_mask = daily["side"].eq("short")
        ax.scatter(
            daily.loc[long_mask, "holding_minutes"],
            daily.loc[long_mask, "trade_return_pct"],
            s=26,
            alpha=0.5,
            color="#2CA02C",
            label="Long",
        )
        ax.scatter(
            daily.loc[short_mask, "holding_minutes"],
            daily.loc[short_mask, "trade_return_pct"],
            s=26,
            alpha=0.5,
            color="#D62728",
            label="Short",
        )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_title(f"{self.symbol} {trading_day.isoformat()} Holding vs Return")
        ax.set_xlabel("Holding Minutes")
        ax.set_ylabel("Trade Return (%)")
        ax.legend(loc="best")

        chart_dir.mkdir(parents=True, exist_ok=True)
        out = chart_dir / f"{self.symbol.lower()}_holding_return.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log_info(f"  Chart: {out.as_posix()}")

    def _save_intraday_review_chart(self, chart_dir: Path, trading_day: date) -> None:
        if not self.minute_signal_path.exists():
            return

        day_df = pd.read_csv(self.minute_signal_path, on_bad_lines="skip")
        if day_df.empty:
            return
        day_df["time_key"] = pd.to_datetime(day_df["time_key"], errors="coerce")
        day_df = day_df.dropna(subset=["time_key"]).sort_values("time_key").reset_index(drop=True)
        day_df = day_df.loc[day_df["time_key"].dt.date == trading_day].copy()
        if day_df.empty:
            return

        day_df["open"] = pd.to_numeric(day_df["open"], errors="coerce")
        day_df["high"] = pd.to_numeric(day_df["high"], errors="coerce")
        day_df["low"] = pd.to_numeric(day_df["low"], errors="coerce")
        day_df["close"] = pd.to_numeric(day_df["close"], errors="coerce")
        for c in [
            "long_entry_signal",
            "short_entry_signal",
            "long_exit_cond",
            "short_exit_cond",
            "entry_event",
            "exit_event",
        ]:
            day_df[c] = pd.to_numeric(day_df[c], errors="coerce").fillna(0).astype(int)
        for c in [
            "entry_price",
            "exit_price",
            "exit_trade_return",
        ]:
            if c in day_df.columns:
                day_df[c] = pd.to_numeric(day_df[c], errors="coerce")
        if "exit_entry_time" in day_df.columns:
            day_df["exit_entry_time"] = pd.to_datetime(day_df["exit_entry_time"], errors="coerce")

        # Build trade markers from event columns in minute_signals.csv.
        pseudo_trades: list[dict[str, object]] = []
        cum = 1.0
        eq_times: list[float] = [mdates.date2num(day_df["time_key"].iloc[0])]
        eq_vals: list[float] = [0.0]

        for _, r in day_df.iterrows():
            ts = pd.Timestamp(r["time_key"])
            if int(r.get("exit_event", 0)) == 1:
                entry_time = pd.Timestamp(r.get("exit_entry_time"))
                entry_price = float(r.get("exit_entry_price", np.nan))
                exit_price = float(r.get("exit_price", np.nan))
                if not np.isfinite(entry_price):
                    entry_price = float(r.get("entry_price", np.nan))
                if not np.isfinite(exit_price):
                    exit_price = float(r.get("close", np.nan))
                if pd.isna(entry_time):
                    entry_time = ts
                trade_ret = float(r.get("exit_trade_return", np.nan))
                if not np.isfinite(trade_ret):
                    side = str(r.get("exit_side", "")).lower()
                    if side == "short" and np.isfinite(entry_price) and np.isfinite(exit_price) and exit_price != 0:
                        trade_ret = entry_price / exit_price - 1.0
                    elif np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price != 0:
                        trade_ret = exit_price / entry_price - 1.0
                    else:
                        trade_ret = 0.0
                cum *= 1.0 + trade_ret
                pseudo_trades.append(
                    {
                        "side": str(r.get("exit_side", "")).lower(),
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "reason": str(r.get("exit_reason", "")),
                    }
                )
                eq_times.append(mdates.date2num(ts))
                eq_vals.append(cum - 1.0)

        if len(eq_times) == 1:
            eq_times.append(mdates.date2num(day_df["time_key"].iloc[-1]))
            eq_vals.append(eq_vals[-1])

        fig, (ax_k, ax_eq) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            height_ratios=[0.62, 0.38],
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )

        # --- K line ---
        t = pd.to_datetime(day_df["time_key"])
        x = mdates.date2num(t)
        width = float(np.median(np.diff(x))) * 0.65 if len(x) > 1 else 0.00025
        o = day_df["open"].astype(float).to_numpy()
        h = day_df["high"].astype(float).to_numpy()
        l_ = day_df["low"].astype(float).to_numpy()
        c = day_df["close"].astype(float).to_numpy()

        for i in range(len(day_df)):
            color = "#2ca02c" if c[i] >= o[i] else "#d62728"
            ax_k.plot([x[i], x[i]], [l_[i], h[i]], color=color, linewidth=0.7, solid_capstyle="round")
            bh = max(abs(c[i] - o[i]), (h[i] - l_[i]) * 0.02 if h[i] != l_[i] else 0.02)
            bottom = min(o[i], c[i])
            ax_k.add_patch(
                Rectangle(
                    (x[i] - width / 2.0, bottom),
                    width,
                    bh,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.3,
                    zorder=3,
                )
            )

        # --- Entry / Exit markers ---
        if pseudo_trades:
            for tr in pseudo_trades:
                ent = tr["entry_time"]
                ext = tr["exit_time"]
                side = str(tr["side"])
                ep = float(tr["entry_price"])
                xp = float(tr["exit_price"])
                x_e = mdates.date2num(ent)
                x_x = mdates.date2num(ext)
                reason = str(tr["reason"])

                ax_k.plot([x_e, x_x], [ep, xp], color="#555555", linewidth=0.85, linestyle="--", alpha=0.75, zorder=4)
                if side == "long":
                    ax_k.scatter([x_e], [ep], marker="^", s=55, c="#1f77b4", zorder=5, edgecolors="k", linewidths=0.4)
                else:
                    ax_k.scatter([x_e], [ep], marker="v", s=55, c="#ff7f0e", zorder=5, edgecolors="k", linewidths=0.4)
                ax_k.scatter([x_x], [xp], marker="s", s=36, c="#9467bd", zorder=5, edgecolors="k", linewidths=0.3)
                ax_k.annotate(
                    reason,
                    (x_x, xp),
                    textcoords="offset points",
                    xytext=(0, -11 if side == "long" else 10),
                    fontsize=7,
                    ha="center",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
                )

        ax_k.set_ylabel("Price")
        ax_k.set_title(f"{self.symbol} {trading_day.isoformat()} - Intraday Kline and Trade Signals")
        ax_k.grid(True, linestyle="--", alpha=0.25)

        ax_eq.step(eq_times, eq_vals, where="post", color="#d62728", linewidth=1.2)
        if len(eq_times) > 1:
            ax_eq.scatter(eq_times[1:], eq_vals[1:], color="#d62728", s=14, zorder=3)

        ax_eq.axhline(0.0, color="#999999", linewidth=0.6, linestyle="-")
        ax_eq.set_ylabel("Intraday Cumulative Return")
        ax_eq.set_xlabel("Time")
        ax_eq.grid(True, linestyle="--", alpha=0.25)

        # X-axis formatting
        ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

        chart_dir.mkdir(parents=True, exist_ok=True)
        out = chart_dir / f"{self.symbol.lower()}_intraday_review.png"
        fig.subplots_adjust(hspace=0.10)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log_info(f"  Chart: {out.as_posix()}")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_info(msg: str) -> None:
    print(f"{_C.DIM}[{_ts()}]{_C.RESET} {msg}", flush=True)


def _log_signal(msg: str) -> None:
    print(f"{_C.BOLD}[{_ts()}]{_C.RESET} {msg}", flush=True)


def _log_warn(msg: str) -> None:
    print(f"{_C.YELLOW}[{_ts()}] WARNING: {msg}{_C.RESET}", flush=True)


def _log_error(msg: str) -> None:
    print(f"{_C.RED}[{_ts()}] ERROR: {msg}{_C.RESET}", flush=True)


# ---------------------------------------------------------------------------
# Futu data fetching
# ---------------------------------------------------------------------------

def fetch_recent_history(
    ctx: "OpenQuoteContext",
    futu_symbol: str,
    days: int = WARMUP_DAYS,
) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days + 3)).strftime("%Y-%m-%d")

    all_frames: list[pd.DataFrame] = []
    page_key = None

    while True:
        ret, data, page_key = ctx.request_history_kline(
            code=futu_symbol,
            start=start_date,
            end=end_date,
            ktype=KLType.K_1M,
            max_count=1000,
            page_req_key=page_key,
        )
        if ret != RET_OK:
            raise RuntimeError(f"Failed to fetch {futu_symbol}: {data}")
        if data is None or data.empty:
            break
        all_frames.append(data.copy())
        if page_key is None:
            break

    if not all_frames:
        raise RuntimeError(f"No data returned for {futu_symbol}")

    df = pd.concat(all_frames, ignore_index=True)
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").drop_duplicates("time_key", keep="last").reset_index(drop=True)
    return df[["time_key", "open", "high", "low", "close", "volume"]].copy()


def fetch_latest_bars(
    ctx: "OpenQuoteContext",
    futu_symbol: str,
    count: int = 10,
) -> pd.DataFrame:
    ret, data = ctx.get_cur_kline(futu_symbol, count, KLType.K_1M)
    if ret != RET_OK:
        _log_error(f"get_cur_kline failed for {futu_symbol}: {data}")
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    data["time_key"] = pd.to_datetime(data["time_key"])
    return data[["time_key", "open", "high", "low", "close", "volume"]].copy()


def fetch_latest_tickers(
    ctx: "OpenQuoteContext",
    futu_symbol: str,
    count: int = 1000,
) -> pd.DataFrame:
    if not hasattr(ctx, "get_rt_ticker"):
        return pd.DataFrame()
    ret, data = ctx.get_rt_ticker(futu_symbol, num=count)
    if ret != RET_OK:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    return data.copy()


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_live(host: str, port: int) -> int:
    if OpenQuoteContext is None:
        _log_error("futu-api not installed. Run: pip install futu-api")
        return 1

    _log_info(f"Connecting to Futu OpenD at {host}:{port}")
    ctx = OpenQuoteContext(host=host, port=port)

    monitors: dict[str, SymbolMonitor] = {}
    ticker_enabled = False
    run_date = datetime.now().date()

    try:
        futu_codes = list(FUTU_SYMBOLS.values())
        ret, data = ctx.subscribe(futu_codes, [SubType.K_1M])
        if ret != RET_OK:
            _log_error(f"Subscription failed: {data}")
            return 1
        ret_t, data_t = ctx.subscribe(futu_codes, [SubType.TICKER])
        ticker_enabled = ret_t == RET_OK
        if not ticker_enabled:
            _log_warn("Ticker subscription unavailable; cvd_real will be empty/flat.")

        for symbol, futu_code in FUTU_SYMBOLS.items():
            params = V3_BEST_PARAMS[symbol]
            benchmark = V3_BENCHMARKS[symbol]
            mon = SymbolMonitor(
                symbol,
                params,
                benchmark,
                run_date=run_date,
            )

            history = fetch_recent_history(ctx, futu_code, days=WARMUP_DAYS)
            mon.warm_up(history)
            monitors[symbol] = mon

        _log_info(
            f"Ready | symbols={','.join(futu_codes)} | poll={POLL_INTERVAL_SECONDS}s | "
            f"session=before_1230 | ticker={'on' if ticker_enabled else 'off'}"
        )
        _log_info(f"Output: {MONITOR_DIR.as_posix()}/daily/YYYY-MM-DD/(spy|qqq)_*.{{csv,png}}")

        while True:
            for symbol, futu_code in FUTU_SYMBOLS.items():
                new_bars = fetch_latest_bars(ctx, futu_code, count=5)
                ticker_df = fetch_latest_tickers(ctx, futu_code, count=1000) if ticker_enabled else pd.DataFrame()
                if not new_bars.empty:
                    mon = monitors[symbol]
                    if mon.last_processed_time is not None:
                        new_bars = new_bars[new_bars["time_key"] > mon.last_processed_time]
                    if not new_bars.empty:
                        mon.process_new_bars(new_bars, ticker_df=ticker_df)

            now = datetime.now()
            if now.hour >= 16 and now.minute >= 5:
                _log_info("\nMarket closed — printing daily summaries")
                for mon in monitors.values():
                    mon.end_of_day_summary(trading_day=now.date())
                _log_info("Waiting for next session...\n")
                time_mod.sleep(3600)
            else:
                time_mod.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        _log_info("\nShutting down...")
        today = datetime.now().date()
        for mon in monitors.values():
            mon.end_of_day_summary(trading_day=today)
    finally:
        ctx.close()

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_param_summary() -> None:
    _log_info(f"{_C.CYAN}{'='*60}{_C.RESET}")
    _log_info(f"{_C.CYAN}  V3 Strategy Parameters{_C.RESET}")
    _log_info(f"{_C.CYAN}{'='*60}{_C.RESET}")
    for symbol in MONITOR_SYMBOLS:
        params = V3_BEST_PARAMS[symbol]
        bench = V3_BENCHMARKS[symbol]
        _log_info(
            f"  {_C.BOLD}{symbol}{_C.RESET}: "
            f"CE({params.ce_length},{params.ce_multiplier}) "
            f"{params.exit_model} "
            f"TP={params.tp_atr_multiple or params.tp_pct} "
            f"SL={params.sl_pct} "
            f"TS={params.time_stop_minutes}m "
            f"session={params.session_filter}"
        )
        _log_info(
            f"    ZLSMA={params.zlsma_length} KAMA={params.kama_er_length}/"
            f"{params.kama_fast_length}/{params.kama_slow_length} "
            f"sizing={params.pa_position_sizing_mode}"
        )
        _log_info(
            f"    Benchmark: sharpe={bench['sharpe']:.2f} "
            f"win={bench['win_rate']:.1%} "
            f"dd={bench['max_drawdown']:.2%} "
            f"hold={bench['avg_holding_minutes']:.1f}m"
        )
    _log_info(f"{_C.CYAN}{'='*60}{_C.RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 real-time monitor (minute market + signal logging)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Futu OpenD host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=11111,
        help="Futu OpenD port (default: 11111)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _log_info(f"{_C.BOLD}V3 Signal Monitor{_C.RESET}")
    _log_info("Mode: LIVE")
    _log_info("Signal engine: STATEFUL(position-aware, TP/SL/time-stop)")
    return run_live(
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main())

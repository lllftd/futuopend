from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class StraddleGreeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float


class StraddleSimulator:
    """Black-Scholes ATM straddle simulator over 1-minute bars."""

    def __init__(self, *, risk_free_rate: float = 0.04):
        self.r = float(risk_free_rate)

    def bs_call(self, s: float, k: float, t: float, sigma: float) -> float:
        s = max(float(s), 1e-9)
        k = max(float(k), 1e-9)
        sigma = max(float(sigma), 1e-6)
        if t <= 0.0:
            return max(s - k, 0.0)
        d1 = (np.log(s / k) + (self.r + 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return float(s * norm.cdf(d1) - k * np.exp(-self.r * t) * norm.cdf(d2))

    def bs_put(self, s: float, k: float, t: float, sigma: float) -> float:
        s = max(float(s), 1e-9)
        k = max(float(k), 1e-9)
        sigma = max(float(sigma), 1e-6)
        if t <= 0.0:
            return max(k - s, 0.0)
        d1 = (np.log(s / k) + (self.r + 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return float(k * np.exp(-self.r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))

    def straddle_price(self, s: float, k: float, t: float, sigma: float) -> float:
        return self.bs_call(s, k, t, sigma) + self.bs_put(s, k, t, sigma)

    def straddle_greeks(self, s: float, k: float, t: float, sigma: float) -> StraddleGreeks:
        s = max(float(s), 1e-9)
        k = max(float(k), 1e-9)
        sigma = max(float(sigma), 1e-6)
        if t <= 0.0:
            price = self.straddle_price(s, k, 0.0, sigma)
            return StraddleGreeks(price=price, delta=0.0, gamma=0.0, theta=0.0, vega=0.0)
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + (self.r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        call = s * norm.cdf(d1) - k * np.exp(-self.r * t) * norm.cdf(d2)
        put = k * np.exp(-self.r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        pdf = norm.pdf(d1)
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1.0
        gamma = pdf / max(s * sigma * sqrt_t, 1e-9)
        call_theta = (
            -(s * pdf * sigma) / (2.0 * sqrt_t)
            - self.r * k * np.exp(-self.r * t) * norm.cdf(d2)
        ) / 252.0
        put_theta = (
            -(s * pdf * sigma) / (2.0 * sqrt_t)
            + self.r * k * np.exp(-self.r * t) * norm.cdf(-d2)
        ) / 252.0
        vega = s * pdf * sqrt_t / 100.0
        return StraddleGreeks(
            price=float(call + put),
            delta=float(call_delta + put_delta),
            gamma=float(2.0 * gamma),
            theta=float(call_theta + put_theta),
            vega=float(2.0 * vega),
        )

    def simulate_trade(
        self,
        df: pd.DataFrame,
        *,
        entry_idx: int,
        dte_days: int,
        entry_iv: float,
        iv_path: np.ndarray | None = None,
        max_minutes: int | None = None,
        timestamp_col: str = "time_key",
        strike_col: str | None = None,
        base_iv_col: str = "l3_base_iv",
    ) -> pd.DataFrame:
        """Simulate one ATM straddle trade from `entry_idx` forward."""
        if entry_idx < 0 or entry_idx >= len(df):
            return pd.DataFrame()
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        base_iv_series = (
            pd.to_numeric(df[base_iv_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if base_iv_col in df.columns
            else pd.Series(float(entry_iv), index=df.index, dtype=np.float64)
        )
        s_entry = float(close.iloc[entry_idx])
        strike = float(df[strike_col].iloc[entry_idx]) if strike_col and strike_col in df.columns else float(s_entry)
        t_total = max(float(dte_days), 1.0) / 252.0
        max_hold = int(max_minutes) if max_minutes is not None else int(dte_days * 390)
        entry_price = self.straddle_price(s_entry, strike, t_total, float(entry_iv))
        rows: list[dict[str, float | int | str | pd.Timestamp]] = []
        upper = min(len(df), entry_idx + max_hold + 1)
        for i, idx in enumerate(range(entry_idx + 1, upper), start=1):
            t_remaining = t_total - i / (390.0 * 252.0)
            if t_remaining <= 0.0:
                break
            s_now = float(close.iloc[idx])
            if iv_path is not None and i - 1 < len(iv_path):
                current_iv = float(iv_path[i - 1])
            else:
                current_iv = float(0.65 * float(entry_iv) + 0.35 * float(base_iv_series.iloc[idx]))
            current_iv = float(np.clip(current_iv, 0.05, 5.0))
            greeks = self.straddle_greeks(s_now, strike, t_remaining, current_iv)
            pnl = float(greeks.price - entry_price)
            pnl_pct = float(pnl / max(entry_price, 1e-9))
            rows.append(
                {
                    "minute": int(i),
                    "timestamp": ts.iloc[idx],
                    "underlying": s_now,
                    "open_price": float(open_.iloc[idx]),
                    "high_price": float(high.iloc[idx]),
                    "low_price": float(low.iloc[idx]),
                    "straddle_value": float(greeks.price),
                    "entry_value": float(entry_price),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "iv": current_iv,
                    "entry_iv": float(entry_iv),
                    "T_remaining": float(t_remaining),
                    "underlying_abs_move": float(abs(s_now / max(s_entry, 1e-9) - 1.0)),
                    "underlying_gap_abs": float(abs(float(open_.iloc[idx]) / max(float(close.iloc[idx - 1]), 1e-9) - 1.0)),
                    "delta": float(greeks.delta),
                    "gamma": float(greeks.gamma),
                    "theta": float(greeks.theta),
                    "vega": float(greeks.vega),
                }
            )
        return pd.DataFrame(rows)

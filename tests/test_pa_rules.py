"""Regression tests for core.pa_rules (causal PA stack)."""
from __future__ import annotations

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from core.pa_rules import _opening_patterns, add_pa_features


class TestOpeningPatterns(unittest.TestCase):
    def test_gap_up_fade_sets_reversal_without_lookahead(self) -> None:
        base = pd.Timestamp("2024-01-03 09:30:00", tz="America/New_York")
        rows = [
            # Prior session last bar (close = prior day close for next day)
            {
                "time_key": base - timedelta(days=1) + timedelta(hours=6, minutes=25),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1e6,
            },
            # Gap up open, still above prior close
            {
                "time_key": base,
                "open": 102.0,
                "high": 102.5,
                "low": 101.8,
                "close": 101.9,
                "volume": 1e6,
            },
            # Sweep through prior close; weak close vs session open
            {
                "time_key": base + timedelta(minutes=5),
                "open": 101.5,
                "high": 101.8,
                "low": 98.5,
                "close": 99.0,
                "volume": 1e6,
            },
        ]
        bars = pd.DataFrame(rows)
        out = _opening_patterns(bars)
        self.assertTrue(out["pa_gap_open_flag"].iloc[-1])
        self.assertFalse(out["pa_opening_reversal"].iloc[0])
        self.assertFalse(out["pa_opening_reversal"].iloc[1])
        self.assertTrue(out["pa_opening_reversal"].iloc[2])

    def test_gap_down_fade(self) -> None:
        base = pd.Timestamp("2024-01-03 09:30:00", tz="America/New_York")
        rows = [
            {
                "time_key": base - timedelta(days=1) + timedelta(hours=6, minutes=25),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1e6,
            },
            {
                "time_key": base,
                "open": 98.0,
                "high": 98.2,
                "low": 97.5,
                "close": 97.8,
                "volume": 1e6,
            },
            {
                "time_key": base + timedelta(minutes=5),
                "open": 98.0,
                "low": 97.0,
                "high": 101.0,
                "close": 99.2,
                "volume": 1e6,
            },
        ]
        bars = pd.DataFrame(rows)
        out = _opening_patterns(bars)
        self.assertTrue(out["pa_opening_reversal"].iloc[2])


class TestAddPaFeaturesSmoke(unittest.TestCase):
    def test_add_pa_features_runs_on_minute_bars(self) -> None:
        idx = pd.date_range("2024-01-03 09:30", periods=120, freq="1min", tz="America/New_York")
        rng = np.random.default_rng(0)
        close = 100.0 + np.cumsum(rng.normal(0, 0.02, size=len(idx)))
        df = pd.DataFrame(
            {
                "time_key": idx,
                "open": close + rng.normal(0, 0.01, size=len(idx)),
                "high": close + np.abs(rng.normal(0.03, 0.01, size=len(idx))),
                "low": close - np.abs(rng.normal(0.03, 0.01, size=len(idx))),
                "close": close,
                "volume": rng.integers(1_000, 10_000, size=len(idx)),
            }
        )
        atr = (df["high"] - df["low"]).ewm(span=14, min_periods=5).mean()
        out = add_pa_features(df, atr, timeframe="5min")
        self.assertGreater(len([c for c in out.columns if c.startswith("pa_")]), 50)
        self.assertIn("pa_opening_reversal", out.columns)
        self.assertIn("pa_bo_strength_up", out.columns)


if __name__ == "__main__":
    unittest.main()

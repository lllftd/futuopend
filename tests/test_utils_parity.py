from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from core.utils import DATA_DIR, calculate_max_drawdown, expand_signal_same_day


def test_data_dir_defaults_to_repo_data_or_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = Path(os.environ.get("DATA_DIR", repo_root / "data")).expanduser()

    assert DATA_DIR == expected


def test_expand_signal_same_day_matches_existing_window_semantics() -> None:
    times = pd.to_datetime(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-03 09:30",
            "2024-01-03 09:31",
        ]
    ).to_series(index=range(5))
    signal = pd.Series([False, True, False, True, False])

    expanded = expand_signal_same_day(signal, times, keep_bars=1)

    assert expanded.tolist() == [False, True, True, True, True]


def test_expand_signal_same_day_non_positive_window_is_identity_bool_series() -> None:
    times = pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:31"]).to_series(index=range(2))
    signal = pd.Series([None, True])

    expanded = expand_signal_same_day(signal, times, keep_bars=0)

    assert expanded.tolist() == [False, True]


def test_calculate_max_drawdown_on_equity_curve() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])

    assert calculate_max_drawdown(equity) == -0.25

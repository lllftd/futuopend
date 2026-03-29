from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
MINUTES_PER_YEAR = 252 * 390


def load_price_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0

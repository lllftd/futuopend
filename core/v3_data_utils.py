from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimize_ce_zlsma_kama_rule import RuleParams, apply_ce_features, build_base_features
from core.utils import load_price_data


def expand_ce_signals_same_day(featured: pd.DataFrame, keep_bars: int) -> pd.DataFrame:
    """
    Mirror live/monitor.py CE_SIGNAL_VALID_BARS: extend ce_buy/sell_signal forward within
    the same session day so triggers stay valid for keep_bars minutes after the trigger bar
    (implementation uses pos..pos+keep_bars inclusive → keep_bars+1 bars total, same as monitor).
    keep_bars <= 0 returns a copy unchanged.
    """
    if keep_bars <= 0:
        return featured.copy()
    df = featured.copy()
    df["ce_buy_signal"] = expand_signal_same_day(df["ce_buy_signal"], df["time_key"], keep_bars)
    df["ce_sell_signal"] = expand_signal_same_day(df["ce_sell_signal"], df["time_key"], keep_bars)
    return df


def expand_signal_same_day(sig: pd.Series, times: pd.Series, keep_bars: int) -> pd.Series:
    s = sig.fillna(False).astype(bool)
    if keep_bars <= 0:
        return s

    out = pd.Series(False, index=s.index)
    by_day = times.dt.date
    for day in by_day.unique():
        idx = s.index[by_day == day]
        arr = s.loc[idx].to_numpy(dtype=bool)
        n = len(arr)
        ext = np.zeros(n, dtype=bool)
        true_pos = np.where(arr)[0]
        for pos in true_pos:
            end = min(n, pos + keep_bars + 1)
            ext[pos:end] = True
        out.loc[idx] = ext
    return out


def prepare_featured(symbol: str, params: RuleParams) -> pd.DataFrame:
    raw = load_price_data(symbol)
    base = build_base_features(
        raw,
        zlsma_length=params.zlsma_length,
        zlsma_offset=params.zlsma_offset,
        kama_er_length=params.kama_er_length,
        kama_fast_length=params.kama_fast_length,
        kama_slow_length=params.kama_slow_length,
        atr_percentile_lookback=params.atr_percentile_lookback,
        pseudo_cvd_method=params.pseudo_cvd_method,
        cvd_lookback=params.cvd_lookback,
        cvd_slope_lookback=params.cvd_slope_lookback,
        pa_timeframe=getattr(params, "pa_timeframe", "5min"),
    )
    return apply_ce_features(base, params.ce_length, params.ce_multiplier)


def prepare_featured_with_monitor_ce(
    symbol: str, params: RuleParams, ce_valid_bars: int
) -> pd.DataFrame:
    """Full-history features + same-day CE validity expansion as live/monitor."""
    return expand_ce_signals_same_day(prepare_featured(symbol, params), ce_valid_bars)


def slice_range(featured: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    out = featured[(featured["time_key"] >= start_ts) & (featured["time_key"] < end_exclusive)].copy()
    return out.reset_index(drop=True)

from __future__ import annotations

import pandas as pd

from core.optimize_ce_zlsma_kama_rule import RuleParams, apply_ce_features, build_base_features
from core.utils import load_price_data


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
    )
    return apply_ce_features(base, params.ce_length, params.ce_multiplier)


def slice_range(featured: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    out = featured[(featured["time_key"] >= start_ts) & (featured["time_key"] < end_exclusive)].copy()
    return out.reset_index(drop=True)

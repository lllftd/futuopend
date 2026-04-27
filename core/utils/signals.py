from __future__ import annotations

import numpy as np
import pandas as pd


def expand_signal_same_day(sig: pd.Series, times: pd.Series, keep_bars: int) -> pd.Series:
    """Extend true signal bars forward within the same calendar day."""
    s = sig.fillna(False).astype(bool)
    if keep_bars <= 0:
        return s

    out = pd.Series(False, index=s.index)
    by_day = times.dt.date
    for day in by_day.unique():
        idx = s.index[by_day == day]
        arr = s.loc[idx].to_numpy(dtype=bool)
        ext = np.zeros(len(arr), dtype=bool)
        for pos in np.where(arr)[0]:
            ext[pos : min(len(arr), pos + keep_bars + 1)] = True
        out.loc[idx] = ext
    return out

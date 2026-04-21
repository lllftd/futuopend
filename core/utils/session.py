from __future__ import annotations

import numpy as np
import pandas as pd


def mark_session_boundaries(
    df: pd.DataFrame,
    *,
    time_col: str = "time_key",
    symbol_col: str = "symbol",
    overnight_gap_seconds: float = 4 * 3600,
    open_skip_bars: int = 0,
    close_cutoff_bars: int = 0,
) -> pd.DataFrame:
    """Annotate RTH-style session breaks (large timestamp gaps) per symbol.

    - New session when adjacent bar gap > ``overnight_gap_seconds`` (default 4h).
    - ``open_skip_bars``: first N bars of each session excluded from ``valid_for_label``.
    - ``close_cutoff_bars``: bars with fewer than this many bars until session end
      excluded (use ``horizon`` so forward labels stay inside the session).

    24/7 series (no large gaps): one long session per symbol; only global tail is
    trimmed by ``close_cutoff_bars``.

    Mutates ``df`` in place (adds/overwrites columns) and returns ``df``.
    """
    if time_col not in df.columns or symbol_col not in df.columns:
        raise KeyError(f"mark_session_boundaries: need columns {time_col!r} and {symbol_col!r}")

    gap_ns = int(max(1.0, float(overnight_gap_seconds)) * 1_000_000_000)
    open_skip_bars = max(0, int(open_skip_bars))
    close_cutoff_bars = max(0, int(close_cutoff_bars))

    session_id = np.zeros(len(df), dtype=np.int32)
    bar_of_session = np.zeros(len(df), dtype=np.int32)
    bars_remaining = np.zeros(len(df), dtype=np.int32)
    valid_for_label = np.ones(len(df), dtype=bool)

    for sym, grp in df.groupby(symbol_col, sort=False):
        pos = df.index.get_indexer(grp.index)
        if (pos < 0).any():
            raise ValueError("mark_session_boundaries: non-unique or missing index in group")
        t = pd.to_datetime(grp[time_col], errors="coerce").values.astype("datetime64[ns]")
        t64 = t.astype(np.int64)
        n = len(t64)
        if n == 0:
            continue
        is_break = np.zeros(n, dtype=bool)
        if n > 1:
            dt = np.diff(t64)
            is_break[1:] = dt > gap_ns
        sid = np.cumsum(is_break).astype(np.int32)

        bos = np.zeros(n, dtype=np.int32)
        br = np.zeros(n, dtype=np.int32)
        v = np.ones(n, dtype=bool)
        start = 0
        for i in range(1, n + 1):
            if i == n or sid[i] != sid[start]:
                for j in range(start, i):
                    bos[j] = j - start
                    br[j] = (i - 1) - j
                    if bos[j] < open_skip_bars:
                        v[j] = False
                    if br[j] < close_cutoff_bars:
                        v[j] = False
                start = i

        session_id[pos] = sid
        bar_of_session[pos] = bos
        bars_remaining[pos] = br
        valid_for_label[pos] = v

    df["session_id"] = session_id
    df["bar_of_session"] = bar_of_session
    df["bars_remaining"] = bars_remaining
    df["valid_for_label"] = valid_for_label
    return df

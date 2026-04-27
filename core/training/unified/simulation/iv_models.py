from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError:  # pragma: no cover
    arch_model = None  # type: ignore[assignment]


def _ewma_base_iv(close: pd.Series, *, span: int = 390 * 5) -> pd.Series:
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ewma_std = log_ret.ewm(span=max(10, int(span)), adjust=False).std().fillna(0.0)
    return ewma_std * np.sqrt(252.0 * 390.0)


def _daily_garch_iv(timestamp: pd.Series, close: pd.Series) -> pd.Series:
    if arch_model is None:
        return pd.Series(index=timestamp.index, dtype=np.float64)
    ts = pd.to_datetime(timestamp, errors="coerce")
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily = pd.DataFrame({"ts": ts, "ret": log_ret}).dropna()
    if daily.empty:
        return pd.Series(index=timestamp.index, dtype=np.float64)
    daily["date"] = daily["ts"].dt.floor("D")
    daily_ret = daily.groupby("date", observed=True)["ret"].sum().dropna()
    if len(daily_ret) < 40:
        return pd.Series(index=timestamp.index, dtype=np.float64)
    try:
        model = arch_model(daily_ret.to_numpy(dtype=np.float64) * 100.0, vol="Garch", p=1, q=1, mean="Zero")
        fit = model.fit(disp="off", show_warning=False)
        cond = pd.Series(fit.conditional_volatility / 100.0 * np.sqrt(252.0), index=daily_ret.index, dtype=np.float64)
    except Exception:
        return pd.Series(index=timestamp.index, dtype=np.float64)
    out = pd.DataFrame({"ts": ts})
    out["date"] = out["ts"].dt.floor("D")
    out["garch_iv"] = out["date"].map(cond).astype(np.float64)
    return out["garch_iv"].ffill().bfill()


def build_base_iv_series(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "time_key",
    close_col: str = "close",
) -> pd.Series:
    """Build a per-bar base IV proxy.

    Default behavior blends GARCH daily conditional vol (if `arch` is present),
    EWMA realized vol, and any long-window vol estimators already present.
    """
    close = pd.to_numeric(df[close_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    ewma_iv = _ewma_base_iv(close)
    gk390 = (
        pd.to_numeric(df.get("gk_vol_390"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "gk_vol_390" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=np.float64)
    )
    rv390 = (
        pd.to_numeric(df.get("rv_390"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "rv_390" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=np.float64)
    )
    base_mode = "garch"
    if base_mode in {"garch", "garch_plus_scenarios"}:
        garch_iv = _daily_garch_iv(df[timestamp_col], close)
        if garch_iv.isna().all():
            out = 0.55 * ewma_iv + 0.25 * gk390.fillna(ewma_iv) + 0.20 * rv390.fillna(ewma_iv)
        else:
            out = 0.45 * garch_iv + 0.30 * ewma_iv + 0.15 * gk390.fillna(ewma_iv) + 0.10 * rv390.fillna(ewma_iv)
    else:
        out = 0.60 * ewma_iv + 0.25 * gk390.fillna(ewma_iv) + 0.15 * rv390.fillna(ewma_iv)
    risk_premium = float(np.clip(0.02, 0.0, 1.0))
    out = out.astype(np.float64).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = (out + risk_premium).clip(lower=0.05, upper=5.0)
    out.name = "l3_base_iv"
    return out

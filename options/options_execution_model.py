from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from options.futu_options_data import FutuOptionDataClient, normalize_underlying


@dataclass(frozen=True)
class ContractSelectorConfig:
    expiry_rule: str = "same_day_if_available"
    otm_steps: int = 1
    skip_suspended: bool = True
    max_dte_days: int = 14
    chain_window_days: int = 7


@dataclass(frozen=True)
class SelectedOptionContract:
    underlying: str
    signal_time: pd.Timestamp
    entry_bar_time: pd.Timestamp
    signal_side: str
    underlying_price: float
    option_code: str
    option_type: str
    expiry: str
    strike_price: float
    lot_size: int | None
    selection_note: str


def signal_side_to_option_type(signal_side: str) -> str:
    normalized = signal_side.strip().lower()
    if normalized == "long":
        return "CALL"
    if normalized == "short":
        return "PUT"
    raise ValueError(f"Unsupported signal side: {signal_side}")


def choose_expiry_date(
    expirations: pd.DataFrame,
    signal_time: pd.Timestamp,
    expiry_rule: str = "same_day_if_available",
    max_dte_days: int = 14,
) -> tuple[str, str]:
    if expirations.empty:
        raise RuntimeError("Option expiration list is empty.")

    frame = expirations.copy()
    frame["expiry_date"] = pd.to_datetime(frame["strike_time"]).dt.date
    signal_date = signal_time.date()
    latest_date = signal_date + timedelta(days=max_dte_days)
    frame = frame.loc[
        (frame["expiry_date"] >= signal_date) & (frame["expiry_date"] <= latest_date)
    ].sort_values("expiry_date").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(
            f"No expiries within {max_dte_days} DTE were found for signal date {signal_date}."
        )

    if expiry_rule != "same_day_if_available":
        raise ValueError(f"Unsupported expiry_rule: {expiry_rule}")

    same_day = frame.loc[frame["expiry_date"] == signal_date]
    if not same_day.empty:
        expiry = same_day.iloc[0]["strike_time"]
        return str(expiry), "same_day"

    expiry = frame.iloc[0]["strike_time"]
    return str(expiry), "nearest_future"


def fetch_expirations_for_signal(
    client: FutuOptionDataClient,
    underlying: str,
    signal_time: pd.Timestamp,
    chain_window_days: int,
    max_dte_days: int,
    refresh: bool = False,
) -> pd.DataFrame:
    signal_date = pd.Timestamp(signal_time).normalize()
    live_expirations = client.get_option_expiration_dates(underlying, refresh=refresh)
    if live_expirations.empty:
        return pd.DataFrame(columns=["strike_time", "underlying"])

    live_expiry_dates = pd.to_datetime(live_expirations["strike_time"]).dt.normalize()
    earliest_live_expiry = live_expiry_dates.min()
    if signal_date < earliest_live_expiry:
        return pd.DataFrame(columns=["strike_time", "underlying"])

    max_end = signal_date + pd.Timedelta(days=max_dte_days)
    windows: list[pd.DataFrame] = []
    cursor = signal_date

    while cursor <= max_end:
        window_end = min(cursor + pd.Timedelta(days=chain_window_days - 1), max_end)
        chain_window = client.get_option_chain_window(
            underlying=underlying,
            start=cursor.strftime("%Y-%m-%d"),
            end=window_end.strftime("%Y-%m-%d"),
            refresh=refresh,
        )
        if not chain_window.empty:
            expiries = chain_window.loc[:, ["strike_time"]].drop_duplicates().copy()
            expiries["underlying"] = normalize_underlying(underlying)
            windows.append(expiries)
        cursor = window_end + pd.Timedelta(days=1)

    if not windows:
        return pd.DataFrame(columns=["strike_time", "underlying"])

    frame = pd.concat(windows, ignore_index=True)
    frame["strike_time"] = pd.to_datetime(frame["strike_time"]).dt.strftime("%Y-%m-%d")
    frame = frame.drop_duplicates(subset=["strike_time"]).sort_values("strike_time").reset_index(drop=True)
    return frame


def select_fixed_otm_contract(
    chain: pd.DataFrame,
    option_type: str,
    underlying_price: float,
    otm_steps: int = 1,
    skip_suspended: bool = True,
) -> tuple[pd.Series, str]:
    if chain.empty:
        raise RuntimeError("Option chain is empty.")

    side = option_type.upper()
    candidates = chain.copy()
    if "option_type" in candidates.columns:
        candidates = candidates.loc[candidates["option_type"].astype(str).str.upper() == side]
    if skip_suspended and "suspension" in candidates.columns:
        candidates = candidates.loc[~candidates["suspension"].fillna(False)]
    candidates["strike_price"] = pd.to_numeric(candidates["strike_price"], errors="coerce")
    candidates = candidates.dropna(subset=["strike_price"]).sort_values("strike_price").reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError(f"No {side} contracts remain after filtering.")

    step_index = max(otm_steps - 1, 0)
    if side == "CALL":
        otm = candidates.loc[candidates["strike_price"] > underlying_price].reset_index(drop=True)
        if not otm.empty:
            selected = otm.iloc[min(step_index, len(otm) - 1)]
            return selected, "fixed_otm_call"
    elif side == "PUT":
        otm = candidates.loc[candidates["strike_price"] < underlying_price].sort_values(
            "strike_price", ascending=False
        ).reset_index(drop=True)
        if not otm.empty:
            selected = otm.iloc[min(step_index, len(otm) - 1)]
            return selected, "fixed_otm_put"
    else:
        raise ValueError(f"Unsupported option_type: {option_type}")

    # Fallback to the nearest strike when a strictly OTM contract is unavailable.
    fallback_position = int((candidates["strike_price"] - underlying_price).abs().to_numpy().argmin())
    fallback = candidates.iloc[fallback_position]
    return fallback, "nearest_strike_fallback"


def select_option_contract_for_signal(
    client: FutuOptionDataClient,
    underlying: str,
    signal_time: pd.Timestamp,
    entry_bar_time: pd.Timestamp,
    signal_side: str,
    underlying_price: float,
    config: ContractSelectorConfig | None = None,
    refresh_expirations: bool = False,
    refresh_chains: bool = False,
) -> SelectedOptionContract:
    selector = config or ContractSelectorConfig()
    normalized = normalize_underlying(underlying)
    option_type = signal_side_to_option_type(signal_side)
    expirations = fetch_expirations_for_signal(
        client=client,
        underlying=normalized,
        signal_time=signal_time,
        chain_window_days=selector.chain_window_days,
        max_dte_days=selector.max_dte_days,
        refresh=refresh_expirations,
    )
    expiry, expiry_note = choose_expiry_date(
        expirations=expirations,
        signal_time=signal_time,
        expiry_rule=selector.expiry_rule,
        max_dte_days=selector.max_dte_days,
    )
    chain = client.get_option_chain(normalized, expiry, refresh=refresh_chains)
    selected, strike_note = select_fixed_otm_contract(
        chain=chain,
        option_type=option_type,
        underlying_price=underlying_price,
        otm_steps=selector.otm_steps,
        skip_suspended=selector.skip_suspended,
    )

    return SelectedOptionContract(
        underlying=normalized,
        signal_time=signal_time,
        entry_bar_time=entry_bar_time,
        signal_side=signal_side.lower(),
        underlying_price=float(underlying_price),
        option_code=str(selected["code"]),
        option_type=option_type,
        expiry=str(pd.to_datetime(selected["strike_time"]).strftime("%Y-%m-%d")),
        strike_price=float(selected["strike_price"]),
        lot_size=int(selected["lot_size"]) if "lot_size" in selected and pd.notna(selected["lot_size"]) else None,
        selection_note=f"{expiry_note}|{strike_note}",
    )

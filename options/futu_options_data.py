from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

try:
    import pandas as pd
    from futu import AuType, KLType, OpenQuoteContext, RET_OK
except ImportError as exc:  # pragma: no cover - only hit when runtime deps are missing
    missing_dependency_error = exc
    pd = None
    AuType = None
    KLType = None
    OpenQuoteContext = None
    RET_OK = None
else:
    missing_dependency_error = None


OPTIONS_CACHE_DIR = Path("data") / "options"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111
DEFAULT_MAX_COUNT = 1000
DEFAULT_KLINE_CALLS_PER_WINDOW = 55
DEFAULT_KLINE_RATE_WINDOW_SECONDS = 30.0
DEFAULT_KLINE_MIN_INTERVAL_SECONDS = 0.65
DEFAULT_RETRY_LIMIT = 3


@dataclass(frozen=True)
class OptionHistoryRequest:
    underlying: str
    option_code: str
    expiry: str
    option_type: str
    strike_price: float
    start: str
    end: str


def normalize_underlying(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Underlying symbol cannot be empty.")
    if "." not in normalized:
        normalized = f"US.{normalized}"
    return normalized


def ticker_from_symbol(symbol: str) -> str:
    return normalize_underlying(symbol).split(".", 1)[1]


def _sanitize_code(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def _strike_key(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_")


def _ensure_runtime_dependencies() -> None:
    if missing_dependency_error is not None:
        raise RuntimeError(
            "Missing dependency. Run `python -m pip install -r requirements.txt` first. "
            f"Import error: {missing_dependency_error}"
        )


class FutuOptionDataClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_count: int = DEFAULT_MAX_COUNT,
        autype_name: str = "NONE",
        extended_time: bool = False,
        cache_dir: Path = OPTIONS_CACHE_DIR,
        kline_calls_per_window: int = DEFAULT_KLINE_CALLS_PER_WINDOW,
        kline_rate_window_seconds: float = DEFAULT_KLINE_RATE_WINDOW_SECONDS,
        kline_min_interval_seconds: float = DEFAULT_KLINE_MIN_INTERVAL_SECONDS,
        retry_limit: int = DEFAULT_RETRY_LIMIT,
    ) -> None:
        self.host = host
        self.port = port
        self.max_count = max_count
        self.autype_name = autype_name.upper()
        self.extended_time = extended_time
        self.cache_dir = cache_dir
        self.kline_calls_per_window = kline_calls_per_window
        self.kline_rate_window_seconds = kline_rate_window_seconds
        self.kline_min_interval_seconds = kline_min_interval_seconds
        self.retry_limit = retry_limit
        self._quote_ctx = None
        self._kline_call_times: deque[float] = deque()
        self._last_kline_call_at = 0.0

    def __enter__(self) -> "FutuOptionDataClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._quote_ctx is not None:
            self._quote_ctx.close()
            self._quote_ctx = None

    @property
    def autype(self) -> Any:
        _ensure_runtime_dependencies()
        return getattr(AuType, self.autype_name)

    def _quote_context(self):
        _ensure_runtime_dependencies()
        if self._quote_ctx is None:
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
        return self._quote_ctx

    def _underlying_dir(self, underlying: str) -> Path:
        return self.cache_dir / ticker_from_symbol(underlying)

    def _expirations_path(self, underlying: str) -> Path:
        return self._underlying_dir(underlying) / "expirations.csv"

    def _chain_path(self, underlying: str, expiry: str) -> Path:
        return self._underlying_dir(underlying) / "chains" / f"{expiry}.csv"

    def _chain_window_path(self, underlying: str, start: str, end: str) -> Path:
        return self._underlying_dir(underlying) / "chain_windows" / f"{start}_to_{end}.csv"

    def _history_path(self, request: OptionHistoryRequest) -> Path:
        return (
            self._underlying_dir(request.underlying)
            / request.expiry
            / request.option_type.upper()
            / _strike_key(request.strike_price)
            / f"{request.start}_to_{request.end}_{_sanitize_code(request.option_code)}.csv"
        )

    def _read_csv(self, path: Path, parse_time_key: bool = False):
        frame = pd.read_csv(path)
        if parse_time_key and "time_key" in frame.columns:
            frame["time_key"] = pd.to_datetime(frame["time_key"])
        return frame

    def _write_csv(self, frame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        return path

    @staticmethod
    def _normalize_date_str(value: str) -> str:
        return pd.to_datetime(value).strftime("%Y-%m-%d")

    @staticmethod
    def _is_rate_limit_error(detail: object) -> bool:
        text = str(detail).lower()
        return "too frequent" in text or "30 seconds" in text

    def _throttle_kline_calls(self) -> None:
        now = time.monotonic()
        while self._kline_call_times and now - self._kline_call_times[0] >= self.kline_rate_window_seconds:
            self._kline_call_times.popleft()

        if self._last_kline_call_at > 0:
            elapsed = now - self._last_kline_call_at
            if elapsed < self.kline_min_interval_seconds:
                time.sleep(self.kline_min_interval_seconds - elapsed)
                now = time.monotonic()
                while self._kline_call_times and now - self._kline_call_times[0] >= self.kline_rate_window_seconds:
                    self._kline_call_times.popleft()

        if len(self._kline_call_times) >= self.kline_calls_per_window:
            sleep_for = self.kline_rate_window_seconds - (now - self._kline_call_times[0]) + 0.25
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
                while self._kline_call_times and now - self._kline_call_times[0] >= self.kline_rate_window_seconds:
                    self._kline_call_times.popleft()

        self._last_kline_call_at = time.monotonic()
        self._kline_call_times.append(self._last_kline_call_at)

    def get_option_expiration_dates(self, underlying: str, refresh: bool = False):
        normalized = normalize_underlying(underlying)
        cache_path = self._expirations_path(normalized)
        if cache_path.exists() and not refresh:
            return self._read_csv(cache_path)

        ret, data = self._quote_context().get_option_expiration_date(code=normalized)
        if ret != RET_OK:
            raise RuntimeError(f"{normalized} expiration lookup failed: {data}")

        frame = data.copy()
        if "strike_time" in frame.columns:
            frame["strike_time"] = pd.to_datetime(frame["strike_time"]).dt.strftime("%Y-%m-%d")
        frame["underlying"] = normalized
        frame = frame.sort_values(["strike_time"]).reset_index(drop=True)
        self._write_csv(frame, cache_path)
        return frame

    def get_option_chain(self, underlying: str, expiry: str, refresh: bool = False):
        normalized = normalize_underlying(underlying)
        expiry_key = self._normalize_date_str(expiry)
        cache_path = self._chain_path(normalized, expiry_key)
        if cache_path.exists() and not refresh:
            return self._read_csv(cache_path)

        ret, data = self._quote_context().get_option_chain(
            code=normalized,
            start=expiry_key,
            end=expiry_key,
        )
        if ret != RET_OK:
            raise RuntimeError(f"{normalized} option chain lookup failed for {expiry_key}: {data}")

        frame = data.copy()
        frame["underlying"] = normalized
        if "strike_time" in frame.columns:
            frame["strike_time"] = pd.to_datetime(frame["strike_time"]).dt.strftime("%Y-%m-%d")
        if "option_type" in frame.columns:
            frame["option_type"] = frame["option_type"].astype(str).str.upper()
        if "strike_price" in frame.columns:
            frame["strike_price"] = pd.to_numeric(frame["strike_price"], errors="coerce")
        if "suspension" in frame.columns:
            frame["suspension"] = frame["suspension"].fillna(False).astype(bool)
        frame = frame.sort_values(["option_type", "strike_price", "code"]).reset_index(drop=True)
        self._write_csv(frame, cache_path)
        return frame

    def get_option_chain_window(self, underlying: str, start: str, end: str, refresh: bool = False):
        normalized = normalize_underlying(underlying)
        start_key = self._normalize_date_str(start)
        end_key = self._normalize_date_str(end)
        cache_path = self._chain_window_path(normalized, start_key, end_key)
        if cache_path.exists() and not refresh:
            return self._read_csv(cache_path)

        ret, data = self._quote_context().get_option_chain(
            code=normalized,
            start=start_key,
            end=end_key,
        )
        if ret != RET_OK:
            raise RuntimeError(
                f"{normalized} option chain lookup failed for window {start_key}..{end_key}: {data}"
            )

        frame = data.copy()
        frame["underlying"] = normalized
        if "strike_time" in frame.columns:
            frame["strike_time"] = pd.to_datetime(frame["strike_time"]).dt.strftime("%Y-%m-%d")
        if "option_type" in frame.columns:
            frame["option_type"] = frame["option_type"].astype(str).str.upper()
        if "strike_price" in frame.columns:
            frame["strike_price"] = pd.to_numeric(frame["strike_price"], errors="coerce")
        if "suspension" in frame.columns:
            frame["suspension"] = frame["suspension"].fillna(False).astype(bool)
        frame = frame.sort_values(["strike_time", "option_type", "strike_price", "code"]).reset_index(drop=True)
        self._write_csv(frame, cache_path)
        return frame

    def get_option_history_bars(self, request: OptionHistoryRequest, refresh: bool = False):
        cache_path = self._history_path(request)
        if cache_path.exists() and not refresh:
            return self._read_csv(cache_path, parse_time_key=True)

        all_frames = []
        page_req_key = None
        page = 1
        while True:
            attempt = 0
            while True:
                self._throttle_kline_calls()
                ret, data, next_page_req_key = self._quote_context().request_history_kline(
                    code=request.option_code,
                    start=request.start,
                    end=request.end,
                    ktype=KLType.K_1M,
                    autype=self.autype,
                    max_count=self.max_count,
                    page_req_key=page_req_key,
                    extended_time=self.extended_time,
                )
                if ret == RET_OK:
                    page_req_key = next_page_req_key
                    break

                if self._is_rate_limit_error(data) and attempt < self.retry_limit:
                    attempt += 1
                    time.sleep(self.kline_rate_window_seconds + 1.0)
                    continue

                raise RuntimeError(
                    f"{request.option_code} history lookup failed on page {page}: {data}"
                )
            if data is None or data.empty:
                break
            all_frames.append(data.copy())
            page += 1
            if page_req_key is None:
                break

        if not all_frames:
            raise RuntimeError(
                f"{request.option_code} returned no option bars for {request.start}..{request.end}."
            )

        frame = pd.concat(all_frames, ignore_index=True)
        if "time_key" in frame.columns:
            frame["time_key"] = pd.to_datetime(frame["time_key"])
            frame = frame.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")

        frame["underlying"] = normalize_underlying(request.underlying)
        frame["option_code"] = request.option_code
        frame["expiry"] = pd.to_datetime(request.expiry).strftime("%Y-%m-%d")
        frame["option_type"] = request.option_type.upper()
        frame["strike_price"] = float(request.strike_price)
        ordered = self._preferred_history_order(frame.columns)
        frame = frame.loc[:, ordered].reset_index(drop=True)
        self._write_csv(frame, cache_path)
        return frame

    @staticmethod
    def _preferred_history_order(columns: list[str] | Any) -> list[str]:
        preferred = [
            "underlying",
            "option_code",
            "code",
            "name",
            "expiry",
            "option_type",
            "strike_price",
            "time_key",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "change_rate",
            "last_close",
        ]
        present = list(columns)
        ordered = [column for column in preferred if column in present]
        ordered.extend(column for column in present if column not in ordered)
        return ordered

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
    from futu import AuType, KLType, OpenQuoteContext, RET_OK
except ImportError as exc:  # pragma: no cover - exercised only in missing dependency environments
    missing_dependency_error = exc
    pd = None
    AuType = None
    KLType = None
    OpenQuoteContext = None
    RET_OK = None
else:
    missing_dependency_error = None


DEFAULT_SYMBOLS = ("US.QQQ", "US.SPY")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111
DEFAULT_MAX_COUNT = 1000
# OpenD returns only data it has; going very early avoids missing older history when the user wants “max” range.
DEFAULT_MAX_HISTORY_START = "1990-01-01"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DownloadConfig:
    symbols: tuple[str, ...]
    start: str
    end: str
    output_dir: Path
    host: str
    port: int
    max_count: int
    autype: object
    extended_time: bool
    merge: bool
    chunk_months: int


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description="Download US 1-minute historical bars from Futu OpenD."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbols to download, e.g. US.QQQ US.SPY",
    )
    parser.add_argument(
        "--max-history",
        action="store_true",
        help=f"Pull maximum available history: from {DEFAULT_MAX_HISTORY_START} through --end (default today). "
        "Ignores --years. Recommended with --chunk-months 1.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years to look back when --start is not provided (ignored if --max-history).",
    )
    parser.add_argument(
        "--start",
        help="Start date in YYYY-MM-DD. Overrides --years when provided (ignored if --max-history).",
    )
    parser.add_argument(
        "--end",
        help="End date in YYYY-MM-DD. Defaults to today when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Directory for downloaded CSV files (default: <repo>/data).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="OpenD host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="OpenD port.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=DEFAULT_MAX_COUNT,
        help="Maximum number of bars per page request.",
    )
    parser.add_argument(
        "--autype",
        choices=("NONE", "QFQ", "HFQ"),
        default="NONE",
        help="Price adjustment mode. NONE is usually safer for intraday backtests.",
    )
    parser.add_argument(
        "--extended-time",
        action="store_true",
        help="Include pre-market and after-hours bars for US symbols.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Append/merge into existing data/{TICKER}.csv (dedupe by time_key, keep last).",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=1,
        help="If >0, request history in monthly slices [start,end] then merge (more reliable for long spans). "
        "If 0, one OpenD range for the whole window (may truncate on some setups). Default: 1.",
    )

    args = parser.parse_args()
    end_date = parse_date(args.end) if args.end else date.today()
    if args.max_history:
        raw_start = os.environ.get("FUTU_US_1M_MAX_START", "").strip()
        start_date = parse_date(raw_start or DEFAULT_MAX_HISTORY_START)
    elif args.start:
        start_date = parse_date(args.start)
    else:
        if args.years <= 0:
            raise SystemExit("--years must be greater than 0")
        start_date = end_date - timedelta(days=args.years * 365)
    if start_date > end_date:
        raise SystemExit("--start must be earlier than or equal to --end")

    return DownloadConfig(
        symbols=tuple(normalize_symbol(symbol) for symbol in args.symbols),
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        max_count=args.max_count,
        autype=getattr(AuType, args.autype),
        extended_time=args.extended_time,
        merge=bool(args.merge),
        chunk_months=max(0, int(args.chunk_months)),
    )


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise SystemExit("Encountered an empty symbol in --symbols")
    if "." not in normalized:
        normalized = f"US.{normalized}"
    return normalized


def _kline_page_sleep_sec() -> float:
    """OpenD: historical candlestick ≤60 calls / 30s → ≥0.5s between calls (margin0.52)."""
    return max(0.05, float(os.environ.get("FUTU_KLINE_PAGE_SLEEP_SEC", "0.52")))


def _kline_chunk_extra_sleep_sec() -> float:
    """Extra pause after finishing one date chunk (on top of last page’s pacing)."""
    return max(0.0, float(os.environ.get("FUTU_KLINE_CHUNK_SLEEP_SEC", "0")))


def _is_futu_kline_rate_limit_message(msg: object) -> bool:
    raw = str(msg)
    s = raw.lower()
    return (
        "too frequent" in s
        or "frequent" in s
        or "频繁" in raw
        or ("60" in s and "30" in s)
    )


def fetch_symbol_bars(
    ctx: OpenQuoteContext,
    config: DownloadConfig,
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
):
    start_s = start if start is not None else config.start
    end_s = end if end is not None else config.end
    all_frames = []
    page_req_key = None
    page = 1
    page_pause = _kline_page_sleep_sec()
    cooldown = max(30.0, float(os.environ.get("FUTU_RATE_LIMIT_COOLDOWN_SEC", "35")))
    first_request = True

    while True:
        if not first_request:
            time.sleep(page_pause)
        first_request = False
        ret, data, page_req_key = ctx.request_history_kline(
            code=symbol,
            start=start_s,
            end=end_s,
            ktype=KLType.K_1M,
            autype=config.autype,
            max_count=config.max_count,
            page_req_key=page_req_key,
            extended_time=config.extended_time,
        )
        if ret != RET_OK:
            if _is_futu_kline_rate_limit_message(data):
                print(f"[{symbol}] page {page}: rate limited ({data!s}); sleep {cooldown:.0f}s …", flush=True)
                time.sleep(cooldown)
                first_request = True
                continue
            raise RuntimeError(f"{symbol} download failed on page {page}: {data}")

        if data is None or data.empty:
            break

        print(f"[{symbol}] page {page}: fetched {len(data)} rows")
        all_frames.append(data.copy())
        page += 1

        if page_req_key is None:
            break

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)

    if "time_key" in result.columns:
        result["time_key"] = result["time_key"].astype(str)
        result = result.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")

    result["symbol"] = symbol
    ordered = preferred_column_order(result.columns)
    return result.loc[:, ordered].reset_index(drop=True)


def fetch_symbol_bars_merged(ctx: OpenQuoteContext, config: DownloadConfig, symbol: str) -> pd.DataFrame:
    """Full window: either one OpenD request window or monthly (multi-month) slices."""
    d0 = parse_date(config.start)
    d1 = parse_date(config.end)
    if config.chunk_months <= 0:
        frame = fetch_symbol_bars(ctx, config, symbol)
        if frame.empty:
            raise RuntimeError(
                f"{symbol} returned no data for {config.start}..{config.end}. "
                "Check market permissions, subscription, and OpenD status."
            )
        return frame

    step = max(1, int(config.chunk_months))
    periods = pd.period_range(pd.Timestamp(d0), pd.Timestamp(d1), freq="M")
    if len(periods) == 0:
        frame = fetch_symbol_bars(ctx, config, symbol)
        if frame.empty:
            raise RuntimeError(f"{symbol} returned no data for {config.start}..{config.end}.")
        return frame

    parts: list[pd.DataFrame] = []
    for i in range(0, len(periods), step):
        batch = periods[i : i + step]
        cs = max(d0, batch[0].start_time.date())
        ce = min(d1, batch[-1].end_time.date())
        label = f"{cs}..{ce}"
        print(f"[{symbol}] chunk {label} (slice {i // step + 1}/{(len(periods) + step - 1) // step})", flush=True)
        chunk = fetch_symbol_bars(ctx, config, symbol, start=cs.isoformat(), end=ce.isoformat())
        if not chunk.empty:
            parts.append(chunk)
        time.sleep(_kline_chunk_extra_sleep_sec())

    if not parts:
        raise RuntimeError(
            f"{symbol} returned no data across monthly chunks {config.start}..{config.end}. "
            "Check US market data permissions in OpenD / account."
        )
    result = pd.concat(parts, ignore_index=True)
    if "time_key" in result.columns:
        result["time_key"] = result["time_key"].astype(str)
        result = result.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")
    result["symbol"] = symbol
    ordered = preferred_column_order(result.columns)
    return result.loc[:, ordered].reset_index(drop=True)


def preferred_column_order(columns: Iterable[str]) -> list[str]:
    preferred = [
        "symbol",
        "code",
        "name",
        "time_key",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "pe_ratio",
        "turnover_rate",
        "change_rate",
        "last_close",
    ]
    present = list(columns)
    ordered = [column for column in preferred if column in present]
    ordered.extend(column for column in present if column not in ordered)
    return ordered


def write_symbol_csv(frame, output_dir: Path, symbol: str, merge: bool = False) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker = symbol.split(".", 1)[1]
    output_path = output_dir / f"{ticker}.csv"

    if merge and output_path.exists():
        old = pd.read_csv(output_path)
        frame = pd.concat([old, frame], ignore_index=True)

    if "time_key" in frame.columns:
        frame["time_key"] = frame["time_key"].astype(str)
    frame = frame.sort_values("time_key", kind="mergesort").drop_duplicates(subset=["time_key"], keep="last")
    frame = frame.reset_index(drop=True)

    ordered = preferred_column_order(frame.columns)
    frame = frame[[c for c in ordered if c in frame.columns]]
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path, len(frame)


def main() -> int:
    if missing_dependency_error is not None:
        print(
            "Missing dependency. Run `python -m pip install -r requirements.txt` first.",
            file=sys.stderr,
        )
        print(f"Import error: {missing_dependency_error}", file=sys.stderr)
        return 1

    config = parse_args()
    print(
        f"Connecting to OpenD at {config.host}:{config.port} | "
        f"range={config.start}..{config.end} | extended_time={config.extended_time} | merge={config.merge} | "
        f"chunk_months={config.chunk_months}",
        flush=True,
    )

    ctx = OpenQuoteContext(host=config.host, port=config.port)
    try:
        for symbol in config.symbols:
            print(f"Starting download for {symbol}")
            frame = fetch_symbol_bars_merged(ctx, config, symbol)
            output_path, nrows = write_symbol_csv(frame, config.output_dir, symbol, merge=config.merge)
            action = "merged into" if config.merge else "wrote"
            print(f"[{symbol}] {action} {nrows} total rows -> {output_path}")
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    finally:
        ctx.close()

    print("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

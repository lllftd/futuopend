from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
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
        "--years",
        type=int,
        default=2,
        help="Number of years to look back when --start is not provided.",
    )
    parser.add_argument(
        "--start",
        help="Start date in YYYY-MM-DD. Overrides --years when provided.",
    )
    parser.add_argument(
        "--end",
        help="End date in YYYY-MM-DD. Defaults to today when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for downloaded CSV files.",
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

    args = parser.parse_args()
    end_date = parse_date(args.end) if args.end else date.today()
    start_date = parse_date(args.start) if args.start else end_date - timedelta(days=args.years * 365)
    if args.years <= 0:
        raise SystemExit("--years must be greater than 0")
    if start_date > end_date:
        raise SystemExit("--start must be earlier than or equal to --end")

    return DownloadConfig(
        symbols=tuple(normalize_symbol(symbol) for symbol in args.symbols),
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        output_dir=Path(args.output_dir),
        host=args.host,
        port=args.port,
        max_count=args.max_count,
        autype=getattr(AuType, args.autype),
        extended_time=args.extended_time,
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


def fetch_symbol_bars(ctx: OpenQuoteContext, config: DownloadConfig, symbol: str):
    all_frames = []
    page_req_key = None
    page = 1

    while True:
        ret, data, page_req_key = ctx.request_history_kline(
            code=symbol,
            start=config.start,
            end=config.end,
            ktype=KLType.K_1M,
            autype=config.autype,
            max_count=config.max_count,
            page_req_key=page_req_key,
            extended_time=config.extended_time,
        )
        if ret != RET_OK:
            raise RuntimeError(f"{symbol} download failed on page {page}: {data}")

        if data is None or data.empty:
            break

        print(f"[{symbol}] page {page}: fetched {len(data)} rows")
        all_frames.append(data.copy())
        page += 1

        if page_req_key is None:
            break

    if not all_frames:
        raise RuntimeError(
            f"{symbol} returned no data. Check market permissions, date range, and OpenD status."
        )

    result = pd.concat(all_frames, ignore_index=True)

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


def write_symbol_csv(frame, output_dir: Path, symbol: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker = symbol.split(".", 1)[1]
    output_path = output_dir / f"{ticker}.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


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
        f"range={config.start}..{config.end} | extended_time={config.extended_time}"
    )

    ctx = OpenQuoteContext(host=config.host, port=config.port)
    try:
        for symbol in config.symbols:
            print(f"Starting download for {symbol}")
            frame = fetch_symbol_bars(ctx, config, symbol)
            output_path = write_symbol_csv(frame, config.output_dir, symbol)
            print(f"[{symbol}] wrote {len(frame)} rows to {output_path}")
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    finally:
        ctx.close()

    print("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

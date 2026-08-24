"""
Fetch and cache OHLCV klines from Binance's public data archive
(https://data.binance.vision).

Downloads monthly kline zip files, extracts and concatenates them into a
single clean CSV per symbol under data/processed/.

Usage:
    python -m src.data_loader --symbol BTCUSDT --interval 1h --start 2023-01 --end 2026-08
    python -m src.data_loader --symbol ETHUSDT --interval 1h --start 2023-01 --end 2026-08
"""

import argparse
import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def month_range(start: str, end: str):
    """Yield 'YYYY-MM' strings from start to end inclusive."""
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month > 12:
            month = 1
            year += 1


def download_month(symbol: str, interval: str, year_month: str, raw_dir: Path):
    """Download and parse a single month's kline zip. Returns None if not available."""
    filename = f"{symbol}-{interval}-{year_month}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [skip] {year_month}: not available (status {resp.status_code})")
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_bytes(resp.content)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    # Binance switched some archives from millisecond to microsecond timestamps
    # partway through 2025. Detect and normalize to milliseconds per month so
    # everything concatenates consistently.
    sample = df["open_time"].iloc[0]
    if sample > 10**14:  # microseconds (16 digits) vs milliseconds (13 digits)
        df["open_time"] = df["open_time"] // 1000
        df["close_time"] = df["close_time"] // 1000

    print(f"  [ok]   {year_month}: {len(df)} rows")
    return df




def download_klines(symbol: str, interval: str, start_month: str, end_month: str,
                     raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Download all months in range, concatenate, clean, and save a single processed CSV."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {symbol} {interval} klines from {start_month} to {end_month}...")
    frames = []
    for ym in month_range(start_month, end_month):
        df = download_month(symbol, interval, ym, raw_path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval} {start_month}..{end_month}")

    full = pd.concat(frames, ignore_index=True)

    full["open_time"] = pd.to_datetime(full["open_time"], unit="ms")
    full["close_time"] = pd.to_datetime(full["close_time"], unit="ms")
    full = full[["open_time", "open", "high", "low", "close", "volume",
                 "close_time", "num_trades"]]
    full = full.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)

    out_file = processed_path / f"{symbol}_{interval}.csv"
    full.to_csv(out_file, index=False)
    print(f"Saved {len(full)} rows to {out_file}")
    return full


def main():
    parser = argparse.ArgumentParser(description="Download Binance klines to data/processed/")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--interval", default="1h", help="e.g. 1h, 1m, 1d")
    parser.add_argument("--start", required=True, help="YYYY-MM")
    parser.add_argument("--end", required=True, help="YYYY-MM")
    args = parser.parse_args()

    download_klines(args.symbol, args.interval, args.start, args.end)


if __name__ == "__main__":
    main()
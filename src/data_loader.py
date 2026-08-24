"""
Fetch and cache OHLCV klines from Binance's public data archive.

Usage:
    python -m src.data_loader --symbol BTCUSDT --interval 1h --start 2023-01 --end 2026-08
"""


def download_klines(symbol: str, interval: str, start_month: str, end_month: str, out_dir: str = "data/raw"):
    """Download monthly klines zip files from data.binance.vision and save to out_dir."""
    raise NotImplementedError


if __name__ == "__main__":
    pass

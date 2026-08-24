"""
Leak-free feature engineering on OHLCV data.

All features must be computable using only information available up to
and including the current timestep (no lookahead). Every feature below uses
only .shift(), .rolling(), or .diff() — all strictly backward-looking pandas
operations — so nothing here can see into the future at the row it's
computed for.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling volatility, momentum, volume z-score, and RSI-style features.

    Expects df to have columns: open_time, open, high, low, close, volume
    (sorted chronologically, as produced by src/data_loader.py).
    """
    df = df.copy()

    # --- Returns (base signal most other features build on) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Rolling realized volatility, multiple horizons ---
    for window in [6, 24, 168]:  # 6h, 1 day, 1 week
        df[f"realized_vol_{window}h"] = df["log_return"].rolling(window).std()

    # --- Momentum: cumulative return over past N hours ---
    for window in [6, 24, 168]:
        df[f"momentum_{window}h"] = np.log(df["close"] / df["close"].shift(window))

    # --- Volume z-score: how unusual is current volume vs recent history ---
    vol_roll_mean = df["volume"].rolling(168).mean()
    vol_roll_std = df["volume"].rolling(168).std()
    df["volume_zscore_168h"] = (df["volume"] - vol_roll_mean) / vol_roll_std

    # --- RSI-style relative strength indicator (14-period, standard convention) ---
    df["rsi_14h"] = _compute_rsi(df["close"], period=14)

    # --- Time-of-day / day-of-week (cyclical encoding, no leakage since these
    #     are deterministic from the timestamp itself) ---
    df["hour_sin"] = np.sin(2 * np.pi * df["open_time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["open_time"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["open_time"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["open_time"].dt.dayofweek / 7)

    return df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI: average gain / average loss over a rolling window, both backward-looking."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_feature_columns() -> list[str]:
    """Names of the engineered feature columns (excludes raw OHLCV and time columns)."""
    return [
        "realized_vol_6h", "realized_vol_24h", "realized_vol_168h",
        "momentum_6h", "momentum_24h", "momentum_168h",
        "volume_zscore_168h", "rsi_14h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
"""
Leak-free feature engineering on OHLCV data.

All features must be computable using only information available up to
and including the current timestep (no lookahead).
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility, momentum, volume z-score, and RSI-style features."""
    raise NotImplementedError

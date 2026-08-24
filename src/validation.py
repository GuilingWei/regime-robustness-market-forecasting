"""
Chronological (walk-forward) train/val/test splitting.

Never shuffle. Never let a validation/test row's timestamp precede any
timestamp used in training for that split.
"""

import pandas as pd


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Return (train_df, val_df, test_df) split strictly in chronological order."""
    raise NotImplementedError


def assert_no_leakage(train_df: pd.DataFrame, other_df: pd.DataFrame, time_col: str = "open_time"):
    """Raise an AssertionError if any timestamp in other_df precedes the max timestamp in train_df incorrectly."""
    raise NotImplementedError

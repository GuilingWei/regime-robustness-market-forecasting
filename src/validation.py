"""
Chronological (walk-forward) train/val/test splitting.

Never shuffle. Never let a validation/test row's timestamp precede any
timestamp used in training for that split. This module is the single most
important piece of correctness in this project -- every model result is only
as trustworthy as this splitter.
"""

import pandas as pd


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15,
                        time_col: str = "open_time"):
    """
    Split df into (train, val, test) strictly in chronological order.

    df must already be sorted by time_col ascending. No row in val may have
    a timestamp earlier than any row in train; no row in test may have a
    timestamp earlier than any row in val.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    assert_no_leakage(train_df, val_df, time_col)
    assert_no_leakage(val_df, test_df, time_col)

    return train_df, val_df, test_df


def assert_no_leakage(earlier_df: pd.DataFrame, later_df: pd.DataFrame, time_col: str = "open_time"):
    """
    Raise an AssertionError if any row in later_df has a timestamp earlier
    than or equal to the maximum timestamp in earlier_df. This is the core
    leakage guard for the whole project -- call it after every split.
    """
    if earlier_df.empty or later_df.empty:
        return

    max_earlier = earlier_df[time_col].max()
    min_later = later_df[time_col].min()

    assert min_later > max_earlier, (
        f"Leakage detected: later_df starts at {min_later}, "
        f"but earlier_df extends up to {max_earlier}. "
        f"The later split must start strictly after the earlier split ends."
    )


def regime_train_test_split(df: pd.DataFrame, train_start: str, train_end: str,
                             test_start: str, test_end: str, time_col: str = "open_time"):
    """
    Split by explicit date ranges rather than fractions -- used for the
    regime-generalization experiment (train on calm window, test on stress
    window), which are not adjacent in time.

    Unlike walk_forward_split, this does NOT assert train comes immediately
    before test (the whole point of the regime experiment is they're
    separated by a gap and a genuinely different market condition). It does
    still assert train_end < test_start, since training on data that
    postdates the test window would be leakage regardless of the experiment
    design.
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    train_df = df[(df[time_col] >= train_start) & (df[time_col] <= train_end)].reset_index(drop=True)
    test_df = df[(df[time_col] >= test_start) & (df[time_col] <= test_end)].reset_index(drop=True)

    assert pd.Timestamp(train_end) < pd.Timestamp(test_start), (
        f"train_end ({train_end}) must be before test_start ({test_start}) "
        f"-- training on data that postdates the test window is leakage."
    )

    return train_df, test_df
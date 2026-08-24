"""Unit tests for src/validation.py -- the most important tests in this repo."""

import pandas as pd
import pytest

from src.validation import walk_forward_split, assert_no_leakage, regime_train_test_split


def _make_dummy_df(n=1000):
    return pd.DataFrame({
        "open_time": pd.date_range("2023-01-01", periods=n, freq="h"),
        "value": range(n),
    })


def test_walk_forward_split_no_leakage():
    df = _make_dummy_df()
    train, val, test = walk_forward_split(df)
    assert train["open_time"].max() < val["open_time"].min()
    assert val["open_time"].max() < test["open_time"].min()


def test_walk_forward_split_covers_all_rows():
    df = _make_dummy_df()
    train, val, test = walk_forward_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_assert_no_leakage_raises_on_violation():
    df = _make_dummy_df()
    # Deliberately construct a bad split: later_df starts before earlier_df ends
    earlier = df.iloc[:100]
    later = df.iloc[50:150]  # overlaps with earlier -- should fail
    with pytest.raises(AssertionError):
        assert_no_leakage(earlier, later)


def test_regime_train_test_split_respects_gap():
    df = _make_dummy_df(n=2000)
    train, test = regime_train_test_split(
        df,
        train_start="2023-01-01", train_end="2023-01-10",
        test_start="2023-02-01", test_end="2023-02-10",
    )
    assert train["open_time"].max() < test["open_time"].min()
    assert len(train) > 0
    assert len(test) > 0


def test_regime_train_test_split_raises_if_train_after_test():
    df = _make_dummy_df(n=2000)
    with pytest.raises(AssertionError):
        regime_train_test_split(
            df,
            train_start="2023-02-01", train_end="2023-02-10",
            test_start="2023-01-01", test_end="2023-01-10",
        )
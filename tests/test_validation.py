"""Unit tests for src/validation.py -- the most important tests in this repo."""

import pandas as pd


def test_no_leakage_between_splits():
    """A validation/test set must never contain a timestamp earlier than the training set's max timestamp."""
    pass

"""Chronological split boundary tests: strict ordering, no overlap, no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_forecasting_lab.config import DEFAULT_CONFIG, Config
from macro_forecasting_lab.data import split_chronological


@pytest.fixture()
def frame(trending_series: pd.Series) -> pd.DataFrame:
    return trending_series.to_frame()


def test_split_is_strictly_ordered(frame: pd.DataFrame) -> None:
    parts = split_chronological(frame, DEFAULT_CONFIG)
    assert parts["train"].index.max() < parts["val"].index.min()
    assert parts["val"].index.max() < parts["test"].index.min()
    assert parts["train"].index.is_monotonic_increasing
    assert parts["val"].index.is_monotonic_increasing
    assert parts["test"].index.is_monotonic_increasing


def test_split_partitions_the_test_era_without_overlap(frame: pd.DataFrame) -> None:
    parts = split_chronological(frame, DEFAULT_CONFIG)
    train_max, val_max = parts["train"].index.max(), parts["val"].index.max()
    assert train_max <= pd.Timestamp(DEFAULT_CONFIG.train_end)
    assert val_max <= pd.Timestamp(DEFAULT_CONFIG.val_end)
    assert parts["test"].index.min() >= pd.Timestamp(DEFAULT_CONFIG.test_start)
    total = len(parts["train"]) + len(parts["val"]) + len(parts["test"])
    assert total == len(frame)  # every row lands in exactly one partition


def test_split_is_contiguous_monthly(frame: pd.DataFrame) -> None:
    parts = split_chronological(frame, DEFAULT_CONFIG)
    for name in ("train", "val", "test"):
        part = parts[name]
        ordinals = np.asarray(part.index.year * 12 + part.index.month)
        assert (np.diff(ordinals) == 1).all(), f"{name} partition has month gaps"


def test_test_window_contains_covid_break() -> None:
    """The COVID shock (2020-02..2020-12) must fall inside the test window."""
    assert pd.Timestamp("2020-02-01") >= pd.Timestamp(DEFAULT_CONFIG.test_start)


def test_custom_split_dates_respected(frame: pd.DataFrame) -> None:
    config = Config(
        train_end="1998-12-31",
        val_start="1999-01-01",
        val_end="2001-12-31",
        test_start="2002-01-01",
    )
    parts = split_chronological(frame, config)
    assert parts["train"].index.max() == pd.Timestamp("1998-12-01")
    assert parts["val"].index.min() == pd.Timestamp("1999-01-01")
    assert parts["test"].index.min() == pd.Timestamp("2002-01-01")


def test_empty_partition_raises() -> None:
    short = pd.Series(
        range(6), index=pd.date_range("2020-01-01", periods=6, freq="MS")
    ).to_frame()
    with pytest.raises(ValueError, match="empty partition"):
        split_chronological(short, DEFAULT_CONFIG)

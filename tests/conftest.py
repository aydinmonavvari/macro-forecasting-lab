"""Shared synthetic fixtures. No network access anywhere in the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_monthly_index(start: str = "1990-01-01", periods: int = 240) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="MS")


@pytest.fixture(scope="session")
def stationary_series() -> pd.Series:
    """Synthetic stationary monthly AR(2)-ish series with mild seasonality."""
    rng = np.random.default_rng(42)
    n = 240
    months = np.arange(n) % 12
    eps = rng.normal(0.0, 1.0, n)
    values = np.empty(n)
    values[0] = 0.0
    values[1] = 0.5
    for t in range(2, n):
        values[t] = 0.5 * values[t - 1] - 0.2 * values[t - 2] + 0.8 * np.sin(
            2 * np.pi * months[t] / 12
        ) + eps[t]
    index = make_monthly_index(periods=n)
    return pd.Series(values, index=index, name="synthetic")


@pytest.fixture(scope="session")
def trending_series() -> pd.Series:
    """Synthetic non-stationary (unit-root) monthly series for split tests.

    Long enough to span the default chronological split (train <= 2009-12,
    validation 2010-2015, test 2016-) with data on both sides.
    """
    rng = np.random.default_rng(7)
    n = 480  # 1990-01 .. 2029-12
    steps = rng.normal(0.0, 0.5, n)
    index = make_monthly_index(periods=n)
    return pd.Series(100.0 + np.cumsum(steps), index=index, name="random_walk")


@pytest.fixture(scope="session")
def aligned_supervised_frame(stationary_series: pd.Series) -> pd.DataFrame:
    """Direct frame built by the production feature builder."""
    from macro_forecasting_lab.features import make_direct_frame

    return make_direct_frame(
        stationary_series, horizons=(1, 12), lags=(0, 1, 2, 3, 6, 12), month_dummies=True
    )

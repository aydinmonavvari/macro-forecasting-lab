"""Leakage and alignment tests — the scientific core of the repo.

These tests assert the direct-horizon target alignment and that no future
information can enter the lag matrix or a training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_forecasting_lab.data import transform_series
from macro_forecasting_lab.features import (
    assert_target_alignment,
    feature_columns,
    make_direct_frame,
    target_column,
    training_mask,
)


def test_direct_target_alignment_h12(stationary_series: pd.Series) -> None:
    """Row with origin t, h=12 must carry the value at t + 12 months."""
    y = stationary_series
    frame = make_direct_frame(y, horizons=(1, 12), lags=(0, 1, 2, 3, 6, 12))
    origin = pd.Timestamp("2000-01-01")
    target_col = target_column(frame, 12)
    expected_date = origin + pd.DateOffset(months=12)
    assert frame.loc[origin, target_col] == pytest.approx(y.loc[expected_date])
    # And the h=1 target is the next month, not the same month:
    assert frame.loc[origin, target_column(frame, 1)] == pytest.approx(
        y.loc[origin + pd.DateOffset(months=1)]
    )


def test_direct_target_alignment_full_index(stationary_series: pd.Series) -> None:
    """The alignment holds for every row where the target is defined."""
    y = stationary_series
    for h in (1, 12):
        frame = make_direct_frame(y, horizons=(h,), lags=(0, 1, 3, 12))
        expected = y.shift(-h).reindex(frame.index)
        actual = frame[target_column(frame, h)]
        both = expected.notna() & actual.notna()
        np.testing.assert_allclose(expected[both].to_numpy(), actual[both].to_numpy())


def test_lag_matrix_contains_no_future_data(stationary_series: pd.Series) -> None:
    """Feature y_l{k} at row t equals the value observed k months earlier."""
    y = stationary_series
    frame = make_direct_frame(y, horizons=(1, 12), lags=(0, 1, 2, 3, 6, 12))
    for k in (0, 1, 2, 3, 6, 12):
        column = f"y_l{k}"
        expected = y.shift(k).reindex(frame.index)
        both = expected.notna() & frame[column].notna()
        np.testing.assert_allclose(expected[both].to_numpy(), frame[column][both].to_numpy())


def test_assert_target_alignment_detects_corruption(stationary_series: pd.Series) -> None:
    """The production guard must flag any tampering with the target column."""
    y = stationary_series
    frame = make_direct_frame(y, horizons=(12,), lags=(0, 1, 12))
    assert_target_alignment(y, frame, 12)  # passes on honest frame
    corrupted = frame.copy()
    first_valid = corrupted[target_column(corrupted, 12)].first_valid_index()
    corrupted.loc[first_valid, target_column(corrupted, 12)] += 999.0
    with pytest.raises(AssertionError, match="misaligned"):
        assert_target_alignment(y, corrupted, 12)


def test_training_mask_forbids_unrealized_targets(stationary_series: pd.Series) -> None:
    """Rows whose direct target was not realized at the origin cannot be used."""
    frame = make_direct_frame(stationary_series, horizons=(12,), lags=(0, 1, 12))
    origin = pd.Timestamp("2000-06-01")
    mask = training_mask(frame, origin, horizon=12)
    usable = frame.index[mask]
    assert (usable + pd.DateOffset(months=12) <= origin).all(), (
        "training window reaches beyond information available at the origin"
    )
    # The origin row itself must never be usable at h=12 (its target is in 2001-06).
    assert not mask.loc[origin]


def test_no_target_columns_in_feature_list(aligned_supervised_frame: pd.DataFrame) -> None:
    cols = feature_columns(aligned_supervised_frame)
    assert not any(c.startswith("target_") for c in cols)
    assert "y_l0" in cols and "y_l12" in cols
    assert any(c.startswith("m_") for c in cols)  # calendar dummies present


def test_month_dummies_reference_level(aligned_supervised_frame: pd.DataFrame) -> None:
    """January is the dropped reference level; 11 dummies remain."""
    month_cols = [c for c in aligned_supervised_frame.columns if c.startswith("m_")]
    assert len(month_cols) == 11
    assert "m_1" not in month_cols


def test_yoy_log_diff_transform() -> None:
    """12-month log difference x100 approximates the YoY percent change."""
    months = pd.date_range("2010-01-01", periods=30, freq="MS")
    base = 100.0 * (1.0 + 0.01) ** np.arange(30)  # exactly 1% growth per month
    series = pd.Series(base, index=months)
    result = transform_series(series, "yoy_log_diff_pct").dropna()
    # 12 log steps of 1% = ~12.68% continuously compounded YoY.
    np.testing.assert_allclose(result.to_numpy(), 100 * 12 * np.log(1.01), rtol=1e-9)
    with pytest.raises(ValueError, match="Unknown transform"):
        transform_series(series, "triple_differencing")

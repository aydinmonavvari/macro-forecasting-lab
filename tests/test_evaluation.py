"""Metric correctness vs hand-computed values + DM test sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from macro_forecasting_lab.evaluation import (
    dm_test,
    exclude_covid_mask,
    mae,
    mape,
    mase,
    mase_scale,
    metric_row,
    rmse,
)


# --------------------------------------------------------------------------
# Hand-computed metrics
# --------------------------------------------------------------------------
def test_rmse_mae_hand_computed() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx(1.25)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 0 + 4 + 4) / 4))


def test_mape_excludes_zeros_and_counts_them() -> None:
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 5.0, 10.0])
    value, excluded = mape(y_true, y_pred)
    assert excluded == 1
    assert value == pytest.approx(np.mean([0.5, 0.5]) * 100)


def test_mase_scale_and_mase_hand_computed() -> None:
    """Seasonal-naive in-sample MAE over a small 4-period seasonal series.

    Every seasonal difference is exactly +2, so the MASE denominator is 2.
    """
    block = [10.0, 12.0, 11.0, 13.0]
    values = np.array([b + 2.0 * k for k in range(4) for b in block])
    train = pd.Series(values)
    scale = mase_scale(train, period=4)
    assert scale == pytest.approx(2.0)
    errors = np.array([1.0, 1.0])
    preds = np.array([0.0, 3.0])
    # MAE = (1 + 2) / 2 = 1.5 -> MASE = 1.5 / 2 = 0.75
    assert mase(errors, preds, scale=scale) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="scale"):
        mase(errors, preds, scale=0.0)


def test_metric_row_fields() -> None:
    row = metric_row(
        series="x", horizon=3, window="full", model="m",
        y_true=np.array([2.0, 4.0]), y_pred=np.array([1.0, 4.0]), scale=1.0,
    )
    assert row["rmse"] == pytest.approx(np.sqrt(0.5))
    assert row["mae"] == pytest.approx(0.5)
    assert row["n_obs"] == 2
    assert row["dm_stat_vs_seasonal_naive"] is None


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
def test_dm_test_detects_clearly_worse_model() -> None:
    """Model 1 (std 1) vs model 2 (std 4): DM must strongly reject equality."""
    rng = np.random.default_rng(42)
    e1 = rng.normal(0.0, 1.0, 400)
    e2 = rng.normal(0.0, 4.0, 400)
    result = dm_test(e1, e2, max_lag=1)
    assert result["stat"] < 0  # model 1 has smaller loss
    assert result["p_normal"] < 0.001
    assert result["p_t"] < 0.001
    assert result["n"] == 400


def test_dm_test_identical_errors_is_null() -> None:
    e = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.4, 0.9, -0.3])
    result = dm_test(e, e, max_lag=1)
    assert result["stat"] == pytest.approx(0.0)
    assert result["p_normal"] == pytest.approx(1.0)


def test_dm_test_symmetric_statistic() -> None:
    rng = np.random.default_rng(7)
    e1 = rng.normal(0.0, 1.0, 300)
    e2 = rng.normal(0.0, 1.4, 300)
    forward = dm_test(e1, e2, max_lag=1)
    backward = dm_test(e2, e1, max_lag=1)
    assert forward["stat"] == pytest.approx(-backward["stat"])
    assert forward["p_normal"] == pytest.approx(backward["p_normal"])


def test_dm_test_input_validation() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        dm_test(np.arange(5.0), np.arange(6.0))
    with pytest.raises(ValueError, match="Unsupported loss"):
        dm_test(np.arange(5.0), np.arange(5.0), loss="absolute")


def test_dm_normal_vs_t_pvalue_relation() -> None:
    """The t p-value is the more conservative (larger) of the two for large |stat|."""
    rng = np.random.default_rng(3)
    e1 = rng.normal(0.0, 1.0, 40)
    e2 = rng.normal(0.0, 2.0, 40)
    result = dm_test(e1, e2, max_lag=1)
    z = abs(result["stat"])
    assert result["p_t"] == pytest.approx(
        2.0 * scipy_stats.t.sf(z, df=39), rel=1e-9
    )
    assert result["p_t"] >= result["p_normal"]


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def test_exclude_covid_mask() -> None:
    index = pd.DatetimeIndex(
        ["2019-12-01", "2020-01-01", "2020-02-01", "2020-06-01", "2020-12-01",
         "2021-01-01"]
    )
    mask = exclude_covid_mask(index)
    np.testing.assert_array_equal(mask, [True, True, False, False, False, True])

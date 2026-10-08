"""Model smoke tests on synthetic data (fast, deterministic, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from macro_forecasting_lab.features import (
    feature_columns,
    make_direct_frame,
    target_column,
    training_mask,
)
from macro_forecasting_lab.models import (
    ARIMAForecaster,
    NaiveLast,
    RollingMean,
    SARIMAForecaster,
    SeasonalNaive,
    SupervisedForecaster,
    build_supervised_models,
)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_naive_last_forecasts_last_value(stationary_series: pd.Series) -> None:
    model = NaiveLast().fit(stationary_series)
    for h in (1, 12):
        assert model.forecast(h) == pytest.approx(stationary_series.iloc[-1])


def test_seasonal_naive_follows_seasonal_lag(stationary_series: pd.Series) -> None:
    """Standard seasonal-naive: forecast at h uses y observed (h mod 12) back.

    For h=12 the forecast is the *last* value (y_{t+12-12} = y_t); for h=1 it
    is the value 11 months before the origin (y_{t-11}); beyond one period it
    wraps.
    """
    model = SeasonalNaive(period=12).fit(stationary_series)
    assert model.forecast(12) == pytest.approx(stationary_series.iloc[-1])
    assert model.forecast(1) == pytest.approx(stationary_series.iloc[-12])
    assert model.forecast(13) == pytest.approx(stationary_series.iloc[-12])


def test_rolling_mean_is_trailing_window_mean(stationary_series: pd.Series) -> None:
    model = RollingMean(window=12).fit(stationary_series)
    expected = stationary_series.iloc[-12:].mean()
    assert model.forecast(7) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Supervised models
# --------------------------------------------------------------------------
def test_supervised_models_smoke_on_synthetic(aligned_supervised_frame: pd.DataFrame) -> None:
    frame = aligned_supervised_frame.dropna()
    fcols = feature_columns(frame)
    target = target_column(frame, 1)
    for model in build_supervised_models(n_estimators=20, seed=42):
        model.fit(frame[fcols], frame[target])
        prediction = model.predict(frame.loc[[frame.index[50]], fcols])
        assert np.isfinite(prediction[0])


def test_supervised_fit_uses_only_realized_targets(
    stationary_series: pd.Series,
) -> None:
    """Direct-h training rows must satisfy origin + h <= forecast origin."""
    frame = make_direct_frame(stationary_series, horizons=(12,), lags=(0, 1, 12))
    origin = frame.index[120]
    mask = training_mask(frame, origin, horizon=12)
    train = frame.loc[mask]
    assert (train.index + pd.DateOffset(months=12) <= origin).all()
    assert len(train) < len(frame)


def test_ols_recovers_ar_signal_exactly() -> None:
    """OLS on a pure AR(1) with no noise should recover the coefficient.

    With ``lags=(0,)`` the feature at row t is y_t and the direct target is
    y_{t+1} = 0.8 y_t + 0.2, so OLS must recover (0.8, 0.2) exactly.
    """
    index = pd.date_range("2000-01-01", periods=60, freq="MS")
    y = pd.Series(0.0, index=index)
    for t in range(1, 60):
        y.iloc[t] = 0.8 * y.iloc[t - 1] + 0.2
    frame = make_direct_frame(y, horizons=(1,), lags=(0,), month_dummies=False)
    frame = frame.dropna()
    fcols = feature_columns(frame)
    model = SupervisedForecaster("ols", LinearRegression())
    model.fit(frame[fcols], frame[target_column(frame, 1)])
    prediction = model.predict(frame.loc[[index[30]], fcols])[0]
    assert prediction == pytest.approx(0.8 * y.loc[index[30]] + 0.2, abs=1e-8)


# --------------------------------------------------------------------------
# Statistical wrappers
# --------------------------------------------------------------------------
def test_arima_wrapper_smoke(stationary_series: pd.Series) -> None:
    model = ARIMAForecaster(order=(1, 0, 0)).fit(stationary_series.iloc[:180])
    forecast = model.forecast(3)
    assert np.isfinite(forecast)
    low, high = model.forecast_interval(3, alpha=0.05)
    assert low <= forecast <= high


def test_sarima_wrapper_smoke(stationary_series: pd.Series) -> None:
    model = SARIMAForecaster(
        order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)
    ).fit(stationary_series.iloc[:120])
    forecast = model.forecast(12)
    assert np.isfinite(forecast)
    low, high = model.forecast_interval(12, alpha=0.05)
    assert low <= forecast <= high


def test_random_state_pinned() -> None:
    """All stochastic estimators must carry random_state=42."""
    for model in build_supervised_models(n_estimators=10, seed=42):
        if hasattr(model.base_estimator, "random_state"):
            assert model.base_estimator.random_state == 42


def test_forecaster_names_are_unique_and_stable() -> None:
    names = [
        NaiveLast().name,
        SeasonalNaive().name,
        RollingMean().name,
        ARIMAForecaster((1, 0, 0)).name,
        SARIMAForecaster((1, 0, 0), (0, 0, 0, 12)).name,
        *[m.name for m in build_supervised_models()],
    ]
    assert len(names) == len(set(names))

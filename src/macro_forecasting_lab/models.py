"""Forecasting models with a uniform fit/forecast interface.

Two model families live here:

* :class:`UnivariateForecaster` — consumes only the history of the target
  series up to the forecast origin (baselines, ARIMA, SARIMA).
* :class:`SupervisedForecaster` — consumes a feature matrix built by
  :mod:`macro_forecasting_lab.features` (OLS, random forest, gradient
  boosting).

Every model predicts horizon by horizon from an explicit origin: it is fit on
data up to ``t`` only, then asked for the value ``h`` steps ahead. Randomness
is pinned with ``random_state=42`` everywhere (OLS and the baselines are
deterministic).
"""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

INF = float("inf")


# --------------------------------------------------------------------------
# Univariate family
# --------------------------------------------------------------------------
class UnivariateForecaster:
    """Base class for models fit on the target series alone."""

    name: str = "univariate"
    kind: str = "base"

    def fit(self, y: pd.Series) -> Self:
        self.y_ = y
        self.n_obs_ = len(y)
        return self

    def forecast(self, horizon: int) -> float:
        msg = "Subclasses must implement forecast()"
        raise NotImplementedError(msg)

    def forecast_interval(self, horizon: int, alpha: float = 0.05) -> tuple[float, float] | None:
        """95%-style prediction interval at ``horizon``; ``None`` if unavailable."""
        return None


class NaiveLast(UnivariateForecaster):
    """Random-walk baseline: forecast the last observed value at every horizon."""

    name = "naive_last"
    kind = "baseline"

    def forecast(self, horizon: int) -> float:
        return float(self.y_.iloc[-1])


class SeasonalNaive(UnivariateForecaster):
    """Seasonal-naive baseline: forecast the value observed one seasonal period ago."""

    name = "seasonal_naive"
    kind = "baseline"

    def __init__(self, period: int = 12) -> None:
        self.period = period

    def forecast(self, horizon: int) -> float:
        offset = (horizon - 1) % self.period
        return float(self.y_.iloc[-self.period + offset])


class RollingMean(UnivariateForecaster):
    """Rolling-mean baseline over the trailing ``window`` months."""

    name = "rolling_mean_12"
    kind = "baseline"

    def __init__(self, window: int = 12) -> None:
        self.window = window

    def forecast(self, horizon: int) -> float:
        return float(self.y_.iloc[-self.window :].mean())


class ARIMAForecaster(UnivariateForecaster):
    """Wrapper around ``statsmodels.tsa.arima.model.ARIMA``.

    ``order`` is ``(p, d, q)``; ``d`` is chosen by unit-root testing (ADF +
    KPSS) before any grid search, and ``(p, q)`` are selected by AIC on
    train+validation only (see :func:`select_arima_order`).
    """

    name = "arima"
    kind = "statistical"

    def __init__(self, order: tuple[int, int, int]) -> None:
        self.order = order

    def fit(self, y: pd.Series) -> Self:
        super().fit(y)
        _, d, _ = self.order
        trend = "c" if d == 0 else "n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", ValueWarning)
            self.result_ = ARIMA(y.astype(float), order=self.order, trend=trend).fit()
        return self

    def forecast(self, horizon: int) -> float:
        return float(self.result_.forecast(steps=horizon).iloc[-1])

    def forecast_interval(self, horizon: int, alpha: float = 0.05) -> tuple[float, float]:
        interval = self.result_.get_forecast(steps=horizon).conf_int(alpha=alpha)
        return float(interval.iloc[-1, 0]), float(interval.iloc[-1, 1])


class SARIMAForecaster(UnivariateForecaster):
    """Wrapper around ``statsmodels`` SARIMAX with an annual seasonal period."""

    name = "sarima"
    kind = "statistical"

    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y: pd.Series) -> Self:
        super().fit(y)
        _, d, _ = self.order
        trend = "c" if d == 0 else "n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", ValueWarning)
            self.result_ = SARIMAX(
                y.astype(float),
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def forecast(self, horizon: int) -> float:
        return float(self.result_.forecast(steps=horizon).iloc[-1])

    def forecast_interval(self, horizon: int, alpha: float = 0.05) -> tuple[float, float]:
        interval = self.result_.get_forecast(steps=horizon).conf_int(alpha=alpha)
        return float(interval.iloc[-1, 0]), float(interval.iloc[-1, 1])


# --------------------------------------------------------------------------
# Supervised family
# --------------------------------------------------------------------------
class SupervisedForecaster:
    """Adapter that gives any scikit-learn regressor the project interface.

    A fresh clone of the estimator is fit at every call so repeated fits
    (rolling-origin evaluation) cannot silently reuse state. Predictions are
    plain point forecasts — tree/linear regressors provide no native
    prediction intervals, which we state honestly in the write-up.
    """

    kind = "ml"

    def __init__(self, name: str, estimator: object) -> None:
        self.name = name
        self.base_estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.feature_names_ = list(X.columns)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X[self.feature_names_].to_numpy())

    def forecast_interval(self, horizon: int, alpha: float = 0.05) -> None:
        return None


def build_supervised_models(n_estimators: int = 200, seed: int = 42) -> list[SupervisedForecaster]:
    """The three supervised regressors used in the study, seeds pinned."""
    return [
        SupervisedForecaster("ols", LinearRegression()),
        SupervisedForecaster(
            "random_forest",
            RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=-1
            ),
        ),
        SupervisedForecaster(
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=seed
            ),
        ),
    ]


# --------------------------------------------------------------------------
# Order selection (AIC on train+validation only)
# --------------------------------------------------------------------------
def select_arima_order(
    y: pd.Series,
    d: int,
    pq_grid: tuple[int, ...] = (0, 1, 2),
) -> tuple[int, int, int]:
    """Pick ``(p, d, q)`` minimizing AIC over a small grid; ``d`` fixed by tests."""
    best_order, best_aic = (1, d, 1), INF
    for p in pq_grid:
        for q in pq_grid:
            order = (p, d, q)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, d_, _ = order
                    trend = "c" if d_ == 0 else "n"
                    result = ARIMA(y.astype(float), order=order, trend=trend).fit()
                aic = float(result.aic)
            except Exception:  # noqa: BLE001 - any failed fit is simply skipped
                continue
            if aic < best_aic:
                best_order, best_aic = order, aic
    return best_order


def select_sarima_order(
    y: pd.Series,
    d: int,
    pq_grid: tuple[int, ...] = (0, 1, 2),
    seasonal_pq_grid: tuple[int, ...] = (0, 1),
    period: int = 12,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Pick ``(p,d,q)(P,0,Q)[period]`` minimizing AIC; seasonal ``D=0`` because
    all inputs are seasonally adjusted (documented in the README)."""
    best, best_aic = ((1, d, 1), (0, 0, 0, period)), INF
    for p in pq_grid:
        for q in pq_grid:
            for big_p in seasonal_pq_grid:
                for big_q in seasonal_pq_grid:
                    order = (p, d, q)
                    seasonal = (big_p, 0, big_q, period)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            trend = "c" if d == 0 else "n"
                            result = SARIMAX(
                                y.astype(float),
                                order=order,
                                seasonal_order=seasonal,
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)
                        aic = float(result.aic)
                    except Exception:  # noqa: BLE001
                        continue
                    if aic < best_aic:
                        best, best_aic = (order, seasonal), aic
    return best

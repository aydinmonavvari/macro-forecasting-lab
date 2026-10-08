"""Evaluation metrics and the Diebold-Mariano test of equal predictive accuracy.

Metrics: RMSE, MAE, MAPE (reported but de-emphasized when the target crosses
zero, e.g. inflation in 2009/2015/2020), and MASE scaled by the in-sample MAE
of the seasonal-naive forecast on the training window, following Hyndman &
Athanasopoulos, *Forecasting: Principles and Practice* (OTexts, 3rd ed.),
Section 5.8.

Diebold-Mariano: implemented after Diebold & Mariano (1995), "Comparing
Predictive Accuracy", *Journal of Business & Economic Statistics* 13(3),
253-263, using the squared (small-square) loss differential and a HAC
long-run variance estimator truncated at lag 1. Both the asymptotic normal
p-value and the small-sample t p-value (T-1 df) are returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2020-12-31")


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """MAPE in percent, plus the number of excluded zero/undefined points.

    MAPE is undefined at ``y_true == 0``; such points are excluded (count
    reported) rather than silently dropped. For inflation the series crosses
    zero in deflationary episodes, so MASE is the primary scale-free metric.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0.0
    excluded = int((~mask).sum())
    value = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return value, excluded


def mase_scale(train: pd.Series, period: int = 12) -> float:
    """In-sample MAE of the seasonal-naive forecast on the training window.

    This is the MASE denominator recommended by Hyndman & Athanasopoulos
    (2018/2021): the scale is estimated *only* on training data, before any
    evaluation window is touched.
    """
    diff = (train - train.shift(period)).dropna()
    return float(np.mean(np.abs(diff)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """Mean absolute scaled error given a precomputed ``scale``."""
    if scale <= 0:
        msg = f"MASE scale must be positive, got {scale}"
        raise ValueError(msg)
    return mae(y_true, y_pred) / scale


# --------------------------------------------------------------------------
# Diebold-Mariano test (Diebold & Mariano 1995)
# --------------------------------------------------------------------------
def dm_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    max_lag: int = 1,
    loss: str = "squared",
) -> dict[str, float]:
    """Two-sided Diebold-Mariano test comparing two forecast error sequences.

    The loss differential is ``d_t = L(e1_t) - L(e2_t)`` with ``L(e) = e**2``
    (squared/"small-square" loss). The long-run variance of ``d_t`` is
    estimated with a HAC estimator truncated at ``max_lag`` lags (default 1,
    following the original paper's treatment of autocorrelated differentials):

        V = (gamma_0 + 2 * sum_{k=1..max_lag} gamma_k) / T

    Returns a dict with ``stat``, ``p_normal`` (asymptotic N(0,1)), and
    ``p_t`` (Student-t with T-1 df, a common small-sample safeguard).
    A negative statistic means model 1 has *smaller* average loss (better).
    """
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)
    if len(e1) != len(e2) or len(e1) < 3:
        msg = "DM test needs equal-length error vectors of length >= 3"
        raise ValueError(msg)
    if loss != "squared":
        msg = f"Unsupported loss {loss!r}; only 'squared' is implemented"
        raise ValueError(msg)

    differential = e1**2 - e2**2
    n = len(differential)
    centered = differential - differential.mean()
    gamma_0 = float(np.dot(centered, centered) / n)
    long_run = gamma_0
    for k in range(1, max_lag + 1):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run += 2.0 * gamma_k

    mean_differential = float(differential.mean())
    if long_run <= 0:
        return {"stat": 0.0, "p_normal": 1.0, "p_t": 1.0, "n": float(n)}
    statistic = mean_differential / np.sqrt(long_run / n)
    return {
        "stat": float(statistic),
        "p_normal": float(2.0 * stats.norm.sf(abs(statistic))),
        "p_t": float(2.0 * stats.t.sf(abs(statistic), df=n - 1)),
        "n": float(n),
    }


# --------------------------------------------------------------------------
# Windows / tidy results
# --------------------------------------------------------------------------
def exclude_covid_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Mask that is False for target dates inside the COVID structural-break
    window (2020-02 through 2020-12) and True otherwise."""
    return np.asarray(~((index >= COVID_START) & (index <= COVID_END)), dtype=bool)


def metric_row(
    series: str,
    horizon: int,
    window: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: float,
    dm_stat: float | None = None,
    dm_p: float | None = None,
    dm_p_t: float | None = None,
) -> dict[str, object]:
    """One tidy record for ``results.csv``."""
    mape_value, mape_excluded = mape(y_true, y_pred)
    return {
        "series": series,
        "horizon": horizon,
        "window": window,
        "model": model,
        "n_obs": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape_pct": mape_value,
        "mape_excluded_zeros": mape_excluded,
        "mase": mase(y_true, y_pred, scale),
        "dm_stat_vs_seasonal_naive": dm_stat,
        "dm_p_value": dm_p,
        "dm_p_value_t": dm_p_t,
    }

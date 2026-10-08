"""Static figures for the macro-forecasting-lab study (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from macro_forecasting_lab.evaluation import COVID_END, COVID_START  # noqa: E402

MODEL_COLORS = {
    "naive_last": "#9e9e9e",
    "seasonal_naive": "#4d4d4d",
    "rolling_mean_12": "#a6761d",
    "arima": "#1f77b4",
    "sarima": "#2ca02c",
    "ols": "#9467bd",
    "random_forest": "#ff7f0e",
    "gradient_boosting": "#d62728",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _shade_covid(ax: plt.Axes) -> None:
    ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.12, zorder=0)


def plot_series_with_splits(
    y: pd.Series,
    train_end: str,
    val_end: str,
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    """Plot the full series with train/validation/test shading."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(y.index, y.to_numpy(), lw=0.9, color="#1a1a1a")
    ax.axvspan(y.index.min(), pd.Timestamp(train_end), color="#2166ac", alpha=0.10, label="train")
    ax.axvspan(
        pd.Timestamp(train_end), pd.Timestamp(val_end), color="#67a9cf", alpha=0.25, label="val"
    )
    ax.axvspan(pd.Timestamp(val_end), y.index.max(), color="#f4a582", alpha=0.35, label="test")
    _style_axis(ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_forecasts(
    actual: pd.Series,
    predictions: dict[str, pd.Series],
    title: str,
    ylabel: str,
    path: Path,
    shade_covid: bool = True,
) -> Path:
    """Forecast-vs-actual plot for all models over the test window."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(actual.index, actual.to_numpy(), color="black", lw=1.8, label="actual", zorder=5)
    for name, pred in predictions.items():
        ax.plot(
            pred.index,
            pred.to_numpy(),
            lw=1.1,
            alpha=0.85,
            color=MODEL_COLORS.get(name),
            label=name,
        )
    if shade_covid:
        _shade_covid(ax)
    _style_axis(ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_intervals(
    actual: pd.Series,
    point: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    """Actual vs. point forecast with 95% prediction band (ARIMA-type models)."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(lower.index, lower.to_numpy(), upper.to_numpy(), alpha=0.25,
                    color="#1f77b4", label="95% prediction interval")
    ax.plot(actual.index, actual.to_numpy(), color="black", lw=1.6, label="actual")
    ax.plot(point.index, point.to_numpy(), color="#1f77b4", lw=1.2, label="point forecast")
    _shade_covid(ax)
    _style_axis(ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residual_acf(residuals: pd.Series, title: str, path: Path, nlags: int = 24) -> Path:
    """ACF of test-window residuals with 95% white-noise bounds."""
    values = residuals.dropna().to_numpy()
    correlations = acf(values, nlags=nlags, fft=False)
    bound = 1.96 / np.sqrt(len(values))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(correlations)), correlations, width=0.4, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="red", ls="--", lw=0.9, label="±1.96/√T")
    ax.axhline(-bound, color="red", ls="--", lw=0.9)
    ax.set_xlabel("lag (months)")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dm_bars(
    dm_stats: dict[str, float], title: str, path: Path, significance: float = 1.96
) -> Path:
    """Bar chart of DM statistics (vs. seasonal naive); negative = better than benchmark."""
    names = list(dm_stats)
    values = [dm_stats[n] for n in names]
    colors = ["#2ca02c" if v < -significance else "#9e9e9e" for v in values]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(names, values, color=colors, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-significance, color="red", ls="--", lw=1.0, label="−1.96 (significant at 5%)")
    ax.axhline(significance, color="red", ls="--", lw=1.0)
    ax.set_ylabel("DM statistic (squared loss, vs seasonal-naive)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_acf_pacf_grid(
    series: pd.Series, name: str, path: Path, nlags: int = 36
) -> Path:
    """Side-by-side ACF/PACF used in the EDA narrative."""
    from statsmodels.tsa.stattools import pacf  # local import keeps module surface small

    a = acf(series.dropna(), nlags=nlags, fft=False)
    p = pacf(series.dropna(), nlags=nlags)
    bound = 1.96 / np.sqrt(len(series.dropna()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, values, label in ((axes[0], a, "ACF"), (axes[1], p, "PACF")):
        ax.bar(range(len(values)), values, width=0.4, color="#4c72b0")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(bound, color="red", ls="--", lw=0.9)
        ax.axhline(-bound, color="red", ls="--", lw=0.9)
        ax.set_title(f"{name} — {label}")
        ax.set_xlabel("lag (months)")
        ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

"""End-to-end experiment orchestration.

Running :func:`run` performs the full study:

1. acquire + transform FRED data (cached),
2. split chronologically (train <= 2009-12, validation 2010-2015, test 2016-),
3. stationarity diagnostics (ADF + KPSS) on train+validation to fix the
   integration order ``d`` for ARIMA/SARIMA,
4. AIC-guided order selection for ARIMA/SARIMA on train+validation only,
5. rolling-origin direct-h forecasts at h=1 and h=12 for every model,
6. metrics (RMSE/MAE/MAPE/MASE) on the full test window and excluding the
   COVID structural-break window (2020-02..2020-12),
7. Diebold-Mariano tests vs. the seasonal-naive benchmark,
8. tidy results CSV, prediction dump, figures, and a Markdown summary.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from macro_forecasting_lab import evaluation as ev
from macro_forecasting_lab import plots
from macro_forecasting_lab.config import DEFAULT_CONFIG, Config
from macro_forecasting_lab.data import build_modeling_frame, split_chronological
from macro_forecasting_lab.features import (
    add_cross_lags,
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
    select_arima_order,
    select_sarima_order,
)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def stationarity_diagnostics(y: pd.Series) -> dict[str, float | int]:
    """ADF + KPSS on a series; integration order ``d`` follows from agreement.

    Rule: ``d = 0`` only when ADF rejects the unit root (p < 0.05) AND KPSS
    fails to reject stationarity (p > 0.05); otherwise ``d = 1``. The rule is
    deliberately conservative about claiming stationarity.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        adf_p = float(adfuller(y.to_numpy(), autolag="AIC")[1])
        kpss_p = float(kpss(y.to_numpy(), regression="c", nlags="auto")[1])
    stationary = adf_p < 0.05 and kpss_p > 0.05
    return {"adf_p": adf_p, "kpss_p": kpss_p, "d": 0 if stationary else 1}


def _offset(h: int) -> pd.DateOffset:
    """Month offset helper (origins/targets move in whole months)."""
    return pd.DateOffset(months=h)


# --------------------------------------------------------------------------
# Evaluation core
# --------------------------------------------------------------------------
def evaluate_series(
    y: pd.Series,
    sup_frame: pd.DataFrame,
    config: Config,
    arima_order: tuple[int, int, int],
    sarima_order: tuple[int, int, int],
    sarima_seasonal: tuple[int, int, int, int],
) -> tuple[dict[int, dict[str, pd.Series]], dict[int, dict[str, dict[str, pd.Series]]]]:
    """Rolling-origin direct-h evaluation for one target series.

    Returns ``(predictions, intervals)`` where ``predictions[h][model]`` is a
    Series indexed by *target* date, and ``intervals[h][model]`` holds
    lower/upper prediction-interval Series for ARIMA-type models.
    """
    test_start = pd.Timestamp(config.test_start)
    # Rolling-origin evaluation step: score on every k-th test month only.
    # This is a standard variance/compute trade-off (Tashman 2000) and is
    # applied identically in the prediction and scoring stages.
    targets = y.index[y.index >= test_start][:: config.target_step]
    horizons = config.horizons
    alpha = 1.0 - config.confidence_level

    univariate: list = [
        NaiveLast(),
        SeasonalNaive(period=config.seasonal_period),
        RollingMean(window=config.seasonal_period),
        ARIMAForecaster(arima_order),
        SARIMAForecaster(sarima_order, sarima_seasonal),
    ]
    supervised: list[SupervisedForecaster] = build_supervised_models(
        config.n_estimators, config.seed
    )
    model_names = [m.name for m in univariate + supervised]

    predictions: dict[int, dict[str, pd.Series]] = {
        h: {name: pd.Series(np.nan, index=targets, dtype=float) for name in model_names}
        for h in horizons
    }
    intervals: dict[int, dict[str, dict[str, pd.Series]]] = {
        h: {} for h in horizons
    }

    origins = sorted({t for h in horizons for t in (targets - _offset(h))})
    fcols = feature_columns(sup_frame)

    for i, origin in enumerate(origins):
        history = y.loc[:origin]
        for model in univariate:
            model.fit(history)
            for h in horizons:
                target_date = origin + _offset(h)
                if target_date in predictions[h][model.name].index:
                    predictions[h][model.name][target_date] = model.forecast(h)
                    if model.name in ("arima", "sarima"):
                        low, high = model.forecast_interval(h, alpha=alpha)
                        bucket = intervals[h].setdefault(
                            model.name,
                            {
                                "lower": pd.Series(np.nan, index=targets, dtype=float),
                                "upper": pd.Series(np.nan, index=targets, dtype=float),
                            },
                        )
                        bucket["lower"][target_date] = low
                        bucket["upper"][target_date] = high

        for h in horizons:
            target_date = origin + _offset(h)
            if target_date not in predictions[h][supervised[0].name].index:
                continue
            target_name = target_column(sup_frame, h)
            mask = training_mask(sup_frame, origin, h)
            train = sup_frame.loc[mask].dropna(subset=[target_name])
            for model in supervised:
                model.fit(train[fcols], train[target_name])
                row = sup_frame.loc[[origin], fcols]
                predictions[h][model.name][target_date] = float(model.predict(row)[0])
        if (i + 1) % 24 == 0:
            print(f"    ... {i + 1}/{len(origins)} origins done", flush=True)

    return predictions, intervals


def score_predictions(
    y: pd.Series,
    predictions: dict[int, dict[str, pd.Series]],
    scale: float,
    column: str,
    config: Config,
) -> pd.DataFrame:
    """Turn raw rolling-origin predictions into tidy metric rows (both windows)."""
    test_start = pd.Timestamp(config.test_start)
    # Rolling-origin evaluation step: score on every k-th test month only.
    # This is a standard variance/compute trade-off (Tashman 2000) and is
    # applied identically in the prediction and scoring stages.
    targets = y.index[y.index >= test_start][:: config.target_step]
    rows: list[dict[str, object]] = []
    benchmark = "seasonal_naive"

    for h, model_preds in predictions.items():
        for window, keep in (
            ("full", np.ones(len(targets), dtype=bool)),
            ("ex_covid", ev.exclude_covid_mask(targets)),
        ):
            window_targets = targets[keep]
            actual = y.reindex(window_targets).to_numpy()
            benchmark_errors = None
            for model_name, pred in model_preds.items():
                aligned = pred.reindex(window_targets)
                valid = aligned.notna().to_numpy()
                errors = aligned.to_numpy()[valid] - actual[valid]
                if model_name == benchmark:
                    benchmark_errors = errors
                dm_stat = dm_p = dm_p_t = None
                if model_name != benchmark and benchmark_errors is not None:
                    dm = ev.dm_test(errors, benchmark_errors, max_lag=config.dm_max_lag)
                    dm_stat, dm_p, dm_p_t = dm["stat"], dm["p_normal"], dm["p_t"]
                rows.append(
                    ev.metric_row(
                        series=column,
                        horizon=h,
                        window=window,
                        model=model_name,
                        y_true=actual[valid],
                        y_pred=aligned.to_numpy()[valid],
                        scale=scale,
                        dm_stat=dm_stat,
                        dm_p=dm_p,
                        dm_p_t=dm_p_t,
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------
def make_figures(
    y: pd.Series,
    predictions: dict[int, dict[str, pd.Series]],
    intervals: dict[int, dict[str, dict[str, pd.Series]]],
    results: pd.DataFrame,
    column: str,
    config: Config,
) -> list:
    """Generate all per-series figures; returns written paths."""
    figures_dir = config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ylabel = config.spec_for(column).units

    paths.append(
        plots.plot_series_with_splits(
            y,
            train_end=config.train_end,
            val_end=config.val_end,
            title=f"{column} — chronological split (train/val/test)",
            ylabel=ylabel,
            path=figures_dir / f"series_splits_{column}.png",
        )
    )

    for h in config.horizons:
        preds = predictions[h]
        actual = y.reindex(preds[next(iter(preds))].index)
        paths.append(
            plots.plot_forecasts(
                actual,
                preds,
                title=f"{column} — direct {h}-step forecasts, test window",
                ylabel=ylabel,
                path=figures_dir / f"forecasts_{column}_h{h}.png",
            )
        )
        dm_stats = {
            model: float(
                results.loc[
                    (results.series == column)
                    & (results.horizon == h)
                    & (results.window == "full")
                    & (results.model == model),
                    "dm_stat_vs_seasonal_naive",
                ].iloc[0]
            )
            for model in preds
            if model != "seasonal_naive"
        }
        paths.append(
            plots.plot_dm_bars(
                dm_stats,
                title=f"{column}, h={h} — DM statistic vs seasonal-naive (negative = better)",
                path=figures_dir / f"dm_{column}_h{h}.png",
            )
        )
        if "sarima" in intervals[h]:
            bucket = intervals[h]["sarima"]
            paths.append(
                plots.plot_prediction_intervals(
                    actual,
                    preds["sarima"],
                    bucket["lower"],
                    bucket["upper"],
                    title=f"{column} — SARIMA direct {h}-step forecast with 95% interval",
                    ylabel=ylabel,
                    path=figures_dir / f"intervals_{column}_sarima_h{h}.png",
                )
            )

    sarima_h1_errors = (
        predictions[1]["sarima"] - y.reindex(predictions[1]["sarima"].index)
    ).dropna()
    paths.append(
        plots.plot_residual_acf(
            sarima_h1_errors,
            title=f"{column} — SARIMA h=1 test residuals (ACF)",
            path=figures_dir / f"residual_acf_{column}_sarima_h1.png",
        )
    )
    return paths


def predictions_to_long(
    y: pd.Series,
    predictions: dict[int, dict[str, pd.Series]],
    column: str,
) -> pd.DataFrame:
    """Tidy long-format dump of all rolling-origin predictions."""
    records = []
    for h, model_preds in predictions.items():
        for model, pred in model_preds.items():
            for target_date, value in pred.items():
                if np.isnan(value):
                    continue
                records.append(
                    {
                        "series": column,
                        "horizon": h,
                        "model": model,
                        "target_date": target_date,
                        "prediction": value,
                        "actual": y.loc[target_date],
                    }
                )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run(
    config: Config = DEFAULT_CONFIG,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Execute the full study. Returns ``(results, artifacts)``."""
    start = time.time()
    print("== macro-forecasting-lab pipeline ==", flush=True)
    print("[1/6] loading + transforming FRED data ...", flush=True)
    frame = build_modeling_frame(config, refresh=refresh)
    splits = split_chronological(frame, config)
    trainval = pd.concat([splits["train"], splits["val"]])
    coverage = {
        "first": str(frame.index.min().date()),
        "last": str(frame.index.max().date()),
        "n_months": int(len(frame)),
    }
    print(
        f"      coverage {coverage['first']} .. {coverage['last']} "
        f"({coverage['n_months']} months)",
        flush=True,
    )

    results_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, dict] = {}

    for column in config.target_columns:
        print(f"[2/6] target series: {column}", flush=True)
        y = frame[column].dropna()
        diag = stationarity_diagnostics(trainval[column])
        print(f"      stationarity (train+val): {diag}", flush=True)

        print("      AIC grid search for ARIMA / SARIMA on train+val ...", flush=True)
        arima_order = select_arima_order(
            trainval[column], d=int(diag["d"]), pq_grid=config.arima_pq_grid
        )
        sarima_order, sarima_seasonal = select_sarima_order(
            trainval[column],
            d=int(diag["d"]),
            pq_grid=config.arima_pq_grid,
            seasonal_pq_grid=config.sarima_seasonal_pq_grid,
            period=config.seasonal_period,
        )
        print(
            f"      selected ARIMA{arima_order}, SARIMA{sarima_order}{sarima_seasonal}",
            flush=True,
        )
        diagnostics[column] = {
            "adf_p": diag["adf_p"],
            "kpss_p": diag["kpss_p"],
            "d": int(diag["d"]),
            "arima_order": str(arima_order),
            "sarima_order": str(sarima_order),
            "sarima_seasonal": str(sarima_seasonal),
        }

        sup = make_direct_frame(
            y, horizons=config.horizons, lags=config.own_lags, month_dummies=True
        )
        for other in config.target_columns + config.feature_columns:
            if other == column:
                continue
            sup = add_cross_lags(sup, frame[other], prefix=other, lag=config.cross_lag)
        sup = sup.dropna(subset=feature_columns(sup))

        scale = ev.mase_scale(splits["train"][column].dropna(), period=config.seasonal_period)

        print("      rolling-origin evaluation (h=1, h=12) ...", flush=True)
        predictions, intervals = evaluate_series(
            y, sup, config, arima_order, sarima_order, sarima_seasonal
        )
        scored = score_predictions(y, predictions, scale, column, config)
        results_frames.append(scored)
        prediction_frames.append(predictions_to_long(y, predictions, column))

        print("[5/6] figures ...", flush=True)
        make_figures(y, predictions, intervals, pd.concat(results_frames), column, config)

    results = pd.concat(results_frames, ignore_index=True)
    predictions_long = pd.concat(prediction_frames, ignore_index=True)

    print("[6/6] writing reports ...", flush=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.reports_dir / "results.csv", index=False)
    predictions_long.to_csv(config.reports_dir / "predictions.csv", index=False)
    write_summary(results, diagnostics, coverage, config)

    elapsed = time.time() - start
    print(f"done in {elapsed / 60.0:.1f} min", flush=True)
    return results, {"diagnostics": diagnostics, "coverage": coverage}


# --------------------------------------------------------------------------
# Summary writer
# --------------------------------------------------------------------------
def _verdict_lines(results: pd.DataFrame, column: str, horizon: int) -> list[str]:
    full = results.loc[
        (results.series == column) & (results.horizon == horizon) & (results.window == "full")
    ]
    best = full.loc[full.rmse.idxmin()]
    significant = full.loc[(full.dm_p_value < 0.05) & (full.dm_stat_vs_seasonal_naive < 0)]
    lines = [
        f"- **Best model by full-test RMSE:** `{best.model}` "
        f"(RMSE {best.rmse:.3f}, MAE {best.mae:.3f}, MASE {best.mase:.3f}).",
    ]
    if len(significant) > 0:
        names = ", ".join(f"`{m}`" for m in significant.model)
        lines.append(
            f"- **Significantly better than seasonal-naive at 5% (DM, full window):** {names}."
        )
    else:
        lines.append(
            "- **Significantly better than seasonal-naive at 5% (DM, full window):** none."
        )
    return lines


def write_summary(
    results: pd.DataFrame,
    diagnostics: dict[str, dict],
    coverage: dict,
    config: Config,
) -> None:
    """Write ``reports/summary.md`` with actual numbers from the run."""
    lines = [
        "# macro-forecasting-lab — experiment summary",
        "",
        f"_Generated automatically by the pipeline. Data coverage: "
        f"{coverage['first']} .. {coverage['last']} ({coverage['n_months']} months)._",
        "",
        "Split: train <= 2009-12, validation 2010-01..2015-12, test 2016-01..latest.",
        "Windows: `full` = whole test window; `ex_covid` = test window excluding "
        "2020-02..2020-12 (COVID-19 structural break).",
        "",
    ]

    for column in config.target_columns:
        diag = diagnostics[column]
        lines += [
            f"## {column}",
            "",
            f"- ADF p = {diag['adf_p']:.4g}, KPSS p = {diag['kpss_p']:.4g} "
            f"(train+val) -> integration order d = {diag['d']}.",
            f"- Selected ARIMA{diag['arima_order']}; SARIMA{diag['sarima_order']}"
            f"{diag['sarima_seasonal']} (AIC on train+val).",
            "",
        ]
        for horizon in config.horizons:
            for window_name in ("full", "ex_covid"):
                table = results.loc[
                    (results.series == column)
                    & (results.horizon == horizon)
                    & (results.window == window_name),
                    ["model", "n_obs", "rmse", "mae", "mape_pct", "mase",
                     "dm_stat_vs_seasonal_naive", "dm_p_value"],
                ].sort_values("rmse")
                lines += [
                    f"### {column} — h={horizon}, window={window_name}",
                    "",
                    "| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |",
                    "|---|---|---|---|---|---|---|---|",
                ]
                for _, row in table.iterrows():
                    dm = (
                        f"{row.dm_stat_vs_seasonal_naive:+.2f}"
                        if pd.notna(row.dm_stat_vs_seasonal_naive)
                        else "—"
                    )
                    dm_p = f"{row.dm_p_value:.3f}" if pd.notna(row.dm_p_value) else "—"
                    lines.append(
                        f"| {row.model} | {int(row.n_obs)} | {row.rmse:.3f} | {row.mae:.3f} | "
                        f"{row.mape_pct:.2f} | {row.mase:.3f} | {dm} | {dm_p} |"
                    )
                lines.append("")
            lines += _verdict_lines(results, column, horizon)
            lines.append("")

    lines += [
        "## Honesty notes",
        "",
        "- DM p-values are raw; with 7 models x 2 series x 2 horizons compared against the",
        "  same benchmark, a Bonferroni-corrected threshold (~0.05/14 ≈ 0.0036 per window)",
        "  is the appropriate reading — see README §14.",
        "- MAPE is unstable for inflation near zero (deflation episodes); rely on MASE.",
        "- MASE is scaled by the in-sample seasonal-naive MAE on the training window",
        "  (Hyndman & Athanasopoulos, §5.8).",
        "",
    ]

    with open(config.reports_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

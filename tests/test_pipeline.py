"""Pipeline scoring tests: DM coverage for every non-benchmark model
(including naive_last), bandwidth resolution, and sensitivity columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_forecasting_lab.config import Config
from macro_forecasting_lab.pipeline import score_predictions

MODELS = [
    "naive_last",
    "seasonal_naive",
    "rolling_mean_12",
    "arima",
    "sarima",
    "ols",
    "random_forest",
    "gradient_boosting",
]


def _make_inputs() -> tuple[pd.Series, dict[int, dict[str, pd.Series]]]:
    rng = np.random.default_rng(42)
    index = pd.date_range("2015-01-01", periods=96, freq="MS")  # 2015-01..2022-12
    y = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.5, len(index))), index=index)
    test_start = pd.Timestamp("2016-01-01")
    targets = index[index >= test_start][::3]

    predictions: dict[int, dict[str, pd.Series]] = {1: {}, 12: {}}
    for h in (1, 12):
        for i, model in enumerate(MODELS):
            noise = rng.normal(0.0, 0.3 * (1 + i), len(targets))
            predictions[h][model] = pd.Series(
                y.reindex(targets).to_numpy() + noise, index=targets
            )
    return y, predictions


def _auto_config() -> Config:
    return Config(
        train_end="2015-12-31",
        val_start="2016-01-01",
        val_end="2016-01-01",
        test_start="2016-01-01",
        horizons=(1, 12),
        target_step=3,
        dm_max_lag="auto",
    )


def test_all_non_benchmark_models_get_dm_including_naive_last() -> None:
    y, predictions = _make_inputs()
    results = score_predictions(y, predictions, scale=1.0, column="x", config=_auto_config())

    non_benchmark = [m for m in MODELS if m != "seasonal_naive"]
    for h in (1, 12):
        for window in ("full", "ex_covid"):
            block = results[(results.horizon == h) & (results.window == window)]
            assert len(block) == len(MODELS)
            tested = block[block.dm_p_value.notna()].model.tolist()
            assert sorted(tested) == sorted(non_benchmark), (
                f"h={h}, window={window}: DM must be computed for every "
                f"non-benchmark model (regression guard for the naive_last "
                f"dict-order bug), got {tested}"
            )
            # 7 DM comparisons per series/horizon/window with 8 models.
            assert block.dm_p_value.notna().sum() == 7
            benchmark_rows = block[block.model == "seasonal_naive"]
            assert benchmark_rows.dm_p_value.isna().all()


def test_auto_bandwidth_is_h_minus_1() -> None:
    """Primary DM lag must resolve to h-1 (0 at h=1, 11 at h=12)."""
    y, predictions = _make_inputs()
    results = score_predictions(y, predictions, scale=1.0, column="x", config=_auto_config())
    for h in (1, 12):
        used = results[(results.horizon == h) & results.dm_max_lag.notna()].dm_max_lag
        assert (used == h - 1).all(), f"h={h}: expected lag {h - 1}, got {sorted(set(used))}"


def test_forced_lag1_matches_sensitivity_columns() -> None:
    """With dm_max_lag forced to 1, the primary DM equals the lag-1 sensitivity."""
    y, predictions = _make_inputs()
    config = Config(
        train_end="2015-12-31",
        val_start="2016-01-01",
        val_end="2016-01-01",
        test_start="2016-01-01",
        horizons=(1, 12),
        target_step=3,
        dm_max_lag=1,
    )
    results = score_predictions(y, predictions, scale=1.0, column="x", config=config)
    tested = results[results.dm_p_value.notna()]
    assert len(tested) > 0
    assert (tested.dm_max_lag == 1).all()
    assert tested.dm_p_value.equals(tested.dm_p_value_lag1)
    assert tested.dm_stat_vs_seasonal_naive.equals(
        tested.dm_stat_vs_seasonal_naive_lag1
    )


def test_metrics_unchanged_by_dm_scoring() -> None:
    """RMSE/MASE columns must not depend on the DM bandwidth setting."""
    y, predictions = _make_inputs()
    auto = score_predictions(y, predictions, scale=1.0, column="x", config=_auto_config())
    forced = score_predictions(
        y,
        predictions,
        scale=1.0,
        column="x",
        config=Config(
            train_end="2015-12-31",
            val_start="2016-01-01",
            val_end="2016-01-01",
            test_start="2016-01-01",
            horizons=(1, 12),
            target_step=3,
            dm_max_lag=1,
        ),
    )
    key = ["series", "horizon", "window", "model"]
    merged = auto.merge(
        forced, on=key, suffixes=("_auto", "_forced"), validate="one_to_one"
    )
    assert np.allclose(merged.rmse_auto, merged.rmse_forced)
    assert np.allclose(merged.mase_auto, merged.mase_forced)
    # The h=12 DM stats must differ between bandwidths on this autocorrelated
    # synthetic data (guarding that both specifications are really computed).
    h12 = merged[(merged.horizon == 12) & merged.dm_p_value_auto.notna()]
    assert not np.allclose(
        h12.dm_stat_vs_seasonal_naive_auto,
        h12.dm_stat_vs_seasonal_naive_forced,
    )


def test_missing_benchmark_degrades_gracefully() -> None:
    """Without a benchmark series present, rows score but carry no DM values."""
    y, predictions = _make_inputs()
    for h in (1, 12):
        predictions[h].pop("seasonal_naive")
    results = score_predictions(y, predictions, scale=1.0, column="x", config=_auto_config())
    assert results.dm_p_value.isna().all()
    assert len(results) == 7 * 2 * 2  # 7 remaining models x 2 horizons x 2 windows

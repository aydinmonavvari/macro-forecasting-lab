"""Central configuration for macro-forecasting-lab.

This module is the single source of truth for series definitions,
transformations, split dates, horizons, and paths. ``configs/default.yaml``
mirrors these values for CLI overrides; the dataclass below is what the code
actually consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Structural break inside the test window: the COVID-19 shock.
COVID_WINDOW_START = "2020-02-01"
COVID_WINDOW_END = "2020-12-31"


@dataclass(frozen=True)
class SeriesSpec:
    """One FRED series and the transformation applied to it.

    Attributes:
        fred_id: FRED series identifier, e.g. ``CPIAUCSL``.
        column: Column name used in the modeling frame.
        transform: ``"level"`` or ``"yoy_log_diff_pct"`` (100 * 12-month log
            difference, an approximation of the year-over-year percent change).
        units: Human-readable units of the transformed series.
        description: Short description including the underlying raw series.
        is_target: Whether this series receives a full forecasting study.
    """

    fred_id: str
    column: str
    transform: str
    units: str
    description: str
    is_target: bool = False


@dataclass(frozen=True)
class Config:
    """Experiment configuration."""

    # --- data ---------------------------------------------------------------
    series: tuple[SeriesSpec, ...] = field(
        default_factory=lambda: (
            SeriesSpec(
                fred_id="CPIAUCSL",
                column="inflation_yoy",
                transform="yoy_log_diff_pct",
                units="percent (12-month log difference x100)",
                description="Consumer Price Index for All Urban Consumers "
                "(seasonally adjusted); transformed to year-over-year inflation.",
                is_target=True,
            ),
            SeriesSpec(
                fred_id="UNRATE",
                column="unemployment_rate",
                transform="level",
                units="percent of labor force",
                description="Civilian unemployment rate (seasonally adjusted), used in levels.",
                is_target=True,
            ),
            SeriesSpec(
                fred_id="INDPRO",
                column="industrial_production_yoy",
                transform="yoy_log_diff_pct",
                units="percent (12-month log difference x100)",
                description="Industrial Production Index (seasonally adjusted); "
                "transformed to year-over-year growth. Feature-only.",
            ),
            SeriesSpec(
                fred_id="FEDFUNDS",
                column="fed_funds_rate",
                transform="level",
                units="percent per year (monthly average)",
                description="Effective federal funds rate, monthly average. Feature-only. "
                "The daily series DFF is an alternative; FEDFUNDS is chosen because it is "
                "the canonical monthly series and aligns natively with our monthly frequency.",
            ),
            SeriesSpec(
                fred_id="M2SL",
                column="m2_yoy",
                transform="yoy_log_diff_pct",
                units="percent (12-month log difference x100)",
                description="M2 money stock (seasonally adjusted); transformed to "
                "year-over-year growth. Feature-only.",
            ),
        )
    )

    # --- chronological split -------------------------------------------------
    train_end: str = "2009-12-31"
    val_start: str = "2010-01-01"
    val_end: str = "2015-12-31"
    test_start: str = "2016-01-01"

    # --- experiment ------------------------------------------------------------
    horizons: tuple[int, ...] = (1, 12)
    own_lags: tuple[int, ...] = (0, 1, 2, 3, 6, 12)
    cross_lag: int = 1  # auxiliary series enter supervised feature sets lagged by 1 month
    seasonal_period: int = 12
    arima_pq_grid: tuple[int, ...] = (0, 1, 2)
    sarima_seasonal_pq_grid: tuple[int, ...] = (0, 1)
    seed: int = 42
    n_estimators: int = 200
    # Rolling-origin evaluation step (Tashman 2000): score every k-th test month.
    target_step: int = 3
    confidence_level: float = 0.95
    dm_max_lag: int = 1  # HAC bandwidth for the Diebold-Mariano variance (lag-1)

    # --- paths -----------------------------------------------------------------
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    reports_dir: Path = REPO_ROOT / "reports"
    figures_dir: Path = REPO_ROOT / "figures"

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Columns that receive the full forecasting study."""
        return tuple(s.column for s in self.series if s.is_target)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Auxiliary columns available as supervised-model features."""
        return tuple(s.column for s in self.series if not s.is_target)

    def spec_for(self, column: str) -> SeriesSpec:
        """Return the :class:`SeriesSpec` whose modeling column is ``column``."""
        for spec in self.series:
            if spec.column == column:
                return spec
        msg = f"No SeriesSpec defines column {column!r}"
        raise KeyError(msg)


DEFAULT_CONFIG = Config()

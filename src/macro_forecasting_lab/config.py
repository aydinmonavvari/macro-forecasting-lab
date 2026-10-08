"""Central configuration for macro-forecasting-lab.

This module is the single source of truth for series definitions,
transformations, split dates, horizons, and paths. ``configs/default.yaml``
mirrors these values as flat top-level keys for CLI overrides; the dataclass
below is what the code actually consumes. :func:`config_from_yaml` is a
strict loader: every YAML key must be a :class:`Config` field and unknown
keys raise ``ValueError`` (typos are never silently ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

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
    # HAC bandwidth for the Diebold-Mariano variance. "auto" (default; the
    # sentinel -1 is accepted as an alias) resolves to h-1 lags for h-step
    # forecasts: overlapping h-step forecast errors follow an MA(h-1) process.
    # An integer >= 1 forces a fixed bandwidth for every horizon (sensitivity
    # runs; the pre-2026-10 default of a fixed lag 1 is kept available as a
    # reported sensitivity specification).
    dm_max_lag: int | str = "auto"

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


# --------------------------------------------------------------------------
# Validation + strict YAML loading
# --------------------------------------------------------------------------
def normalize_dm_max_lag(value: int | str) -> int | str:
    """Validate/normalize a ``dm_max_lag`` setting.

    Returns ``"auto"`` (bandwidth = horizon - 1, resolved per horizon at test
    time) or a forced integer bandwidth >= 1. ``-1`` is accepted as an alias
    for ``"auto"``. Anything else raises ``ValueError``.
    """
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        msg = (
            f"dm_max_lag must be 'auto', -1, or an integer >= 1; got {value!r}"
        )
        raise ValueError(msg)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"dm_max_lag must be 'auto', -1, or an integer >= 1; got {value!r}"
        raise ValueError(msg)
    if value == -1:
        return "auto"
    if value >= 1:
        return int(value)
    msg = f"dm_max_lag must be 'auto', -1, or an integer >= 1; got {value!r}"
    raise ValueError(msg)


_DATE_KEYS = ("train_end", "val_start", "val_end", "test_start")
_TUPLE_KEYS = (
    "horizons",
    "own_lags",
    "arima_pq_grid",
    "sarima_seasonal_pq_grid",
)
_INT_KEYS = (
    "cross_lag",
    "seasonal_period",
    "seed",
    "n_estimators",
    "target_step",
)


def config_from_yaml(path: str | Path) -> Config:
    """Build a :class:`Config` from a YAML file (strict).

    Every top-level YAML key must name a :class:`Config` field (including
    ``series``); unknown keys raise ``ValueError`` so typos can never be
    silently ignored. Known keys are type-checked and range-checked where
    meaningful (dates must be ISO, ``confidence_level`` in (0, 1),
    ``dm_max_lag`` via :func:`normalize_dm_max_lag`). Missing keys fall back
    to the dataclass defaults.
    """
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Config file {path} must contain a YAML mapping at top level"
        raise ValueError(msg)

    allowed = set(Config.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = (
            f"Unknown config key(s) in {path}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )
        raise ValueError(msg)

    kwargs: dict[str, object] = {}
    if "series" in payload:
        kwargs["series"] = tuple(SeriesSpec(**spec) for spec in payload["series"])
    for key in _DATE_KEYS:
        if key in payload:
            try:
                date.fromisoformat(str(payload[key])[:10])
            except ValueError as exc:
                msg = f"Config key {key!r} must be an ISO date: {exc}"
                raise ValueError(msg) from exc
            kwargs[key] = str(payload[key])
    for key in _TUPLE_KEYS:
        if key in payload:
            values = payload[key]
            if not isinstance(values, (list, tuple)):
                msg = f"Config key {key!r} must be a list of integers"
                raise ValueError(msg)
            try:
                kwargs[key] = tuple(int(v) for v in values)
            except (TypeError, ValueError) as exc:
                msg = f"Config key {key!r} must contain only integers: {exc}"
                raise ValueError(msg) from exc
    for key in _INT_KEYS:
        if key in payload:
            try:
                kwargs[key] = int(payload[key])
            except (TypeError, ValueError) as exc:
                msg = f"Config key {key!r} must be an integer: {exc}"
                raise ValueError(msg) from exc
    if "confidence_level" in payload:
        value = float(payload["confidence_level"])
        if not 0.0 < value < 1.0:
            msg = f"confidence_level must be in (0, 1); got {value}"
            raise ValueError(msg)
        kwargs["confidence_level"] = value
    if "dm_max_lag" in payload:
        kwargs["dm_max_lag"] = normalize_dm_max_lag(payload["dm_max_lag"])

    return Config(**kwargs)

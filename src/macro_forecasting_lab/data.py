"""FRED data acquisition, caching, transformation, and validation.

Two acquisition paths are supported:

1. **Default — public CSV endpoint** (no key required):
   ``https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>``
   This is the official FRED graph download endpoint and returns the full
   history of a series as CSV.

2. **Optional — official FRED API** (only when ``FRED_API_KEY`` is present in
   the environment): ``https://api.stlouisfed.org/fred/series/observations``.
   The key is read from the environment at call time and is *never* hardcoded,
   logged, or committed.

Downloaded series are cached as CSV under ``data/raw/``; the transformed
modeling frame is cached under ``data/processed/``. Datasets are never
committed to version control.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from macro_forecasting_lab.config import DEFAULT_CONFIG, Config

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
USER_AGENT = (
    "macro-forecasting-lab/1.0 (educational research; "
    "https://github.com/aydinmonavvari/macro-forecasting-lab)"
)


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """Fetch a URL with a descriptive User-Agent (FRED asks for identification)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def download_fred_csv(series_id: str, timeout: float = 60.0) -> pd.Series:
    """Download one series through the public CSV endpoint (no API key needed)."""
    url = f"{FRED_CSV_URL}?id={series_id}"
    raw = _http_get(url, timeout=timeout)
    frame = pd.read_csv(pd.io.common.BytesIO(raw))
    date_col = frame.columns[0]
    value_col = frame.columns[-1]
    series = pd.Series(
        frame[value_col].astype(str).to_numpy(),
        index=pd.to_datetime(frame[date_col]),
        name=series_id,
    )
    series = series.replace(".", np.nan).astype(float).dropna()
    return _to_monthly(series)


def download_fred_api(series_id: str, api_key: str, timeout: float = 60.0) -> pd.Series:
    """Download one series through the official FRED API (requires a key).

    The API key must be supplied by the caller (read from the ``FRED_API_KEY``
    environment variable upstream); it is never persisted.
    """
    url = f"{FRED_API_URL}?series_id={series_id}&api_key={api_key}&file_type=json"
    payload = json.loads(_http_get(url, timeout=timeout))
    records = [(obs["date"], obs["value"]) for obs in payload["observations"]]
    frame = pd.DataFrame(records, columns=["date", "value"])
    series = pd.Series(
        frame["value"].astype(str).to_numpy(),
        index=pd.to_datetime(frame["date"]),
        name=series_id,
    )
    series = series.replace(".", np.nan).astype(float).dropna()
    return _to_monthly(series)


def _to_monthly(series: pd.Series) -> pd.Series:
    """Normalize the index to monthly periods (first day of month), sorted."""
    series = series.sort_index()
    series.index = series.index.to_period("M").to_timestamp()
    grouped = series.groupby(level=0).mean()  # safe for daily->monthly coercion
    return grouped.sort_index()


def fill_monthly_gaps(series: pd.Series, max_consecutive: int = 2) -> pd.Series:
    """Reindex to a gap-free monthly grid and time-interpolate interior gaps.

    Real releases occasionally skip a month (e.g. BLS did not publish an
    October 2025 CPI/unemployment observation during the federal data
    shutdown). Rather than truncating all series, interior gaps of at most
    ``max_consecutive`` months are linearly interpolated *on the raw level*
    before any transformation; the number and location of filled months is a
    documented data-handling decision (see README §7 and the research report).
    """
    grid = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    reindexed = series.reindex(grid)
    missing = reindexed.isna()
    if not missing.any():
        return reindexed.rename(series.name)
    filled = reindexed.interpolate(method="time", limit=max_consecutive, limit_area="inside")
    if filled.isna().any():
        bad = filled.index[filled.isna()][:3].tolist()
        msg = f"{series.name}: unfillable monthly gaps at {[str(d.date()) for d in bad]}"
        raise ValueError(msg)
    return filled.rename(series.name)


def load_series(
    spec_fred_id: str,
    raw_dir: Path = DEFAULT_CONFIG.raw_dir,
    refresh: bool = False,
    api_key: str | None = None,
) -> pd.Series:
    """Load one FRED series, using and updating the local CSV cache.

    Priority: local cache (unless ``refresh``) -> official API (if a key is
    given) -> public CSV endpoint.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{spec_fred_id}.csv"

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        series = _to_monthly(frame.iloc[:, 0].dropna().rename(spec_fred_id))
        return fill_monthly_gaps(series)

    key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
    if key:
        series = download_fred_api(spec_fred_id, key)
    else:
        series = download_fred_csv(spec_fred_id)

    series.to_csv(cache_path, index=True, index_label="observation_date")
    return fill_monthly_gaps(series)


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
def transform_series(level: pd.Series, transform: str) -> pd.Series:
    """Apply the configured transformation to a raw monthly level series.

    ``yoy_log_diff_pct`` computes ``100 * log(x).diff(12)``, i.e. the
    continuously compounded year-over-year growth rate. We use the log
    difference rather than the exact percent change because it is
    time-additive, symmetric, and the standard convention in macroeconomic
    modeling; for small rates the two coincide to the first order.
    """
    if transform == "level":
        return level.copy()
    if transform == "yoy_log_diff_pct":
        return 100.0 * np.log(level).diff(12)
    msg = f"Unknown transform {transform!r}"
    raise ValueError(msg)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_monthly(series: pd.Series, name: str = "series") -> pd.Series:
    """Validate that a series is a clean, sorted, gap-free monthly series.

    Checks: strictly increasing unique index, contiguous monthly periods, no
    missing values, and no observations dated in the future.
    """
    if series.index.has_duplicates:
        msg = f"{name}: duplicate timestamps in index"
        raise ValueError(msg)
    if not series.index.is_monotonic_increasing:
        msg = f"{name}: index is not sorted"
        raise ValueError(msg)
    periods = series.index.year * 12 + series.index.month
    gaps = np.diff(np.asarray(periods))
    if (gaps != 1).any():
        position = int(np.argmax(gaps != 1))
        bad = series.index[position + 1]
        msg = f"{name}: monthly frequency violated at {bad.date()}"
        raise ValueError(msg)
    if series.isna().any():
        msg = f"{name}: {int(series.isna().sum())} missing values after transform"
        raise ValueError(msg)
    today = pd.Timestamp.today().normalize()
    if series.index.max() > today + pd.Timedelta(days=1):
        msg = f"{name}: index contains future-dated rows (max {series.index.max().date()})"
        raise ValueError(msg)
    return series


# --------------------------------------------------------------------------
# Modeling frame
# --------------------------------------------------------------------------
def build_modeling_frame(
    config: Config = DEFAULT_CONFIG,
    refresh: bool = False,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download, transform, and merge all configured series into one frame.

    Returns a DataFrame indexed by month with one column per
    :class:`~macro_forecasting_lab.config.SeriesSpec`. Rows dropped by the
    12-month differencing are removed and validated; the frame is cached to
    ``data/processed/modeling_frame.csv``.
    """
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / "modeling_frame.csv"

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return frame.asfreq("MS")

    columns: dict[str, pd.Series] = {}
    for spec in config.series:
        level = load_series(spec.fred_id, raw_dir=config.raw_dir, refresh=refresh, api_key=api_key)
        transformed = transform_series(level, spec.transform).dropna()
        validate_monthly(transformed, name=spec.column)
        columns[spec.column] = transformed.rename(spec.column)

    frame = pd.DataFrame(columns).sort_index()
    frame = frame.dropna(how="any")  # align all series to their common coverage
    validate_monthly(frame.iloc[:, 0], name="modeling_frame")
    frame.asfreq("MS").to_csv(cache_path, index=True, index_label="observation_date")
    return frame.asfreq("MS")


def split_chronological(
    frame: pd.DataFrame, config: Config = DEFAULT_CONFIG
) -> dict[str, pd.DataFrame]:
    """Split a monthly frame into strictly ordered train/validation/test parts."""
    train_end = pd.Timestamp(config.train_end)
    val_start = pd.Timestamp(config.val_start)
    val_end = pd.Timestamp(config.val_end)
    test_start = pd.Timestamp(config.test_start)

    train = frame.loc[:train_end]
    val = frame.loc[val_start:val_end]
    test = frame.loc[test_start:]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        msg = "Chronological split produced an empty partition."
        raise ValueError(msg)
    if train.index.max() >= val.index.min():
        msg = "Split violation: train overlaps validation."
        raise ValueError(msg)
    if val.index.max() >= test.index.min():
        msg = "Split violation: validation overlaps test."
        raise ValueError(msg)
    return {"train": train, "val": val, "test": test}

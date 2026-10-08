"""Direct-horizon feature/target construction with explicit leakage discipline.

The forecasting design is **direct multi-step**: for horizon ``h`` the model
target at forecast origin ``t`` is ``y_{t+h}``, and every feature is a function
of observations at or before ``t``. All shifts are explicit (``pandas.shift``)
so the alignment is auditable; a dedicated test asserts
``frame.loc[t, "target_h12"] == y.loc[t + 12 months]``.

Training-set discipline for supervised models: when fitting at origin ``t``,
only rows whose own target was already realized at time ``t`` may be used,
i.e. rows with ``origin + h <= t`` (see :func:`training_mask`).
"""

from __future__ import annotations

import pandas as pd

TARGET_PREFIX = "target_h"


def make_direct_frame(
    y: pd.Series,
    horizons: tuple[int, ...] = (1, 12),
    lags: tuple[int, ...] = (0, 1, 2, 3, 6, 12),
    month_dummies: bool = True,
) -> pd.DataFrame:
    """Build the direct-horizon regression frame for one target series.

    Columns:
      - ``y_l{k}``: the target series shifted by ``k`` months (``k=0`` is the
        contemporaneous value at the forecast origin).
      - ``target_h{h}``: ``y`` shifted by ``-h`` months; the direct target.
      - ``m_{2..12}``: month-of-year dummies (calendar features; January is
        the reference level).
    """
    frame = pd.DataFrame(index=y.index)
    for k in lags:
        if k < 0:
            msg = f"Lags must be non-negative, got {k}"
            raise ValueError(msg)
        frame[f"y_l{k}"] = y.shift(k)
    for h in horizons:
        if h <= 0:
            msg = f"Horizons must be positive, got {h}"
            raise ValueError(msg)
        frame[f"{TARGET_PREFIX}{h}"] = y.shift(-h)
    if month_dummies:
        months = pd.Series(y.index.month, index=y.index)
        dummies = pd.get_dummies(months, prefix="m", drop_first=True).astype(float)
        frame = frame.join(dummies)
    return frame


def add_cross_lags(
    frame: pd.DataFrame, other: pd.Series, prefix: str, lag: int = 1
) -> pd.DataFrame:
    """Add a lagged auxiliary series as extra features (no future data)."""
    if lag < 0:
        msg = f"Lag must be non-negative, got {lag}"
        raise ValueError(msg)
    aligned = other.reindex(frame.index)
    out = frame.copy()
    out[f"{prefix}_lag{lag}"] = aligned.shift(lag)
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All modeling columns except direct targets."""
    return [c for c in frame.columns if not c.startswith(TARGET_PREFIX)]


def target_column(frame: pd.DataFrame, horizon: int) -> str:
    """Name of the direct-target column for ``horizon``; raises if absent."""
    name = f"{TARGET_PREFIX}{horizon}"
    if name not in frame.columns:
        msg = f"Frame has no target column {name!r}; available: {list(frame.columns)}"
        raise KeyError(msg)
    return name


def training_mask(frame: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.Series:
    """Boolean mask over the frame: rows usable to fit a direct-h model at ``origin``.

    A row (with index ``s``) may only be used when its target ``y_{s+h}`` was
    already realized at the forecast origin, i.e. ``s + h months <= origin``.
    This is the direct-strategy analogue of "no future data in the training
    set" and is explicitly unit-tested.
    """
    cutoff = origin - pd.DateOffset(months=horizon)
    return pd.Series(frame.index <= cutoff, index=frame.index)


def assert_target_alignment(y: pd.Series, frame: pd.DataFrame, horizon: int) -> None:
    """Raise ``AssertionError`` unless direct targets align exactly with ``y``.

    For every row with a defined target: ``frame.loc[t, target_h{h}]`` must
    equal ``y`` at ``t + h`` months, and features ``y_l{k}`` must equal ``y``
    at ``t - k`` months. This is the core leakage guard.
    """
    target = target_column(frame, horizon)
    expected = y.shift(-horizon).reindex(frame.index)
    actual = frame[target]
    both = expected.notna() & actual.notna()
    if not expected[both].equals(actual[both]):
        msg = f"target_h{horizon} misaligned with source series"
        raise AssertionError(msg)
    for column in frame.columns:
        if column.startswith("y_l"):
            k = int(column.split("l")[1])
            expected_lag = y.shift(k).reindex(frame.index)
            actual_lag = frame[column]
            both_lag = expected_lag.notna() & actual_lag.notna()
            if not expected_lag[both_lag].equals(actual_lag[both_lag]):
                msg = f"Feature {column} misaligned with source series"
                raise AssertionError(msg)

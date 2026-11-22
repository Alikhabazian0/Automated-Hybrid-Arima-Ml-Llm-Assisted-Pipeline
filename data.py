"""Block 1: validate a numerical time series and reserve a future test period.

Requires Python 3.10+, NumPy and pandas. No fitted transformations are performed
here. The development series will supply the later rolling-validation folds.
"""

from dataclasses import dataclass
from math import ceil
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Sizes are observation counts; test_fraction is rounded upward."""

    test_fraction: float = 0.20
    min_development_size: int = 100
    min_test_size: int = 20

    def __post_init__(self) -> None:
        fraction = self.test_fraction
        if (
            isinstance(fraction, (bool, np.bool_))
            or not isinstance(fraction, Real)
            or not 0 < fraction < 1
        ):
            raise ValueError("test_fraction must be a number between 0 and 1.")
        for name in ("min_development_size", "min_test_size"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class HoldoutSplit:
    """Development is for diagnostics/tuning; test is for final evaluation."""

    development: pd.Series
    test: pd.Series


def validate_series(y: pd.Series) -> pd.Series:
    """Return a float copy with valid, unique timestamps in increasing order.

    Missing observations and irregular spacing are not filled automatically.
    A forecast step will mean the next observation, not a fixed clock duration.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a pandas DatetimeIndex.")
    if y.empty:
        raise ValueError("The time series is empty.")
    if y.index.hasnans:
        raise ValueError("Timestamps contain missing values.")
    if not y.index.is_unique:
        raise ValueError("Timestamps must be unique.")
    if not y.index.is_monotonic_increasing:
        raise ValueError("Sort observations chronologically before validation.")
    if y.isna().any():
        raise ValueError(
            "Target values contain missing values; choose a handling policy first."
        )

    types = pd.api.types
    if (
        types.is_bool_dtype(y.dtype)
        or types.is_complex_dtype(y.dtype)
        or types.is_datetime64_any_dtype(y.dtype)
        or types.is_timedelta64_dtype(y.dtype)
    ):
        raise TypeError(
            "Target values must be real numbers, "
            "not booleans, complex numbers or dates."
        )

    try:
        numeric = pd.to_numeric(y, errors="raise")
        if types.is_complex_dtype(numeric.dtype):
            raise TypeError("Complex targets are not supported.")
        clean = numeric.astype("float64").copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Target values must be convertible to real numbers.") from exc
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Target values contain NaN or infinity.")
    return clean


def load_series_csv(
    path: str | Path,
    *,
    time_column: str,
    target_column: str,
    time_format: str | None = None,
    utc: bool = False,
) -> pd.Series:
    """Load one target and sort by time; input timestamps must parse consistently.

    By default, naive timestamps stay naive and consistent offsets are kept.
    Use utc=True to normalize offset-aware dates, including DST changes, to UTC.
    With utc=True, naive timestamps are interpreted as UTC, not local time.
    """
    if time_column == target_column:
        raise ValueError("time_column and target_column must be different.")
    frame = pd.read_csv(path)
    missing = {time_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    timestamps = pd.to_datetime(
        frame[time_column], format=time_format, errors="raise", utc=utc
    )
    y = pd.Series(
        frame[target_column].to_numpy(copy=True),
        index=pd.DatetimeIndex(timestamps, name=time_column),
        name=target_column,
    ).sort_index()
    return validate_series(y)


def chronological_holdout(
    y: pd.Series,
    config: SplitConfig | None = None,
) -> HoldoutSplit:
    """Reserve the last ceil(n * test_fraction) observations without shuffling."""
    config = SplitConfig() if config is None else config
    clean = validate_series(y)
    n_test = ceil(len(clean) * config.test_fraction)
    n_development = len(clean) - n_test
    if (
        n_development < config.min_development_size
        or n_test < config.min_test_size
    ):
        raise ValueError(
            f"Insufficient observations: development={n_development}, test={n_test}; "
            f"required minimums are {config.min_development_size} and {config.min_test_size}."
        )
    return HoldoutSplit(
        development=clean.iloc[:n_development].copy(),
        test=clean.iloc[n_development:].copy(),
    )

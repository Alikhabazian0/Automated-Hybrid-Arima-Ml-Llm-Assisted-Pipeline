"""Block 2: causal features for one-observation-ahead forecasting.

Keep this file beside data.py. Training rows are indexed by target time:
X.loc[t] uses observations strictly before t, and its label is y.loc[t].
There are no fitted scalers or other learned transformations in this module.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from data import validate_series


@dataclass(frozen=True)
class FeatureConfig:
    """Lags and windows count observations, not clock hours or days."""

    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple[int, ...] = (6, 12, 24)

    def __post_init__(self) -> None:
        for name, minimum in (("lags", 1), ("rolling_windows", 2)):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise TypeError(f"{name} must be a tuple or list of integers.")
            if name == "lags" and not values:
                raise ValueError("At least one lag is required.")
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
                for value in values
            ):
                raise ValueError(f"{name} must contain integers >= {minimum}.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")
            normalized = tuple(sorted(int(value) for value in values))
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class SupervisedData:
    """Features and labels share the same target-time index."""

    X: pd.DataFrame
    y: pd.Series


class LagFeatureBuilder:
    """Create past-only training rows and a row for the next forecast."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = FeatureConfig() if config is None else config

    @property
    def required_history(self) -> int:
        """Number of known observations needed to make one feature row."""
        return max(self.config.lags + self.config.rolling_windows)

    def _features_at_origins(self, y: pd.Series) -> pd.DataFrame:
        """Features at origin t may use y_t to forecast the next observation."""
        columns = {
            f"lag_{lag}": y.shift(lag - 1)
            for lag in self.config.lags
        }
        for window in self.config.rolling_windows:
            rolling = y.rolling(
                window=window, min_periods=window, center=False
            )
            columns[f"rolling_mean_{window}"] = rolling.mean()
            columns[f"rolling_std_{window}"] = rolling.std(ddof=1)
        return pd.DataFrame(columns, index=y.index)

    def make_training_data(self, y: pd.Series) -> SupervisedData:
        """Pair each known target with features from its preceding origin.

        Pass only the history available for the current training fold.
        The first required_history targets are discarded as warm-up.
        """
        clean = validate_series(y)
        if len(clean) <= self.required_history:
            raise ValueError(
                f"Need at least {self.required_history + 1} observations "
                "to construct one labeled training row."
            )
        # Move origin-t features onto the next observation's target-time row.
        X = self._features_at_origins(clean).shift(1)
        X = X.iloc[self.required_history:].copy()
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Feature calculation produced NaN or infinity.")
        target = clean.loc[X.index].copy()
        return SupervisedData(X=X, y=target)

    def next_features(self, history: pd.Series) -> pd.DataFrame:
        """Build one row to predict the observation after the supplied history.

        Its index is the last known timestamp, named forecast_origin. A future
        target timestamp is deliberately not guessed for irregular observations.
        At test time, history may contain only observations already revealed.
        """
        clean = validate_series(history)
        if len(clean) < self.required_history:
            raise ValueError(
                f"Need at least {self.required_history} known observations "
                "to construct the next forecast's features."
            )
        row = self._features_at_origins(clean).iloc[[-1]].copy()
        if not np.isfinite(row.to_numpy()).all():
            raise ValueError("Feature calculation produced NaN or infinity.")
        row.index = row.index.rename("forecast_origin")
        return row

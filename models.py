"""Block 4: common one-step forecasting interface and baseline models.

Requires scikit-learn>=1.8,<2, NumPy and pandas. Keep this file beside data.py
and features.py. A forecast is for the observation after the supplied history.
New observations can update the forecasting context without re-estimating model
parameters; call fit() again only when the evaluation policy permits refitting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import validate_series
from features import FeatureConfig, LagFeatureBuilder


class BaseForecaster(ABC):
    """Template for fit-on-history, predict-one-next-observation models."""

    name = "base"

    def __init__(self) -> None:
        self._fitted_until: pd.Timestamp | None = None

    @property
    def min_train_size(self) -> int:
        return 1

    @property
    def fitted_until(self) -> pd.Timestamp | None:
        """Last timestamp used in the most recent successful fit."""
        return self._fitted_until

    def fit(self, history: pd.Series) -> "BaseForecaster":
        """Estimate parameters using only the supplied history.

        Starting a new fit invalidates the previous fitted state. If fitting
        fails, prediction remains disabled until a successful fit.
        """
        self._fitted_until = None
        clean = validate_series(history)
        if len(clean) < self.min_train_size:
            raise ValueError(
                f"{self.name} needs at least {self.min_train_size} "
                f"training observations; received {len(clean)}."
            )
        self._fit(clean)
        self._fitted_until = clean.index[-1]
        return self

    def predict_next(self, history: pd.Series) -> float:
        """Forecast one observation beyond known history, without refitting."""
        if self._fitted_until is None:
            raise NotFittedError("Call fit() successfully before prediction.")
        clean = validate_series(history)
        try:
            precedes_fit = clean.index[-1] < self._fitted_until
        except TypeError as exc:
            raise ValueError(
                "Training and prediction timestamps need compatible timezones."
            ) from exc
        if precedes_fit:
            raise ValueError(
                "Forecast origin precedes the end of training data. "
                "Fit a fresh model on that earlier training window."
            )
        prediction = float(self._predict_next(clean))
        if not np.isfinite(prediction):
            raise ValueError(f"{self.name} produced a non-finite forecast.")
        return prediction

    @abstractmethod
    def _fit(self, history: pd.Series) -> None:
        """Subclass implementation of parameter estimation."""
        raise NotImplementedError

    @abstractmethod
    def _predict_next(self, history: pd.Series) -> float:
        """Subclass implementation of one-step prediction."""
        raise NotImplementedError


class ZeroForecaster(BaseForecaster):
    """Zero conditional-mean benchmark for returns or forecast errors."""

    name = "zero"

    def _fit(self, history: pd.Series) -> None:
        # This benchmark has no parameters to estimate.
        return None

    def _predict_next(self, history: pd.Series) -> float:
        return 0.0


class PersistenceForecaster(BaseForecaster):
    """Predict the last observed target; random-walk benchmark for price levels."""

    name = "persistence"

    def _fit(self, history: pd.Series) -> None:
        # The latest observation is supplied at each prediction call.
        return None

    def _predict_next(self, history: pd.Series) -> float:
        return float(history.iloc[-1])


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0
    min_training_rows: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("alpha must be a finite positive number.")
        rows = self.min_training_rows
        if (
            isinstance(rows, (bool, np.bool_))
            or not isinstance(rows, Integral)
            or rows < 2
        ):
            raise ValueError("min_training_rows must be an integer >= 2.")


class RidgeForecaster(BaseForecaster):
    """Standardized Ridge regression with causal lag/rolling features.

    The default uses only raw lags, making it a linear autoregressive baseline.
    Supplying rolling standard deviations adds nonlinear functions of history
    as input features, although the regression remains linear in its inputs.
    """

    name = "ridge"

    def __init__(
        self,
        config: RidgeConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = RidgeConfig() if config is None else config
        feature_config = (
            FeatureConfig(rolling_windows=())
            if feature_config is None else feature_config
        )
        self._builder = LagFeatureBuilder(feature_config)
        self._pipeline: Pipeline | None = None

    @property
    def min_train_size(self) -> int:
        return self._builder.required_history + self.config.min_training_rows

    def _fit(self, history: pd.Series) -> None:
        training = self._builder.make_training_data(history)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(self.config.alpha), solver="svd")),
        ])
        pipeline.fit(training.X, training.y)
        self._pipeline = pipeline

    def _predict_next(self, history: pd.Series) -> float:
        if self._pipeline is None:
            raise NotFittedError("The Ridge estimator has not been fitted.")
        row = self._builder.next_features(history)
        return float(self._pipeline.predict(row)[0])

"""Block 6: XGBoost regression with causal features and fixed hyperparameters.

Requires xgboost>=3.1,<4 (or its xgboost-cpu variant), scikit-learn>=1.8,<2,
NumPy and pandas. Keep this file beside models.py, features.py and data.py.
The target scaler is fitted on labeled training rows; forecasts are returned
in original target units. No validation or test data is supplied to training.
"""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import FeatureConfig, LagFeatureBuilder
from models import BaseForecaster


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_training_rows: int = 50
    random_state: int = 42
    n_jobs: int = 1

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_estimators", 1),
            ("max_depth", 1),
            ("min_training_rows", 2),
            ("random_state", 0),
            ("n_jobs", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
            object.__setattr__(self, name, int(value))
        if self.random_state > 2**31 - 1:
            raise ValueError("random_state must be <= 2**31 - 1.")

        for name in ("learning_rate", "subsample", "colsample_bytree"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be a finite number in (0, 1].")
            object.__setattr__(self, name, float(value))

        for name in ("min_child_weight", "reg_alpha", "reg_lambda", "gamma"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number.")
            object.__setattr__(self, name, float(value))


class XGBoostForecaster(BaseForecaster):
    """Predict one next observation; trees and target scaling stay fixed.

    New history changes the latest feature row. A new fit() is required to
    re-estimate the target scaler or train another collection of trees.
    """

    name = "xgboost"

    def __init__(
        self,
        config: XGBoostConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = XGBoostConfig() if config is None else config
        self._builder = LagFeatureBuilder(feature_config)
        self._estimator: TransformedTargetRegressor | None = None
        self._n_training_rows = 0
        self._fitted_config: dict[str, object] = {}
        self._fitted_features: dict[str, object] = {}
        self._fit_warnings: tuple[str, ...] = ()

    @property
    def min_train_size(self) -> int:
        return self._builder.required_history + self.config.min_training_rows

    def _require_fit(self) -> None:
        if self.fitted_until is None or self._estimator is None:
            raise NotFittedError("Call fit() successfully before using XGBoost results.")

    def _fit(self, history: pd.Series) -> None:
        self._estimator = None
        self._n_training_rows = 0
        self._fitted_config = {}
        self._fitted_features = {}
        self._fit_warnings = ()

        training = self._builder.make_training_data(history)
        params = asdict(self.config)
        params.pop("min_training_rows")
        estimator = TransformedTargetRegressor(
            regressor=XGBRegressor(
                objective="reg:squarederror",
                booster="gbtree",
                tree_method="hist",
                device="cpu",
                validate_parameters=True,
                **params,
            ),
            transformer=StandardScaler(),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(training.X, training.y)
            probe = self._predict_original_units(
                estimator, self._builder.next_features(history)
            )
        self._fit_warnings = tuple(str(item.message).strip() for item in captured)
        if not np.isfinite(probe):
            raise ValueError("The fitted XGBoost model produced a non-finite forecast.")

        self._estimator = estimator
        self._n_training_rows = len(training.y)
        self._fitted_config = asdict(self.config)
        self._fitted_features = asdict(self._builder.config)

    @staticmethod
    def _predict_original_units(
        estimator: TransformedTargetRegressor, row: pd.DataFrame
    ) -> float:
        # Decode float32 tree output using the float64 training statistics.
        normalized = float(estimator.regressor_.predict(row)[0])
        scaler = estimator.transformer_
        return float(normalized * scaler.scale_[0] + scaler.mean_[0])

    def _predict_next(self, history: pd.Series) -> float:
        self._require_fit()
        row = self._builder.next_features(history)
        return self._predict_original_units(self._estimator, row)

    def fit_summary(self) -> dict[str, object]:
        """Return JSON-safe fit metadata; prediction does not update this fit."""
        self._require_fit()
        regressor = self._estimator.regressor_
        scaler = self._estimator.transformer_
        return {
            "model": self.name,
            "n_training_rows": self._n_training_rows,
            "fitted_until": self.fitted_until.isoformat(),
            "hyperparameters": dict(self._fitted_config),
            "feature_config": dict(self._fitted_features),
            "feature_names": [str(name) for name in regressor.feature_names_in_],
            "boosting_rounds": int(regressor.get_booster().num_boosted_rounds()),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "target_scaling": {
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
            },
            "warnings": list(self._fit_warnings),
        }

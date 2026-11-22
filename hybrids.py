"""Block 8: additive residual correction with chronological error targets.

The correction model learns e_t = y_t - first_stage_forecast_t from predictions
made using earlier observations. After fitting, the first stage is re-estimated
on the whole supplied training window. Both stages then keep their parameters
fixed until the enclosing hybrid's next fit(). New actuals extend the original
first-stage error history, never the combined hybrid's error history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from features import FeatureConfig
from models import BaseForecaster
from validation import ValidationConfig, walk_forward

if TYPE_CHECKING:
    from arima_model import ARIMAConfig
    from xgboost_model import XGBoostConfig


@dataclass(frozen=True)
class HybridConfig:
    initial_train_size: int = 100
    base_refit_every: int | None = 24
    correction_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("initial_train_size", "base_refit_every"):
            value = getattr(self, name)
            if name == "base_refit_every" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        value = self.correction_weight
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("correction_weight must be a finite number in [0, 1].")
        object.__setattr__(self, "correction_weight", float(value))


def _new_component(factory: Callable[[], BaseForecaster]) -> BaseForecaster:
    component = factory()
    if not isinstance(component, BaseForecaster):
        raise TypeError("Each factory must return a BaseForecaster instance.")
    return component


class ResidualHybrid(BaseForecaster):
    """Generic first-stage forecast plus a weighted forecast of its errors.

    Factories must construct independent, fixed-configuration components.
    Prediction history must extend the previously processed history unchanged.
    A failed component update invalidates the hybrid until a fresh fit().
    """

    def __init__(
        self,
        base_factory: Callable[[], BaseForecaster],
        correction_factory: Callable[[], BaseForecaster],
        config: HybridConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if not callable(base_factory) or not callable(correction_factory):
            raise TypeError("Provide two callable component factories.")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a nonempty string.")
        self.config = HybridConfig() if config is None else config
        self._base_factory = base_factory
        self._correction_factory = correction_factory
        first = _new_component(base_factory)
        second = _new_component(correction_factory)
        if first is second:
            raise ValueError("The two stages must be independent model instances.")
        self.name = name or f"{first.name}_then_{second.name}"
        self._base_minimum = int(first.min_train_size)
        self._inner_initial = max(self.config.initial_train_size, self._base_minimum)
        self._correction_minimum = int(second.min_train_size)
        if self._base_minimum < 1 or self._correction_minimum < 1:
            raise ValueError("Components need positive minimum sample sizes.")
        self._clear_components()

    def _clear_components(self) -> None:
        self._base_model: BaseForecaster | None = None
        self._correction_model: BaseForecaster | None = None
        self._training_errors: pd.Series | None = None
        self._error_history: pd.Series | None = None
        self._observed_history: pd.Series | None = None
        self._last_components: dict[str, object] | None = None
        self._fit_options: dict[str, object] = {}
        self._weight = 1.0
        self._n_training_observations = 0
        self._inner_refits = 0

    @property
    def min_train_size(self) -> int:
        initial = max(self.config.initial_train_size, self._base_minimum)
        return initial + self._correction_minimum

    @property
    def state_until(self) -> pd.Timestamp | None:
        if self.fitted_until is None or self._observed_history is None:
            return None
        return self._observed_history.index[-1]

    def _require_fit(self) -> None:
        if (
            self.fitted_until is None
            or self._base_model is None
            or self._correction_model is None
        ):
            raise NotFittedError("Call fit() successfully before using hybrid results.")

    @property
    def training_errors(self) -> pd.Series:
        """Original chronological errors used to fit the correction model."""
        self._require_fit()
        return self._training_errors.copy()

    @property
    def error_history(self) -> pd.Series:
        """Training errors plus first-stage errors from newly revealed actuals."""
        self._require_fit()
        return self._error_history.copy()

    def _fit(self, history: pd.Series) -> None:
        self._clear_components()
        self._inner_initial = max(self.config.initial_train_size, self._base_minimum)
        inner_config = ValidationConfig(
            initial_train_size=self._inner_initial,
            refit_every=self.config.base_refit_every,
        )
        inner_run = walk_forward(
            self._base_factory, history, inner_config,
            model_name=f"{self.name}_error_generator",
        )
        errors = inner_run.errors.rename("first_stage_error")
        base = _new_component(self._base_factory)
        correction = _new_component(self._correction_factory)
        if base is correction:
            raise ValueError("The two stages must be independent model instances.")
        base.fit(history)
        correction.fit(errors)

        self._base_model = base
        self._correction_model = correction
        self._training_errors = errors.copy()
        self._error_history = errors.copy()
        self._observed_history = history.copy()
        self._n_training_observations = len(history)
        self._inner_refits = inner_run.n_fits
        self._fit_options = asdict(self.config)
        self._weight = self.config.correction_weight

    def _predict_next(self, history: pd.Series) -> float:
        self._require_fit()
        seen = self._observed_history
        n_seen = len(seen)
        if (
            len(history) < n_seen
            or not history.index[:n_seen].equals(seen.index)
            or not np.array_equal(history.iloc[:n_seen].to_numpy(), seen.to_numpy())
        ):
            raise ValueError(
                "History must extend all previously processed timestamps and "
                "values unchanged. Refit before moving backward or revising history."
            )
        if len(history) == n_seen and self._last_components is not None:
            return float(self._last_components["prediction"])

        succeeded = False
        try:
            new_errors = []
            for position in range(n_seen, len(history)):
                if position == n_seen and self._last_components is not None:
                    base_prediction = float(self._last_components["base_prediction"])
                else:
                    base_prediction = self._base_model.predict_next(
                        history.iloc[:position].copy()
                    )
                # Reveal this actual only after its first-stage forecast.
                error = float(history.iloc[position]) - base_prediction
                if not np.isfinite(error):
                    raise ValueError("The first-stage forecast error is not finite.")
                new_errors.append(error)

            error_history = self._error_history
            if new_errors:
                revealed = pd.Series(
                    new_errors,
                    index=history.index[n_seen:].rename(error_history.index.name),
                    name=error_history.name,
                    dtype=float,
                )
                error_history = pd.concat([error_history, revealed])

            base_prediction = self._base_model.predict_next(history)
            correction_prediction = self._correction_model.predict_next(error_history)
            weighted_correction = self._weight * correction_prediction
            prediction = base_prediction + weighted_correction
            if not np.isfinite(prediction):
                raise ValueError("The combined hybrid forecast is not finite.")

            self._error_history = error_history.copy()
            self._observed_history = history.copy()
            self._last_components = {
                "forecast_origin": history.index[-1].isoformat(),
                "base_prediction": float(base_prediction),
                "correction_prediction": float(correction_prediction),
                "correction_weight": float(self._weight),
                "weighted_correction": float(weighted_correction),
                "prediction": float(prediction),
            }
            succeeded = True
            return float(prediction)
        finally:
            # Components may have advanced independently before an exception.
            # Keep a partially updated hybrid unavailable until a fresh fit.
            if not succeeded:
                self._fitted_until = None
                self._last_components = None

    def predict_components(self, history: pd.Series) -> dict[str, object]:
        self.predict_next(history)
        return dict(self._last_components)

    def fit_summary(self) -> dict[str, object]:
        self._require_fit()

        def describe(model):
            method = getattr(model, "fit_summary", None)
            if callable(method):
                return method()
            return {
                "model": model.name,
                "fitted_until": model.fitted_until.isoformat(),
            }

        return {
            "model": self.name,
            "config": dict(self._fit_options),
            "inner_initial_train_size": self._inner_initial,
            "minimum_training_observations": self._inner_initial + self._correction_minimum,
            "n_training_observations": self._n_training_observations,
            "n_training_errors": len(self._training_errors),
            "n_observed_errors": len(self._error_history),
            "inner_refits": self._inner_refits,
            "fitted_until": self.fitted_until.isoformat(),
            "state_until": self.state_until.isoformat(),
            "base_fit": describe(self._base_model),
            "correction_fit": describe(self._correction_model),
            "last_forecast": (
                dict(self._last_components) if self._last_components else None
            ),
        }


class ARIMAXGBoostForecaster(ResidualHybrid):
    """ARIMA forecast plus XGBoost forecast of its chronological errors."""

    def __init__(
        self,
        arima_config: ARIMAConfig | None = None,
        ml_config: XGBoostConfig | None = None,
        feature_config: FeatureConfig | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        from arima_model import ARIMAConfig, ARIMAForecaster
        from xgboost_model import XGBoostConfig, XGBoostForecaster

        arima_config = ARIMAConfig() if arima_config is None else arima_config
        ml_config = XGBoostConfig() if ml_config is None else ml_config
        super().__init__(
            lambda: ARIMAForecaster(arima_config),
            lambda: XGBoostForecaster(ml_config, feature_config),
            config,
            name="arima_to_xgboost",
        )


class XGBoostARIMAForecaster(ResidualHybrid):
    """XGBoost forecast plus ARIMA forecast of its chronological errors."""

    def __init__(
        self,
        ml_config: XGBoostConfig | None = None,
        arima_config: ARIMAConfig | None = None,
        feature_config: FeatureConfig | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        from arima_model import ARIMAConfig, ARIMAForecaster
        from xgboost_model import XGBoostConfig, XGBoostForecaster

        ml_config = XGBoostConfig() if ml_config is None else ml_config
        arima_config = ARIMAConfig() if arima_config is None else arima_config
        super().__init__(
            lambda: XGBoostForecaster(ml_config, feature_config),
            lambda: ARIMAForecaster(arima_config),
            config,
            name="xgboost_to_arima",
        )

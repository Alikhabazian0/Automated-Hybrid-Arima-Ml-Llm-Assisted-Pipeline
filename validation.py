"""Block 7b: one-step chronological validation with an explicit refit policy.

Use only development data while choosing architectures and hyperparameters.
The evaluator supplies past observations to the model, obtains a forecast,
and only then reads the next actual value. A failure invalidates the candidate's
complete run; failed dates are never silently removed from model comparisons.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd

from data import validate_series
from metrics import ForecastMetrics, MetricConfig, score_forecasts
from models import BaseForecaster


@dataclass(frozen=True)
class ValidationConfig:
    initial_train_size: int = 100
    refit_every: int | None = 1
    window_size: int | None = None
    metrics: MetricConfig = field(default_factory=MetricConfig)

    def __post_init__(self) -> None:
        for name in ("initial_train_size", "refit_every", "window_size"):
            value = getattr(self, name)
            if value is None and name != "initial_train_size":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.metrics, MetricConfig):
            raise TypeError("metrics must be a MetricConfig instance.")


@dataclass(frozen=True)
class ValidationResult:
    model_name: str
    config: ValidationConfig
    forecasts: pd.DataFrame
    metrics: ForecastMetrics
    n_fits: int
    fit_seconds: float
    predict_seconds: float
    total_seconds: float

    @property
    def errors(self) -> pd.Series:
        """Chronological forecast errors, indexed by their target times."""
        return self.forecasts["error"].copy().rename("forecast_error")

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "config": asdict(self.config),
            **self.metrics.to_dict(),
            "first_target": self.forecasts.index[0].isoformat(),
            "last_target": self.forecasts.index[-1].isoformat(),
            "n_fits": self.n_fits,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "total_seconds": self.total_seconds,
        }


class ValidationError(RuntimeError):
    """A candidate failed at a particular forecast target and stage."""

    def __init__(self, model_name: str, target_time, stage: str, cause: Exception):
        self.model_name = model_name
        self.target_time = target_time
        self.stage = stage
        super().__init__(
            f"{model_name}: {stage} failed for target {target_time}: "
            f"{type(cause).__name__}: {cause}"
        )


def _check_length(y: pd.Series, config: ValidationConfig) -> None:
    if config.initial_train_size >= len(y):
        raise ValueError("initial_train_size must leave at least one forecast target.")


def _new_model(factory: Callable[[], BaseForecaster]) -> BaseForecaster:
    model = factory()
    if not isinstance(model, BaseForecaster):
        raise TypeError("The model factory must return a BaseForecaster instance.")
    return model


def walk_forward(
    model_factory: Callable[[], BaseForecaster],
    y: pd.Series,
    config: ValidationConfig | None = None,
    *,
    model_name: str | None = None,
) -> ValidationResult:
    """Forecast every observation following the initial training segment.

    refit_every=1 refits at every origin; k refits every k observations; None
    fits once. Factories should construct fresh models. window_size limits the
    parameter-estimation sample at refits. Between refits, forecasting context
    grows from that fit's start, preserving the ARIMA state-update contract.
    """
    started = perf_counter()
    config = ValidationConfig() if config is None else config
    clean = validate_series(y)
    _check_length(clean, config)
    if not callable(model_factory):
        raise TypeError("model_factory must be callable.")
    if model_name is not None and (not isinstance(model_name, str) or not model_name):
        raise ValueError("model_name must be a nonempty string.")
    label = model_name or "candidate"
    initial = config.initial_train_size
    fit_start = 0 if config.window_size is None else max(0, initial - config.window_size)
    reference = clean.iloc[fit_start:initial].copy()
    model = None
    records = []
    n_fits = 0
    fit_total = predict_total = 0.0
    training_size = 0

    for position in range(initial, len(clean)):
        target_time = clean.index[position]
        refitted = position == initial or (
            config.refit_every is not None
            and (position - initial) % config.refit_every == 0
        )
        fit_elapsed = 0.0
        if refitted:
            fit_start = (
                0 if config.window_size is None
                else max(0, position - config.window_size)
            )
            training = clean.iloc[fit_start:position].copy()
            fit_started = perf_counter()
            try:
                model = _new_model(model_factory)
                if model_name is None:
                    label = model.name
                model.fit(training)
            except Exception as exc:
                raise ValidationError(label, target_time, "fit", exc) from exc
            fit_elapsed = perf_counter() - fit_started
            fit_total += fit_elapsed
            n_fits += 1
            training_size = len(training)

        history = clean.iloc[fit_start:position].copy()
        predict_started = perf_counter()
        try:
            prediction = float(model.predict_next(history))
            if not np.isfinite(prediction):
                raise ValueError("The model produced a non-finite forecast.")
            if model.fitted_until is None or model.fitted_until > history.index[-1]:
                raise ValueError("The model's fit endpoint exceeds the forecast origin.")
        except Exception as exc:
            raise ValidationError(label, target_time, "predict", exc) from exc
        predict_elapsed = perf_counter() - predict_started
        predict_total += predict_elapsed

        # The target is revealed only after prediction.
        actual = float(clean.iloc[position])
        error = actual - prediction
        if not np.isfinite(error):
            raise ValidationError(
                label, target_time, "score", ValueError("Non-finite forecast error.")
            )
        records.append({
            "forecast_origin": history.index[-1],
            "actual": actual,
            "prediction": prediction,
            "error": error,
            "last_observed": float(history.iloc[-1]),
            "fit_start": clean.index[fit_start],
            "fit_end": model.fitted_until,
            "training_size": training_size,
            "refitted": refitted,
            "fit_seconds": fit_elapsed,
            "predict_seconds": predict_elapsed,
        })

    forecasts = pd.DataFrame(
        records,
        index=clean.index[initial:].rename("target_time"),
    )
    try:
        metrics = score_forecasts(
            forecasts["actual"],
            forecasts["prediction"],
            reference=reference,
            last_observed=forecasts["last_observed"],
            config=config.metrics,
        )
    except Exception as exc:
        raise ValidationError(label, forecasts.index[-1], "score", exc) from exc
    return ValidationResult(
        model_name=label,
        config=config,
        forecasts=forecasts,
        metrics=metrics,
        n_fits=n_fits,
        fit_seconds=float(fit_total),
        predict_seconds=float(predict_total),
        total_seconds=float(perf_counter() - started),
    )


def compare_models(
    model_factories: Mapping[str, Callable[[], BaseForecaster]],
    y: pd.Series,
    config: ValidationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, ValidationResult]]:
    """Rank complete runs by RMSE/MAE; retain failed candidates in the table."""
    if not isinstance(model_factories, Mapping) or not model_factories:
        raise ValueError("Provide a nonempty mapping of candidate names to factories.")
    if any(not isinstance(name, str) or not name for name in model_factories):
        raise ValueError("Candidate names must be nonempty strings.")
    config = ValidationConfig() if config is None else config
    clean = validate_series(y)
    _check_length(clean, config)
    runs: dict[str, ValidationResult] = {}
    rows = []
    for name, factory in model_factories.items():
        try:
            result = walk_forward(factory, clean, config, model_name=name)
        except (ValidationError, ValueError, TypeError, ArithmeticError) as exc:
            rows.append({
                "model": name, "status": "failed", "error": str(exc),
                "rmse": None, "mae": None, "n_observations": None,
            })
            continue
        runs[name] = result
        rows.append({
            "model": name, "status": "ok", "error": "", **result.metrics.to_dict(),
            "n_fits": result.n_fits, "fit_seconds": result.fit_seconds,
            "predict_seconds": result.predict_seconds, "total_seconds": result.total_seconds,
        })
    ranking = pd.DataFrame(rows).set_index("model").sort_values(
        ["rmse", "mae"], na_position="last", kind="stable"
    )
    return ranking, runs

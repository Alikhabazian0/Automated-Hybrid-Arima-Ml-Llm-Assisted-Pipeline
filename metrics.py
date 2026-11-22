"""Block 7a: aligned point-forecast metrics in original target units.

Directional accuracy uses three classes (-1, 0, +1). Choose returns for signs
of the target itself, levels for changes from the last known observation, or
None to disable directional scoring. Scaled-error denominators use only the
initial training reference, never the validation or test observations.
"""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd

from data import validate_series


@dataclass(frozen=True)
class MetricConfig:
    direction_mode: str | None = None
    zero_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.direction_mode not in (None, "returns", "levels"):
            raise ValueError("direction_mode must be None, 'returns' or 'levels'.")
        value = self.zero_tolerance
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("zero_tolerance must be a finite nonnegative number.")
        object.__setattr__(self, "zero_tolerance", float(value))


@dataclass(frozen=True)
class ForecastMetrics:
    n_observations: int
    rmse: float
    mae: float
    mean_error: float
    mase: float | None
    rmsse: float | None
    directional_accuracy: float | None
    directional_coverage: float | None
    directional_accuracy_active: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _error_statistics(errors: np.ndarray) -> tuple[float, float, float]:
    """Compute RMSE, MAE and bias without squaring large raw errors."""
    if not np.isfinite(errors).all():
        raise ValueError("Forecast differences contain non-finite values.")
    scale = float(np.max(np.abs(errors)))
    if scale == 0:
        return 0.0, 0.0, 0.0
    normalized = errors / scale
    return (
        float(scale * np.sqrt(np.mean(normalized ** 2))),
        float(scale * np.mean(np.abs(normalized))),
        float(scale * np.mean(normalized)),
    )


def _aligned(y: pd.Series, index: pd.DatetimeIndex, name: str) -> pd.Series:
    clean = validate_series(y)
    if not clean.index.equals(index):
        raise ValueError(f"{name} timestamps must exactly match forecast targets.")
    return clean


def score_forecasts(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    reference: pd.Series | None = None,
    last_observed: pd.Series | None = None,
    config: MetricConfig | None = None,
) -> ForecastMetrics:
    """Score matching target-time rows; no automatic alignment or row dropping.

    For levels, last_observed must contain each forecast origin's last known
    value, indexed by the corresponding target time. Directional scores and
    coverage are fractions in [0, 1]. An inactive prediction has direction zero.
    """
    config = MetricConfig() if config is None else config
    truth = validate_series(actual)
    forecast = _aligned(predicted, truth.index, "predicted")
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = truth.to_numpy() - forecast.to_numpy()
    except FloatingPointError as exc:
        raise ValueError("Forecast differences exceed numerical limits.") from exc
    rmse, mae, bias = _error_statistics(errors)

    mase = rmsse = None
    if reference is not None:
        training = validate_series(reference)
        if training.index[-1] >= truth.index[0]:
            raise ValueError("The scale reference must precede all scored targets.")
        if len(training) > 1:
            with np.errstate(over="raise", invalid="raise"):
                baseline_errors = np.diff(training.to_numpy())
            baseline_rmse, baseline_mae, _ = _error_statistics(baseline_errors)
            if baseline_mae > 0:
                value = mae / baseline_mae
                mase = float(value) if np.isfinite(value) else None
            if baseline_rmse > 0:
                value = rmse / baseline_rmse
                rmsse = float(value) if np.isfinite(value) else None

    accuracy = coverage = active_accuracy = None
    if config.direction_mode is not None:
        observed_values = truth.to_numpy()
        predicted_values = forecast.to_numpy()
        if config.direction_mode == "levels":
            if last_observed is None:
                raise ValueError("levels directional scoring needs last_observed.")
            previous = _aligned(last_observed, truth.index, "last_observed")
            with np.errstate(over="raise", invalid="raise"):
                observed_values = observed_values - previous.to_numpy()
                predicted_values = predicted_values - previous.to_numpy()
        tolerance = config.zero_tolerance
        true_direction = np.where(
            observed_values > tolerance, 1,
            np.where(observed_values < -tolerance, -1, 0),
        )
        predicted_direction = np.where(
            predicted_values > tolerance, 1,
            np.where(predicted_values < -tolerance, -1, 0),
        )
        matches = true_direction == predicted_direction
        active = predicted_direction != 0
        accuracy = float(np.mean(matches))
        coverage = float(np.mean(active))
        if np.any(active):
            active_accuracy = float(np.mean(matches[active]))

    return ForecastMetrics(
        n_observations=len(truth),
        rmse=rmse,
        mae=mae,
        mean_error=bias,
        mase=mase,
        rmsse=rmsse,
        directional_accuracy=accuracy,
        directional_coverage=coverage,
        directional_accuracy_active=active_accuracy,
    )

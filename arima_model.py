"""Block 5: non-seasonal ARIMA / ARMA with fixed-order estimation.

Requires statsmodels>=0.15,<0.16 and the existing models.py interface. Forecast
steps count observations. Supply the fitted history, extended by observations
already revealed, on each prediction call. State updates do not re-estimate
parameters; a new fit() explicitly resets estimation and forecasting history.
"""

from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.arima.model import ARIMA

from models import BaseForecaster


@dataclass(frozen=True)
class ARIMAConfig:
    order: tuple[int, int, int] = (1, 0, 1)
    trend: str | None = None
    min_training_size: int = 50
    maxiter: int = 200
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.order, (tuple, list)) or len(self.order) != 3:
            raise ValueError("order must contain exactly three integers: (p, d, q).")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 0
            for value in self.order
        ):
            raise ValueError("p, d and q must be nonnegative integers.")
        object.__setattr__(self, "order", tuple(int(v) for v in self.order))
        if self.trend not in (None, "n", "c", "t", "ct"):
            raise ValueError("trend must be None, 'n', 'c', 't' or 'ct'.")
        d = self.order[1]
        if d > 0 and self.trend in ("c", "ct"):
            raise ValueError("A constant trend is incompatible with differencing.")
        if d > 1 and self.trend == "t":
            raise ValueError("A linear trend requires d <= 1 in this version.")
        for name, minimum in (("min_training_size", 10), ("maxiter", 1)):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("enforce_stationarity", "enforce_invertibility"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be Boolean.")
            object.__setattr__(self, name, bool(getattr(self, name)))


class ARIMAForecaster(BaseForecaster):
    """Estimate coefficients at fit(), then update states with known data.

    Prediction history must start with all previously processed observations,
    unchanged and in the same order. Use a fresh fit for each validation fold.
    """

    name = "arima"

    def __init__(self, config: ARIMAConfig | None = None) -> None:
        super().__init__()
        self.config = ARIMAConfig() if config is None else config
        self._fit_result = None
        self._state_result = None
        self._fit_history: pd.Series | None = None
        self._observed_history: pd.Series | None = None
        self._fit_warnings: tuple[str, ...] = ()

    @property
    def min_train_size(self) -> int:
        # Structural guard only; statistical adequacy depends on the data.
        return max(self.config.min_training_size, sum(self.config.order) + 5)

    @property
    def state_until(self) -> pd.Timestamp | None:
        if self.fitted_until is None or self._observed_history is None:
            return None
        return self._observed_history.index[-1]

    def _require_fit(self) -> None:
        if self.fitted_until is None or self._fit_result is None:
            raise NotFittedError("Call fit() successfully before using ARIMA results.")

    def _fit(self, history: pd.Series) -> None:
        self._fit_result = None
        self._state_result = None
        self._fit_history = None
        self._observed_history = None
        self._fit_warnings = ()

        trend = self.config.trend
        if trend is None:
            trend = "c" if self.config.order[1] == 0 else "n"

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = ARIMA(
                history.to_numpy(),
                order=self.config.order,
                trend=trend,
                enforce_stationarity=self.config.enforce_stationarity,
                enforce_invertibility=self.config.enforce_invertibility,
                missing="raise",
            )
            result = model.fit(
                method="statespace",
                method_kwargs={"maxiter": int(self.config.maxiter), "disp": 0},
            )

        self._fit_warnings = tuple(str(item.message).strip() for item in captured)
        if not bool(result.mle_retvals.get("converged", False)):
            raise RuntimeError("ARIMA optimization did not converge.")
        if not np.isfinite(np.asarray(result.params)).all():
            raise ValueError("ARIMA produced non-finite estimated parameters.")
        if not np.isfinite(np.asarray(result.forecast(steps=1))).all():
            raise ValueError("The fitted ARIMA model produced a non-finite forecast.")

        self._fit_result = result
        self._state_result = result
        self._fit_history = history.copy()
        self._observed_history = history.copy()

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

        candidate = self._state_result
        if len(history) > n_seen:
            # Keep the complete time index, including deterministic trends.
            # This re-filters the accumulated data but does not re-fit parameters.
            candidate = candidate.append(
                history.iloc[n_seen:].to_numpy(),
                refit=False,
            )

        prediction = float(np.asarray(candidate.forecast(steps=1)).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise ValueError("Updated ARIMA state produced a non-finite forecast.")
        # Commit only after a successful forecast; failed updates leave state intact.
        if len(history) > n_seen:
            self._state_result = candidate
            self._observed_history = history.copy()
        return prediction

    def fitted_residuals(self, burn_in: int | None = None) -> pd.Series:
        """Return in-sample training innovations for diagnostics.

        Parameters were estimated on the entire supplied fitting window. Hybrid
        training errors must instead be generated by chronological forecasting.
        """
        self._require_fit()
        if burn_in is None:
            burn_in = int(self._fit_result.loglikelihood_burn)
        if (
            isinstance(burn_in, (bool, np.bool_))
            or not isinstance(burn_in, Integral)
            or not 0 <= burn_in < len(self._fit_history)
        ):
            raise ValueError("burn_in must be an integer within the training window.")
        residuals = pd.Series(
            np.asarray(self._fit_result.resid).reshape(-1),
            index=self._fit_history.index,
            name="arima_fitted_residual",
        ).iloc[int(burn_in):].copy()
        if not np.isfinite(residuals.to_numpy()).all():
            raise ValueError("Training residuals contain non-finite values.")
        return residuals

    def fit_summary(self) -> dict[str, object]:
        """Describe the original parameter fit; state updates do not alter it."""
        self._require_fit()
        result = self._fit_result
        criteria = {}
        for key in ("aic", "aicc", "bic"):
            value = float(getattr(result, key))
            criteria[key] = value if np.isfinite(value) else None
        return {
            "order": list(result.model.order),
            "trend": result.model.trend,
            "n_training_observations": len(self._fit_history),
            "fitted_until": self.fitted_until.isoformat(),
            "state_until": self.state_until.isoformat(),
            "converged": True,
            "parameters": {
                name: float(value)
                for name, value in zip(result.param_names, result.params)
            },
            **criteria,
            "warnings": list(self._fit_warnings),
        }

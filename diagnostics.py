"""Block 3: target and residual diagnostics, without architecture routing.

Requires statsmodels>=0.15,<0.16, SciPy, NumPy and pandas. Matplotlib is needed
only for plot_qq(). Keep this module beside data.py. Use current development
folds; remove any model-initialization burn-in explicitly before diagnostics.
"""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from data import validate_series


@dataclass(frozen=True)
class DiagnosticConfig:
    alpha: float = 0.05
    max_lag: int = 10
    min_observations: int = 40
    stationarity_regression: str = "c"

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not 0 < self.alpha < 1
        ):
            raise ValueError("alpha must be a number between 0 and 1.")
        for name, minimum in (("max_lag", 1), ("min_observations", 8)):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.stationarity_regression not in ("c", "ct"):
            raise ValueError("stationarity_regression must be 'c' or 'ct'.")


@dataclass(frozen=True)
class DiagnosticReport:
    series_kind: str
    n_obs: int
    settings: dict[str, object]
    summary: dict[str, float | None]
    tests: dict[str, dict[str, object]]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return ordinary Python values suitable for strict JSON export."""
        return asdict(self)


def _finite_float(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


class TimeSeriesDiagnostics:
    def __init__(self, config: DiagnosticConfig | None = None) -> None:
        self.config = DiagnosticConfig() if config is None else config

    def _prepare(
        self, y: pd.Series, kind: str
    ) -> tuple[np.ndarray, DiagnosticReport, str | None]:
        clean = validate_series(y)
        x = clean.to_numpy()
        n = len(x)
        constant = bool(np.all(x == x[0]))
        reason = None
        if n < self.config.min_observations:
            reason = f"Need at least {self.config.min_observations} observations."
        elif constant:
            reason = "The series is constant; these tests are undefined."

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(over="ignore", invalid="ignore"):
                summary = {
                    "mean": _finite_float(np.mean(x)),
                    "std": _finite_float(np.std(x, ddof=1)) if n > 1 else None,
                    "skewness": _finite_float(stats.skew(x, bias=False))
                    if n > 2 and not constant else None,
                    "excess_kurtosis": _finite_float(
                        stats.kurtosis(x, fisher=True, bias=False)
                    ) if n > 3 and not constant else None,
                }
        report = DiagnosticReport(
            series_kind=kind,
            n_obs=n,
            settings={
                "alpha": float(self.config.alpha),
                "lag": int(min(self.config.max_lag, max(1, n // 5))),
                "stationarity_regression": self.config.stationarity_regression,
            },
            summary=summary,
            tests={},
            notes=(
                "P-values are exploratory and individual, without multiplicity correction.",
                "Non-rejection does not prove the null hypothesis.",
                "These diagnostics do not determine ARIMA-ML order.",
            ),
        )
        return x, report, reason

    def _run_test(
        self,
        null: str,
        calculation: Callable[[], object],
        reason: str | None = None,
    ) -> dict[str, object]:
        result = {
            "null_hypothesis": null,
            "status": "skipped" if reason else "ok",
            "statistic": None,
            "pvalue": None,
            "pvalue_relation": None,
            "reject_null": None,
            "notes": [reason] if reason else [],
        }
        if reason:
            return result
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                statistic, pvalue = calculation()[:2]
                statistic, pvalue = float(statistic), float(pvalue)
                if not np.isfinite(statistic) or not np.isfinite(pvalue):
                    raise ValueError("Test produced a non-finite statistic or p-value.")
                if not 0 <= pvalue <= 1:
                    raise ValueError("Test produced a p-value outside [0, 1].")
                result.update(
                    statistic=statistic,
                    pvalue=pvalue,
                    pvalue_relation="approximate",
                    reject_null=bool(pvalue < self.config.alpha),
                )
            except (ValueError, FloatingPointError, ZeroDivisionError,
                    OverflowError, np.linalg.LinAlgError) as exc:
                result["status"] = "error"
                result["notes"].append(f"{type(exc).__name__}: {exc}")
        result["notes"].extend(str(item.message).strip() for item in captured)
        return result

    def target_report(self, y: pd.Series) -> DiagnosticReport:
        """ADF and KPSS have different null hypotheses; inspect both."""
        x, report, reason = self._prepare(y, "target")
        regression = self.config.stationarity_regression
        report.tests["adf"] = self._run_test(
            "The series has a unit root.",
            lambda: adfuller(
                x, regression=regression, autolag="AIC", result_object=True
            ),
            reason,
        )
        stationary_null = (
            "The series is stationary around a constant."
            if regression == "c"
            else "The series is stationary around a linear trend."
        )
        report.tests["kpss"] = self._run_test(
            stationary_null,
            lambda: kpss(
                x, regression=regression, nlags="auto", result_object=True
            ),
            reason,
        )
        # KPSS may return a table boundary, rather than an interior p-value.
        result = report.tests["kpss"]
        notes = " ".join(result["notes"])
        if result["status"] == "ok":
            p, alpha = result["pvalue"], self.config.alpha
            if "actual p-value is smaller" in notes:
                result["pvalue_relation"] = "less_than"
                result["reject_null"] = True if alpha >= p else None
            elif "actual p-value is greater" in notes:
                result["pvalue_relation"] = "greater_than"
                result["reject_null"] = False if alpha <= p else None
        return report

    def residual_report(
        self, residuals: pd.Series, *, model_df: int = 0
    ) -> DiagnosticReport:
        """Check residual correlation, ARCH effects and normal moments.

        For fitted ARMA residuals, model_df is usually p+q. For chronological
        forecast errors, leave it at zero; calibration remains exploratory.
        Standardized residuals may be passed after fitting a variance model.
        """
        if (
            isinstance(model_df, (bool, np.bool_))
            or not isinstance(model_df, Integral)
            or model_df < 0
        ):
            raise ValueError("model_df must be a nonnegative integer.")
        model_df = int(model_df)
        x, report, reason = self._prepare(residuals, "residual")
        lag = report.settings["lag"]
        report.settings["model_df"] = model_df

        def ljung_box():
            row = acorr_ljungbox(x, lags=[lag], model_df=model_df).iloc[0]
            return row["lb_stat"], row["lb_pvalue"]

        report.tests["ljung_box"] = self._run_test(
            f"Autocorrelations through lag {lag} are zero.",
            ljung_box,
            reason or ("Ljung-Box requires lag > model_df." if lag <= model_df else None),
        )
        report.tests["arch_lm"] = self._run_test(
            f"No ARCH dependence through lag {lag}.",
            lambda: het_arch(x, nlags=lag, ddof=model_df, result_object=True),
            reason or (
                "ARCH-LM requires n_obs - lag > model_df."
                if len(x) - lag <= model_df else None
            ),
        )
        normality = self._run_test(
            "Skewness and kurtosis match a normal distribution.",
            lambda: stats.jarque_bera(x),
            reason,
        )
        normality["notes"].append(
            "Jarque-Bera uses asymptotic calibration; dependence can affect its p-value."
        )
        if len(x) <= 2000:
            normality["notes"].append(
                "Small-sample approximation: SciPy recommends more than 2000 observations."
            )
        report.tests["jarque_bera"] = normality
        return report

    def correlation_table(self, y: pd.Series) -> pd.DataFrame:
        """Return ACF/PACF at observation lags, excluding lag zero."""
        x, report, reason = self._prepare(y, "correlation")
        if reason:
            raise ValueError(reason)
        lag = min(report.settings["lag"], len(x) // 2 - 1)
        return pd.DataFrame({
            "lag": np.arange(1, lag + 1),
            "acf": acf(x, nlags=lag, fft=False)[1:],
            "pacf": pacf(x, nlags=lag, method="ywm")[1:],
        })


def plot_qq(y: pd.Series, ax=None):
    """Return an Axes containing a normal Q-Q plot and a scaled reference line."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import qqplot

    clean = validate_series(y)
    if len(clean) < 3 or clean.nunique() < 2:
        raise ValueError("A Q-Q plot needs at least three values and some variation.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    qqplot(clean.to_numpy(), line="s", ax=ax)
    ax.set_title("Normal Q-Q plot")
    return ax

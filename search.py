"""Block 9: bounded hyperparameter grids and chronological model selection.

Pass development data only. The returned scores are model-selection scores;
evaluate the locked winner on the reserved test segment separately. Search
uses the same target dates and refit policy for every candidate. A failed run
remains in the ranking without a score. No automatic grid truncation occurs.

Optional AICc screening uses only the first validation training window and
ranks within each d. It is a heuristic for standalone ARIMA, not a screen of
ARIMA orders inside a residual hybrid. Validate complete hybrid candidates.

Requires all earlier project modules and their dependencies. No new package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from numbers import Integral, Real

import numpy as np
import pandas as pd

from arima_model import ARIMAConfig, ARIMAForecaster
from data import validate_series
from features import FeatureConfig
from hybrids import HybridConfig, ARIMAXGBoostForecaster, XGBoostARIMAForecaster
from models import BaseForecaster, ZeroForecaster, PersistenceForecaster, RidgeConfig, RidgeForecaster
from validation import ValidationConfig, ValidationResult, compare_models
from xgboost_model import XGBoostConfig, XGBoostForecaster


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _axis(values, name: str) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a nonempty sequence.")
    return tuple(values)


def _budget(size: int, maximum: int) -> None:
    maximum = _positive_integer(maximum, "max_candidates")
    if size > maximum:
        raise ValueError(
            f"Grid contains {size} candidates, exceeding max_candidates={maximum}. "
            "Reduce the grid or explicitly raise the budget. No candidates were fitted."
        )


def _plain(value):
    """Copy metadata into finite, JSON-compatible Python values."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and np.isfinite(value):
        return float(value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    raise ValueError("Candidate metadata must contain finite JSON-compatible values.")


@dataclass(frozen=True)
class ARIMASearchSpace:
    p_values: tuple[int, ...] = (0, 1, 2, 3)
    d_values: tuple[int, ...] = (0, 1)
    q_values: tuple[int, ...] = (0, 1, 2, 3)
    # None tries n/c for d=0, n/t for d=1, and n for d>=2.
    # In statsmodels ARIMA, trend='t' with d=1 represents drift.
    trends: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("p_values", "d_values", "q_values"):
            values = _axis(getattr(self, name), name)
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values):
                raise ValueError(f"{name} must contain nonnegative integers.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates.")
            object.__setattr__(self, name, tuple(sorted(int(v) for v in values)))
        if self.trends is not None:
            trends = _axis(self.trends, "trends")
            if any(t not in ("n", "c", "t", "ct") for t in trends):
                raise ValueError("trends may contain 'n', 'c', 't', or 'ct'.")
            if len(set(trends)) != len(trends):
                raise ValueError("trends contains duplicates.")
            object.__setattr__(self, "trends", trends)

    def _trends_for(self, d: int) -> tuple[str, ...]:
        choices = self.trends
        if choices is None:
            return ("n", "c") if d == 0 else (("n", "t") if d == 1 else ("n",))
        return tuple(t for t in choices if not (d > 0 and t in ("c", "ct")) and not (d > 1 and t == "t"))

    def configs(self, template: ARIMAConfig | None = None, *, max_candidates: int = 256) -> tuple[ARIMAConfig, ...]:
        """Enumerate the complete valid grid; template supplies fitting settings."""
        template = ARIMAConfig() if template is None else template
        if not isinstance(template, ARIMAConfig):
            raise TypeError("template must be an ARIMAConfig.")
        size = len(self.p_values) * len(self.q_values) * sum(len(self._trends_for(d)) for d in self.d_values)
        if not size:
            raise ValueError("No valid order/trend combinations in this grid.")
        _budget(size, max_candidates)
        return tuple(
            replace(template, order=(p, d, q), trend=trend)
            for p, d, q in product(self.p_values, self.d_values, self.q_values)
            for trend in self._trends_for(d)
        )


@dataclass(frozen=True)
class XGBoostSearchSpace:
    n_estimators: tuple[int, ...] = (150, 300)
    max_depth: tuple[int, ...] = (2, 3)
    learning_rate: tuple[float, ...] = (0.03, 0.1)

    def __post_init__(self) -> None:
        for name in ("n_estimators", "max_depth", "learning_rate"):
            values = _axis(getattr(self, name), name)
            normalized = tuple(getattr(replace(XGBoostConfig(), **{name: v}), name) for v in values)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} contains duplicates.")
            object.__setattr__(self, name, normalized)

    def configs(self, template: XGBoostConfig | None = None, *, max_candidates: int = 256) -> tuple[XGBoostConfig, ...]:
        """Other XGBoost settings, including the seed, come from template."""
        template = XGBoostConfig() if template is None else template
        if not isinstance(template, XGBoostConfig):
            raise TypeError("template must be an XGBoostConfig.")
        _budget(len(self.n_estimators) * len(self.max_depth) * len(self.learning_rate), max_candidates)
        return tuple(
            replace(template, n_estimators=n, max_depth=depth, learning_rate=rate)
            for n, depth, rate in product(self.n_estimators, self.max_depth, self.learning_rate)
        )


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    pipeline: str
    parameters: dict[str, object]
    factory: Callable[[], BaseForecaster] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.pipeline, str) or not self.pipeline:
            raise ValueError("Candidate name and pipeline must be nonempty strings.")
        if not callable(self.factory):
            raise TypeError("Candidate factory must be callable.")
        if not isinstance(self.parameters, dict):
            raise TypeError("Candidate parameters must be a dictionary.")
        object.__setattr__(self, "parameters", _plain(self.parameters))


def _configs(values, cls, name: str) -> tuple:
    values = _axis(values, name)
    if any(not isinstance(v, cls) for v in values):
        raise TypeError(f"{name} must contain {cls.__name__} instances.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate configurations.")
    return values


def build_candidates(
    arima_configs: Sequence[ARIMAConfig] | None = None,
    ml_configs: Sequence[XGBoostConfig] | None = None,
    feature_configs: Sequence[FeatureConfig] | None = None,
    *,
    pipelines: tuple[str, ...] = ("arima",),
    hybrid_config: HybridConfig | None = None,
    include_baselines: bool = True,
    max_candidates: int = 256,
) -> tuple[ModelCandidate, ...]:
    """Build standalone or complete hybrid grids without fitting anything.

    Default: all 64 valid ARIMA order/trend choices plus three benchmarks.
    ML settings stay fixed unless ml_configs is supplied. Adding both hybrids
    searches their complete ARIMA x ML x feature grid. A large Cartesian product
    requires an explicit budget; it is never silently truncated or sampled.
    """
    allowed = ("arima", "xgboost", "arima_to_xgboost", "xgboost_to_arima")
    pipelines = _axis(pipelines, "pipelines")
    if any(p not in allowed for p in pipelines) or len(set(pipelines)) != len(pipelines):
        raise ValueError(f"Choose distinct pipelines from {allowed}.")
    if not isinstance(include_baselines, (bool, np.bool_)):
        raise TypeError("include_baselines must be Boolean.")
    arimas = _configs(ARIMASearchSpace().configs() if arima_configs is None else arima_configs, ARIMAConfig, "arima_configs")
    mls = _configs((XGBoostConfig(),) if ml_configs is None else ml_configs, XGBoostConfig, "ml_configs")
    features = _configs((FeatureConfig(),) if feature_configs is None else feature_configs, FeatureConfig, "feature_configs")
    hybrid_config = HybridConfig() if hybrid_config is None else hybrid_config
    if not isinstance(hybrid_config, HybridConfig):
        raise TypeError("hybrid_config must be a HybridConfig.")
    a, m = len(arimas), len(mls) * len(features)
    count = 3 * bool(include_baselines) + a * ("arima" in pipelines) + m * ("xgboost" in pipelines)
    count += a * m * sum(p in pipelines for p in ("arima_to_xgboost", "xgboost_to_arima"))
    _budget(count, max_candidates)
    candidates = []

    def add(name, pipeline, parameters, factory):
        candidates.append(ModelCandidate(name, pipeline, parameters, factory))

    if include_baselines:
        for label, factory in (("zero", ZeroForecaster), ("persistence", PersistenceForecaster)):
            add(label, label, {}, factory)
        ridge_config = RidgeConfig()
        ridge_features = FeatureConfig(rolling_windows=())
        add("ridge", "ridge", {"ridge": asdict(ridge_config), "features": asdict(ridge_features)},
            lambda: RidgeForecaster(ridge_config, ridge_features))
    for i, ac in enumerate(arimas, 1):
        if "arima" in pipelines:
            add(f"arima:a{i:03d}", "arima", {"arima": asdict(ac)}, lambda ac=ac: ARIMAForecaster(ac))
    for j, (mc, fc) in enumerate(product(mls, features), 1):
        ml_params = {"ml": asdict(mc), "features": asdict(fc)}
        if "xgboost" in pipelines:
            add(f"xgboost:m{j:03d}", "xgboost", ml_params, lambda mc=mc, fc=fc: XGBoostForecaster(mc, fc))
        for i, ac in enumerate(arimas, 1):
            params = {"arima": asdict(ac), **ml_params, "hybrid": asdict(hybrid_config)}
            if "arima_to_xgboost" in pipelines:
                add(f"arima_to_xgboost:a{i:03d}:m{j:03d}", "arima_to_xgboost", params,
                    lambda ac=ac, mc=mc, fc=fc: ARIMAXGBoostForecaster(ac, mc, fc, hybrid_config))
            if "xgboost_to_arima" in pipelines:
                add(f"xgboost_to_arima:a{i:03d}:m{j:03d}", "xgboost_to_arima", params,
                    lambda ac=ac, mc=mc, fc=fc: XGBoostARIMAForecaster(mc, ac, fc, hybrid_config))
    return tuple(candidates)


@dataclass(frozen=True)
class SearchConfig:
    primary_metric: str = "rmse"
    max_candidates: int = 256

    def __post_init__(self) -> None:
        if self.primary_metric not in ("rmse", "mae"):
            raise ValueError("primary_metric must be 'rmse' or 'mae'.")
        object.__setattr__(self, "max_candidates", _positive_integer(self.max_candidates, "max_candidates"))


@dataclass(frozen=True)
class SearchResult:
    ranking: pd.DataFrame
    runs: dict[str, ValidationResult]
    candidates: tuple[ModelCandidate, ...]
    validation_config: ValidationConfig
    search_config: SearchConfig

    @property
    def best_name(self) -> str | None:
        successful = self.ranking.index[self.ranking["status"] == "ok"]
        return str(successful[0]) if len(successful) else None

    @property
    def best_parameters(self) -> dict[str, object] | None:
        candidate = next((c for c in self.candidates if c.name == self.best_name), None)
        return _plain(candidate.parameters) if candidate else None

    def make_best(self) -> BaseForecaster:
        """Construct a fresh, unfitted winner with the selected settings."""
        if self.best_name is None:
            raise RuntimeError("All candidates failed; inspect ranking['error'].")
        candidate = next(c for c in self.candidates if c.name == self.best_name)
        model = candidate.factory()
        if not isinstance(model, BaseForecaster) or model.fitted_until is not None:
            raise TypeError("The winning factory must construct a fresh BaseForecaster.")
        return model

    def fit_best(self, development: pd.Series) -> BaseForecaster:
        """Refit selected settings on development data for final test evaluation."""
        return self.make_best().fit(development)

    def summary(self) -> dict[str, object]:
        winner = self.best_name
        return {
            "evaluation": "development model selection; reserved test required",
            "search_config": asdict(self.search_config),
            "validation_config": _plain(asdict(self.validation_config)),
            "n_candidates": len(self.candidates),
            "n_successful": len(self.runs),
            "n_failed": len(self.candidates) - len(self.runs),
            "best_name": winner,
            "best_parameters": self.best_parameters,
            "best_metrics": self.runs[winner].metrics.to_dict() if winner else None,
        }


def run_search(
    candidates: Sequence[ModelCandidate],
    development: pd.Series,
    validation_config: ValidationConfig | None = None,
    search_config: SearchConfig | None = None,
) -> SearchResult:
    """Select by chronological RMSE or MAE; preserve all failed candidates."""
    candidates = _axis(candidates, "candidates")
    if any(not isinstance(c, ModelCandidate) for c in candidates):
        raise TypeError("candidates must contain ModelCandidate instances.")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique.")
    validation_config = ValidationConfig(initial_train_size=200, refit_every=None) if validation_config is None else validation_config
    search_config = SearchConfig() if search_config is None else search_config
    if not isinstance(validation_config, ValidationConfig) or not isinstance(search_config, SearchConfig):
        raise TypeError("Provide ValidationConfig and SearchConfig instances.")
    _budget(len(candidates), search_config.max_candidates)
    ranking, runs = compare_models({c.name: c.factory for c in candidates}, development, validation_config)
    lookup = {c.name: c for c in candidates}
    ranking["pipeline"] = [lookup[name].pipeline for name in ranking.index]
    ranking["parameters"] = [_plain(lookup[name].parameters) for name in ranking.index]
    keys = [search_config.primary_metric, "mae" if search_config.primary_metric == "rmse" else "rmse"]
    ranking = ranking.sort_values(keys, kind="stable", na_position="last")
    return SearchResult(ranking, runs, candidates, validation_config, search_config)


@dataclass(frozen=True)
class ARIMAShortlist:
    configs: tuple[ARIMAConfig, ...]
    table: pd.DataFrame
    screening_start: pd.Timestamp
    screening_end: pd.Timestamp


def shortlist_arima(
    arima_configs: Sequence[ARIMAConfig],
    development: pd.Series,
    validation_config: ValidationConfig,
    *,
    top_per_d: int = 3,
    max_candidates: int = 256,
) -> ARIMAShortlist:
    """Optional standalone-ARIMA screen on the initial training window only.

    Keep the lowest finite AICc values separately within each d. The screen is
    approximate: a discarded model might have better validation forecasts.
    For a full bounded grid comparison, skip this function. For hybrids, search
    complete pipeline candidates instead of screening on the raw target here.
    """
    configs = _configs(arima_configs, ARIMAConfig, "arima_configs")
    _budget(len(configs), max_candidates)
    top_per_d = _positive_integer(top_per_d, "top_per_d")
    if not isinstance(validation_config, ValidationConfig):
        raise TypeError("validation_config must be a ValidationConfig.")
    y = validate_series(development)
    stop = validation_config.initial_train_size
    if stop >= len(y):
        raise ValueError("initial_train_size must leave a validation target.")
    start = 0 if validation_config.window_size is None else max(0, stop - validation_config.window_size)
    training = y.iloc[start:stop].copy()
    records = []
    for config in configs:
        row = {"p": config.order[0], "d": config.order[1], "q": config.order[2], "trend": config.trend,
               "aicc": np.nan, "status": "failed", "error": "", "selected": False}
        try:
            fitted = ARIMAForecaster(config).fit(training)
            aicc = fitted.fit_summary()["aicc"]
            if aicc is None or not np.isfinite(aicc):
                raise ValueError("The fit has no finite AICc.")
            row.update(aicc=float(aicc), status="ok")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
    table = pd.DataFrame(records)
    table.index.name = "configuration_index"
    table["rank_within_d"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    for d in sorted(set(config.order[1] for config in configs)):
        ranked = table.loc[(table["d"] == d) & (table["status"] == "ok")].sort_values("aicc", kind="stable")
        table.loc[ranked.index, "rank_within_d"] = np.arange(1, len(ranked) + 1)
        table.loc[ranked.index[:top_per_d], "selected"] = True
    selected = tuple(configs[i] for i in table.index[table["selected"]])
    return ARIMAShortlist(selected, table, training.index[0], training.index[-1])

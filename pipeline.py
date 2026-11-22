"""Block 11: one runnable ARIMA-ML forecasting experiment.

Examples, from the folder containing all project modules:
    python pipeline.py --demo --output results/demo_01
    python pipeline.py --csv data.csv --time-column date --target-column y \
        --mode fixed --output results/fixed_01

Fixed mode compares one configuration per requested architecture. Search mode
enumerates the bounded order/trend grid and optionally an ML grid. Both modes
include zero, persistence, and Ridge benchmarks by default. LLM review remains
advisory. Only the development winner is evaluated on the final test segment.

All forecasts are one observation ahead, conditional on earlier observed values.
The test inherits the predeclared validation window/refit/metric policy. With
refit_every=None, coefficients are fitted once on development; states/features
update as test observations arrive. A positive interval also re-estimates
coefficients on the permitted past data, keeping architecture/settings fixed.

Uses the earlier modules and their dependencies. Matplotlib is optional when
plots and image review are disabled. No external data or API call is needed
for --demo. Use a new output directory for every experiment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

from arima_model import ARIMAConfig
from data import HoldoutSplit, SplitConfig, chronological_holdout, load_series_csv, validate_series
from diagnostics import DiagnosticConfig, TimeSeriesDiagnostics, plot_qq
from features import FeatureConfig
from hybrids import HybridConfig
from metrics import MetricConfig
from models import BaseForecaster
from search import ARIMASearchSpace, XGBoostSearchSpace, ModelCandidate, SearchConfig, SearchResult, build_candidates, run_search
from selection import OpenAIReviewer, ReviewRequest, SelectionConfig, SelectionResult, review_selection, select_pipeline
from validation import ValidationConfig, ValidationError, ValidationResult, walk_forward
from xgboost_model import XGBoostConfig


_PIPELINES = ("arima", "xgboost", "arima_to_xgboost", "xgboost_to_arima")


def _plain(value):
    if value is None or isinstance(value, str):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    raise TypeError(f"Unsupported report value: {type(value).__name__}.")


def _series_summary(y: pd.Series) -> dict[str, object]:
    digest = hashlib.sha256(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes()).hexdigest()
    return {"n_observations": len(y), "first_time": y.index[0].isoformat(),
            "last_time": y.index[-1].isoformat(), "fingerprint": digest}


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "xgboost", "xgboost-cpu", "matplotlib", "openai"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _new_output(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}. Choose a new experiment directory.")
    return destination


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "fixed"
    split: SplitConfig = field(default_factory=SplitConfig)
    validation: ValidationConfig = field(default_factory=lambda: ValidationConfig(initial_train_size=200, refit_every=None))
    search: SearchConfig = field(default_factory=SearchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    ml: XGBoostConfig = field(default_factory=XGBoostConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    arima_space: ARIMASearchSpace = field(default_factory=ARIMASearchSpace)
    ml_space: XGBoostSearchSpace | None = None
    pipelines: tuple[str, ...] = _PIPELINES
    include_baselines: bool = True
    review_qq: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "search"):
            raise ValueError("mode must be 'fixed' or 'search'.")
        typed = (("split", SplitConfig), ("validation", ValidationConfig),
                 ("search", SearchConfig), ("selection", SelectionConfig),
                 ("diagnostics", DiagnosticConfig), ("arima", ARIMAConfig),
                 ("ml", XGBoostConfig), ("features", FeatureConfig),
                 ("hybrid", HybridConfig), ("arima_space", ARIMASearchSpace))
        for name, cls in typed:
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be a {cls.__name__} instance.")
        if self.ml_space is not None and not isinstance(self.ml_space, XGBoostSearchSpace):
            raise TypeError("ml_space must be an XGBoostSearchSpace or None.")
        if not isinstance(self.pipelines, (tuple, list)) or not self.pipelines:
            raise ValueError("pipelines must be a nonempty sequence.")
        if any(name not in _PIPELINES for name in self.pipelines) or len(set(self.pipelines)) != len(self.pipelines):
            raise ValueError(f"Choose distinct pipelines from {_PIPELINES}.")
        object.__setattr__(self, "pipelines", tuple(self.pipelines))
        for name in ("include_baselines", "review_qq"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be Boolean.")
            object.__setattr__(self, name, bool(value))

    def candidates(self) -> tuple[ModelCandidate, ...]:
        maximum = self.search.max_candidates
        arimas = (self.arima,) if self.mode == "fixed" else self.arima_space.configs(self.arima, max_candidates=maximum)
        mls = (self.ml,)
        if self.mode == "search" and self.ml_space is not None:
            mls = self.ml_space.configs(self.ml, max_candidates=maximum)
        return build_candidates(
            arimas, mls, (self.features,), pipelines=self.pipelines,
            hybrid_config=self.hybrid, include_baselines=self.include_baselines,
            max_candidates=maximum,
        )


@dataclass(frozen=True)
class ExperimentResult:
    config: PipelineConfig
    split: HoldoutSplit
    search_result: SearchResult
    selection: SelectionResult
    test_result: ValidationResult | None
    test_diagnostics: dict[str, object] | None
    status: str
    error: str | None
    created_at_utc: str
    timings: dict[str, float]
    versions: dict[str, str]
    output_dir: Path | None = None

    def summary(self) -> dict[str, object]:
        selected = self.selection.selected_name
        score = self.selection.leaderboard.loc[selected, self.selection.primary_metric] if selected else None
        return _plain({
            "status": self.status, "mode": self.config.mode,
            "selected_name": selected, "primary_metric": self.selection.primary_metric,
            "validation_score": score,
            "test_metrics": self.test_result.metrics.to_dict() if self.test_result else None,
            "llm_status": self.selection.review_status, "error": self.error,
            "n_candidates": len(self.search_result.candidates),
            "n_successful": len(self.search_result.runs),
            "output_dir": str(self.output_dir) if self.output_dir else None,
        })

    def to_dict(self) -> dict[str, object]:
        return _plain({
            "schema_version": 1, "created_at_utc": self.created_at_utc,
            "summary": self.summary(), "config": asdict(self.config),
            "data": {"development": _series_summary(self.split.development),
                     "test": _series_summary(self.split.test)},
            "selection": self.selection.to_dict(),
            "test_evaluation": self.test_result.summary() if self.test_result else None,
            "test_diagnostics": self.test_diagnostics,
            "timings_seconds": self.timings, "versions": self.versions,
            "protocol": [
                "Architecture and hyperparameters are selected using development data only.",
                "LLM review precedes test evaluation and remains advisory.",
                "Only the locked development winner is evaluated on the test segment.",
                "Each test forecast precedes its target; earlier revealed test values may update context and scheduled fits.",
                "The test uses the development refit/window/metric policy, restarting its refit schedule at the first test origin.",
                "Test diagnostics are descriptive and cannot trigger model reselection in this run.",
            ],
        })

    def save(self, output_dir: str | Path, *, include_plots: bool = True) -> Path:
        """Write a new results directory; preserve existing experiments."""
        if not isinstance(include_plots, (bool, np.bool_)):
            raise TypeError("include_plots must be Boolean.")
        destination = _new_output(output_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            ranking = self.selection.leaderboard.copy()
            ranking["parameters"] = ranking["parameters"].map(lambda value: json.dumps(_plain(value), allow_nan=False, sort_keys=True))
            ranking.to_csv(staging / "validation_ranking.csv", index_label="candidate")
            if self.search_result.runs:
                frames = []
                for name, run in self.search_result.runs.items():
                    frame = run.forecasts.copy()
                    frame.insert(0, "candidate", name)
                    frames.append(frame)
                pd.concat(frames).to_csv(staging / "validation_forecasts.csv", index_label="target_time")
            if self.test_result is not None:
                self.test_result.forecasts.to_csv(staging / "test_forecasts.csv", index_label="target_time")
            plot_status = self._save_plots(staging) if include_plots else {"status": "not_requested"}
            report = self.to_dict()
            report["summary"]["output_dir"] = str(destination)
            report["plots"] = plot_status
            (staging / "experiment.json").write_text(
                json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
            )
            if destination.exists():
                raise FileExistsError(f"Output was created by another process: {destination}.")
            staging.rename(destination)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _save_plots(self, directory: Path) -> dict[str, object]:
        status = {}
        chosen = self.selection.selected_name
        runs = {}
        if chosen:
            runs["validation"] = self.search_result.runs[chosen]
        if self.test_result:
            runs["test"] = self.test_result
        for label, run in runs.items():
            errors = run.errors
            name = f"{label}_qq.png"
            if len(errors) < 3 or errors.nunique() < 2:
                status[name] = "skipped: too few errors or no variation"
                continue
            try:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure
                fig = Figure(figsize=(6, 5), dpi=130)
                FigureCanvasAgg(fig)
                try:
                    ax = fig.add_subplot(111)
                    plot_qq(errors, ax=ax)
                    ax.set_title(f"{label.capitalize()} forecast errors: {chosen}", fontsize=10)
                    fig.tight_layout()
                    fig.savefig(directory / name)
                finally:
                    fig.clear()
                status[name] = "saved"
            except Exception as exc:
                (directory / name).unlink(missing_ok=True)
                status[name] = f"unavailable: {type(exc).__name__}: {str(exc)[:300]}"
        return status


class ForecastPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        reviewer: Callable[[ReviewRequest], dict[str, object]] | None = None,
    ) -> None:
        self.config = PipelineConfig() if config is None else config
        if not isinstance(self.config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig.")
        if reviewer is not None and not callable(reviewer):
            raise TypeError("reviewer must be callable or None.")
        self.reviewer = reviewer

    def run(
        self,
        y: pd.Series,
        *,
        output_dir: str | Path | None = None,
        include_plots: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> ExperimentResult:
        started = perf_counter()
        created = datetime.now(timezone.utc).isoformat()
        config = self.config
        if output_dir is not None:
            _new_output(output_dir)
        if not isinstance(include_plots, (bool, np.bool_)):
            raise TypeError("include_plots must be Boolean.")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None.")

        def update(message):
            if progress is not None:
                progress(message)

        clean = validate_series(y)
        split = chronological_holdout(clean, config.split)
        initial = config.validation.initial_train_size
        if len(split.development) <= initial:
            raise ValueError(f"Development has {len(split.development)} observations; initial_train_size={initial} must leave validation targets.")
        candidates = config.candidates()
        available = initial if config.validation.window_size is None else min(initial, config.validation.window_size)
        too_small = []
        for candidate in candidates:
            model = candidate.factory()
            if not isinstance(model, BaseForecaster):
                raise TypeError("Candidate factories must construct BaseForecaster instances.")
            if model.min_train_size > available:
                too_small.append((candidate.name, int(model.min_train_size)))
        if too_small:
            required = max(size for _, size in too_small)
            raise ValueError(f"The initial fitting window has {available} observations; these candidates require up to {required}. Increase initial_train_size/window_size or reduce model history requirements. Examples: {too_small[:3]}")
        timings = {"preparation": float(perf_counter() - started)}

        update(f"Comparing {len(candidates)} candidates on development data ({config.mode} mode).")
        stage = perf_counter()
        search_result = run_search(candidates, split.development, config.validation, config.search)
        timings["search"] = float(perf_counter() - stage)
        stage = perf_counter()
        decision = select_pipeline(search_result, split.development, config.selection, config.diagnostics)
        timings["selection"] = float(perf_counter() - stage)
        stage = perf_counter()
        if self.reviewer is not None:
            update("Reviewing development evidence with the configured LLM.")
            decision = review_selection(decision, self.reviewer, include_qq=config.review_qq)
        timings["review"] = float(perf_counter() - stage)

        test_run, test_report, error = None, None, None
        status = "selection_failed" if decision.selected_name is None else "completed"
        stage = perf_counter()
        if decision.selected_name is None:
            error = "All development candidates failed; inspect the validation ranking."
        else:
            update(f"Evaluating locked candidate {decision.selected_name} on the reserved test segment.")
            test_config = replace(config.validation, initial_train_size=len(split.development))
            try:
                # walk_forward fits a fresh selected model on development at the
                # first test origin. It reveals each test value after forecasting.
                test_run = walk_forward(decision.make_model, clean, test_config, model_name=decision.selected_name)
            except (ValidationError, ValueError, TypeError, ArithmeticError) as exc:
                status = "test_failed"
                error = f"{type(exc).__name__}: {exc}"
        timings["test"] = float(perf_counter() - stage)
        stage = perf_counter()
        if test_run is not None:
            test_report = TimeSeriesDiagnostics(config.diagnostics).residual_report(test_run.errors, model_df=0).to_dict()
        timings["test_diagnostics"] = float(perf_counter() - stage)
        timings["total_compute"] = float(perf_counter() - started)
        experiment = ExperimentResult(
            config, split, search_result, decision, test_run, _plain(test_report),
            status, error, created, timings, _versions(),
        )
        if output_dir is not None:
            update("Saving experiment report, rankings, and forecasts.")
            destination = experiment.save(output_dir, include_plots=include_plots)
            experiment = replace(experiment, output_dir=destination)
        return experiment


def run_pipeline(
    y: pd.Series,
    config: PipelineConfig | None = None,
    *,
    reviewer: Callable[[ReviewRequest], dict[str, object]] | None = None,
    output_dir: str | Path | None = None,
    include_plots: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    return ForecastPipeline(config, reviewer=reviewer).run(
        y, output_dir=output_dir, include_plots=include_plots, progress=progress
    )


def run_csv_pipeline(
    path: str | Path,
    *,
    time_column: str,
    target_column: str,
    config: PipelineConfig | None = None,
    time_format: str | None = None,
    utc: bool = False,
    reviewer: Callable[[ReviewRequest], dict[str, object]] | None = None,
    output_dir: str | Path | None = None,
    include_plots: bool = True,
) -> ExperimentResult:
    y = load_series_csv(path, time_column=time_column, target_column=target_column,
                        time_format=time_format, utc=utc)
    return run_pipeline(y, config, reviewer=reviewer, output_dir=output_dir, include_plots=include_plots)


def _demo_series() -> pd.Series:
    rng = np.random.default_rng(42)
    values = np.zeros(650)
    for t in range(1, len(values)):
        values[t] = .55 * values[t-1] + .15 * np.tanh(3 * values[t-1]) + .08 * np.sin(t/13) + rng.normal(0, .12)
    return pd.Series(values[50:], index=pd.date_range("2020-01-01", periods=600, freq="h"), name="synthetic_target")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--demo", action="store_true", help="Use a reproducible synthetic series.")
    parser.add_argument("--time-column")
    parser.add_argument("--target-column")
    parser.add_argument("--utc", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this experiment.")
    parser.add_argument("--mode", choices=("fixed", "search"), default="fixed")
    parser.add_argument("--order", type=int, nargs=3, default=(1, 0, 1), metavar=("P", "D", "Q"))
    parser.add_argument("--p-max", type=int, default=3)
    parser.add_argument("--q-max", type=int, default=3)
    parser.add_argument("--d-values", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--tune-ml", action="store_true", help="Also search the 8-setting XGBoost grid.")
    parser.add_argument("--pipelines", nargs="+", choices=_PIPELINES, default=_PIPELINES)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--initial-train-size", type=int, default=200)
    parser.add_argument("--refit-every", type=int, default=0, help="0 fits once; k refits every k observations.")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--metric", choices=("rmse", "mae"), default="rmse")
    parser.add_argument("--direction-mode", choices=("returns", "levels"))
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--llm-model", help="Optional Responses model ID; uses OPENAI_API_KEY.")
    parser.add_argument("--no-qq-review", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.csv and (not args.time_column or not args.target_column):
        parser.error("--csv requires --time-column and --target-column.")
    if args.tune_ml and args.mode != "search":
        parser.error("--tune-ml requires --mode search.")
    if args.p_max < 0 or args.q_max < 0 or args.refit_every < 0:
        parser.error("p-max, q-max, and refit-every must be nonnegative.")
    try:
        config = PipelineConfig(
            mode=args.mode, split=SplitConfig(test_fraction=args.test_fraction),
            validation=ValidationConfig(initial_train_size=args.initial_train_size,
                                        refit_every=args.refit_every or None,
                                        window_size=args.window_size,
                                        metrics=MetricConfig(direction_mode=args.direction_mode)),
            search=SearchConfig(primary_metric=args.metric, max_candidates=args.max_candidates),
            arima=ARIMAConfig(order=tuple(args.order)),
            arima_space=ARIMASearchSpace(p_values=tuple(range(args.p_max + 1)),
                                        d_values=tuple(args.d_values),
                                        q_values=tuple(range(args.q_max + 1))),
            ml_space=XGBoostSearchSpace() if args.tune_ml else None,
            pipelines=tuple(args.pipelines), review_qq=not args.no_qq_review,
        )
        reviewer = OpenAIReviewer(args.llm_model) if args.llm_model else None
        y = _demo_series() if args.demo else load_series_csv(
            args.csv, time_column=args.time_column, target_column=args.target_column, utc=args.utc
        )
        experiment = run_pipeline(y, config, reviewer=reviewer, output_dir=args.output,
                                  include_plots=not args.no_plots, progress=print)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Experiment could not start or save: {exc}\n")
    print(json.dumps(experiment.summary(), allow_nan=False, indent=2))
    return 0 if experiment.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

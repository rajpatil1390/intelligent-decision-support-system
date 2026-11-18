"""AutoML trainer: tune each planned candidate on train/valid, pick the best acceptable one, refit it on
train+valid, and evaluate it exactly once on the untouched test split.

The test split is wrapped in a Holdout that records every access, so "evaluated once, at the end" is
checked rather than assumed.
"""

import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from idss.core.logger import get_logger
from idss.core.schemas import (
    Baseline,
    DatasetProfile,
    LeaderboardEntry,
    ModelConstraints,
    ModelPlan,
    TaskType,
    TrainingBudget,
    TrainingResult,
    UserRequirement,
)
from idss.ml import model_zoo, tuner
from idss.ml.evaluator import (
    THRESHOLD_TUNABLE_METRICS,
    Direction,
    EvalData,
    apply_threshold,
    complexity,
    evaluate,
    is_supported,
    metric_direction,
    tune_threshold,
)
from idss.ml.model_zoo import Candidate, ModelContext
from idss.ml.preprocess import build_preprocessor, clean_dataset, split_data
from idss.ml.ts_features import TimeSeriesFeatures, choose_lags, infer_seasonal_period

logger = get_logger()

MODEL_FILENAME = "best_model.joblib"
LEADERBOARD_FILENAME = "leaderboard.json"
METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "test_predictions.csv"
INFERENCE_SAMPLE_ROWS = 500


class TrainingError(Exception):
    """No candidate model could be trained."""


class Holdout:
    """The test split. Reading it is logged so tests and the Verifier can prove it was used only once."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.access_log: list[str] = []

    def __len__(self) -> int:
        return len(self._frame)

    def reveal(self, purpose: str) -> pd.DataFrame:
        self.access_log.append(purpose)
        logger.info("Test split accessed for: {}.", purpose)
        return self._frame


class TrainedModel:
    """What best_model.joblib contains: the fitted pipeline plus what is needed to use it on raw data.

    For time-series models, the input frame must include the target column with past values (NaN for
    rows to forecast); lag features are built from it.
    """

    def __init__(self, pipeline: Pipeline, task_type: TaskType, target: str | None, feature_columns: list[str],
                 model_name: str, classes: list[Any] | None = None, decision_threshold: float | None = None,
                 pos_label: int | None = None) -> None:
        self.pipeline = pipeline
        self.task_type = task_type
        self.target = target
        self.feature_columns = feature_columns
        self.model_name = model_name
        self.classes = classes
        self.decision_threshold = decision_threshold  # binary classification: tuned probability cut-off
        self.pos_label = pos_label

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.decision_threshold is not None and self.pos_label is not None:
            predictions = apply_threshold(self.predict_proba(X), self.pos_label, self.decision_threshold)
        else:
            predictions = self.pipeline.predict(X[self.feature_columns])
        if self.classes is not None:
            return np.asarray(self.classes, dtype=object)[np.asarray(predictions, dtype=int)]
        return np.asarray(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, columns in the order of `classes` (classification only)."""
        if self.classes is None:
            raise AttributeError("predict_proba is only available for classification models.")
        return self.pipeline.predict_proba(X[self.feature_columns])


@dataclass
class _Prepared:
    """Data and fitted feature pipeline shared by all candidates."""

    train: pd.DataFrame
    valid: pd.DataFrame
    holdout: Holdout
    feature_columns: list[str]
    features: Pipeline  # fitted on train
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray | None
    y_valid: np.ndarray | None
    classes: list[Any] | None
    context: ModelContext
    label_column: str | None  # supervised target, or clustering ground truth (evaluation only)


class AutoMLTrainer:
    """Runs the model search described by a ModelPlan within a TrainingBudget.

    The switches turn single components off for ablation studies (scripts/evaluate.py): without
    `tune_hyperparameters` every candidate is fitted once with its library defaults; without `tune_threshold`
    binary classifiers keep the 0.5 cut-off; without `train_baselines` the always-trained baseline candidates
    (e.g. seasonal naive) are skipped. The test-split baseline score is computed either way.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        clock: Callable[[], float] = time.monotonic,
        tune_hyperparameters: bool = True,
        tune_threshold: bool = True,
        train_baselines: bool = True,
    ) -> None:
        self.seed = seed
        self._clock = clock  # used only for budget checks, so tests can simulate elapsed time
        self.tune_hyperparameters = tune_hyperparameters
        self.tune_threshold = tune_threshold
        self.train_baselines = train_baselines

    def run(
        self,
        df: pd.DataFrame,
        requirement: UserRequirement,
        profile: DatasetProfile,
        run_dir: Path,
        budget: TrainingBudget | dict[str, Any] | None = None,
        plan: ModelPlan | None = None,
    ) -> TrainingResult:
        started = self._clock()
        wall_start = time.perf_counter()
        if plan is None:
            from idss.agents.model_agent import ModelAgent  # deterministic default plan; local import avoids a cycle

            plan = ModelAgent(None).plan(requirement, profile)
        budget = TrainingBudget.model_validate(budget) if isinstance(budget, dict) else (budget or plan.budget)
        notes: list[str] = []

        data = self._prepare(df, plan, profile, notes)
        metric = self._usable_metric(plan, data, notes)
        direction = metric_direction(metric)
        candidates = self._candidates(plan, budget, notes)
        logger.info(
            "AutoML: task={}, metric={} ({}), candidates={}, budget={}.",
            plan.task_type, metric, direction, [c.name for c in candidates], budget.model_dump(),
        )

        entries: list[LeaderboardEntry] = []
        for candidate in candidates:
            remaining = budget.total_timeout_s - (self._clock() - started)
            if remaining <= 0:
                entries.append(LeaderboardEntry(model=candidate.name, status="skipped", error="Total time budget used up."))
                notes.append(f"Skipped '{candidate.name}': the total time budget of {budget.total_timeout_s:g}s was used up.")
                continue
            entries.append(self._train_candidate(candidate, data, metric, budget, remaining))

        if not any(entry.status == "ok" for entry in entries):
            details = "; ".join(f"{e.model}: {e.error}" for e in entries)
            raise TrainingError(f"No candidate model could be trained ({details}).")

        mark_constraint_violations(entries, plan.constraints)
        entries = rank_entries(entries, direction)
        best, selection_notes = select_best(entries)
        notes += selection_notes

        final, fit_s = self._refit(best, data, plan.task_type)
        evaluation = self._final_evaluation(final, data, plan.task_type, fit_s, best.decision_threshold)
        best.selected = True
        best.test_metrics = evaluation.test

        result = self._save(
            run_dir, final, best, entries, data, plan, metric, direction, budget, evaluation, notes,
            time.perf_counter() - wall_start,
        )
        logger.info("AutoML finished: best={} test {}={}.", best.model, metric, evaluation.test.get(metric))
        return result

    # ------------------------------------------------------------------ data preparation

    def _prepare(self, df: pd.DataFrame, plan: ModelPlan, profile: DatasetProfile, notes: list[str]) -> _Prepared:
        task = plan.task_type
        family = model_zoo.task_family(task)
        supervised = family != "clustering"
        label_column = plan.target if plan.target in df.columns else None
        if supervised and label_column is None:
            raise TrainingError(f"Target column '{plan.target}' is not in the data.")

        cleaned, _ = clean_dataset(df, label_column if supervised else None, task, profile)
        if not supervised and label_column is not None and cleaned[label_column].isna().any():
            notes.append(f"Ground-truth column '{label_column}' has missing values; clustering is scored without it.")
            label_column = None
        train, valid, test = split_data(cleaned, label_column if supervised else None, task, seed=self.seed)
        holdout = Holdout(test)

        if family == "time_series":
            time_column = profile.time_column or next(iter(profile.columns_of("datetime")), None)
            if time_column is None:
                raise TrainingError("Time-series forecasting needs a date/time column, but none was found.")
            period = infer_seasonal_period(train[time_column])
            lags, windows = choose_lags(period, len(train))
            ts = TimeSeriesFeatures(time_column, label_column, lags, windows)
            features = Pipeline([
                ("ts_features", ts),
                ("preprocess", build_preprocessor(profile, label_column, extra_numeric=ts.feature_names())),
            ])
            feature_columns = list(train.columns)  # the target column supplies lag history
        else:
            period = 1
            features = Pipeline([("preprocess", build_preprocessor(profile, label_column))])
            feature_columns = [col for col in train.columns if col != label_column]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train = features.fit_transform(train[feature_columns])
            X_valid = features.transform(valid[feature_columns])
        y_train, y_valid, classes = _encode_targets(train, valid, label_column, family)

        class_counts: tuple[int, ...] = ()
        pos_label = None
        if classes is not None and family == "classification":
            class_counts = tuple(int(c) for c in np.bincount(y_train, minlength=len(classes)))
            if len(classes) == 2:
                pos_label = int(np.argmin(class_counts))
        context = ModelContext(
            task_type=task,
            n_rows=len(train),
            seed=self.seed,
            n_classes=len(classes) if classes is not None and family == "classification" else None,
            class_counts=class_counts,
            pos_label=pos_label,
            seasonal_period=period,
            feature_names=tuple(str(name) for name in features[-1].get_feature_names_out()),
        )
        return _Prepared(train, valid, holdout, feature_columns, features, np.asarray(X_train, dtype="float64"),
                         np.asarray(X_valid, dtype="float64"), y_train, y_valid, classes, context, label_column)

    def _usable_metric(self, plan: ModelPlan, data: _Prepared, notes: list[str]) -> str:
        metric = plan.primary_metric
        fallback = {"regression": "rmse", "time_series": "rmse", "classification": "f1", "clustering": "silhouette"}
        family = model_zoo.task_family(plan.task_type)
        labels = data.y_train is not None
        problem = None
        if not is_supported(metric, plan.task_type, labels_available=labels):
            problem = "is not available for this task/data"
        elif metric == "rmsle" and data.y_train is not None and np.any(data.y_train < 0):
            problem = "needs non-negative targets"
        elif metric == "mape" and data.y_train is not None and np.any(data.y_train == 0):
            problem = "is undefined when the target contains zeros"
        if problem:
            notes.append(f"Primary metric '{metric}' {problem}; using '{fallback[family]}' instead.")
            return fallback[family]
        return metric

    def _candidates(self, plan: ModelPlan, budget: TrainingBudget, notes: list[str]) -> list[Candidate]:
        chosen: list[Candidate] = []
        for ranked in plan.candidates:
            try:
                candidate = model_zoo.get_candidate(plan.task_type, ranked.name)
            except KeyError as exc:
                notes.append(str(exc))
                continue
            if candidate.baseline or candidate in chosen:
                continue
            chosen.append(candidate)
            if len(chosen) == budget.max_candidates:
                break
        # Baselines are cheap and always trained so the real models can be compared against them.
        if self.train_baselines:
            chosen += [c for c in model_zoo.candidates_for(plan.task_type, include_baselines=True) if c.baseline]
        if not chosen:
            raise TrainingError("The plan contains no usable candidate models.")
        return chosen

    # ------------------------------------------------------------------ training

    def _train_candidate(
        self, candidate: Candidate, data: _Prepared, metric: str, budget: TrainingBudget, remaining_s: float
    ) -> LeaderboardEntry:
        tune = candidate.tunable and self.tune_hyperparameters
        searched = candidate if tune else replace(candidate, search_space=_library_defaults)
        result = tuner.tune(
            searched, data.X_train, data.y_train, data.X_valid, data.y_valid, metric,
            n_trials=budget.n_trials_per_model if tune else 1,
            timeout_s=min(budget.timeout_per_model_s, remaining_s),
            seed=self.seed,
            context=data.context,
        )
        base = {"model": candidate.name, "n_trials": result.n_trials, "n_failed_trials": result.n_failed,
                "tuning_time_s": result.elapsed_s}
        if result.best_score is None:
            return LeaderboardEntry(**base, status="failed", error="; ".join(result.errors) or "No trial completed in time.")

        threshold = default_score = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = time.perf_counter()
                model = candidate.build(result.best_params, data.context).fit(data.X_train, data.y_train)
                fit_s = time.perf_counter() - start
                if self.tune_threshold and _threshold_applies(model, metric, data.context):
                    # Tuned on the validation split with the train-only model, i.e. before the final refit.
                    threshold, _, default_score = tune_threshold(
                        metric, data.y_valid, model.predict_proba(data.X_valid), data.context.pos_label, seed=self.seed
                    )
                valid_metrics = _evaluate_model(model, data.X_valid, data.y_valid, data.context, threshold)
                sample = data.valid[data.feature_columns].head(INFERENCE_SAMPLE_ROWS)
                stats = complexity(model, lambda: model.predict(data.features.transform(sample)), len(sample),
                                   fit_s, data.X_train.shape[1])
        except Exception as exc:  # refitting the best params can still fail (e.g. memory); skip this candidate
            return LeaderboardEntry(**base, status="failed", error=f"Refit failed: {type(exc).__name__}: {exc}")

        return LeaderboardEntry(
            **base,
            status="ok",
            params=result.best_params,
            valid_score=valid_metrics.get(metric, result.best_score),
            valid_metrics=valid_metrics,
            decision_threshold=threshold,
            valid_score_default_threshold=default_score,
            train_time_s=stats["train_time_s"],
            inference_ms_per_row=stats["inference_ms_per_row"],
            model_size_kb=stats["model_size_kb"],
            n_features=stats["n_features_after_preprocessing"],
        )

    def _refit(self, best: LeaderboardEntry, data: _Prepared, task: TaskType) -> tuple[Pipeline, float]:
        candidate = model_zoo.get_candidate(task, best.model)
        full = pd.concat([data.train, data.valid])  # chronological order is kept for time series
        context = replace(data.context, n_rows=len(full))
        final = Pipeline([*clone(data.features).steps, ("model", candidate.build(best.params, context))])
        y_full = None if data.y_train is None else np.concatenate([data.y_train, data.y_valid])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            final.fit(full[data.feature_columns], y_full)
        return final, time.perf_counter() - start

    def _final_evaluation(
        self, final: Pipeline, data: _Prepared, task: TaskType, fit_s: float, threshold: float | None
    ) -> "_Evaluation":
        """Score the refit model on the test split (read once), plus baseline and in-sample train metrics."""
        family = model_zoo.task_family(task)
        test = data.holdout.reveal("final_evaluation")
        X_test = test[data.feature_columns]
        y_test = None if data.label_column is None else _encode_like(test[data.label_column], data.classes)
        full = pd.concat([data.train, data.valid])
        y_full = None if data.y_train is None else np.concatenate([data.y_train, data.y_valid])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = np.asarray(final[:-1].transform(X_test), dtype="float64")
            sample = X_test.head(INFERENCE_SAMPLE_ROWS)
            stats = complexity(final, lambda: final.predict(sample), len(sample), fit_s, transformed.shape[1])
            test_metrics = self._score(final, X_test, y_test, transformed, data, task, threshold)
            default_metrics = (
                self._score(final, X_test, y_test, transformed, data, task, None) if threshold is not None else {}
            )
            train_transformed = np.asarray(final[:-1].transform(full[data.feature_columns]), dtype="float64")
            train_metrics = self._score(final, full[data.feature_columns], y_full, train_transformed, data, task, threshold)
            baseline = _baseline(final, transformed, y_test, y_full, data, task, self.seed)
            predictions = _test_predictions(final, test, X_test, data, task, threshold)

        if family == "classification" and data.classes is not None:
            test_metrics["class_labels"] = [str(c) for c in data.classes]
            if data.context.pos_label is not None:
                test_metrics["positive_class"] = str(data.classes[data.context.pos_label])
        test_metrics["n_test_rows"] = len(test)
        return _Evaluation(test_metrics, default_metrics, train_metrics, baseline, stats, predictions)

    def _score(
        self, model: Pipeline, X: pd.DataFrame, y: np.ndarray | None, transformed: np.ndarray, data: _Prepared,
        task: TaskType, threshold: float | None,
    ) -> dict[str, Any]:
        is_classification = model_zoo.task_family(task) == "classification"
        y_proba = model.predict_proba(X) if is_classification and hasattr(model, "predict_proba") else None
        if threshold is not None and y_proba is not None and data.context.pos_label is not None:
            y_pred = apply_threshold(y_proba, data.context.pos_label, threshold)
        else:
            y_pred = np.asarray(model.predict(X))
        return evaluate(task, EvalData(
            y_pred=y_pred, y_true=y, y_proba=y_proba, X=transformed,
            n_classes=data.context.n_classes, pos_label=data.context.pos_label, seed=self.seed,
        ))

    # ------------------------------------------------------------------ persistence

    def _save(
        self, run_dir: Path, final: Pipeline, best: LeaderboardEntry, entries: list[LeaderboardEntry], data: _Prepared,
        plan: ModelPlan, metric: str, direction: Direction, budget: TrainingBudget, evaluation: "_Evaluation",
        notes: list[str], total_s: float,
    ) -> TrainingResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        family = model_zoo.task_family(plan.task_type)
        trained = TrainedModel(
            pipeline=final,
            task_type=plan.task_type,
            target=data.label_column if family != "clustering" else None,
            feature_columns=data.feature_columns,
            model_name=best.model,
            classes=data.classes if family == "classification" else None,
            decision_threshold=best.decision_threshold,
            pos_label=data.context.pos_label,
        )
        model_path = run_dir / MODEL_FILENAME
        leaderboard_path = run_dir / LEADERBOARD_FILENAME
        metrics_path = run_dir / METRICS_FILENAME
        predictions_path = run_dir / PREDICTIONS_FILENAME
        joblib.dump(trained, model_path)
        evaluation.predictions.to_csv(predictions_path, index=False)
        leaderboard_path.write_text(json.dumps([e.model_dump() for e in entries], indent=2, default=str), encoding="utf-8")
        metrics_path.write_text(json.dumps({
            "model": best.model,
            "primary_metric": metric,
            "direction": direction,
            "decision_threshold": best.decision_threshold,
            "valid": best.valid_metrics,
            "test": evaluation.test,
            "test_at_threshold_0.5": evaluation.test_default_threshold,
            "train": evaluation.train,
            "baseline": evaluation.baseline.model_dump() if evaluation.baseline else None,
            "complexity": evaluation.complexity,
        }, indent=2, default=str), encoding="utf-8")

        return TrainingResult(
            task_type=plan.task_type,
            target=data.label_column,
            primary_metric=metric,
            direction=direction,
            best_model=best.model,
            best_params=best.params,
            valid_score=best.valid_score,
            test_metrics=evaluation.test,
            decision_threshold=best.decision_threshold,
            test_metrics_default_threshold=evaluation.test_default_threshold,
            train_metrics=evaluation.train,
            baseline=evaluation.baseline,
            complexity=evaluation.complexity,
            leaderboard=entries,
            budget=budget,
            total_time_s=round(total_s, 2),
            model_path=str(model_path),
            leaderboard_path=str(leaderboard_path),
            metrics_path=str(metrics_path),
            predictions_path=str(predictions_path),
            test_set_accesses=list(data.holdout.access_log),
            warnings=notes,
        )


# --------------------------------------------------------------------------- ranking & constraints


def mark_constraint_violations(entries: list[LeaderboardEntry], constraints: ModelConstraints) -> None:
    """Record which trained candidates break the user's inference-time or training-time limits."""
    for entry in entries:
        if entry.status != "ok":
            continue
        limit = constraints.max_inference_ms_per_row
        if limit is not None and entry.inference_ms_per_row is not None and entry.inference_ms_per_row > limit:
            entry.violations.append(f"inference {entry.inference_ms_per_row:.3g} ms/row > {limit:g} ms limit")
        limit = constraints.max_train_time_s
        if limit is not None and entry.train_time_s is not None and entry.train_time_s > limit:
            entry.violations.append(f"training {entry.train_time_s:.3g} s > {limit:g} s limit")


def rank_entries(entries: list[LeaderboardEntry], direction: Direction) -> list[LeaderboardEntry]:
    """Trained models by validation score (best first), then failed/skipped ones. Sets `rank`."""
    ok = [e for e in entries if e.status == "ok" and e.valid_score is not None]
    ok.sort(key=lambda e: e.valid_score, reverse=direction == "maximize")
    for position, entry in enumerate(ok, start=1):
        entry.rank = position
    return ok + [e for e in entries if e not in ok]


def select_best(ranked: list[LeaderboardEntry]) -> tuple[LeaderboardEntry, list[str]]:
    """Best-ranked model that meets all constraints; the best overall if none does (with a note)."""
    ok = [e for e in ranked if e.rank is not None]
    for entry in ok:
        if not entry.violations:
            if entry is not ok[0]:
                return entry, [f"Selected '{entry.model}' because better-scoring models break the constraints."]
            return entry, []
    return ok[0], [f"No model meets every constraint; selected the best-scoring model '{ok[0].model}' anyway."]


# --------------------------------------------------------------------------- helpers


def _encode_targets(
    train: pd.DataFrame, valid: pd.DataFrame, label: str | None, family: str
) -> tuple[np.ndarray | None, np.ndarray | None, list[Any] | None]:
    if label is None:
        return None, None, None
    if family in ("regression", "time_series"):
        return train[label].to_numpy(dtype="float64"), valid[label].to_numpy(dtype="float64"), None
    classes = sorted(pd.concat([train[label], valid[label]]).dropna().unique().tolist(), key=str)
    return _encode_like(train[label], classes), _encode_like(valid[label], classes), classes


def _encode_like(values: pd.Series, classes: list[Any] | None) -> np.ndarray:
    """Encode labels as indices into `classes`; labels never seen in training become -1."""
    if classes is None:
        return values.to_numpy(dtype="float64")
    index = {value: i for i, value in enumerate(classes)}
    return np.asarray([index.get(value, -1) for value in values], dtype=int)


def _evaluate_model(
    model: Any, X: np.ndarray, y: np.ndarray | None, ctx: ModelContext, threshold: float | None = None
) -> dict[str, Any]:
    family = model_zoo.task_family(ctx.task_type)
    proba = model.predict_proba(X) if family == "classification" and hasattr(model, "predict_proba") else None
    if threshold is not None and proba is not None and ctx.pos_label is not None:
        y_pred = apply_threshold(proba, ctx.pos_label, threshold)
    else:
        y_pred = np.asarray(model.predict(X))
    metrics = evaluate(ctx.task_type, EvalData(
        y_pred=y_pred, y_true=y, y_proba=proba,
        X=X if family == "clustering" else None, n_classes=ctx.n_classes, pos_label=ctx.pos_label, seed=ctx.seed,
    ))
    metrics.pop("confusion_matrix", None)  # kept for the final test evaluation only
    return metrics


def _library_defaults(trial: Any, ctx: ModelContext) -> dict[str, Any]:
    """Search space of a candidate whose tuning is switched off: no parameters, i.e. the library defaults."""
    return {}


def _threshold_applies(model: Any, metric: str, ctx: ModelContext) -> bool:
    return (
        ctx.n_classes == 2
        and ctx.pos_label is not None
        and metric in THRESHOLD_TUNABLE_METRICS
        and hasattr(model, "predict_proba")
    )


@dataclass
class _Evaluation:
    test: dict[str, Any]
    test_default_threshold: dict[str, Any]
    train: dict[str, Any]
    baseline: Baseline | None
    complexity: dict[str, Any]
    predictions: pd.DataFrame


def _test_predictions(
    final: Pipeline, test: pd.DataFrame, X_test: pd.DataFrame, data: _Prepared, task: TaskType, threshold: float | None
) -> pd.DataFrame:
    """Test-split predictions (original labels/values), saved so charts never need to re-read the test split."""
    family = model_zoo.task_family(task)
    out = pd.DataFrame(index=test.index)
    if family == "time_series" and "ts_features" in final.named_steps:
        out["time"] = test[final.named_steps["ts_features"].time_column].to_numpy()
    if family == "clustering":
        out["cluster"] = np.asarray(final.predict(X_test))
        return out.reset_index(drop=True)
    out["actual"] = test[data.label_column].to_numpy()
    if family == "classification" and data.classes is not None:
        proba = final.predict_proba(X_test)
        pos = data.context.pos_label
        encoded = apply_threshold(proba, pos, threshold) if threshold is not None and pos is not None else final.predict(X_test)
        out["predicted"] = np.asarray(data.classes, dtype=object)[np.asarray(encoded, dtype=int)]
        if pos is not None:
            out["probability"] = proba[:, pos]
    else:
        out["predicted"] = np.asarray(final.predict(X_test), dtype="float64")
    return out.reset_index(drop=True)


def _baseline(
    final: Pipeline, X_test_transformed: np.ndarray, y_test: np.ndarray | None, y_full: np.ndarray | None,
    data: _Prepared, task: TaskType, seed: int,
) -> Baseline | None:
    """Score a trivial reference model on the test split: majority class, mean, seasonal naive, or (clustering
    with ground truth) a random assignment into as many clusters as there are true groups."""
    family = model_zoo.task_family(task)
    if y_test is None or y_full is None:
        return None
    n = len(y_test)
    if family == "clustering":
        n_groups = len(np.unique(y_full))
        y_pred = np.random.default_rng(seed).integers(0, n_groups, n)
        metrics = evaluate(task, EvalData(y_pred=y_pred, y_true=y_test, X=X_test_transformed, seed=seed))
        return Baseline(name="random_assignment", metrics=metrics)
    if family == "classification":
        counts = np.bincount(y_full, minlength=data.context.n_classes or 0)
        priors = counts / counts.sum()
        name, y_pred, y_proba = "majority_class", np.full(n, int(np.argmax(counts))), np.tile(priors, (n, 1))
    elif family == "time_series":
        naive = model_zoo.get_candidate(task, "seasonal_naive").build({}, data.context)
        name, y_pred, y_proba = "seasonal_naive", naive.predict(X_test_transformed), None
    else:
        name, y_pred, y_proba = "mean", np.full(n, float(np.mean(y_full))), None
    metrics = evaluate(task, EvalData(
        y_pred=y_pred, y_true=y_test, y_proba=y_proba, n_classes=data.context.n_classes,
        pos_label=data.context.pos_label, seed=seed,
    ))
    metrics.pop("confusion_matrix", None)
    return Baseline(name=name, metrics=metrics)

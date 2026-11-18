"""Hyperparameter tuning with Optuna (TPE sampler + median pruner).

On larger training sets each trial is evaluated in two stages: first fitted on half of the training rows
(reported as step 0, so the median pruner can stop clearly worse settings early), then on all of them.
A trial that raises is recorded and pruned; tune() itself never raises because of a bad trial.
"""

import time
import warnings
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from pydantic import BaseModel, Field

from idss.core.logger import get_logger
from idss.ml.evaluator import METRICS, Direction, EvalData, compute_metric, metric_direction, normalize_metric_name
from idss.ml.model_zoo import Candidate, ModelContext, task_family

logger = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)

TWO_STAGE_MIN_ROWS = 2000
MEDIAN_PRUNER_STARTUP_TRIALS = 5
_MAX_REPORTED_ERRORS = 3


class TuneResult(BaseModel):
    candidate: str
    metric: str
    direction: Direction
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float | None = None
    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = Field(default_factory=list)


def score_model(model: Any, X: np.ndarray, y: np.ndarray | None, metric: str, ctx: ModelContext) -> float | None:
    """Score a fitted model on (X, y) with `metric`; None if the metric is undefined for these predictions."""
    metric = normalize_metric_name(metric)
    y_proba = model.predict_proba(X) if METRICS[metric].needs_proba and hasattr(model, "predict_proba") else None
    data = EvalData(
        y_pred=np.asarray(model.predict(X)),
        y_true=y,
        y_proba=y_proba,
        X=X if task_family(ctx.task_type) == "clustering" else None,
        n_classes=ctx.n_classes,
        pos_label=ctx.pos_label,
        seed=ctx.seed,
    )
    return compute_metric(metric, data)


def tune(
    candidate: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_valid: np.ndarray,
    y_valid: np.ndarray | None,
    metric: str,
    n_trials: int,
    timeout_s: float,
    seed: int = 42,
    *,
    context: ModelContext,
) -> TuneResult:
    """Search `candidate`'s hyperparameters, fitting on train and scoring `metric` on valid."""
    direction = metric_direction(metric)
    two_stage = candidate.tunable and len(X_train) >= TWO_STAGE_MIN_ROWS
    half = np.sort(np.random.default_rng(seed).choice(len(X_train), len(X_train) // 2, replace=False))

    def fit_and_score(params: dict[str, Any], rows: np.ndarray | None) -> float:
        model = candidate.build(params, context)
        if rows is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train[rows], None if y_train is None else y_train[rows])
        score = score_model(model, X_valid, y_valid, metric, context)
        if score is None:
            raise ValueError(f"metric '{metric}' is undefined for this model's predictions")
        return score

    def objective(trial: optuna.Trial) -> float:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = candidate.search_space(trial, context)
                if two_stage:
                    trial.report(fit_and_score(params, half), step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned("worse than the median on the half-data stage")
                score = fit_and_score(params, None)
            trial.report(score, step=1)
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # a bad configuration must never stop the search
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:300])
            raise optuna.TrialPruned("trial raised an exception") from exc

    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=MEDIAN_PRUNER_STARTUP_TRIALS),
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, catch=(Exception,), gc_after_trial=False)
    elapsed = time.perf_counter() - start

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if "error" in t.user_attrs or t.state == TrialState.FAIL]
    errors = list(dict.fromkeys(t.user_attrs.get("error", "trial failed") for t in failed))[:_MAX_REPORTED_ERRORS]
    result = TuneResult(
        candidate=candidate.name,
        metric=normalize_metric_name(metric),
        direction=direction,
        n_trials=len(study.trials),
        n_completed=len(completed),
        n_pruned=sum(1 for t in study.trials if t.state == TrialState.PRUNED) - sum(1 for t in failed if t.state == TrialState.PRUNED),
        n_failed=len(failed),
        elapsed_s=round(elapsed, 3),
        errors=errors,
    )
    if completed:
        best = study.best_trial
        result.best_params = dict(best.params)
        result.best_score = float(best.value)
    logger.info(
        "Tuned {}: best {}={} after {} trials ({} failed) in {:.1f}s.",
        candidate.name, result.metric, result.best_score, result.n_trials, result.n_failed, elapsed,
    )
    return result

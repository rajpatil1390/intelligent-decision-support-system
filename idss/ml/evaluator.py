"""Metrics for every task type, plus model-complexity measurements.

One registry (METRICS) defines every metric's family, direction and requirements, so tuning, ranking,
requirement repair and reporting all agree on what a metric means.
"""

import io
import math
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import joblib
import numpy as np
from sklearn import metrics as skm

Direction = Literal["maximize", "minimize"]
MetricFamily = Literal["classification", "regression", "clustering"]

METRIC_FAMILY_BY_TASK: dict[str, MetricFamily] = {
    "tabular_classification": "classification",
    "text_classification": "classification",
    "tabular_regression": "regression",
    "time_series_forecasting": "regression",
    "tabular_clustering": "clustering",
}
METRIC_ALIASES = {
    "f1_score": "f1", "auc": "roc_auc", "roc": "roc_auc", "accuracy_score": "accuracy",
    "root_mean_squared_error": "rmse", "mean_squared_error": "mse", "mean_absolute_error": "mae",
    "r_squared": "r2", "r^2": "r2", "silhouette_score": "silhouette", "ari": "adjusted_rand_index",
    "normalized_mutual_info": "nmi", "normalized_mutual_information": "nmi",
}
_SILHOUETTE_SAMPLE = 2000
_TIMING_REPEATS = 3


@dataclass(frozen=True)
class EvalData:
    """Inputs for computing metrics. Classification labels are encoded as 0..n_classes-1."""

    y_pred: np.ndarray
    y_true: np.ndarray | None = None
    y_proba: np.ndarray | None = None
    X: np.ndarray | None = None  # features, for internal clustering metrics
    n_classes: int | None = None
    pos_label: int | None = None  # binary classification: the class of interest (the minority)
    seed: int = 42


@dataclass(frozen=True)
class MetricSpec:
    family: MetricFamily
    direction: Direction
    fn: Callable[[EvalData], float | None]
    needs_proba: bool = False
    needs_labels: bool = True  # clustering: needs ground-truth labels


# --------------------------------------------------------------------------- classification


def _binary(d: EvalData) -> bool:
    return d.n_classes == 2 and d.pos_label is not None


def _averaged(score: Callable[..., float]) -> Callable[[EvalData], float | None]:
    def compute(d: EvalData) -> float | None:
        if _binary(d):
            return score(d.y_true, d.y_pred, average="binary", pos_label=d.pos_label, zero_division=0)
        return score(d.y_true, d.y_pred, average="weighted", zero_division=0)

    return compute


def _roc_auc(d: EvalData) -> float | None:
    if d.y_proba is None:
        return None
    try:
        if _binary(d):
            return skm.roc_auc_score((d.y_true == d.pos_label).astype(int), d.y_proba[:, d.pos_label])
        return skm.roc_auc_score(d.y_true, d.y_proba, multi_class="ovr", average="weighted", labels=list(range(d.n_classes or 0)))
    except (ValueError, IndexError):
        return None  # e.g. only one class present in y_true


def _log_loss(d: EvalData) -> float | None:
    if d.y_proba is None:
        return None
    try:
        return skm.log_loss(d.y_true, d.y_proba, labels=list(range(d.n_classes or d.y_proba.shape[1])))
    except ValueError:
        return None


# --------------------------------------------------------------------------- regression


def _mape(d: EvalData) -> float | None:
    if np.any(d.y_true == 0):
        return None
    return skm.mean_absolute_percentage_error(d.y_true, d.y_pred)


def _rmsle(d: EvalData) -> float | None:
    if np.any(d.y_true < 0) or np.any(d.y_pred < 0):
        return None  # only defined for non-negative values
    return skm.root_mean_squared_log_error(d.y_true, d.y_pred)


def _r2(d: EvalData) -> float | None:
    return skm.r2_score(d.y_true, d.y_pred) if len(d.y_true) >= 2 else None


# --------------------------------------------------------------------------- clustering


def _internal(score: Callable[..., float], sample: bool = False) -> Callable[[EvalData], float | None]:
    def compute(d: EvalData) -> float | None:
        n_clusters = len(np.unique(d.y_pred))
        if d.X is None or not 2 <= n_clusters < len(d.y_pred):
            return None
        if sample and len(d.y_pred) > _SILHOUETTE_SAMPLE:
            return score(d.X, d.y_pred, sample_size=_SILHOUETTE_SAMPLE, random_state=d.seed)
        return score(d.X, d.y_pred)

    return compute


def _external(score: Callable[[Any, Any], float]) -> Callable[[EvalData], float | None]:
    return lambda d: None if d.y_true is None else score(d.y_true, d.y_pred)


METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("classification", "maximize", lambda d: skm.accuracy_score(d.y_true, d.y_pred)),
    "balanced_accuracy": MetricSpec("classification", "maximize", lambda d: skm.balanced_accuracy_score(d.y_true, d.y_pred)),
    "f1": MetricSpec("classification", "maximize", _averaged(skm.f1_score)),
    "precision": MetricSpec("classification", "maximize", _averaged(skm.precision_score)),
    "recall": MetricSpec("classification", "maximize", _averaged(skm.recall_score)),
    "roc_auc": MetricSpec("classification", "maximize", _roc_auc, needs_proba=True),
    "log_loss": MetricSpec("classification", "minimize", _log_loss, needs_proba=True),
    "rmse": MetricSpec("regression", "minimize", lambda d: skm.root_mean_squared_error(d.y_true, d.y_pred)),
    "mse": MetricSpec("regression", "minimize", lambda d: skm.mean_squared_error(d.y_true, d.y_pred)),
    "mae": MetricSpec("regression", "minimize", lambda d: skm.mean_absolute_error(d.y_true, d.y_pred)),
    "r2": MetricSpec("regression", "maximize", _r2),
    "mape": MetricSpec("regression", "minimize", _mape),
    "rmsle": MetricSpec("regression", "minimize", _rmsle),
    "silhouette": MetricSpec("clustering", "maximize", _internal(skm.silhouette_score, sample=True), needs_labels=False),
    "davies_bouldin": MetricSpec("clustering", "minimize", _internal(skm.davies_bouldin_score), needs_labels=False),
    "calinski_harabasz": MetricSpec("clustering", "maximize", _internal(skm.calinski_harabasz_score), needs_labels=False),
    "rand_index": MetricSpec("clustering", "maximize", _external(skm.rand_score)),
    "adjusted_rand_index": MetricSpec("clustering", "maximize", _external(skm.adjusted_rand_score)),
    "nmi": MetricSpec("clustering", "maximize", _external(skm.normalized_mutual_info_score)),
}


# --------------------------------------------------------------------------- public API


def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(key, key)


def metric_direction(metric: str) -> Direction:
    return METRICS[normalize_metric_name(metric)].direction


def is_supported(metric: str, task_type: str, *, labels_available: bool = True) -> bool:
    """Whether `metric` can be computed for `task_type` (clustering label metrics need ground truth)."""
    spec = METRICS.get(normalize_metric_name(metric))
    if spec is None or spec.family != METRIC_FAMILY_BY_TASK[task_type]:
        return False
    return labels_available or spec.family != "clustering" or not spec.needs_labels


def compute_metric(metric: str, data: EvalData) -> float | None:
    """One metric value, or None when it is not defined for this data."""
    spec = METRICS[normalize_metric_name(metric)]
    if spec.needs_labels and data.y_true is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spec.fn(data)
    return None if value is None or not math.isfinite(value) else float(value)


def evaluate(task_type: str, data: EvalData) -> dict[str, Any]:
    """Every metric of the task's family that is defined for `data` (values rounded to 6 places)."""
    family = METRIC_FAMILY_BY_TASK[task_type]
    results: dict[str, Any] = {}
    for name, spec in METRICS.items():
        if spec.family != family:
            continue
        value = compute_metric(name, data)
        if value is not None:
            results[name] = round(value, 6)
    if family == "classification" and data.y_true is not None:
        labels = list(range(data.n_classes or 0))
        results["confusion_matrix"] = skm.confusion_matrix(data.y_true, data.y_pred, labels=labels).tolist()
    return results


# Metrics worth tuning a decision threshold for. Precision or recall alone would be maximised by a
# degenerate threshold (predict almost nothing / everything positive), so they keep 0.5.
THRESHOLD_TUNABLE_METRICS = frozenset({"f1", "accuracy", "balanced_accuracy"})
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
DEFAULT_THRESHOLD = 0.5


def apply_threshold(proba: np.ndarray, pos_label: int, threshold: float) -> np.ndarray:
    """Binary predictions: the positive class wherever its probability reaches `threshold`."""
    return np.where(proba[:, pos_label] >= threshold, pos_label, 1 - pos_label)


def tune_threshold(
    metric: str, y_true: np.ndarray, proba: np.ndarray, pos_label: int, *, seed: int = 42
) -> tuple[float, float | None, float | None]:
    """Threshold maximising `metric` on (y_true, proba); returns (threshold, score_at_it, score_at_0.5).

    Ties keep 0.5, so the tuned score is never below the default one on the data used for tuning.
    """
    def score(threshold: float) -> float | None:
        data = EvalData(y_pred=apply_threshold(proba, pos_label, threshold), y_true=y_true, n_classes=2,
                        pos_label=pos_label, seed=seed)
        return compute_metric(metric, data)

    default_score = score(DEFAULT_THRESHOLD)
    best_threshold, best_score = DEFAULT_THRESHOLD, default_score
    for threshold in THRESHOLD_GRID:
        value = score(float(threshold))
        if value is not None and (best_score is None or value > best_score + 1e-12):
            best_threshold, best_score = float(threshold), value
    return best_threshold, best_score, default_score


def model_size_kb(model: Any) -> float:
    """Size of the model when serialised with joblib."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return round(len(buffer.getvalue()) / 1024, 2)


def inference_ms_per_row(predict: Callable[[], Any], n_rows: int, repeats: int = _TIMING_REPEATS) -> float:
    """Median over `repeats` runs of the time `predict()` takes, per row, in milliseconds."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict()
        timings.append(time.perf_counter() - start)
    return round(statistics.median(timings) * 1000 / max(n_rows, 1), 4)


def complexity(model: Any, predict: Callable[[], Any], n_rows: int, train_time_s: float, n_features: int) -> dict[str, Any]:
    """Standard complexity block reported for every trained model."""
    return {
        "train_time_s": round(train_time_s, 3),
        "inference_ms_per_row": inference_ms_per_row(predict, n_rows),
        "model_size_kb": model_size_kb(model),
        "n_features_after_preprocessing": n_features,
    }

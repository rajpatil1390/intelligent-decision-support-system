"""Registry of every model the AutoML engine may train, each with a factory and an Optuna search space.

The Model Agent (LLM) may only pick names from this registry; it never writes training code. Factories
receive a ModelContext with facts about the training data (size, classes, imbalance, season length).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

from idss.core.schemas import TaskType
from idss.ml.ts_features import FEATURE_PREFIX

ModelFamily = Literal["classification", "regression", "clustering", "time_series"]
Params = dict[str, Any]

IMBALANCE_THRESHOLD = 0.20  # minority share below which class weighting is switched on
MINIBATCH_KMEANS_ROWS = 20_000
AGGLOMERATIVE_MAX_FIT_ROWS = 5_000
MAX_CLUSTERS = 10

TASK_TO_FAMILY: dict[str, ModelFamily] = {
    "tabular_classification": "classification",
    "text_classification": "classification",
    "tabular_regression": "regression",
    "time_series_forecasting": "time_series",
    "tabular_clustering": "clustering",
}
MODEL_ALIASES = {
    "xgb": "xgboost", "xgbclassifier": "xgboost", "xgbregressor": "xgboost",
    "lgbm": "lightgbm", "light_gbm": "lightgbm", "lgbmclassifier": "lightgbm", "lgbmregressor": "lightgbm",
    "rf": "random_forest", "random_forests": "random_forest", "randomforest": "random_forest",
    "extra_tree": "extra_trees", "extratrees": "extra_trees", "logistic": "logistic_regression",
    "logisticregression": "logistic_regression", "hist_gradient_boosting_classifier": "hist_gradient_boosting",
    "histgradientboosting": "hist_gradient_boosting", "hgb": "hist_gradient_boosting", "gbm": "lightgbm",
    "k_means": "kmeans", "gmm": "gaussian_mixture", "hierarchical": "agglomerative",
    "agglomerative_clustering": "agglomerative", "naive": "seasonal_naive", "ridge_regression": "ridge",
}


@dataclass(frozen=True)
class ModelContext:
    """Facts about the training data that factories and search spaces may use."""

    task_type: TaskType
    n_rows: int
    seed: int = 42
    n_classes: int | None = None
    class_counts: tuple[int, ...] = ()  # by encoded class index
    pos_label: int | None = None  # binary: encoded index of the minority class
    seasonal_period: int = 1
    feature_names: tuple[str, ...] = ()  # model input columns after preprocessing

    @property
    def minority_share(self) -> float | None:
        return min(self.class_counts) / sum(self.class_counts) if self.class_counts else None

    @property
    def imbalanced(self) -> bool:
        share = self.minority_share
        return share is not None and share < IMBALANCE_THRESHOLD


@dataclass(frozen=True)
class Candidate:
    name: str
    family: ModelFamily
    description: str
    build: Callable[[Params, ModelContext], Any]
    search_space: Callable[[optuna.Trial, ModelContext], Params] = field(default=lambda trial, ctx: {})
    interpretable: bool = False
    baseline: bool = False

    @property
    def tunable(self) -> bool:
        return not self.baseline


# --------------------------------------------------------------------------- helper estimators


class SeasonalNaiveRegressor(RegressorMixin, BaseEstimator):
    """Baseline forecast: the value one season ago (read from the lag feature at `lag_index`)."""

    def __init__(self, lag_index: int = 0) -> None:
        self.lag_index = lag_index

    def fit(self, X: np.ndarray, y: object = None) -> "SeasonalNaiveRegressor":
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype="float64")[:, self.lag_index]


class InductiveClusterer(ClusterMixin, BaseEstimator):
    """Agglomerative clustering with a predict() method: a 1-nearest-neighbour model labels new rows.

    Fits on at most `max_fit_rows` rows, because agglomerative clustering needs O(n^2) memory.
    """

    def __init__(self, n_clusters: int = 3, linkage: str = "ward", max_fit_rows: int = AGGLOMERATIVE_MAX_FIT_ROWS,
                 random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.max_fit_rows = max_fit_rows
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: object = None) -> "InductiveClusterer":
        X = np.asarray(X, dtype="float64")
        rows = np.arange(len(X))
        if len(X) > self.max_fit_rows:
            rows = np.sort(np.random.default_rng(self.random_state).choice(len(X), self.max_fit_rows, replace=False))
        labels = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage).fit_predict(X[rows])
        self.classifier_ = KNeighborsClassifier(n_neighbors=1).fit(X[rows], labels)
        self.labels_ = self.classifier_.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier_.predict(np.asarray(X, dtype="float64"))


# --------------------------------------------------------------------------- shared search spaces


def _class_weight(ctx: ModelContext) -> str | None:
    return "balanced" if ctx.imbalanced else None


def _scale_pos_weight(ctx: ModelContext) -> float:
    if ctx.imbalanced and ctx.n_classes == 2 and len(ctx.class_counts) == 2:
        return ctx.class_counts[0] / ctx.class_counts[1]  # XGBoost: weight of class 1 relative to class 0
    return 1.0


def _forest_space(trial: optuna.Trial, ctx: ModelContext) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }


def _xgb_space(trial: optuna.Trial, ctx: ModelContext) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lgbm_space(trial: optuna.Trial, ctx: ModelContext) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, max(5, min(50, ctx.n_rows // 10))),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _hgb_space(trial: optuna.Trial, ctx: ModelContext) -> Params:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 50, 300, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, max(5, min(50, ctx.n_rows // 10))),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }


def _lgbm_common(ctx: ModelContext) -> Params:
    return {"subsample_freq": 1, "random_state": ctx.seed, "verbosity": -1, "n_jobs": -1}


def _xgb_common(ctx: ModelContext) -> Params:
    return {"tree_method": "hist", "random_state": ctx.seed, "n_jobs": -1, "verbosity": 0}


# --------------------------------------------------------------------------- classification


_CLASSIFICATION = [
    Candidate(
        "logistic_regression", "classification", "Linear model; fast, well calibrated and easy to interpret.",
        build=lambda p, ctx: LogisticRegression(**p, max_iter=2000, class_weight=_class_weight(ctx)),
        search_space=lambda t, ctx: {"C": t.suggest_float("C", 1e-3, 100.0, log=True)},
        interpretable=True,
    ),
    Candidate(
        "random_forest", "classification", "Bagged decision trees; robust with little tuning.",
        build=lambda p, ctx: RandomForestClassifier(**p, class_weight=_class_weight(ctx), random_state=ctx.seed, n_jobs=-1),
        search_space=_forest_space,
    ),
    Candidate(
        "extra_trees", "classification", "Extremely randomised trees; like random forest but faster and smoother.",
        build=lambda p, ctx: ExtraTreesClassifier(**p, class_weight=_class_weight(ctx), random_state=ctx.seed, n_jobs=-1),
        search_space=_forest_space,
    ),
    Candidate(
        "xgboost", "classification", "Gradient-boosted trees (XGBoost); often the most accurate on tabular data.",
        build=lambda p, ctx: XGBClassifier(**p, **_xgb_common(ctx), scale_pos_weight=_scale_pos_weight(ctx)),
        search_space=_xgb_space,
    ),
    Candidate(
        "lightgbm", "classification", "Gradient-boosted trees (LightGBM); accurate and very fast.",
        build=lambda p, ctx: LGBMClassifier(**p, **_lgbm_common(ctx), class_weight=_class_weight(ctx)),
        search_space=_lgbm_space,
    ),
    Candidate(
        "hist_gradient_boosting", "classification", "scikit-learn histogram gradient boosting; handles larger data well.",
        build=lambda p, ctx: HistGradientBoostingClassifier(**p, class_weight=_class_weight(ctx), random_state=ctx.seed),
        search_space=_hgb_space,
    ),
]

# --------------------------------------------------------------------------- regression


def _ridge(p: Params, ctx: ModelContext) -> Ridge:
    return Ridge(**p)


def _ridge_space(trial: optuna.Trial, ctx: ModelContext) -> Params:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


_RIDGE = Candidate("ridge", "regression", "Regularised linear regression; fast and easy to interpret.",
                   build=_ridge, search_space=_ridge_space, interpretable=True)
_LGBM_REG = Candidate("lightgbm", "regression", "Gradient-boosted trees (LightGBM); accurate and very fast.",
                      build=lambda p, ctx: LGBMRegressor(**p, **_lgbm_common(ctx)), search_space=_lgbm_space)
_XGB_REG = Candidate("xgboost", "regression", "Gradient-boosted trees (XGBoost); often the most accurate on tabular data.",
                     build=lambda p, ctx: XGBRegressor(**p, **_xgb_common(ctx)), search_space=_xgb_space)

_REGRESSION = [
    _RIDGE,
    Candidate("random_forest", "regression", "Bagged decision trees; robust with little tuning.",
              build=lambda p, ctx: RandomForestRegressor(**p, random_state=ctx.seed, n_jobs=-1), search_space=_forest_space),
    Candidate("extra_trees", "regression", "Extremely randomised trees; like random forest but faster and smoother.",
              build=lambda p, ctx: ExtraTreesRegressor(**p, random_state=ctx.seed, n_jobs=-1), search_space=_forest_space),
    _XGB_REG,
    _LGBM_REG,
    Candidate("hist_gradient_boosting", "regression", "scikit-learn histogram gradient boosting; handles larger data well.",
              build=lambda p, ctx: HistGradientBoostingRegressor(**p, random_state=ctx.seed), search_space=_hgb_space),
]

# --------------------------------------------------------------------------- clustering


def _max_clusters(ctx: ModelContext) -> int:
    return max(2, min(MAX_CLUSTERS, ctx.n_rows // 5))


def _kmeans(p: Params, ctx: ModelContext) -> KMeans | MiniBatchKMeans:
    if ctx.n_rows > MINIBATCH_KMEANS_ROWS:
        return MiniBatchKMeans(**p, random_state=ctx.seed, n_init=3)
    return KMeans(**p, random_state=ctx.seed, n_init=5)


def _gaussian_mixture(p: Params, ctx: ModelContext) -> GaussianMixture:
    # Several EM starts: a single one can stop in a poor local optimum, so the final refit on train+valid would
    # no longer match the configuration that won the search (seen on the segments sample: ARI 0.53 instead of 0.84).
    return GaussianMixture(**p, random_state=ctx.seed, n_init=3 if ctx.n_rows > MINIBATCH_KMEANS_ROWS else 5)


_CLUSTERING = [
    Candidate("kmeans", "clustering", "K-means (mini-batch above 20k rows); fast, round clusters.",
              build=_kmeans,
              search_space=lambda t, ctx: {"n_clusters": t.suggest_int("n_clusters", 2, _max_clusters(ctx))},
              interpretable=True),
    Candidate("gaussian_mixture", "clustering", "Gaussian mixture; soft, elliptical clusters.",
              build=_gaussian_mixture,
              search_space=lambda t, ctx: {
                  "n_components": t.suggest_int("n_components", 2, _max_clusters(ctx)),
                  "covariance_type": t.suggest_categorical("covariance_type", ["full", "diag", "tied", "spherical"]),
              }),
    Candidate("agglomerative", "clustering", "Hierarchical (agglomerative) clustering; flexible cluster shapes.",
              build=lambda p, ctx: InductiveClusterer(**p, random_state=ctx.seed),
              search_space=lambda t, ctx: {
                  "n_clusters": t.suggest_int("n_clusters", 2, _max_clusters(ctx)),
                  "linkage": t.suggest_categorical("linkage", ["ward", "complete", "average"]),
              }),
]

# --------------------------------------------------------------------------- time series


def _seasonal_naive(p: Params, ctx: ModelContext) -> SeasonalNaiveRegressor:
    for lag in (ctx.seasonal_period, 1):
        suffix = f"{FEATURE_PREFIX}lag_{lag}"
        matches = [i for i, name in enumerate(ctx.feature_names) if name.endswith(suffix)]
        if matches:
            return SeasonalNaiveRegressor(lag_index=matches[0])
    raise ValueError("Seasonal-naive baseline needs a lag feature, but none was generated.")


_TIME_SERIES = [
    Candidate("lightgbm", "time_series", _LGBM_REG.description + " Uses lag, rolling and calendar features.",
              build=_LGBM_REG.build, search_space=_lgbm_space),
    Candidate("xgboost", "time_series", _XGB_REG.description + " Uses lag, rolling and calendar features.",
              build=_XGB_REG.build, search_space=_xgb_space),
    Candidate("ridge", "time_series", "Linear model on lag, rolling and calendar features; fast and interpretable.",
              build=_ridge, search_space=_ridge_space, interpretable=True),
    Candidate("seasonal_naive", "time_series", "Baseline: repeats the value from one season earlier.",
              build=_seasonal_naive, interpretable=True, baseline=True),
]

REGISTRY: dict[ModelFamily, dict[str, Candidate]] = {
    family: {candidate.name: candidate for candidate in candidates}
    for family, candidates in (
        ("classification", _CLASSIFICATION),
        ("regression", _REGRESSION),
        ("clustering", _CLUSTERING),
        ("time_series", _TIME_SERIES),
    )
}

DEFAULT_RANKING: dict[ModelFamily, list[str]] = {
    "classification": ["lightgbm", "logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting", "extra_trees"],
    "regression": ["lightgbm", "ridge", "random_forest", "xgboost", "hist_gradient_boosting", "extra_trees"],
    "clustering": ["kmeans", "gaussian_mixture", "agglomerative"],
    "time_series": ["lightgbm", "ridge", "xgboost"],
}


def task_family(task_type: str) -> ModelFamily:
    return TASK_TO_FAMILY[task_type]


def normalize_model_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return MODEL_ALIASES.get(key, MODEL_ALIASES.get(key.replace("_", ""), key))


def candidates_for(task_type: str, *, include_baselines: bool = False) -> list[Candidate]:
    return [c for c in REGISTRY[task_family(task_type)].values() if include_baselines or not c.baseline]


def candidate_names(task_type: str, *, include_baselines: bool = False) -> list[str]:
    return [c.name for c in candidates_for(task_type, include_baselines=include_baselines)]


def get_candidate(task_type: str, name: str) -> Candidate:
    """Look up a candidate by (normalised) name. Raises KeyError for names outside the zoo."""
    family = task_family(task_type)
    key = normalize_model_name(name)
    if key not in REGISTRY[family]:
        raise KeyError(f"'{name}' is not an allowed {family} model. Allowed: {', '.join(REGISTRY[family])}.")
    return REGISTRY[family][key]

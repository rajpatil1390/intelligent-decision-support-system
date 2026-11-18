"""Explanations for business users: global drivers, per-row explanations, cluster profiles, forecasts
and the PNG charts used in the report.

SHAP values are computed on the model's input columns after preprocessing and then summed back into the
user's ORIGINAL columns (one-hot categories -> their column, date parts -> the date column, lag/rolling
features -> "<target> (history)"), so drivers read like the user's own data. If SHAP is unavailable for a
model, sklearn's permutation importance on the original columns is used instead.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: charts are only ever written to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from idss.core.logger import get_logger  # noqa: E402
from idss.core.schemas import (  # noqa: E402
    ClusterProfile,
    DatasetProfile,
    ExplanationResult,
    FeatureImpact,
    Forecast,
    ForecastPoint,
    GlobalExplanation,
    LocalContribution,
    LocalExplanation,
    TaskType,
    TrainingResult,
)
from idss.ml import model_zoo  # noqa: E402
from idss.ml.profiler import parse_datetimes  # noqa: E402
from idss.ml.trainer import TrainedModel  # noqa: E402
from idss.ml.ts_features import infer_seasonal_period  # noqa: E402

logger = get_logger()

MAX_EXPLAIN_ROWS = 500
SHAP_BACKGROUND_ROWS = 100
TOP_FEATURES = 10
LOCAL_TOP = 5
DIRECTION_MIN_CORR = 0.1
FORECAST_HORIZON = 30
Z_95 = 1.96
TREE_MODELS = (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
               XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor,
               HistGradientBoostingClassifier, HistGradientBoostingRegressor)
LINEAR_MODELS = (LogisticRegression, Ridge)
_BRANCH_PREFIXES = ("num__", "cat__", "date__", "extra__")


class ModelExplainer:
    """SHAP (or permutation-importance) explanations for a TrainedModel, reported per ORIGINAL column."""

    def __init__(self, model: TrainedModel, data: pd.DataFrame, *, seed: int = 42) -> None:
        self.model = model
        self.seed = seed
        self.features = model.pipeline[:-1]
        self.estimator = model.pipeline[-1]
        self.family = model_zoo.task_family(model.task_type)
        self.binary = model.classes is not None and len(model.classes) == 2 and model.pos_label is not None
        self.raw, self.Xt = self._sample(data)
        preprocess = model.pipeline.named_steps["preprocess"]
        self.names = [str(name) for name in preprocess.get_feature_names_out()]
        self.origins = original_feature_names(model)
        self.method, self._explainer = self._make_explainer()
        self._values: np.ndarray | None = None
        self.signed = True

    # ------------------------------------------------------------------ public

    @property
    def output(self) -> str:
        target = self.model.target or "the target"
        if self.binary:
            return f"probability that {target} = '{self.model.classes[self.model.pos_label]}'"
        if self.model.classes is not None:
            return f"predicted class of {target}"
        return f"predicted {target}"

    def global_explanation(self) -> GlobalExplanation:
        notes: list[str] = []
        if self.method in ("shap_tree", "shap_linear"):
            try:
                features = self._shap_global()
                if self.method == "shap_tree" and isinstance(self.estimator, (LGBMClassifier, XGBClassifier,
                                                                               HistGradientBoostingClassifier)):
                    notes.append("SHAP values for boosted trees are on the log-odds scale; shares and directions are unaffected.")
                return GlobalExplanation(method=self.method, output=self.output, features=features,
                                         n_samples=len(self.raw), notes=notes)
            except Exception as exc:  # fall back to permutation importance
                logger.warning("SHAP explanation failed ({}: {}); using permutation importance.", type(exc).__name__, exc)
                notes.append(f"SHAP failed ({type(exc).__name__}); permutation importance used instead.")
                self.method = "permutation"
        features = self._permutation_global()
        method = "permutation" if features else "none"
        if not features:
            notes.append("No explanation could be computed (no target values available for permutation importance).")
        return GlobalExplanation(method=method, output=self.output, features=features, n_samples=len(self.raw), notes=notes)

    def contributions(self) -> pd.DataFrame:
        """Signed SHAP contribution of every original column for every sampled row (index = sample rows)."""
        values = self._shap_values()
        return _group_columns(values, self.origins).set_index(self.raw.index)

    def local(self, position: int) -> LocalExplanation:
        """Top contributing original columns for the sampled row at `position`."""
        prediction = self._predict_output(self.raw.iloc[[position]])[0]
        if self.method not in ("shap_tree", "shap_linear") or not self.signed:
            return LocalExplanation(method="unavailable", prediction=_py(prediction))
        row = self.contributions().iloc[position]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(LOCAL_TOP)
        contributions = [
            LocalContribution(feature=name, value=_py(self.raw.iloc[position].get(name)), impact=round(float(impact), 5))
            for name, impact in top.items()
        ]
        return LocalExplanation(method=self.method, prediction=_py(prediction), contributions=contributions)

    def predict_output(self, raw: pd.DataFrame) -> np.ndarray:
        return self._predict_output(raw)

    # ------------------------------------------------------------------ internals

    def _sample(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        columns = self.model.feature_columns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.family == "time_series":
                # Lags need consecutive rows: transform the recent history, then keep the last rows.
                ts = self.model.pipeline.named_steps["ts_features"]
                ordered = data.sort_values(ts.time_column, key=parse_datetimes, kind="stable")
                recent = ordered.tail(MAX_EXPLAIN_ROWS + max([*ts.lags, *ts.windows]) + 1)
                Xt = np.asarray(self.features.transform(recent[columns]), dtype="float64")
                return recent.iloc[-MAX_EXPLAIN_ROWS:].copy(), Xt[-MAX_EXPLAIN_ROWS:]
            raw = data if len(data) <= MAX_EXPLAIN_ROWS else data.sample(MAX_EXPLAIN_ROWS, random_state=self.seed)
            return raw.copy(), np.asarray(self.features.transform(raw[columns]), dtype="float64")

    def _make_explainer(self) -> tuple[str, Any]:
        try:
            if isinstance(self.estimator, TREE_MODELS):
                return "shap_tree", shap.TreeExplainer(self.estimator)
            if isinstance(self.estimator, LINEAR_MODELS):
                background = self.Xt[: min(SHAP_BACKGROUND_ROWS, len(self.Xt))]
                masker = shap.maskers.Independent(background, max_samples=len(background))
                return "shap_linear", shap.LinearExplainer(self.estimator, masker)
        except Exception as exc:
            logger.warning("Could not create a SHAP explainer ({}: {}).", type(exc).__name__, exc)
        return "permutation", None

    def _shap_values(self) -> np.ndarray:
        if self._values is not None:
            return self._values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.method == "shap_tree":
                explanation = self._explainer(self.Xt, check_additivity=False)
            else:
                explanation = self._explainer(self.Xt)
        values = np.asarray(explanation.values, dtype="float64")
        if values.ndim == 3:
            if self.binary:
                values = values[:, :, self.model.pos_label]
            else:  # multiclass: magnitude only, no single direction
                values, self.signed = np.abs(values).sum(axis=2), False
        elif self.binary and self.model.pos_label == 0:
            values = -values  # single-output models explain class 1
        self._values = values
        return values

    def _shap_global(self) -> list[FeatureImpact]:
        values = self._shap_values()
        importance = _group_columns(np.abs(values), self.origins).mean()  # sum of |SHAP| per original column
        details: dict[str, tuple[str, str]] = {}
        for origin in importance.index:
            positions = [i for i, o in enumerate(self.origins) if o == origin]
            details[origin] = self._direction(origin, positions, values)
        return _impacts(importance, details)

    def _direction(self, origin: str, positions: list[int], values: np.ndarray) -> tuple[str, str]:
        ts = self.model.pipeline.named_steps.get("ts_features")
        if ts is not None and origin == ts.time_column:
            return "mixed", "Trend over time and calendar effects (year, season, day of week)."
        if ts is not None and origin == f"{ts.target} (history)":
            return "mixed", f"Recent values of {ts.target} (lags and rolling averages)."
        if not self.signed:
            return "mixed", ""
        branch = self.names[positions[0]].split("__", 1)[0]
        if len(positions) == 1 and branch == "num":
            x, s = self.Xt[:, positions[0]], values[:, positions[0]]
            corr = _corr(x, s)
            if corr is not None and abs(corr) >= DIRECTION_MIN_CORR:
                word = "increases" if corr > 0 else "decreases"
                return word, f"Higher {origin} {word} the {self.output}."
        if branch == "cat":
            effects = {}
            for position in positions:
                present = self.Xt[:, position] > 0.5
                if present.any():
                    category = self.names[position].split("__", 1)[1][len(origin) + 1:]
                    effects[category] = float(values[present, position].mean())
            if effects:
                high = max(effects, key=effects.get)
                low = min(effects, key=effects.get)
                return "mixed", f"Highest for {origin} = '{high}', lowest for '{low}'."
        return "mixed", ""

    def _permutation_global(self) -> list[FeatureImpact]:
        target = self.model.target
        if target is None or target not in self.raw.columns:
            return []
        rows = self.raw.dropna(subset=[target])
        if rows.empty:
            return []
        y = rows[target].to_numpy()
        if self.model.classes is not None:
            index = {value: i for i, value in enumerate(self.model.classes)}
            y = np.asarray([index.get(value, -1) for value in y])
        X = rows[self.model.feature_columns]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(self.model.pipeline, X, y, n_repeats=5, random_state=self.seed)
        names = [self._permutation_name(column) for column in X.columns]
        importance = pd.Series(np.clip(result.importances_mean, 0, None), index=names).groupby(level=0, sort=False).sum()
        outputs = self._predict_output(X)
        details = {}
        for column, name in zip(X.columns, names, strict=True):
            corr = _corr(pd.to_numeric(X[column], errors="coerce").to_numpy(dtype="float64"), outputs) \
                if pd.api.types.is_numeric_dtype(X[column]) else None
            if corr is not None and abs(corr) >= DIRECTION_MIN_CORR:
                word = "increases" if corr > 0 else "decreases"
                details[name] = (word, f"Higher {name} {word} the {self.output}.")
            else:
                details.setdefault(name, ("mixed", ""))
        return _impacts(importance, details)

    def _permutation_name(self, column: str) -> str:
        if self.family == "time_series" and column == self.model.target:
            return f"{column} (history)"
        return column

    def _predict_output(self, raw: pd.DataFrame) -> np.ndarray:
        X = raw[self.model.feature_columns]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.binary:
                return self.model.pipeline.predict_proba(X)[:, self.model.pos_label]
            return np.asarray(self.model.pipeline.predict(X), dtype="float64")  # value, or encoded class


# --------------------------------------------------------------------------- module API


def explain_global(model: TrainedModel, X_sample: pd.DataFrame, task: TaskType | None = None, *,
                   seed: int = 42) -> GlobalExplanation:
    """Global drivers of `model` on (up to 500 rows of) `X_sample`, per original column."""
    if task is not None and task != model.task_type:
        raise ValueError(f"Model was trained for {model.task_type}, not {task}.")
    return ModelExplainer(model, X_sample, seed=seed).global_explanation()


def original_feature_names(model: TrainedModel) -> list[str]:
    """For each model input column (after preprocessing), the original column it came from."""
    preprocess = model.pipeline.named_steps["preprocess"]
    branches = {name: [str(c) for c in columns] for name, _, columns in preprocess.transformers_ if name != "remainder"}
    ts = model.pipeline.named_steps.get("ts_features")
    origins = []
    for name in preprocess.get_feature_names_out():
        branch, rest = str(name).split("__", 1)
        if branch == "extra" and ts is not None:
            is_history = rest.startswith(("ts_lag_", "ts_rolling_"))
            origins.append(f"{ts.target} (history)" if is_history else ts.time_column)
            continue
        candidates = [c for c in branches.get(branch, []) if rest == c or rest.startswith(f"{c}_")]
        origins.append(max(candidates, key=len) if candidates else rest)
    return origins


def cluster_profiles(model: TrainedModel, data: pd.DataFrame, profile: DatasetProfile) -> tuple[list[ClusterProfile], GlobalExplanation]:
    """Describe each cluster (size, typical values, what sets it apart) and which columns separate clusters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = np.asarray(model.predict(data[model.feature_columns]))
    usable = [c.name for c in profile.columns if c.name in model.feature_columns and c.n_unique > 1
              and c.semantic_type in ("numeric", "categorical", "boolean")]
    numeric = [c for c in usable if pd.api.types.is_numeric_dtype(data[c]) and profile.column(c).semantic_type == "numeric"]
    categorical = [c for c in usable if c not in numeric]
    means, stds = data[numeric].mean(), data[numeric].std().replace(0, np.nan)
    overall_shares = {c: data[c].astype(str).value_counts(normalize=True) for c in categorical}

    profiles: list[ClusterProfile] = []
    separation: dict[str, float] = dict.fromkeys(usable, 0.0)
    for cluster in sorted(np.unique(labels)):
        members = data[labels == cluster]
        share = len(members) / len(data)
        summary: dict[str, Any] = {}
        differences: list[tuple[float, str]] = []
        for column in numeric:
            mean = float(members[column].mean())
            summary[column] = round(mean, 3)
            z = (mean - means[column]) / stds[column] if pd.notna(stds[column]) else 0.0
            separation[column] += share * float(z) ** 2
            differences.append((abs(float(z)), f"{column}: {mean:.3g} vs {means[column]:.3g} overall"))
        for column in categorical:
            counts = members[column].astype(str).value_counts(normalize=True)
            if counts.empty:
                continue
            mode, mode_share = counts.index[0], float(counts.iloc[0])
            summary[column] = mode
            overall = overall_shares[column]
            categories = overall.index.union(counts.index)
            # total variation distance between the cluster's and the overall distribution of this column
            tvd = 0.5 * float((counts.reindex(categories, fill_value=0) - overall.reindex(categories, fill_value=0)).abs().sum())
            separation[column] += share * tvd
            lift = mode_share - float(overall_shares[column].get(mode, 0.0))
            differences.append((2 * lift, f"{column}: mostly '{mode}' ({mode_share:.0%} vs "
                                          f"{overall_shares[column].get(mode, 0.0):.0%} overall)"))
        differences.sort(reverse=True)
        profiles.append(ClusterProfile(cluster=int(cluster), size=len(members), share=round(share, 4), summary=summary,
                                       distinguishing=[text for score, text in differences[:3] if score > 0.1]))

    importance = pd.Series(separation, dtype="float64")
    explanation = GlobalExplanation(
        method="cluster_profile", output="cluster membership", features=_impacts(importance, {}), n_samples=len(data),
        notes=["Importance = how strongly a column differs between clusters (standardised mean gap / distribution gap)."],
    )
    return profiles, explanation


def forecast_future(model: TrainedModel, history: pd.DataFrame, horizon: int = FORECAST_HORIZON,
                    residual_std: float | None = None) -> Forecast:
    """Recursive forecast of the next `horizon` periods after the last observed row of `history`."""
    ts = model.pipeline.named_steps["ts_features"]
    ordered = history.sort_values(ts.time_column, key=parse_datetimes, kind="stable").reset_index(drop=True)
    times = parse_datetimes(ordered[ts.time_column])
    step = times.diff().dropna().median()
    period = max(1, infer_seasonal_period(ordered[ts.time_column]))
    context = max([*ts.lags, *ts.windows, period]) + 1
    frame = ordered[model.feature_columns].tail(context + period).reset_index(drop=True)
    is_text_time = not pd.api.types.is_datetime64_any_dtype(ordered[ts.time_column])
    last_time = times.iloc[-1]

    points: list[ForecastPoint] = []
    sigma = float(residual_std) if residual_std and np.isfinite(residual_std) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(1, horizon + 1):
            when = last_time + k * step
            future = frame.iloc[[len(frame) - period]].copy()  # other columns repeat their value from one season ago
            future[ts.time_column] = when.strftime("%Y-%m-%d") if is_text_time and step >= pd.Timedelta(days=1) else (
                when.isoformat() if is_text_time else when)
            future[ts.target] = np.nan
            frame = pd.concat([frame, future], ignore_index=True)
            prediction = float(model.pipeline.predict(frame.tail(context + 1))[-1])
            frame.loc[frame.index[-1], ts.target] = prediction
            points.append(ForecastPoint(timestamp=str(when.date() if step >= pd.Timedelta(days=1) else when),
                                        prediction=round(prediction, 4), lower=round(prediction - Z_95 * sigma, 4),
                                        upper=round(prediction + Z_95 * sigma, 4)))
    exogenous = [c for c in model.feature_columns if c not in (ts.time_column, ts.target)]
    assumptions = ["Forecast is recursive: each predicted value becomes history for the next step."]
    if exogenous:
        assumptions.append(f"Future values of {', '.join(exogenous)} are assumed to repeat their values from one season "
                           f"({period} periods) earlier.")
    interval = (f"±{Z_95} × validation RMSE ({sigma:.4g}), from one-step errors; uncertainty grows further ahead."
                if sigma else "No validation error available; no interval.")
    return Forecast(horizon=horizon, points=points, interval=interval, assumptions=assumptions)


def explain_run(model: TrainedModel, data: pd.DataFrame, training: TrainingResult, profile: DatasetProfile,
                run_dir: Path, *, background: pd.DataFrame | None = None, horizon: int = FORECAST_HORIZON,
                seed: int = 42) -> tuple[ExplanationResult, ModelExplainer | None]:
    """Everything the decision layer needs to explain a trained model, plus charts in run_dir/charts/.

    `data` is the full dataset (cluster profiles, forecast history); `background` (default: `data`) is
    the sample the drivers are computed on — normally the training rows.
    """
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    family = model_zoo.task_family(model.task_type)
    charts: dict[str, str] = {}

    if family == "clustering":
        clusters, global_explanation = cluster_profiles(model, data, profile)
        _save_chart(charts, "feature_importance", plot_feature_importance(global_explanation, charts_dir / "feature_importance.png"))
        _save_chart(charts, "cluster_sizes", plot_cluster_sizes(clusters, charts_dir / "cluster_sizes.png"))
        return ExplanationResult(global_explanation=global_explanation, clusters=clusters, charts=charts), None

    explainer = ModelExplainer(model, background if background is not None else data, seed=seed)
    global_explanation = explainer.global_explanation()
    _save_chart(charts, "feature_importance", plot_feature_importance(global_explanation, charts_dir / "feature_importance.png"))
    if explainer.method in ("shap_tree", "shap_linear"):
        _save_chart(charts, "shap_summary", plot_shap_summary(explainer, charts_dir / "shap_summary.png"))

    local_examples: list[LocalExplanation] = []
    forecast = None
    if explainer.method in ("shap_tree", "shap_linear"):
        outputs = explainer.predict_output(explainer.raw)
        position = len(outputs) - 1 if family == "time_series" else int(np.argmax(outputs))
        local_examples.append(explainer.local(position))

    if family == "classification":
        _save_chart(charts, "confusion_matrix", plot_confusion_matrix(training.test_metrics, charts_dir / "confusion_matrix.png"))
    elif family == "regression":
        _save_chart(charts, "actual_vs_pred", plot_actual_vs_pred(training.predictions_path, charts_dir / "actual_vs_pred.png"))
    elif family == "time_series":
        selected = next((e for e in training.leaderboard if e.selected), None)
        sigma = (selected.valid_metrics.get("rmse") if selected else None) or training.test_metrics.get("rmse")
        forecast = forecast_future(model, data, horizon, sigma)
        _save_chart(charts, "forecast", plot_forecast(model, data, training.predictions_path, forecast, charts_dir / "forecast.png"))

    return ExplanationResult(global_explanation=global_explanation, local_examples=local_examples, forecast=forecast,
                             charts=charts), explainer


# --------------------------------------------------------------------------- charts


def plot_feature_importance(explanation: GlobalExplanation, path: Path) -> Path | None:
    features = explanation.features[:TOP_FEATURES]
    if not features:
        return None
    colors = {"increases": "#c0392b", "decreases": "#2e86c1", "mixed": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(features) + 1.2))
    ax.barh([f.feature for f in reversed(features)], [f.share * 100 for f in reversed(features)],
            color=[colors[f.direction] for f in reversed(features)])
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title(f"Key drivers of the {explanation.output}", fontsize=10)
    return _finish(fig, path)


def plot_shap_summary(explainer: ModelExplainer, path: Path) -> Path | None:
    if not explainer.signed:
        return None
    try:
        values = explainer._shap_values()
        names = [_short_name(name) for name in explainer.names]
        plt.figure()
        shap.summary_plot(values, explainer.Xt, feature_names=names, max_display=TOP_FEATURES, show=False, plot_size=(8, 5))
        plt.title(f"SHAP: effect on the {explainer.output}", fontsize=10)
        return _finish(plt.gcf(), path)
    except Exception as exc:  # charts are optional
        logger.warning("SHAP summary chart failed ({}: {}).", type(exc).__name__, exc)
        plt.close("all")
        return None


def plot_confusion_matrix(test_metrics: dict[str, Any], path: Path) -> Path | None:
    matrix, labels = test_metrics.get("confusion_matrix"), test_metrics.get("class_labels")
    if not matrix or not labels:
        return None
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test split)", fontsize=10)
    return _finish(fig, path)


def plot_actual_vs_pred(predictions_path: str | None, path: Path) -> Path | None:
    if not predictions_path or not Path(predictions_path).is_file():
        return None
    predictions = pd.read_csv(predictions_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(predictions["actual"], predictions["predicted"], s=12, alpha=0.6)
    low = float(min(predictions["actual"].min(), predictions["predicted"].min()))
    high = float(max(predictions["actual"].max(), predictions["predicted"].max()))
    ax.plot([low, high], [low, high], color="#c0392b", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs predicted (test split)", fontsize=10)
    return _finish(fig, path)


def plot_forecast(model: TrainedModel, data: pd.DataFrame, predictions_path: str | None, forecast: Forecast,
                  path: Path) -> Path | None:
    ts = model.pipeline.named_steps["ts_features"]
    ordered = data.sort_values(ts.time_column, key=parse_datetimes, kind="stable").tail(120)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(parse_datetimes(ordered[ts.time_column]), ordered[ts.target], label="Actual", color="#34495e", linewidth=1)
    if predictions_path and Path(predictions_path).is_file():
        test = pd.read_csv(predictions_path)
        if "time" in test:
            ax.plot(parse_datetimes(test["time"]), test["predicted"], label="Model (test split)", color="#2e86c1", linewidth=1)
    future = pd.to_datetime([p.timestamp for p in forecast.points])
    ax.plot(future, [p.prediction for p in forecast.points], label="Forecast", color="#c0392b", linewidth=1.5)
    ax.fill_between(future, [p.lower for p in forecast.points], [p.upper for p in forecast.points],
                    color="#c0392b", alpha=0.15, label="Approx. 95% interval")
    ax.set_title(f"{ts.target}: recent history and {forecast.horizon}-period forecast", fontsize=10)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_cluster_sizes(profiles: list[ClusterProfile], path: Path) -> Path | None:
    if not profiles:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([f"Cluster {p.cluster}" for p in profiles], [p.size for p in profiles], color="#2e86c1")
    ax.set_ylabel("Records")
    ax.set_title("Cluster sizes", fontsize=10)
    return _finish(fig, path)


# --------------------------------------------------------------------------- helpers


def _group_columns(values: np.ndarray, origins: list[str]) -> pd.DataFrame:
    """Sum model-input columns that belong to the same original column."""
    frame = pd.DataFrame(values)
    return frame.T.groupby(np.asarray(origins), sort=False).sum().T


def _impacts(importance: pd.Series, details: dict[str, tuple[str, str]]) -> list[FeatureImpact]:
    importance = importance.clip(lower=0).sort_values(ascending=False)
    total = float(importance.sum())
    return [
        FeatureImpact(feature=str(name), importance=round(float(value), 6),
                      share=round(float(value) / total, 4) if total > 0 else 0.0,
                      direction=details.get(name, ("mixed", ""))[0], detail=details.get(name, ("mixed", ""))[1])
        for name, value in importance.items()
    ]


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _short_name(name: str) -> str:
    for prefix in _BRANCH_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_chart(charts: dict[str, str], name: str, path: Path | None) -> None:
    if path is not None:
        charts[name] = str(path)

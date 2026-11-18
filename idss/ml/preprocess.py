"""Cleaning, train/valid/test splitting, and the sklearn preprocessing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from idss.core.logger import get_logger
from idss.core.schemas import DatasetProfile, TaskType
from idss.ml.profiler import CLASSIFICATION_TASKS, parse_datetimes
from idss.storage.runs import run_dir

logger = get_logger()

TRAIN_SHARE, VALID_SHARE, TEST_SHARE = 0.70, 0.20, 0.10
MAX_ONEHOT_CATEGORIES = 20
MIN_ROWS_TO_SPLIT = 10
PREPROCESSOR_FILENAME = "preprocessor.joblib"

Split = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


@dataclass(frozen=True)
class ColumnPlan:
    """Which preprocessing branch each feature column goes to, and why dropped columns are dropped."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    def describe(self) -> list[str]:
        """Human-readable preprocessing steps."""
        steps = []
        if self.numeric:
            steps.append(f"Numeric columns ({len(self.numeric)}): fill missing with median, then standardise.")
        if self.categorical:
            steps.append(
                f"Categorical columns ({len(self.categorical)}): fill missing with most frequent value, "
                f"then one-hot encode (top {MAX_ONEHOT_CATEGORIES} categories per column)."
            )
        if self.datetime:
            steps.append(f"Date columns ({len(self.datetime)}): extract year, month, day and day of week.")
        steps += [f"Dropped column '{name}': {reason}." for name, reason in self.dropped.items()]
        return steps


class DateFeatureExtractor(TransformerMixin, BaseEstimator):
    """Turn datetime-like columns into numeric year, month, day and day-of-week features."""

    PARTS = ("year", "month", "day", "dayofweek")

    def fit(self, X: pd.DataFrame, y: object = None) -> "DateFeatureExtractor":
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray([str(col) for col in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        features = {}
        for col in X.columns:
            parsed = parse_datetimes(X[col])
            for part in self.PARTS:
                features[f"{col}_{part}"] = getattr(parsed.dt, part).to_numpy(dtype="float64", na_value=np.nan)
        return pd.DataFrame(features, index=X.index).to_numpy(dtype="float64")

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray([f"{name}_{part}" for name in names for part in self.PARTS], dtype=object)


def _as_object(X: pd.DataFrame) -> pd.DataFrame:
    """Cast to plain object columns with np.nan for missing, which SimpleImputer and OneHotEncoder expect."""
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)


def plan_columns(profile: DatasetProfile, target: str | None) -> ColumnPlan:
    """Assign every non-target column to a preprocessing branch or drop it."""
    plan = ColumnPlan()
    for col in profile.columns:
        if col.name == target:
            continue
        if col.n_unique <= 1:
            plan.dropped[col.name] = "constant"
        elif col.semantic_type == "id":
            plan.dropped[col.name] = "identifier, not predictive"
        elif col.semantic_type == "text":
            plan.dropped[col.name] = "free text is not supported yet"
        elif col.semantic_type == "numeric":
            plan.numeric.append(col.name)
        elif col.semantic_type == "datetime":
            plan.datetime.append(col.name)
        else:  # categorical, boolean
            plan.categorical.append(col.name)
    return plan


def build_preprocessor(
    profile: DatasetProfile, target: str | None, *, extra_numeric: list[str] | tuple[str, ...] = ()
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the feature columns described by `profile`.

    `extra_numeric` names generated columns (e.g. time-series lags) that are median-imputed but not scaled.
    """
    plan = plan_columns(profile, target)
    for name, reason in plan.dropped.items():
        logger.info("Dropping column '{}': {}.", name, reason)

    transformers = []
    if plan.numeric:
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("num", numeric, plan.numeric))
    if plan.categorical:
        categorical = Pipeline([
            ("as_object", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=MAX_ONEHOT_CATEGORIES, sparse_output=False)),
        ])
        transformers.append(("cat", categorical, plan.categorical))
    if plan.datetime:
        dates = Pipeline([("extract", DateFeatureExtractor()), ("impute", SimpleImputer(strategy="median"))])
        transformers.append(("date", dates, plan.datetime))
    if extra_numeric:
        transformers.append(("extra", SimpleImputer(strategy="median"), list(extra_numeric)))
    if not transformers:
        raise ValueError("No usable feature columns: every column is the target, constant, an identifier or free text.")
    return ColumnTransformer(transformers, remainder="drop")


def clean_dataset(
    df: pd.DataFrame, target: str | None, task_type: TaskType, profile: DatasetProfile
) -> tuple[pd.DataFrame, list[str]]:
    """Remove duplicate rows and rows without a target; sort time series chronologically.

    Returns the cleaned frame and a list of human-readable steps that were applied.
    """
    steps: list[str] = []
    n_before = len(df)
    df = df.drop_duplicates()
    if removed := n_before - len(df):
        steps.append(f"Removed {removed:,} duplicate rows.")

    if target is not None:
        missing = df[target].isna()
        if missing.any():
            df = df.loc[~missing]
            steps.append(f"Removed {int(missing.sum()):,} rows with missing target '{target}'.")

    if task_type == "time_series_forecasting":
        time_columns = [name for name in profile.columns_of("datetime") if name != target]
        if time_columns:
            df = df.sort_values(time_columns[0], key=parse_datetimes, kind="stable")
            steps.append(f"Sorted rows chronologically by '{time_columns[0]}'.")
    return df, steps


def split_data(df: pd.DataFrame, target: str | None, task_type: TaskType, seed: int = 42) -> Split:
    """Split into train/valid/test = 70/20/10.

    Classification is stratified on the target (falling back to a random split if a class is too rare).
    Time series are split chronologically without shuffling, so `df` must already be in time order.
    """
    if len(df) < MIN_ROWS_TO_SPLIT:
        raise ValueError(f"Need at least {MIN_ROWS_TO_SPLIT} rows to split, got {len(df)}.")
    if target is not None and df[target].isna().any():
        raise ValueError(f"Target '{target}' has missing values; run clean_dataset first.")

    # Integer sizes: fractional test_size values suffer float rounding (0.2 / 0.9 * 900 -> 201 rows).
    n_test = max(1, round(len(df) * TEST_SHARE))
    n_valid = max(1, round(len(df) * VALID_SHARE))
    if task_type == "time_series_forecasting":
        return df.iloc[: -(n_valid + n_test)].copy(), df.iloc[-(n_valid + n_test) : -n_test].copy(), df.iloc[-n_test:].copy()

    stratify = target if target is not None and task_type in CLASSIFICATION_TASKS else None
    try:
        return _random_split(df, n_valid, n_test, stratify, seed)
    except ValueError as exc:
        if stratify is None:
            raise
        logger.warning("Stratified split not possible ({}); using a random split instead.", exc)
        return _random_split(df, n_valid, n_test, None, seed)


def _random_split(df: pd.DataFrame, n_valid: int, n_test: int, stratify: str | None, seed: int) -> Split:
    rest, test = train_test_split(df, test_size=n_test, random_state=seed, stratify=df[stratify] if stratify else None)
    train, valid = train_test_split(
        rest, test_size=n_valid, random_state=seed, stratify=rest[stratify] if stratify else None
    )
    return train.copy(), valid.copy(), test.copy()


def save_preprocessor(preprocessor: ColumnTransformer, run_id: str, runs_dir: Path | None = None) -> Path:
    """Save a fitted preprocessor to <runs_dir>/<run_id>/preprocessor.joblib and return the path."""
    path = run_dir(run_id, runs_dir) / PREPROCESSOR_FILENAME
    joblib.dump(preprocessor, path)
    logger.info("Saved preprocessor to {}.", path)
    return path


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load a preprocessor saved by save_preprocessor. Only load files this application created."""
    return joblib.load(path)

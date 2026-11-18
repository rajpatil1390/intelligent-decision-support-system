"""Dataset profiling: semantic column types, data-quality warnings, and target / task-type suggestions."""

import math
import re
import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from idss.core.logger import get_logger
from idss.core.schemas import ColumnProfile, DatasetProfile, SemanticType, TaskType

logger = get_logger()

TARGET_KEYWORDS = ("target", "label", "class", "outcome", "churn", "price", "sales", "y")
CLASSIFICATION_TASKS: frozenset[str] = frozenset({"tabular_classification", "text_classification"})
MAX_INT_CLASSES = 20  # an integer target with at most this many values is treated as classification
HIGH_MISSING_PCT = 40.0
MINORITY_CLASS_SHARE = 0.10
HIGH_CARDINALITY = 50
LEAKAGE_CORRELATION = 0.98
ID_UNIQUE_RATIO = 0.95
TEXT_MIN_AVG_LENGTH = 50
TEXT_MIN_UNIQUE_RATIO = 0.5

_MIN_ROWS_FOR_STRING_ID = 20
_DATETIME_PARSE_RATIO = 0.9
_N_SAMPLE_VALUES = 5
_MAX_SAMPLE_CHARS = 60
_BOOLEAN_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})
_ID_NAME_TOKENS = frozenset({"id", "uuid", "guid", "key", "no", "number"})
_NAME_TOKEN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_PLAIN_NUMBER = r"\s*[-+]?\d+(?:\.\d+)?\s*"


def profile_dataset(df: pd.DataFrame, target: str | None = None, *, suggest_target: bool = True) -> DatasetProfile:
    """Profile `df` and suggest a target column and task type.

    Args:
        df: the raw dataset.
        target: the target column, if known (matched case-insensitively). Raises ValueError if absent.
        suggest_target: when no target is given, pick one heuristically. If False (or no column is
            eligible), the suggested task is clustering.
    """
    if df.empty:
        raise ValueError("Cannot profile an empty DataFrame.")
    if target is not None:
        target = resolve_column(df, target)

    columns = [_profile_column(df[name]) for name in df.columns]
    by_name = {col.name: col for col in columns}
    time_column = _time_index_column(df, columns)
    if target is None and suggest_target:
        target = _suggest_target(columns)
    task_type = _infer_task_type(columns, target, time_column) if target else "tabular_clustering"

    issues = _column_warnings(columns)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues.append(f"{duplicate_rows:,} duplicate rows found; they will be removed.")
    if target:
        issues += _target_warnings(df, by_name[target], task_type)
        issues += _leakage_warnings(df, target, columns)

    profile = DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=columns,
        duplicate_rows=duplicate_rows,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        suggested_target=target,
        suggested_task_type=task_type,
        time_column=time_column,
        warnings=issues,
    )
    logger.info(
        "Profiled {:,} rows x {} cols: target={}, task={}, {} warning(s).",
        profile.n_rows, profile.n_cols, target, task_type, len(issues),
    )
    return profile


def task_type_for_target(profile: DatasetProfile, target: str) -> TaskType | None:
    """Task type the data implies for predicting `target`; None if that column cannot be a target."""
    return _infer_task_type(profile.columns, target, profile.time_column)


def resolve_column(df: pd.DataFrame, name: str) -> str:
    """Return the column of `df` matching `name` exactly, or else case/whitespace-insensitively."""
    if name in df.columns:
        return name
    wanted = name.strip().lower()
    matches = [col for col in df.columns if str(col).strip().lower() == wanted]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Column '{name}' not found. Available columns: {', '.join(map(str, df.columns))}.")


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse `series` to datetime64; values that cannot be parsed become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
        if parsed.notna().sum() < series.notna().sum():
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed


# --------------------------------------------------------------------------- column profiling


def _profile_column(series: pd.Series) -> ColumnProfile:
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    semantic = _infer_semantic_type(str(series.name), non_null, n_unique)
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        semantic_type=semantic,
        missing_pct=round(100 * float(series.isna().mean()), 2),
        n_unique=n_unique,
        sample_values=[_to_builtin(value) for value in non_null.drop_duplicates().head(_N_SAMPLE_VALUES)],
        stats=_column_stats(non_null, semantic),
    )


def _infer_semantic_type(name: str, non_null: pd.Series, n_unique: int) -> SemanticType:
    if non_null.empty:
        return "categorical"  # all missing; reported as constant and dropped
    if _is_boolean(non_null, n_unique):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return "datetime"

    unique_ratio = n_unique / len(non_null)
    if pd.api.types.is_numeric_dtype(non_null):
        # Unique integers are an ID only with an ID-like name or a 1,2,3,... sequence; otherwise they may be real values.
        if _is_integral(non_null) and unique_ratio >= ID_UNIQUE_RATIO and (_has_id_name(name) or _is_sequence(non_null)):
            return "id"
        return "numeric"

    text = non_null.astype(str)
    if _looks_datetime(text):
        return "datetime"
    if text.str.len().mean() >= TEXT_MIN_AVG_LENGTH and unique_ratio >= TEXT_MIN_UNIQUE_RATIO:
        return "text"
    if unique_ratio >= ID_UNIQUE_RATIO and (len(non_null) >= _MIN_ROWS_FOR_STRING_ID or _has_id_name(name)):
        return "id"
    return "categorical"


def _is_boolean(non_null: pd.Series, n_unique: int) -> bool:
    if pd.api.types.is_bool_dtype(non_null):
        return True
    if n_unique != 2:
        return False
    if pd.api.types.is_numeric_dtype(non_null):
        return set(non_null.unique().tolist()) <= {0, 1}
    return {str(value).strip().lower() for value in non_null.unique()} <= _BOOLEAN_TOKENS


def _is_integral(non_null: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(non_null):
        return True
    if not pd.api.types.is_float_dtype(non_null):
        return False
    values = non_null.to_numpy(dtype="float64")
    return bool(np.all(np.mod(values, 1) == 0))


def _is_sequence(non_null: pd.Series) -> bool:
    values = np.sort(non_null.to_numpy(dtype="float64"))
    return len(values) >= 2 and bool(np.all(np.diff(values) == 1))


def _has_id_name(name: str) -> bool:
    return any(token.lower() in _ID_NAME_TOKENS for token in _NAME_TOKEN.findall(name))


def _looks_datetime(text: pd.Series) -> bool:
    sample = text.drop_duplicates().head(200)
    if sample.str.fullmatch(_PLAIN_NUMBER).mean() > 0.5:
        return False  # plain numbers (years, codes) are not dates
    if sample.str.contains(r"\d").mean() < _DATETIME_PARSE_RATIO:
        return False  # month names alone ("Jan") are categorical
    return bool(parse_datetimes(sample).notna().mean() >= _DATETIME_PARSE_RATIO)


def _column_stats(non_null: pd.Series, semantic: SemanticType) -> dict[str, Any]:
    if non_null.empty:
        return {}
    if semantic == "numeric":
        values = pd.Series(non_null.to_numpy(dtype="float64"))
        return {
            "mean": _num(values.mean()),
            "std": _num(values.std()),
            "min": _num(values.min()),
            "median": _num(values.median()),
            "max": _num(values.max()),
            "integer": _is_integral(non_null),
        }
    if semantic in ("categorical", "boolean"):
        counts = non_null.astype(str).value_counts().head(5)
        return {"top_values": {str(value): int(count) for value, count in counts.items()}}
    if semantic == "datetime":
        parsed = parse_datetimes(non_null).dropna()
        return {"min": parsed.min().isoformat(), "max": parsed.max().isoformat()} if not parsed.empty else {}
    if semantic == "text":
        lengths = non_null.astype(str).str.len()
        return {"avg_length": _num(lengths.mean()), "max_length": int(lengths.max())}
    return {}


# --------------------------------------------------------------------------- suggestions


def _suggest_target(columns: list[ColumnProfile]) -> str | None:
    eligible = [col for col in columns if col.semantic_type in ("numeric", "categorical", "boolean") and col.n_unique > 1]
    if not eligible:
        return None
    by_lower_name = {col.name.strip().lower(): col.name for col in eligible}
    for keyword in TARGET_KEYWORDS:
        if keyword in by_lower_name:
            return by_lower_name[keyword]
    for keyword in TARGET_KEYWORDS:
        if len(keyword) < 4:
            continue  # short keywords like "y" only match exactly
        for col in eligible:
            if keyword in col.name.lower():
                return col.name
    return eligible[-1].name


def _infer_task_type(columns: list[ColumnProfile], target: str, time_column: str | None) -> TaskType | None:
    col = next((c for c in columns if c.name == target), None)
    if col is None:
        return None
    if col.semantic_type in ("categorical", "boolean"):
        is_classification = True
    elif col.semantic_type == "numeric":
        is_classification = bool(col.stats.get("integer")) and col.n_unique <= MAX_INT_CLASSES
    else:
        return None  # id / text / datetime targets cannot be predicted directly

    if is_classification:
        features = [c for c in columns if c.name != target and c.semantic_type != "id" and c.n_unique > 1]
        if features and all(c.semantic_type == "text" for c in features):
            return "text_classification"
        return "tabular_classification"
    if time_column is not None and time_column != target:
        return "time_series_forecasting"
    return "tabular_regression"


def _time_index_column(df: pd.DataFrame, columns: list[ColumnProfile]) -> str | None:
    """A complete, unique, increasing datetime column means the rows form a time series."""
    for col in columns:
        if col.semantic_type != "datetime":
            continue
        if col.missing_pct > 0 or col.n_unique != len(df):
            continue
        if parse_datetimes(df[col.name]).is_monotonic_increasing:
            return col.name
    return None


# --------------------------------------------------------------------------- warnings


def _column_warnings(columns: list[ColumnProfile]) -> list[str]:
    issues: list[str] = []
    for col in columns:
        if col.n_unique <= 1:
            detail = " (all values missing)" if col.missing_pct == 100 else ""
            issues.append(f"Column '{col.name}' is constant{detail}; it will be dropped.")
        elif col.missing_pct > HIGH_MISSING_PCT:
            issues.append(f"Column '{col.name}' is {col.missing_pct:.0f}% missing.")
        if col.semantic_type == "categorical" and col.n_unique > HIGH_CARDINALITY:
            issues.append(
                f"Column '{col.name}' has {col.n_unique} categories (high cardinality); rare categories will be grouped."
            )
    return issues


def _target_warnings(df: pd.DataFrame, target: ColumnProfile, task_type: TaskType | None) -> list[str]:
    issues: list[str] = []
    if target.missing_pct > 0:
        issues.append(f"Target '{target.name}' is {target.missing_pct:g}% missing; those rows will be dropped.")
    if task_type is None:
        issues.append(f"Target '{target.name}' looks like a {target.semantic_type} column and cannot be predicted; choose another target.")
    elif task_type in CLASSIFICATION_TASKS:
        shares = df[target.name].value_counts(normalize=True)
        if len(shares) >= 2 and shares.iloc[-1] < MINORITY_CLASS_SHARE:
            issues.append(
                f"Target '{target.name}' is imbalanced: minority class '{shares.index[-1]}' is only {shares.iloc[-1]:.1%} of rows."
            )
    return issues


def _leakage_warnings(df: pd.DataFrame, target: str, columns: list[ColumnProfile]) -> list[str]:
    y = _numeric_view(df[target])
    if y is None:
        return []
    issues: list[str] = []
    for col in columns:
        if col.name == target or col.semantic_type not in ("numeric", "boolean"):
            continue
        x = _numeric_view(df[col.name])
        if x is None:
            continue
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            corr = x.corr(y)
        if pd.notna(corr) and abs(corr) >= LEAKAGE_CORRELATION:
            issues.append(
                f"Column '{col.name}' is almost perfectly correlated with the target (r={corr:.3f}); possible data leakage."
            )
    return issues


def _numeric_view(series: pd.Series) -> pd.Series | None:
    """Numeric version of `series` for correlation: numbers as-is, two-valued columns as 0/1, else None."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(series.to_numpy(dtype="float64", na_value=np.nan), index=series.index)
    if series.nunique() == 2:
        codes, _ = pd.factorize(series)
        return pd.Series(codes, index=series.index, dtype="float64").where(codes >= 0)
    return None


# --------------------------------------------------------------------------- helpers


def _num(value: Any) -> float | None:
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str):
        return value if len(value) <= _MAX_SAMPLE_CHARS else value[: _MAX_SAMPLE_CHARS - 3] + "..."
    if isinstance(value, bool | int | float) or value is None:
        return value
    return str(value)

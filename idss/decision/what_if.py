"""What-if analysis: how the model's output changes when some inputs change.

Changes are validated against the dataset profile (known column, correct type, observed range or
categories) and segments are selected with structured filters {column, op, value} only — no user text is
ever evaluated as code. A change is a value (set), or {"set": v}, {"add": x} or {"pct": p} (+p percent).
"""

import operator
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from idss.core.schemas import DatasetProfile, FilterCondition, SegmentImpact, WhatIfResult
from idss.ml import model_zoo
from idss.ml.profiler import parse_datetimes
from idss.ml.trainer import TrainedModel

_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}
_CHANGE_KINDS = ("set", "add", "pct")


class WhatIfError(ValueError):
    """Invalid what-if request (unknown column, wrong type, bad filter)."""


def output_label(model: TrainedModel) -> str:
    target = model.target or "cluster"
    if model.classes is not None and len(model.classes) == 2 and model.pos_label is not None:
        return f"probability that {target} = '{model.classes[model.pos_label]}'"
    if model.classes is not None or model_zoo.task_family(model.task_type) == "clustering":
        return f"predicted {target}"
    return f"predicted {target}"


def model_output(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    """Positive-class probability (binary), predicted value (regression), else predicted label."""
    X = frame[model.feature_columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model.classes is not None and len(model.classes) == 2 and model.pos_label is not None:
            return model.pipeline.predict_proba(X)[:, model.pos_label]
        return np.asarray(model.predict(X))


def simulate(model: TrainedModel, base_row: dict[str, Any], changes: dict[str, Any], profile: DatasetProfile, *,
             history: pd.DataFrame | None = None) -> WhatIfResult:
    """Model output for `base_row` before and after `changes`.

    Time-series models need `history` (earlier rows) so lag features can be computed for the row.
    """
    warnings_: list[str] = []
    _validate_changes(model, profile, changes)
    missing = [c for c in model.feature_columns if c not in base_row and c != model.target]
    if missing:
        warnings_.append(f"Missing inputs filled by the model's defaults: {', '.join(missing)}.")
    base = pd.DataFrame([base_row]).reindex(columns=model.feature_columns)
    changed, range_warnings = apply_changes(base, changes, model, profile)
    warnings_ += range_warnings

    if model_zoo.task_family(model.task_type) == "time_series":
        if history is None:
            raise WhatIfError("Time-series what-if needs the history of earlier rows.")
        context = history[model.feature_columns].tail(_context_rows(model))
        before = model_output(model, pd.concat([context, base], ignore_index=True))[-1]
        after = model_output(model, pd.concat([context, changed], ignore_index=True))[-1]
    else:
        before, after = model_output(model, base)[0], model_output(model, changed)[0]
    return WhatIfResult(output=output_label(model), before=_py(before), after=_py(after), changes=changes,
                        warnings=warnings_, **_deltas(before, after))


def simulate_segment(model: TrainedModel, df: pd.DataFrame, filters: list[FilterCondition | dict[str, Any]],
                     changes: dict[str, Any], profile: DatasetProfile) -> SegmentImpact:
    """Average model output over the rows matching `filters`, before and after applying `changes` to them."""
    conditions = parse_filters(filters)
    _validate_changes(model, profile, changes)
    mask = filter_mask(df, conditions)
    count = int(mask.sum())
    if count == 0:
        return SegmentImpact(output=output_label(model), filters=conditions, changes=changes, count=0,
                             warnings=["No rows match the filters."])

    if model_zoo.task_family(model.task_type) == "time_series":
        # Predict on the whole series so lags come from real history; average over the selected rows.
        ordered_index = df.sort_values(model.pipeline.named_steps["ts_features"].time_column,
                                       key=parse_datetimes, kind="stable").index
        frame = df.loc[ordered_index]
        changed_rows, range_warnings = apply_changes(frame[mask.loc[ordered_index]], changes, model, profile)
        changed = frame.copy()
        changed.loc[changed_rows.index, changed_rows.columns] = changed_rows
        selected = mask.loc[ordered_index].to_numpy()
        before_values = model_output(model, frame)[selected]
        after_values = model_output(model, changed)[selected]
    else:
        rows = df[mask]
        changed_rows, range_warnings = apply_changes(rows[model.feature_columns], changes, model, profile)
        before_values, after_values = model_output(model, rows), model_output(model, changed_rows)

    if not np.issubdtype(np.asarray(before_values).dtype, np.number):
        flipped = int((np.asarray(before_values) != np.asarray(after_values)).sum())
        return SegmentImpact(output=output_label(model), filters=conditions, changes=changes, count=count,
                             warnings=[*range_warnings, f"Predicted label changes for {flipped} of {count} rows."])
    avg_before, avg_after = float(np.mean(before_values)), float(np.mean(after_values))
    return SegmentImpact(output=output_label(model), filters=conditions, changes=changes, count=count,
                         avg_before=round(avg_before, 6), avg_after=round(avg_after, 6), warnings=range_warnings,
                         **_deltas(avg_before, avg_after))


def parse_filters(filters: list[FilterCondition | dict[str, Any]]) -> list[FilterCondition]:
    try:
        return [f if isinstance(f, FilterCondition) else FilterCondition.model_validate(f) for f in filters]
    except ValidationError as exc:
        allowed = ", ".join([*_COMPARISONS, "in"])
        raise WhatIfError(f"Invalid filter: {exc.errors()[0]['msg']}. Use {{column, op, value}} with op in: {allowed}.") from exc


def filter_mask(df: pd.DataFrame, conditions: list[FilterCondition]) -> pd.Series:
    """Rows matching every condition (structured comparison only)."""
    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        if condition.column not in df.columns:
            raise WhatIfError(f"Unknown filter column '{condition.column}'.")
        column = df[condition.column]
        if condition.op == "in":
            if not isinstance(condition.value, (list, tuple, set)):
                raise WhatIfError(f"Filter 'in' on '{condition.column}' needs a list of values.")
            mask &= column.isin(list(condition.value))
        elif condition.op in ("<", "<=", ">", ">="):
            if not pd.api.types.is_numeric_dtype(column) or not _is_number(condition.value):
                raise WhatIfError(f"Filter '{condition.op}' needs a numeric column and value ('{condition.column}').")
            mask &= _COMPARISONS[condition.op](column, condition.value).fillna(False)
        else:
            mask &= _COMPARISONS[condition.op](column, condition.value).fillna(condition.op == "!=")
    return mask


def apply_changes(frame: pd.DataFrame, changes: dict[str, Any], model: TrainedModel,
                  profile: DatasetProfile) -> tuple[pd.DataFrame, list[str]]:
    """Return a changed copy of `frame` and warnings for values outside the observed range/categories."""
    changed = frame.copy()
    notes: list[str] = []
    for column, change in changes.items():
        kind, amount = _change_kind(change)
        if kind == "set":
            changed[column] = amount
        elif kind == "add":
            changed[column] = pd.to_numeric(changed[column], errors="coerce") + amount
        else:
            changed[column] = pd.to_numeric(changed[column], errors="coerce") * (1 + amount / 100)
        notes += _range_warnings(column, changed[column], model, profile)
    return changed, notes


# --------------------------------------------------------------------------- validation


def _validate_changes(model: TrainedModel, profile: DatasetProfile, changes: dict[str, Any]) -> None:
    if not changes:
        raise WhatIfError("No changes given.")
    columns = {c.name: c for c in profile.columns}
    for column, change in changes.items():
        if column not in model.feature_columns or column == model.target or column not in columns:
            raise WhatIfError(f"Unknown or non-input column '{column}'.")
        kind, amount = _change_kind(change)
        semantic = columns[column].semantic_type
        if semantic in ("id", "text"):
            raise WhatIfError(f"Column '{column}' is {semantic} and is not used by the model.")
        if semantic == "numeric":
            if not _is_number(amount):
                raise WhatIfError(f"Column '{column}' is numeric; got {amount!r}.")
        elif kind != "set":
            raise WhatIfError(f"Column '{column}' is {semantic}; only a new value can be set, not '{kind}'.")


def _change_kind(change: Any) -> tuple[str, Any]:
    if isinstance(change, dict):
        if len(change) != 1 or next(iter(change)) not in _CHANGE_KINDS:
            raise WhatIfError(f"A change must be a value or one of {{'set'|'add'|'pct': value}}; got {change!r}.")
        kind, amount = next(iter(change.items()))
        if kind in ("add", "pct") and not _is_number(amount):
            raise WhatIfError(f"'{kind}' needs a number; got {amount!r}.")
        return kind, amount
    return "set", change


def _range_warnings(column: str, values: pd.Series, model: TrainedModel, profile: DatasetProfile) -> list[str]:
    stats = profile.column(column).stats
    if "min" in stats and "max" in stats and pd.api.types.is_numeric_dtype(values):
        low, high = stats["min"], stats["max"]
        outside = values[(values < low) | (values > high)]
        if len(outside):
            return [f"{column}: {len(outside)} value(s) outside the observed range [{low:g}, {high:g}]; "
                    "the model is extrapolating."]
        return []
    known = _known_categories(model, column)
    if known is not None:
        unseen = sorted({str(v) for v in values.dropna().unique()} - known)
        if unseen:
            return [f"{column}: value(s) {', '.join(unseen)} never seen in training; the model treats them as unknown."]
    return []


def _known_categories(model: TrainedModel, column: str) -> set[str] | None:
    preprocess = model.pipeline.named_steps["preprocess"]
    for name, transformer, columns in preprocess.transformers_:
        if name == "cat" and column in list(columns):
            encoder = transformer.named_steps["onehot"]
            return {str(value) for value in encoder.categories_[list(columns).index(column)]}
    return None


def _context_rows(model: TrainedModel) -> int:
    ts = model.pipeline.named_steps["ts_features"]
    return max([*ts.lags, *ts.windows]) + 1


def _deltas(before: Any, after: Any) -> dict[str, float | None]:
    if not (_is_number(before) and _is_number(after)):
        return {"delta": None, "delta_pct": None}
    delta = float(after) - float(before)
    pct = delta / abs(float(before)) * 100 if float(before) != 0 else None
    return {"delta": round(delta, 6), "delta_pct": round(pct, 4) if pct is not None else None}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _py(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value

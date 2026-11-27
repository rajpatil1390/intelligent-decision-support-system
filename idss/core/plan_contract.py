"""The plan contract: the ONLY output of the planning agents that can drive execution (v1.1).

The Requirement Agent and the Model Agent may propose anything; what reaches the AutoML engine is an
AnalysisPlan that passed this strict schema (no extra keys, no type coercion, names only from the model zoo and
the metric registry) and then `idss.execution.plan_validator.validate_plan`, which checks it against the data.
Nothing is dropped silently here: an unknown key, model or metric makes the whole plan invalid, and the
validator then replaces it with a deterministic fallback plan built from the data profile.
"""

from collections import Counter
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from idss.core.schemas import BudgetMode, TaskType
from idss.ml import model_zoo
from idss.ml.evaluator import METRIC_FAMILY_BY_TASK, METRICS

MAX_CANDIDATES = 6
MAX_ACTIONABLE_FEATURES = 20
MAX_INFERENCE_MS = 60_000.0  # one minute per row: anything above is not a real constraint
MAX_TRAIN_MINUTES = 24 * 60.0
_MAX_NAME_CHARS = 200

TASK_TYPES: tuple[str, ...] = get_args(TaskType)
SUPERVISED_TASKS = frozenset(task for task in TASK_TYPES if task != "tabular_clustering")

# Allowed names come from the registries themselves, so the contract cannot drift from what the engine can run.
# Baselines (seasonal naive) are always trained by the engine and are not candidates.
TASK_MODELS: dict[str, tuple[str, ...]] = {task: tuple(model_zoo.candidate_names(task)) for task in TASK_TYPES}
TASK_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    task: tuple(name for name, spec in METRICS.items() if spec.family == METRIC_FAMILY_BY_TASK[task]) for task in TASK_TYPES
}
MODEL_NAMES: tuple[str, ...] = tuple(dict.fromkeys(name for names in TASK_MODELS.values() for name in names))
METRIC_NAMES: tuple[str, ...] = tuple(METRICS)

ModelName = Literal[MODEL_NAMES]  # type: ignore[valid-type]  # built from the model zoo at import time
MetricName = Literal[METRIC_NAMES]  # type: ignore[valid-type]  # built from the metric registry at import time
ColumnName = Annotated[str, StringConstraints(min_length=1, max_length=_MAX_NAME_CHARS)]

_STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)


class PlanConstraints(BaseModel):
    """User limits the engine must respect. Bounded so a nonsensical number is rejected, not obeyed."""

    model_config = _STRICT

    max_inference_ms_per_row: float | None = Field(default=None, gt=0, le=MAX_INFERENCE_MS)
    max_train_minutes: float | None = Field(default=None, gt=0, le=MAX_TRAIN_MINUTES)
    prefer_interpretable: bool = False


class AnalysisPlan(BaseModel):
    """A structured plan proposed by the agents. Valid here means well-formed, not yet checked against the data."""

    model_config = _STRICT

    task_type: TaskType
    target: ColumnName | None = Field(
        default=None, description="Column to predict; for clustering an optional ground-truth column (evaluation only)."
    )
    primary_metric: MetricName
    candidate_models: list[ModelName] = Field(min_length=1, max_length=MAX_CANDIDATES)
    budget_mode: BudgetMode
    constraints: PlanConstraints = Field(default_factory=PlanConstraints)
    actionable_features: list[ColumnName] = Field(default_factory=list, max_length=MAX_ACTIONABLE_FEATURES)
    time_column: ColumnName | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "AnalysisPlan":
        task = self.task_type
        problems: list[str] = []
        if duplicates := _duplicates(self.candidate_models):
            problems.append(f"candidate_models lists {', '.join(duplicates)} more than once")
        if wrong := [name for name in self.candidate_models if name not in TASK_MODELS[task]]:
            problems.append(f"model(s) {', '.join(wrong)} cannot be used for {task} (allowed: {', '.join(TASK_MODELS[task])})")
        if self.primary_metric not in TASK_METRIC_NAMES[task]:
            problems.append(
                f"metric '{self.primary_metric}' does not apply to {task} (allowed: {', '.join(TASK_METRIC_NAMES[task])})"
            )
        if task in SUPERVISED_TASKS and self.target is None:
            problems.append(f"{task} needs a target column")
        if task == "time_series_forecasting" and self.time_column is None:
            problems.append("time_series_forecasting needs a time_column")
        if self.time_column is not None and self.time_column == self.target:
            problems.append("time_column must differ from the target")
        if duplicates := _duplicates(self.actionable_features):
            problems.append(f"actionable_features lists {', '.join(duplicates)} more than once")
        if self.target is not None and self.target in self.actionable_features:
            problems.append(f"the target '{self.target}' cannot also be an actionable feature")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class PlanProposal(BaseModel):
    """What the planning agents hand over. `payload` is untrusted until validate_plan accepts it."""

    payload: dict[str, Any] = Field(default_factory=dict, description="Raw AnalysisPlan fields.")
    source: Literal["llm", "default"] = "default"
    reasons: dict[str, str] = Field(default_factory=dict, description="Model name -> why it was chosen (display only).")
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list, description="The planner's own output was rejected before it became a payload."
    )


class PlanContractError(ValueError):
    """A proposal violates the plan contract."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def parse_plan(payload: AnalysisPlan | dict[str, Any]) -> AnalysisPlan:
    """Validate `payload` against the contract. Raises PlanContractError with one readable reason per problem.

    An AnalysisPlan instance is checked again too: it may have been altered after it was built."""
    data = payload.model_dump() if isinstance(payload, AnalysisPlan) else payload
    try:
        return AnalysisPlan.model_validate(data)
    except ValidationError as exc:
        raise PlanContractError(describe_errors(exc)) from None


def describe_errors(exc: ValidationError) -> list[str]:
    """One short sentence per validation error (allowed values are listed only where they help)."""
    reasons: list[str] = []
    for error in exc.errors(include_url=False):
        field = ".".join(str(part) for part in error["loc"])
        kind, value = error["type"], error.get("input")
        if kind == "extra_forbidden":
            reasons.append(f"Unknown field '{field}' is not part of the plan contract.")
        elif kind == "literal_error" and field.startswith("candidate_models"):
            reasons.append(f"Model {value!r} is not in the model zoo.")
        elif kind == "literal_error" and field == "primary_metric":
            reasons.append(f"Metric {value!r} is not a metric the system can measure.")
        elif kind == "missing":
            reasons.append(f"Required field '{field}' is missing.")
        elif kind == "value_error":
            reasons.append(f"Plan is inconsistent: {error['msg'].removeprefix('Value error, ')}.")
        else:
            reasons.append(f"Field '{field or 'plan'}' is invalid ({error['msg']}; got {_short(value)}).")
    return reasons


def _duplicates(values: list[str]) -> list[str]:
    return [value for value, count in Counter(values).items() if count > 1]


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."

"""Independent plan validation (v1.1): the engine never trusts a plan because an LLM proposed it.

`validate_plan(plan, df, profile, settings)` checks a proposed AnalysisPlan against the contract, the actual data
and the hard limits in the settings, and returns either a sealed ValidatedPlan (the only thing the Orchestrator
lets into TRAIN) or PlanRejected(reasons). Every check is deterministic and independent of the LLM: columns must
exist, the target must suit the task, forecasting needs a datetime column, actionable features must be real
non-ID columns, and the budget must sit inside the limits; the public demo always runs in fast mode.

On rejection `validate_or_fallback` builds a fallback plan from a fresh profile of the data (never from the LLM's
text), validates that too, and records the rejection reasons on the result so the Run Monitor can show them. A
data problem that no plan can fix (too many rows, file too large, nothing to learn from) is fatal.
"""

import hashlib
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from configs.settings import Settings
from idss.agents.requirement_agent import TASK_METRICS
from idss.core.logger import get_logger
from idss.core.plan_contract import (
    MAX_CANDIDATES,
    AnalysisPlan,
    PlanConstraints,
    PlanContractError,
    PlanProposal,
    parse_plan,
)
from idss.core.schemas import (
    BUDGET_PRESETS,
    BudgetMode,
    ColumnProfile,
    DatasetProfile,
    ModelConstraints,
    ModelPlan,
    RankedCandidate,
    TrainingBudget,
)
from idss.ml import model_zoo
from idss.ml.evaluator import is_supported
from idss.ml.profiler import CLASSIFICATION_TASKS, MAX_INT_CLASSES, profile_dataset

logger = get_logger()

TRAIN_TIME_HEADROOM = 0.8  # share of a training-time limit given to the search; the rest covers the final refit
PlanSource = Literal["llm", "default", "fallback"]
_UNPREDICTABLE = frozenset({"id", "text", "datetime"})
_FEATURE_TYPES = frozenset({"numeric", "categorical", "boolean", "datetime"})  # what the preprocessor can use


class PlanRejected(BaseModel):
    """Why a plan was refused. `fatal` = a data problem that no plan (not even the fallback) can fix."""

    model_config = ConfigDict(frozen=True)

    reasons: list[str] = Field(min_length=1)
    fatal: bool = False


class ValidatedPlan(BaseModel):
    """A plan that passed validate_plan, plus the engine's ModelPlan built from it.

    Only validate_plan seals one. A ValidatedPlan that is not sealed (read back from state.json, or built by hand)
    or that changed after sealing is validated again before TRAIN may start.
    """

    model_config = ConfigDict(frozen=True)

    analysis: AnalysisPlan
    model_plan: ModelPlan
    source: PlanSource
    reasons: dict[str, str] = Field(default_factory=dict, description="Model name -> why it was chosen.")
    notes: list[str] = Field(default_factory=list)
    fallback: bool = Field(default=False, description="True when the proposed plan was rejected.")
    rejected_reasons: list[str] = Field(default_factory=list, description="Why the proposed plan was rejected.")
    fingerprint: str = ""
    _sealed: bool = PrivateAttr(default=False)

    @property
    def sealed(self) -> bool:
        """Sealed by validate_plan in this process and unchanged since."""
        return self._sealed and self.fingerprint == _fingerprint(self)


class PlanValidationError(Exception):
    """Neither the proposed plan nor the fallback plan can run on this data."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


# --------------------------------------------------------------------------- public API


def validate_plan(
    plan: AnalysisPlan | dict[str, Any],
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: Settings,
    *,
    budget: TrainingBudget | None = None,
    public_demo: bool | None = None,
    data_path: str | Path | None = None,
    source: PlanSource = "llm",
    reasons: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> ValidatedPlan | PlanRejected:
    """Check `plan` against the contract, the data and the settings' hard limits.

    `budget` replaces the budget derived from the plan's mode (it is checked against the same limits);
    `public_demo` defaults to the setting; `data_path` lets the file size be checked against the upload limit.
    """
    demo = settings.public_demo if public_demo is None else public_demo
    if data_problems := check_data_limits(df, settings, data_path=data_path, public_demo=demo):
        return PlanRejected(reasons=data_problems, fatal=True)
    try:
        analysis = parse_plan(plan)
    except PlanContractError as exc:
        return PlanRejected(reasons=exc.reasons)

    notes = list(notes or [])
    if demo and analysis.budget_mode == "full":
        analysis = analysis.model_copy(update={"budget_mode": "fast"})
        notes.append("Public demo: the fast budget is enforced.")

    problems = [
        *_check_target(analysis, df, profile),
        *_check_time_column(analysis, df, profile),
        *_check_actionable(analysis, df, profile),
        *_check_metric(analysis, df),
        *_check_features(analysis, profile),
    ]
    model_plan = build_model_plan(analysis, source=source, reasons=reasons, notes=notes, budget=budget)
    problems += _check_budget(model_plan.budget, analysis.constraints, settings)
    if problems:
        return PlanRejected(reasons=problems)
    return _seal(analysis, model_plan, source=source, reasons=reasons, notes=model_plan.notes)


def validate_or_fallback(
    proposal: PlanProposal | AnalysisPlan | dict[str, Any],
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: Settings,
    *,
    budget: TrainingBudget | None = None,
    public_demo: bool | None = None,
    data_path: str | Path | None = None,
) -> ValidatedPlan:
    """Validate the proposal; if it is rejected, validate and return the fallback plan instead.

    Raises PlanValidationError when the data itself is unusable (fatal) or the fallback is rejected too.
    """
    if not isinstance(proposal, PlanProposal):
        proposal = PlanProposal(payload=proposal.model_dump() if isinstance(proposal, AnalysisPlan) else dict(proposal))
    options: dict[str, Any] = {"public_demo": public_demo, "data_path": data_path}
    if proposal.errors:
        result: ValidatedPlan | PlanRejected = PlanRejected(reasons=proposal.errors)
    else:
        result = validate_plan(proposal.payload, df, profile, settings, budget=budget, source=proposal.source,
                               reasons=proposal.reasons, notes=proposal.notes, **options)
    if isinstance(result, ValidatedPlan):
        return result
    if result.fatal:
        raise PlanValidationError(result.reasons)
    for reason in result.reasons:
        logger.warning("Plan rejected: {}", reason)

    fallback = validate_plan(fallback_plan(df, settings), df, profile, settings, budget=clamp_budget(budget, settings),
                             source="fallback", notes=["The proposed plan was rejected; this fallback plan was built "
                                                       "from the data profile."], **options)
    if isinstance(fallback, PlanRejected):
        raise PlanValidationError([*result.reasons, *(f"Fallback plan rejected too: {r}" for r in fallback.reasons)])
    logger.warning("Using the fallback plan: {} on '{}' with {}.", fallback.analysis.task_type,
                   fallback.analysis.target, ", ".join(fallback.analysis.candidate_models))
    return _seal(fallback.analysis, fallback.model_plan, source="fallback", notes=fallback.notes, fallback=True,
                 rejected_reasons=result.reasons)


def revalidate(
    validated: ValidatedPlan,
    df: pd.DataFrame,
    profile: DatasetProfile,
    settings: Settings,
    *,
    budget: TrainingBudget | None = None,
    public_demo: bool | None = None,
    data_path: str | Path | None = None,
) -> ValidatedPlan | PlanRejected:
    """Validate a stored plan again (unsealed or changed since sealing); keeps its fallback record."""
    if validated.fallback:
        budget = clamp_budget(budget, settings)
    result = validate_plan(validated.analysis, df, profile, settings, budget=budget, public_demo=public_demo,
                           data_path=data_path, source=validated.source, reasons=validated.reasons,
                           notes=validated.notes)
    if isinstance(result, PlanRejected):
        return result
    return _seal(result.analysis, result.model_plan, source=validated.source, reasons=validated.reasons,
                 notes=result.notes, fallback=validated.fallback, rejected_reasons=validated.rejected_reasons)


def fallback_plan(df: pd.DataFrame, settings: Settings) -> AnalysisPlan:
    """Deterministic plan from a fresh profile of the data: suggested target and task, default metric and models."""
    profile = profile_dataset(df)  # fresh heuristics: no hint from the LLM-parsed requirement
    task = profile.suggested_task_type or "tabular_clustering"
    target = None if task == "tabular_clustering" else profile.suggested_target
    labels = target is not None or task != "tabular_clustering"
    metric = next(m for m in [*TASK_METRICS[task], "silhouette"] if is_supported(m, task, labels_available=labels))
    limit = min(settings.plan_max_candidates, MAX_CANDIDATES)
    models = model_zoo.DEFAULT_RANKING[model_zoo.task_family(task)][:limit]
    time_column = default_time_column(profile) if task == "time_series_forecasting" else None
    return AnalysisPlan(task_type=task, target=target, primary_metric=metric, candidate_models=models,
                        budget_mode="fast", constraints=PlanConstraints(), actionable_features=[], time_column=time_column)


def check_data_limits(
    df: pd.DataFrame, settings: Settings, *, data_path: str | Path | None = None, public_demo: bool = False
) -> list[str]:
    """Data-size problems that no plan can fix."""
    problems: list[str] = []
    if len(df) > settings.plan_max_rows:
        problems.append(f"The data has {len(df):,} rows; the limit is {settings.plan_max_rows:,} rows.")
    if data_path is not None and Path(data_path).is_file():
        size_mb = Path(data_path).stat().st_size / 2**20
        limit = settings.public_demo_max_upload_mb if public_demo else settings.max_upload_mb
        if size_mb > limit:
            problems.append(f"The data file is {size_mb:.1f} MB; the limit is {limit:g} MB.")
    return problems


def build_model_plan(
    analysis: AnalysisPlan,
    *,
    source: PlanSource,
    reasons: dict[str, str] | None = None,
    notes: list[str] | None = None,
    budget: TrainingBudget | None = None,
) -> ModelPlan:
    """The engine's view of a plan: concrete budget, constraints in seconds, candidates with reasons."""
    notes = list(dict.fromkeys(notes or []))
    limits = analysis.constraints
    constraints = ModelConstraints(
        max_inference_ms_per_row=limits.max_inference_ms_per_row,
        max_train_time_s=None if limits.max_train_minutes is None else round(limits.max_train_minutes * 60, 6),
        prefer_interpretable=limits.prefer_interpretable,
    )
    if budget is None:
        budget = budget_for(analysis.budget_mode, constraints, notes)
    family = model_zoo.task_family(analysis.task_type)
    reasons = reasons or {}
    candidates = [RankedCandidate(name=name, reason=reasons.get(name) or model_zoo.REGISTRY[family][name].description)
                  for name in analysis.candidate_models]
    candidates = ensure_interpretable(candidates, constraints, budget.max_candidates, analysis.task_type, notes)
    return ModelPlan(task_type=analysis.task_type, target=analysis.target, candidates=candidates,
                     primary_metric=analysis.primary_metric, budget_mode=analysis.budget_mode, budget=budget,
                     constraints=constraints, notes=list(dict.fromkeys(notes)), source=source,
                     time_column=analysis.time_column)


def budget_for(mode: BudgetMode, constraints: ModelConstraints, notes: list[str]) -> TrainingBudget:
    """The preset for `mode`, shrunk to fit a training-time limit if the user gave one."""
    budget = BUDGET_PRESETS[mode].model_copy()
    if constraints.max_train_time_s is not None:
        total = max(1.0, constraints.max_train_time_s * TRAIN_TIME_HEADROOM)
        if total < budget.total_timeout_s:
            budget = budget.model_copy(update={
                "total_timeout_s": total,
                "timeout_per_model_s": min(budget.timeout_per_model_s, max(1.0, total / budget.max_candidates)),
            })
            notes.append(f"Search time limited to {total:g}s to respect the training-time constraint.")
    return budget


def clamp_budget(budget: TrainingBudget | None, settings: Settings) -> TrainingBudget:
    """`budget` (default: the fast preset) reduced to fit every hard limit; used for the fallback plan."""
    base = budget or BUDGET_PRESETS["fast"]
    return TrainingBudget(
        max_candidates=min(base.max_candidates, settings.plan_max_candidates),
        n_trials_per_model=min(base.n_trials_per_model, settings.plan_max_trials_per_model),
        timeout_per_model_s=min(base.timeout_per_model_s, settings.plan_max_seconds_per_model),
        total_timeout_s=min(base.total_timeout_s, settings.plan_max_total_seconds),
    )


def ensure_interpretable(
    candidates: list[RankedCandidate], constraints: ModelConstraints, max_candidates: int, task: str, notes: list[str]
) -> list[RankedCandidate]:
    """If the user wants explainability, make sure an interpretable model is among those that will be trained."""
    if not constraints.prefer_interpretable:
        return candidates
    interpretable = [c for c in model_zoo.candidates_for(task) if c.interpretable]
    names = {c.name for c in interpretable}
    if not interpretable or any(c.name in names for c in candidates[:max_candidates]):
        return candidates
    index = next((i for i, c in enumerate(candidates) if c.name in names), None)
    if index is None:
        moved, chosen = list(candidates), RankedCandidate(name=interpretable[0].name, reason=interpretable[0].description)
    else:
        moved, chosen = candidates[:index] + candidates[index + 1:], candidates[index]
    moved.insert(max_candidates - 1, chosen)
    notes.append(f"Included '{chosen.name}' because the user asked for an explainable model.")
    return moved


def default_time_column(profile: DatasetProfile) -> str | None:
    """The profile's time index, else its first datetime column."""
    return profile.time_column or next(iter(profile.columns_of("datetime")), None)


# --------------------------------------------------------------------------- checks


def _column(name: str, df: pd.DataFrame, profile: DatasetProfile) -> ColumnProfile | None:
    if name not in df.columns:
        return None
    try:
        return profile.column(name)
    except KeyError:
        return None


def _check_target(plan: AnalysisPlan, df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    target, task = plan.target, plan.task_type
    if target is None:
        return []
    col = _column(target, df, profile)
    if col is None:
        return [f"Target column '{target}' is not in the data."]
    if col.semantic_type == "id":
        return [f"Target '{target}' is an identifier column (a different value on almost every row); it cannot be predicted."]
    if col.n_unique == 0:
        return [f"Target '{target}' has no values (every row is missing)."]
    if col.n_unique == 1:
        return [f"Target '{target}' has a single value, so there is nothing to predict."]
    if col.semantic_type in _UNPREDICTABLE:
        return [f"Target '{target}' is a {col.semantic_type} column; it cannot be predicted."]
    if task in CLASSIFICATION_TASKS:
        if col.semantic_type == "numeric" and not (col.stats.get("integer") and col.n_unique <= MAX_INT_CLASSES):
            return [f"Target '{target}' is a continuous number ({col.n_unique:,} distinct values); "
                    f"{task} needs a categorical target (use regression instead)."]
        if task == "text_classification" and not any(c.semantic_type == "text" for c in profile.columns if c.name != target):
            return ["text_classification needs a free-text column, but the data has none."]
    elif task in ("tabular_regression", "time_series_forecasting") and col.semantic_type != "numeric":
        return [f"Target '{target}' is {col.semantic_type}; {task} needs a numeric target."]
    return []


def _check_time_column(plan: AnalysisPlan, df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    name = plan.time_column
    if name is None:
        return []
    col = _column(name, df, profile)
    if col is None:
        return [f"Time column '{name}' is not in the data."]
    if col.semantic_type != "datetime":
        return [f"Time column '{name}' is not a date/time column (it is {col.semantic_type})."]
    return []


def _check_actionable(plan: AnalysisPlan, df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    problems: list[str] = []
    for name in plan.actionable_features:
        col = _column(name, df, profile)
        if col is None:
            problems.append(f"Actionable feature '{name}' is not in the data.")
        elif col.semantic_type == "id":
            problems.append(f"Actionable feature '{name}' is an identifier column; the business cannot change it.")
    return problems


def _check_metric(plan: AnalysisPlan, df: pd.DataFrame) -> list[str]:
    metric, task, target = plan.primary_metric, plan.task_type, plan.target
    if not is_supported(metric, task, labels_available=target is not None):
        return [f"Metric '{metric}' needs ground-truth labels, but the plan names no ground-truth column."]
    if target is None or target not in df.columns or metric not in ("rmsle", "mape"):
        return []
    values = pd.to_numeric(df[target], errors="coerce")
    if metric == "rmsle" and (values < 0).any():
        return [f"Metric 'rmsle' needs a non-negative target, but '{target}' has negative values."]
    if metric == "mape" and (values == 0).any():
        return [f"Metric 'mape' is undefined because '{target}' contains zeros."]
    return []


def _check_features(plan: AnalysisPlan, profile: DatasetProfile) -> list[str]:
    if plan.task_type == "time_series_forecasting":
        return []  # lag and rolling features are built from the target itself
    usable = [c.name for c in profile.columns
              if c.name != plan.target and c.n_unique > 1 and c.semantic_type in _FEATURE_TYPES]
    if usable:
        return []
    return ["No usable feature columns besides the target: every other column is constant, an identifier or free text."]


def _check_budget(budget: TrainingBudget, constraints: PlanConstraints, settings: Settings) -> list[str]:
    limits = [
        ("models", budget.max_candidates, settings.plan_max_candidates),
        ("tuning trials per model", budget.n_trials_per_model, settings.plan_max_trials_per_model),
        ("seconds per model", budget.timeout_per_model_s, settings.plan_max_seconds_per_model),
        ("seconds in total", budget.total_timeout_s, settings.plan_max_total_seconds),
    ]
    problems = [f"The budget asks for {value:g} {label}; the limit is {limit:g}." for label, value, limit in limits
                if value > limit]
    minutes = constraints.max_train_minutes
    if minutes is not None and minutes > settings.plan_max_train_minutes:
        problems.append(f"The training-time constraint of {minutes:g} minutes exceeds the limit of "
                        f"{settings.plan_max_train_minutes:g} minutes.")
    return problems


# --------------------------------------------------------------------------- sealing


def _seal(
    analysis: AnalysisPlan,
    model_plan: ModelPlan,
    *,
    source: PlanSource,
    reasons: dict[str, str] | None = None,
    notes: list[str] | None = None,
    fallback: bool = False,
    rejected_reasons: list[str] | None = None,
) -> ValidatedPlan:
    plan = ValidatedPlan(analysis=analysis, model_plan=model_plan, source=source, reasons=reasons or {},
                         notes=list(dict.fromkeys(notes or [])), fallback=fallback,
                         rejected_reasons=list(rejected_reasons or []))
    plan = plan.model_copy(update={"fingerprint": _fingerprint(plan)})
    plan._sealed = True
    return plan


def _fingerprint(plan: ValidatedPlan) -> str:
    return hashlib.sha256(plan.model_dump_json(exclude={"fingerprint"}).encode("utf-8")).hexdigest()

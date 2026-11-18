"""Model Agent: decide which zoo models to train, which metric to optimise and how much to search.

The LLM only ranks names from the model zoo (with reasons) and suggests a budget mode; it never writes
code. Unknown names are dropped, models the user asked for go first, and if the LLM fails or picks
nothing usable a deterministic ranking is used. User constraints such as "inference < 5 ms" or
"training < 30 min" are parsed without the LLM and applied to the budget and to model selection.
"""

import re

from pydantic import BaseModel, Field

from idss.agents.requirement_agent import TASK_METRICS
from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.prompts import format_profile, render_prompt
from idss.core.schemas import (
    BUDGET_PRESETS,
    BudgetMode,
    DatasetProfile,
    KnowledgeBrief,
    ModelConstraints,
    ModelPlan,
    RankedCandidate,
    TrainingBudget,
    UserRequirement,
)
from idss.ml import model_zoo
from idss.ml.evaluator import is_supported, normalize_metric_name

logger = get_logger()

TRAIN_TIME_HEADROOM = 0.8  # share of a training-time limit given to the search; the rest covers the final refit
_DURATION = r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b"
_INFERENCE_LIMIT = re.compile(r"(?:inference|predict\w*|latency|response\w*|scoring)\D{0,40}?" + _DURATION, re.IGNORECASE)
_TRAINING_LIMIT = re.compile(r"(?:train\w*|fit\w*|build\w*)\D{0,40}?" + _DURATION, re.IGNORECASE)
_EXPLAINABLE = re.compile(r"explain|interpret|transparen|understand", re.IGNORECASE)


class ModelChoice(BaseModel):
    """What the LLM returns."""

    candidates: list[RankedCandidate] = Field(default_factory=list)
    primary_metric: str | None = None
    budget_mode: BudgetMode = "fast"


class ModelAgent:
    """Builds the ModelPlan for a run. Never raises because of the LLM."""

    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    def plan(
        self,
        requirement: UserRequirement,
        profile: DatasetProfile | None,
        *,
        budget_mode: BudgetMode | None = None,
        knowledge: KnowledgeBrief | None = None,
        feedback: list[str] | None = None,
    ) -> ModelPlan:
        """Plan the model search. `knowledge` (retrieved best practices) and `feedback` (verifier
        recommendations from a previous attempt) are shown to the LLM when it ranks the models."""
        task = requirement.problem.task_type
        notes: list[str] = []
        constraints = parse_constraints(requirement.problem.constraints)
        if feedback:
            notes.append("Re-planned using the verifier's recommendations.")
        choice = self._ask_llm(requirement, profile, knowledge, feedback or [])

        candidates, source = self._rank(choice, requirement, notes)
        mode: BudgetMode = budget_mode or (choice.budget_mode if choice else "fast")
        budget = budget_for(mode, constraints, notes)
        candidates = _ensure_interpretable(candidates, constraints, budget.max_candidates, task, notes)

        labels_available = _labels_available(requirement, profile)
        primary = choose_primary_metric(requirement, choice.primary_metric if choice else None, labels_available, notes)
        target = requirement.target if task != "tabular_clustering" or labels_available else None
        for note in notes:
            logger.info("Model plan: {}", note)
        return ModelPlan(
            task_type=task,
            target=target,
            candidates=candidates,
            primary_metric=primary,
            budget_mode=mode,
            budget=budget,
            constraints=constraints,
            notes=notes,
            source=source,
        )

    def _ask_llm(
        self,
        requirement: UserRequirement,
        profile: DatasetProfile | None,
        knowledge: KnowledgeBrief | None,
        feedback: list[str],
    ) -> ModelChoice | None:
        if self.llm_client is None:
            return None
        task = requirement.problem.task_type
        allowed = "\n".join(f"- {c.name}: {c.description}" for c in model_zoo.candidates_for(task))
        try:
            messages = render_prompt(
                "model_selection",
                requirement_json=requirement.model_dump_json(exclude={"system_notes"}, indent=2),
                dataset_context=format_profile(profile),
                allowed_models=allowed,
                constraints="\n".join(f"- {c}" for c in requirement.problem.constraints) or "- none",
                knowledge=knowledge.as_prompt_text() if knowledge else "None available.",
                feedback="\n".join(f"- {item}" for item in feedback) or "None (first attempt).",
            )
            return self.llm_client.chat_structured(messages, ModelChoice)
        except Exception as exc:  # the default ranking is always a safe fallback
            logger.warning("Model selection by LLM failed ({}: {}); using the default ranking.", type(exc).__name__, exc)
            return None

    @staticmethod
    def _rank(
        choice: ModelChoice | None, requirement: UserRequirement, notes: list[str]
    ) -> tuple[list[RankedCandidate], str]:
        task = requirement.problem.task_type
        allowed = set(model_zoo.candidate_names(task))
        ranked: dict[str, RankedCandidate] = {}
        for item in choice.candidates if choice else []:
            name = model_zoo.normalize_model_name(item.name)
            if name not in allowed:
                notes.append(f"Model '{item.name}' is not in the model zoo for {task}; ignored.")
            elif name not in ranked:
                ranked[name] = RankedCandidate(name=name, reason=item.reason or "Chosen by the model agent.")
        source = "llm" if ranked else "default"
        if choice is not None and not ranked:
            notes.append("The LLM chose no usable model; using the default ranking.")

        requested = [model_zoo.normalize_model_name(m.name) for m in requirement.models if m.name]
        user_first = [RankedCandidate(name=n, reason="Requested by the user.") for n in dict.fromkeys(requested) if n in allowed]
        ordered = {c.name: c for c in user_first} | {n: c for n, c in ranked.items() if n not in requested}

        family = model_zoo.task_family(task)
        for name in model_zoo.DEFAULT_RANKING[family]:
            if name not in ordered:
                ordered[name] = RankedCandidate(name=name, reason=model_zoo.REGISTRY[family][name].description)
        return list(ordered.values()), source


def parse_constraints(texts: list[str]) -> ModelConstraints:
    """Read inference-time, training-time and explainability limits from free-text constraints."""
    inference_ms: float | None = None
    train_s: float | None = None
    interpretable = False
    for text in texts:
        if match := _INFERENCE_LIMIT.search(text):
            value = _seconds(*match.groups()) * 1000
            inference_ms = value if inference_ms is None else min(inference_ms, value)
        elif match := _TRAINING_LIMIT.search(text):
            value = _seconds(*match.groups())
            train_s = value if train_s is None else min(train_s, value)
        if _EXPLAINABLE.search(text):
            interpretable = True
    return ModelConstraints(max_inference_ms_per_row=inference_ms, max_train_time_s=train_s, prefer_interpretable=interpretable)


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


def choose_primary_metric(
    requirement: UserRequirement, suggestion: str | None, labels_available: bool, notes: list[str]
) -> str:
    """First supported metric from: the requirement, the LLM's suggestion, the task defaults."""
    task = requirement.problem.task_type
    wanted = [m.name for m in requirement.problem.performance_metrics]
    options = [*wanted, *([suggestion] if suggestion else []), *TASK_METRICS[task], "silhouette"]
    for option in options:
        name = normalize_metric_name(option)
        if is_supported(name, task, labels_available=labels_available):
            if wanted and name != normalize_metric_name(wanted[0]):
                notes.append(f"Metric '{wanted[0]}' cannot be used here; optimising '{name}' instead.")
            return name
    raise ValueError(f"No supported metric for task {task}.")  # unreachable: task defaults are always supported


def _ensure_interpretable(
    candidates: list[RankedCandidate], constraints: ModelConstraints, max_candidates: int, task: str, notes: list[str]
) -> list[RankedCandidate]:
    """If the user wants explainability, make sure an interpretable model is among those that will be trained."""
    if not constraints.prefer_interpretable:
        return candidates
    interpretable = {c.name for c in model_zoo.candidates_for(task) if c.interpretable}
    if any(c.name in interpretable for c in candidates[:max_candidates]):
        return candidates
    for index, candidate in enumerate(candidates):
        if candidate.name in interpretable:
            moved = candidates[:index] + candidates[index + 1:]
            moved.insert(max_candidates - 1, candidate)
            notes.append(f"Included '{candidate.name}' because the user asked for an explainable model.")
            return moved
    return candidates


def _labels_available(requirement: UserRequirement, profile: DatasetProfile | None) -> bool:
    """Clustering can be scored against ground truth only if the requirement names an existing column."""
    if requirement.problem.task_type != "tabular_clustering":
        return True
    return bool(requirement.target and profile and requirement.target in {c.name for c in profile.columns})


def _seconds(value: str, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith(("ms", "milli")):
        return float(value) / 1000
    if unit.startswith("h"):
        return float(value) * 3600
    if unit.startswith("m"):
        return float(value) * 60
    return float(value)

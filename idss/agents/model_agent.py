"""Model Agent: propose which zoo models to train, which metric to optimise and how much to search.

The LLM only ranks names from the model zoo (with reasons) and suggests a metric and a budget mode; it never
writes code. Since v1.1 the agent no longer cleans up the LLM's answer: its output is a PlanProposal whose payload
must pass the plan contract (idss.core.plan_contract) and the independent validator
(idss.execution.plan_validator). An unknown or duplicated model name, a metric that does not fit the task or an
unknown key therefore rejects the proposal (the validator then uses a fallback plan) instead of being dropped.
Models the user asked for go first and the default ranking fills the list. User constraints such as
"inference < 5 ms" or "training < 30 min" are parsed without the LLM.
"""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from idss.agents.requirement_agent import TASK_METRICS
from idss.core.llm_client import LLMClient, LLMValidationError
from idss.core.logger import get_logger
from idss.core.plan_contract import MAX_CANDIDATES, PlanContractError, PlanProposal, parse_plan
from idss.core.prompts import format_profile, format_requirement, render_prompt
from idss.core.sanitize import data_lines
from idss.core.schemas import (
    BudgetMode,
    DatasetProfile,
    KnowledgeBrief,
    ModelConstraints,
    ModelPlan,
    RankedCandidate,
    UserRequirement,
)
from idss.execution.plan_validator import (  # noqa: F401  (budget_for / TRAIN_TIME_HEADROOM re-exported)
    TRAIN_TIME_HEADROOM,
    budget_for,
    build_model_plan,
    default_time_column,
)
from idss.ml import model_zoo
from idss.ml.evaluator import is_supported, normalize_metric_name

logger = get_logger()

_DURATION = r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b"
_INFERENCE_LIMIT = re.compile(r"(?:inference|predict\w*|latency|response\w*|scoring)\D{0,40}?" + _DURATION, re.IGNORECASE)
_TRAINING_LIMIT = re.compile(r"(?:train\w*|fit\w*|build\w*)\D{0,40}?" + _DURATION, re.IGNORECASE)
_EXPLAINABLE = re.compile(r"explain|interpret|transparen|understand", re.IGNORECASE)


class ModelChoice(BaseModel):
    """What the LLM returns. Unknown keys are an error (the LLM client asks the model to correct them)."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[RankedCandidate] = Field(default_factory=list)
    primary_metric: str | None = None
    budget_mode: BudgetMode = "fast"


class ModelAgent:
    """Builds the plan proposal for a run. Never raises because of the LLM."""

    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    def propose(
        self,
        requirement: UserRequirement,
        profile: DatasetProfile | None,
        *,
        budget_mode: BudgetMode | None = None,
        knowledge: KnowledgeBrief | None = None,
        feedback: list[str] | None = None,
    ) -> PlanProposal:
        """Assemble an (unvalidated) AnalysisPlan payload from the requirement and the LLM's model choice.

        `budget_mode` overrides the LLM's fast/full suggestion. `knowledge` (retrieved best practices) and
        `feedback` (verifier recommendations from a previous attempt) are shown to the LLM.
        """
        task = requirement.problem.task_type
        notes: list[str] = []
        if feedback:
            notes.append("Re-planned using the verifier's recommendations.")
        choice, rejection = self._ask_llm(requirement, profile, knowledge, feedback or [])
        if rejection is not None:
            return PlanProposal(source="llm", notes=notes, errors=[rejection])

        names, reasons, source = self._rank(choice, requirement, notes)
        labels_available = _labels_available(requirement, profile)
        if choice is not None and choice.primary_metric:
            metric = normalize_metric_name(choice.primary_metric)  # checked by the contract, not repaired here
        else:
            metric = choose_primary_metric(requirement, None, labels_available, notes)
        payload: dict[str, Any] = {
            "task_type": task,
            "target": requirement.target,
            "primary_metric": metric,
            "candidate_models": names,
            "budget_mode": budget_mode or (choice.budget_mode if choice else "fast"),
            "constraints": parse_constraint_values(requirement.problem.constraints),
            "actionable_features": list(requirement.actionable_features),
            "time_column": default_time_column(profile) if task == "time_series_forecasting" and profile else None,
        }
        return PlanProposal(payload=payload, source=source, reasons=reasons, notes=notes)

    def plan(
        self,
        requirement: UserRequirement,
        profile: DatasetProfile | None,
        *,
        budget_mode: BudgetMode | None = None,
        knowledge: KnowledgeBrief | None = None,
        feedback: list[str] | None = None,
    ) -> ModelPlan:
        """Contract-checked ModelPlan for scripts and tests that call the trainer directly.

        The Orchestrator never uses this: every run plan goes through plan_validator.validate_plan, which also
        checks the plan against the data. Raises PlanContractError if the proposal breaks the contract.
        """
        proposal = self.propose(requirement, profile, budget_mode=budget_mode, knowledge=knowledge, feedback=feedback)
        if proposal.errors:
            raise PlanContractError(proposal.errors)
        plan = build_model_plan(parse_plan(proposal.payload), source=proposal.source, reasons=proposal.reasons,
                                notes=proposal.notes)
        for note in plan.notes:
            logger.info("Model plan: {}", note)
        return plan

    def _ask_llm(
        self,
        requirement: UserRequirement,
        profile: DatasetProfile | None,
        knowledge: KnowledgeBrief | None,
        feedback: list[str],
    ) -> tuple[ModelChoice | None, str | None]:
        """(choice, None) on success; (None, reason) if the LLM's output was invalid; (None, None) if unavailable."""
        if self.llm_client is None:
            return None, None
        task = requirement.problem.task_type
        allowed = "\n".join(f"- {c.name}: {c.description}" for c in model_zoo.candidates_for(task))
        try:
            messages = render_prompt(
                "model_selection",
                requirement_json=format_requirement(requirement),
                dataset_context=format_profile(profile),
                allowed_models=allowed,
                constraints=data_lines(requirement.problem.constraints or ["none"]),
                knowledge=data_lines(knowledge.insights if knowledge and knowledge.insights else ["None available."]),
                feedback=data_lines(feedback or ["None (first attempt)."]),
            )
            return self.llm_client.chat_structured(messages, ModelChoice), None
        except LLMValidationError as exc:  # malformed output after the client's retries: reject, never guess
            logger.warning("Model selection output rejected: {}", exc)
            return None, f"The model-selection output did not match the plan schema: {exc.errors or exc}"
        except Exception as exc:  # unavailable LLM: the deterministic default ranking is a safe proposal
            logger.warning("Model selection by LLM failed ({}: {}); using the default ranking.", type(exc).__name__, exc)
            return None, None

    @staticmethod
    def _rank(
        choice: ModelChoice | None, requirement: UserRequirement, notes: list[str]
    ) -> tuple[list[str], dict[str, str], str]:
        """User-requested models first, then the LLM's ranking, then the default ranking (up to MAX_CANDIDATES).

        Names are only canonicalised (case, spaces, known aliases); unknown or repeated names stay in the list so
        the plan contract rejects them.
        """
        family = model_zoo.task_family(requirement.problem.task_type)
        items = choice.candidates if choice else []
        llm_names = [model_zoo.normalize_model_name(item.name) for item in items]
        reasons: dict[str, str] = {}
        for item, name in zip(items, llm_names, strict=True):
            reasons.setdefault(name, item.reason or "Chosen by the model agent.")
        source = "llm" if llm_names else "default"
        if choice is not None and not llm_names:
            notes.append("The LLM chose no model; using the default ranking.")

        requested = list(dict.fromkeys(model_zoo.normalize_model_name(m.name) for m in requirement.models if m.name))
        reasons |= {name: "Requested by the user." for name in requested}
        ordered = requested + [name for name in llm_names if name not in requested]
        for name in model_zoo.DEFAULT_RANKING[family]:
            if len(ordered) >= MAX_CANDIDATES:
                break
            if name not in ordered:
                ordered.append(name)
        return ordered, reasons, source


def parse_constraint_values(texts: list[str]) -> dict[str, Any]:
    """Inference-time, training-time and explainability limits from free-text constraints, as PlanConstraints fields."""
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
    return {"max_inference_ms_per_row": inference_ms, "max_train_minutes": None if train_s is None else train_s / 60,
            "prefer_interpretable": interpretable}


def parse_constraints(texts: list[str]) -> ModelConstraints:
    """The same limits as ModelConstraints (training time in seconds)."""
    values = parse_constraint_values(texts)
    minutes = values["max_train_minutes"]
    return ModelConstraints(max_inference_ms_per_row=values["max_inference_ms_per_row"],
                            max_train_time_s=None if minutes is None else round(minutes * 60, 6),
                            prefer_interpretable=values["prefer_interpretable"])


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

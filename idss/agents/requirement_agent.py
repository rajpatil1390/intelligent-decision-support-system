"""Requirement Agent: turn a business question into a validated, repaired UserRequirement.

The LLM interprets free text; everything that can be checked against the data is then verified and
repaired deterministically (repair), so a confused LLM can never send an impossible target or task
downstream. Optional LLM steps (relevance, sufficiency, questions, summary) fall back to safe defaults.
"""

from typing import NamedTuple

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient, LLMError
from idss.core.logger import get_logger
from idss.core.prompts import format_profile, format_requirement, render_prompt
from idss.core.schemas import (
    TASK_DESCRIPTIONS,
    DatasetProfile,
    DatasetSpec,
    PerformanceMetric,
    ProblemSpec,
    TaskType,
    UserRequirement,
)
from idss.ml.evaluator import METRIC_FAMILY_BY_TASK, METRICS, normalize_metric_name
from idss.ml.profiler import task_type_for_target

logger = get_logger()

TASK_METRICS: dict[str, list[str]] = {
    "tabular_classification": ["f1"],
    "text_classification": ["f1"],
    "tabular_regression": ["rmse"],
    "time_series_forecasting": ["rmse"],
    # Chance-corrected ARI when ground truth exists (plain Rand index rewards over-splitting), else silhouette.
    "tabular_clustering": ["adjusted_rand_index", "silhouette"],
}
_TARGET_TYPES = frozenset({"numeric", "categorical", "boolean"})

MIN_CONFIDENCE = 0.4
MAX_QUESTIONS = 3
MISSING_GOAL = "what you want to achieve"
MISSING_TARGET = "which column to predict"
MISSING_CLARITY = "a clearer description of the business goal"
_FALLBACK_QUESTIONS = {
    MISSING_GOAL: "What business decision or outcome should this analysis help you with?",
    MISSING_TARGET: "Which column in your data should we predict?",
    MISSING_CLARITY: "Could you describe in a sentence or two what you want to find out from this data?",
}


class RelevanceCheck(BaseModel):
    relevant: bool
    reason: str = ""


class SufficiencyCheck(BaseModel):
    sufficient: bool
    missing: list[str] = Field(default_factory=list)


class ClarifyingQuestions(BaseModel):
    questions: list[str] = Field(min_length=1)


class RequirementSummary(BaseModel):
    summary: str = Field(min_length=1)


class Sufficiency(NamedTuple):
    ok: bool
    missing: list[str]
    questions: list[str]


class RequirementAgent:
    """Parses, repairs, checks and summarises user requirements."""

    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    # ------------------------------------------------------------------ relevance

    def check_relevance(self, text: str) -> RelevanceCheck:
        """Classify `text` as a data/decision request or chit-chat. Fails open (relevant) if the LLM is unavailable."""
        try:
            return self._llm().chat_structured(render_prompt("relevance_check", question=text), RelevanceCheck)
        except LLMError as exc:
            logger.warning("Relevance check failed ({}); assuming the request is relevant.", exc)
            return RelevanceCheck(relevant=True, reason="Relevance could not be checked automatically.")

    def is_relevant(self, text: str) -> bool:
        return self.check_relevance(text).relevant

    # ------------------------------------------------------------------ parsing

    def parse(self, text: str, profile: DatasetProfile | None) -> UserRequirement:
        """Parse `text` with the LLM, then repair it against `profile`. Raises LLMError if the LLM fails."""
        messages = render_prompt("requirement_parse", question=text, dataset_context=format_profile(profile))
        requirement = self._llm().chat_structured(messages, UserRequirement)
        return self.repair(requirement, profile)

    def repair(self, requirement: UserRequirement, profile: DatasetProfile | None) -> UserRequirement:
        """Fix target, task type and metrics without the LLM. Each correction is recorded in system_notes."""
        req = requirement.model_copy(deep=True)
        notes: list[str] = []
        if profile is not None and req.problem.task_type != "tabular_clustering":
            self._repair_target(req, profile, notes)
            self._repair_task_type(req, profile, notes)
        self._repair_metrics(req, notes)
        req.system_notes = notes
        for note in notes:
            logger.info("Requirement repaired: {}", note)
        return req

    @staticmethod
    def _repair_target(req: UserRequirement, profile: DatasetProfile, notes: list[str]) -> None:
        requested = req.target
        usable = {col.name: col for col in profile.columns if col.semantic_type in _TARGET_TYPES and col.n_unique > 1}
        resolved = _match_name(requested, usable) if requested else None
        if resolved is None:
            fallback = profile.suggested_target
            if requested:
                known = _match_name(requested, {col.name: col for col in profile.columns})
                reason = (
                    f"is {_article(profile.column(known).semantic_type)} column and cannot be predicted"
                    if known else "is not in the dataset"
                )
                instead = f"using the suggested target '{fallback}' instead" if fallback else "no other usable target was found"
                notes.append(f"Target column '{requested}' {reason}; {instead}.")
            elif fallback:
                notes.append(f"No target column was given; using the suggested target '{fallback}'.")
            resolved = fallback
        _set_target(req, resolved)

    @staticmethod
    def _repair_task_type(req: UserRequirement, profile: DatasetProfile, notes: list[str]) -> None:
        target = req.target
        expected = task_type_for_target(profile, target) if target else None
        if expected and expected != req.problem.task_type:
            notes.append(
                f"Task type '{req.problem.task_type}' does not match the data for '{target}'; using '{expected}' instead."
            )
            req.problem.task_type = expected

    @staticmethod
    def _repair_metrics(req: UserRequirement, notes: list[str]) -> None:
        family = METRIC_FAMILY_BY_TASK[req.problem.task_type]
        kept: list[PerformanceMetric] = []
        for metric in req.problem.performance_metrics:
            name = normalize_metric_name(metric.name)
            spec = METRICS.get(name)
            if spec is not None and spec.family != family:
                notes.append(f"Metric '{metric.name}' does not apply to {family}; removed.")
                continue
            kept.append(metric.model_copy(update={"name": name}))
        if not kept:
            defaults = TASK_METRICS[req.problem.task_type]
            kept = [PerformanceMetric(name=name) for name in defaults]
            notes.append(f"No suitable success metric was given; using {', '.join(defaults)}.")
        req.problem.performance_metrics = kept

    # ------------------------------------------------------------------ sufficiency

    def check_sufficiency(self, req: UserRequirement, profile: DatasetProfile | None = None) -> Sufficiency:
        """Decide whether work can start. Deterministic checks first, then an optional LLM review."""
        missing: list[str] = []
        if req.user.intent == "unclear":
            missing.append(MISSING_GOAL)
        if req.problem.task_type != "tabular_clustering" and not req.target:
            missing.append(MISSING_TARGET)
        if req.confidence < MIN_CONFIDENCE:
            missing.append(MISSING_CLARITY)

        try:
            review = self._llm().chat_structured(
                render_prompt(
                    "sufficiency_check",
                    requirement_json=_requirement_json(req),
                    dataset_context=format_profile(profile),
                ),
                SufficiencyCheck,
            )
            if not review.sufficient:
                reported = [item.strip() for item in review.missing if item.strip()] or [MISSING_CLARITY]
                missing += [item for item in reported if item not in missing]
        except LLMError as exc:
            logger.warning("LLM sufficiency review failed ({}); using rule-based checks only.", exc)

        if not missing:
            return Sufficiency(ok=True, missing=[], questions=[])
        return Sufficiency(ok=False, missing=missing, questions=self._questions(req, missing, profile))

    def _questions(self, req: UserRequirement, missing: list[str], profile: DatasetProfile | None) -> list[str]:
        try:
            result = self._llm().chat_structured(
                render_prompt(
                    "clarifying_questions",
                    question=req.problem.description,
                    missing="\n".join(f"- {item}" for item in missing),
                    dataset_context=format_profile(profile),
                ),
                ClarifyingQuestions,
            )
            questions = [q.strip() for q in result.questions if q.strip()][:MAX_QUESTIONS]
            if questions:
                return questions
        except LLMError as exc:
            logger.warning("Generating clarifying questions failed ({}); using template questions.", exc)
        return [_FALLBACK_QUESTIONS.get(item, f"Could you tell us more about {item}?") for item in missing][:MAX_QUESTIONS]

    # ------------------------------------------------------------------ summary

    def summarize(self, req: UserRequirement) -> str:
        """Plain-English paragraph describing the requirement. Never raises: falls back to a template."""
        try:
            result = self._llm().chat_structured(
                render_prompt("requirement_summary", requirement_json=_requirement_json(req)), RequirementSummary
            )
            return result.summary.strip()
        except Exception as exc:  # the summary is cosmetic; never let it break the run
            logger.warning("Requirement summary failed ({}: {}); using the template summary.", type(exc).__name__, exc)
            return template_summary(req)

    def _llm(self) -> LLMClient:
        if self.llm_client is None:
            raise LLMError("No LLM client is configured.")
        return self.llm_client


def requirement_from_profile(question: str, profile: DatasetProfile) -> UserRequirement:
    """What the parse would produce without an LLM: the profile's suggested task and target, then repaired.

    Used when no LLM is configured, so a whole run can still work offline.
    """
    task_type = profile.suggested_task_type or "tabular_clustering"
    requirement = UserRequirement(
        problem=ProblemSpec(task_type=task_type, description=question.strip() or "Analyse the dataset."),
        datasets=[DatasetSpec(target_column=profile.suggested_target)],
        confidence=1.0,
    )
    return RequirementAgent(None).repair(requirement, profile)


def align_requirement(
    requirement: UserRequirement, task_type: TaskType, target: str | None, actionable_features: list[str]
) -> UserRequirement:
    """Copy of `requirement` that matches a fallback plan (v1.1), so the verifier, recommendations and report
    describe what was actually trained. Metrics of another task family are replaced by that task's defaults."""
    req = requirement.model_copy(deep=True)
    notes = [f"The proposed plan was rejected; following the fallback plan ({task_type}, target '{target}')."]
    req.problem.task_type = task_type
    _set_target(req, target)
    req.actionable_features = list(actionable_features)
    RequirementAgent._repair_metrics(req, notes)
    req.system_notes = [*req.system_notes, *notes]
    return req


def template_summary(req: UserRequirement) -> str:
    """Deterministic requirement summary used when the LLM is unavailable."""
    problem = req.problem
    parts = [f"Goal: {problem.description.rstrip('.')}."]
    if req.target:
        parts.append(f"We will predict '{req.target}', which is {TASK_DESCRIPTIONS[problem.task_type]}.")
    else:
        parts.append(f"This is {TASK_DESCRIPTIONS[problem.task_type]}.")
    if problem.performance_metrics:
        metrics = ", ".join(
            m.name if m.target is None else f"{m.name} (target {m.target:g})" for m in problem.performance_metrics
        )
        parts.append(f"Success will be measured by {metrics}.")
    if problem.constraints:
        parts.append(f"Constraints: {'; '.join(problem.constraints)}.")
    return " ".join(parts)


def _requirement_json(req: UserRequirement) -> str:
    return format_requirement(req)  # cleaned, in a <data> block, no server paths


def _set_target(req: UserRequirement, target: str | None) -> None:
    if req.datasets:
        req.datasets[0].target_column = target
    elif target:
        req.datasets.append(DatasetSpec(source="upload", target_column=target))


def _match_name[T](name: str, candidates: dict[str, T]) -> str | None:
    """Exact match first, then case/whitespace-insensitive."""
    if name in candidates:
        return name
    wanted = name.strip().lower()
    matches = [candidate for candidate in candidates if candidate.strip().lower() == wanted]
    return matches[0] if len(matches) == 1 else None


def _article(word: str) -> str:
    return f"an {word}" if word[:1] in "aeiou" else f"a {word}"

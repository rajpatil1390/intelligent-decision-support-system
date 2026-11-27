"""Orchestrator: a resumable state machine that drives one run from question to report.

Our own design, not the reference AgentManager: every state has exactly one handler that does its
work and returns (next_state, message). After every transition the whole RunContext is written to
workspace/runs/<run_id>/state.json, so a run survives restarts and can wait for the user's answers.
Loops are bounded: at most `max_revisions` clarification rounds, `MAX_PLAN_REVISIONS` re-plans after a
failed verification, and `MAX_STEPS` steps per call.
PROFILE also builds a KnowledgeBrief; PLAN asks the ModelAgent for a plan proposal and has it checked by the
independent plan validator (a rejected proposal is replaced by a fallback plan built from the data profile, and
the reasons are kept for the Run Monitor); TRAIN starts only from a ValidatedPlan sealed by that validator (the
transition into TRAIN is refused otherwise) and runs the AutoMLTrainer; VERIFY runs the Verifier; EXPLAIN computes
drivers, recommendations and a what-if example; INSIGHT writes the grounded decision brief; REPORT saves report.pdf and report.md. `ask()` answers ad-hoc
questions through the sandboxed AnalysisAgent and `what_if()` simulates changes with the trained model.
Every transition is also mirrored into the SQLite database (runs + events tables) for listing; a database
problem is logged and never stops a run, because state.json is the source of truth.
"""

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field, computed_field

from configs.settings import get_settings
from idss.agents.analysis_agent import AnalysisAgent, AnalysisAnswer
from idss.agents.data_agent import DataAgent, DataAgentResult
from idss.agents.insight_agent import InsightAgent
from idss.agents.knowledge_agent import KnowledgeAgent
from idss.agents.model_agent import ModelAgent
from idss.agents.requirement_agent import RequirementAgent, align_requirement
from idss.agents.verifier import Verifier
from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.schemas import (
    BudgetMode,
    DatasetProfile,
    DatasetSpec,
    DecisionInsight,
    ExplanationResult,
    FilterCondition,
    KnowledgeBrief,
    ModelPlan,
    RecommendationResult,
    SegmentImpact,
    TrainingBudget,
    TrainingResult,
    UserRequirement,
    VerificationReport,
    WhatIfResult,
)
from idss.decision.facts import build_registry
from idss.decision.recommender import example_what_if, recommend
from idss.decision.report_builder import build_markdown, build_pdf
from idss.decision.what_if import simulate, simulate_segment
from idss.execution.plan_validator import (
    PlanRejected,
    PlanValidationError,
    ValidatedPlan,
    revalidate,
    validate_or_fallback,
)
from idss.ml import model_zoo
from idss.ml.data_loader import load_dataset
from idss.ml.explain import explain_run
from idss.ml.profiler import parse_datetimes, profile_dataset
from idss.ml.trainer import AutoMLTrainer, TrainedModel
from idss.storage.repository import EventRecord, Repository, RunRecord, get_repository
from idss.storage.runs import new_run_id, run_dir

logger = get_logger()

MAX_REVISIONS = 2
MAX_PLAN_REVISIONS = 1  # re-plans with the full budget after a failed verification
MAX_STEPS = 50
STATE_FILENAME = "state.json"
FILE_RETRIES = 3  # Windows: replacing/reading state.json can briefly fail while another thread has it open
FILE_RETRY_BASE_S = 0.05

NOT_RELEVANT_MESSAGE = (
    "Thanks for your message! I can help with business questions that can be answered from your data, "
    "for example predicting which customers will churn, forecasting sales, or estimating prices. "
    "Could you describe the decision you would like support with?"
)
REPORT_PDF = "report.pdf"
REPORT_MARKDOWN = "report.md"


class PlanNotValidatedError(RuntimeError):
    """TRAIN was asked to run without a plan accepted by the plan validator."""


class RunState(StrEnum):
    INIT = "INIT"
    PARSE = "PARSE"
    VALIDATE = "VALIDATE"
    PROFILE = "PROFILE"
    PLAN = "PLAN"
    TRAIN = "TRAIN"
    VERIFY = "VERIFY"
    EXPLAIN = "EXPLAIN"
    INSIGHT = "INSIGHT"
    REPORT = "REPORT"
    DONE = "DONE"
    FAILED = "FAILED"
    NEEDS_INPUT = "NEEDS_INPUT"


TERMINAL_STATES = frozenset({RunState.DONE, RunState.FAILED})
BLOCKED_STATES = TERMINAL_STATES | {RunState.NEEDS_INPUT}


class RunEvent(BaseModel):
    """One transition: `message` describes the work done in `from_state` that led to `state`."""

    timestamp: datetime
    from_state: RunState | None = None
    state: RunState
    message: str


class RunContext(BaseModel):
    """Everything about a run. Persisted as state.json after every transition."""

    run_id: str
    state: RunState = RunState.INIT
    question: str
    data_path: str
    dataset_id: str | None = Field(default=None, description="Uploaded dataset id (API runs).")
    actionable_features: list[str] = Field(
        default_factory=list, description="Columns the user says the business can change; override the parsed list."
    )
    budget_mode: BudgetMode | None = Field(default=None, description="fast/full chosen for this run (API runs).")
    requirement: UserRequirement | None = None
    profile: DatasetProfile | None = None
    summary: str | None = None
    data: DataAgentResult | None = None
    knowledge: KnowledgeBrief | None = None
    validated_plan: ValidatedPlan | None = Field(
        default=None, description="The plan TRAIN runs: accepted by the plan validator, or its fallback plan."
    )
    training: TrainingResult | None = None
    verification: VerificationReport | None = None
    explanation: ExplanationResult | None = None
    recommendations: RecommendationResult | None = None
    what_if_example: WhatIfResult | None = None
    insight: DecisionInsight | None = None
    report_paths: dict[str, str] = Field(default_factory=dict)
    requirement_provided: bool = Field(default=False, description="Requirement given to start(); PARSE only repairs it.")
    pending_questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    revision_count: int = 0
    plan_revision_count: int = 0
    analyses: list[AnalysisAnswer] = Field(default_factory=list)
    what_ifs: list[WhatIfResult | SegmentImpact] = Field(default_factory=list)
    history: list[RunEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    message: str | None = Field(default=None, description="Final user-facing message.")
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def plan(self) -> ModelPlan | None:
        """The engine's ModelPlan of the validated plan (read-only: a plan can only be set through the validator)."""
        return self.validated_plan.model_plan if self.validated_plan is not None else None


Handler = Callable[[RunContext], tuple[RunState, str]]


class Orchestrator:
    """Runs the IDSS pipeline as a persisted state machine."""

    def __init__(
        self,
        llm_client: LLMClient | None,
        run_dir_root: Path | None = None,
        *,
        max_revisions: int = MAX_REVISIONS,
        budget_mode: BudgetMode | None = None,
        budget: TrainingBudget | None = None,
        repository: Repository | None = None,
        fast_only: bool = False,
        max_explain_rows: int | None = None,
    ) -> None:
        """`budget_mode` overrides the Model Agent's fast/full choice; `budget` overrides the budget entirely.
        `repository` is where transitions are mirrored (default: the workspace database). `fast_only` (public
        demo) never uses the full budget, not even for the re-plan after a failed verification.
        `max_explain_rows` caps the SHAP sample in EXPLAIN (None = MAX_EXPLAIN_ROWS of idss.ml.explain; the public
        demo passes a smaller one to save memory)."""
        self.run_dir_root = run_dir_root if run_dir_root is not None else get_settings().runs_dir
        self.repository = repository
        self.max_revisions = max_revisions
        self.budget_mode = budget_mode
        self.budget = budget
        self.fast_only = fast_only
        self.max_explain_rows = max_explain_rows
        self.requirement_agent = RequirementAgent(llm_client)
        self.data_agent = DataAgent(llm_client, runs_dir=self.run_dir_root)
        self.knowledge_agent = KnowledgeAgent(llm_client)
        self.model_agent = ModelAgent(llm_client)
        self.trainer = AutoMLTrainer()
        self.verifier = Verifier(llm_client)
        self.analysis_agent = AnalysisAgent(llm_client)
        self.insight_agent = InsightAgent(llm_client)
        self.context: RunContext | None = None
        self._handlers: dict[RunState, Handler] = {
            RunState.INIT: self._handle_init,
            RunState.PARSE: self._handle_parse,
            RunState.VALIDATE: self._handle_validate,
            RunState.PROFILE: self._handle_profile,
            RunState.PLAN: self._handle_plan,
            RunState.TRAIN: self._handle_train,
            RunState.VERIFY: self._handle_verify,
            RunState.EXPLAIN: self._handle_explain,
            RunState.INSIGHT: self._handle_insight,
            RunState.REPORT: self._handle_report,
        }

    # ------------------------------------------------------------------ public API

    def start(
        self,
        question: str,
        data_path: str | Path,
        *,
        requirement: UserRequirement | None = None,
        dataset_id: str | None = None,
        actionable_features: list[str] | None = None,
        budget_mode: BudgetMode | None = None,
    ) -> str:
        """Create a new run in state INIT and return its id. Call step() or run_until_blocked() to advance.

        Passing a ready-made `requirement` skips the LLM parse (it is still repaired against the data),
        which lets a whole run work offline without an LLM. `actionable_features` (if given) replaces the
        requirement's list after parsing; `budget_mode` (if given) is this run's fast/full choice.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question must not be empty.")
        now = datetime.now(UTC)
        ctx = RunContext(run_id=new_run_id(), question=question, data_path=str(data_path), dataset_id=dataset_id,
                         actionable_features=[f.strip() for f in actionable_features or [] if f.strip()],
                         budget_mode=budget_mode, created_at=now, updated_at=now, requirement=requirement,
                         requirement_provided=requirement is not None)
        ctx.history.append(RunEvent(timestamp=now, state=RunState.INIT, message="Run created."))
        self._save(ctx)
        self._record(ctx, ctx.history[-1])
        self.context = ctx
        logger.info("Started run {} for data '{}'.", ctx.run_id, ctx.data_path)
        return ctx.run_id

    def step(self, run_id: str | None = None) -> RunState:
        """Run the current state's handler once and move to the state it returns."""
        ctx = self._get(run_id)
        if ctx.state in BLOCKED_STATES:
            return ctx.state
        state = ctx.state
        try:
            next_state, message = self._handlers[state](ctx)
        except Exception as exc:  # any failure ends the run cleanly instead of crashing the caller
            logger.exception("Run {} failed in state {}.", ctx.run_id, state)
            message = f"{state} failed: {type(exc).__name__}: {exc}"
            ctx.errors.append(message)
            ctx.message = f"Sorry, the run stopped during the {state.lower()} stage: {exc}"
            next_state = RunState.FAILED
        self._transition(ctx, next_state, message)
        return next_state

    def run_until_blocked(self, run_id: str | None = None, *, max_steps: int = MAX_STEPS) -> RunContext:
        """Step until the run is DONE, FAILED or NEEDS_INPUT (or fail it after `max_steps` steps)."""
        ctx = self._get(run_id)
        for _ in range(max_steps):
            if ctx.state in BLOCKED_STATES:
                return ctx
            self.step(ctx.run_id)
        if ctx.state not in BLOCKED_STATES:
            message = f"Stopped after {max_steps} steps without finishing (possible loop)."
            ctx.errors.append(message)
            ctx.message = "Sorry, the run did not finish. Please try again."
            self._transition(ctx, RunState.FAILED, message)
        return ctx

    def provide_answer(self, run_id: str, answer: str) -> RunState:
        """Merge the user's answer into the question and re-enter PARSE (counts as one revision)."""
        ctx = self._get(run_id)
        if ctx.state != RunState.NEEDS_INPUT:
            raise ValueError(f"Run {run_id} is in state {ctx.state}, not waiting for input.")
        if not answer.strip():
            raise ValueError("The answer must not be empty.")
        ctx.answers.append(answer.strip())
        ctx.question = _merge_answer(ctx.question, ctx.pending_questions, answer)
        ctx.pending_questions = []
        ctx.revision_count += 1
        self._transition(ctx, RunState.PARSE, f"User answered (revision {ctx.revision_count}/{self.max_revisions}).")
        return ctx.state

    def ask(self, run_id: str, question: str) -> AnalysisAnswer:
        """Answer an ad-hoc question about the run's data with sandboxed, LLM-written pandas code."""
        ctx = self._get(run_id)
        if not question.strip():
            raise ValueError("The question must not be empty.")
        profile = ctx.profile or profile_dataset(load_dataset(ctx.data_path))
        answer = self.analysis_agent.ask(question.strip(), profile, ctx.data_path)
        ctx.analyses.append(answer)
        ctx.updated_at = datetime.now(UTC)
        self._save(ctx)
        self._record(ctx)
        return answer

    def what_if(
        self,
        run_id: str,
        changes: dict[str, Any],
        *,
        base_row: dict[str, Any] | None = None,
        segment: list[FilterCondition | dict[str, Any]] | None = None,
    ) -> WhatIfResult | SegmentImpact:
        """Simulate `changes` with the run's trained model, for one row (`base_row`) or a filtered segment."""
        if (base_row is None) == (segment is None):
            raise ValueError("Give exactly one of base_row or segment.")
        ctx = self._get(run_id)
        training, profile = _required(ctx.training, "trained model"), _required(ctx.profile, "profile")
        model: TrainedModel = joblib.load(training.model_path)  # our own artifact from TRAIN
        data = load_dataset(ctx.data_path)
        if base_row is not None:
            history = None
            if model_zoo.task_family(model.task_type) == "time_series":
                history = data.sort_values(model.pipeline.named_steps["ts_features"].time_column,
                                           key=parse_datetimes, kind="stable")
            result: WhatIfResult | SegmentImpact = simulate(model, base_row, changes, profile, history=history)
        else:
            result = simulate_segment(model, data, segment or [], changes, profile)
        ctx.what_ifs.append(result)
        ctx.updated_at = datetime.now(UTC)
        self._save(ctx)
        self._record(ctx)
        return result

    def resume(self, run_id: str) -> RunContext:
        """Load a saved run from state.json and make it the active run."""
        self.context = load_context(run_id, self.run_dir_root)
        logger.info("Resumed run {} in state {}.", run_id, self.context.state)
        return self.context

    def abort(self, run_id: str, reason: str) -> RunState:
        """Move an unfinished run to FAILED (e.g. the process running it was restarted). Blocked runs are unchanged."""
        ctx = self._get(run_id)
        if ctx.state in BLOCKED_STATES:
            return ctx.state
        ctx.errors.append(reason)
        ctx.message = f"Sorry, the run stopped: {reason}"
        self._transition(ctx, RunState.FAILED, reason)
        return ctx.state

    # ------------------------------------------------------------------ state handlers

    def _handle_init(self, ctx: RunContext) -> tuple[RunState, str]:
        relevance = self.requirement_agent.check_relevance(ctx.question)
        if not relevance.relevant:
            ctx.message = NOT_RELEVANT_MESSAGE
            return RunState.DONE, f"Not a data or decision request ({relevance.reason}); nothing to train."
        ctx.profile = profile_dataset(load_dataset(ctx.data_path))
        return RunState.PARSE, f"Request is relevant. Profiled data: {ctx.profile.n_rows:,} rows x {ctx.profile.n_cols} columns."

    def _handle_parse(self, ctx: RunContext) -> tuple[RunState, str]:
        if ctx.requirement_provided and ctx.requirement is not None and ctx.revision_count == 0:
            requirement = self.requirement_agent.repair(ctx.requirement, ctx.profile)  # given requirement: no LLM parse
        else:
            requirement = self.requirement_agent.parse(ctx.question, ctx.profile)
        _attach_dataset(requirement, ctx.data_path)
        if ctx.actionable_features:
            requirement.actionable_features = list(ctx.actionable_features)
        ctx.requirement = requirement
        message = (
            f"Parsed requirement: task={requirement.problem.task_type}, target={requirement.target}, "
            f"confidence={requirement.confidence:.2f}."
        )
        if requirement.system_notes:
            message += " Corrections: " + " ".join(requirement.system_notes)
        return RunState.VALIDATE, message

    def _handle_validate(self, ctx: RunContext) -> tuple[RunState, str]:
        requirement = _required(ctx.requirement, "requirement")
        result = self.requirement_agent.check_sufficiency(requirement, ctx.profile)
        if not result.ok:
            missing = "; ".join(result.missing)
            if ctx.revision_count >= self.max_revisions:
                ctx.message = (
                    f"We still do not have enough information to start after {ctx.revision_count} round(s) "
                    f"of questions. Missing: {missing}."
                )
                return RunState.FAILED, f"Requirement still incomplete after {ctx.revision_count} revision(s): {missing}."
            ctx.pending_questions = result.questions
            return RunState.NEEDS_INPUT, f"Need more information: {missing}."
        ctx.pending_questions = []
        ctx.summary = self.requirement_agent.summarize(requirement)
        return RunState.PROFILE, "Requirement is complete."

    def _handle_profile(self, ctx: RunContext) -> tuple[RunState, str]:
        requirement = _required(ctx.requirement, "requirement")
        result = self.data_agent.run(ctx.data_path, requirement, run_id=ctx.run_id)
        ctx.data = result
        ctx.profile = result.profile
        ctx.knowledge = self.knowledge_agent.brief(requirement, result.profile)
        sizes = result.split_sizes
        return RunState.PLAN, (
            f"Prepared data: train/valid/test = {sizes.train:,}/{sizes.valid:,}/{sizes.test:,} rows. "
            f"Knowledge brief: {len(ctx.knowledge.insights)} insight(s) from {len(ctx.knowledge.sources)} source(s)."
        )

    def _handle_plan(self, ctx: RunContext) -> tuple[RunState, str]:
        revising = ctx.plan_revision_count > 0
        feedback = ctx.verification.recommendations if revising and ctx.verification else None
        requirement, profile = _required(ctx.requirement, "requirement"), _required(ctx.profile, "profile")
        proposal = self.model_agent.propose(
            requirement, profile, budget_mode=self._budget_mode(ctx, revising), knowledge=ctx.knowledge, feedback=feedback
        )
        try:
            validated = validate_or_fallback(proposal, load_dataset(ctx.data_path), profile, get_settings(),
                                             **self._plan_options(ctx))
        except PlanValidationError as exc:
            ctx.message = f"Sorry, no safe analysis plan could be built for this data: {exc}"
            ctx.errors.append(f"PLAN failed: no valid plan ({exc}).")
            return RunState.FAILED, f"No valid plan could be built: {exc}"
        ctx.validated_plan = validated
        plan = validated.model_plan
        names = ", ".join(c.name for c in plan.candidates[: plan.budget.max_candidates])
        message = (
            f"{'Revised plan' if revising else 'Plan'} ({plan.source}): optimise {plan.primary_metric} with {names} "
            f"({plan.budget_mode} mode, {plan.budget.n_trials_per_model} trials each)."
        )
        if validated.fallback:
            ctx.requirement = align_requirement(requirement, validated.analysis.task_type, validated.analysis.target,
                                                validated.analysis.actionable_features)
            message = (f"Proposed plan rejected ({' '.join(validated.rejected_reasons)}) Using a fallback plan built "
                       f"from the data profile: {plan.task_type} on '{plan.target}'. {message}")
        return RunState.TRAIN, message

    def _plan_options(self, ctx: RunContext) -> dict[str, Any]:
        """This orchestrator's inputs to the plan validator (the hard limits come from the settings)."""
        return {"budget": self.budget, "public_demo": True if self.fast_only else None, "data_path": ctx.data_path}

    def _train_refusal(self, ctx: RunContext) -> str | None:
        """Why TRAIN may not start, or None. Only a ValidatedPlan sealed by the validator in this process passes
        directly; any other ValidatedPlan (read back from state.json, built by hand, changed since) is validated
        again against the data first."""
        validated = ctx.validated_plan
        if not isinstance(validated, ValidatedPlan):
            return "Refused to start TRAIN: the run has no validated plan."
        if validated.sealed:
            return None
        try:
            result = revalidate(validated, load_dataset(ctx.data_path), _required(ctx.profile, "profile"),
                                get_settings(), **self._plan_options(ctx))
        except Exception as exc:  # the guard itself must never crash the state machine
            return f"Refused to start TRAIN: the plan could not be validated ({type(exc).__name__}: {exc})."
        if isinstance(result, PlanRejected):
            return "Refused to start TRAIN: the stored plan failed validation: " + " ".join(result.reasons)
        ctx.validated_plan = result
        return None

    def _budget_mode(self, ctx: RunContext, revising: bool) -> BudgetMode | None:
        if self.fast_only:
            return "fast"
        return "full" if revising else (ctx.budget_mode or self.budget_mode)

    def _handle_train(self, ctx: RunContext) -> tuple[RunState, str]:
        if refusal := self._train_refusal(ctx):  # e.g. a run resumed in TRAIN from a tampered state.json
            raise PlanNotValidatedError(refusal)
        validated = _required(ctx.validated_plan, "validated plan")
        result = self.trainer.run(
            load_dataset(ctx.data_path),
            _required(ctx.requirement, "requirement"),
            _required(ctx.profile, "profile"),
            run_dir(ctx.run_id, self.run_dir_root),
            plan=validated.model_plan,
        )
        ctx.training = result
        trained = sum(1 for entry in result.leaderboard if entry.status == "ok")
        score = result.test_metrics.get(result.primary_metric)
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        return RunState.VERIFY, (
            f"Trained {trained} model(s) in {result.total_time_s:.1f}s; best: {result.best_model} "
            f"({result.primary_metric} = {score_text} on held-out test data)."
        )

    def _handle_verify(self, ctx: RunContext) -> tuple[RunState, str]:
        report = self.verifier.verify(
            _required(ctx.requirement, "requirement"), _required(ctx.training, "training"), ctx.profile
        )
        ctx.verification = report
        if report.passed:
            return RunState.EXPLAIN, f"Verification passed ({len(report.checks)} checks)."
        failed = ", ".join(report.failed_checks)
        if ctx.plan_revision_count < MAX_PLAN_REVISIONS:
            ctx.plan_revision_count += 1
            budget = "the fast budget (public demo)" if self.fast_only else "the full budget"
            return RunState.PLAN, (
                f"Verification failed ({failed}); re-planning once with {budget} and the verifier's recommendations."
            )
        # Never fail the whole run here: continue, with the results flagged as not verified.
        return RunState.EXPLAIN, f"Verification still failing ({failed}); continuing with results flagged as not verified."

    def _handle_explain(self, ctx: RunContext) -> tuple[RunState, str]:
        training, profile = _required(ctx.training, "training"), _required(ctx.profile, "profile")
        model: TrainedModel = joblib.load(training.model_path)  # our own artifact from TRAIN
        data = load_dataset(ctx.data_path)
        explanation, explainer = explain_run(model, data, training, profile, run_dir(ctx.run_id, self.run_dir_root),
                                             background=_training_rows(ctx), max_rows=self.max_explain_rows)
        ctx.explanation = explanation
        ctx.recommendations = recommend(model, data, ctx.requirement, profile, training, explanation, explainer,
                                        ctx.verification)
        ctx.what_if_example = example_what_if(model, data, profile, ctx.recommendations, explanation)
        drivers = ", ".join(f.feature for f in explanation.global_explanation.features[:3]) or "none"
        return RunState.INSIGHT, (
            f"Explained the model ({explanation.global_explanation.method}); top drivers: {drivers}; "
            f"{len(ctx.recommendations.recommendations)} recommendation(s)."
        )

    def _handle_insight(self, ctx: RunContext) -> tuple[RunState, str]:
        _required(ctx.requirement, "requirement")
        _required(ctx.training, "training")
        ctx.insight = self.insight_agent.generate(build_registry(ctx))  # numbers only through registry placeholders
        grounding = ctx.insight.grounding
        detail = "" if grounding is None else f" (grounding {'passed' if grounding.passed else 'failed'})"
        return RunState.REPORT, f"Decision brief written by {ctx.insight.generated_by}{detail}."

    def _handle_report(self, ctx: RunContext) -> tuple[RunState, str]:
        out_dir = run_dir(ctx.run_id, self.run_dir_root)
        charts = ctx.explanation.charts if ctx.explanation else {}
        pdf = build_pdf(ctx, ctx.insight, charts, out_dir / REPORT_PDF)
        markdown = build_markdown(ctx, ctx.insight, charts, out_dir / REPORT_MARKDOWN)
        ctx.report_paths = {"pdf": str(pdf), "markdown": str(markdown)}
        summary = ctx.insight.executive_summary if ctx.insight else "The analysis is complete."
        if ctx.verification is not None and not ctx.verification.passed:
            summary += (f" Note: the model did not pass all quality checks ({', '.join(ctx.verification.failed_checks)}); "
                        "treat its results with caution.")
        ctx.message = f"{summary} Full report: {pdf}"
        return RunState.DONE, f"Wrote {REPORT_PDF} and {REPORT_MARKDOWN}."

    # ------------------------------------------------------------------ persistence

    def _get(self, run_id: str | None) -> RunContext:
        if run_id is None:
            if self.context is None:
                raise RuntimeError("No active run: call start() or resume() first.")
            return self.context
        if self.context is not None and self.context.run_id == run_id:
            return self.context
        return self.resume(run_id)

    def _transition(self, ctx: RunContext, state: RunState, message: str) -> None:
        if state == RunState.TRAIN and (refusal := self._train_refusal(ctx)):
            logger.error("Run {}: {}", ctx.run_id, refusal)
            ctx.errors.append(refusal)
            ctx.message = "Sorry, the run stopped: training may only start from a validated plan."
            state, message = RunState.FAILED, refusal
        now = datetime.now(UTC)
        ctx.history.append(RunEvent(timestamp=now, from_state=ctx.state, state=state, message=message))
        ctx.state = state
        ctx.updated_at = now
        self._save(ctx)
        self._record(ctx, ctx.history[-1])
        logger.info("Run {} -> {}: {}", ctx.run_id, state, message)

    def _save(self, ctx: RunContext) -> None:
        path = run_dir(ctx.run_id, self.run_dir_root) / STATE_FILENAME
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
        _with_file_retries(lambda: tmp.replace(path))  # atomic: a crash never leaves a half-written state.json

    def _record(self, ctx: RunContext, event: RunEvent | None = None) -> None:
        """Mirror the run's summary (and its newest event) into the database; failures are only logged."""
        try:
            repository = self.repository if self.repository is not None else get_repository()
            record = None if event is None else EventRecord(
                run_id=ctx.run_id, ts=event.timestamp, from_state=event.from_state, to_state=event.state,
                message=event.message)
            repository.record_transition(run_record(ctx), record)
        except Exception as exc:  # the database is an index; state.json is the source of truth
            logger.warning("Could not record run {} in the database ({}: {}).", ctx.run_id, type(exc).__name__, exc)


def load_context(run_id: str, run_dir_root: Path | None = None) -> RunContext:
    """Read a run's state.json (FileNotFoundError if there is none, ValueError for an invalid run id)."""
    path = run_dir(run_id, run_dir_root, create=False) / STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No saved run '{run_id}' (expected {path}).")
    return RunContext.model_validate_json(_with_file_retries(lambda: path.read_text(encoding="utf-8")))


def run_record(ctx: RunContext) -> RunRecord:
    """The database summary row of a run."""
    training = ctx.training
    task = training.task_type if training else (ctx.requirement.problem.task_type if ctx.requirement else None)
    value = training.test_metrics.get(training.primary_metric) if training else None
    numeric = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    return RunRecord(
        run_id=ctx.run_id, dataset_id=ctx.dataset_id, question=ctx.question, state=ctx.state, task_type=task,
        best_model=training.best_model if training else None,
        primary_metric=training.primary_metric if training else None,
        metric_value=float(value) if numeric else None,
        verification_passed=ctx.verification.passed if ctx.verification else None,
        created_at=ctx.created_at, updated_at=ctx.updated_at,
        report_pdf=ctx.report_paths.get("pdf"), report_md=ctx.report_paths.get("markdown"),
    )


def _with_file_retries[T](action: Callable[[], T]) -> T:
    """Retry a file operation that failed because another thread briefly held the file (Windows sharing rules)."""
    for attempt in range(FILE_RETRIES):
        try:
            return action()
        except PermissionError:
            time.sleep(FILE_RETRY_BASE_S * 2 ** attempt)
    return action()  # last attempt: let the error propagate


def _training_rows(ctx: RunContext) -> pd.DataFrame | None:
    """Train + valid rows saved by the DataAgent: explanations describe the model on the data it learned from."""
    artifacts = ctx.data.artifacts if ctx.data else {}
    paths = [Path(artifacts[name]) for name in ("train", "valid") if name in artifacts]
    if len(paths) != 2 or not all(path.is_file() for path in paths):
        return None
    return pd.concat([pd.read_parquet(path) for path in paths])


def _attach_dataset(requirement: UserRequirement, data_path: str) -> None:
    """Record the uploaded file on the requirement's first dataset entry."""
    if not requirement.datasets:
        requirement.datasets.append(DatasetSpec())
    dataset = requirement.datasets[0]
    dataset.path = data_path
    dataset.source = "upload"
    dataset.name = dataset.name or Path(data_path).name


def _merge_answer(question: str, asked: list[str], answer: str) -> str:
    lines = [question, "", "Clarification from the user:"]
    lines += [f"Q: {q}" for q in asked]
    lines.append(f"A: {answer.strip()}")
    return "\n".join(lines)


def _required[T](value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"Run has no {name} yet; the state machine reached this state out of order.")
    return value

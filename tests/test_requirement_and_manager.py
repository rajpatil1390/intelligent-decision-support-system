"""Requirement Agent, prompts and Orchestrator tests. The LLM is always faked; no network calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from configs.settings import Settings
from idss.agents.manager import NOT_RELEVANT_MESSAGE, STATE_FILENAME, Orchestrator, RunState
from idss.agents.requirement_agent import (
    MISSING_CLARITY,
    MISSING_GOAL,
    ClarifyingQuestions,
    RelevanceCheck,
    RequirementAgent,
    RequirementSummary,
    SufficiencyCheck,
)
from idss.core.llm_client import LLMClient, LLMError
from idss.core.prompts import PROMPT_NAMES, format_profile, load_prompt, render_prompt
from idss.core.schemas import (
    DatasetProfile,
    DatasetSpec,
    ModelSpec,
    PerformanceMetric,
    ProblemSpec,
    TrainingBudget,
    UserInfo,
    UserRequirement,
)
from idss.ml.data_loader import load_dataset
from idss.ml.profiler import profile_dataset
from tests.fakes import FakeLLM, PassingVerifier

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
CHURN_CSV = SAMPLES_DIR / "customer_churn.csv"
CHURN_QUESTION = "Which customers are likely to churn and what should we do to retain them?"
PIPELINE_STATES = [
    "INIT", "PARSE", "VALIDATE", "PROFILE", "PLAN", "TRAIN", "VERIFY", "EXPLAIN", "INSIGHT", "REPORT", "DONE",
]
# TRAIN really trains now; keep it to one model and two trials so these state-machine tests stay fast.
TEST_BUDGET = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)


def make_orchestrator(llm: Any, run_dir_root: Path) -> Orchestrator:
    orchestrator = Orchestrator(llm, run_dir_root, budget=TEST_BUDGET)
    orchestrator.verifier = PassingVerifier()  # these tests cover the state machine; verification has its own tests
    return orchestrator


def make_requirement(
    task_type: str = "tabular_classification",
    target: str | None = "churn",
    metrics: tuple[str, ...] = ("f1",),
    intent: str = "build",
    confidence: float = 0.85,
    models: tuple[str, ...] = (),
) -> UserRequirement:
    return UserRequirement(
        user=UserInfo(intent=intent),
        problem=ProblemSpec(
            task_type=task_type,
            description="Predict which customers will churn so we can retain them.",
            performance_metrics=[PerformanceMetric(name=m) for m in metrics],
        ),
        datasets=[DatasetSpec(target_column=target)] if target else [],
        models=[ModelSpec(name=m) for m in models],
        confidence=confidence,
    )


def good_llm(**overrides: Any) -> FakeLLM:
    responses: dict[type[BaseModel], Any] = {
        RelevanceCheck: RelevanceCheck(relevant=True, reason="Churn prediction request."),
        UserRequirement: make_requirement(),
        SufficiencyCheck: SufficiencyCheck(sufficient=True),
        ClarifyingQuestions: ClarifyingQuestions(questions=["Which column should we predict?"]),
        RequirementSummary: RequirementSummary(summary="We will predict which customers are likely to churn."),
    }
    responses.update({globals()[name]: value for name, value in overrides.items()})
    return FakeLLM(responses)


@pytest.fixture(scope="module")
def churn_profile() -> DatasetProfile:
    return profile_dataset(load_dataset(CHURN_CSV))


# --------------------------------------------------------------------------- prompts


def test_all_prompts_load_and_demand_json() -> None:
    for name in PROMPT_NAMES:
        prompt = load_prompt(name)
        assert "JSON" in prompt.system.template, name


def test_prompts_include_profile_summary_but_never_raw_values(churn_profile: DatasetProfile) -> None:
    messages = render_prompt("requirement_parse", question=CHURN_QUESTION, dataset_context=format_profile(churn_profile))
    user = messages[1]["content"]
    assert CHURN_QUESTION in user
    assert "- customer_id | id | 0% | 3000" in user
    assert "Suggested target column: churn." in user and "Suggested task type: tabular_classification." in user
    for raw_value in ("CUST-00001", "Month-to-month", "Electronic check", "2019-01-02"):
        assert raw_value not in user


def test_render_prompt_requires_every_variable() -> None:
    with pytest.raises(KeyError):
        render_prompt("requirement_parse", question="only the question")


# --------------------------------------------------------------------------- relevance


@pytest.mark.parametrize("relevant", [True, False])
def test_is_relevant(relevant: bool) -> None:
    agent = RequirementAgent(good_llm(RelevanceCheck=RelevanceCheck(relevant=relevant, reason="r")))
    assert agent.is_relevant("some text") is relevant


def test_relevance_fails_open_when_llm_unavailable() -> None:
    assert RequirementAgent(FakeLLM({})).is_relevant("Predict churn") is True
    assert RequirementAgent(None).is_relevant("Predict churn") is True


def test_relevance_through_real_llm_client_with_mocked_api() -> None:
    api = MagicMock()
    api.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"relevant": false, "reason": "small talk"}'))],
        usage=None,
    )
    client = LLMClient(Settings(_env_file=None, llm_model="m", llm_api_key="k"), client=api)
    assert RequirementAgent(client).is_relevant("Hi there, how are you?") is False
    system_prompt = api.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "JSON Schema" in system_prompt and "intake assistant" in system_prompt


# --------------------------------------------------------------------------- parse + repair


def test_parse_keeps_a_valid_requirement(churn_profile: DatasetProfile) -> None:
    llm = good_llm()
    req = RequirementAgent(llm).parse(CHURN_QUESTION, churn_profile)
    assert (req.target, req.problem.task_type, req.system_notes) == ("churn", "tabular_classification", [])
    [(_, messages)] = [call for call in llm.calls if call[0] == "UserRequirement"]
    assert "customer_id | id" in messages[1]["content"]


@pytest.mark.parametrize(
    "bad_target, expected_note",
    [("will_leave", "is not in the dataset"), ("customer_id", "is an id column and cannot be predicted")],
)
def test_repair_replaces_unusable_target(churn_profile: DatasetProfile, bad_target: str, expected_note: str) -> None:
    agent = RequirementAgent(good_llm(UserRequirement=make_requirement(target=bad_target)))
    req = agent.parse(CHURN_QUESTION, churn_profile)
    assert req.target == "churn"
    assert any(expected_note in note and "suggested target 'churn'" in note for note in req.system_notes)


def test_repair_matches_target_case_insensitively(churn_profile: DatasetProfile) -> None:
    req = RequirementAgent(good_llm(UserRequirement=make_requirement(target=" CHURN "))).parse("q", churn_profile)
    assert req.target == "churn" and req.system_notes == []


def test_repair_fills_target_when_none_given(churn_profile: DatasetProfile) -> None:
    req = RequirementAgent(good_llm(UserRequirement=make_requirement(target=None))).parse("q", churn_profile)
    assert req.target == "churn" and "No target column was given" in req.system_notes[0]


def test_repair_prefers_profile_task_type_and_fixes_metrics(churn_profile: DatasetProfile) -> None:
    llm_output = make_requirement(task_type="tabular_regression", metrics=("rmse",))
    req = RequirementAgent(good_llm(UserRequirement=llm_output)).parse(CHURN_QUESTION, churn_profile)
    assert req.problem.task_type == "tabular_classification"
    assert [m.name for m in req.problem.performance_metrics] == ["f1"]
    notes = " ".join(req.system_notes)
    assert "'tabular_regression' does not match" in notes and "'rmse' does not apply" in notes and "using f1" in notes


def test_repair_detects_time_series_from_profile() -> None:
    profile = profile_dataset(load_dataset(SAMPLES_DIR / "store_sales.csv"))
    llm_output = make_requirement(task_type="tabular_regression", target="sales", metrics=("mae",))
    req = RequirementAgent(None).repair(llm_output, profile)
    assert req.problem.task_type == "time_series_forecasting"
    assert [m.name for m in req.problem.performance_metrics] == ["mae"]  # still a valid regression metric


@pytest.mark.parametrize(
    "task_type, expected",
    [
        ("tabular_classification", ["f1"]),
        ("tabular_regression", ["rmse"]),
        ("time_series_forecasting", ["rmse"]),
        ("tabular_clustering", ["adjusted_rand_index", "silhouette"]),
    ],
)
def test_repair_adds_default_metrics(task_type: str, expected: list[str]) -> None:
    req = RequirementAgent(None).repair(make_requirement(task_type=task_type, metrics=()), profile=None)
    assert [m.name for m in req.problem.performance_metrics] == expected


def test_repair_normalizes_metric_aliases() -> None:
    req = RequirementAgent(None).repair(make_requirement(metrics=("F1 Score", "AUC")), profile=None)
    assert [m.name for m in req.problem.performance_metrics] == ["f1", "roc_auc"]


# --------------------------------------------------------------------------- sufficiency + summary


def test_sufficiency_ok() -> None:
    assert RequirementAgent(good_llm()).check_sufficiency(make_requirement()) == (True, [], [])


def test_sufficiency_rules_and_llm_questions() -> None:
    agent = RequirementAgent(good_llm(ClarifyingQuestions=ClarifyingQuestions(questions=["What is your goal?"])))
    ok, missing, questions = agent.check_sufficiency(make_requirement(intent="unclear", confidence=0.2))
    assert not ok and missing == [MISSING_GOAL, MISSING_CLARITY] and questions == ["What is your goal?"]


def test_sufficiency_llm_review_adds_missing_items_and_template_questions() -> None:
    llm = good_llm(SufficiencyCheck=SufficiencyCheck(sufficient=False, missing=["time horizon for the prediction"]))
    del llm.responses[ClarifyingQuestions]  # question generation fails -> template questions
    ok, missing, questions = RequirementAgent(llm).check_sufficiency(make_requirement())
    assert not ok and missing == ["time horizon for the prediction"]
    assert questions == ["Could you tell us more about time horizon for the prediction?"]


def test_sufficiency_falls_back_to_rules_when_llm_unavailable() -> None:
    assert RequirementAgent(FakeLLM({})).check_sufficiency(make_requirement()).ok is True


def test_summarize_uses_llm() -> None:
    assert RequirementAgent(good_llm()).summarize(make_requirement()) == "We will predict which customers are likely to churn."


@pytest.mark.parametrize("failure", [LLMError("down"), RuntimeError("unexpected bug")])
def test_summarize_falls_back_to_template(failure: Exception) -> None:
    summary = RequirementAgent(good_llm(RequirementSummary=failure)).summarize(make_requirement())
    assert "'churn'" in summary and "f1" in summary and summary.startswith("Goal:")


# --------------------------------------------------------------------------- orchestrator


def states(ctx: Any) -> list[str]:
    return [str(event.state) for event in ctx.history]


def test_full_run_reaches_done_with_a_report(tmp_path: Path) -> None:
    llm = good_llm(UserRequirement=make_requirement(models=("XGBoost",)))
    orchestrator = make_orchestrator(llm, tmp_path)
    run_id = orchestrator.start(CHURN_QUESTION, CHURN_CSV)
    ctx = orchestrator.run_until_blocked()

    assert ctx.state == RunState.DONE and states(ctx) == PIPELINE_STATES and ctx.errors == []
    assert [str(e.from_state) for e in ctx.history[1:]] == PIPELINE_STATES[:-1]  # each event links from -> to
    assert ctx.requirement.target == "churn" and ctx.requirement.datasets[0].path == str(CHURN_CSV)
    assert ctx.summary == "We will predict which customers are likely to churn."
    assert ctx.plan.primary_metric == "f1" and ctx.plan.candidates[0].name == "xgboost"  # user's model first
    assert ctx.data.split_sizes.train == 2100 and Path(ctx.data.artifacts["preprocessor"]).is_file()
    assert ctx.training.best_model == "xgboost" and Path(ctx.training.model_path).is_file()
    assert not any("not implemented" in event.message for event in ctx.history)  # no stub stages left
    assert ctx.explanation is not None and ctx.recommendations is not None
    assert ctx.insight is not None and ctx.insight.generated_by == "template"  # no InsightDraft fake → template
    assert all(Path(path).is_file() for path in ctx.report_paths.values()) and set(ctx.report_paths) == {"pdf", "markdown"}
    assert ctx.knowledge is not None and ctx.knowledge.generated_by == "retrieval"  # no KnowledgeBrief fake → fallback
    saved = json.loads((tmp_path / run_id / STATE_FILENAME).read_text(encoding="utf-8"))
    assert saved["state"] == "DONE" and len(saved["history"]) == len(PIPELINE_STATES)


def test_irrelevant_question_ends_politely_without_training(tmp_path: Path) -> None:
    llm = good_llm(RelevanceCheck=RelevanceCheck(relevant=False, reason="small talk"))
    orchestrator = make_orchestrator(llm, tmp_path)
    orchestrator.start("Hello! How is your day going?", CHURN_CSV)
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.DONE and states(ctx) == ["INIT", "DONE"]
    assert ctx.message == NOT_RELEVANT_MESSAGE and ctx.requirement is None and ctx.data is None
    assert [name for name, _ in llm.calls] == ["RelevanceCheck"]


def test_needs_input_then_answer_continues_to_done(tmp_path: Path) -> None:
    llm = good_llm(SufficiencyCheck=[SufficiencyCheck(sufficient=False, missing=["retention budget"]), SufficiencyCheck(sufficient=True)])
    orchestrator = make_orchestrator(llm, tmp_path)
    run_id = orchestrator.start(CHURN_QUESTION, CHURN_CSV)

    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.NEEDS_INPUT and ctx.pending_questions == ["Which column should we predict?"]

    assert orchestrator.provide_answer(run_id, "Budget is 10 lakh per quarter.") == RunState.PARSE
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.DONE and ctx.revision_count == 1
    assert "Clarification from the user:" in ctx.question and "A: Budget is 10 lakh per quarter." in ctx.question
    assert llm.count("UserRequirement") == 2 and llm.count("RelevanceCheck") == 1
    assert states(ctx)[:5] == ["INIT", "PARSE", "VALIDATE", "NEEDS_INPUT", "PARSE"]


def test_revision_limit_fails_the_run(tmp_path: Path) -> None:
    llm = good_llm(SufficiencyCheck=SufficiencyCheck(sufficient=False, missing=["what to predict"]))
    orchestrator = make_orchestrator(llm, tmp_path)
    run_id = orchestrator.start("Help me with my data", CHURN_CSV)

    assert orchestrator.run_until_blocked().state == RunState.NEEDS_INPUT
    for answer in ("not sure", "still not sure"):
        orchestrator.provide_answer(run_id, answer)
        ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.FAILED and ctx.revision_count == 2
    assert "after 2 round(s)" in ctx.message and "what to predict" in ctx.message
    with pytest.raises(ValueError, match="not waiting for input"):
        orchestrator.provide_answer(run_id, "one more try")


def test_state_is_persisted_and_resumable(tmp_path: Path) -> None:
    first = make_orchestrator(good_llm(), tmp_path)
    run_id = first.start(CHURN_QUESTION, CHURN_CSV)
    for expected in (RunState.PARSE, RunState.VALIDATE, RunState.PROFILE):
        assert first.step() == expected
    assert json.loads((tmp_path / run_id / STATE_FILENAME).read_text(encoding="utf-8"))["state"] == "PROFILE"

    second = make_orchestrator(good_llm(), tmp_path)  # e.g. after a restart
    ctx = second.resume(run_id)
    assert ctx.state == RunState.PROFILE and ctx.requirement == first.context.requirement
    assert second.run_until_blocked().state == RunState.DONE
    with pytest.raises(FileNotFoundError):
        second.resume("no-such-run")


def test_waiting_run_can_be_answered_after_restart(tmp_path: Path) -> None:
    sufficiency = [SufficiencyCheck(sufficient=False, missing=["goal"]), SufficiencyCheck(sufficient=True)]
    run_id = make_orchestrator(good_llm(SufficiencyCheck=sufficiency[:1]), tmp_path).start(CHURN_QUESTION, CHURN_CSV)
    make_orchestrator(good_llm(SufficiencyCheck=sufficiency[:1]), tmp_path).run_until_blocked(run_id)

    fresh = make_orchestrator(good_llm(SufficiencyCheck=sufficiency[1:]), tmp_path)
    fresh.provide_answer(run_id, "Reduce churn next quarter.")  # loads the saved run by id
    assert fresh.run_until_blocked(run_id).state == RunState.DONE


def test_llm_failure_during_parse_fails_run_cleanly(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(good_llm(UserRequirement=LLMError("API down")), tmp_path)
    orchestrator.start(CHURN_QUESTION, CHURN_CSV)
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.FAILED and "PARSE failed: LLMError: API down" in ctx.errors[0]
    assert "parse stage" in ctx.message


def test_missing_dataset_fails_run(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(good_llm(), tmp_path)
    orchestrator.start(CHURN_QUESTION, tmp_path / "missing.csv")
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.FAILED and "DataLoadError" in ctx.errors[0]


def test_step_limit_stops_a_looping_run(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(good_llm(), tmp_path)
    orchestrator.start(CHURN_QUESTION, CHURN_CSV)
    orchestrator._handlers[RunState.TRAIN] = lambda ctx: (RunState.TRAIN, "looping")  # simulate a buggy handler
    ctx = orchestrator.run_until_blocked(max_steps=20)
    assert ctx.state == RunState.FAILED and "possible loop" in ctx.errors[-1]


def test_finished_runs_do_not_move_and_bad_input_is_rejected(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(good_llm(RelevanceCheck=RelevanceCheck(relevant=False)), tmp_path)
    orchestrator.start("hi", CHURN_CSV)
    assert orchestrator.run_until_blocked().state == RunState.DONE
    assert orchestrator.step() == RunState.DONE
    with pytest.raises(ValueError):
        orchestrator.start("   ", CHURN_CSV)
    with pytest.raises(RuntimeError):
        make_orchestrator(good_llm(), tmp_path).step()

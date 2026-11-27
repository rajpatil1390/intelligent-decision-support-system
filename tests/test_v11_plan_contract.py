"""v1.1 (A): strict plan contract, independent plan validation, the TRAIN guard and malformed-file hardening.

No network calls: the LLM is faked or absent, budgets are tiny.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from configs.settings import Settings, get_settings
from idss.agents.manager import Orchestrator, RunContext, RunState
from idss.agents.model_agent import ModelAgent, ModelChoice
from idss.agents.requirement_agent import RequirementAgent, requirement_from_profile
from idss.core.llm_client import LLMValidationError
from idss.core.logger import get_logger
from idss.core.plan_contract import (
    METRIC_NAMES,
    MODEL_NAMES,
    TASK_MODELS,
    AnalysisPlan,
    PlanContractError,
    PlanProposal,
    parse_plan,
)
from idss.core.schemas import (
    BUDGET_PRESETS,
    DatasetProfile,
    DatasetSpec,
    ModelPlan,
    ProblemSpec,
    RankedCandidate,
    TrainingBudget,
    UserRequirement,
)
from idss.execution.plan_validator import (
    PlanRejected,
    PlanValidationError,
    ValidatedPlan,
    fallback_plan,
    revalidate,
    validate_or_fallback,
    validate_plan,
)
from idss.ml import model_zoo
from idss.ml.data_loader import WIDE_DATASET_COLUMNS, DataLoadError, load_dataset, load_warnings
from idss.ml.evaluator import METRICS
from idss.ml.profiler import profile_dataset
from tests.fakes import FakeLLM, PassingVerifier
from tests.test_ui_pages import FakeClient, assert_ok, open_page, texts
from ui import api_client

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
CHURN_CSV = SAMPLES / "customer_churn.csv"
TINY = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)
QUESTION = "Which customers are likely to churn?"


def churn_plan(**changes: Any) -> dict[str, Any]:
    """A valid AnalysisPlan payload for the churn sample, with `changes` applied."""
    payload: dict[str, Any] = {
        "task_type": "tabular_classification", "target": "churn", "primary_metric": "f1",
        "candidate_models": ["lightgbm", "logistic_regression"], "budget_mode": "fast",
        "constraints": {"max_inference_ms_per_row": None, "max_train_minutes": None, "prefer_interpretable": False},
        "actionable_features": ["contract_type", "monthly_charges"], "time_column": None,
    }
    return payload | changes


@pytest.fixture(scope="module")
def churn() -> tuple[pd.DataFrame, DatasetProfile]:
    df = load_dataset(CHURN_CSV)
    return df, profile_dataset(df)


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy()  # the session's test settings (public demo off), safe to modify


@pytest.fixture
def warnings_logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = get_logger().add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    get_logger().remove(sink)


def rejected(result: ValidatedPlan | PlanRejected) -> list[str]:
    assert isinstance(result, PlanRejected), f"expected a rejection, got {result}"
    return result.reasons


# --------------------------------------------------------------------------- 1. the contract


def test_contract_names_come_from_the_registries() -> None:
    for task, names in TASK_MODELS.items():
        assert list(names) == model_zoo.candidate_names(task)
    assert "seasonal_naive" not in MODEL_NAMES  # baselines are always trained, never candidates
    assert set(METRIC_NAMES) == set(METRICS)


def test_contract_accepts_a_valid_plan() -> None:
    plan = parse_plan(churn_plan())
    assert isinstance(plan, AnalysisPlan) and plan.candidate_models == ["lightgbm", "logistic_regression"]
    with pytest.raises(ValidationError):
        plan.target = "tenure_months"  # type: ignore[misc]  # frozen


def test_contract_rejects_unknown_model_name() -> None:
    with pytest.raises(PlanContractError) as caught:
        parse_plan(churn_plan(candidate_models=["lightgbm", "svm"]))
    assert caught.value.reasons == ["Model 'svm' is not in the model zoo."]


def test_contract_rejects_extra_field() -> None:
    with pytest.raises(PlanContractError) as caught:
        parse_plan(churn_plan(run_shell="rm -rf /"))
    assert caught.value.reasons == ["Unknown field 'run_shell' is not part of the plan contract."]
    with pytest.raises(PlanContractError, match="constraints.gpu"):
        parse_plan(churn_plan(constraints={"gpu": True}))


def test_contract_rejects_wrong_metric_for_task() -> None:
    with pytest.raises(PlanContractError, match="metric 'rmse' does not apply to tabular_classification"):
        parse_plan(churn_plan(primary_metric="rmse"))
    with pytest.raises(PlanContractError, match="Metric 'profit' is not a metric the system can measure"):
        parse_plan(churn_plan(primary_metric="profit"))


def test_contract_rejects_models_of_another_family_and_duplicates() -> None:
    with pytest.raises(PlanContractError) as caught:
        parse_plan(churn_plan(candidate_models=["ridge", "lightgbm", "lightgbm"]))
    reason = caught.value.reasons[0]
    assert "lists lightgbm more than once" in reason and "ridge cannot be used for tabular_classification" in reason


def test_contract_is_strict_about_types_and_bounds() -> None:
    for bad in (
        churn_plan(budget_mode="turbo"),
        churn_plan(target=7),  # no coercion to str
        churn_plan(candidate_models=[]),
        churn_plan(candidate_models=["lightgbm"] * 7),
        churn_plan(actionable_features=[f"col{i}" for i in range(21)]),
        churn_plan(constraints={"max_train_minutes": "10"}),  # no coercion from str
        churn_plan(constraints={"max_train_minutes": 0}),
        churn_plan(constraints={"max_train_minutes": 5000}),  # above 24 h
        churn_plan(constraints={"max_inference_ms_per_row": -1.0}),
    ):
        with pytest.raises(PlanContractError):
            parse_plan(bad)


def test_contract_cross_field_rules() -> None:
    with pytest.raises(PlanContractError, match="needs a target column"):
        parse_plan(churn_plan(target=None))
    with pytest.raises(PlanContractError, match="cannot also be an actionable feature"):
        parse_plan(churn_plan(actionable_features=["churn"]))
    with pytest.raises(PlanContractError, match="needs a time_column"):
        parse_plan(churn_plan(task_type="time_series_forecasting", target="sales", primary_metric="rmse",
                              candidate_models=["lightgbm"]))


def test_model_choice_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ModelChoice.model_validate({"candidates": [{"name": "lightgbm"}], "temperature": 2})
    with pytest.raises(ValidationError):
        ModelChoice.model_validate({"candidates": [{"name": "lightgbm", "hyperparameters": {"n": 9}}]})


def test_model_agent_passes_unknown_names_through_to_the_contract(churn: tuple[pd.DataFrame, DatasetProfile]) -> None:
    _, profile = churn
    llm = FakeLLM({ModelChoice: ModelChoice(candidates=[RankedCandidate(name="SVM", reason="margins")],
                                            primary_metric="rmse")})
    proposal = ModelAgent(llm).propose(requirement_from_profile(QUESTION, profile), profile)
    assert proposal.source == "llm" and proposal.payload["candidate_models"][0] == "svm"
    assert proposal.payload["primary_metric"] == "rmse"  # the LLM's metric is checked, not silently replaced
    with pytest.raises(PlanContractError) as caught:
        parse_plan(proposal.payload)
    assert any("'svm' is not in the model zoo" in reason for reason in caught.value.reasons)


def test_invalid_llm_output_is_rejected_not_guessed(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    llm = FakeLLM({ModelChoice: LLMValidationError("bad JSON", errors="- budget: extra fields not permitted")})
    proposal = ModelAgent(llm).propose(requirement_from_profile(QUESTION, profile), profile)
    assert proposal.payload == {} and "extra fields not permitted" in proposal.errors[0]
    validated = validate_or_fallback(proposal, df, profile, settings)
    assert validated.fallback and validated.rejected_reasons == proposal.errors


# --------------------------------------------------------------------------- 2. the validator


def test_valid_plan_is_accepted_and_sealed(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    result = validate_plan(churn_plan(), df, profile, settings, source="llm")
    assert isinstance(result, ValidatedPlan) and result.sealed and not result.fallback
    plan = result.model_plan
    assert plan.task_type == "tabular_classification" and plan.target == "churn" and plan.source == "llm"
    assert [c.name for c in plan.candidates] == ["lightgbm", "logistic_regression"]
    assert plan.budget == BUDGET_PRESETS["fast"]


def test_bad_target_is_rejected(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    assert rejected(validate_plan(churn_plan(target="loyalty"), df, profile, settings)) == [
        "Target column 'loyalty' is not in the data."]
    continuous = churn_plan(target="monthly_charges", actionable_features=["contract_type"])
    assert "continuous number" in rejected(validate_plan(continuous, df, profile, settings))[0]
    regression = churn_plan(task_type="tabular_regression", primary_metric="rmse", candidate_models=["ridge"])
    assert "needs a numeric target" in rejected(validate_plan(regression, df, profile, settings))[0]


def test_target_cannot_be_an_id_column(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    reasons = rejected(validate_plan(churn_plan(target="customer_id"), df, profile, settings))
    assert reasons == ["Target 'customer_id' is an identifier column (a different value on almost every row); "
                       "it cannot be predicted."]


def test_out_of_range_trials_and_time_are_rejected(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    huge = TrainingBudget(max_candidates=9, n_trials_per_model=500, timeout_per_model_s=600, total_timeout_s=7200)
    reasons = rejected(validate_plan(churn_plan(), df, profile, settings, budget=huge))
    assert reasons == [
        "The budget asks for 9 models; the limit is 6.",
        "The budget asks for 500 tuning trials per model; the limit is 40.",
        "The budget asks for 600 seconds per model; the limit is 180.",
        "The budget asks for 7200 seconds in total; the limit is 1080.",
    ]
    long_training = churn_plan(constraints={"max_train_minutes": 300.0})
    assert rejected(validate_plan(long_training, df, profile, settings)) == [
        "The training-time constraint of 300 minutes exceeds the limit of 240 minutes."]
    strict = settings.model_copy(update={"plan_max_trials_per_model": 10})
    assert "15 tuning trials per model" in rejected(validate_plan(churn_plan(), df, profile, strict))[0]


def test_wrong_metric_for_task_is_rejected(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    assert "does not apply to tabular_classification" in rejected(
        validate_plan(churn_plan(primary_metric="mae"), df, profile, settings))[0]
    segments = load_dataset(SAMPLES / "customer_segments.csv")
    clustering = {"task_type": "tabular_clustering", "target": None, "primary_metric": "adjusted_rand_index",
                  "candidate_models": ["kmeans"], "budget_mode": "fast"}
    assert rejected(validate_plan(clustering, segments, profile_dataset(segments), settings)) == [
        "Metric 'adjusted_rand_index' needs ground-truth labels, but the plan names no ground-truth column."]
    zeros = pd.DataFrame({"x": np.arange(40.0) * 0.5, "y": np.r_[0.0, np.arange(1.0, 40.0) * 1.5]})
    mape = {"task_type": "tabular_regression", "target": "y", "primary_metric": "mape", "candidate_models": ["ridge"],
            "budget_mode": "fast"}
    assert rejected(validate_plan(mape, zeros, profile_dataset(zeros), settings)) == [
        "Metric 'mape' is undefined because 'y' contains zeros."]


def test_forecasting_needs_a_datetime_column(settings: Settings) -> None:
    sales = load_dataset(SAMPLES / "store_sales.csv")
    profile = profile_dataset(sales)
    forecast = {"task_type": "time_series_forecasting", "target": "sales", "primary_metric": "rmse",
                "candidate_models": ["lightgbm"], "budget_mode": "fast", "time_column": "date"}
    assert isinstance(validate_plan(forecast, sales, profile, settings), ValidatedPlan)
    assert rejected(validate_plan(forecast | {"time_column": "day_of_week"}, sales, profile, settings)) == [
        "Time column 'day_of_week' is not a date/time column (it is categorical)."]
    assert "needs a time_column" in rejected(validate_plan(forecast | {"time_column": None}, sales, profile, settings))[0]
    crops = load_dataset(SAMPLES / "crop_price.csv")
    no_dates = forecast | {"target": "price", "time_column": "year"}
    assert rejected(validate_plan(no_dates, crops, profile_dataset(crops), settings)) == [
        "Time column 'year' is not a date/time column (it is numeric)."]


def test_actionable_features_must_exist_and_not_be_ids(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    reasons = rejected(validate_plan(churn_plan(actionable_features=["discount", "customer_id"]), df, profile, settings))
    assert reasons == ["Actionable feature 'discount' is not in the data.",
                       "Actionable feature 'customer_id' is an identifier column; the business cannot change it."]


def test_public_demo_forces_fast_mode(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    result = validate_plan(churn_plan(budget_mode="full"), df, profile, settings, public_demo=True)
    assert isinstance(result, ValidatedPlan) and result.analysis.budget_mode == "fast"
    assert result.model_plan.budget == BUDGET_PRESETS["fast"]
    assert "Public demo: the fast budget is enforced." in result.notes
    full = validate_plan(churn_plan(budget_mode="full"), df, profile, settings, public_demo=False)
    assert isinstance(full, ValidatedPlan) and full.model_plan.budget == BUDGET_PRESETS["full"]


def test_data_limits_are_fatal(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    small = settings.model_copy(update={"plan_max_rows": 1000})
    result = validate_plan(churn_plan(), df, profile, small)
    assert isinstance(result, PlanRejected) and result.fatal
    assert result.reasons == ["The data has 3,000 rows; the limit is 1,000 rows."]
    with pytest.raises(PlanValidationError, match="3,000 rows"):
        validate_or_fallback(churn_plan(), df, profile, small)
    tiny_upload = settings.model_copy(update={"max_upload_mb": 0.01})
    file_check = validate_plan(churn_plan(), df, profile, tiny_upload, data_path=CHURN_CSV)
    assert isinstance(file_check, PlanRejected) and file_check.fatal and "the limit is 0.01 MB" in file_check.reasons[0]


def test_fallback_plan_is_used_and_logged(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings,
                                          warnings_logged: list[str]) -> None:
    df, profile = churn
    # The LLM-side proposal names a real numeric target and an unknown model: the fallback must not reuse its text.
    proposal = PlanProposal(payload=churn_plan(task_type="tabular_regression", target="tenure_months",
                                               primary_metric="rmse", candidate_models=["ridge", "svm"],
                                               actionable_features=[]), source="llm")
    validated = validate_or_fallback(proposal, df, profile, settings, budget=TINY)
    assert validated.fallback and validated.sealed and validated.source == "fallback"
    assert validated.rejected_reasons == ["Model 'svm' is not in the model zoo."]
    analysis = validated.analysis
    assert (analysis.task_type, analysis.target, analysis.budget_mode) == ("tabular_classification", "churn", "fast")
    assert analysis.candidate_models == model_zoo.DEFAULT_RANKING["classification"] and analysis.actionable_features == []
    assert validated.model_plan.source == "fallback" and validated.model_plan.budget == TINY
    assert "Plan rejected: Model 'svm' is not in the model zoo." in warnings_logged
    assert any(message.startswith("Using the fallback plan: tabular_classification on 'churn'") for message in warnings_logged)


def test_fallback_budget_is_clamped_to_the_limits(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    strict = settings.model_copy(update={"plan_max_trials_per_model": 5, "plan_max_total_seconds": 90})
    validated = validate_or_fallback(churn_plan(), df, profile, strict)  # fast preset: 15 trials, 180 s > limits
    assert validated.fallback and validated.model_plan.budget.n_trials_per_model == 5
    assert validated.model_plan.budget.total_timeout_s == 90


def test_fallback_for_time_series_and_regression(settings: Settings) -> None:
    plan = fallback_plan(load_dataset(SAMPLES / "store_sales.csv"), settings)
    assert (plan.task_type, plan.target, plan.time_column) == ("time_series_forecasting", "sales", "date")
    assert plan.candidate_models == model_zoo.DEFAULT_RANKING["time_series"]
    plan = fallback_plan(load_dataset(SAMPLES / "crop_price.csv"), settings)
    assert (plan.task_type, plan.target, plan.primary_metric) == ("tabular_regression", "price", "rmse")


def test_sealing_and_revalidation(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df, profile = churn
    validated = validate_plan(churn_plan(), df, profile, settings)
    assert isinstance(validated, ValidatedPlan) and validated.sealed
    reloaded = ValidatedPlan.model_validate_json(validated.model_dump_json())
    assert reloaded.model_dump() == validated.model_dump() and not reloaded.sealed  # read back: validate again
    again = revalidate(reloaded, df, profile, settings)
    assert isinstance(again, ValidatedPlan) and again.sealed and again.model_plan == validated.model_plan
    validated.model_plan.candidates.append(RankedCandidate(name="svm"))
    assert not validated.sealed  # changed after sealing
    hand_built = ValidatedPlan(analysis=parse_plan(churn_plan()), model_plan=again.model_plan, source="llm")
    assert not hand_built.sealed


# --------------------------------------------------------------------------- 3. the Orchestrator: TRAIN needs a ValidatedPlan


class SpyTrainer:
    """Records the plan TRAIN hands to the engine, then stops the run (these tests are not about training)."""

    def __init__(self) -> None:
        self.plans: list[ModelPlan] = []

    def run(self, *args: Any, plan: ModelPlan | None = None, **kwargs: Any) -> Any:
        assert plan is not None
        self.plans.append(plan)
        raise RuntimeError("spy trainer stops here")


def offline_orchestrator(tmp_path: Path) -> tuple[Orchestrator, str, SpyTrainer]:
    orchestrator = Orchestrator(None, tmp_path, budget=TINY)
    orchestrator.verifier = PassingVerifier()
    spy = SpyTrainer()
    orchestrator.trainer = spy  # type: ignore[assignment]
    profile = profile_dataset(load_dataset(CHURN_CSV))
    run_id = orchestrator.start(QUESTION, CHURN_CSV, requirement=requirement_from_profile(QUESTION, profile))
    return orchestrator, run_id, spy


def run_to(orchestrator: Orchestrator, run_id: str, state: RunState) -> RunContext:
    ctx = orchestrator._get(run_id)
    while ctx.state != state and ctx.state not in (RunState.DONE, RunState.FAILED):
        orchestrator.step(run_id)
    return ctx


def states(ctx: RunContext) -> list[str]:
    return [str(event.state) for event in ctx.history]


def test_unvalidated_plan_cannot_reach_train(tmp_path: Path) -> None:
    orchestrator, run_id, spy = offline_orchestrator(tmp_path)
    orchestrator._handlers[RunState.PLAN] = lambda ctx: (RunState.TRAIN, "Plan accepted without validation.")
    ctx = orchestrator.run_until_blocked(run_id)

    assert ctx.state == RunState.FAILED and "TRAIN" not in states(ctx) and spy.plans == []
    assert ctx.errors == ["Refused to start TRAIN: the run has no validated plan."]
    assert ctx.history[-1].from_state == RunState.PLAN and ctx.plan is None
    with pytest.raises((AttributeError, ValueError)):  # read-only: a plan can only come from the validator
        ctx.plan = ModelPlan(task_type="tabular_classification", target="churn",  # type: ignore[misc]
                             candidates=[RankedCandidate(name="lightgbm")], primary_metric="f1", budget=TINY)


def test_hand_built_invalid_plan_cannot_reach_train(tmp_path: Path) -> None:
    orchestrator, run_id, spy = offline_orchestrator(tmp_path)

    def bypass(ctx: RunContext) -> tuple[RunState, str]:
        analysis = parse_plan(churn_plan(target="customer_id"))  # well-formed, but the target is an ID column
        model_plan = ModelPlan(task_type="tabular_classification", target="customer_id",
                               candidates=[RankedCandidate(name="lightgbm")], primary_metric="f1", budget=TINY)
        ctx.validated_plan = ValidatedPlan(analysis=analysis, model_plan=model_plan, source="llm")
        return RunState.TRAIN, "Plan built by hand."

    orchestrator._handlers[RunState.PLAN] = bypass
    ctx = orchestrator.run_until_blocked(run_id)
    assert ctx.state == RunState.FAILED and "TRAIN" not in states(ctx) and spy.plans == []
    assert ctx.errors[-1].startswith("Refused to start TRAIN: the stored plan failed validation: Target 'customer_id'")


def test_plan_changed_after_validation_cannot_reach_train(tmp_path: Path) -> None:
    orchestrator, run_id, spy = offline_orchestrator(tmp_path)
    real_plan = orchestrator._handlers[RunState.PLAN]

    def tamper(ctx: RunContext) -> tuple[RunState, str]:
        result = real_plan(ctx)
        object.__setattr__(ctx.validated_plan.analysis, "target", "customer_id")  # force past the frozen model
        return result

    orchestrator._handlers[RunState.PLAN] = tamper
    ctx = orchestrator.run_until_blocked(run_id)
    assert ctx.state == RunState.FAILED and "TRAIN" not in states(ctx) and spy.plans == []
    assert "failed validation" in ctx.errors[-1]


def test_train_never_uses_a_model_plan_edited_after_validation(tmp_path: Path) -> None:
    orchestrator, run_id, spy = offline_orchestrator(tmp_path)
    real_plan = orchestrator._handlers[RunState.PLAN]

    def tamper(ctx: RunContext) -> tuple[RunState, str]:
        result = real_plan(ctx)
        ctx.validated_plan.model_plan.candidates.insert(0, RankedCandidate(name="svm"))
        return result

    orchestrator._handlers[RunState.PLAN] = tamper
    run_to(orchestrator, run_id, RunState.FAILED)
    assert len(spy.plans) == 1 and spy.plans[0].candidates[0].name == "lightgbm"  # rebuilt from the validated plan
    assert all(c.name != "svm" for c in spy.plans[0].candidates)


def test_validated_plan_reaches_train_and_survives_a_restart(tmp_path: Path) -> None:
    orchestrator = Orchestrator(None, tmp_path, budget=TINY)
    profile = profile_dataset(load_dataset(CHURN_CSV))
    run_id = orchestrator.start(QUESTION, CHURN_CSV, requirement=requirement_from_profile(QUESTION, profile))
    ctx = run_to(orchestrator, run_id, RunState.TRAIN)
    assert ctx.state == RunState.TRAIN and ctx.validated_plan is not None and ctx.validated_plan.sealed
    assert ctx.plan == ctx.validated_plan.model_plan and ctx.plan.source == "default"

    restarted = Orchestrator(None, tmp_path, budget=TINY)  # a new process: the plan read from state.json is unsealed
    assert not restarted.resume(run_id).validated_plan.sealed
    assert restarted.step(run_id) == RunState.VERIFY  # validated again, then trained
    assert restarted.context.validated_plan.sealed and restarted.context.training.best_model == "lightgbm"


def test_rejected_plan_falls_back_in_the_run_and_is_shown() -> None:
    orchestrator = Orchestrator(None, budget=TINY)  # default runs dir, so the API can read the run
    orchestrator.model_agent = ModelAgent(FakeLLM({ModelChoice: ModelChoice(candidates=[RankedCandidate(name="svm")])}))
    profile = profile_dataset(load_dataset(CHURN_CSV))
    requirement = RequirementAgent(None).repair(UserRequirement(
        problem=ProblemSpec(task_type="tabular_regression", description="Predict monthly charges."),
        datasets=[DatasetSpec(target_column="monthly_charges")], actionable_features=["contract_type"], confidence=1.0,
    ), profile)
    run_id = orchestrator.start("What drives monthly charges?", CHURN_CSV, requirement=requirement)
    ctx = run_to(orchestrator, run_id, RunState.VERIFY)

    assert ctx.state == RunState.VERIFY and ctx.validated_plan.fallback
    assert ctx.validated_plan.rejected_reasons == ["Model 'svm' is not in the model zoo."]
    plan_event = next(e for e in ctx.history if e.from_state == RunState.PLAN)
    assert plan_event.state == RunState.TRAIN and plan_event.message.startswith("Proposed plan rejected (Model 'svm'")
    assert ctx.training.target == "churn" and ctx.training.task_type == "tabular_classification"
    # The requirement now describes what was trained, so verification and recommendations stay consistent.
    assert ctx.requirement.target == "churn" and ctx.requirement.problem.task_type == "tabular_classification"
    assert [m.name for m in ctx.requirement.problem.performance_metrics] == ["f1"]
    assert ctx.requirement.actionable_features == []
    assert any("following the fallback plan" in note for note in ctx.requirement.system_notes)

    with TestClient(create_app(llm_factory=lambda: None, public_demo=False)) as client:
        body = client.get(f"/runs/{run_id}").json()
    assert body["plan_fallback"] is True and body["plan_source"] == "fallback"
    assert body["plan_rejections"] == ["Model 'svm' is not in the model zoo."]


def test_api_refuses_unusable_actionable_features() -> None:
    with TestClient(create_app(llm_factory=lambda: None, public_demo=False)) as client:
        files = {"file": ("churn.csv", CHURN_CSV.read_bytes(), "text/csv")}
        dataset_id = client.post("/datasets", files=files).json()["id"]
        for features, status in ((["customer_id"], 400), (["loyalty"], 400), ([f"c{i}" for i in range(21)], 422)):
            response = client.post("/runs", json={"dataset_id": dataset_id, "question": QUESTION, "mode": "fast",
                                                  "actionable_features": features})
            assert response.status_code == status, response.json()
            if status == 400:
                assert response.json()["error"] == "invalid_actionable_features"
                assert features[0] in response.json()["detail"]
        assert client.get("/runs", params={"dataset_id": dataset_id}).json() == []  # nothing was started


def test_run_monitor_shows_the_plan_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    fake.details["run-done"].update(plan_source="fallback", plan_fallback=True,
                                    plan_rejections=["Model 'svm' is not in the model zoo."])
    monkeypatch.setattr(api_client, "get_client", lambda: fake)
    at = open_page("pages/4_Run_Monitor.py", run_id="run-done")
    assert_ok(at)
    page = texts(at)
    assert "The proposed analysis plan was rejected" in page and "Model 'svm' is not in the model zoo." in page


# --------------------------------------------------------------------------- 4. malformed files


def write(tmp_path: Path, name: str, content: str | bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return path


@pytest.mark.parametrize(("content", "message"), [
    (b"", "File is empty"),
    ("   \n\n  \n", "is empty: it has no header and no data"),
    ("customer,age,churn\n", "has a header but no data rows"),
])
def test_empty_and_header_only_files(tmp_path: Path, content: str | bytes, message: str) -> None:
    with pytest.raises(DataLoadError, match=message):
        load_dataset(write(tmp_path, "data.csv", content))


@pytest.mark.parametrize("header", ["a,b,a", "a, a ,b", "price;Price;price"])
def test_duplicate_column_names(tmp_path: Path, header: str) -> None:
    sep = ";" if ";" in header else ","
    rows = "\n".join(sep.join(str(i + j) for j in range(3)) for i in range(5))
    with pytest.raises(DataLoadError, match="duplicate column names: '(a|price)'"):
        load_dataset(write(tmp_path, "dupes.csv", f"{header}\n{rows}\n"))


def test_duplicate_column_names_in_excel(tmp_path: Path) -> None:
    path = tmp_path / "dupes.xlsx"
    pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["x", "y", "x"]).to_excel(path, index=False)
    with pytest.raises(DataLoadError, match="duplicate column names: 'x'"):
        load_dataset(path)


def test_wrong_and_mixed_encodings(tmp_path: Path) -> None:
    windows = load_dataset(write(tmp_path, "cp1252.csv", "item,price\nCafé €,5\nThé,3\n".encode("cp1252")))
    assert windows["item"].tolist() == ["Café €", "Thé"]
    assert load_warnings(windows) == ["The file is not UTF-8; it was read as Windows-1252."]

    mixed_bytes = "name,n\nJosé,1\n".encode() + "Zoë,2\n".encode("cp1252")
    mixed = load_dataset(write(tmp_path, "mixed.csv", mixed_bytes))
    assert len(mixed) == 2 and "mixes UTF-8 with another encoding" in load_warnings(mixed)[0]
    assert "mixes UTF-8" in profile_dataset(mixed).warnings[0]  # surfaced to the user through the profile

    latin = load_dataset(write(tmp_path, "latin.csv", b"code,n\n\x81x,1\nab,2\n"))  # 0x81 is undefined in cp1252
    assert load_warnings(latin) == ["The file is not UTF-8; it was read as Latin-1."]

    utf16 = load_dataset(write(tmp_path, "utf16.csv", "a,b\n1,2\n3,4\n".encode("utf-16")))
    assert list(utf16.columns) == ["a", "b"] and len(utf16) == 2 and load_warnings(utf16) == []


def test_all_missing_target(tmp_path: Path, churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df = churn[0].assign(churn=np.nan)
    path = tmp_path / "no_target.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)  # loads: an empty target is a data problem, not a file problem
    profile = profile_dataset(loaded, target="churn")
    assert profile.suggested_task_type is None
    assert "Target 'churn' has no values (every row is missing); it cannot be predicted." in profile.warnings
    assert rejected(validate_plan(churn_plan(), loaded, profile, settings)) == [
        "Target 'churn' has no values (every row is missing)."]
    repaired = RequirementAgent(None).repair(requirement_from_profile(QUESTION, churn[1]), profile_dataset(loaded))
    assert repaired.target != "churn"


def test_single_class_target(churn: tuple[pd.DataFrame, DatasetProfile], settings: Settings) -> None:
    df = churn[0].assign(churn="No")
    profile = profile_dataset(df, target="churn")
    assert profile.suggested_task_type is None
    assert "Target 'churn' has a single value ('No'); there is nothing to predict." in profile.warnings
    assert rejected(validate_plan(churn_plan(), df, profile, settings)) == [
        "Target 'churn' has a single value, so there is nothing to predict."]
    validated = validate_or_fallback(churn_plan(), df, profile, settings)
    assert validated.fallback and validated.analysis.target != "churn"


def test_one_unique_value_per_column(tmp_path: Path, settings: Settings) -> None:
    path = write(tmp_path, "constant.csv", "region,plan,score\n" + "north,basic,1\n" * 30)
    df = load_dataset(path)
    profile = profile_dataset(df)
    assert any(w.startswith("No column varies") for w in profile.warnings)
    assert profile.suggested_target is None and profile.suggested_task_type == "tabular_clustering"
    with pytest.raises(PlanValidationError, match="No usable feature columns"):
        validate_or_fallback(fallback_plan(df, settings), df, profile, settings)

    orchestrator = Orchestrator(None, tmp_path / "runs", budget=TINY)
    orchestrator.start("Group my customers.", path, requirement=requirement_from_profile("Group my customers.", profile))
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.FAILED and "nothing to learn" in ctx.message  # a clear stop, not a crash


def test_extremely_wide_file(tmp_path: Path) -> None:
    n_cols = WIDE_DATASET_COLUMNS + 200
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(30, n_cols)).round(3), columns=[f"f{i}" for i in range(n_cols)])
    path = tmp_path / "wide.csv"
    wide.to_csv(path, index=False)
    df = load_dataset(path)
    assert df.shape == (30, n_cols)
    expected = (f"The file has {n_cols:,} columns (more than {WIDE_DATASET_COLUMNS:,}); profiling and training will "
                "be slow. Consider keeping only the relevant columns.")
    assert load_warnings(df) == [expected] and expected in profile_dataset(df).warnings


def test_ragged_and_binary_files_give_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="Could not read 'ragged.csv' as CSV"):
        load_dataset(write(tmp_path, "ragged.csv", "a,b\n1,2\n3,4,5,6\n"))
    with pytest.raises(DataLoadError, match="looks like a binary file"):
        load_dataset(write(tmp_path, "binary.csv", b"\x00\x01\x02PK\x03\x04" * 10))


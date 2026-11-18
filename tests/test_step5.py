"""Step 5 tests: threshold tuning, knowledge retrieval, verifier, sandbox, analysis agent and the
VERIFY → PLAN revision loop. The LLM is always faked; there are no network calls."""

import json
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from configs.settings import Settings
from idss.agents.analysis_agent import AnalysisAgent, AnalysisCode
from idss.agents.knowledge_agent import BriefInsights, KnowledgeAgent
from idss.agents.manager import Orchestrator, RunState
from idss.agents.model_agent import ModelAgent, ModelChoice
from idss.agents.requirement_agent import RelevanceCheck, RequirementSummary, SufficiencyCheck
from idss.agents.verifier import RECOMMENDATIONS, LLMReview, Verifier
from idss.core.schemas import (
    BUDGET_PRESETS,
    Baseline,
    Check,
    DatasetProfile,
    KnowledgeBrief,
    LeaderboardEntry,
    RankedCandidate,
    TrainingBudget,
    TrainingResult,
    UserRequirement,
    VerificationReport,
)
from idss.execution.sandbox import MAX_STDOUT_CHARS, _child_env, check_code, run_code
from idss.knowledge.retriever import retrieve
from idss.knowledge.vector_store import KnowledgeIndex, chunk_text
from idss.knowledge.web_search import web_search
from idss.ml.data_loader import load_dataset
from idss.ml.evaluator import apply_threshold, tune_threshold
from idss.ml.profiler import profile_dataset
from tests.fakes import FakeLLM
from tests.test_automl import (
    classification_df,
    clustering_df,
    make_requirement,
    regression_df,
    run_automl,
    timeseries_df,
)

CHURN_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "customer_churn.csv"
ONE_MODEL = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)


@pytest.fixture(scope="module")
def churn_profile() -> DatasetProfile:
    return profile_dataset(load_dataset(CHURN_CSV))


def churn_requirement(*metrics: tuple[str, float | None]) -> UserRequirement:
    requirement = make_requirement("tabular_classification", "churn", metrics=tuple(m for m, _ in metrics) or ("f1",))
    requirement.problem.description = "Predict which customers will churn so we can retain them."
    requirement.problem.domain = "telecom"
    for spec, (_, target) in zip(requirement.problem.performance_metrics, metrics, strict=False):
        spec.target = target
    return requirement


# --------------------------------------------------------------------------- 1. threshold tuning


def test_tune_threshold_finds_a_better_cutoff() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.where(y == 1, rng.uniform(0.25, 0.6, 400), rng.uniform(0.0, 0.35, 400))  # positives rarely reach 0.5
    threshold, tuned, default = tune_threshold("f1", y, np.column_stack([1 - p, p]), pos_label=1)
    assert threshold < 0.5 and tuned > default


def test_threshold_tuning_on_imbalanced_data(tmp_path: Path) -> None:
    df = classification_df(n=600, minority=0.12)
    result = run_automl(df, "tabular_classification", "target", ["logistic_regression"], tmp_path)
    [selected] = [e for e in result.leaderboard if e.selected]

    assert result.decision_threshold is not None and selected.decision_threshold == result.decision_threshold
    assert selected.valid_score >= selected.valid_score_default_threshold  # tuning never hurts on valid
    assert result.test_metrics["f1"] >= result.test_metrics_default_threshold["f1"]  # and here it helps on test
    model = joblib.load(result.model_path)  # the saved model applies the tuned threshold
    proba = model.predict_proba(df.head(50))
    expected = np.asarray(model.classes, dtype=object)[apply_threshold(proba, model.pos_label, result.decision_threshold)]
    assert list(model.predict(df.head(50))) == list(expected)


def test_no_threshold_for_regression_or_ranking_metrics(tmp_path: Path) -> None:
    regression = run_automl(regression_df(), "tabular_regression", "target", ["ridge"], tmp_path / "reg")
    assert regression.decision_threshold is None and regression.test_metrics_default_threshold == {}
    profile = profile_dataset(classification_df(), target="target")
    requirement = make_requirement("tabular_classification", "target", metrics=("roc_auc",))
    plan = ModelAgent(None).plan(requirement, profile).model_copy(
        update={"candidates": [RankedCandidate(name="logistic_regression")], "budget": ONE_MODEL})
    from idss.ml.trainer import AutoMLTrainer
    ranking = AutoMLTrainer().run(classification_df(), requirement, profile, tmp_path / "auc", plan=plan)
    assert ranking.primary_metric == "roc_auc" and ranking.decision_threshold is None


# --------------------------------------------------------------------------- 2. knowledge


def test_chunking_respects_size_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=500, overlap=80)
    assert len(chunks) >= 6 and all(len(chunk) <= 500 for chunk in chunks)
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first.split()[-1] in second.split()  # consecutive chunks overlap
    assert chunk_text("   ") == []


def test_index_builds_persists_and_rebuilds_on_change(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "imbalance.md").write_text("# Imbalance\nUse class weights when the minority class is rare.", encoding="utf-8")
    (notes / "forecasting.md").write_text("# Forecasting\nUse lag features and a seasonal naive baseline.", encoding="utf-8")
    index_path = tmp_path / "index.joblib"

    assert KnowledgeIndex(notes, index_path).ensure_fresh() is True and index_path.is_file()
    reloaded = KnowledgeIndex(notes, index_path)
    assert reloaded.ensure_fresh() is False  # loaded from the cache, not rebuilt
    assert reloaded.search("rare minority class weights")[0].source == "imbalance.md"

    (notes / "clustering.md").write_text("# Clustering\nK-means groups customers into segments.", encoding="utf-8")
    assert reloaded.ensure_fresh() is True  # a new note changes the content hash
    assert reloaded.search("kmeans customer segments")[0].source == "clustering.md"
    assert reloaded.search("   ") == []


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("forecast daily sales with weekly seasonality and lag features", "time_series_forecasting.md"),
        ("rare positive class: class weights and decision threshold", "class_imbalance.md"),
        ("customer segmentation with k-means and silhouette score", "clustering_segmentation.md"),
        ("which customers will leave and which retention offer to make", "churn_retention_strategies.md"),
        ("a column almost perfectly correlated with the target", "feature_leakage.md"),
        ("how much stock to order and which price to set", "pricing_and_demand_decisions.md"),
    ],
)
def test_retrieval_finds_the_relevant_note(query: str, expected_source: str) -> None:
    # Keyword retrieval can rank a closely related note first (e.g. a classification note that also covers
    # thresholds), and the agent receives all top-k chunks, so the check is recall@2 at note level.
    results = retrieve(query, k=4)
    top_notes = list(dict.fromkeys(r.source for r in results))
    assert expected_source in top_notes[:2], top_notes
    assert len(results) <= 4 and max(sum(r.source == s for r in results) for s in top_notes) <= 2


def test_web_search_is_silently_empty_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("web search must not touch the network without a key")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert web_search("customer churn best practices", settings=Settings(_env_file=None)) == []


def test_web_search_errors_are_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: Any, **kwargs: Any) -> None:
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    assert web_search("churn", settings=Settings(_env_file=None, tavily_api_key="key")) == []


def test_knowledge_agent_summarises_or_falls_back(churn_profile: DatasetProfile) -> None:
    requirement = churn_requirement()
    llm = FakeLLM({BriefInsights: BriefInsights(insights=["Weight the minority class.", "Tune the threshold on validation data."])})
    brief = KnowledgeAgent(llm, use_web=False).brief(requirement, churn_profile)
    assert brief.generated_by == "llm" and brief.insights[0] == "Weight the minority class." and brief.sources
    prompt = llm.calls[0][1][1]["content"]
    assert "[churn_retention_strategies.md]" in prompt and "CUST-00001" not in prompt

    fallback = KnowledgeAgent(FakeLLM({}), use_web=False).brief(requirement, churn_profile)
    assert fallback.generated_by == "retrieval" and len(fallback.insights) == len(fallback.chunks) > 0
    assert all(len(insight) <= 303 for insight in fallback.insights)


def test_model_agent_prompt_includes_knowledge_and_feedback(churn_profile: DatasetProfile) -> None:
    brief = KnowledgeBrief(query="q", insights=["Prefer gradient boosting for medium-sized data."],
                           sources=["model_selection_by_data_size.md"], generated_by="llm")
    llm = FakeLLM({ModelChoice: ModelChoice(candidates=[RankedCandidate(name="lightgbm")])})
    plan = ModelAgent(llm).plan(churn_requirement(), churn_profile, knowledge=brief, feedback=["Reduce overfitting."])
    prompt = llm.calls[0][1][1]["content"]
    assert "Prefer gradient boosting for medium-sized data." in prompt and "- Reduce overfitting." in prompt
    assert "Re-planned using the verifier's recommendations." in plan.notes


# --------------------------------------------------------------------------- 3. verifier


def make_training(**overrides: Any) -> TrainingResult:
    values: dict[str, Any] = {
        "task_type": "tabular_classification", "target": "churn", "primary_metric": "f1", "direction": "maximize",
        "best_model": "lightgbm", "best_params": {}, "valid_score": 0.7,
        "test_metrics": {"f1": 0.70, "accuracy": 0.80, "n_test_rows": 300},
        "train_metrics": {"f1": 0.75, "accuracy": 0.84},
        "baseline": Baseline(name="majority_class", metrics={"f1": 0.0, "accuracy": 0.67}),
        "complexity": {}, "budget": BUDGET_PRESETS["fast"], "total_time_s": 1.0,
        "leaderboard": [LeaderboardEntry(model="lightgbm", status="ok", rank=1, valid_score=0.7, selected=True)],
        "model_path": "m", "leaderboard_path": "l", "metrics_path": "x", "test_set_accesses": ["final_evaluation"],
    }
    return TrainingResult(**(values | overrides))


def make_profile(warnings: list[str] = ()) -> DatasetProfile:
    return DatasetProfile(n_rows=3000, n_cols=12, columns=[], duplicate_rows=0, memory_mb=0.4, warnings=list(warnings))


def test_verifier_passes_a_good_model() -> None:
    report = Verifier(None).verify(churn_requirement(("f1", 0.6)), make_training(), make_profile())
    assert report.passed and len(report.checks) == 7 and report.recommendations == []
    assert report.llm_verdict is None  # no LLM configured: rule-based only


@pytest.mark.parametrize(
    "metrics, training, warnings, failed",
    [
        ((("f1", 0.8),), {}, [], "metric_targets"),
        ((("accuracy", 85),), {}, [], "metric_targets"),  # 85 is read as 85%
        ((), {"test_metrics": {"f1": 0.0, "n_test_rows": 300}, "train_metrics": {"f1": 0.0}}, [], "beats_baseline"),
        ((), {"leaderboard": [LeaderboardEntry(model="lightgbm", status="ok", rank=1, selected=True,
                                               violations=["inference 8 ms/row > 5 ms limit"])]}, [], "constraints"),
        ((), {"test_metrics": {"f1": 0.7, "accuracy": 0.8, "n_test_rows": 20}}, [], "test_size"),
        ((), {"test_set_accesses": ["final_evaluation", "final_evaluation"]}, [], "test_set_used_once"),
        ((), {}, ["Column 'refund' is almost perfectly correlated with the target (r=0.999); possible data leakage."],
         "no_leakage"),
        ((), {"train_metrics": {"f1": 0.95, "accuracy": 0.99}}, [], "overfitting"),
    ],
)
def test_verifier_detects_each_problem(metrics: tuple, training: dict, warnings: list[str], failed: str) -> None:
    report = Verifier(None).verify(churn_requirement(*metrics), make_training(**training), make_profile(warnings))
    assert not report.passed and report.failed_checks == [failed]
    assert RECOMMENDATIONS[failed] in report.recommendations


def test_verifier_reads_percent_targets_and_uses_r2_gap_for_regression() -> None:
    assert Verifier(None).verify(churn_requirement(("accuracy", 75)), make_training(), make_profile()).passed
    regression = {
        "task_type": "tabular_regression", "primary_metric": "rmse", "direction": "minimize",
        "test_metrics": {"rmse": 10.0, "r2": 0.80, "n_test_rows": 100},
        "baseline": Baseline(name="mean", metrics={"rmse": 30.0, "r2": 0.0}),
    }
    requirement = make_requirement("tabular_regression", "target", metrics=("rmse",))
    overfit = Verifier(None).verify(requirement, make_training(**regression, train_metrics={"rmse": 2.0, "r2": 0.97}))
    healthy = Verifier(None).verify(requirement, make_training(**regression, train_metrics={"rmse": 8.0, "r2": 0.90}))
    assert overfit.failed_checks == ["overfitting"] and "r2: train 0.970 vs test 0.800" in overfit.checks[-1].detail
    assert healthy.passed


def test_verifier_llm_review_is_optional() -> None:
    llm = FakeLLM({LLMReview: LLMReview(verdict="Solid model for a first campaign.", concerns=["Modest F1."],
                                        recommendations=["Collect a few more months of data."])})
    report = Verifier(llm).verify(churn_requirement(), make_training(), make_profile())
    assert report.passed and report.llm_verdict == "Solid model for a first campaign."
    assert report.llm_concerns == ["Modest F1."] and "Collect a few more months of data." in report.recommendations
    skipped = Verifier(FakeLLM({})).verify(churn_requirement(), make_training(), make_profile())
    assert skipped.passed and skipped.llm_verdict is None


@pytest.mark.parametrize(
    "task, make_df, target, models, baseline_name",
    [
        ("tabular_classification", classification_df, "target", ["logistic_regression"], "majority_class"),
        ("tabular_regression", regression_df, "target", ["ridge"], "mean"),
        ("time_series_forecasting", timeseries_df, "sales", ["ridge"], "seasonal_naive"),
        ("tabular_clustering", clustering_df, None, ["kmeans"], None),
    ],
)
def test_trainer_baselines_and_verifier_comparison(tmp_path: Path, task: str, make_df: Any, target: str | None,
                                                    models: list[str], baseline_name: str | None) -> None:
    result = run_automl(make_df(), task, target, models, tmp_path, budget=ONE_MODEL)
    assert (result.baseline.name if result.baseline else None) == baseline_name
    assert result.train_metrics  # in-sample scores for the overfitting check
    report = Verifier(None).verify(make_requirement(task, target), result)
    beats = next(c for c in report.checks if c.name == "beats_baseline")
    assert beats.passed, beats.detail


# --------------------------------------------------------------------------- 4. sandbox


def test_sandbox_runs_pandas_code() -> None:
    code = (
        "import pandas as pd\n"
        "rate = (df['churn'] == 'Yes').groupby(df['contract_type']).mean().round(3)\n"
        "print('rows:', len(df))\n"
        "result = rate"
    )
    result = run_code(code, CHURN_CSV)
    assert result.ok and result.stdout.strip() == "rows: 3000" and not result.truncated
    assert result.result == {"Month-to-month": 0.465, "One year": 0.2, "Two year": 0.101}


def test_sandbox_returns_dataframes_as_records_and_reports_errors() -> None:
    table = run_code("result = df[['customer_id', 'tenure_months']].head(3)", CHURN_CSV)
    assert table.ok and table.result[0] == {"customer_id": "CUST-00001", "tenure_months": table.result[0]["tenure_months"]}
    failed = run_code("x = 1\nresult = df['no_such_column'].mean()", CHURN_CSV)
    assert not failed.ok and failed.error == "KeyError: 'no_such_column' (line 2)"


@pytest.mark.parametrize(
    "code, reason",
    [
        ("import os\nresult = os.getcwd()", "import of 'os'"),
        ("import subprocess", "import of 'subprocess'"),
        ("from os import path", "import from 'os'"),
        ("import pandas.io.common as c", "import of 'pandas.io.common'"),
        ("from pandas import read_csv", "importing 'read_csv'"),
        ("result = open('secrets.txt').read()", "use of 'open'"),
        ("result = eval('1 + 1')", "use of 'eval'"),
        ("result = __import__('os')", "use of '__import__'"),
        ("result = getattr(df, 'shape')", "use of 'getattr'"),
        ("result = ().__class__.__bases__", "attribute '.__class__'"),
        ("result = df._data", "attribute '._data'"),
        ("result = pd.read_csv('other.csv')", "attribute '.read_csv'"),
        ("df.to_csv('copy.csv')", "attribute '.to_csv'"),
        ("result = df.query('tenure_months > 12')", "attribute '.query'"),
        ("result = np.load('x.npy')", "attribute '.load'"),
    ],
)
def test_sandbox_blocks_unsafe_code_before_running(code: str, reason: str) -> None:
    assert any(reason in problem for problem in check_code(code)), check_code(code)
    result = run_code(code, CHURN_CSV)
    assert not result.ok and result.error.startswith("Blocked by the sandbox") and result.duration_s < 1


def test_sandbox_kills_infinite_loops() -> None:
    result = run_code("while True:\n    pass", CHURN_CSV, timeout_s=3)
    assert not result.ok and "Timed out after 3s" in result.error and result.duration_s < 15


def test_sandbox_truncates_huge_output() -> None:
    result = run_code("print('x' * 100_000)\nresult = 'done'", CHURN_CSV)
    assert result.ok and result.truncated and len(result.stdout) == MAX_STDOUT_CHARS and result.result == "done"


def test_sandbox_child_gets_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "also-secret")
    env = _child_env()
    assert "LLM_API_KEY" not in env and "TAVILY_API_KEY" not in env and env["PYTHONIOENCODING"] == "utf-8"


# --------------------------------------------------------------------------- 5. analysis agent


def test_analysis_agent_repairs_failing_code_once(churn_profile: DatasetProfile) -> None:
    llm = FakeLLM({AnalysisCode: [
        AnalysisCode(code="result = df['tenure'].mean()", explanation="Average tenure."),
        AnalysisCode(code="result = round(float(df['tenure_months'].mean()), 2)", explanation="Average tenure in months."),
    ]})
    answer = AnalysisAgent(llm).ask("What is the average customer tenure?", churn_profile, CHURN_CSV)

    assert answer.ok and answer.attempts == 2 and isinstance(answer.result, float)
    assert answer.answer.startswith("Average tenure in months.") and "Result: " in answer.answer
    first_prompt, repair_prompt = (messages[1]["content"] for name, messages in llm.calls)
    assert "- tenure_months (numeric" in first_prompt and "CUST-00001" not in first_prompt  # columns only
    assert "KeyError: 'tenure'" in repair_prompt


def test_analysis_agent_gives_up_after_one_repair(churn_profile: DatasetProfile) -> None:
    llm = FakeLLM({AnalysisCode: AnalysisCode(code="import os\nresult = 1")})
    answer = AnalysisAgent(llm).ask("Anything", churn_profile, CHURN_CSV)
    assert not answer.ok and answer.attempts == 2 and llm.count("AnalysisCode") == 2
    assert "Blocked by the sandbox" in answer.error
    unavailable = AnalysisAgent(FakeLLM({})).ask("Anything", churn_profile, CHURN_CSV)
    assert not unavailable.ok and unavailable.attempts == 0


# --------------------------------------------------------------------------- 6. orchestrator


def churn_llm(**extra: Any) -> FakeLLM:
    responses = {
        RelevanceCheck: RelevanceCheck(relevant=True),
        UserRequirement: churn_requirement(),
        SufficiencyCheck: SufficiencyCheck(sufficient=True),
        RequirementSummary: RequirementSummary(summary="Predict churn."),
        ModelChoice: ModelChoice(candidates=[RankedCandidate(name="logistic_regression")]),
        BriefInsights: BriefInsights(insights=["Weight the minority class."]),
    }
    return FakeLLM(responses | extra)


def test_failed_verification_replans_at_most_once(tmp_path: Path) -> None:
    llm = churn_llm()
    orchestrator = Orchestrator(llm, tmp_path, budget=ONE_MODEL)
    failing = VerificationReport(passed=False, checks=[Check(name="metric_targets", passed=False, detail="f1 too low")],
                                 recommendations=["Add more informative features."])
    verify_calls: list[int] = []
    orchestrator.verifier.verify = lambda *args, **kwargs: verify_calls.append(1) or failing  # type: ignore[method-assign]

    orchestrator.start("Which customers will churn?", CHURN_CSV)
    ctx = orchestrator.run_until_blocked()

    states = [str(event.state) for event in ctx.history]
    assert ctx.state == RunState.DONE and ctx.plan_revision_count == 1 and len(verify_calls) == 2
    assert states.count("PLAN") == 2 and states.count("TRAIN") == 2 and states.count("EXPLAIN") == 1
    assert ctx.verification.passed is False and "did not pass all quality checks (metric_targets)" in ctx.message
    assert ctx.plan.budget_mode == "full"  # the revision asks for the full search
    second_plan_prompt = [m for name, m in llm.calls if name == "ModelChoice"][1][1]["content"]
    assert "- Add more informative features." in second_plan_prompt
    assert ctx.knowledge.generated_by == "llm" and "Weight the minority class." in second_plan_prompt


def test_real_verifier_report_is_stored(tmp_path: Path) -> None:
    orchestrator = Orchestrator(churn_llm(), tmp_path, budget=ONE_MODEL)
    run_id = orchestrator.start("Which customers will churn?", CHURN_CSV)
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.DONE and len(ctx.verification.checks) == 7
    saved = json.loads((tmp_path / run_id / "state.json").read_text(encoding="utf-8"))
    assert len(saved["verification"]["checks"]) == 7 and saved["knowledge"]["insights"]


def test_orchestrator_ask_answers_and_persists(tmp_path: Path) -> None:
    llm = churn_llm(AnalysisCode=None)
    llm.responses[AnalysisCode] = AnalysisCode(code="result = int((df['churn'] == 'Yes').sum())",
                                               explanation="Number of customers who churned.")
    orchestrator = Orchestrator(llm, tmp_path, budget=ONE_MODEL)
    run_id = orchestrator.start("Which customers will churn?", CHURN_CSV)

    answer = orchestrator.ask(run_id, "How many customers churned?")
    assert answer.ok and answer.result == 985 and "Number of customers who churned." in answer.answer
    saved = json.loads((tmp_path / run_id / "state.json").read_text(encoding="utf-8"))
    assert saved["analyses"][0]["result"] == 985 and saved["state"] == "INIT"  # asking does not advance the run

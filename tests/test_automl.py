"""AutoML engine tests: model zoo, tuner, trainer, Model Agent and the TRAIN state. Tiny data, few trials."""

import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.samplers import RandomSampler

from idss.agents.manager import Orchestrator, RunState
from idss.agents.model_agent import ModelAgent, ModelChoice, parse_constraints
from idss.agents.requirement_agent import RelevanceCheck, RequirementAgent, RequirementSummary, SufficiencyCheck
from idss.core.schemas import (
    BUDGET_PRESETS,
    DatasetSpec,
    LeaderboardEntry,
    ModelConstraints,
    ModelSpec,
    PerformanceMetric,
    ProblemSpec,
    RankedCandidate,
    TrainingBudget,
    TrainingResult,
    UserRequirement,
)
from idss.ml import model_zoo, tuner
from idss.ml.model_zoo import REGISTRY, ModelContext, get_candidate
from idss.ml.profiler import profile_dataset
from idss.ml.trainer import AutoMLTrainer, TrainingError, mark_constraint_violations, rank_entries, select_best
from idss.ml.ts_features import TimeSeriesFeatures
from idss.ml.tuner import TuneResult
from tests.fakes import FakeLLM

CHURN_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "customer_churn.csv"
TINY = TrainingBudget(max_candidates=2, n_trials_per_model=3, timeout_per_model_s=30, total_timeout_s=120)


# --------------------------------------------------------------------------- synthetic data


def classification_df(n: int = 240, minority: float = 0.35, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], n)
    score = 1.5 * x1 - x2 + np.where(region == "south", 0.8, 0.0) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region,
                         "target": np.where(score > np.quantile(score, 1 - minority), "yes", "no")})


def regression_df(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], n)
    y = 3 * x1 - 2 * x2 + np.where(region == "east", 2.0, 0.0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "target": y})


def clustering_df(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])
    points = centers[rng.integers(0, 3, n)] + rng.normal(0, 0.7, (n, 2))
    return pd.DataFrame({"f1": points[:, 0], "f2": points[:, 1], "size": rng.choice(["s", "m"], n)})


def timeseries_df(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    promo = (rng.random(n) < 0.2).astype(int)
    weekly = np.array([0, 2, 4, 6, 10, 25, 18])[dates.dayofweek]
    sales = 100 + 0.5 * np.arange(n) + weekly + 10 * promo + rng.normal(0, 2, n)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "promo": promo, "sales": sales.round(2)})


def make_requirement(task: str, target: str | None, *, metrics: tuple[str, ...] = (), constraints: tuple[str, ...] = (),
                     models: tuple[str, ...] = ()) -> UserRequirement:
    return UserRequirement(
        problem=ProblemSpec(task_type=task, description="Test problem.", constraints=list(constraints),
                            performance_metrics=[PerformanceMetric(name=m) for m in metrics]),
        datasets=[DatasetSpec(target_column=target)] if target else [],
        models=[ModelSpec(name=m) for m in models],
        confidence=1.0,
    )


def run_automl(df: pd.DataFrame, task: str, target: str | None, models: list[str], run_dir: Path, *,
               budget: TrainingBudget = TINY, constraints: tuple[str, ...] = (),
               trainer: AutoMLTrainer | None = None) -> TrainingResult:
    profile = profile_dataset(df, target=target, suggest_target=target is not None)
    requirement = RequirementAgent(None).repair(make_requirement(task, target, constraints=constraints), profile)
    plan = ModelAgent(None).plan(requirement, profile).model_copy(
        update={"candidates": [RankedCandidate(name=m) for m in models], "budget": budget}
    )
    return (trainer or AutoMLTrainer()).run(df, requirement, profile, run_dir, plan=plan)


# --------------------------------------------------------------------------- end-to-end per task


@pytest.mark.parametrize(
    "task, make_df, target, models, metric",
    [
        ("tabular_classification", classification_df, "target", ["logistic_regression", "lightgbm"], "f1"),
        ("tabular_regression", regression_df, "target", ["ridge", "lightgbm"], "rmse"),
        ("tabular_clustering", clustering_df, None, ["kmeans", "agglomerative"], "silhouette"),
        ("time_series_forecasting", timeseries_df, "sales", ["ridge", "lightgbm"], "rmse"),
    ],
)
def test_each_task_trains_saves_and_reloads(tmp_path: Path, task: str, make_df: Any, target: str | None,
                                            models: list[str], metric: str) -> None:
    df = make_df()
    result = run_automl(df, task, target, models, tmp_path)

    assert result.task_type == task and result.primary_metric == metric
    assert result.test_metrics[metric] is not None
    ok = [e for e in result.leaderboard if e.status == "ok"]
    assert len(ok) >= 2 and [e.rank for e in ok] == list(range(1, len(ok) + 1))
    [selected] = [e for e in result.leaderboard if e.selected]
    assert selected.model == result.best_model and selected.test_metrics == result.test_metrics
    if task == "time_series_forecasting":
        assert "seasonal_naive" in {e.model for e in result.leaderboard}  # baseline is always trained

    assert json.loads(Path(result.leaderboard_path).read_text(encoding="utf-8"))[0]["rank"] == 1
    assert json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))["model"] == result.best_model
    model = joblib.load(result.model_path)
    predictions = model.predict(df.tail(10))
    assert len(predictions) == 10
    if task == "tabular_classification":
        assert set(predictions) <= {"yes", "no"} and model.predict_proba(df.tail(10)).shape == (10, 2)
        assert result.test_metrics["positive_class"] == "yes"  # the minority class
    for key in ("train_time_s", "inference_ms_per_row", "model_size_kb", "n_features_after_preprocessing"):
        assert result.complexity[key] is not None


@pytest.mark.parametrize(
    "family, name", [(family, name) for family, candidates in REGISTRY.items() for name in candidates]
)
def test_every_zoo_candidate_builds_and_fits(family: str, name: str) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    task, y, extra = {
        "classification": ("tabular_classification", (X[:, 0] > 0).astype(int), {"n_classes": 2, "class_counts": (40, 40), "pos_label": 1}),
        "regression": ("tabular_regression", 2 * X[:, 0] + rng.normal(0, 0.1, 80), {}),
        "time_series": ("time_series_forecasting", 2 * X[:, 0], {"seasonal_period": 7,
                        "feature_names": ("extra__ts_lag_1", "extra__ts_lag_7", "num__a", "num__b")}),
        "clustering": ("tabular_clustering", None, {}),
    }[family]
    context = ModelContext(task_type=task, n_rows=80, **extra)
    candidate = REGISTRY[family][name]
    params = candidate.search_space(optuna.create_study(sampler=RandomSampler(seed=0)).ask(), context)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = candidate.build(params, context).fit(X, y)
    assert len(model.predict(X)) == 80


def test_class_imbalance_switches_on_weighting() -> None:
    imbalanced = ModelContext("tabular_classification", n_rows=100, n_classes=2, class_counts=(90, 10), pos_label=1)
    balanced = ModelContext("tabular_classification", n_rows=100, n_classes=2, class_counts=(50, 50), pos_label=1)
    build = lambda name, ctx: get_candidate("tabular_classification", name).build({}, ctx)  # noqa: E731

    assert build("logistic_regression", imbalanced).class_weight == "balanced"
    assert build("lightgbm", imbalanced).get_params()["class_weight"] == "balanced"
    assert build("xgboost", imbalanced).get_params()["scale_pos_weight"] == pytest.approx(9.0)
    assert build("logistic_regression", balanced).class_weight is None
    assert build("xgboost", balanced).get_params()["scale_pos_weight"] == 1.0


# --------------------------------------------------------------------------- leakage


def test_test_split_is_untouched_until_final_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, int]] = []
    real_tune = tuner.tune

    def spy(candidate: Any, X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, metric: str, **kwargs: Any) -> TuneResult:
        seen.append((len(X_train), len(X_valid)))
        return real_tune(candidate, X_train, y_train, X_valid, y_valid, metric, **kwargs)

    monkeypatch.setattr(tuner, "tune", spy)
    result = run_automl(classification_df(240), "tabular_classification", "target", ["logistic_regression", "lightgbm"], tmp_path)

    assert seen == [(168, 48), (168, 48)]  # tuning only ever sees the 70% train and 20% valid splits
    assert result.test_set_accesses == ["final_evaluation"]  # the 10% test split is read exactly once
    assert result.test_metrics["n_test_rows"] == 24


def test_time_series_features_use_only_past_values() -> None:
    df = timeseries_df(60)
    ts = TimeSeriesFeatures("date", "sales", lags=(1, 7), windows=(7,))
    features = ts.feature_names()
    before = ts.fit_transform(df)
    changed = df.copy()
    changed.loc[30, "sales"] += 1000
    after = ts.fit_transform(changed)

    pd.testing.assert_frame_equal(before.loc[:30, features], after.loc[:30, features])  # row 30 never sees itself
    assert after.loc[31, "ts_lag_1"] == changed.loc[30, "sales"] and "sales" not in before.columns

    ts.fit(df.iloc[:40])  # a later window continues from the fitted history
    assert ts.transform(df.iloc[40:]).iloc[0]["ts_lag_1"] == df.loc[39, "sales"]


# --------------------------------------------------------------------------- budget & failures


def test_total_time_budget_skips_remaining_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts: list[float] = []
    real_tune = tuner.tune

    def spy(*args: Any, **kwargs: Any) -> TuneResult:
        timeouts.append(kwargs["timeout_s"])
        return real_tune(*args, **kwargs)

    monkeypatch.setattr(tuner, "tune", spy)
    ticks = itertools.count(0, 50)  # every budget check sees 50 more simulated seconds
    budget = TrainingBudget(max_candidates=3, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)
    result = run_automl(classification_df(), "tabular_classification", "target",
                        ["logistic_regression", "lightgbm", "random_forest"], tmp_path,
                        budget=budget, trainer=AutoMLTrainer(clock=lambda: next(ticks)))

    assert {e.model: e.status for e in result.leaderboard} == {
        "logistic_regression": "ok", "lightgbm": "skipped", "random_forest": "skipped",
    }
    assert timeouts == [10]  # min(per-model 30s, 10s left of the 60s budget)
    assert any("time budget" in note for note in result.warnings)


def test_tuner_respects_its_timeout() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(400, 5)), rng.integers(0, 2, 400)
    context = ModelContext("tabular_classification", n_rows=400, n_classes=2, class_counts=(200, 200), pos_label=1)
    start = time.perf_counter()
    result = tuner.tune(get_candidate("tabular_classification", "random_forest"), X, y, X, y, "f1",
                        n_trials=10_000, timeout_s=1.0, context=context)
    assert time.perf_counter() - start < 10 and 0 < result.n_trials < 10_000


def test_failing_trials_are_pruned_not_raised() -> None:
    context = ModelContext("tabular_regression", n_rows=50)
    X, y = np.random.default_rng(0).normal(size=(50, 3)), np.arange(50.0)
    broken = model_zoo.Candidate("broken", "regression", "always fails",
                                 build=lambda p, ctx: (_ for _ in ()).throw(RuntimeError("bad config")))
    result = tuner.tune(broken, X, y, X, y, "rmse", n_trials=3, timeout_s=10, context=context)
    assert result.best_score is None and result.n_failed == 3 and "RuntimeError: bad config" in result.errors[0]


def test_all_candidates_failing_raises_training_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuner, "tune", lambda candidate, *args, **kwargs: TuneResult(
        candidate=candidate.name, metric="f1", direction="maximize", errors=["boom"]))
    with pytest.raises(TrainingError, match="boom"):
        run_automl(classification_df(), "tabular_classification", "target", ["logistic_regression"], tmp_path)


# --------------------------------------------------------------------------- constraints


def test_constraint_violations_are_marked_and_avoided() -> None:
    entries = [
        LeaderboardEntry(model="fast_but_worse", status="ok", valid_score=0.80, inference_ms_per_row=0.5, train_time_s=1),
        LeaderboardEntry(model="slow_but_better", status="ok", valid_score=0.90, inference_ms_per_row=8.0, train_time_s=1),
        LeaderboardEntry(model="broken", status="failed", error="x"),
    ]
    mark_constraint_violations(entries, ModelConstraints(max_inference_ms_per_row=5))
    ranked = rank_entries(entries, "maximize")
    best, notes = select_best(ranked)

    assert [e.model for e in ranked] == ["slow_but_better", "fast_but_worse", "broken"] and ranked[2].rank is None
    assert ranked[0].violations == ["inference 8 ms/row > 5 ms limit"] and ranked[1].violations == []
    assert best.model == "fast_but_worse" and "break the constraints" in notes[0]


def test_when_every_model_violates_the_best_is_still_returned(tmp_path: Path) -> None:
    result = run_automl(classification_df(), "tabular_classification", "target", ["logistic_regression", "lightgbm"],
                        tmp_path, constraints=("Inference must take under 0.000001 ms per row",))
    assert all(e.violations for e in result.leaderboard if e.status == "ok")
    assert result.best_model == next(e.model for e in result.leaderboard if e.rank == 1)
    assert any("No model meets every constraint" in note for note in result.warnings)


@pytest.mark.parametrize(
    "text, inference_ms, train_s",
    [
        ("inference < 5 ms", 5.0, None),
        ("Predictions must be returned within 0.2 s", 200.0, None),
        ("Training must take under 30 minutes", None, 1800.0),
        ("train in less than 1 hour", None, 3600.0),
        ("The model must be accurate", None, None),
    ],
)
def test_parse_constraints(text: str, inference_ms: float | None, train_s: float | None) -> None:
    constraints = parse_constraints([text])
    assert (constraints.max_inference_ms_per_row, constraints.max_train_time_s) == (inference_ms, train_s)


# --------------------------------------------------------------------------- Model Agent


@pytest.fixture(scope="module")
def clf_profile() -> Any:
    return profile_dataset(classification_df(), target="target")


def test_model_agent_keeps_only_zoo_names(clf_profile: Any) -> None:
    llm = FakeLLM({ModelChoice: ModelChoice(budget_mode="full", candidates=[
        RankedCandidate(name="XGBoost", reason="strong on tabular data"),
        RankedCandidate(name="svm", reason="not allowed"),
        RankedCandidate(name="Random Forest", reason="robust"),
        RankedCandidate(name="xgboost", reason="duplicate"),
    ])})
    plan = ModelAgent(llm).plan(make_requirement("tabular_classification", "target"), clf_profile)
    names = [c.name for c in plan.candidates]

    assert names[:2] == ["xgboost", "random_forest"] and plan.candidates[0].reason == "strong on tabular data"
    assert sorted(names) == sorted(model_zoo.candidate_names("tabular_classification"))  # rest filled from defaults
    assert plan.source == "llm" and plan.budget_mode == "full" and plan.budget == BUDGET_PRESETS["full"]
    assert any("'svm' is not in the model zoo" in note for note in plan.notes)
    model_selection_prompt = llm.calls[0][1][1]["content"]
    assert "- lightgbm:" in model_selection_prompt and "x1 | numeric" in model_selection_prompt


@pytest.mark.parametrize("llm", [None, FakeLLM({}), FakeLLM({ModelChoice: ModelChoice(candidates=[RankedCandidate(name="svm")])})])
def test_model_agent_falls_back_to_default_ranking(clf_profile: Any, llm: Any) -> None:
    plan = ModelAgent(llm).plan(make_requirement("tabular_classification", "target"), clf_profile)
    assert plan.source == "default" and plan.budget_mode == "fast"
    assert [c.name for c in plan.candidates] == model_zoo.DEFAULT_RANKING["classification"]


def test_model_agent_puts_user_requested_models_first(clf_profile: Any) -> None:
    requirement = make_requirement("tabular_classification", "target", models=("Extra Trees",))
    plan = ModelAgent(None).plan(requirement, clf_profile)
    assert plan.candidates[0].name == "extra_trees" and plan.candidates[0].reason == "Requested by the user."


def test_model_agent_applies_constraints(clf_profile: Any) -> None:
    llm = FakeLLM({ModelChoice: ModelChoice(candidates=[RankedCandidate(name=n) for n in ("xgboost", "lightgbm", "random_forest")])})
    requirement = make_requirement("tabular_classification", "target", constraints=(
        "Inference must be under 5 ms per prediction", "Training should finish within 2 minutes", "The model must be explainable",
    ))
    plan = ModelAgent(llm).plan(requirement, clf_profile)

    assert plan.constraints == ModelConstraints(max_inference_ms_per_row=5.0, max_train_time_s=120.0, prefer_interpretable=True)
    assert plan.budget.total_timeout_s == pytest.approx(96.0) and plan.budget.timeout_per_model_s == pytest.approx(32.0)
    assert [c.name for c in plan.candidates[:3]] == ["xgboost", "lightgbm", "logistic_regression"]  # explainable model added


def test_model_agent_primary_metric(clf_profile: Any) -> None:
    requirement = make_requirement("tabular_classification", "target", metrics=("AUC",))
    assert ModelAgent(None).plan(requirement, clf_profile).primary_metric == "roc_auc"
    clustering = RequirementAgent(None).repair(make_requirement("tabular_clustering", None), None)
    plan = ModelAgent(None).plan(clustering, profile_dataset(clustering_df(), suggest_target=False))
    assert plan.primary_metric == "silhouette" and plan.target is None  # rand_index needs labels we don't have


# --------------------------------------------------------------------------- orchestrator


def churn_llm(**extra: Any) -> FakeLLM:
    requirement = make_requirement("tabular_classification", "churn", metrics=("f1",))
    responses = {
        RelevanceCheck: RelevanceCheck(relevant=True),
        UserRequirement: requirement,
        SufficiencyCheck: SufficiencyCheck(sufficient=True),
        RequirementSummary: RequirementSummary(summary="Predict churn."),
        ModelChoice: ModelChoice(candidates=[RankedCandidate(name="logistic_regression", reason="simple baseline")]),
    }
    return FakeLLM(responses | extra)


def test_orchestrator_trains_and_reaches_done(tmp_path: Path) -> None:
    budget = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)
    orchestrator = Orchestrator(churn_llm(), tmp_path, budget=budget)
    run_id = orchestrator.start("Which customers will churn?", CHURN_CSV)
    ctx = orchestrator.run_until_blocked()

    assert ctx.state == RunState.DONE and ctx.errors == []
    assert ctx.plan.source == "llm" and ctx.plan.candidates[0].name == "logistic_regression"
    assert ctx.training.best_model == "logistic_regression" and Path(ctx.training.model_path).is_file()
    assert ctx.training.test_set_accesses == ["final_evaluation"]
    messages = {str(e.from_state): e.message for e in ctx.history if e.from_state}
    assert messages["TRAIN"].startswith("Trained 1 model(s)") and "Verification" in messages["VERIFY"]
    assert messages["EXPLAIN"].startswith("Explained the model") and "Decision brief written by" in messages["INSIGHT"]
    assert messages["REPORT"] == "Wrote report.pdf and report.md." and Path(ctx.report_paths["pdf"]).is_file()
    assert "logistic_regression" in ctx.message
    saved = json.loads((tmp_path / run_id / "state.json").read_text(encoding="utf-8"))
    assert saved["training"]["best_model"] == "logistic_regression"


def test_training_error_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = Orchestrator(churn_llm(), tmp_path, budget=TINY)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise TrainingError("No candidate model could be trained (boom).")

    monkeypatch.setattr(orchestrator.trainer, "run", explode)
    orchestrator.start("Which customers will churn?", CHURN_CSV)
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.FAILED and ctx.errors[0].startswith("TRAIN failed: TrainingError")

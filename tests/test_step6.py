"""Step 6 tests: explanations, what-if, recommendations, grounded insights, reports and the offline run.
Tiny synthetic data, one model with 2 trials, SHAP on at most 100 rows. The LLM is always faked."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

import idss.ml.explain as explain_module
from idss.agents.insight_agent import (
    InsightAgent,
    InsightDraft,
    build_facts,
    fact_numbers,
    template_insight,
    unsupported_numbers,
)
from idss.agents.manager import Orchestrator, RunContext, RunState
from idss.agents.model_agent import ModelAgent
from idss.agents.requirement_agent import RelevanceCheck, RequirementAgent
from idss.agents.verifier import Verifier
from idss.core.schemas import (
    DatasetProfile,
    FilterCondition,
    RankedCandidate,
    SegmentImpact,
    TrainingBudget,
    TrainingResult,
    UserRequirement,
    WhatIfResult,
)
from idss.decision.recommender import (
    candidate_values,
    counterfactual,
    example_what_if,
    infer_actionable,
    recommend,
    risk_tiers,
    tier_cutoffs,
)
from idss.decision.report_builder import build_markdown, build_pdf
from idss.decision.what_if import WhatIfError, model_output, simulate, simulate_segment
from idss.ml.explain import ModelExplainer, explain_global, explain_run, original_feature_names
from idss.ml.profiler import profile_dataset
from idss.ml.trainer import AutoMLTrainer, TrainedModel
from tests.fakes import FakeLLM
from tests.test_automl import classification_df, clustering_df, make_requirement, regression_df, timeseries_df

ONE = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)
SHAP_ROWS = 100


@dataclass
class Trained:
    df: pd.DataFrame
    requirement: UserRequirement
    profile: DatasetProfile
    result: TrainingResult
    model: TrainedModel
    run_dir: Path


def train(run_dir: Path, df: pd.DataFrame, task: str, target: str | None, model_name: str,
          actionable: tuple[str, ...] = ()) -> Trained:
    profile = profile_dataset(df, target=target, suggest_target=target is not None)
    requirement = RequirementAgent(None).repair(make_requirement(task, target), profile)
    requirement.actionable_features = list(actionable)
    plan = ModelAgent(None).plan(requirement, profile).model_copy(
        update={"candidates": [RankedCandidate(name=model_name)], "budget": ONE})
    result = AutoMLTrainer().run(df, requirement, profile, run_dir, plan=plan)
    return Trained(df, requirement, profile, result, joblib.load(result.model_path), run_dir)


@pytest.fixture(autouse=True)
def _small_shap_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain_module, "MAX_EXPLAIN_ROWS", SHAP_ROWS)


@pytest.fixture(scope="module")
def clf_lgbm(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train(tmp_path_factory.mktemp("lgbm"), classification_df(300, minority=0.3), "tabular_classification", "target", "lightgbm")


@pytest.fixture(scope="module")
def clf_lr(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train(tmp_path_factory.mktemp("lr"), classification_df(300, minority=0.2), "tabular_classification", "target",
                 "logistic_regression", actionable=("x1",))


@pytest.fixture(scope="module")
def reg(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train(tmp_path_factory.mktemp("reg"), regression_df(240), "tabular_regression", "target", "ridge")


@pytest.fixture(scope="module")
def ts(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train(tmp_path_factory.mktemp("ts"), timeseries_df(160), "time_series_forecasting", "sales", "ridge")


@pytest.fixture(scope="module")
def clu(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return train(tmp_path_factory.mktemp("clu"), clustering_df(240), "tabular_clustering", None, "kmeans")


@pytest.fixture(scope="module")
def clf_facts(clf_lr: Trained, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(explain_module, "MAX_EXPLAIN_ROWS", SHAP_ROWS)
        explanation, explainer = explain_run(clf_lr.model, clf_lr.df, clf_lr.result, clf_lr.profile,
                                             tmp_path_factory.mktemp("facts"))
        recommendations = recommend(clf_lr.model, clf_lr.df, clf_lr.requirement, clf_lr.profile, clf_lr.result,
                                    explanation, explainer)
    what_if = example_what_if(clf_lr.model, clf_lr.df, clf_lr.profile, recommendations, explanation)
    verification = Verifier(None).verify(clf_lr.requirement, clf_lr.result, clf_lr.profile)
    return build_facts(clf_lr.requirement, clf_lr.result, explanation, recommendations, verification, None, what_if,
                       clf_lr.profile.warnings)


# --------------------------------------------------------------------------- explanations


def test_explain_tree_model_uses_shap(clf_lgbm: Trained) -> None:
    explanation = explain_global(clf_lgbm.model, clf_lgbm.df, "tabular_classification")
    by_feature = {f.feature: f for f in explanation.features}

    assert explanation.method == "shap_tree" and explanation.n_samples == SHAP_ROWS
    assert set(by_feature) <= {"x1", "x2", "region"} and explanation.features[0].feature == "x1"
    assert sum(f.share for f in explanation.features) == pytest.approx(1.0, abs=0.01)
    assert by_feature["x1"].direction == "increases" and by_feature["x2"].direction == "decreases"
    assert explanation.output == "probability that target = 'yes'"


def test_explainer_max_rows_caps_the_sample(clf_lgbm: Trained, ts: Trained, tmp_path: Path) -> None:
    """The public demo explains fewer rows (memory); the default stays MAX_EXPLAIN_ROWS."""
    assert len(ModelExplainer(clf_lgbm.model, clf_lgbm.df).raw) == SHAP_ROWS
    capped = ModelExplainer(clf_lgbm.model, clf_lgbm.df, max_rows=40)
    assert len(capped.raw) == len(capped.Xt) == 40
    assert capped.global_explanation().n_samples == 40
    assert len(ModelExplainer(ts.model, ts.df, max_rows=30).raw) == 30  # time series: the latest rows
    explanation, explainer = explain_run(clf_lgbm.model, clf_lgbm.df, clf_lgbm.result, clf_lgbm.profile, tmp_path,
                                         max_rows=25)
    assert explanation.global_explanation.n_samples == 25 and explainer is not None and len(explainer.raw) == 25
    with pytest.raises(ValueError, match="at least 1"):
        ModelExplainer(clf_lgbm.model, clf_lgbm.df, max_rows=0)


def test_explain_linear_model_uses_shap(clf_lr: Trained) -> None:
    explanation = explain_global(clf_lr.model, clf_lr.df)
    by_feature = {f.feature: f for f in explanation.features}
    assert explanation.method == "shap_linear"
    assert by_feature["x1"].direction == "increases" and by_feature["x2"].direction == "decreases"
    assert "region = 'south'" in by_feature["region"].detail  # south raises the score in the generator


@pytest.mark.parametrize("break_at", ["create", "compute"])
def test_explain_falls_back_to_permutation_importance(clf_lgbm: Trained, monkeypatch: pytest.MonkeyPatch,
                                                      break_at: str) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SHAP unavailable")

    if break_at == "create":
        monkeypatch.setattr(explain_module.shap, "TreeExplainer", broken)
    else:
        monkeypatch.setattr(ModelExplainer, "_shap_values", broken)
    explanation = explain_global(clf_lgbm.model, clf_lgbm.df)
    assert explanation.method == "permutation" and explanation.features[0].feature == "x1"
    assert (break_at == "create") or any("SHAP failed" in note for note in explanation.notes)


def test_derived_columns_fold_back_to_original_names(clf_lgbm: Trained, ts: Trained) -> None:
    names = [str(n) for n in clf_lgbm.model.pipeline.named_steps["preprocess"].get_feature_names_out()]
    origins = original_feature_names(clf_lgbm.model)
    assert len(origins) == len(names)
    one_hot = [origin for name, origin in zip(names, origins, strict=True) if name.startswith("cat__")]
    assert len(one_hot) >= 2 and set(one_hot) == {"region"}

    ts_origins = set(original_feature_names(ts.model))
    assert {"sales (history)", "date", "promo"} <= ts_origins  # lags, calendar parts and one-hot folded back
    assert not any("ts_" in origin or "__" in origin for origin in ts_origins)


# --------------------------------------------------------------------------- what-if


def test_what_if_single_row(clf_lr: Trained) -> None:
    row = clf_lr.df.iloc[0].to_dict()
    result = simulate(clf_lr.model, row, {"x1": row["x1"] + 1.0}, clf_lr.profile)
    assert 0 <= result.before <= 1 and result.after > result.before and result.warnings == []
    assert result.delta == pytest.approx(result.after - result.before, abs=1e-6)

    far = simulate(clf_lr.model, row, {"x1": 100.0}, clf_lr.profile)
    assert any("outside the observed range" in w for w in far.warnings)
    unseen = simulate(clf_lr.model, row, {"region": "mars"}, clf_lr.profile)
    assert any("never seen in training" in w for w in unseen.warnings)
    assert simulate(clf_lr.model, row, {"x1": {"pct": 10}}, clf_lr.profile).after is not None


@pytest.mark.parametrize(
    "changes",
    [{"x1": "high"}, {"region": {"pct": 10}}, {"unknown": 1}, {"target": "yes"}, {}, {"x1": {"times": 2}}],
    ids=["text-for-number", "pct-on-category", "unknown-column", "target-column", "empty", "bad-change-kind"],
)
def test_what_if_rejects_invalid_changes(clf_lr: Trained, changes: dict[str, Any]) -> None:
    with pytest.raises(WhatIfError):
        simulate(clf_lr.model, clf_lr.df.iloc[0].to_dict(), changes, clf_lr.profile)


def test_what_if_segment_with_structured_filters(clf_lr: Trained) -> None:
    df = clf_lr.df
    filters = [{"column": "region", "op": "==", "value": "south"}, {"column": "x1", "op": ">", "value": 0}]
    impact = simulate_segment(clf_lr.model, df, filters, {"x1": {"add": 0.5}}, clf_lr.profile)
    assert impact.count == int(((df["region"] == "south") & (df["x1"] > 0)).sum())
    assert impact.avg_after > impact.avg_before and impact.delta == pytest.approx(impact.avg_after - impact.avg_before, abs=1e-6)

    members = simulate_segment(clf_lr.model, df, [FilterCondition(column="region", op="in", value=["north", "east"])],
                               {"x2": 0.0}, clf_lr.profile)
    assert members.count == int(df["region"].isin(["north", "east"]).sum())
    empty = simulate_segment(clf_lr.model, df, [{"column": "x1", "op": ">", "value": 1e9}], {"x1": 0.0}, clf_lr.profile)
    assert empty.count == 0 and empty.warnings == ["No rows match the filters."]


@pytest.mark.parametrize(
    "filters",
    [
        [{"column": "region", "op": "~=", "value": "south"}],
        [{"column": "nope", "op": "==", "value": 1}],
        [{"column": "region", "op": "<", "value": 3}],
        [{"column": "region", "op": "in", "value": "south"}],
        [{"column": "region", "op": "==", "value": "south'); import os; ('"}],
    ],
    ids=["unknown-op", "unknown-column", "order-on-text", "in-without-list", "injection-is-just-a-value"],
)
def test_segment_filters_reject_bad_input(clf_lr: Trained, filters: list[dict[str, Any]]) -> None:
    if filters[0]["value"] == "south'); import os; ('":  # never evaluated: simply matches nothing
        assert simulate_segment(clf_lr.model, clf_lr.df, filters, {"x1": 0.0}, clf_lr.profile).count == 0
        return
    with pytest.raises(WhatIfError):
        simulate_segment(clf_lr.model, clf_lr.df, filters, {"x1": 0.0}, clf_lr.profile)


def test_time_series_what_if_needs_history(ts: Trained) -> None:
    last = ts.df.iloc[-1].to_dict()
    with pytest.raises(WhatIfError, match="history"):
        simulate(ts.model, last, {"promo": 1}, ts.profile)
    result = simulate(ts.model, last, {"promo": 1 - int(last["promo"])}, ts.profile, history=ts.df.iloc[:-1])
    assert isinstance(result.before, float) and result.delta is not None


# --------------------------------------------------------------------------- recommendations


def test_counterfactual_lowers_risk_using_actionable_features_only(clf_lr: Trained) -> None:
    df, model = clf_lr.df, clf_lr.model
    assert infer_actionable(model, clf_lr.profile, clf_lr.requirement) == ["x1"]  # from requirement.actionable_features
    threshold = clf_lr.result.decision_threshold or 0.5
    worst = df.iloc[[int(np.argmax(model_output(model, df)))]]
    grids = candidate_values(df, ["x1", "x2", "region"], clf_lr.profile)

    single = counterfactual(model, worst, ["x1"], grids, threshold)
    assert single.success and single.probability_after < threshold < single.probability_before
    assert [c.feature for c in single.changes] == ["x1"] and df["x1"].min() <= single.changes[0].after <= df["x1"].max()

    several = counterfactual(model, worst, ["x1", "x2", "region"], grids, threshold)
    assert several.probability_after < several.probability_before and 1 <= len(several.changes) <= 3
    for change in several.changes:
        if change.feature == "region":
            assert change.after in set(df["region"])
        else:
            assert df[change.feature].min() <= change.after <= df[change.feature].max()

    impossible = counterfactual(model, worst, ["x1", "x2", "region"], grids, threshold=0.0)  # p < 0 can never happen
    assert not impossible.success and len(impossible.changes) <= 3  # never more than 3 changes


def test_infer_actionable_excludes_ids_dates_and_demographics() -> None:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n)], "age": rng.integers(18, 70, n), "gender": rng.choice(["f", "m"], n),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "price": rng.normal(50, 10, n).round(2), "plan": rng.choice(["basic", "pro"], n), "target": rng.choice(["y", "n"], n),
    })
    profile = profile_dataset(df, target="target")
    stub = SimpleNamespace(feature_columns=[c for c in df.columns if c != "target"], target="target")
    assert infer_actionable(stub, profile, None) == ["price", "plan"]
    requirement = make_requirement("tabular_classification", "target")
    requirement.actionable_features = ["AGE", "not_a_column"]
    assert infer_actionable(stub, profile, requirement) == ["age"]  # the user's choice wins, matched case-insensitively


def test_risk_tiers_cover_every_record(clf_lr: Trained, tmp_path: Path) -> None:
    explanation, explainer = explain_run(clf_lr.model, clf_lr.df, clf_lr.result, clf_lr.profile, tmp_path)
    result = recommend(clf_lr.model, clf_lr.df, clf_lr.requirement, clf_lr.profile, clf_lr.result, explanation, explainer)
    threshold = clf_lr.result.decision_threshold or 0.5

    high, medium, low = result.risk_tiers
    cutoffs = result.tier_cutoffs
    assert [t.name for t in result.risk_tiers] == ["High", "Medium", "Low"]
    assert sum(t.count for t in result.risk_tiers) == len(clf_lr.df)
    assert sum(t.share for t in result.risk_tiers) == pytest.approx(1.0, abs=0.002)
    assert cutoffs.medium == pytest.approx(threshold, abs=1e-4) and cutoffs.high >= cutoffs.medium
    assert cutoffs.high == pytest.approx(max(threshold, cutoffs.percentile_value), abs=1e-4)
    assert high.min_probability == cutoffs.high and medium.max_probability == cutoffs.high
    assert medium.min_probability == low.max_probability == cutoffs.medium
    assert high.share <= 0.31  # never more than ~30% of records in High
    prioritise = result.recommendations[0]
    assert prioritise.evidence["high_tier_count"] == high.count and prioritise.evidence["high_cutoff"] == cutoffs.high
    assert result.recommendations and all(r.evidence for r in result.recommendations)
    assert result.counterfactual_example is not None and result.counterfactual_example.success


@pytest.mark.parametrize(
    "threshold, high_cutoff, counts",
    [(0.2, 0.6993, (300, 500, 200)),  # low tuned threshold: the 70th percentile caps High at the top 30%
     (0.85, 0.85, (150, 0, 850))],    # high threshold: it wins, so Medium is empty
)
def test_tier_cutoffs_use_threshold_and_percentile(threshold: float, high_cutoff: float,
                                                   counts: tuple[int, int, int]) -> None:
    probability = np.arange(1000) / 1000
    cutoffs = tier_cutoffs(probability, threshold)
    assert cutoffs.medium == threshold and cutoffs.high == pytest.approx(high_cutoff)
    assert cutoffs.percentile == 70 and cutoffs.percentile_value == pytest.approx(0.6993)
    explanation = SimpleNamespace(global_explanation=SimpleNamespace(features=[]))
    tiers = risk_tiers(probability, cutoffs, None, explanation)
    assert tuple(t.count for t in tiers) == counts


def test_regression_levers_have_evidence(reg: Trained, tmp_path: Path) -> None:
    explanation, explainer = explain_run(reg.model, reg.df, reg.result, reg.profile, tmp_path)
    result = recommend(reg.model, reg.df, reg.requirement, reg.profile, reg.result, explanation, explainer)
    assert result.actionable_features == ["x1", "x2"]  # 'region' is a location, not a lever
    assert result.recommendations and all("delta_increase" in r.evidence for r in result.recommendations)
    actions = {r.action.split(" by ")[0] for r in result.recommendations}
    assert actions == {"Increase x1", "Decrease x2"}  # y = 3*x1 - 2*x2 + ...
    assert all(r.expected_impact > 0 for r in result.recommendations)
    # x1 is centred on zero, so a percentage change would be meaningless: the step is a share of its std.
    assert "standard deviation" in result.recommendations[0].action
    assert any("higher target is better" in note for note in result.notes)


# --------------------------------------------------------------------------- insight agent


def grounded_draft(facts: dict[str, Any], extra: str = "") -> InsightDraft:
    f1 = facts["model"]["test_metrics"]["f1"]
    high = next(t for t in facts["risk_tiers"] if t["name"] == "High")
    return InsightDraft(
        executive_summary=f"The model reaches an F1 score of {f1:.2f} on held-out data. {extra}".strip(),
        key_findings=[f"{high['count']} records are high risk.", f"That is about {high['share']:.0%} of all records.",
                      "x1 is the strongest driver."],
        recommended_actions=["Contact the high-risk group first."],
        risks_and_limitations=["The model shows associations, not causes."],
        next_steps=["Run a small pilot with a comparison group."],
    )


def test_grounded_llm_insight_is_accepted(clf_facts: dict[str, Any]) -> None:
    llm = FakeLLM({InsightDraft: grounded_draft(clf_facts)})
    insight = InsightAgent(llm).generate(clf_facts)
    assert insight.generated_by == "llm" and insight.grounding.passed and insight.grounding.attempts == 1
    assert "JSON" in llm.calls[0][1][0]["content"] and "never invent" in llm.calls[0][1][0]["content"].lower()


def test_invented_number_triggers_one_regeneration_then_template(clf_facts: dict[str, Any]) -> None:
    invented = grounded_draft(clf_facts, "Churn will fall by 37.5% next quarter.")
    llm = FakeLLM({InsightDraft: [invented, invented]})
    insight = InsightAgent(llm).generate(clf_facts)

    assert insight.generated_by == "template" and llm.count("InsightDraft") == 2
    assert not insight.grounding.passed and insight.grounding.attempts == 2 and "37.5%" in insight.grounding.unsupported_numbers
    retry_prompt = llm.calls[1][1][1]["content"]
    assert "not in the facts" in retry_prompt and "37.5%" in retry_prompt


def test_regeneration_can_fix_the_draft(clf_facts: dict[str, Any]) -> None:
    llm = FakeLLM({InsightDraft: [grounded_draft(clf_facts, "Revenue rises by 12.34%."), grounded_draft(clf_facts)]})
    insight = InsightAgent(llm).generate(clf_facts)
    assert insight.generated_by == "llm" and insight.grounding.passed and insight.grounding.attempts == 2


def test_template_insight_is_complete_and_grounded(clf_facts: dict[str, Any]) -> None:
    insight = template_insight(clf_facts)
    assert insight.generated_by == "template" and insight.executive_summary
    assert 3 <= len(insight.key_findings) <= 5
    assert insight.recommended_actions and insight.risks_and_limitations and insight.next_steps
    text = " ".join([insight.executive_summary, *insight.key_findings, *insight.recommended_actions,
                     *insight.risks_and_limitations, *insight.next_steps])
    assert unsupported_numbers(text, fact_numbers(clf_facts)) == []  # the template never invents numbers either
    assert InsightAgent(None).generate(clf_facts).generated_by == "template"


@pytest.mark.parametrize(
    "text, ok",
    [("F1 is 0.62", True), ("about 62%", True), ("risk drops by 0.18", True), ("a change of -0.18", True),
     ("we have 3 actions", True), ("1,234 customers", True), ("37.5% fewer", False), ("worth 999", False)],
)
def test_number_grounding_rules(text: str, ok: bool) -> None:
    facts = {"f1": 0.6231, "delta": -0.18, "customers": 1234}
    assert (unsupported_numbers(text, fact_numbers(facts)) == []) is ok


# --------------------------------------------------------------------------- reports


def run_context(t: Trained, **extra: Any) -> RunContext:
    now = datetime.now(UTC)
    return RunContext(run_id="test-run", question="What drives the outcome?", data_path="data.csv", created_at=now,
                      updated_at=now, requirement=t.requirement, profile=t.profile, training=t.result, **extra)


@pytest.mark.parametrize(
    "name, marker",
    [("clf_lr", "### Risk tiers"), ("reg", "](charts/actual_vs_pred.png)"), ("ts", "### Forecast (next 30 periods)"),
     ("clu", "### Segments")],
)
def test_reports_for_every_task(name: str, marker: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    t: Trained = request.getfixturevalue(name)
    explanation, explainer = explain_run(t.model, t.df, t.result, t.profile, tmp_path)
    recommendations = recommend(t.model, t.df, t.requirement, t.profile, t.result, explanation, explainer)
    what_if = example_what_if(t.model, t.df, t.profile, recommendations, explanation)
    verification = Verifier(None).verify(t.requirement, t.result, t.profile)
    insight = template_insight(build_facts(t.requirement, t.result, explanation, recommendations, verification, None, what_if))
    ctx = run_context(t, verification=verification, explanation=explanation, recommendations=recommendations,
                      what_if_example=what_if, insight=insight)

    pdf = build_pdf(ctx, insight, explanation.charts, tmp_path / "report.pdf")
    markdown = build_markdown(ctx, insight, explanation.charts, tmp_path / "report.md")
    assert pdf.read_bytes()[:4] == b"%PDF" and pdf.stat().st_size > 5_000
    text = markdown.read_text(encoding="utf-8")
    for heading in ("# Intelligent Decision Support System Using Generative AI and Machine Learning",
                    "## 1. Executive summary", "## 2. Data overview", "## 3. Model results", "## 4. Verification",
                    "## 5. Key drivers", "## 6. Recommendations", "## 7. What-if example", "## Appendix"):
        assert heading in text
    assert "![feature_importance](charts/feature_importance.png)" in text and marker in text


def test_report_handles_missing_charts_and_sections(clf_lr: Trained, tmp_path: Path) -> None:
    ctx = run_context(clf_lr)  # no verification, explanation, recommendations or insight
    charts = {"feature_importance": str(tmp_path / "missing.png")}
    pdf = build_pdf(ctx, None, charts, tmp_path / "report.pdf")
    text = build_markdown(ctx, None, charts, tmp_path / "report.md").read_text(encoding="utf-8")
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert "missing.png" not in text and "No explanation was computed." in text
    assert "Verification was not run." in text and "No what-if example was computed" in text


# --------------------------------------------------------------------------- orchestrator


def test_offline_run_produces_report_and_supports_what_if(tmp_path: Path) -> None:
    df = classification_df(300)
    csv = tmp_path / "customers.csv"
    df.to_csv(csv, index=False)
    requirement = make_requirement("tabular_classification", "target")
    requirement.actionable_features = ["x1"]
    llm = FakeLLM({RelevanceCheck: RelevanceCheck(relevant=True)})  # every other LLM call fails -> deterministic paths
    orchestrator = Orchestrator(llm, tmp_path / "runs", budget=ONE)
    run_id = orchestrator.start("Which customers will say yes?", csv, requirement=requirement)
    ctx = orchestrator.run_until_blocked()

    assert ctx.state == RunState.DONE and ctx.errors == [] and llm.count("UserRequirement") == 0  # no LLM parse
    assert Path(ctx.report_paths["pdf"]).read_bytes()[:4] == b"%PDF" and Path(ctx.report_paths["markdown"]).is_file()
    assert ctx.insight.generated_by == "template" and ctx.recommendations.actionable_features == ["x1"]
    assert ctx.explanation.global_explanation.features and ctx.message.endswith(ctx.report_paths["pdf"])

    row = df.iloc[0].to_dict()
    single = orchestrator.what_if(run_id, {"x1": float(row["x1"]) + 1.0}, base_row=row)
    segment = orchestrator.what_if(run_id, {"x1": {"pct": 10}}, segment=[{"column": "region", "op": "==", "value": "north"}])
    assert isinstance(single, WhatIfResult) and single.after > single.before
    assert isinstance(segment, SegmentImpact) and segment.count == int((df["region"] == "north").sum())
    with pytest.raises(ValueError):
        orchestrator.what_if(run_id, {"x1": 1.0})

    saved = json.loads((tmp_path / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert len(saved["what_ifs"]) == 2 and saved["report_paths"]["pdf"]
    resumed = Orchestrator(None, tmp_path / "runs").resume(run_id)
    assert isinstance(resumed.what_ifs[0], WhatIfResult) and isinstance(resumed.what_ifs[1], SegmentImpact)

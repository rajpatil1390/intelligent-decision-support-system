"""Insight Agent: write the business decision brief from computed facts only.

The LLM sees a JSON of numbers the system computed (metrics, drivers, risk tiers, what-if results,
recommendations, verification, knowledge) and must not introduce any other number. A grounding check
extracts every number from its text; anything not found in the facts (±1% or rounding) triggers one
regeneration with feedback, and a second failure falls back to a deterministic template brief.
"""

import json
import math
import re
from typing import Any

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.prompts import render_prompt
from idss.core.schemas import (
    TASK_DESCRIPTIONS,
    DecisionInsight,
    ExplanationResult,
    GroundingCheck,
    KnowledgeBrief,
    RecommendationResult,
    TrainingResult,
    UserRequirement,
    VerificationReport,
    WhatIfResult,
)

logger = get_logger()

MAX_ATTEMPTS = 2  # first draft + one regeneration
RELATIVE_TOLERANCE = 0.01
# Small whole numbers ("3 actions", "top 5 drivers") are counting words, not claims about the data.
EXEMPT_NUMBERS = frozenset({*range(11), 100})
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


class InsightDraft(BaseModel):
    executive_summary: str = Field(min_length=1)
    key_findings: list[str] = Field(min_length=3, max_length=5)
    recommended_actions: list[str] = Field(min_length=1)
    risks_and_limitations: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


class InsightAgent:
    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    def generate(self, facts: dict[str, Any]) -> DecisionInsight:
        """LLM brief that passes the grounding check, else the template brief. Never raises."""
        if self.llm_client is None:
            return template_insight(facts)
        allowed = fact_numbers(facts)
        feedback = ""
        unsupported: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                draft = self.llm_client.chat_structured(
                    render_prompt("decision_insight", facts_json=json.dumps(facts, indent=2, default=str), feedback=feedback),
                    InsightDraft,
                )
            except Exception as exc:  # any LLM problem -> template
                logger.warning("Insight generation failed ({}: {}); using the template brief.", type(exc).__name__, exc)
                # Only report a grounding result if an earlier draft was actually checked.
                checked = GroundingCheck(passed=False, attempts=attempt - 1, unsupported_numbers=unsupported)
                return template_insight(facts, checked if attempt > 1 else None)
            unsupported = unsupported_numbers(_draft_text(draft), allowed)
            logger.info("Insight grounding check (attempt {}): {}", attempt,
                        "passed" if not unsupported else f"unsupported numbers {unsupported}")
            if not unsupported:
                return DecisionInsight(**draft.model_dump(), generated_by="llm",
                                       grounding=GroundingCheck(passed=True, attempts=attempt))
            feedback = ("Your previous draft used numbers that are not in the facts: " + ", ".join(unsupported)
                        + ". Rewrite it using only numbers from the facts, or describe those effects in words.")
        logger.warning("Insight still ungrounded after {} attempts; using the template brief.", MAX_ATTEMPTS)
        return template_insight(facts, GroundingCheck(passed=False, attempts=MAX_ATTEMPTS, unsupported_numbers=unsupported))


# --------------------------------------------------------------------------- facts


def build_facts(
    requirement: UserRequirement,
    training: TrainingResult,
    explanation: ExplanationResult | None = None,
    recommendations: RecommendationResult | None = None,
    verification: VerificationReport | None = None,
    knowledge: KnowledgeBrief | None = None,
    what_if: WhatIfResult | None = None,
    data_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """The only information the Insight Agent may use: computed numbers and short descriptions."""
    test = {k: v for k, v in training.test_metrics.items() if k not in ("confusion_matrix", "class_labels")}
    facts: dict[str, Any] = {
        "goal": requirement.problem.description,
        "task": TASK_DESCRIPTIONS[training.task_type],
        "target": training.target,
        "model": {"name": training.best_model, "primary_metric": training.primary_metric,
                  "test_metrics": test, "decision_threshold": training.decision_threshold,
                  "test_metrics_at_threshold_0_5": {k: v for k, v in training.test_metrics_default_threshold.items()
                                                    if k not in ("confusion_matrix", "class_labels")}},
        "baseline": training.baseline.model_dump() if training.baseline else None,
        "constraints": requirement.problem.constraints,
        "data_warnings": data_warnings or [],
    }
    if explanation is not None:
        facts["key_drivers"] = [
            {"feature": f.feature, "share_of_importance": f.share, "direction": f.direction, "detail": f.detail}
            for f in explanation.global_explanation.features[:5]
        ]
        facts["explained_output"] = explanation.global_explanation.output
        if explanation.clusters:
            facts["clusters"] = [c.model_dump() for c in explanation.clusters]
        if explanation.forecast is not None:
            points = explanation.forecast.points
            facts["forecast"] = {"horizon": explanation.forecast.horizon, "first": points[0].model_dump(),
                                 "last": points[-1].model_dump(),
                                 "average": round(sum(p.prediction for p in points) / len(points), 4),
                                 "interval": explanation.forecast.interval}
    if recommendations is not None:
        facts["risk_tiers"] = [t.model_dump() for t in recommendations.risk_tiers]
        if recommendations.tier_cutoffs is not None:
            facts["risk_tier_cutoffs"] = recommendations.tier_cutoffs.model_dump()
        facts["recommendations"] = [r.model_dump() for r in recommendations.recommendations]
        facts["actionable_features"] = recommendations.actionable_features
        if recommendations.counterfactual_example is not None:
            facts["counterfactual_example"] = recommendations.counterfactual_example.model_dump()
    if what_if is not None:
        facts["what_if_example"] = what_if.model_dump()
    if verification is not None:
        facts["verification"] = {"passed": verification.passed, "failed_checks": verification.failed_checks,
                                 "checks": [c.model_dump() for c in verification.checks]}
    if knowledge is not None and knowledge.insights:
        facts["best_practices"] = knowledge.insights[:4]
    return facts


def fact_numbers(facts: Any) -> list[float]:
    """Every number in the facts, including numbers written inside strings."""
    numbers: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                numbers.append(float(value))
        elif isinstance(value, str):
            numbers.extend(_parse_number(token)[0] for token in _NUMBER.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(facts)
    return numbers


def unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    """Numbers in `text` that do not match any fact (±1%, rounding, or as a percentage of a fraction)."""
    problems = []
    for token in _NUMBER.findall(text):
        value, is_percent = _parse_number(token)
        if value in EXEMPT_NUMBERS or _supported(value, token, is_percent, allowed):
            continue
        problems.append(token)
    return list(dict.fromkeys(problems))


def _supported(value: float, token: str, is_percent: bool, allowed: list[float]) -> bool:
    """Magnitudes are compared, so "reduces risk by 0.18" matches a fact of -0.18."""
    decimals = len(token.rstrip("%").split(".")[1]) if "." in token else 0
    written = abs(value)
    readings = [written, written / 100] if is_percent or written <= 100 else [written]  # 62% may mean 0.62
    for fact in (abs(f) for f in allowed):
        if any(math.isclose(reading, fact, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-9) for reading in readings):
            return True
        # Rounded for display: 0.6230 -> "0.62"; 0.6230 -> "62%".
        if round(fact, decimals) == round(written, decimals) or round(fact * 100, decimals) == round(written, decimals):
            return True
    return False


def _parse_number(token: str) -> tuple[float, bool]:
    is_percent = token.endswith("%")
    return float(token.rstrip("%").replace(",", "")), is_percent


def _draft_text(draft: InsightDraft) -> str:
    return " ".join([draft.executive_summary, *draft.key_findings, *draft.recommended_actions,
                     *draft.risks_and_limitations, *draft.next_steps])


# --------------------------------------------------------------------------- template


def template_insight(facts: dict[str, Any], grounding: GroundingCheck | None = None) -> DecisionInsight:
    """A complete, readable brief built only from the facts (used without an LLM or when grounding fails)."""
    model = facts["model"]
    metric = model["primary_metric"]
    score = model["test_metrics"].get(metric)
    baseline = facts.get("baseline") or {}
    baseline_score = (baseline.get("metrics") or {}).get(metric)
    drivers = facts.get("key_drivers", [])
    tiers = {t["name"]: t for t in facts.get("risk_tiers", [])}
    recommendations = facts.get("recommendations", [])
    verification = facts.get("verification") or {}

    goal = facts["goal"].strip()
    summary = [f"Goal: {goal if goal.endswith(('.', '?', '!')) else goal + '.'} We treated this as {facts['task']}."]
    if score is not None:
        baseline_name = str(baseline.get("name", "baseline")).replace("_", " ")
        summary.append(f"The best model ({model['name']}) scored {metric} = {_fmt(score)} on held-out test data"
                       + (f", against {_fmt(baseline_score)} for a simple {baseline_name} baseline."
                          if baseline_score is not None else "."))
    if drivers:
        summary.append("The strongest drivers are " + _join([d["feature"] for d in drivers[:3]]) + ".")
    if "High" in tiers:
        summary.append(f"{tiers['High']['count']:,} records ({tiers['High']['share']:.1%}) are in the high-risk tier.")
    if verification and not verification.get("passed", True):
        summary.append("Note: the model did not pass every quality check, so treat the results with caution.")

    findings = []
    if score is not None:
        findings.append(f"Model quality: {metric} = {_fmt(score)} on test data with {model['name']}"
                        + (f" (decision threshold {_fmt(model['decision_threshold'])})." if model.get("decision_threshold") else "."))
    for driver in drivers[:3]:
        findings.append(f"{driver['feature']} accounts for {driver['share_of_importance']:.0%} of the model's explanation"
                        + (f"; {driver['detail']}" if driver.get("detail") else "."))
    if "High" in tiers:
        high = tiers["High"]
        cutoffs = facts.get("risk_tier_cutoffs")
        findings.append(f"Risk tiers: High {high['count']:,}, Medium {tiers['Medium']['count']:,}, Low {tiers['Low']['count']:,} records"
                        + (f" (High from probability {_fmt(cutoffs['high'])}, Medium from the decision threshold "
                           f"{_fmt(cutoffs['medium'])})." if cutoffs else "."))
    if "forecast" in facts:
        forecast = facts["forecast"]
        findings.append(f"The {forecast['horizon']}-period forecast averages {_fmt(forecast['average'])} "
                        f"(from {_fmt(forecast['first']['prediction'])} to {_fmt(forecast['last']['prediction'])}).")
    for cluster in facts.get("clusters", [])[:2]:
        findings.append(f"Segment {cluster['cluster']} has {cluster['size']:,} records ({cluster['share']:.0%}).")
    while len(findings) < 3:
        findings.append(f"The analysis covered {len(facts.get('actionable_features', []))} actionable column(s).")

    actions = [f"{r['action']} Expected impact: {_fmt(r['expected_impact'])} ({r['impact_unit']}); confidence {r['confidence']}."
               for r in recommendations] or ["Review the key drivers with the business team before acting."]

    risks = []
    if verification.get("failed_checks"):
        risks.append("Failed quality checks: " + ", ".join(verification["failed_checks"]) + ".")
    risks += [str(w) for w in facts.get("data_warnings", [])[:3]]
    risks.append("The model shows associations in historical data, not guaranteed cause and effect; "
                 "test changes on a small group before rolling them out.")
    if facts.get("what_if_example"):
        risks.append("What-if results assume all other inputs stay the same.")

    next_steps = ["Validate the top recommendation with a controlled pilot and a comparison group.",
                  "Share the key drivers with domain experts to confirm they make business sense.",
                  "Retrain the model regularly as new data arrives."]
    return DecisionInsight(executive_summary=" ".join(summary), key_findings=findings[:5], recommended_actions=actions,
                           risks_and_limitations=risks, next_steps=next_steps, generated_by="template", grounding=grounding)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"

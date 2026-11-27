"""Insight Agent: write the business decision brief from computed facts only.

Since v1.1 the LLM never types a number. It receives the facts registry's view (idss.decision.facts), in which
every number is a placeholder such as {{metrics.test.f1}} with its value listed, and must write its brief with
those placeholders. Code then checks the draft: a placeholder naming an unknown key, or a digit written outside a
placeholder (names such as F1 excepted), rejects it. A rejected draft is regenerated once with feedback, then the
deterministic template brief is used. Accepted placeholders are filled in by code, and the earlier grounding check
(every number in the final text must match a fact, ±1% or rounding) still runs as a second layer.
The template brief is written with the same placeholders, so every number in any brief has a registry key.
"""

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.prompts import render_prompt
from idss.core.sanitize import data_json
from idss.core.schemas import DecisionInsight, GroundingCheck
from idss.decision.facts import FactsRegistry, UnknownFactKey, build_facts, build_registry  # noqa: F401  (re-exported)

logger = get_logger()

MAX_ATTEMPTS = 2  # first draft + one regeneration
RELATIVE_TOLERANCE = 0.01
# Small whole numbers ("3 actions", "top 5 drivers") are counting words, not claims about the data.
EXEMPT_NUMBERS = frozenset({*range(11), 100})
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
FIELDS = ("executive_summary", "key_findings", "recommended_actions", "risks_and_limitations", "next_steps")


class InsightDraft(BaseModel):
    executive_summary: str = Field(min_length=1)
    key_findings: list[str] = Field(min_length=3, max_length=5)
    recommended_actions: list[str] = Field(min_length=1)
    risks_and_limitations: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


class DraftCheck(BaseModel):
    """Result of checking one draft against the registry."""

    unknown_keys: list[str] = Field(default_factory=list)
    raw_numbers: list[str] = Field(default_factory=list)
    ungrounded: list[str] = Field(default_factory=list)
    rendered: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.rendered is not None and not (self.unknown_keys or self.raw_numbers or self.ungrounded)


class InsightAgent:
    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    def generate(self, facts: FactsRegistry) -> DecisionInsight:
        """LLM brief that passes every check, else the template brief. Never raises."""
        if self.llm_client is None:
            return template_insight(facts)
        feedback = ""
        check = DraftCheck()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                draft = self.llm_client.chat_structured(
                    render_prompt("decision_insight", facts_json=data_json(facts.llm_view()), feedback=feedback),
                    InsightDraft,
                )
            except Exception as exc:  # any LLM problem -> template
                logger.warning("Insight generation failed ({}: {}); using the template brief.", type(exc).__name__, exc)
                # Only report a check result if an earlier draft was actually checked.
                return template_insight(facts, _grounding(check, attempt - 1) if attempt > 1 else None)
            check = check_draft(draft, facts)
            logger.info("Insight draft check (attempt {}): {}", attempt, "passed" if check.ok else check.model_dump(
                exclude={"rendered"}))
            if check.ok and check.rendered is not None:
                return DecisionInsight(**check.rendered, generated_by="llm", placeholder_text=draft.model_dump(),
                                       grounding=GroundingCheck(passed=True, attempts=attempt))
            feedback = _feedback(check)
        logger.warning("Insight draft still rejected after {} attempts; using the template brief.", MAX_ATTEMPTS)
        return template_insight(facts, _grounding(check, MAX_ATTEMPTS))


def check_draft(draft: InsightDraft, facts: FactsRegistry) -> DraftCheck:
    """Placeholders must name known facts and no number may be typed; then fill in the values and ground them."""
    text = _draft_text(draft)
    unknown, raw = facts.unknown_keys(text), facts.raw_numbers(text)
    if unknown or raw:
        return DraftCheck(unknown_keys=unknown, raw_numbers=raw)
    rendered = {field: _apply(getattr(draft, field), facts.render) for field in FIELDS}
    ungrounded = unsupported_numbers(_text_of(rendered), facts.numbers())  # second layer
    return DraftCheck(ungrounded=ungrounded, rendered=rendered)


def _feedback(check: DraftCheck) -> str:
    parts = ["Your previous draft broke the number rules."]
    if check.raw_numbers:
        parts.append("It typed numbers instead of placeholders: " + ", ".join(check.raw_numbers) + ".")
    if check.unknown_keys:
        parts.append("It used placeholder keys that do not exist: " + ", ".join(check.unknown_keys) + ".")
    if check.ungrounded:
        parts.append("Its numbers did not match the facts: " + ", ".join(check.ungrounded) + ".")
    parts.append("Rewrite it. Write every number only as a {{key}} placeholder copied from the facts, or describe "
                 "the effect in words.")
    return " ".join(parts)


def _grounding(check: DraftCheck, attempts: int) -> GroundingCheck:
    return GroundingCheck(passed=False, attempts=attempts, unsupported_numbers=[*check.raw_numbers, *check.ungrounded],
                          unknown_keys=check.unknown_keys)


# --------------------------------------------------------------------------- grounding (second layer)


def fact_numbers(facts: Any) -> list[float]:
    """Every number in a facts structure, including numbers written inside strings."""
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
    return _text_of(draft.model_dump())


def _text_of(fields: dict[str, Any]) -> str:
    return " ".join(item for field in FIELDS for item in _as_list(fields[field]))


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _apply(value: str | list[str], function: Any) -> str | list[str]:
    return function(value) if isinstance(value, str) else [function(item) for item in value]


# --------------------------------------------------------------------------- template


def template_insight(facts: FactsRegistry, grounding: GroundingCheck | None = None) -> DecisionInsight:
    """A complete, readable brief written with fact placeholders (used without an LLM or when its drafts fail)."""
    texts = template_text(facts)
    rendered = {field: _apply(value, facts.render) for field, value in texts.items()}
    return DecisionInsight(**rendered, generated_by="template", grounding=grounding, placeholder_text=texts)


def template_text(facts: FactsRegistry) -> dict[str, Any]:
    """The template brief with {{key}} placeholders for every number."""
    tree = facts.tree
    model = tree.get("model") or {}
    metric = model.get("primary_metric")
    score_key = f"metrics.test.{metric}"
    has_score = score_key in facts and facts.value(score_key) is not None
    baseline = tree.get("baseline") or {}
    baseline_key = f"baseline.{metric}"
    drivers = tree.get("drivers") or []
    tiers = tree.get("risk_tiers") or {}
    verification = tree.get("verification") or {}

    name = facts.templated("model.name") if "model" in tree else ""
    metric_name = facts.templated("model.primary_metric") if "model" in tree else ""
    feature_names = [facts.templated(f"drivers[{index}].feature") for index in range(len(drivers))]
    goal = facts.templated("requirement.goal").strip() if "requirement" in tree else "Analyse the dataset."
    summary = [f"Goal: {goal if goal.endswith(('.', '?', '!', '}}')) else goal + '.'} We treated this as {model.get('task', 'an analysis')}."]
    if has_score:
        baseline_name = str(baseline.get("name", "baseline")).replace("_", " ")
        summary.append(f"The best model ({name}) scored {metric_name} = {{{{{score_key}}}}} on held-out test data"
                       + (f", against {{{{{baseline_key}}}}} for a simple {baseline_name} baseline."
                          if baseline_key in facts and facts.value(baseline_key) is not None else "."))
    if drivers:
        summary.append("The strongest drivers are " + _join(feature_names[:3]) + ".")
    if "high" in tiers:
        summary.append("{{risk_tiers.high.count}} records ({{risk_tiers.high.share}}) are in the high-risk tier.")
    if verification and not verification.get("passed", True):
        summary.append("Note: the model did not pass every quality check, so treat the results with caution.")

    findings = []
    if has_score:
        findings.append(f"Model quality: {metric_name} = {{{{{score_key}}}}} on test data with {name}"
                        + (" (decision threshold {{threshold.tuned}})." if "threshold.tuned" in facts else "."))
    for index, driver in enumerate(drivers[:3]):
        detail = facts.templated(f"drivers[{index}].detail") if driver.get("detail") else ""
        findings.append(f"{feature_names[index]} accounts for {{{{drivers[{index}].share}}}} of the model's explanation"
                        + (f"; {detail}" if detail else "."))
    if "high" in tiers:
        cutoffs = (" (High from probability {{risk_tiers.cutoffs.high}}, Medium from the decision threshold "
                   "{{risk_tiers.cutoffs.medium}})." if "cutoffs" in tiers else ".")
        findings.append("Risk tiers: High {{risk_tiers.high.count}}, Medium {{risk_tiers.medium.count}}, "
                        "Low {{risk_tiers.low.count}} records" + cutoffs)
    forecast = tree.get("forecast") or {}
    if forecast.get("points"):
        last = len(forecast["points"]) - 1
        findings.append("The {{forecast.horizon}}-period forecast averages {{forecast.average}} "
                        f"(from {{{{forecast.points[0].prediction}}}} to {{{{forecast.points[{last}].prediction}}}}).")
    for index, _ in enumerate((tree.get("clusters") or [])[:2]):
        findings.append(f"Segment {{{{clusters[{index}].id}}}} has {{{{clusters[{index}].size}}}} records "
                        f"({{{{clusters[{index}].share}}}}).")
    if len(findings) < 3 and "actionable.count" in facts:
        findings.append("The analysis covered {{actionable.count}} actionable column(s).")
    while len(findings) < 3:
        findings.append("The key drivers above come from the model's own explanation of its predictions.")

    actions = [f"{facts.templated(f'recommendations[{index}].action')} Expected impact: "
               f"{{{{recommendations[{index}].expected_impact}}}} ({facts.templated(f'recommendations[{index}].impact_unit')}); "
               f"confidence {item['confidence']}." for index, item in enumerate(tree.get("recommendations") or [])]
    actions = actions or ["Review the key drivers with the business team before acting."]

    risks = []
    if verification.get("failed_checks"):
        risks.append("Failed quality checks: " + ", ".join(verification["failed_checks"]) + ".")
    warnings = (tree.get("data") or {}).get("warnings") or []
    risks += [facts.templated(f"data.warnings[{index}]") for index in range(min(3, len(warnings)))]
    risks.append("The model shows associations in historical data, not guaranteed cause and effect; "
                 "test changes on a small group before rolling them out.")
    if tree.get("whatif"):
        risks.append("What-if results assume all other inputs stay the same.")

    next_steps = ["Validate the top recommendation with a controlled pilot and a comparison group.",
                  "Share the key drivers with domain experts to confirm they make business sense.",
                  "Retrain the model regularly as new data arrives."]
    return {"executive_summary": " ".join(summary), "key_findings": findings[:5], "recommended_actions": actions,
            "risks_and_limitations": risks, "next_steps": next_steps}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"

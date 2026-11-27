"""Number provenance (v1.1): every number the system reports has a key, a value and a source file.

`build_registry(ctx)` turns a run's computed results into a tree of facts (metrics.test.f1, baseline.f1,
threshold.tuned, risk_tiers.high.count, whatif[0].before, ...). Each branch records the file the numbers come from
(metrics.json, leaderboard.json, state.json sections). Numbers written inside text (a warning, a check detail) are
facts too, keyed `<path>.n0`, `<path>.n1`, ...

Text that mentions numbers is written with placeholders such as {{metrics.test.f1}} and filled in by code with
`render()`, so no number in the decision brief or the report is typed by an LLM. `show()` returns a number's
display text for the report and remembers it, so the report can list every number it printed with its key and
source ("Sources of numbers").
"""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from idss.core.plan_contract import MODEL_NAMES
from idss.core.schemas import (
    TASK_DESCRIPTIONS,
    DatasetProfile,
    ExplanationResult,
    KnowledgeBrief,
    ModelPlan,
    RecommendationResult,
    TrainingResult,
    UserRequirement,
    VerificationReport,
    WhatIfResult,
)
from idss.ml.evaluator import METRICS

if TYPE_CHECKING:
    from idss.agents.manager import RunContext

Format = Literal["num", "count", "g", "f2", "pct", "pct0", "signed", "signed_pct", "secs", "gsecs", "text"]

NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?%?")  # "1,988" is one number; "900," ends at 0
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\[\]-]+)\s*\}\}")
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w.\-]*$")  # names such as f1, lag_7 or q1_sales: allowed as written
_DIGIT_WORD = re.compile(r"\S*\d\S*")
_SEGMENT = re.compile(r"\[\d+\]$")
MISSING = "-"
DEFAULT_THRESHOLD = 0.5

SOURCE_METRICS = "metrics.json"
SOURCE_LEADERBOARD = "leaderboard.json"
_COUNT_FIELDS = frozenset({
    "count", "size", "n_rows", "n_cols", "duplicate_rows", "n_unique", "n_samples", "horizon", "rank", "n_test_rows",
    "n_passed", "n_checks", "max_candidates", "n_trials_per_model", "plan_revisions", "percentile", "id",
})
_FIELD_FORMATS: dict[str, Format] = {
    "share": "pct", "delta": "signed", "impact": "signed", "expected_impact": "signed", "delta_pct": "signed_pct",
    "missing_pct": "g", "total_time_s": "secs", "total_timeout_s": "gsecs", "memory_mb": "f2",
}
MAX_REPORT_DRIVERS = 10


class UnknownFactKey(KeyError):
    """A placeholder or report lookup names a key that is not in the registry."""


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: float | int | str | None
    display: str
    source: str
    kind: Format


def format_value(value: Any, kind: Format) -> str:
    """Display text of a number in the given format (strings are returned as they are)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return MISSING
    if isinstance(value, str) or kind == "text":
        return str(value)
    number = float(value)
    match kind:
        case "count":
            return f"{int(round(number)):,}"
        case "g":
            return f"{number:g}"
        case "f2":
            return f"{number:.2f}"
        case "pct":
            return f"{number:.1%}"
        case "pct0":
            return f"{number:.0%}"
        case "signed":
            return f"{number:+.4g}"
        case "signed_pct":
            return f"{number:+.2f}%"
        case "secs":
            return f"{number:.1f}s"
        case "gsecs":
            return f"{number:g}s"
    if isinstance(value, int):
        return f"{value:,}"
    if abs(number) >= 10_000:
        return f"{number:,.0f}"
    if abs(number) >= 1_000:
        return f"{number:,.1f}"
    return f"{number:.4g}"


class FactsRegistry:
    """Every computed number (and every number inside computed text), by key, with its source."""

    def __init__(self) -> None:
        self.tree: dict[str, Any] = {}  # the plain values, for code that needs structure (lengths, names)
        self.view: dict[str, Any] = {}  # the same tree with numbers replaced by {{key}} placeholders
        self._facts: dict[str, Fact] = {}
        self._strings: dict[str, str] = {}  # path -> original text of every string leaf
        self._templates: dict[str, str] = {}  # path -> text with its numbers replaced by placeholders
        self._used: dict[tuple[str, str], None] = {}
        self.names: set[str] = {*METRICS, *MODEL_NAMES}  # names that may contain digits and be written as they are

    # ------------------------------------------------------------------ building

    def load(self, path: str, value: Any, source: str) -> Any:
        """Register `value` (any JSON-like structure) under `path`; returns and stores its placeholder view."""
        view = self._load(path, value, source)
        self.tree[path], self.view[path] = value, view
        return view

    def _load(self, path: str, value: Any, source: str) -> Any:
        if type(value).__module__ == "numpy" and hasattr(value, "item"):
            value = value.item()  # numpy scalar -> Python number
        if isinstance(value, dict):
            return {key: self._load(f"{path}.{key}", item, source) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._load(f"{path}[{index}]", item, source) for index, item in enumerate(value)]
        if isinstance(value, bool):
            return value
        if value is None or isinstance(value, int | float):
            number = None if isinstance(value, float) and not math.isfinite(value) else value
            kind = _kind(path, number)
            self._facts[path] = Fact(key=path, value=number, display=format_value(number, kind), source=source, kind=kind)
            return None if number is None else f"{{{{{path}}}}}"
        return self._load_text(path, str(value), source)

    def _load_text(self, path: str, text: str, source: str) -> str:
        text = text.replace("{{", "{ {").replace("}}", "} }")  # text from the data can never form a placeholder
        self._strings[path] = text
        if not any(char.isdigit() for char in text):
            self._templates[path] = text
            return text
        self._facts[path] = Fact(key=path, value=text, display=text, source=source, kind="text")
        tokens = iter(range(10_000))

        def placeholder(match: re.Match[str]) -> str:
            key = f"{path}.n{next(tokens)}"
            token = match.group(0)
            number = float(token.rstrip("%").replace(",", ""))
            self._facts[key] = Fact(key=key, value=number, display=token, source=source, kind="text")
            return f"{{{{{key}}}}}"

        self._templates[path] = NUMBER_TOKEN.sub(placeholder, text)
        if _IDENTIFIER.match(text):
            self.names.add(text)
            return text
        return f"{{{{{path}}}}}" if not any(char.isspace() for char in text) else self._templates[path]

    # ------------------------------------------------------------------ lookups

    def __contains__(self, key: str) -> bool:
        return key in self._facts

    def fact(self, key: str) -> Fact:
        try:
            return self._facts[key]
        except KeyError:
            raise UnknownFactKey(key) from None

    def value(self, key: str) -> Any:
        return self.fact(key).value

    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    def numbers(self) -> list[float]:
        """Every numeric value, including numbers inside registered text (for the grounding check)."""
        return [float(f.value) for f in self._facts.values()
                if isinstance(f.value, int | float) and not isinstance(f.value, bool)]

    def show(self, key: str, fmt: Format | None = None) -> str:
        """Display text of `key` (in `fmt` if given), remembered for the "Sources of numbers" list."""
        fact = self.fact(key)
        if fact.value is None:
            return MISSING
        display = fact.display if fmt is None or isinstance(fact.value, str) else format_value(fact.value, fmt)
        self._used.setdefault((display, key), None)
        return display

    def show_or(self, key: str, fmt: Format | None = None) -> str:
        """Like show(), but MISSING for an unknown key."""
        return self.show(key, fmt) if key in self else MISSING

    def text(self, path: str) -> str:
        """The original text at `path`; the numbers in it are remembered as used."""
        if path not in self._strings:
            raise UnknownFactKey(path)
        self.render(self._templates[path])
        return self._strings[path]

    def display(self, path: str, fmt: Format | None = None) -> str:
        """Whatever `path` holds, ready to print and remembered: a number (in `fmt`), a name, or prose whose
        numbers are remembered one by one. MISSING for an unknown path."""
        if path in self._strings:
            text = self._strings[path]
            whole = path in self._facts and not any(char.isspace() for char in text)
            return self.show(path) if whole else self.text(path)
        return self.show(path, fmt) if path in self._facts else MISSING

    def note(self, path: str, text: str, source: str) -> str:
        """Register a piece of text that is not part of the tree (e.g. stored brief text), then use it."""
        self._load_text(path, text, source)
        return self.text(path)

    def templated(self, path: str) -> str:
        """The text at `path` with each of its numbers replaced by a placeholder."""
        if path not in self._templates:
            raise UnknownFactKey(path)
        return self._templates[path]

    def used(self) -> list[Fact]:
        """Every number shown so far, as (display, key, source) facts, in first-use order."""
        return [self._facts[key].model_copy(update={"display": display}) for display, key in self._used]

    # ------------------------------------------------------------------ placeholders

    @staticmethod
    def placeholders(text: str) -> list[str]:
        return PLACEHOLDER.findall(text)

    def unknown_keys(self, text: str) -> list[str]:
        return [key for key in dict.fromkeys(self.placeholders(text)) if key not in self]

    def raw_numbers(self, text: str) -> list[str]:
        """Words with digits outside placeholders and known names: numbers typed instead of looked up."""
        stripped = PLACEHOLDER.sub(" ", text)
        for name in sorted((n for n in self.names if any(c.isdigit() for c in n)), key=len, reverse=True):
            stripped = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", " ", stripped, flags=re.IGNORECASE)
        return list(dict.fromkeys(_DIGIT_WORD.findall(stripped)))

    def render(self, text: str) -> str:
        """Replace every placeholder with its value's display text. Raises UnknownFactKey for an unknown key."""
        return PLACEHOLDER.sub(lambda match: self.show(match.group(1)), text)

    def llm_view(self) -> dict[str, Any]:
        """The facts the Insight Agent may use, with numbers as placeholders and their values listed."""
        view = self.view
        selected: dict[str, Any] = {key: view[key] for key in (
            "requirement", "model", "metrics", "baseline", "threshold", "drivers", "clusters", "risk_tiers",
            "recommendations", "actionable", "counterfactual", "whatif", "verification") if key in view}
        if "data" in view:
            selected["data_warnings"] = view["data"].get("warnings", [])
        if "explanation" in view:
            selected["explained_output"] = view["explanation"].get("output")
        if "forecast" in view:
            forecast = view["forecast"]
            points = forecast.get("points") or []
            selected["forecast"] = {key: forecast[key] for key in ("horizon", "average", "interval") if key in forecast}
            if points:
                selected["forecast"] |= {"first_point": points[0], "last_point": points[-1]}
        if "knowledge" in view:
            selected["best_practices"] = view["knowledge"].get("insights", [])
        keys = dict.fromkeys(self.placeholders(str(selected)))
        selected["values"] = {key: self.fact(key).display for key in keys if self.fact(key).value is not None}
        return selected


# --------------------------------------------------------------------------- building from a run


def build_registry(ctx: "RunContext") -> FactsRegistry:
    """Every number of a run, for the decision brief and the report."""
    return build_facts(ctx.requirement, ctx.training, ctx.explanation, ctx.recommendations, ctx.verification,
                       ctx.knowledge, ctx.what_if_example, profile=ctx.profile, plan=ctx.plan, run=ctx)


def build_facts(
    requirement: UserRequirement | None,
    training: TrainingResult | None,
    explanation: ExplanationResult | None = None,
    recommendations: RecommendationResult | None = None,
    verification: VerificationReport | None = None,
    knowledge: KnowledgeBrief | None = None,
    what_if: WhatIfResult | None = None,
    data_warnings: list[str] | None = None,
    *,
    profile: DatasetProfile | None = None,
    plan: ModelPlan | None = None,
    run: "RunContext | None" = None,
) -> FactsRegistry:
    """The facts registry from a run's parts (any part may be missing)."""
    facts = FactsRegistry()
    if run is not None:
        facts.load("run", {"id": run.run_id, "plan_revisions": run.plan_revision_count}, "state.json")
        facts.load("question", {"text": run.question.split("\n")[0]}, "user question")
        facts.load("dataset", {"file": Path(run.data_path).name}, "uploaded file name")
        facts.load("report", {"date": f"{datetime.now():%Y-%m-%d %H:%M}"}, "report build time")
    if requirement is not None:
        facts.load("requirement", {"goal": requirement.problem.description,
                                   "constraints": list(requirement.problem.constraints)}, "state.json (requirement)")
    if profile is not None:
        facts.load("data", {
            "n_rows": profile.n_rows, "n_cols": profile.n_cols, "duplicate_rows": profile.duplicate_rows,
            "memory_mb": profile.memory_mb, "warnings": list(profile.warnings),
            "columns": [{"name": c.name, "type": c.semantic_type, "missing_pct": c.missing_pct, "n_unique": c.n_unique}
                        for c in profile.columns],
        }, "state.json (profile)")
    elif data_warnings:
        facts.load("data", {"warnings": list(data_warnings)}, "state.json (profile)")
    if training is not None:
        _load_training(facts, training)
    if plan is not None:
        facts.load("plan", {"source": plan.source, "budget_mode": plan.budget_mode, "primary_metric": plan.primary_metric,
                            "candidates": ", ".join(c.name for c in plan.candidates[: plan.budget.max_candidates]),
                            "max_candidates": plan.budget.max_candidates,
                            "n_trials_per_model": plan.budget.n_trials_per_model,
                            "total_timeout_s": plan.budget.total_timeout_s}, "state.json (validated_plan)")
    if verification is not None:
        facts.load("verification", {
            "passed": verification.passed, "n_passed": sum(c.passed for c in verification.checks),
            "n_checks": len(verification.checks), "failed_checks": verification.failed_checks,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in verification.checks],
        }, "state.json (verification)")
    if explanation is not None:
        _load_explanation(facts, explanation)
    if recommendations is not None:
        _load_recommendations(facts, recommendations)
    if what_if is not None:
        facts.load("whatif", [{
            "output": what_if.output, "before": what_if.before, "after": what_if.after, "delta": what_if.delta,
            "delta_pct": what_if.delta_pct, "changes": [{"feature": k, "value": v} for k, v in what_if.changes.items()],
            "warnings": list(what_if.warnings),
        }], "state.json (what_if_example)")
    if knowledge is not None:
        facts.load("knowledge", {"insights": knowledge.insights[:4], "sources": list(knowledge.sources)},
                   "state.json (knowledge brief)")
    return facts


def _load_training(facts: FactsRegistry, training: TrainingResult) -> None:
    facts.load("model", {"name": training.best_model, "task": TASK_DESCRIPTIONS[training.task_type],
                         "target": training.target, "primary_metric": training.primary_metric,
                         "direction": training.direction}, SOURCE_METRICS)
    facts.load("metrics", {"test": _scalars(training.test_metrics),
                           "test_at_default_threshold": _scalars(training.test_metrics_default_threshold)}, SOURCE_METRICS)
    if training.baseline is not None:
        facts.load("baseline", {"name": training.baseline.name, **_scalars(training.baseline.metrics)}, SOURCE_METRICS)
    if training.decision_threshold is not None:
        facts.load("threshold", {"tuned": training.decision_threshold}, SOURCE_METRICS)
        facts.load("threshold_default", {"value": DEFAULT_THRESHOLD}, "fixed default cut-off")
    facts.load("leaderboard", [
        {"rank": e.rank, "model": e.model, "valid_score": e.valid_score, "inference_ms_per_row": e.inference_ms_per_row,
         "violations": "; ".join(e.violations)} for e in training.leaderboard if e.rank is not None
    ], SOURCE_LEADERBOARD)
    facts.load("training", {"total_time_s": training.total_time_s}, "state.json (training)")


def _load_explanation(facts: FactsRegistry, explanation: ExplanationResult) -> None:
    explained = explanation.global_explanation
    section: dict[str, Any] = {"method": explained.method, "output": explained.output,
                               "n_samples": explained.n_samples, "notes": list(explained.notes)}
    if explanation.local_examples:
        local = explanation.local_examples[0]
        section["example"] = {"prediction": local.prediction, "contributions": [
            {"feature": c.feature, "value": c.value, "impact": c.impact} for c in local.contributions]}
    source = "state.json (explanation)"
    facts.load("explanation", section, source)
    facts.load("drivers", [{"feature": f.feature, "share": f.share, "direction": f.direction, "detail": f.detail}
                           for f in explained.features[:MAX_REPORT_DRIVERS]], source)
    if explanation.clusters:
        facts.load("clusters", [{"id": c.cluster, "size": c.size, "share": c.share,
                                 "distinguishing": list(c.distinguishing)} for c in explanation.clusters], source)
    if explanation.forecast is not None:
        forecast = explanation.forecast
        points = forecast.points
        facts.load("forecast", {
            "horizon": forecast.horizon, "interval": forecast.interval, "assumptions": list(forecast.assumptions),
            "average": round(sum(p.prediction for p in points) / len(points), 4) if points else None,
            "points": [{"timestamp": p.timestamp, "prediction": p.prediction, "lower": p.lower, "upper": p.upper}
                       for p in points],
        }, source)


def _load_recommendations(facts: FactsRegistry, recommendations: RecommendationResult) -> None:
    source = "state.json (recommendations)"
    facts.load("recommendations", [{"action": r.action, "expected_impact": r.expected_impact,
                                    "impact_unit": r.impact_unit, "confidence": r.confidence}
                                   for r in recommendations.recommendations], source)
    facts.load("recommendation_notes", list(recommendations.notes), source)
    facts.load("actionable", {"features": list(recommendations.actionable_features),
                              "count": len(recommendations.actionable_features)}, source)
    if recommendations.risk_tiers:
        tiers: dict[str, Any] = {tier.name.lower(): {
            "min_probability": tier.min_probability, "max_probability": tier.max_probability, "count": tier.count,
            "share": tier.share, "avg_probability": tier.avg_probability, "top_drivers": list(tier.top_drivers),
        } for tier in recommendations.risk_tiers}
        if recommendations.tier_cutoffs is not None:
            cutoffs = recommendations.tier_cutoffs
            tiers["cutoffs"] = {"high": cutoffs.high, "medium": cutoffs.medium, "percentile": cutoffs.percentile,
                                "percentile_value": cutoffs.percentile_value}
        facts.load("risk_tiers", tiers, source)
    example = recommendations.counterfactual_example
    if example is not None:
        facts.load("counterfactual", {
            "probability_before": example.probability_before, "probability_after": example.probability_after,
            "threshold": example.threshold, "success": example.success,
            "changes": [{"feature": c.feature, "before": c.before, "after": c.after} for c in example.changes],
        }, source)


def _scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    """Numbers and labels only (confusion matrices and label lists are not reported as facts)."""
    return {name: value for name, value in metrics.items()
            if isinstance(value, int | float | str) and not isinstance(value, bool)}


def _kind(path: str, value: Any) -> Format:
    field = _SEGMENT.sub("", path.rsplit(".", 1)[-1])
    if field in _FIELD_FORMATS:
        return _FIELD_FORMATS[field]
    if field in _COUNT_FIELDS or isinstance(value, int):
        return "count"
    return "num"

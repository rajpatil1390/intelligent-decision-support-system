"""Decision report as PDF (reportlab) and Markdown, from the same list of content blocks.

The report is assembled from the run's computed results only. Any missing part (a chart, verification,
recommendations, ...) is skipped or replaced by a short note, never an error. Since v1.1 every number is printed
through the run's facts registry (idss.decision.facts), and the appendix "Sources of numbers" lists each printed
number with its key and the file it comes from. Dataset warnings about possible prompt-injection text get their
own "Security checks" section.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from idss.core.logger import get_logger
from idss.core.sanitize import INJECTION_WARNING, clean
from idss.core.schemas import TASK_DESCRIPTIONS, DecisionInsight
from idss.decision.facts import MISSING, FactsRegistry, UnknownFactKey, build_registry

if TYPE_CHECKING:
    from idss.agents.manager import RunContext

logger = get_logger()

PROJECT_NAME = "Intelligent Decision Support System Using Generative AI and Machine Learning"
MAX_PROFILE_ROWS = 25
MAX_LEADERBOARD_ROWS = 5
MAX_DRIVER_ROWS = 10
MAX_FORECAST_ROWS = 10
IMAGE_WIDTH_CM = 15.5
SOURCES_HEADING = "Sources of numbers"
BRIEF_FIELDS = ("executive_summary", "key_findings", "recommended_actions", "risks_and_limitations", "next_steps")
_PDF_REPLACEMENTS = {"→": "->", "≥": ">=", "≤": "<=", "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
                     "…": "...", "•": "-", "≈": "~", "−": "-"}


@dataclass
class Block:
    kind: Literal["title", "heading", "subheading", "paragraph", "bullets", "table", "image", "pagebreak"]
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)  # first row = header
    path: str = ""


def report_blocks(ctx: "RunContext", insight: DecisionInsight | None, charts: dict[str, str]) -> list[Block]:
    """All report content, in order. Shared by the PDF and Markdown renderers."""
    facts = build_registry(ctx)
    d = facts.display
    training, profile, requirement = ctx.training, ctx.profile, ctx.requirement
    brief = _brief(facts, insight)
    blocks: list[Block] = [Block("title", PROJECT_NAME), Block("paragraph", "Decision report")]
    task = training.task_type if training else (requirement.problem.task_type if requirement else None)
    if training is not None and training.target is not None:
        target = d("model.target")
    else:
        target = facts.note("report.target", str(requirement.target if requirement else MISSING), "state.json (requirement)")
    blocks.append(Block("table", rows=[
        ["Item", "Value"],
        ["Question", d("question.text")],
        ["Date", facts.show("report.date")],
        ["Dataset", d("dataset.file")],
        ["Run id", d("run.id")],
        ["Task", TASK_DESCRIPTIONS.get(task, MISSING) if task else MISSING],
        ["Target", target],
        ["Best model", d("model.name") if training else MISSING],
    ]))
    blocks.append(Block("pagebreak"))

    blocks.append(Block("heading", "1. Executive summary"))
    if brief is not None:
        blocks += [Block("paragraph", brief["executive_summary"]), Block("subheading", "Key findings"),
                   Block("bullets", items=brief["key_findings"])]
    else:
        blocks.append(Block("paragraph", facts.note("run.message", ctx.message or "No summary is available for this run.",
                                                    "state.json (message)")))

    blocks.append(Block("heading", "2. Data overview"))
    if profile is not None:
        blocks += _data_blocks(facts, profile)
    else:
        blocks.append(Block("paragraph", "No data profile is available."))

    blocks.append(Block("heading", "3. Model results"))
    if training is not None:
        blocks += _model_blocks(facts, training, charts)
    else:
        blocks.append(Block("paragraph", "No model was trained in this run."))

    blocks.append(Block("heading", "4. Verification"))
    verification = ctx.verification
    if verification is not None:
        blocks.append(Block("paragraph", f"Overall: {'PASSED' if verification.passed else 'NOT PASSED'} "
                                         f"({d('verification.n_passed')}/{d('verification.n_checks')} checks)."))
        blocks.append(Block("table", rows=[["Check", "Result", "Detail"]] + [
            [c.name, "pass" if c.passed else "FAIL", d(f"verification.checks[{i}].detail")]
            for i, c in enumerate(verification.checks)]))
        if verification.llm_verdict:
            if facts.raw_numbers(verification.llm_verdict):  # an LLM's number cannot be traced to a computed result
                blocks.append(Block("paragraph", "The LLM reviewer's verdict is not shown because it contains numbers "
                                                 "that cannot be traced to a computed result."))
            else:
                blocks.append(Block("paragraph", f"Reviewer verdict: {clean(verification.llm_verdict, 1_000)}"))
    else:
        blocks.append(Block("paragraph", "Verification was not run."))

    blocks.append(Block("heading", "5. Key drivers"))
    blocks += _driver_blocks(facts, ctx, charts)

    blocks.append(Block("heading", "6. Recommendations"))
    recommendations = ctx.recommendations
    if recommendations is not None and recommendations.recommendations:
        blocks.append(Block("table", rows=[["Action", "Expected impact", "Confidence"]] + [
            [d(f"recommendations[{i}].action"),
             f"{d(f'recommendations[{i}].expected_impact')} ({d(f'recommendations[{i}].impact_unit')})", r.confidence]
            for i, r in enumerate(recommendations.recommendations)]))
    if brief is not None:
        blocks += [Block("subheading", "In business terms"), Block("bullets", items=brief["recommended_actions"])]
    if recommendations is not None and recommendations.notes:
        blocks.append(Block("bullets", items=[d(f"recommendation_notes[{i}]") for i in range(len(recommendations.notes))]))
    if (recommendations is None or not recommendations.recommendations) and brief is None:
        blocks.append(Block("paragraph", "No recommendations are available."))

    blocks.append(Block("heading", "7. What-if example"))
    example = ctx.what_if_example
    if example is not None:
        changes = ", ".join(f"{d(f'whatif[0].changes[{i}].feature')} -> {d(f'whatif[0].changes[{i}].value')}"
                            for i in range(len(example.changes)))
        delta = MISSING if example.delta is None else d("whatif[0].delta") + (
            "" if example.delta_pct is None else f" ({d('whatif[0].delta_pct')})")
        rows = [["Item", "Value"], ["Change", changes], ["Output", d("whatif[0].output")],
                ["Before", d("whatif[0].before")], ["After", d("whatif[0].after")], ["Change in output", delta]]
        blocks.append(Block("table", rows=rows))
        if example.warnings:
            blocks.append(Block("bullets", items=[d(f"whatif[0].warnings[{i}]") for i in range(len(example.warnings))]))
    else:
        blocks.append(Block("paragraph", "No what-if example was computed for this task."))

    blocks.append(Block("heading", "8. Risks and limitations"))
    blocks.append(Block("bullets", items=brief["risks_and_limitations"] if brief else
                        ["Results show associations in historical data, not guaranteed cause and effect."]))
    if brief is not None:
        blocks += [Block("subheading", "Next steps"), Block("bullets", items=brief["next_steps"])]

    blocks.append(Block("heading", "Appendix"))
    blocks += _appendix_blocks(facts, ctx, insight)
    blocks += _sources_blocks(facts)  # last: it lists every number printed above
    return blocks


def _brief(facts: FactsRegistry, insight: DecisionInsight | None) -> dict[str, Any] | None:
    """The decision brief with its numbers filled in from this registry (so each one is listed with its key).
    A brief stored without placeholders (or with keys this registry lacks) is shown as stored, its numbers
    registered as coming from the brief."""
    if insight is None:
        return None
    templates = insight.placeholder_text or {}
    brief: dict[str, Any] = {}
    for name in BRIEF_FIELDS:
        stored = getattr(insight, name)
        try:
            if name not in templates:
                raise UnknownFactKey(name)
            brief[name] = _each(templates[name], lambda text, _: facts.render(text))
        except UnknownFactKey:
            brief[name] = _each(stored, lambda text, i, name=name: facts.note(f"brief.{name}[{i}]", text,
                                                                               "decision brief (state.json)"))
    return brief


def _each(value: str | list[str], function: Callable[[str, int], str]) -> str | list[str]:
    return function(value, 0) if isinstance(value, str) else [function(item, i) for i, item in enumerate(value)]


def _data_blocks(facts: FactsRegistry, profile: Any) -> list[Block]:
    d = facts.display
    blocks = [Block("paragraph", f"{d('data.n_rows')} rows and {d('data.n_cols')} columns; "
                                 f"{d('data.duplicate_rows')} duplicate rows; {d('data.memory_mb')} MB in memory.")]
    rows = [["Column", "Type", "Missing %", "Distinct"]] + [
        [d(f"data.columns[{i}].name"), c.semantic_type, d(f"data.columns[{i}].missing_pct"), d(f"data.columns[{i}].n_unique")]
        for i, c in enumerate(profile.columns[:MAX_PROFILE_ROWS])]
    blocks.append(Block("table", rows=rows))
    warnings = [d(f"data.warnings[{i}]") for i in range(len(profile.warnings))]
    blocks += [Block("subheading", "Data-quality warnings"), Block("bullets", items=warnings or ["No data-quality warnings."])]
    flagged = [warning for warning in warnings if warning.startswith(INJECTION_WARNING)]
    blocks += [Block("subheading", "Security checks"), Block("bullets", items=[
        *(flagged or ["No prompt-injection text was found in the column names or values."]),
        "Column names, values and other dataset text were sent to the LLM only as untrusted data inside "
        "<data> blocks, and no raw rows were sent.",
    ])]
    return blocks


def _model_blocks(facts: FactsRegistry, training: Any, charts: dict[str, str]) -> list[Block]:
    d = facts.display
    metric = training.primary_metric
    text = (f"Best model: {d('model.name')}. Primary metric: {d('model.primary_metric')} "
            f"({'higher' if training.direction == 'maximize' else 'lower'} is better) = {d(f'metrics.test.{metric}')} "
            f"on the held-out test split ({d('metrics.test.n_test_rows')} rows, evaluated once).")
    if f"baseline.{metric}" in facts:
        text += f" A simple {d('baseline.name')} baseline scores {d(f'baseline.{metric}')}."
    blocks = [Block("paragraph", text)]
    if "threshold.tuned" in facts:
        tuned, default = d("threshold.tuned", "f2"), d("threshold_default.value", "f2")
        blocks.append(Block("paragraph", f"Decision threshold tuned on the validation split: {tuned} "
                                         f"(instead of the default {default})."))
        at_default = training.test_metrics_default_threshold
        shared = [m for m in ("f1", "precision", "recall", "accuracy") if m in training.test_metrics and m in at_default]
        blocks.append(Block("table", rows=[["Metric", f"At {default}", f"At {tuned}"]] + [
            [m, d(f"metrics.test_at_default_threshold.{m}"), d(f"metrics.test.{m}")] for m in shared]))
    metrics = [[name, d(f"metrics.test.{name}")] for name, value in training.test_metrics.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool) and name != "n_test_rows"]
    if metrics:
        blocks += [Block("subheading", "Test metrics"), Block("table", rows=[["Metric", "Value"]] + metrics)]
    ranked = [e for e in training.leaderboard if e.rank is not None][:MAX_LEADERBOARD_ROWS]
    if ranked:
        blocks += [Block("subheading", f"Leaderboard (validation {metric})"), Block("table", rows=[
            ["Rank", "Model", "Valid score", "Inference ms/row", "Constraint issues"]] + [
            [d(f"leaderboard[{i}].rank"), d(f"leaderboard[{i}].model") + (" (selected)" if e.selected else ""),
             d(f"leaderboard[{i}].valid_score"), d(f"leaderboard[{i}].inference_ms_per_row"),
             d(f"leaderboard[{i}].violations") if e.violations else MISSING] for i, e in enumerate(ranked)])]
    for name in ("confusion_matrix", "actual_vs_pred", "forecast"):
        if _chart(charts, name):
            blocks.append(Block("image", path=charts[name]))
    return blocks


def _driver_blocks(facts: FactsRegistry, ctx: "RunContext", charts: dict[str, str]) -> list[Block]:
    d = facts.display
    explanation = ctx.explanation
    if explanation is None:
        return [Block("paragraph", "No explanation was computed.")]
    explained = explanation.global_explanation
    blocks = [Block("paragraph", f"Method: {explained.method.replace('_', ' ')}; explains the "
                                 f"{d('explanation.output')}; {d('explanation.n_samples')} rows analysed.")]
    if _chart(charts, "feature_importance"):
        blocks.append(Block("image", path=charts["feature_importance"]))
    if explained.features:
        blocks.append(Block("table", rows=[["Feature", "Share", "Direction", "Detail"]] + [
            [d(f"drivers[{i}].feature"), d(f"drivers[{i}].share"), f.direction,
             d(f"drivers[{i}].detail") if f.detail else MISSING] for i, f in enumerate(explained.features[:MAX_DRIVER_ROWS])]))
    if _chart(charts, "shap_summary"):
        blocks.append(Block("image", path=charts["shap_summary"]))
    if explained.notes:
        blocks.append(Block("bullets", items=[d(f"explanation.notes[{i}]") for i in range(len(explained.notes))]))
    recommendations = ctx.recommendations
    if recommendations is not None and recommendations.risk_tiers:
        blocks.append(Block("subheading", "Risk tiers"))
        if recommendations.tier_cutoffs is not None:
            high, medium = d("risk_tiers.cutoffs.high", "f2"), d("risk_tiers.cutoffs.medium", "f2")
            blocks.append(Block("paragraph", (
                f"High: probability >= {high}, the larger of the decision threshold ({medium}) and the "
                f"{d('risk_tiers.cutoffs.percentile')}th percentile of predicted probabilities "
                f"({d('risk_tiers.cutoffs.percentile_value', 'f2')}). Medium: {medium} <= probability < {high}. "
                f"Low: probability < {medium}.")))
        rows = [["Tier", "Probability range", "Records", "Share", "Top drivers"]]
        for tier in recommendations.risk_tiers:
            key = f"risk_tiers.{tier.name.lower()}"
            drivers = ", ".join(d(f"{key}.top_drivers[{j}]") for j in range(len(tier.top_drivers))) or MISSING
            rows.append([tier.name, f"{d(f'{key}.min_probability', 'f2')} - {d(f'{key}.max_probability', 'f2')}",
                         d(f"{key}.count"), d(f"{key}.share"), drivers])
        blocks.append(Block("table", rows=rows))
    if explanation.clusters:
        if _chart(charts, "cluster_sizes"):
            blocks.append(Block("image", path=charts["cluster_sizes"]))
        blocks += [Block("subheading", "Segments"), Block("table", rows=[["Segment", "Records", "Share", "What sets it apart"]] + [
            [d(f"clusters[{i}].id"), d(f"clusters[{i}].size"), d(f"clusters[{i}].share"),
             "; ".join(d(f"clusters[{i}].distinguishing[{j}]") for j in range(len(c.distinguishing))) or MISSING]
            for i, c in enumerate(explanation.clusters)])]
    if explanation.forecast is not None:
        forecast = explanation.forecast
        blocks += [Block("subheading", f"Forecast (next {d('forecast.horizon')} periods)"), Block("table", rows=[
            ["Date", "Forecast", "Lower", "Upper"]] + [
            [d(f"forecast.points[{i}].timestamp"), d(f"forecast.points[{i}].prediction"), d(f"forecast.points[{i}].lower"),
             d(f"forecast.points[{i}].upper")] for i in range(min(MAX_FORECAST_ROWS, len(forecast.points)))]),
            Block("bullets", items=[d("forecast.interval"),
                                    *(d(f"forecast.assumptions[{j}]") for j in range(len(forecast.assumptions)))])]
    if explanation.local_examples and explanation.local_examples[0].contributions:
        local = explanation.local_examples[0]
        blocks += [Block("subheading", f"Example: why one record got {d('explanation.example.prediction')}"),
                   Block("table", rows=[["Feature", "Value", "Effect"]] + [
                       [d(f"explanation.example.contributions[{i}].feature"), d(f"explanation.example.contributions[{i}].value"),
                        d(f"explanation.example.contributions[{i}].impact")] for i in range(len(local.contributions))])]
    return blocks


def _appendix_blocks(facts: FactsRegistry, ctx: "RunContext", insight: DecisionInsight | None) -> list[Block]:
    d = facts.display
    rows = [["Setting", "Value"]]
    if ctx.plan is not None:
        rows += [["Plan source", d("plan.source")], ["Candidates", d("plan.candidates")],
                 ["Budget", f"{d('plan.budget_mode')}: {d('plan.max_candidates')} models x {d('plan.n_trials_per_model')} "
                            f"trials, {d('plan.total_timeout_s')} total"], ["Primary metric", d("plan.primary_metric")]]
    if ctx.requirement is not None:
        constraints = [d(f"requirement.constraints[{i}]") for i in range(len(ctx.requirement.problem.constraints))]
        rows.append(["Constraints", "; ".join(constraints) or "none"])
    if ctx.training is not None:
        rows.append(["Training time", d("training.total_time_s")])
    rows.append(["Re-plans after verification", d("run.plan_revisions")])
    if insight is not None:
        grounding = insight.grounding
        failed = grounding is not None and bool(grounding.unsupported_numbers or grounding.unknown_keys)
        rows.append(["Brief written by", "LLM (number checks passed)" if insight.generated_by == "llm"
                     else "template" + (" (LLM draft failed the number checks)" if failed else "")])
    blocks = [Block("table", rows=rows)]
    knowledge = ctx.knowledge
    if knowledge is not None and knowledge.sources:
        blocks += [Block("subheading", "Knowledge sources"),
                   Block("bullets", items=[d(f"knowledge.sources[{i}]") for i in range(len(knowledge.sources))])]
    return blocks


def _sources_blocks(facts: FactsRegistry) -> list[Block]:
    used = facts.used()
    blocks = [Block("subheading", SOURCES_HEADING), Block("paragraph", (
        "Every number in this report, with the key it was looked up by and the file it comes from. Numbers in the "
        "decision brief were written as placeholders and filled in by code; none was typed by the language model."))]
    if used:
        blocks.append(Block("table", rows=[["Number", "Key", "Source"]] + [[f.display, f.key, f.source] for f in used]))
    return blocks


# --------------------------------------------------------------------------- renderers


def build_markdown(ctx: "RunContext", insight: DecisionInsight | None, charts: dict[str, str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for block in report_blocks(ctx, insight, charts):
        if block.kind == "title":
            lines += [f"# {block.text}", ""]
        elif block.kind == "heading":
            lines += [f"## {block.text}", ""]
        elif block.kind == "subheading":
            lines += [f"### {block.text}", ""]
        elif block.kind == "paragraph":
            lines += [block.text, ""]
        elif block.kind == "bullets":
            lines += [f"- {item}" for item in block.items] + [""]
        elif block.kind == "table" and block.rows:
            header, *body = block.rows
            lines += ["| " + " | ".join(_md_cell(c) for c in header) + " |", "|" + "---|" * len(header)]
            lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in body] + [""]
        elif block.kind == "image":
            relative = Path(os.path.relpath(block.path, out_path.parent)).as_posix()
            lines += [f"![{Path(block.path).stem}]({relative})", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_pdf(ctx: "RunContext", insight: DecisionInsight | None, charts: dict[str, str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8, leading=10)
    header = ParagraphStyle("header", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    story: list[Any] = []
    for block in report_blocks(ctx, insight, charts):
        if block.kind == "title":
            story += [Spacer(1, 4 * cm), Paragraph(_pdf(block.text), styles["Title"])]
        elif block.kind == "heading":
            story += [Spacer(1, 0.3 * cm), Paragraph(_pdf(block.text), styles["Heading1"])]
        elif block.kind == "subheading":
            story.append(Paragraph(_pdf(block.text), styles["Heading3"]))
        elif block.kind == "paragraph":
            story.append(Paragraph(_pdf(block.text), styles["BodyText"]))
        elif block.kind == "bullets":
            story += [Paragraph(_pdf(item), styles["BodyText"], bulletText="-") for item in block.items]
        elif block.kind == "table" and block.rows:
            story += [_pdf_table(block.rows, cell, header), Spacer(1, 0.25 * cm)]
        elif block.kind == "image":
            image = _pdf_image(block.path)
            if image is not None:
                story += [image, Spacer(1, 0.25 * cm)]
        elif block.kind == "pagebreak":
            story.append(PageBreak())

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, f"IDSS decision report - run {ctx.run_id} - page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                 topMargin=2 * cm, bottomMargin=2 * cm, title="IDSS decision report", author="IDSS")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    logger.info("Wrote PDF report {}.", out_path)
    return out_path


def _pdf_table(rows: list[list[str]], cell: ParagraphStyle, header: ParagraphStyle) -> Table:
    data = [[Paragraph(_pdf(str(value)), header if i == 0 else cell) for value in row] for i, row in enumerate(rows)]
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e4057")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b7c3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def _pdf_image(path: str) -> Image | None:
    if not path or not Path(path).is_file():
        return None
    try:
        width, height = ImageReader(path).getSize()
    except Exception as exc:  # unreadable chart: skip it
        logger.warning("Skipping unreadable chart {} ({}).", path, exc)
        return None
    target_width = IMAGE_WIDTH_CM * cm
    target_height = min(target_width * height / width, 18 * cm)
    return Image(path, width=target_height * width / height, height=target_height)


def _pdf(text: str) -> str:
    """Escape reportlab markup and replace characters the built-in PDF fonts cannot draw."""
    for old, new in _PDF_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _chart(charts: dict[str, str], name: str) -> bool:
    return bool(charts.get(name)) and Path(charts[name]).is_file()

"""Decision report as PDF (reportlab) and Markdown, from the same list of content blocks.

The report is assembled from the run's computed results only. Any missing part (a chart, verification,
recommendations, ...) is skipped or replaced by a short note, never an error.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from idss.core.logger import get_logger
from idss.core.schemas import TASK_DESCRIPTIONS, DecisionInsight

if TYPE_CHECKING:
    from idss.agents.manager import RunContext

logger = get_logger()

PROJECT_NAME = "Intelligent Decision Support System Using Generative AI and Machine Learning"
MAX_PROFILE_ROWS = 25
MAX_LEADERBOARD_ROWS = 5
MAX_DRIVER_ROWS = 10
MAX_FORECAST_ROWS = 10
IMAGE_WIDTH_CM = 15.5
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
    training, profile, requirement = ctx.training, ctx.profile, ctx.requirement
    blocks: list[Block] = [Block("title", PROJECT_NAME), Block("paragraph", "Decision report")]
    task = training.task_type if training else (requirement.problem.task_type if requirement else None)
    blocks.append(Block("table", rows=[
        ["Item", "Value"],
        ["Question", ctx.question.split("\n")[0]],
        ["Date", f"{datetime.now():%Y-%m-%d %H:%M}"],
        ["Dataset", Path(ctx.data_path).name],
        ["Run id", ctx.run_id],
        ["Task", TASK_DESCRIPTIONS.get(task, "-") if task else "-"],
        ["Target", str(training.target if training else (requirement.target if requirement else "-"))],
        ["Best model", training.best_model if training else "-"],
    ]))
    blocks.append(Block("pagebreak"))

    blocks.append(Block("heading", "1. Executive summary"))
    if insight is not None:
        blocks += [Block("paragraph", insight.executive_summary), Block("subheading", "Key findings"),
                   Block("bullets", items=insight.key_findings)]
    else:
        blocks.append(Block("paragraph", ctx.message or "No summary is available for this run."))

    blocks.append(Block("heading", "2. Data overview"))
    if profile is not None:
        blocks.append(Block("paragraph", f"{profile.n_rows:,} rows and {profile.n_cols} columns; "
                                         f"{profile.duplicate_rows:,} duplicate rows; {profile.memory_mb:.2f} MB in memory."))
        rows = [["Column", "Type", "Missing %", "Distinct"]] + [
            [c.name, c.semantic_type, f"{c.missing_pct:g}", f"{c.n_unique:,}"] for c in profile.columns[:MAX_PROFILE_ROWS]
        ]
        blocks.append(Block("table", rows=rows))
        blocks += [Block("subheading", "Data-quality warnings"),
                   Block("bullets", items=profile.warnings or ["No data-quality warnings."])]
    else:
        blocks.append(Block("paragraph", "No data profile is available."))

    blocks.append(Block("heading", "3. Model results"))
    if training is not None:
        blocks += _model_blocks(training, charts)
    else:
        blocks.append(Block("paragraph", "No model was trained in this run."))

    blocks.append(Block("heading", "4. Verification"))
    verification = ctx.verification
    if verification is not None:
        blocks.append(Block("paragraph", f"Overall: {'PASSED' if verification.passed else 'NOT PASSED'} "
                                         f"({sum(c.passed for c in verification.checks)}/{len(verification.checks)} checks)."))
        blocks.append(Block("table", rows=[["Check", "Result", "Detail"]] + [
            [c.name, "pass" if c.passed else "FAIL", c.detail] for c in verification.checks]))
        if verification.llm_verdict:
            blocks.append(Block("paragraph", f"Reviewer verdict: {verification.llm_verdict}"))
    else:
        blocks.append(Block("paragraph", "Verification was not run."))

    blocks.append(Block("heading", "5. Key drivers"))
    blocks += _driver_blocks(ctx, charts)

    blocks.append(Block("heading", "6. Recommendations"))
    recommendations = ctx.recommendations
    if recommendations is not None and recommendations.recommendations:
        blocks.append(Block("table", rows=[["Action", "Expected impact", "Confidence"]] + [
            [r.action, f"{r.expected_impact:,.4g} ({r.impact_unit})", r.confidence] for r in recommendations.recommendations]))
    if insight is not None:
        blocks += [Block("subheading", "In business terms"), Block("bullets", items=insight.recommended_actions)]
    if recommendations is not None and recommendations.notes:
        blocks.append(Block("bullets", items=recommendations.notes))
    if (recommendations is None or not recommendations.recommendations) and insight is None:
        blocks.append(Block("paragraph", "No recommendations are available."))

    blocks.append(Block("heading", "7. What-if example"))
    example = ctx.what_if_example
    if example is not None:
        changes = ", ".join(f"{k} -> {v}" for k, v in example.changes.items())
        rows = [["Item", "Value"], ["Change", changes], ["Output", example.output],
                ["Before", _fmt(example.before)], ["After", _fmt(example.after)],
                ["Change in output", "-" if example.delta is None else f"{example.delta:+.4g}"
                 + ("" if example.delta_pct is None else f" ({example.delta_pct:+.2f}%)")]]
        blocks.append(Block("table", rows=rows))
        if example.warnings:
            blocks.append(Block("bullets", items=example.warnings))
    else:
        blocks.append(Block("paragraph", "No what-if example was computed for this task."))

    blocks.append(Block("heading", "8. Risks and limitations"))
    blocks.append(Block("bullets", items=insight.risks_and_limitations if insight else
                        ["Results show associations in historical data, not guaranteed cause and effect."]))
    if insight is not None:
        blocks += [Block("subheading", "Next steps"), Block("bullets", items=insight.next_steps)]

    blocks.append(Block("heading", "Appendix"))
    blocks += _appendix_blocks(ctx, insight)
    return blocks


def _model_blocks(training: Any, charts: dict[str, str]) -> list[Block]:
    metric = training.primary_metric
    score = training.test_metrics.get(metric)
    text = (f"Best model: {training.best_model}. Primary metric: {metric} "
            f"({'higher' if training.direction == 'maximize' else 'lower'} is better) = {_fmt(score)} on the held-out "
            f"test split ({training.test_metrics.get('n_test_rows', '-')} rows, evaluated once).")
    if training.baseline is not None and metric in training.baseline.metrics:
        text += f" A simple {training.baseline.name} baseline scores {_fmt(training.baseline.metrics[metric])}."
    blocks = [Block("paragraph", text)]
    if training.decision_threshold is not None:
        default = training.test_metrics_default_threshold
        blocks.append(Block("paragraph", f"Decision threshold tuned on the validation split: {training.decision_threshold:.2f} "
                                         "(instead of the default 0.50)."))
        shared = [m for m in ("f1", "precision", "recall", "accuracy") if m in training.test_metrics and m in default]
        blocks.append(Block("table", rows=[["Metric", "At 0.50", f"At {training.decision_threshold:.2f}"]] + [
            [m, _fmt(default[m]), _fmt(training.test_metrics[m])] for m in shared]))
    metrics = [[k, _fmt(v)] for k, v in training.test_metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "n_test_rows"]
    if metrics:
        blocks += [Block("subheading", "Test metrics"), Block("table", rows=[["Metric", "Value"]] + metrics)]
    leaderboard = [e for e in training.leaderboard if e.rank is not None][:MAX_LEADERBOARD_ROWS]
    if leaderboard:
        blocks += [Block("subheading", f"Leaderboard (validation {metric})"), Block("table", rows=[
            ["Rank", "Model", "Valid score", "Inference ms/row", "Constraint issues"]] + [
            [str(e.rank), e.model + (" (selected)" if e.selected else ""), _fmt(e.valid_score),
             _fmt(e.inference_ms_per_row), "; ".join(e.violations) or "-"] for e in leaderboard])]
    for name in ("confusion_matrix", "actual_vs_pred", "forecast"):
        if _chart(charts, name):
            blocks.append(Block("image", path=charts[name]))
    return blocks


def _driver_blocks(ctx: "RunContext", charts: dict[str, str]) -> list[Block]:
    explanation = ctx.explanation
    if explanation is None:
        return [Block("paragraph", "No explanation was computed.")]
    global_explanation = explanation.global_explanation
    blocks = [Block("paragraph", f"Method: {global_explanation.method.replace('_', ' ')}; explains the "
                                 f"{global_explanation.output}; {global_explanation.n_samples:,} rows analysed.")]
    if _chart(charts, "feature_importance"):
        blocks.append(Block("image", path=charts["feature_importance"]))
    if global_explanation.features:
        blocks.append(Block("table", rows=[["Feature", "Share", "Direction", "Detail"]] + [
            [f.feature, f"{f.share:.1%}", f.direction, f.detail or "-"] for f in global_explanation.features[:MAX_DRIVER_ROWS]]))
    if _chart(charts, "shap_summary"):
        blocks.append(Block("image", path=charts["shap_summary"]))
    if global_explanation.notes:
        blocks.append(Block("bullets", items=global_explanation.notes))
    recommendations = ctx.recommendations
    if recommendations is not None and recommendations.risk_tiers:
        blocks.append(Block("subheading", "Risk tiers"))
        cutoffs = recommendations.tier_cutoffs
        if cutoffs is not None:
            blocks.append(Block("paragraph", (
                f"High: probability >= {cutoffs.high:.2f}, the larger of the decision threshold ({cutoffs.medium:.2f}) "
                f"and the {cutoffs.percentile}th percentile of predicted probabilities ({cutoffs.percentile_value:.2f}). "
                f"Medium: {cutoffs.medium:.2f} <= probability < {cutoffs.high:.2f}. Low: probability < {cutoffs.medium:.2f}.")))
        blocks += [Block("table", rows=[
            ["Tier", "Probability range", "Records", "Share", "Top drivers"]] + [
            [t.name, f"{t.min_probability:.2f} - {t.max_probability:.2f}", f"{t.count:,}", f"{t.share:.1%}",
             ", ".join(t.top_drivers) or "-"] for t in recommendations.risk_tiers])]
    if explanation.clusters:
        if _chart(charts, "cluster_sizes"):
            blocks.append(Block("image", path=charts["cluster_sizes"]))
        blocks += [Block("subheading", "Segments"), Block("table", rows=[["Segment", "Records", "Share", "What sets it apart"]] + [
            [str(c.cluster), f"{c.size:,}", f"{c.share:.1%}", "; ".join(c.distinguishing) or "-"] for c in explanation.clusters])]
    if explanation.forecast is not None:
        forecast = explanation.forecast
        blocks += [Block("subheading", f"Forecast (next {forecast.horizon} periods)"), Block("table", rows=[
            ["Date", "Forecast", "Lower", "Upper"]] + [
            [p.timestamp, _fmt(p.prediction), _fmt(p.lower), _fmt(p.upper)] for p in forecast.points[:MAX_FORECAST_ROWS]]),
            Block("bullets", items=[forecast.interval, *forecast.assumptions])]
    for local in explanation.local_examples[:1]:
        if local.contributions:
            blocks += [Block("subheading", f"Example: why one record got {_fmt(local.prediction)}"), Block("table", rows=[
                ["Feature", "Value", "Effect"]] + [[c.feature, _fmt(c.value), f"{c.impact:+.4g}"] for c in local.contributions])]
    return blocks


def _appendix_blocks(ctx: "RunContext", insight: DecisionInsight | None) -> list[Block]:
    rows = [["Setting", "Value"]]
    plan = ctx.plan
    if plan is not None:
        rows += [["Plan source", plan.source], ["Candidates", ", ".join(c.name for c in plan.candidates[: plan.budget.max_candidates])],
                 ["Budget", f"{plan.budget_mode}: {plan.budget.max_candidates} models x {plan.budget.n_trials_per_model} trials, "
                            f"{plan.budget.total_timeout_s:g}s total"], ["Primary metric", plan.primary_metric]]
    if ctx.requirement is not None:
        rows.append(["Constraints", "; ".join(ctx.requirement.problem.constraints) or "none"])
    if ctx.training is not None:
        rows.append(["Training time", f"{ctx.training.total_time_s:.1f}s"])
    rows.append(["Re-plans after verification", str(ctx.plan_revision_count)])
    if insight is not None:
        grounding = insight.grounding
        rows.append(["Brief written by", "LLM (grounding check passed)" if insight.generated_by == "llm"
                     else "template" + (" (LLM draft failed the grounding check)" if grounding and grounding.unsupported_numbers else "")])
    blocks = [Block("table", rows=rows)]
    knowledge = ctx.knowledge
    if knowledge is not None and knowledge.sources:
        blocks += [Block("subheading", "Knowledge sources"), Block("bullets", items=knowledge.sources)]
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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)

"""Step 6 demo: full offline runs (no LLM) on the churn and store-sales samples, ending in PDF/Markdown reports.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\demo_step6.py
The requirement is built deterministically from the data profile, so no API key is needed; insights come
from the template writer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idss.agents.manager import Orchestrator, RunContext, RunState  # noqa: E402  (needs PROJECT_ROOT on sys.path)
from idss.agents.requirement_agent import requirement_from_profile  # noqa: E402
from idss.core.logger import setup_logging  # noqa: E402
from idss.core.schemas import UserRequirement  # noqa: E402
from idss.ml.data_loader import load_dataset  # noqa: E402
from idss.ml.profiler import profile_dataset  # noqa: E402

RUNS = (
    ("customer_churn.csv", "Which customers are likely to churn and what should we do to retain them?"),
    ("store_sales.csv", "Forecast daily sales for the next month and show what drives them."),
)


def offline_requirement(path: Path, question: str) -> UserRequirement:
    """What the Requirement Agent would produce, built from the data profile instead of an LLM."""
    return requirement_from_profile(question, profile_dataset(load_dataset(path)))


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    setup_logging(level="WARNING", force=True)
    orchestrator = Orchestrator(None, budget_mode="fast")  # no LLM: every agent uses its deterministic path
    exit_code = 0
    for filename, question in RUNS:
        path = PROJECT_ROOT / "data" / "samples" / filename
        print(f"\n{'=' * 78}\n{filename}: {question}\n{'=' * 78}", flush=True)
        orchestrator.start(question, path, requirement=offline_requirement(path, question))
        ctx = orchestrator.run_until_blocked()
        _report(ctx)
        exit_code |= ctx.state != RunState.DONE
    return exit_code


def _report(ctx: RunContext) -> None:
    print(f"State: {ctx.state}" + (f" | errors: {ctx.errors}" if ctx.errors else ""))
    training = ctx.training
    if training is None:
        return
    score = training.test_metrics.get(training.primary_metric)
    threshold = f", threshold {training.decision_threshold:.2f}" if training.decision_threshold is not None else ""
    print(f"Best model: {training.best_model} | test {training.primary_metric} = {score:.4f}{threshold} | "
          f"verification {'passed' if ctx.verification and ctx.verification.passed else 'NOT passed'}")

    explanation = ctx.explanation
    if explanation is not None:
        print(f"\nTop drivers ({explanation.global_explanation.method}, {explanation.global_explanation.output}):")
        for feature in explanation.global_explanation.features[:5]:
            print(f"  {feature.feature:22s} {feature.share:6.1%}  {feature.direction:9s} {feature.detail}")

    recommendations = ctx.recommendations
    if recommendations is not None and recommendations.risk_tiers:
        print("\nRisk tiers:")
        for tier in recommendations.risk_tiers:
            print(f"  {tier.name:6s} p in [{tier.min_probability:.2f}, {tier.max_probability:.2f}]  {tier.count:5,} records "
                  f"({tier.share:5.1%})  avg p {tier.avg_probability or 0:.3f}  drivers: {', '.join(tier.top_drivers)}")

    if explanation is not None and explanation.forecast is not None:
        forecast = explanation.forecast
        points = forecast.points
        average = sum(p.prediction for p in points) / len(points)
        print(f"\nForecast: next {forecast.horizon} periods {points[0].timestamp} .. {points[-1].timestamp}, "
              f"average {average:,.0f} (first {points[0].prediction:,.0f}, last {points[-1].prediction:,.0f})")
        print(f"  Interval: {forecast.interval}")

    if recommendations is not None:
        print("\nTop recommendations:")
        for number, item in enumerate(recommendations.recommendations[:3], start=1):
            print(f"  {number}. {item.action}\n     impact {item.expected_impact:+,.4g} ({item.impact_unit}); "
                  f"confidence {item.confidence}")
        for note in recommendations.notes:
            print(f"  note: {note}")

    if ctx.what_if_example is not None:
        example = ctx.what_if_example
        delta = f"{example.delta:+.4g}" if example.delta is not None else "n/a"
        print(f"\nWhat-if example: {example.changes} -> {example.output}: {example.before:.4g} -> {example.after:.4g} ({delta})")
    if ctx.insight is not None:
        print(f"\nDecision brief ({ctx.insight.generated_by}): {ctx.insight.executive_summary}")
    print(f"\nReports:\n  PDF: {ctx.report_paths.get('pdf')}\n  MD:  {ctx.report_paths.get('markdown')}")


if __name__ == "__main__":
    sys.exit(main())

"""Step 4 benchmark: run the AutoML engine on the three sample datasets with the default (no-LLM) plan.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\benchmark_step4.py                     # fast mode, all samples
    .venv\\Scripts\\python.exe scripts\\benchmark_step4.py --mode full
    .venv\\Scripts\\python.exe scripts\\benchmark_step4.py --threshold-update  # Step 5: churn row + verification
Writes docs/results/step4_benchmark.md (the --threshold-update mode only adds/replaces the churn
"after threshold tuning" row and section, keeping the Step 4 results).
"""

import argparse
import os
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idss.agents.model_agent import ModelAgent  # noqa: E402  (needs PROJECT_ROOT on sys.path)
from idss.agents.requirement_agent import RequirementAgent  # noqa: E402
from idss.agents.verifier import Verifier  # noqa: E402
from idss.core.logger import setup_logging  # noqa: E402
from idss.core.schemas import (  # noqa: E402
    DatasetProfile,
    DatasetSpec,
    ProblemSpec,
    TrainingResult,
    UserRequirement,
    VerificationReport,
)
from idss.ml.data_loader import load_dataset  # noqa: E402
from idss.ml.profiler import profile_dataset  # noqa: E402
from idss.ml.trainer import AutoMLTrainer  # noqa: E402
from idss.storage.runs import run_dir  # noqa: E402

DATASETS = ("customer_churn.csv", "crop_price.csv", "store_sales.csv")
OUTPUT = PROJECT_ROOT / "docs" / "results" / "step4_benchmark.md"
THRESHOLD_DATASET = "customer_churn.csv"
THRESHOLD_ROW_LABEL = f"{THRESHOLD_DATASET} — after threshold tuning"
SECTION_START, SECTION_END = "<!-- step5-threshold:start -->", "<!-- step5-threshold:end -->"


def default_requirement(profile: DatasetProfile, name: str) -> UserRequirement:
    """What the Requirement Agent would produce for 'predict the obvious target', without an LLM."""
    requirement = UserRequirement(
        problem=ProblemSpec(task_type=profile.suggested_task_type, description=f"Benchmark on {name}."),
        datasets=[DatasetSpec(target_column=profile.suggested_target)],
        confidence=1.0,
    )
    return RequirementAgent(None).repair(requirement, profile)


def train(filename: str, mode: str) -> tuple[int, TrainingResult, UserRequirement, DatasetProfile]:
    df = load_dataset(PROJECT_ROOT / "data" / "samples" / filename)
    profile = profile_dataset(df)
    requirement = default_requirement(profile, filename)
    plan = ModelAgent(None).plan(requirement, profile, budget_mode=mode)
    print(f"Training on {filename} ({len(df):,} rows, {plan.task_type}) ...", flush=True)
    output_dir = run_dir(f"bench-step4-{mode}-{Path(filename).stem}")
    return len(df), AutoMLTrainer().run(df, requirement, profile, output_dir, plan=plan), requirement, profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the AutoML engine on the sample datasets (no LLM).")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--threshold-update", action="store_true",
                        help=f"retrain only {THRESHOLD_DATASET} and add the 'after threshold tuning' results")
    args = parser.parse_args(argv)
    setup_logging(level="WARNING", force=True)
    if args.threshold_update:
        return threshold_update(args.mode)

    results: list[tuple[str, int, TrainingResult]] = []
    for filename in DATASETS:
        rows, result, _, _ = train(filename, args.mode)
        results.append((filename, rows, result))

    summary = _summary_table(results)
    print("\n" + summary)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(_report(results, summary, args.mode), encoding="utf-8")
    print(f"\nSaved {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


SUMMARY_HEADER = (
    "| Dataset | Task | Rows | Best model | Primary metric | Test score | Search time (s) | Final fit (s) |\n"
    "|---|---|---:|---|---|---:|---:|---:|"
)


def _summary_row(label: str, rows: int, r: TrainingResult) -> str:
    score = r.test_metrics.get(r.primary_metric)
    return (
        f"| {label} | {r.task_type} | {rows:,} | {r.best_model} | {r.primary_metric} "
        f"({'higher' if r.direction == 'maximize' else 'lower'} is better) | {score:.4f} | "
        f"{r.total_time_s:.1f} | {r.complexity['train_time_s']:.2f} |"
    )


def _summary_table(results: list[tuple[str, int, TrainingResult]]) -> str:
    return "\n".join([SUMMARY_HEADER, *(_summary_row(filename, rows, r) for filename, rows, r in results)])


def threshold_update(mode: str) -> int:
    """Step 5: retrain the churn model with threshold tuning, verify it, and update the benchmark report."""
    rows, result, requirement, profile = train(THRESHOLD_DATASET, mode)
    report = Verifier(None).verify(requirement, result, profile)  # offline: rule-based checks only
    label = f"{THRESHOLD_ROW_LABEL} (t = {result.decision_threshold:.2f})"
    row = _summary_row(label, rows, result)
    print("\n" + SUMMARY_HEADER + "\n" + row)
    print("\nVerificationReport (rule-based, no LLM):")
    print(report.model_dump_json(indent=2))

    text = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
    if f"| {THRESHOLD_DATASET} |" not in text:
        print(f"\n{OUTPUT.relative_to(PROJECT_ROOT)} has no Step 4 churn row; run the full benchmark first.")
        return 1
    lines = [line for line in text.splitlines() if not line.startswith(f"| {THRESHOLD_ROW_LABEL}")]
    churn_row = next(i for i, line in enumerate(lines) if line.startswith(f"| {THRESHOLD_DATASET} |"))
    lines.insert(churn_row + 1, row)
    text = "\n".join(lines) + "\n"
    if SECTION_START in text:
        text = text[: text.index(SECTION_START)].rstrip() + "\n" + text[text.index(SECTION_END) + len(SECTION_END):].lstrip("\n")
    OUTPUT.write_text(text.rstrip() + "\n\n" + _threshold_section(result, report) + "\n", encoding="utf-8")
    print(f"\nUpdated {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


def _threshold_section(r: TrainingResult, report: VerificationReport) -> str:
    tuned, default = r.test_metrics, r.test_metrics_default_threshold

    def pair(metric: str) -> str:
        return f"{default.get(metric, float('nan')):.4f} → {tuned.get(metric, float('nan')):.4f}"

    failed = ", ".join(report.failed_checks) or "none"
    lines = [
        SECTION_START,
        f"## {THRESHOLD_DATASET} — decision-threshold tuning (Step 5, {datetime.now():%Y-%m-%d})",
        "",
        f"The probability cut-off for the positive class ('{tuned.get('positive_class')}') is tuned on the validation "
        f"split to maximise {r.primary_metric}, then applied unchanged to the test split (read once).",
        "",
        "| Model | Threshold | F1 (0.5 → tuned) | Precision (0.5 → tuned) | Recall (0.5 → tuned) | Accuracy (0.5 → tuned) | ROC AUC |",
        "|---|---:|---|---|---|---|---:|",
        f"| {r.best_model} | {r.decision_threshold:.2f} | {pair('f1')} | {pair('precision')} | {pair('recall')} | "
        f"{pair('accuracy')} | {tuned.get('roc_auc', float('nan')):.4f} |",
        "",
        f"Baseline ({r.baseline.name if r.baseline else 'n/a'}) test {r.primary_metric}: "
        f"{(r.baseline.metrics.get(r.primary_metric) if r.baseline else float('nan')):.4f}.",
        "",
        f"Rule-based verification (no LLM): **{'PASSED' if report.passed else 'FAILED'}** — "
        f"{sum(c.passed for c in report.checks)}/{len(report.checks)} checks passed; failed: {failed}.",
        "",
        *(f"- `{c.name}`: {'pass' if c.passed else 'FAIL'} — {c.detail}" for c in report.checks),
        SECTION_END,
    ]
    return "\n".join(lines)


def _report(results: list[tuple[str, int, TrainingResult]], summary: str, mode: str) -> str:
    budget = results[0][2].budget
    packages = ", ".join(f"{p} {version(p)}" for p in ("scikit-learn", "lightgbm", "xgboost", "optuna", "pandas"))
    parts = [
        "# Step 4 benchmark: AutoML engine",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Mode: **{mode}** — up to {budget.max_candidates} candidates, {budget.n_trials_per_model} Optuna trials "
        f"each, {budget.timeout_per_model_s:g}s per model, {budget.total_timeout_s:g}s total",
        "- Plan: default ranking from `ModelAgent(None)` (no LLM); target and task suggested by the profiler",
        "- Data: synthetic samples from `scripts/make_samples.py`; split 70/20/10 (stratified / chronological)",
        "- Selection uses the validation split; the test split is scored once, for the selected model only",
        f"- Environment: Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{os.cpu_count()} CPUs; {packages}",
        "",
        "## Summary (test split)",
        "",
        summary,
    ]
    for filename, _, r in results:
        parts += [
            "",
            f"## {filename} — leaderboard (validation {r.primary_metric})",
            "",
            "| Rank | Model | Status | Valid score | Inference (ms/row) | Size (KB) | Trials (failed) |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        for e in r.leaderboard:
            valid = f"{e.valid_score:.4f}" if e.valid_score is not None else "–"
            marker = " (selected)" if e.selected else ""
            parts.append(
                f"| {e.rank or '–'} | {e.model}{marker} | {e.status} | {valid} | {e.inference_ms_per_row or '–'} | "
                f"{e.model_size_kb or '–'} | {e.n_trials} ({e.n_failed_trials}) |"
            )
        other = {k: v for k, v in r.test_metrics.items() if isinstance(v, float) and k != r.primary_metric}
        parts += ["", "Other test metrics: " + ", ".join(f"{k} = {v:.4f}" for k, v in other.items())]
        if r.warnings:
            parts += ["", "Notes: " + " ".join(r.warnings)]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    sys.exit(main())

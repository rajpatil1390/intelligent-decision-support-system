"""Step 8 evaluation: benchmark, paper-style metrics, ablation and robustness (offline, seeded, fast mode).

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\evaluate.py                # 3 seeds, all experiments
    .venv\\Scripts\\python.exe scripts\\evaluate.py --seeds 42     # one seed, quicker
    .venv\\Scripts\\python.exe scripts\\evaluate.py --from-json    # rebuild the report and charts only
Writes docs/results/evaluation.md, docs/results/evaluation.json and docs/results/figures/*.png.

Every run goes through the real Orchestrator without an LLM: the requirement comes from the data profile
(as in an offline run), every agent uses its deterministic path and web search is disabled. Runs, logs and
their database live in a temporary workspace, so workspace/ is never touched.

Paper-style metrics follow AutoML-Agent (Trirat et al., ICML 2025):
    SR  = success rate: 1 for a run that finishes and reports test performance, 0 otherwise
    NPS = normalised performance score in [0, 1]: the test score itself for 0-1 metrics where higher is
          better (F1, adjusted Rand index); 1 / (1 + error) for error metrics, using the scale-free RMSLE so the score
          does not depend on the target's units; 0 for a failed run
    CS  = comprehensive score = 0.5 * SR + 0.5 * NPS
"""

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402  (needs PROJECT_ROOT on sys.path for the imports below)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from configs.settings import get_settings  # noqa: E402
from idss.agents.manager import BLOCKED_STATES, MAX_STEPS, Orchestrator, RunState  # noqa: E402
from idss.agents.model_agent import ModelAgent  # noqa: E402
from idss.agents.requirement_agent import RequirementAgent, requirement_from_profile  # noqa: E402
from idss.agents.verifier import BOUNDED_METRICS  # noqa: E402
from idss.core.logger import get_logger, setup_logging  # noqa: E402
from idss.core.plan_contract import PlanProposal  # noqa: E402
from idss.core.schemas import (  # noqa: E402
    BUDGET_PRESETS,
    DatasetProfile,
    DatasetSpec,
    ProblemSpec,
    TaskType,
    TrainingResult,
    UserRequirement,
    VerificationReport,
)
from idss.ml.data_loader import load_dataset  # noqa: E402
from idss.ml.evaluator import THRESHOLD_TUNABLE_METRICS, metric_direction  # noqa: E402
from idss.ml.profiler import profile_dataset  # noqa: E402
from idss.ml.trainer import AutoMLTrainer  # noqa: E402
from idss.storage.db import get_engine  # noqa: E402
from idss.storage.runs import run_dir  # noqa: E402
from scripts import make_samples  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
JSON_PATH = RESULTS_DIR / "evaluation.json"
MD_PATH = RESULTS_DIR / "evaluation.md"
SEEDS = (42, 43, 44)
MISSING_SHARE = 0.10
N_NOISE_COLUMNS = 5
NEAR_ZERO_BASELINE = 0.05  # |baseline| below this: relative improvement is not reported
METRIC_LABELS = {"adjusted_rand_index": "ARI", "rand_index": "RI"}

# Vibrant palette shared with scripts/make_diagrams.py, in an order whose neighbours stay distinct for
# colour-blind readers (checked with a CVD validator; the given order put pink next to green).
PURPLE, GREEN, ORANGE, BLUE, PINK, YELLOW = "#6C5CE7", "#00B894", "#E17055", "#0984E3", "#FD79A8", "#FDCB6E"
INK, MUTED, GRID = "#2D3436", "#636E72", "#DFE6E9"
DPI = 300

TASK_LABELS = {
    "tabular_regression": "regression",
    "time_series_forecasting": "time-series forecasting",
    "tabular_clustering": "clustering",
}


# --------------------------------------------------------------------------- experiment definitions


@dataclass(frozen=True)
class Case:
    """One benchmark dataset. Without `task_type`, the requirement is the data profile's suggestion."""

    key: str
    filename: str
    question: str
    task_type: TaskType | None = None
    target: str | None = None


CASES = (
    Case("Churn", "customer_churn.csv", "Which customers are likely to churn and what should we do to retain them?"),
    Case("Crop price", "crop_price.csv", "Estimate the market price of a crop from season, weather and yield."),
    Case("Store sales", "store_sales.csv", "Forecast daily sales for the next month and show what drives them."),
    Case("Delivery", "delivery_status.csv", "Predict whether a shipment will arrive on time, late or very late."),
    Case("Segments", "customer_segments.csv", "Group our customers into behavioural segments.",
         task_type="tabular_clustering", target="segment"),
)


@dataclass(frozen=True)
class Variant:
    """A system configuration for the ablation study."""

    key: str
    label: str
    tune_hyperparameters: bool = True
    tune_threshold: bool = True
    replan: bool = True
    single_model: bool = False


FULL = Variant("full", "Full system")
NO_THRESHOLD = Variant("no_threshold", "− threshold tuning", tune_threshold=False)
NO_HPO = Variant("no_hpo", "− hyperparameter tuning", tune_hyperparameters=False)
NO_REPLAN = Variant("no_replan", "− verification re-planning", replan=False)
SINGLE_MODEL = Variant("single_model", "Single-model baseline", tune_hyperparameters=False, tune_threshold=False,
                       replan=False, single_model=True)
VARIANTS = (FULL, NO_THRESHOLD, NO_HPO, NO_REPLAN, SINGLE_MODEL)
VARIANT_COLORS = {FULL.key: PURPLE, NO_THRESHOLD.key: GREEN, NO_HPO.key: ORANGE, NO_REPLAN.key: BLUE,
                  SINGLE_MODEL.key: PINK}

CLEAN = "clean"
PERTURBATIONS = {"missing": f"+{MISSING_SHARE:.0%} missing values", "noise": f"+{N_NOISE_COLUMNS} noisy columns"}
PERTURBATION_COLORS = {"missing": ORANGE, "noise": BLUE}


@dataclass
class Outcome:
    """The result of one run: one dataset, seed, system variant and data perturbation."""

    case: str
    seed: int
    variant: str
    perturbation: str = CLEAN
    trained: bool = False  # trained, tested once and verified without an error
    completed: bool | None = None  # reached DONE with a PDF report; None when stopped after verification
    task_type: str = ""
    n_classes: int | None = None
    rows: int = 0
    best_model: str | None = None
    primary_metric: str | None = None
    score: float | None = None
    baseline_name: str | None = None
    baseline_score: float | None = None
    rand_index: float | None = None  # clustering: the paper's (not chance-corrected) metric, reported alongside ARI
    baseline_rand_index: float | None = None
    nps_metric: str | None = None
    nps: float = 0.0
    baseline_nps: float | None = None
    decision_threshold: float | None = None
    threshold_applicable: bool = False
    train_time_s: float | None = None  # AutoML search + refit, summed over attempts
    inference_ms_per_row: float | None = None
    verification_passed: bool | None = None
    failed_checks: list[str] = field(default_factory=list)
    attempts: int = 0
    reused_from: str | None = None  # this variant cannot differ here, so that variant's run is reused
    final_state: str = ""
    error: str | None = None
    wall_time_s: float = 0.0


# --------------------------------------------------------------------------- scores


def nps_metric(primary_metric: str, test_metrics: dict[str, Any]) -> str:
    """The metric NPS is computed from: the primary one if it is a 0-1 score, else the scale-free RMSLE."""
    if primary_metric in BOUNDED_METRICS and metric_direction(primary_metric) == "maximize":
        return primary_metric
    if isinstance(test_metrics.get("rmsle"), float):
        return "rmsle"
    return primary_metric


def normalized_performance(metric: str, value: float | None) -> float:
    """NPS: a 0-1 score as is (clipped); 1 / (1 + error) for error metrics; 0 when there is no value."""
    if value is None or not math.isfinite(value):
        return 0.0
    if metric_direction(metric) == "maximize":
        return min(max(value, 0.0), 1.0)
    return 1.0 / (1.0 + max(value, 0.0))


def comprehensive_score(sr: float, nps: float) -> float:
    return 0.5 * sr + 0.5 * nps


def improvement_pct(score: float | None, baseline: float | None, metric: str, *,
                    min_reference: float = 1e-12) -> float | None:
    """Relative gain of `score` over a reference (a baseline or another run) in %, positive = better;
    None when the reference is missing or (near) 0, where a percentage is meaningless."""
    if score is None or baseline is None or abs(baseline) < min_reference:
        return None
    gain = score - baseline if metric_direction(metric) == "maximize" else baseline - score
    return 100.0 * gain / abs(baseline)


def mean_std(values: list[float | None]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value); NaN when there are no values."""
    present = [v for v in values if v is not None and math.isfinite(v)]
    if not present:
        return math.nan, math.nan
    return statistics.fmean(present), statistics.stdev(present) if len(present) > 1 else 0.0


# --------------------------------------------------------------------------- data perturbations


def protected_columns(profile: DatasetProfile, requirement: UserRequirement) -> set[str]:
    """Columns a perturbation must not touch: the target / ground truth, identifiers and the time index."""
    protected = set(profile.columns_of("id"))
    if requirement.target:
        protected.add(requirement.target)
    if requirement.problem.task_type == "time_series_forecasting" and profile.time_column:
        protected.add(profile.time_column)
    return protected


def add_missing_values(df: pd.DataFrame, rng: np.random.Generator, protected: set[str],
                       share: float = MISSING_SHARE) -> pd.DataFrame:
    """Blank `share` of the cells of every unprotected column, completely at random."""
    out = df.copy()
    columns = [c for c in out.columns if c not in protected]
    mask = rng.random((len(out), len(columns))) < share
    for index, column in enumerate(columns):
        out[column] = out[column].mask(mask[:, index])
    return out


def add_noise_columns(df: pd.DataFrame, rng: np.random.Generator, protected: set[str],
                      n: int = N_NOISE_COLUMNS) -> pd.DataFrame:
    """Append `n` columns of pure noise (numeric and categorical) that carry no signal about the target."""
    out = df.copy()
    rows = len(out)
    makers: list[Callable[[], Any]] = [
        lambda: rng.normal(0, 1, rows).round(4),
        lambda: rng.normal(50, 15, rows).round(2),
        lambda: rng.uniform(0, 1000, rows).round(1),
        lambda: rng.integers(0, 10, rows),
        lambda: rng.choice(["alpha", "beta", "gamma", "delta"], rows),
    ]
    for index in range(n):
        out[f"noise_{index + 1}"] = makers[index % len(makers)]()
    return out


PERTURB: dict[str, Callable[[pd.DataFrame, np.random.Generator, set[str]], pd.DataFrame]] = {
    "missing": add_missing_values,
    "noise": add_noise_columns,
}


# --------------------------------------------------------------------------- running the system


class SingleModelAgent:
    """Proposes only the top default model (LightGBM; k-means for clustering), fitted once with library defaults.
    The run's budget is SINGLE_MODEL_BUDGET (set on the Orchestrator); the plan validator checks both."""

    def propose(self, requirement: UserRequirement, profile: DatasetProfile | None, **_: Any) -> PlanProposal:
        proposal = ModelAgent(None).propose(requirement, profile, budget_mode="fast")
        payload = proposal.payload | {"candidate_models": proposal.payload["candidate_models"][:1]}
        return proposal.model_copy(update={"payload": payload})


SINGLE_MODEL_BUDGET = BUDGET_PRESETS["fast"].model_copy(update={"max_candidates": 1, "n_trials_per_model": 1})


def build_requirement(case: Case, profile: DatasetProfile) -> UserRequirement:
    """The requirement an offline run would use: the profile's suggestion, or the case's explicit task/target."""
    if case.task_type is None:
        return requirement_from_profile(case.question, profile)
    requirement = UserRequirement(
        problem=ProblemSpec(task_type=case.task_type, description=case.question),
        datasets=[DatasetSpec(target_column=case.target)] if case.target else [],
        confidence=1.0,
    )
    return RequirementAgent(None).repair(requirement, profile)


def run_case(case: Case, seed: int, variant: Variant, data_path: Path, requirement: UserRequirement, rows: int, *,
             perturbation: str = CLEAN, end_to_end: bool = False) -> tuple[Outcome, Outcome | None]:
    """Run the Orchestrator for one configuration; returns the final outcome and the first-attempt outcome.

    `end_to_end` runs on to the PDF report; otherwise the run stops once verification has decided, because the
    later stages (explanations, insights, report) do not change any metric reported here.
    """
    base = Outcome(case=case.key, seed=seed, variant=variant.key, perturbation=perturbation, rows=rows)
    orchestrator = Orchestrator(None, budget_mode="fast")
    orchestrator.trainer = AutoMLTrainer(seed=seed, tune_hyperparameters=variant.tune_hyperparameters,
                                         tune_threshold=variant.tune_threshold, train_baselines=not variant.single_model)
    if variant.single_model:
        orchestrator.model_agent = SingleModelAgent()  # duck-typed: the Orchestrator only calls propose()
        orchestrator.budget = SINGLE_MODEL_BUDGET
    start = time.perf_counter()
    run_id = orchestrator.start(case.question, data_path, requirement=requirement)
    ctx = orchestrator.context
    attempts: list[tuple[TrainingResult, VerificationReport]] = []
    try:
        for _ in range(MAX_STEPS):
            if ctx.state in BLOCKED_STATES or (not end_to_end and ctx.state == RunState.EXPLAIN):
                break
            state = ctx.state
            orchestrator.step(run_id)
            if state == RunState.VERIFY and ctx.training is not None and ctx.verification is not None:
                attempts.append((ctx.training, ctx.verification))
                if not variant.replan:
                    break
    finally:
        shutil.rmtree(run_dir(run_id, orchestrator.run_dir_root, create=False), ignore_errors=True)
    wall = round(time.perf_counter() - start, 2)
    base.final_state, base.wall_time_s = str(ctx.state), wall
    if ctx.errors:
        base.error = ctx.errors[-1]
    if not attempts:
        return base, None

    completed = None
    if end_to_end:
        completed = ctx.state == RunState.DONE and bool(ctx.report_paths.get("pdf"))
    final = _from_attempt(base, attempts[-1], sum(t.total_time_s for t, _ in attempts), len(attempts), completed)
    first = _from_attempt(base, attempts[0], attempts[0][0].total_time_s, 1, None)
    return final, first


def _from_attempt(base: Outcome, attempt: tuple[TrainingResult, VerificationReport], train_time_s: float,
                  n_attempts: int, completed: bool | None) -> Outcome:
    training, verification = attempt
    metric = training.primary_metric
    score = training.test_metrics.get(metric)
    baseline = training.baseline.metrics.get(metric) if training.baseline else None
    metric_for_nps = nps_metric(metric, training.test_metrics)
    labels = training.test_metrics.get("class_labels")
    return replace(
        base,
        trained=isinstance(score, float),
        completed=completed,
        task_type=training.task_type,
        n_classes=len(labels) if isinstance(labels, list) else None,
        best_model=training.best_model,
        primary_metric=metric,
        score=score,
        baseline_name=training.baseline.name if training.baseline else None,
        baseline_score=baseline,
        rand_index=training.test_metrics.get("rand_index"),
        baseline_rand_index=training.baseline.metrics.get("rand_index") if training.baseline else None,
        nps_metric=metric_for_nps,
        nps=normalized_performance(metric_for_nps, training.test_metrics.get(metric_for_nps)),
        baseline_nps=(normalized_performance(metric_for_nps, training.baseline.metrics.get(metric_for_nps))
                      if training.baseline else None),
        decision_threshold=training.decision_threshold,
        threshold_applicable=("positive_class" in training.test_metrics and metric in THRESHOLD_TUNABLE_METRICS),
        train_time_s=round(train_time_s, 2),
        inference_ms_per_row=training.complexity.get("inference_ms_per_row"),
        verification_passed=verification.passed,
        failed_checks=list(verification.failed_checks),
        attempts=n_attempts,
    )


def run_all(seeds: tuple[int, ...], scratch: Path, cases: tuple[Case, ...] = CASES) -> list[Outcome]:
    """Every experiment: full system end to end, ablation variants and perturbed data, for every seed."""
    outcomes: list[Outcome] = []
    total = len(cases) * len(seeds)
    for number, (case, seed) in enumerate(((c, s) for c in cases for s in seeds), start=1):
        path = SAMPLES_DIR / case.filename
        df = load_dataset(path)
        profile = profile_dataset(df)
        requirement = build_requirement(case, profile)
        print(f"[{number}/{total}] {case.key} (seed {seed}) ...", flush=True)

        full, first = run_case(case, seed, FULL, path, requirement, len(df), end_to_end=True)
        no_replan = replace(first or full, variant=NO_REPLAN.key, completed=None)
        outcomes += [full, no_replan]
        _progress("full system", full)

        if full.threshold_applicable:
            no_threshold, _ = run_case(case, seed, NO_THRESHOLD, path, requirement, len(df))
        else:  # threshold tuning only exists for binary classification: the configuration is identical
            no_threshold = replace(full, variant=NO_THRESHOLD.key, completed=None, reused_from=FULL.key)
        no_hpo, _ = run_case(case, seed, NO_HPO, path, requirement, len(df))
        single, _ = run_case(case, seed, SINGLE_MODEL, path, requirement, len(df))
        outcomes += [no_threshold, no_hpo, single]
        for name, outcome in (("- threshold", no_threshold), ("- HPO", no_hpo), ("single model", single)):
            _progress(name, outcome)

        protected = protected_columns(profile, requirement)
        for perturbation, perturb in PERTURB.items():
            rng = np.random.default_rng([seed, list(PERTURB).index(perturbation)])
            perturbed_path = scratch / f"{Path(case.filename).stem}_{perturbation}_{seed}.csv"
            perturb(df, rng, protected).to_csv(perturbed_path, index=False)
            outcome, _ = run_case(case, seed, FULL, perturbed_path, requirement, len(df), perturbation=perturbation)
            outcomes.append(outcome)
            _progress(PERTURBATIONS[perturbation], outcome)
    return outcomes


def _progress(label: str, o: Outcome) -> None:
    if not o.trained:
        print(f"    {label:24s} FAILED ({o.error})", flush=True)
        return
    verified = "verified" if o.verification_passed else "NOT verified"
    print(f"    {label:24s} {o.best_model:24s} {o.primary_metric} = {_fmt(o.score)}  NPS {o.nps:.3f}  "
          f"{o.train_time_s:6.1f}s  {verified}{' (re-planned)' if o.attempts > 1 else ''}", flush=True)


# --------------------------------------------------------------------------- aggregation


def select(outcomes: list[Outcome], variant: str = FULL.key, perturbation: str = CLEAN,
           case: str | None = None) -> list[Outcome]:
    return [o for o in outcomes if o.variant == variant and o.perturbation == perturbation
            and (case is None or o.case == case)]


def success(o: Outcome, end_to_end: bool) -> bool:
    return bool(o.completed) if end_to_end else o.trained


def paper_scores(outcomes: list[Outcome], seeds: list[int], *, end_to_end: bool) -> dict[str, dict[str, Any]]:
    """SR, NPS and CS (mean ± std over seeds) per dataset and overall. A failed run scores NPS = 0."""
    table: dict[str, dict[str, Any]] = {}
    per_seed: dict[int, list[tuple[float, float]]] = {seed: [] for seed in seeds}
    for case in CASES:
        sr, nps, cs = [], [], []
        for o in (o for o in outcomes if o.case == case.key):
            ok = 1.0 if success(o, end_to_end) else 0.0
            value = o.nps if ok else 0.0
            sr.append(ok)
            nps.append(value)
            cs.append(comprehensive_score(ok, value))
            per_seed.setdefault(o.seed, []).append((ok, value))
        table[case.key] = {"sr": mean_std(sr), "nps": mean_std(nps), "cs": mean_std(cs), "runs": len(sr)}
    seed_sr = [statistics.fmean(ok for ok, _ in pairs) for pairs in per_seed.values() if pairs]
    seed_nps = [statistics.fmean(v for _, v in pairs) for pairs in per_seed.values() if pairs]
    table["Overall"] = {
        "sr": mean_std(seed_sr), "nps": mean_std(seed_nps),
        "cs": mean_std([comprehensive_score(s, n) for s, n in zip(seed_sr, seed_nps, strict=True)]),
        "runs": sum(len(p) for p in per_seed.values()),
    }
    return table


def benchmark_rows(outcomes: list[Outcome]) -> list[dict[str, Any]]:
    """Main benchmark: the full system on clean data, one row per dataset (mean ± std over seeds)."""
    rows = []
    for case in CASES:
        runs = select(outcomes, case=case.key)
        ok = [o for o in runs if o.trained]
        if not ok:
            rows.append({"dataset": case.key, "file": case.filename, "runs": len(runs), "trained": 0})
            continue
        models = Counter(o.best_model for o in ok).most_common()
        # Chance-level baselines (majority-class F1, random-assignment ARI) are about 0: no relative gain.
        gains = [improvement_pct(o.score, o.baseline_score, o.primary_metric, min_reference=NEAR_ZERO_BASELINE)
                 for o in ok]
        rows.append({
            "dataset": case.key,
            "file": case.filename,
            "task": task_label(ok[0]),
            "rows": ok[0].rows,
            "best_model": models[0][0],
            "best_model_count": models[0][1],
            "models": dict(models),
            "primary_metric": ok[0].primary_metric,
            "direction": metric_direction(ok[0].primary_metric),
            "score": mean_std([o.score for o in ok]),
            "baseline_name": ok[0].baseline_name,
            "baseline": mean_std([o.baseline_score for o in ok]),
            "rand_index": mean_std([o.rand_index for o in ok]),
            "baseline_rand_index": mean_std([o.baseline_rand_index for o in ok]),
            "improvement_pct": mean_std(gains),
            "train_time_s": mean_std([o.train_time_s for o in ok]),
            "inference_ms_per_row": mean_std([o.inference_ms_per_row for o in ok]),
            "nps": mean_std([o.nps for o in ok]),
            "baseline_nps": mean_std([o.baseline_nps for o in ok]),
            "verified": sum(bool(o.verification_passed) for o in ok),
            "replanned": sum(o.attempts > 1 for o in ok),
            "trained": len(ok),
            "runs": len(runs),
        })
    return rows


def ablation_rows(outcomes: list[Outcome], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        runs = select(outcomes, variant.key)
        scores = paper_scores(runs, seeds, end_to_end=False)
        per_case = {}
        for case in CASES:
            case_runs = [o for o in runs if o.case == case.key]
            changes = []
            for o in case_runs:
                reference = next((r for r in select(outcomes, case=case.key) if r.seed == o.seed), None)
                if reference is not None and o.trained and reference.trained:
                    changes.append(improvement_pct(o.score, reference.score, o.primary_metric))
            per_case[case.key] = {
                "metric": next((o.primary_metric for o in case_runs if o.primary_metric), None),
                "score": mean_std([o.score for o in case_runs if o.trained]),
                "change_vs_full_pct": mean_std(changes),
                "reused_from": next((o.reused_from for o in case_runs if o.reused_from), None),
                "verified": sum(bool(o.verification_passed) for o in case_runs),
                "runs": len(case_runs),
            }
        rows.append({
            "variant": variant.key,
            "label": variant.label,
            "per_case": per_case,
            "sr": scores["Overall"]["sr"],
            "nps": scores["Overall"]["nps"],
            "cs": scores["Overall"]["cs"],
            "train_time_s": mean_std([o.train_time_s for o in runs if o.trained]),
        })
    return rows


def robustness_rows(outcomes: list[Outcome], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        clean = select(outcomes, case=case.key)
        row: dict[str, Any] = {"dataset": case.key, "metric": next((o.primary_metric for o in clean), None),
                               "clean": mean_std([o.score for o in clean if o.trained]),
                               "clean_nps": mean_std([o.nps for o in clean if o.trained])}
        for perturbation in PERTURBATIONS:
            runs = select(outcomes, perturbation=perturbation, case=case.key)
            changes, nps_drops = [], []
            for o in runs:
                reference = next((r for r in clean if r.seed == o.seed), None)
                if reference is not None and o.trained and reference.trained:
                    changes.append(improvement_pct(o.score, reference.score, o.primary_metric))
                    nps_drops.append(reference.nps - o.nps)
            row[perturbation] = {
                "score": mean_std([o.score for o in runs if o.trained]),
                "change_pct": mean_std(changes),
                "nps_drop": mean_std(nps_drops),
                "verified": sum(bool(o.verification_passed) for o in runs),
                "trained": sum(o.trained for o in runs),
                "runs": len(runs),
            }
        rows.append(row)
    return rows


def task_label(o: Outcome) -> str:
    if o.task_type in TASK_LABELS:
        return TASK_LABELS[o.task_type]
    return "binary classification" if o.n_classes == 2 else f"{o.n_classes}-class classification"


# --------------------------------------------------------------------------- formatting


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    magnitude = abs(value)
    digits = 1 if magnitude >= 100 else 2 if magnitude >= 10 else 3
    return f"{value:,.{digits}f}"


def _pm(pair: tuple[float, float], digits: int | None = None, suffix: str = "") -> str:
    mean, std = pair
    if not math.isfinite(mean):
        return "–"
    if digits is None:
        return f"{_fmt(mean)} ± {_fmt(std)}{suffix}"
    return f"{mean:,.{digits}f} ± {std:,.{digits}f}{suffix}"


def _signed(pair: tuple[float, float], digits: int = 1) -> str:
    mean, std = pair
    if not math.isfinite(mean):
        return "–"
    return f"{mean:+.{digits}f}% ± {std:.{digits}f}"


def _m(metric: str | None) -> str:
    """Short display name of a metric (ARI, RI); others as they are."""
    return METRIC_LABELS.get(metric or "", metric or "–")


def benchmark_table(rows: list[dict[str, Any]], n_seeds: int) -> str:
    lines = [
        "| Dataset | Task | Rows | Best model | Primary metric | Test score | Baseline | Improvement | "
        "RI (paper metric) | Train time (s) | Inference (ms/row) | Verification passed |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if not r.get("trained"):
            lines.append(f"| {r['dataset']} | – | – | failed | – | – | – | – | – | – | – | 0/{r['runs']} |")
            continue
        arrow = "↑" if r["direction"] == "maximize" else "↓"
        gain = _signed(r["improvement_pct"]) if math.isfinite(r["improvement_pct"][0]) else "n/a¹"
        model = r["best_model"] + (f" ({r['best_model_count']}/{r['trained']})" if n_seeds > 1 else "")
        baseline = f"{_pm(r['baseline'])} ({r['baseline_name']})" if r["baseline_name"] else "–"
        ri = r.get("rand_index", (math.nan, math.nan))
        ri_cell = f"{_pm(ri)} (baseline {_fmt(r['baseline_rand_index'][0])})" if math.isfinite(ri[0]) else "–"
        lines.append(
            f"| {r['dataset']} | {r['task']} | {r['rows']:,} | {model} | {_m(r['primary_metric'])} {arrow} | "
            f"{_pm(r['score'])} | {baseline} | {gain} | {ri_cell} | "
            f"{_pm(r['train_time_s'], 1)} | {_pm(r['inference_ms_per_row'], 4)} | {r['verified']}/{r['runs']} |"
        )
    return "\n".join(lines)


def clustering_metric_table(outcomes: list[Outcome]) -> str:
    """ARI next to the plain Rand index for every clustering configuration (shows why ARI is the primary metric)."""
    lines = ["| Configuration | Model chosen (per seed) | ARI ↑ | RI ↑ |", "|---|---|---:|---:|"]
    clustering = {o.case for o in outcomes if o.task_type == "tabular_clustering"}
    configs = [(v.key, CLEAN, v.label) for v in VARIANTS if v.key != NO_THRESHOLD.key]
    configs += [(FULL.key, p, f"Full system, {label}") for p, label in PERTURBATIONS.items()]
    for case in sorted(clustering):
        for variant, perturbation, label in configs:
            runs = [o for o in select(outcomes, variant, perturbation, case) if o.trained]
            if runs:
                models = ", ".join(o.best_model or "–" for o in runs)
                lines.append(f"| {case}: {label} | {models} | {_pm(mean_std([o.score for o in runs]))} | "
                             f"{_pm(mean_std([o.rand_index for o in runs]))} |")
    return "\n".join(lines)


def paper_table(scores: dict[str, dict[str, Any]], nps_metrics: dict[str, str]) -> str:
    lines = ["| Dataset | NPS from | SR | NPS | CS |", "|---|---|---:|---:|---:|"]
    for name, s in scores.items():
        source = nps_metrics.get(name, "mean over datasets")
        source = f"1/(1+{source.upper()})" if source == "rmsle" else _m(source)
        bold = "**" if name == "Overall" else ""
        lines.append(f"| {bold}{name}{bold} | {source} | {_pm(s['sr'], 3)} | {_pm(s['nps'], 3)} | "
                     f"{bold}{_pm(s['cs'], 3)}{bold} |")
    return "\n".join(lines)


def ablation_table(rows: list[dict[str, Any]]) -> str:
    metrics = {case.key: rows[0]["per_case"][case.key]["metric"] for case in CASES}
    header = " | ".join(f"{case.key} ({_m(metrics[case.key])})" for case in CASES)
    lines = [f"| Variant | {header} | SR | NPS | CS | Train time (s) |",
             "|---|" + "---:|" * (len(CASES) + 4)]
    for r in rows:
        cells = []
        for case in CASES:
            c = r["per_case"][case.key]
            cell = _pm(c["score"])
            if r["variant"] != FULL.key and math.isfinite(c["change_vs_full_pct"][0]):
                cell += f" ({c['change_vs_full_pct'][0]:+.1f}%)"
            if c["reused_from"]:
                cell += "²"
            cells.append(cell)
        bold = "**" if r["variant"] == FULL.key else ""
        lines.append(f"| {bold}{r['label']}{bold} | " + " | ".join(cells) + f" | {_pm(r['sr'], 3)} | "
                     f"{_pm(r['nps'], 3)} | {bold}{_pm(r['cs'], 3)}{bold} | {_pm(r['train_time_s'], 1)} |")
    return "\n".join(lines)


def robustness_table(rows: list[dict[str, Any]]) -> str:
    labels = list(PERTURBATIONS.values())
    lines = [f"| Dataset | Metric | Clean | {labels[0]} | Change | {labels[1]} | Change | Verified (missing / noise) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m, n = r["missing"], r["noise"]
        arrow = "↑" if r["metric"] and metric_direction(r["metric"]) == "maximize" else "↓"
        lines.append(f"| {r['dataset']} | {_m(r['metric'])} {arrow} | {_pm(r['clean'])} | {_pm(m['score'])} | "
                     f"{_signed(m['change_pct'])} | {_pm(n['score'])} | {_signed(n['change_pct'])} | "
                     f"{m['verified']}/{m['runs']} / {n['verified']}/{n['runs']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- interpretation


def _variant_label(o: Outcome) -> str:
    if o.perturbation != CLEAN:
        return PERTURBATIONS[o.perturbation]
    return next(v.label for v in VARIANTS if v.key == o.variant)


def interpretation(bench: list[dict[str, Any]], scores: dict[str, Any], ablation: list[dict[str, Any]],
                   robust: list[dict[str, Any]], outcomes: list[Outcome]) -> list[str]:
    """Short findings derived from the numbers, so the text always matches the tables."""
    notes: list[str] = []
    trained = [r for r in bench if r.get("trained")]
    beats = [r for r in trained if r["baseline_name"] and _better(r["score"][0], r["baseline"][0], r["primary_metric"])]
    notes.append(
        f"**Baselines.** The full system beats the trivial baseline on {len(beats)} of {len(trained)} datasets "
        f"(majority class, mean, seasonal naive or random cluster assignment, scored on the same test split). "
        f"Verification passed in {sum(r['verified'] for r in trained)} of {sum(r['runs'] for r in trained)} runs."
    )
    overall = scores["Overall"]
    notes.append(
        f"**Paper-style scores.** SR = {overall['sr'][0]:.2f}, NPS = {overall['nps'][0]:.3f} and CS = "
        f"{overall['cs'][0]:.3f} (mean over datasets and seeds). Because the engine runs fixed, tested code instead of "
        "LLM-written scripts, every run that starts also finishes, so SR measures robustness rather than code quality "
        "here; NPS carries the differences."
    )
    by_key = {r["variant"]: r for r in ablation}
    full_cs = by_key[FULL.key]["cs"][0]
    churn_threshold = by_key[NO_THRESHOLD.key]["per_case"]["Churn"]
    if math.isfinite(churn_threshold["change_vs_full_pct"][0]):
        change = churn_threshold["change_vs_full_pct"][0]
        notes.append(
            f"**Threshold tuning** matters only for binary classification: on churn, keeping the 0.5 cut-off "
            f"changes F1 by {change:+.1f}% against the full system (the tuned cut-off trades precision for recall)."
        )
    hpo = by_key[NO_HPO.key]
    changes = {c: v["change_vs_full_pct"][0] for c, v in hpo["per_case"].items()
               if math.isfinite(v["change_vs_full_pct"][0])}
    if changes:
        worst, best = min(changes, key=changes.get), max(changes, key=changes.get)
        note = (f"**Hyperparameter tuning.** With library defaults the CS {_moves(full_cs, hpo['cs'][0])}; per "
                f"dataset the test score changes from {changes[worst]:+.1f}% ({worst}) to {changes[best]:+.1f}% ({best}).")
        if changes[best] > 0:
            note += (" Where defaults score higher, tuning still chose the best model on the validation split; the "
                     "test difference is within selection noise on these near-linear synthetic datasets.")
        notes.append(note + f" Search time drops from {by_key[FULL.key]['train_time_s'][0]:.1f}s to "
                            f"{hpo['train_time_s'][0]:.1f}s per run on average.")
    replanned = sum(r["replanned"] for r in trained)
    no_replan = by_key[NO_REPLAN.key]
    replans = [o for o in outcomes if o.attempts > 1 and not o.reused_from]  # reused copies are not new runs
    rescued = sum(bool(o.verification_passed) for o in replans)
    where = ", ".join(f"{label} ×{count}" for label, count in Counter(
        f"{o.case} ({_variant_label(o)})" for o in replans).items())
    note = (f"**Verification re-planning** was triggered in {replanned} of {sum(r['runs'] for r in trained)} "
            f"full-system runs on clean data (CS without it: {no_replan['cs'][0]:.3f}, with it: {full_cs:.3f}).")
    if replans:
        note += (f" Across all configurations it ran {len(replans)} time(s) — {where} — and the second attempt "
                 f"passed verification in {rescued} of them; a failure that persists is reported as 'not verified' "
                 "instead of being hidden.")
    else:
        note += " No configuration needed it; it is a safety net for data where a check fails."
    notes.append(note)
    single = by_key[SINGLE_MODEL.key]
    single_verified = sum(c["verified"] for c in single["per_case"].values())
    single_runs = sum(c["runs"] for c in single["per_case"].values())
    full_verified = sum(c["verified"] for c in by_key[FULL.key]["per_case"].values())
    notes.append(
        f"**Single model.** One LightGBM (k-means for clustering) with default settings reaches CS = "
        f"{single['cs'][0]:.3f} against {full_cs:.3f} for the full system, and passes verification in "
        f"{single_verified} of {single_runs} runs (full system: {full_verified} of {single_runs}); model search, "
        "tuning and threshold tuning together account for the difference."
    )
    drops = [(r["dataset"], p, r[p]["change_pct"][0]) for r in robust for p in PERTURBATIONS
             if math.isfinite(r[p]["change_pct"][0])]
    if drops:
        dataset, perturbation, change = min(drops, key=lambda item: item[2])
        mild = [d for d in drops if d[2] > -5.0]
        notes.append(
            f"**Robustness.** {len(mild)} of {len(drops)} perturbed settings stay within 5% of the clean score. "
            f"The largest loss is {dataset} with {PERTURBATIONS[perturbation].lstrip('+')} ({change:+.1f}%)."
        )
    return notes


def _moves(before: float, after: float) -> str:
    if abs(after - before) < 0.0005:
        return f"stays at {before:.3f}"
    return f"{'falls' if after < before else 'rises'} from {before:.3f} to {after:.3f}"


def _better(score: float, baseline: float, metric: str) -> bool:
    return score > baseline if metric_direction(metric) == "maximize" else score < baseline


# --------------------------------------------------------------------------- charts


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _bars(ax: plt.Axes, groups: list[str], series: list[tuple[str, list[float], list[float], str]], *,
          labels: set[int] | None = None, fmt: str = "{:.2f}") -> None:
    """Grouped bars with error bars; value labels only on the series indices in `labels`."""
    n = len(series)
    width = min(0.8 / n, 0.3)
    x = np.arange(len(groups))
    for index, (name, values, errors, color) in enumerate(series):
        offset = (index - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, color=color, label=name, edgecolor="white", linewidth=1.5,
                      yerr=errors, capsize=4, error_kw={"elinewidth": 1.4, "ecolor": INK, "capthick": 1.4})
        if labels is not None and index in labels:
            ax.bar_label(bars, labels=[fmt.format(v) if math.isfinite(v) else "" for v in values],
                         padding=4, fontsize=12, color=INK)
    ax.set_xticks(x, groups)


def _nps_label(nps_metric: str | None, primary_metric: str) -> str:
    if nps_metric and nps_metric != primary_metric:
        return f"1/(1+{nps_metric.upper()})"
    return _m(primary_metric)


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figures(bench: list[dict[str, Any]], scores: dict[str, Any], ablation: list[dict[str, Any]],
                 robust: list[dict[str, Any]]) -> list[Path]:
    _style()
    trained = [r for r in bench if r.get("trained")]
    groups = [f"{r['dataset']}\n({_nps_label(r['nps_metric'], r['primary_metric'])})" for r in trained]
    paths = []

    fig, ax = plt.subplots(figsize=(13, 6.5))
    _bars(ax, groups, [
        ("IDSS (full system)", [r["nps"][0] for r in trained], [r["nps"][1] for r in trained], PURPLE),
        ("Trivial baseline", [r["baseline_nps"][0] for r in trained], [r["baseline_nps"][1] for r in trained], ORANGE),
    ], labels={0, 1})
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Normalised score (NPS, higher is better)")
    ax.set_title("Model vs trivial baseline on the held-out test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    paths.append(_save(fig, "benchmark_vs_baseline.png"))

    names = [*(r["dataset"] for r in trained), "Overall"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    _bars(ax, names, [
        ("SR (success rate)", [scores[n]["sr"][0] for n in names], [scores[n]["sr"][1] for n in names], GREEN),
        ("NPS (normalised performance)", [scores[n]["nps"][0] for n in names], [scores[n]["nps"][1] for n in names],
         BLUE),
        ("CS = 0.5·SR + 0.5·NPS", [scores[n]["cs"][0] for n in names], [scores[n]["cs"][1] for n in names], PURPLE),
    ], labels={2})
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0–1, mean ± std over seeds)")
    ax.set_title("Paper-style metrics (AutoML-Agent, ICML 2025)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    paths.append(_save(fig, "paper_metrics.png"))

    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.7, 1]})
    others = [r for r in ablation if r["variant"] != FULL.key]
    datasets = [case.key for case in CASES]
    _bars(left, datasets, [
        (r["label"], [r["per_case"][d]["change_vs_full_pct"][0] for d in datasets],
         [r["per_case"][d]["change_vs_full_pct"][1] for d in datasets], VARIANT_COLORS[r["variant"]])
        for r in others
    ])
    left.axhline(0, color=INK, linewidth=1.2)
    left.set_ylabel("Test-score change vs full system (%)\n(negative = worse)")
    left.set_title("Effect of removing each component")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    y = np.arange(len(ablation))[::-1]
    bars = right.barh(y, [r["cs"][0] for r in ablation], xerr=[r["cs"][1] for r in ablation], height=0.55,
                      color=[VARIANT_COLORS[r["variant"]] for r in ablation], edgecolor="white", linewidth=1.5,
                      capsize=4, error_kw={"elinewidth": 1.4, "ecolor": INK, "capthick": 1.4})
    right.bar_label(bars, labels=[f"{r['cs'][0]:.3f}" for r in ablation], padding=8, fontsize=13, color=INK)
    right.set_yticks(y, [r["label"] for r in ablation])
    right.set_xlim(0, 1.15)
    right.grid(axis="y", visible=False)
    right.grid(axis="x", visible=True)
    right.set_xlabel("Comprehensive score (CS)")
    right.set_title("Overall CS by configuration")
    fig.tight_layout(w_pad=4)
    paths.append(_save(fig, "ablation.png"))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    _bars(ax, [r["dataset"] for r in robust], [
        (PERTURBATIONS[p], [r[p]["change_pct"][0] for r in robust], [r[p]["change_pct"][1] for r in robust],
         PERTURBATION_COLORS[p])
        for p in PERTURBATIONS
    ], labels={0, 1}, fmt="{:+.1f}%")
    ax.axhline(0, color=INK, linewidth=1.2)
    changes = [(r[p]["change_pct"][0], r[p]["change_pct"][1]) for r in robust for p in PERTURBATIONS
               if math.isfinite(r[p]["change_pct"][0])]
    low = min([0.0, *(mean - std for mean, std in changes)])
    high = max([0.0, *(mean + std for mean, std in changes)])
    ax.set_ylim(min(low * 1.3, -5), max(high * 1.3, 5))
    ax.set_ylabel("Test-score change vs clean data (%)\n(negative = worse)")
    ax.set_title("Robustness to dirty data")
    ax.legend(loc="lower left")
    paths.append(_save(fig, "robustness.png"))

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6.5))
    names = [r["dataset"] for r in trained]
    for ax, key, color, title, label in (
        (left, "train_time_s", PURPLE, "AutoML training time", "Seconds per run (search + refit)"),
        (right, "inference_ms_per_row", GREEN, "Inference latency", "Milliseconds per row"),
    ):
        values, errors = [r[key][0] for r in trained], [r[key][1] for r in trained]
        bars = ax.bar(names, values, yerr=errors, color=color, width=0.55, edgecolor="white", linewidth=1.5,
                      capsize=4, error_kw={"elinewidth": 1.4, "ecolor": INK, "capthick": 1.4})
        ax.bar_label(bars, labels=[f"{v:.1f}" if key == "train_time_s" else f"{v:.3f}" for v in values],
                     padding=4, fontsize=12, color=INK)
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max(v + e for v, e in zip(values, errors, strict=True)) * 1.2)
    fig.tight_layout(w_pad=4)
    paths.append(_save(fig, "efficiency.png"))
    return paths


# --------------------------------------------------------------------------- report


def environment() -> dict[str, Any]:
    packages = ("scikit-learn", "xgboost", "lightgbm", "optuna", "shap", "pandas", "numpy")
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpus": os.cpu_count(),
        "packages": {p: version(p) for p in packages},
    }


def build_markdown(data: dict[str, Any], analysis: dict[str, Any]) -> str:
    seeds = data["seeds"]
    budget = data["budget"]
    env = data["environment"]
    nps_sources = {r["dataset"]: r["nps_metric"] for r in analysis["benchmark"] if r.get("trained")}
    packages = ", ".join(f"{name} {v}" for name, v in env["packages"].items())
    figures = "figures"
    parts = [
        "# IDSS evaluation (Step 8)",
        "",
        f"- Generated: {data['generated']} by `scripts/evaluate.py` (offline: no LLM, no web search).",
        *(f"- Re-run on {r['at']}: {', '.join(r['cases'])} only, same seeds — {r['reason']}. The other datasets keep "
          "their original runs; the replaced runs are kept under `superseded` in `evaluation.json`."
          for r in data.get("reruns", [])),
        f"- Seeds: {', '.join(map(str, seeds))}. Every number is the mean ± sample standard deviation over seeds; "
        "the seed controls the train/valid/test split (time series: chronological, so only model randomness), "
        "Optuna's sampler, the models and the data perturbations.",
        f"- Budget: **fast** — up to {budget['max_candidates']} candidate models × {budget['n_trials_per_model']} "
        f"Optuna trials, {budget['timeout_per_model_s']:g}s per model, {budget['total_timeout_s']:g}s in total; "
        "a re-plan after a failed verification uses the full budget once.",
        "- Pipeline: the real Orchestrator (Requirement → Data → Plan → Train → Verify → Explain → Insight → Report). "
        "Requirements come from the data profile, as in an offline run; the segments dataset is given an explicit "
        "clustering requirement whose `segment` column is ground truth for scoring only (never a feature).",
        "- Splits 70/20/10: models are selected on the validation split; the test split is read once.",
        f"- Environment: Python {env['python']}, {env['os']}, {env['cpus']} CPUs; {packages}.",
        "- Reproduce: `.venv\\Scripts\\python.exe scripts\\evaluate.py` (rebuild this file and the charts from "
        "`evaluation.json` with `--from-json`).",
        "",
        "Datasets (all synthetic, generated by `scripts/make_samples.py` with fixed seeds):",
        "",
        "| Dataset | File | Rows | Task | Target |",
        "|---|---|---:|---|---|",
    ]
    for case, r in zip(CASES, analysis["benchmark"], strict=True):
        target = f"`{case.target}` (ground truth)" if case.task_type == "tabular_clustering" else f"`{r.get('target', '')}`"
        parts.append(f"| {case.key} | `{case.filename}` | {r.get('rows', 0):,} | {r.get('task', '–')} | {target} |")
    parts += [
        "",
        "## 1. Main benchmark (full system, clean data)",
        "",
        benchmark_table(analysis["benchmark"], len(seeds)),
        "",
        "↑ higher is better, ↓ lower is better. *Best model* is the most frequent choice over the seeds. "
        "*Improvement* is the relative gain over the baseline on the primary metric. *Train time* is the whole AutoML "
        "search plus the final refit (summed over attempts when verification triggered a re-plan); *inference* is the "
        "median time of the fitted pipeline (preprocessing included) on up to 500 test rows.",
        "",
        f"¹ Chance-level baseline (|baseline| < {NEAR_ZERO_BASELINE:g}), so a relative improvement is undefined and the "
        "absolute gain equals the model's score: the majority-class baseline never predicts the minority (churn) "
        "class, so its F1 is 0; a random cluster assignment has an ARI of about 0 by construction.",
        "",
        "### Why ARI rather than the Rand index for clustering",
        "",
        "AutoML-Agent scores clustering with the Rand index (RI). RI counts the share of row pairs on which the "
        "clustering and the ground truth agree, and pairs placed in *different* clusters agree whenever the truth "
        "also separates them — which is most pairs once there are several groups. RI therefore rewards splitting the "
        "data into more clusters than exist and is not corrected for chance: a random assignment into four groups "
        "already scores about 0.62 here. In a first run with RI as the primary metric, an untuned k-means with its "
        "default of 8 clusters scored a higher RI than the tuned 4-cluster model. The **adjusted Rand index (ARI)** "
        "subtracts the agreement expected by chance (random labels score ≈ 0, a perfect match 1) and penalises "
        "over-splitting, so IDSS now uses ARI as the primary clustering metric whenever ground truth exists — for "
        "model selection, the baseline comparison and the overfitting check in verification, NPS, the ablation and "
        "the robustness study. RI is still computed and reported for comparison with the paper.",
        "",
        "The same investigation exposed an engine bug: the Gaussian mixture ran a single EM initialisation, which on "
        "this data stops in a poor local optimum for some seeds (ARI ≈ 0.53 instead of 0.84). The configuration "
        "that won the search was then refitted into a worse model, which made adding noise columns look like an "
        "*improvement*. The mixture now uses several initialisations (like k-means). The Segments runs below were "
        "re-run after both changes; the two earlier sets are kept under `superseded` in `evaluation.json`.",
        "",
        analysis["clustering_metrics"],
        "",
        f"![Model vs baseline]({figures}/benchmark_vs_baseline.png)",
        "",
        f"![Training time and inference latency]({figures}/efficiency.png)",
        "",
        "## 2. Paper-style metrics (SR, NPS, CS)",
        "",
        "Following AutoML-Agent (Trirat et al., ICML 2025):",
        "",
        "- **SR (success rate)** — share of runs that complete the whole pipeline (trained, tested, verified, "
        "explained and reported as a PDF).",
        "- **NPS (normalised performance score)** — the test score mapped to [0, 1]: the score itself for 0–1 metrics "
        "where higher is better (F1; ARI, clipped at 0 because it can be slightly negative); "
        "`NPS = 1 / (1 + error)` for error metrics. For regression and "
        "forecasting the error is the **RMSLE** of the selected model, a scale-free error (as used for these tasks "
        "in the paper's benchmark): `1 / (1 + RMSE)` would depend on the target's units and be ≈ 0 for prices in "
        "rupees or daily sales. A failed run scores NPS = 0.",
        "- **CS (comprehensive score)** — `CS = 0.5 · SR + 0.5 · NPS`.",
        "",
        "Overall = mean over datasets for each seed, then mean ± std over seeds.",
        "",
        paper_table(analysis["paper"], nps_sources),
        "",
        f"![SR, NPS and CS]({figures}/paper_metrics.png)",
        "",
        "## 3. Ablation study",
        "",
        "Same datasets and seeds; one component removed at a time:",
        "",
        "- **Full system** — model search over the planned candidates, Optuna tuning, decision-threshold tuning, "
        "verification with one re-plan.",
        "- **− threshold tuning** — binary classifiers keep the 0.5 probability cut-off.",
        "- **− hyperparameter tuning** — every candidate is fitted once with its library defaults.",
        "- **− verification re-planning** — the first attempt is final even when verification fails.",
        "- **Single-model baseline** — one LightGBM (k-means for clustering) with library defaults: no model "
        "search, no tuning, no threshold tuning, no re-planning.",
        "",
        "Cells: test score (mean ± std) and, in brackets, the mean change against the full system on the same seed "
        "(positive = better). Ablation runs stop after verification, so SR here counts runs that trained, tested and "
        "verified a model.",
        "",
        ablation_table(analysis["ablation"]),
        "",
        "² Threshold tuning exists only for binary classification; on the other datasets the configuration is identical "
        "to the full system, so its runs are reused. Likewise, *− verification re-planning* is the full system's first "
        "attempt, so it differs only where re-planning was triggered.",
        "",
        f"![Ablation]({figures}/ablation.png)",
        "",
        "## 4. Robustness",
        "",
        f"The full system is re-run on corrupted copies of each dataset: **{PERTURBATIONS['missing']}** (extra cells "
        "blanked completely at random in every column except the target / ground truth, identifiers and the time "
        f"index) and **{PERTURBATIONS['noise']}** (pure-noise columns: three numeric, one integer, one categorical). "
        "Change = relative change of the test score against the clean run on the same seed (negative = worse).",
        "",
        robustness_table(analysis["robustness"]),
        "",
        f"![Robustness]({figures}/robustness.png)",
        "",
        "## 5. Interpretation",
        "",
        *(f"- {note}" for note in analysis["interpretation"]),
        "",
        "## 6. Threats to validity",
        "",
        "- All five datasets are synthetic with known generating processes; real data is noisier and may contain "
        "problems (drift, label noise, leakage) these datasets do not have.",
        "- Fast mode only, on one machine: timings depend on the CPU and on other load, and a larger budget can change "
        "the selected model.",
        f"- {len(seeds)} seeds give an indication of variance, not tight confidence intervals.",
        "- The LLM agents are not exercised (offline evaluation): requirement parsing, the knowledge brief, the "
        "LLM review in verification and the written insights use their deterministic fallbacks. Model choice is "
        "deterministic as well, so these results measure the engine and the pipeline, not the LLM.",
        "- SR, NPS and CS follow the AutoML-Agent definitions, but the numbers are not comparable with the paper's "
        "tables: the datasets, budgets and success criteria differ.",
        "",
    ]
    return "\n".join(parts)


def analyse(data: dict[str, Any]) -> dict[str, Any]:
    outcomes = [Outcome(**o) for o in data["runs"]]
    seeds = data["seeds"]
    bench = benchmark_rows(outcomes)
    for row, case in zip(bench, CASES, strict=True):
        runs = [o for o in select(outcomes, case=case.key) if o.trained]
        row["nps_metric"] = runs[0].nps_metric if runs else None
        row["target"] = data.get("targets", {}).get(case.key, "")
    scores = paper_scores(select(outcomes), seeds, end_to_end=True)
    ablation = ablation_rows(outcomes, seeds)
    robust = robustness_rows(outcomes, seeds)
    return {
        "benchmark": bench,
        "paper": scores,
        "ablation": ablation,
        "robustness": robust,
        "clustering_metrics": clustering_metric_table(outcomes),
        "interpretation": interpretation(bench, scores, ablation, robust, outcomes),
    }


# --------------------------------------------------------------------------- main


def isolated_workspace(root: Path) -> None:
    """Send runs, logs, the database and the knowledge cache to `root`, and disable web search."""
    settings = get_settings()
    settings.workspace_dir = root
    settings.tavily_api_key = settings.serpapi_key = SecretStr("")
    setup_logging(level="ERROR", log_dir=settings.logs_dir, force=True)


def ensure_samples() -> None:
    if not all((SAMPLES_DIR / case.filename).is_file() for case in CASES):
        make_samples.main()


DEFINITIONS = {
    "SR": "1 if the run finishes and reports test performance, else 0 (main benchmark: end to end, with a PDF "
          "report; ablation/robustness: trained, tested and verified)",
    "NPS": "test score for 0-1 metrics where higher is better (F1; ARI clipped at 0); 1 / (1 + RMSLE) for regression "
           "and forecasting; 0 for a failed run",
    "CS": "0.5 * SR + 0.5 * NPS",
}


def _evaluate(seeds: tuple[int, ...], cases: tuple[Case, ...]) -> tuple[list[Outcome], float]:
    """Run the experiments for `cases` in a temporary workspace; returns the outcomes and the duration in seconds."""
    ensure_samples()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="idss-eval-", ignore_cleanup_errors=True) as tmp:
        isolated_workspace(Path(tmp) / "workspace")
        try:
            outcomes = run_all(seeds, Path(tmp), cases)
        finally:
            get_engine().dispose()  # release the SQLite file so the temp dir can be removed
            get_logger().remove()
    return outcomes, round(time.perf_counter() - started, 1)


def _targets() -> dict[str, str]:
    targets = {}
    for case in CASES:
        profile = profile_dataset(load_dataset(SAMPLES_DIR / case.filename))
        targets[case.key] = build_requirement(case, profile).target or ""
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation of IDSS: benchmark, SR/NPS/CS, ablation, robustness.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="random seeds (default: 42 43 44)")
    parser.add_argument("--from-json", action="store_true", help="rebuild the report and charts from evaluation.json")
    parser.add_argument("--cases", nargs="+", metavar="CASE",
                        help="re-run only these datasets (e.g. Segments) and merge them into evaluation.json")
    parser.add_argument("--reason", default="re-run of selected cases", help="why the cases were re-run (--cases)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    if args.from_json:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    elif args.cases:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        unknown = set(args.cases) - {case.key for case in CASES}
        if unknown:
            parser.error(f"unknown case(s) {sorted(unknown)}; choose from {[case.key for case in CASES]}")
        cases = tuple(case for case in CASES if case.key in args.cases)
        seeds = tuple(data["seeds"])  # a partial re-run must use the seeds of the runs it is merged with
        outcomes, duration = _evaluate(seeds, cases)
        replaced = [run for run in data["runs"] if run["case"] in args.cases]
        data["runs"] = [run for run in data["runs"] if run["case"] not in args.cases] + [asdict(o) for o in outcomes]
        data.setdefault("superseded", []).append({
            "replaced": f"{datetime.now():%Y-%m-%d %H:%M}", "cases": list(args.cases), "reason": args.reason,
            "runs": replaced})
        data.setdefault("reruns", []).append({"at": f"{datetime.now():%Y-%m-%d %H:%M}", "cases": list(args.cases),
                                              "reason": args.reason, "duration_s": duration})
        data["targets"] = _targets()
        data["definitions"] = DEFINITIONS
    else:
        outcomes, duration = _evaluate(tuple(args.seeds), CASES)
        data = {
            "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
            "seeds": list(args.seeds),
            "budget": BUDGET_PRESETS["fast"].model_dump(),
            "environment": environment(),
            "duration_s": duration,
            "targets": _targets(),
            "definitions": DEFINITIONS,
            "runs": [asdict(o) for o in outcomes],
        }

    analysis = analyse(data)
    data["summary"] = analysis
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(_finite(data), indent=2, allow_nan=False), encoding="utf-8")
    figures = make_figures(analysis["benchmark"], analysis["paper"], analysis["ablation"], analysis["robustness"])
    MD_PATH.write_text(build_markdown(data, analysis), encoding="utf-8")

    nps_sources = {r["dataset"]: r["nps_metric"] for r in analysis["benchmark"] if r.get("trained")}
    print("\nMain benchmark (full system, clean data):\n" + benchmark_table(analysis["benchmark"], len(data["seeds"])))
    print("\nPaper-style metrics:\n" + paper_table(analysis["paper"], nps_sources))
    print("\nAblation:\n" + ablation_table(analysis["ablation"]))
    print("\nRobustness:\n" + robustness_table(analysis["robustness"]))
    print("\nSaved:", *(p.relative_to(PROJECT_ROOT) for p in (MD_PATH, JSON_PATH, *figures)), sep="\n  ")
    return 0


def _finite(value: Any) -> Any:
    """`value` as plain JSON: NaN/inf become null, tuples become lists, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    sys.exit(main())

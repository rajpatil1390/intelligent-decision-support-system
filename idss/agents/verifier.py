"""Verifier: two-stage check of a trained model before its results are explained to the user.

Stage 1 (always, no LLM) runs deterministic checks and alone decides `passed`. Stage 2 (optional) asks
the LLM for a short verdict, concerns and recommendations; if it fails, it is skipped.
"""

import json

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.prompts import render_prompt
from idss.core.schemas import Check, DatasetProfile, TrainingResult, UserRequirement, VerificationReport
from idss.ml.evaluator import METRIC_FAMILY_BY_TASK, METRICS, metric_direction, normalize_metric_name

logger = get_logger()

MIN_TEST_ROWS = 30
MAX_OVERFIT_GAP = 0.15  # train minus test, in points of a 0-1 scaled metric
# Metrics on a 0..1 scale (a target above 1 for them is read as a percentage, e.g. 85 -> 0.85).
BOUNDED_METRICS = frozenset({"accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "r2",
                             "rand_index", "adjusted_rand_index", "nmi"})
_GAP_METRIC_BY_FAMILY = {"classification": "accuracy", "regression": "r2", "clustering": "silhouette"}

RECOMMENDATIONS = {
    "metric_targets": "Run the full search budget, add more informative features or more data to reach the target metric.",
    "beats_baseline": "The model is no better than a trivial baseline: check the target definition and add features with real signal.",
    "constraints": "Prefer faster or smaller model families (e.g. logistic_regression or ridge) to meet the constraints.",
    "test_size": "Collect more data: the test split is too small for reliable metrics.",
    "test_set_used_once": "Re-run training: the test split must be evaluated exactly once, at the end.",
    "no_leakage": "Remove the suspected leaking column(s) and retrain.",
    "overfitting": "Reduce overfitting: prefer simpler or more regularised models, or gather more training data.",
}


class LLMReview(BaseModel):
    verdict: str
    concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class Verifier:
    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm_client = llm_client

    def verify(
        self, requirement: UserRequirement, training: TrainingResult, profile: DatasetProfile | None = None
    ) -> VerificationReport:
        checks = [
            check_metric_targets(requirement, training),
            check_beats_baseline(training),
            check_constraints(training),
            check_test_size(training),
            check_test_set_used_once(training),
            check_no_leakage(profile),
            check_overfitting(training),
        ]
        passed = all(check.passed for check in checks)
        recommendations = [RECOMMENDATIONS[check.name] for check in checks if not check.passed]
        review = self._llm_review(requirement, training, checks)
        if review is not None:
            recommendations += [item for item in review.recommendations if item not in recommendations]
        logger.info("Verification {}: failed checks = {}.", "passed" if passed else "failed",
                    [c.name for c in checks if not c.passed])
        return VerificationReport(
            passed=passed,
            checks=checks,
            llm_verdict=review.verdict if review else None,
            llm_concerns=review.concerns if review else [],
            recommendations=recommendations,
        )

    def _llm_review(self, requirement: UserRequirement, training: TrainingResult, checks: list[Check]) -> LLMReview | None:
        if self.llm_client is None:
            return None
        metrics = {
            "model": training.best_model,
            "primary_metric": training.primary_metric,
            "test": {k: v for k, v in training.test_metrics.items() if k != "confusion_matrix"},
            "train": training.train_metrics,
            "baseline": training.baseline.model_dump() if training.baseline else None,
            "decision_threshold": training.decision_threshold,
            "leaderboard": [{"model": e.model, "valid_score": e.valid_score, "violations": e.violations}
                            for e in training.leaderboard if e.status == "ok"][:5],
        }
        try:
            messages = render_prompt(
                "verification_review",
                requirement_json=requirement.model_dump_json(exclude={"system_notes"}, indent=2),
                checks_json=json.dumps([c.model_dump() for c in checks], indent=2),
                metrics_json=json.dumps(metrics, indent=2, default=str),
            )
            return self.llm_client.chat_structured(messages, LLMReview)
        except Exception as exc:  # the review is optional
            logger.warning("LLM verification review skipped ({}: {}).", type(exc).__name__, exc)
            return None


# --------------------------------------------------------------------------- rule-based checks


def check_metric_targets(requirement: UserRequirement, training: TrainingResult) -> Check:
    targets = [m for m in requirement.problem.performance_metrics if m.target is not None]
    if not targets:
        return Check(name="metric_targets", passed=True, detail="No target values were stated for the metrics.")
    results, failed = [], False
    for metric in targets:
        name = normalize_metric_name(metric.name)
        if name not in METRICS:
            results.append(f"{metric.name}: not a metric the system can measure (not checked)")
            continue
        value = training.test_metrics.get(name)
        target = metric.target / 100 if metric.target > 1 and name in BOUNDED_METRICS else metric.target
        if value is None:
            failed = True
            results.append(f"{name}: not measured")
            continue
        met = value >= target if metric_direction(name) == "maximize" else value <= target
        failed |= not met
        results.append(f"{name} = {value:.4g} vs target {target:g} ({'met' if met else 'NOT met'})")
    return Check(name="metric_targets", passed=not failed, detail="; ".join(results) + ".")


def check_beats_baseline(training: TrainingResult) -> Check:
    if training.baseline is None:
        return Check(name="beats_baseline", passed=True, detail="No trivial baseline applies to this task.")
    metric = training.primary_metric
    model_value = training.test_metrics.get(metric)
    baseline_value = training.baseline.metrics.get(metric)
    if model_value is None or baseline_value is None:
        return Check(name="beats_baseline", passed=True,
                     detail=f"{metric} is undefined for the {training.baseline.name} baseline; not compared.")
    better = model_value > baseline_value if training.direction == "maximize" else model_value < baseline_value
    return Check(
        name="beats_baseline",
        passed=better,
        detail=f"{training.best_model} {metric} = {model_value:.4g} vs {training.baseline.name} baseline "
               f"{baseline_value:.4g} ({'better' if better else 'NOT better'}).",
    )


def check_constraints(training: TrainingResult) -> Check:
    selected = next((e for e in training.leaderboard if e.selected), None)
    violations = selected.violations if selected else []
    if violations:
        return Check(name="constraints", passed=False, detail="Selected model breaks: " + "; ".join(violations) + ".")
    return Check(name="constraints", passed=True, detail="The selected model meets all user constraints.")


def check_test_size(training: TrainingResult) -> Check:
    rows = int(training.test_metrics.get("n_test_rows", 0))
    return Check(name="test_size", passed=rows >= MIN_TEST_ROWS,
                 detail=f"{rows} test rows (minimum {MIN_TEST_ROWS} for reliable metrics).")


def check_test_set_used_once(training: TrainingResult) -> Check:
    accesses = training.test_set_accesses
    return Check(name="test_set_used_once", passed=accesses == ["final_evaluation"],
                 detail=f"Test split accesses: {accesses or 'none'}.")


def check_no_leakage(profile: DatasetProfile | None) -> Check:
    if profile is None:
        return Check(name="no_leakage", passed=True, detail="No dataset profile available; leakage not checked.")
    leaks = [warning for warning in profile.warnings if "leakage" in warning.lower()]
    if leaks:
        return Check(name="no_leakage", passed=False, detail="Unresolved leakage warnings: " + " ".join(leaks))
    return Check(name="no_leakage", passed=True, detail="No feature is suspiciously correlated with the target.")


def check_overfitting(training: TrainingResult) -> Check:
    """Gap between in-sample (train) and test scores, on a 0-1 scaled metric: the primary metric when it is one
    (classification; clustering against ground truth, e.g. ARI), else accuracy / r2 / silhouette."""
    family = METRIC_FAMILY_BY_TASK[training.task_type]
    bounded_primary = training.primary_metric in BOUNDED_METRICS and family in ("classification", "clustering")
    metric = training.primary_metric if bounded_primary else _GAP_METRIC_BY_FAMILY[family]
    train_value, test_value = training.train_metrics.get(metric), training.test_metrics.get(metric)
    if train_value is None or test_value is None:
        return Check(name="overfitting", passed=True, detail=f"{metric} not available on both splits; not checked.")
    gap = train_value - test_value
    return Check(
        name="overfitting",
        passed=gap < MAX_OVERFIT_GAP,
        detail=f"{metric}: train {train_value:.3f} vs test {test_value:.3f}; gap {gap:.3f} "
               f"(limit {MAX_OVERFIT_GAP:.2f}).",
    )

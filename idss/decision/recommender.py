"""Recommendations: turn model outputs into concrete, evidence-backed actions.

- Classification (binary): risk tiers from the predicted probability (High from max(tuned threshold, 70th
  percentile of the scores), Medium from the threshold), the drivers of each tier, and a greedy counterfactual
  search over ACTIONABLE features (observed values only, at most 3 changes, pruned to a minimal set) that moves
  high-risk records below the decision threshold.
- Regression / time series: the estimated effect of ±10% (or switching category) on the top levers.
- Clustering: one targeted action per distinguishable segment.
Every recommendation carries `evidence`: the computed numbers behind it.
"""

import re
import warnings
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from idss.core.logger import get_logger
from idss.core.schemas import (
    Counterfactual,
    CounterfactualChange,
    DatasetProfile,
    ExplanationResult,
    Recommendation,
    RecommendationResult,
    RiskTier,
    RiskTierCutoffs,
    TrainingResult,
    UserRequirement,
    VerificationReport,
    WhatIfResult,
)
from idss.decision.what_if import WhatIfError, model_output, simulate, simulate_segment
from idss.ml import model_zoo
from idss.ml.explain import ModelExplainer
from idss.ml.profiler import parse_datetimes
from idss.ml.trainer import TrainedModel

logger = get_logger()

# Columns describing who/where/when someone is, rather than something the business can change.
NON_ACTIONABLE_TOKENS = frozenset({
    "age", "gender", "sex", "race", "ethnicity", "ethnic", "religion", "marital", "senior", "citizen", "birth",
    "nationality", "region", "state", "country", "city", "location", "zip", "postcode", "postal",
    "rainfall", "rain", "temperature", "temp", "humidity", "weather",
    "year", "month", "day", "week", "date", "time", "season", "holiday", "tenure",
})
HIGH_TIER_PERCENTILE = 70  # High tier = at most the top 30% of scores (and above the threshold)
MAX_COUNTERFACTUAL_CHANGES = 3
COUNTERFACTUAL_RECORDS = 20
GRID_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
MAX_CATEGORY_OPTIONS = 10
LEVER_PCT = 10
MAX_LEVERS = 3
MAX_POPULATION = 5_000
Confidence = Literal["high", "medium", "low"]


def infer_actionable(model: TrainedModel, profile: DatasetProfile, requirement: UserRequirement | None) -> list[str]:
    """Columns the business can change: the requirement's list if given, else inferred (no IDs, dates,
    text, the target, constants, or demographic / location / weather / time attributes)."""
    inputs = [c for c in model.feature_columns if c != model.target]
    if requirement is not None and requirement.actionable_features:
        lookup = {c.lower(): c for c in inputs}
        chosen = [lookup[name.strip().lower()] for name in requirement.actionable_features if name.strip().lower() in lookup]
        if chosen:
            return list(dict.fromkeys(chosen))
    actionable = []
    for name in inputs:
        column = next((c for c in profile.columns if c.name == name), None)
        if column is None or column.n_unique <= 1 or column.semantic_type not in ("numeric", "categorical", "boolean"):
            continue
        if set(re.findall(r"[a-z]+", name.lower())) & NON_ACTIONABLE_TOKENS:
            continue
        actionable.append(name)
    return actionable


def recommend(
    model: TrainedModel,
    data: pd.DataFrame,
    requirement: UserRequirement | None,
    profile: DatasetProfile,
    training: TrainingResult,
    explanation: ExplanationResult,
    explainer: ModelExplainer | None = None,
    verification: VerificationReport | None = None,
    *,
    seed: int = 42,
) -> RecommendationResult:
    actionable = infer_actionable(model, profile, requirement)
    family = model_zoo.task_family(model.task_type)
    verified = verification is None or verification.passed
    notes = [] if actionable else ["No actionable columns were found; recommendations are limited to prioritisation."]
    if verification is not None and not verification.passed:
        notes.append("The model did not pass every verification check, so confidence is capped at 'low'.")

    if family == "clustering":
        return RecommendationResult(actionable_features=actionable, notes=notes,
                                    recommendations=_cluster_recommendations(explanation, training))
    population = data if len(data) <= MAX_POPULATION or family == "time_series" else data.sample(MAX_POPULATION, random_state=seed)
    if family == "classification":
        if model.classes is None or len(model.classes) != 2 or model.pos_label is None:
            notes.append("Risk tiers and counterfactuals are only computed for two-class problems.")
            return RecommendationResult(actionable_features=actionable, notes=notes)
        return _classification(model, population, actionable, profile, training, explanation, explainer, verified, notes)
    return _levers(model, population, actionable, profile, training, explanation, verified, notes)


# --------------------------------------------------------------------------- classification


def _classification(
    model: TrainedModel, population: pd.DataFrame, actionable: list[str], profile: DatasetProfile,
    training: TrainingResult, explanation: ExplanationResult, explainer: ModelExplainer | None, verified: bool,
    notes: list[str],
) -> RecommendationResult:
    threshold = training.decision_threshold or 0.5
    probability = model_output(model, population)
    cutoffs = tier_cutoffs(probability, threshold)
    tiers = risk_tiers(probability, cutoffs, explainer, explanation)
    in_high = probability >= cutoffs.high
    high = population[in_high]
    target, positive = model.target, model.classes[model.pos_label]
    recommendations: list[Recommendation] = []

    precision = training.test_metrics.get("precision")
    expected = float(probability[in_high].sum())
    if len(high):
        why = (f"the top {100 - cutoffs.percentile}% of scores" if cutoffs.high > threshold
               else "the tuned decision threshold")
        rationale = f"They have a predicted probability of {target} = '{positive}' of at least {cutoffs.high:.2f} ({why})"
        if precision is not None:
            rationale += (f"; on the test split, {precision:.0%} of records flagged at the decision threshold of "
                          f"{threshold:.2f} were truly '{positive}', and this tier is the most confident part of that group.")
        recommendations.append(Recommendation(
            action=f"Prioritise the {len(high):,} high-risk records ({len(high) / len(population):.1%}) for proactive action.",
            rationale=rationale + ("" if precision is not None else "."),
            expected_impact=round(expected, 2),
            impact_unit=f"expected '{positive}' cases in the high-risk tier",
            confidence=_confidence(verified, precision or 0.0),
            evidence={"high_tier_count": len(high), "high_tier_share": round(len(high) / len(population), 4),
                      "expected_positive_cases": round(expected, 2), "high_cutoff": cutoffs.high,
                      "threshold": round(threshold, 4), "test_precision": precision},
        ))

    counterfactual_example = None
    if actionable and len(high):
        grids = candidate_values(population, actionable, profile)
        worst = high.assign(_p=probability[in_high]).sort_values("_p", ascending=False).head(COUNTERFACTUAL_RECORDS)
        results = [counterfactual(model, worst.loc[[index]], actionable, grids, threshold) for index in worst.index]
        counterfactual_example = results[0]
        successes = [r for r in results if r.success]
        occurrences = Counter((c.feature, _key(c.after)) for r in successes for c in r.changes)
        for (feature, value), count in occurrences.most_common(2):
            recommendations.append(_change_recommendation(
                model, high, feature, value, count, len(results), len(successes), profile, verified))
        if not successes:
            notes.append(f"No combination of up to {MAX_COUNTERFACTUAL_CHANGES} actionable changes moved the "
                         f"{len(results)} highest-risk records below the threshold.")
    return RecommendationResult(actionable_features=actionable, recommendations=recommendations, risk_tiers=tiers,
                                tier_cutoffs=cutoffs, counterfactual_example=counterfactual_example, notes=notes)


def tier_cutoffs(probability: np.ndarray, threshold: float) -> RiskTierCutoffs:
    """High starts at the larger of the decision threshold and the 70th percentile of the scores, so the High tier
    stays a focused group even when the tuned threshold is low; Medium is the rest of the records above the threshold."""
    percentile_value = float(np.percentile(probability, HIGH_TIER_PERCENTILE)) if len(probability) else threshold
    return RiskTierCutoffs(high=round(max(threshold, percentile_value), 4), medium=round(threshold, 4),
                           percentile=HIGH_TIER_PERCENTILE, percentile_value=round(percentile_value, 4))


def risk_tiers(probability: np.ndarray, cutoffs: RiskTierCutoffs, explainer: ModelExplainer | None,
               explanation: ExplanationResult) -> list[RiskTier]:
    """High: p >= cutoffs.high; Medium: cutoffs.medium <= p < cutoffs.high; Low: p < cutoffs.medium."""
    bounds = {"High": (cutoffs.high, 1.0), "Medium": (cutoffs.medium, cutoffs.high), "Low": (0.0, cutoffs.medium)}
    drivers = _tier_drivers(explainer, bounds) if explainer is not None else {}
    fallback = [f.feature for f in explanation.global_explanation.features[:3]]
    tiers = []
    for name, (low, high) in bounds.items():
        mask = (probability >= low) & ((probability < high) if name != "High" else True)
        tiers.append(RiskTier(
            name=name, min_probability=round(low, 4), max_probability=round(high, 4), count=int(mask.sum()),
            share=round(float(mask.mean()), 4), avg_probability=round(float(probability[mask].mean()), 4) if mask.any() else None,
            top_drivers=drivers.get(name, fallback),
        ))
    return tiers


def _tier_drivers(explainer: ModelExplainer, bounds: dict[str, tuple[float, float]]) -> dict[str, list[str]]:
    """Columns pushing risk up most in High/Medium, and keeping it down most in Low (mean signed SHAP)."""
    if explainer.method not in ("shap_tree", "shap_linear") or not explainer.signed:
        return {}
    try:
        contributions = explainer.contributions()
    except Exception as exc:
        logger.warning("Tier drivers unavailable ({}: {}).", type(exc).__name__, exc)
        return {}
    probability = explainer.predict_output(explainer.raw)
    drivers = {}
    for name, (low, high) in bounds.items():
        mask = (probability >= low) & ((probability < high) if name != "High" else True)
        if not mask.any():
            continue
        mean = contributions[mask].mean()
        ordered = mean.sort_values(ascending=name == "Low")
        drivers[name] = [str(feature) for feature, value in ordered.head(3).items() if (value > 0) != (name == "Low")]
    return drivers


def candidate_values(population: pd.DataFrame, actionable: list[str], profile: DatasetProfile) -> dict[str, list[Any]]:
    """Values each actionable column may take: observed quantiles (numeric) or observed categories."""
    grids: dict[str, list[Any]] = {}
    for feature in actionable:
        column = profile.column(feature)
        values = population[feature].dropna()
        if values.empty:
            continue
        if column.semantic_type == "numeric":
            grid = values.quantile(list(GRID_QUANTILES)).to_numpy(dtype="float64")
            if column.stats.get("integer"):
                grid = np.round(grid)
            grids[feature] = sorted({_py(v) for v in grid})
        else:
            grids[feature] = [_py(v) for v in values.value_counts().head(MAX_CATEGORY_OPTIONS).index]
    return grids


def counterfactual(model: TrainedModel, row: pd.DataFrame, actionable: list[str], grids: dict[str, list[Any]],
                   threshold: float) -> Counterfactual:
    """Greedy search for at most 3 changes to actionable columns that bring the probability below `threshold`,
    then drop any change that is not needed (minimal set)."""
    current = row[model.feature_columns].copy()
    start = float(model_output(model, current)[0])
    probability = start
    changed: list[str] = []
    for _ in range(MAX_COUNTERFACTUAL_CHANGES):
        if probability < threshold:
            break
        options = [(f, v) for f in actionable if f not in changed for v in grids.get(f, []) if not _same(current[f].iloc[0], v)]
        if not options:
            break
        batch = pd.concat([current] * len(options), ignore_index=True)
        for i, (feature, value) in enumerate(options):
            batch.loc[i, feature] = value
        outputs = model_output(model, batch)
        best = int(np.argmin(outputs))
        if outputs[best] >= probability - 1e-9:
            break  # nothing helps any more
        feature, value = options[best]
        current.loc[current.index[0], feature] = value
        changed.append(feature)
        probability = float(outputs[best])

    if probability < threshold:  # prune: undo changes that are not needed to stay below the threshold
        for feature in list(reversed(changed)):
            trial = current.copy()
            trial.loc[trial.index[0], feature] = row[feature].iloc[0]
            trial_probability = float(model_output(model, trial)[0])
            if trial_probability < threshold:
                current, probability = trial, trial_probability
                changed.remove(feature)
    return Counterfactual(
        row_index=_py(row.index[0]), probability_before=round(start, 4), probability_after=round(probability, 4),
        threshold=round(threshold, 4), success=probability < threshold,
        changes=[CounterfactualChange(feature=f, before=_py(row[f].iloc[0]), after=_py(current[f].iloc[0])) for f in changed],
    )


def _change_recommendation(model: TrainedModel, high: pd.DataFrame, feature: str, value: Any, occurrences: int,
                           records: int, successes: int, profile: DatasetProfile, verified: bool) -> Recommendation:
    impact = simulate_segment(model, high, [], {feature: value}, profile)
    if profile.column(feature).semantic_type == "numeric":
        typical = float(high[feature].mean())
        verb = "Lower" if float(value) < typical else "Raise"
        action = f"{verb} {feature} towards {value:g} for high-risk records (currently about {typical:.3g} on average)."
    else:
        action = f"Move high-risk records to {feature} = '{value}'."
    success_rate = successes / records if records else 0.0
    return Recommendation(
        action=action,
        rationale=(f"This change is part of the minimal change set for {occurrences} of {records} high-risk records "
                   f"tested; applied to all {impact.count:,} high-risk records it moves the average "
                   f"{impact.output} from {impact.avg_before:.3f} to {impact.avg_after:.3f}."),
        expected_impact=round(impact.delta or 0.0, 4),
        impact_unit=f"change in average {impact.output} (high-risk tier)",
        confidence=_confidence(verified, success_rate),
        evidence={"high_tier_count": impact.count, "avg_probability_before": impact.avg_before,
                  "avg_probability_after": impact.avg_after, "counterfactual_records": records,
                  "counterfactual_successes": successes, "change_occurrences": occurrences,
                  "counterfactual_success_rate": round(success_rate, 4)},
    )


# --------------------------------------------------------------------------- regression / time series


def _levers(model: TrainedModel, population: pd.DataFrame, actionable: list[str], profile: DatasetProfile,
            training: TrainingResult, explanation: ExplanationResult, verified: bool, notes: list[str]) -> RecommendationResult:
    ranked = [f.feature for f in explanation.global_explanation.features if f.feature in actionable][:MAX_LEVERS]
    r2 = training.test_metrics.get("r2") or 0.0
    notes.append(f"Lever directions assume a higher {model.target} is better; reverse them if lower is better.")
    recommendations = []
    for feature in ranked:
        try:
            recommendations.append(lever_recommendation(model, population, feature, profile, verified, r2))
        except WhatIfError as exc:
            notes.append(f"Lever '{feature}' skipped: {exc}")
    recommendations.sort(key=lambda r: abs(r.expected_impact), reverse=True)
    return RecommendationResult(actionable_features=actionable, recommendations=recommendations, notes=notes)


def lever_recommendation(model: TrainedModel, population: pd.DataFrame, feature: str, profile: DatasetProfile,
                         verified: bool, r2: float) -> Recommendation:
    column = profile.column(feature)
    if column.semantic_type == "numeric":
        if (column.stats.get("min") or 0) >= 0:
            step_text = f"{LEVER_PCT}%"
            increase, decrease = {"pct": LEVER_PCT}, {"pct": -LEVER_PCT}
        else:
            # A percentage change is meaningless for a column with negative values (scaling -2 by +10% lowers it),
            # so step by 10% of its standard deviation instead.
            step = round(LEVER_PCT / 100 * float(column.stats.get("std") or 1.0), 6)
            step_text = f"{step:.3g} ({LEVER_PCT}% of its standard deviation)"
            increase, decrease = {"add": step}, {"add": -step}
        up = simulate_segment(model, population, [], {feature: increase}, profile)
        down = simulate_segment(model, population, [], {feature: decrease}, profile)
        best, sign = (up, "Increase") if (up.delta or 0) >= (down.delta or 0) else (down, "Decrease")
        action = f"{sign} {feature} by {step_text}."
        rationale = (f"Across {up.count:,} records, increasing {feature} by {step_text} changes the average "
                     f"{up.output} by {up.delta:+.4g} ({_pct(up.delta_pct)}); decreasing it changes it by "
                     f"{down.delta:+.4g} ({_pct(down.delta_pct)}).")
        evidence = {"records": up.count, "avg_before": up.avg_before, "change_step": step_text,
                    "delta_increase": up.delta, "delta_decrease": down.delta,
                    "delta_pct_increase": up.delta_pct, "delta_pct_decrease": down.delta_pct}
    else:
        current = population[feature].mode().iloc[0]
        options = [v for v in population[feature].dropna().value_counts().head(MAX_CATEGORY_OPTIONS).index]
        impacts = {v: simulate_segment(model, population, [], {feature: _py(v)}, profile) for v in options}
        value, best = max(impacts.items(), key=lambda item: item[1].delta or 0)
        action = f"Set {feature} to '{_py(value)}' where possible."
        rationale = (f"Setting {feature} = '{_py(value)}' for all {best.count:,} records changes the average "
                     f"{best.output} from {best.avg_before:.4g} to {best.avg_after:.4g} (most common today: '{_py(current)}').")
        evidence = {"records": best.count, "avg_before": best.avg_before, "avg_after": best.avg_after,
                    "delta": best.delta, "delta_pct": best.delta_pct}
    return Recommendation(action=action, rationale=rationale, expected_impact=round(best.delta or 0.0, 4),
                          impact_unit=f"change in average {best.output}", confidence=_confidence(verified, r2),
                          evidence=evidence)


# --------------------------------------------------------------------------- clustering


def _cluster_recommendations(explanation: ExplanationResult, training: TrainingResult) -> list[Recommendation]:
    silhouette = training.test_metrics.get("silhouette") or 0.0
    confidence: Confidence = "medium" if silhouette >= 0.25 else "low"
    recommendations = []
    for cluster in sorted(explanation.clusters, key=lambda c: c.size, reverse=True)[:3]:
        traits = "; ".join(cluster.distinguishing) or "no single column sets it apart"
        recommendations.append(Recommendation(
            action=f"Design a tailored approach for segment {cluster.cluster} ({cluster.size:,} records, {cluster.share:.0%}).",
            rationale=f"What sets this segment apart: {traits}.",
            expected_impact=float(cluster.size), impact_unit="records in the segment", confidence=confidence,
            evidence={"segment_size": cluster.size, "segment_share": cluster.share, "test_silhouette": silhouette},
        ))
    return recommendations


# --------------------------------------------------------------------------- example what-if


def example_what_if(model: TrainedModel, data: pd.DataFrame, profile: DatasetProfile, result: RecommendationResult,
                    explanation: ExplanationResult) -> WhatIfResult | None:
    """One illustrative what-if on the top actionable driver, for the report."""
    family = model_zoo.task_family(model.task_type)
    if family == "clustering" or not result.actionable_features:
        return None
    cf = result.counterfactual_example
    if cf is not None and cf.changes:
        row = data.loc[cf.row_index] if cf.row_index in data.index else data.iloc[0]
        changes = {cf.changes[0].feature: cf.changes[0].after}
        return simulate(model, row.to_dict(), changes, profile)
    driver = next((f.feature for f in explanation.global_explanation.features if f.feature in result.actionable_features), None)
    if driver is None:
        return None
    column = profile.column(driver)
    if column.semantic_type == "numeric":
        changes: dict[str, Any] = {driver: {"pct": -LEVER_PCT if family == "classification" else LEVER_PCT}}
    else:
        options = data[driver].dropna().value_counts().index
        changes = {driver: _py(options[1] if len(options) > 1 else options[0])}
    try:
        if family == "time_series":
            ts = model.pipeline.named_steps["ts_features"]
            ordered = data.sort_values(ts.time_column, key=parse_datetimes, kind="stable")
            return simulate(model, ordered.iloc[-1].to_dict(), changes, profile, history=ordered.iloc[:-1])
        if family == "classification":
            row = data.iloc[int(np.argmax(model_output(model, data)))]
        else:
            row = data.iloc[len(data) // 2]
        return simulate(model, row.to_dict(), changes, profile)
    except WhatIfError as exc:
        logger.warning("Example what-if skipped: {}", exc)
        return None


# --------------------------------------------------------------------------- helpers


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def _confidence(verified: bool, strength: float) -> Confidence:
    if not verified:
        return "low"
    return "high" if strength >= 0.6 else "medium" if strength >= 0.3 else "low"


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b) or (isinstance(a, float) and isinstance(b, float) and np.isclose(a, b))
    except (TypeError, ValueError):
        return False


def _key(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _py(value: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return value.item() if isinstance(value, np.generic) else value

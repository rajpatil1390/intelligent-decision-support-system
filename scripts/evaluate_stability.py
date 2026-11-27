"""Explanation stability: how much do the drivers and what-if answers move between runs (seeds)?

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\evaluate_stability.py              # 5 datasets x 3 seeds, offline (~5 min)
    .venv\\Scripts\\python.exe scripts\\evaluate_stability.py --from-json  # rebuild stability.md + figure only
Writes docs/results/stability.json, docs/results/stability.md and docs/results/figures/stability.png.
The Step 8 evaluation (evaluation.json / evaluation.md and its figures) is not touched.

For every sample dataset and seed (42, 43, 44) the real Orchestrator runs offline exactly as in
scripts/evaluate.py (fast budget, requirement from the data profile, no LLM, web search off, temporary
workspace) until the EXPLAIN stage has finished. The seed changes the train/valid/test split, the Optuna
search and the models' random state: everything that differs between two runs of the same analysis.

Driver stability. Each seed's global drivers (SHAP values, or permutation importance, summed back to the
original columns; cluster-separation scores for clustering) are ranked. For every pair of seeds:
  - Spearman rank correlation over the union of both seeds' top-10 drivers, each feature ranked by its
    position in that seed's full driver list (a feature missing from a list shares the last rank);
  - top-5 overlap = Jaccard index |A n B| / |A u B| of the two top-5 sets.
Both are reported as mean +- sample standard deviation over the three seed pairs.

What-if stability (classification datasets). One fixed scenario per dataset is applied to the same base row
under each seed's model: the probability of the at-risk class (binary: the positive class; multiclass: the
least frequent class) before and after the change, and the change itself, as mean +- std over the seeds.
The base row is the typical record of the at-risk class (median of numeric columns, most common value of
the others, latest date), so it depends on the data only, never on one seed's model.
"""

import argparse
import itertools
import json
import math
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402  (needs PROJECT_ROOT on sys.path for the imports below)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from idss.agents.manager import BLOCKED_STATES, MAX_STEPS, Orchestrator  # noqa: E402
from idss.core.logger import get_logger  # noqa: E402
from idss.core.schemas import BUDGET_PRESETS, DatasetProfile  # noqa: E402
from idss.decision.what_if import WhatIfError, apply_changes, simulate  # noqa: E402
from idss.ml.data_loader import load_dataset  # noqa: E402
from idss.ml.profiler import profile_dataset  # noqa: E402
from idss.ml.trainer import AutoMLTrainer, TrainedModel  # noqa: E402
from idss.storage.db import get_engine  # noqa: E402
from idss.storage.runs import run_dir  # noqa: E402
from scripts import evaluate as ev  # noqa: E402

RESULTS_DIR = ev.RESULTS_DIR
JSON_PATH = RESULTS_DIR / "stability.json"
MD_PATH = RESULTS_DIR / "stability.md"
FIGURE_NAME = "stability.png"
TOP_K = 10  # drivers saved and compared by rank
TOP_OVERLAP = 5  # drivers compared as a set (Jaccard)
STABLE_RHO, STABLE_JACCARD = 0.8, 0.8  # "stable" when both means reach these values
MODERATE_RHO, MODERATE_JACCARD = 0.5, 0.6  # below both: "unstable"
SMALL_FEATURE_SET = TOP_OVERLAP + 1  # with this few features the top-5 sets overlap almost by construction
CLOSE_SHARES = 0.06  # top-5 importance shares within 6 percentage points: rank swaps are expected

# Facts about how the synthetic samples are generated (scripts/make_samples.py), added to the interpretation
# only when the features they explain are the ones that move in or out of the top 5.
CONTEXT: dict[str, list[dict[str, Any]]] = {
    "Delivery": [{
        "requires": {"peak_season", "order_date"},
        "text": "In the synthetic data peak_season is defined by the order month (October to December), so a model "
                "can credit the same seasonal signal to either column.",
    }],
}


@dataclass(frozen=True)
class Scenario:
    """A fixed what-if: the same changes on the same base row for every seed's model."""

    changes: dict[str, Any]
    label: str


SCENARIOS = {
    "Churn": Scenario({"contract_type": "Two year"}, "contract_type → 'Two year'"),
    "Delivery": Scenario({"shipping_mode": "Express"}, "shipping_mode → 'Express'"),
}


# --------------------------------------------------------------------------- one run


def run_seed(case: ev.Case, seed: int, data_path: Path, df: pd.DataFrame, profile: DatasetProfile) -> dict[str, Any]:
    """Run the system offline until EXPLAIN has finished; return the drivers and (classification) the what-if."""
    requirement = ev.build_requirement(case, profile)
    orchestrator = Orchestrator(None, budget_mode="fast")
    orchestrator.trainer = AutoMLTrainer(seed=seed)
    record: dict[str, Any] = {"case": case.key, "seed": seed, "task_type": None, "n_classes": None, "best_model": None,
                              "method": None, "n_features": 0, "ranking": [], "top10": [], "what_if": None, "final_state": "",
                              "error": None, "wall_time_s": 0.0}
    start = time.perf_counter()
    run_id = orchestrator.start(case.question, data_path, requirement=requirement)
    ctx = orchestrator.context
    try:
        for _ in range(MAX_STEPS):
            if ctx.state in BLOCKED_STATES or ctx.explanation is not None:
                break
            orchestrator.step(run_id)
        record["final_state"] = str(ctx.state)
        if ctx.errors:
            record["error"] = ctx.errors[-1]
        if ctx.explanation is not None and ctx.training is not None:
            features = ctx.explanation.global_explanation.features
            labels = ctx.training.test_metrics.get("class_labels")
            record.update(
                task_type=ctx.training.task_type,
                n_classes=len(labels) if isinstance(labels, list) else None,
                best_model=ctx.training.best_model,
                method=ctx.explanation.global_explanation.method,
                n_features=len(features),
                ranking=[f.feature for f in features],
                top10=[{"rank": rank, "feature": f.feature, "share": f.share, "importance": f.importance,
                        "direction": f.direction} for rank, f in enumerate(features[:TOP_K], start=1)],
            )
            scenario = SCENARIOS.get(case.key)
            if scenario is not None:
                model: TrainedModel = joblib.load(ctx.training.model_path)  # our own artifact from TRAIN
                record["what_if"] = replay_what_if(model, df, profile, scenario)
    finally:
        shutil.rmtree(run_dir(run_id, orchestrator.run_dir_root, create=False), ignore_errors=True)
    record["wall_time_s"] = round(time.perf_counter() - start, 2)
    return record


def at_risk_class(model: TrainedModel, df: pd.DataFrame) -> Any:
    """Binary: the model's positive (minority) class; multiclass: the least frequent class in the data."""
    if model.pos_label is not None:
        return model.classes[model.pos_label]
    return df[model.target].value_counts().idxmin()


def typical_row(df: pd.DataFrame, profile: DatasetProfile, target: str, label: Any) -> dict[str, Any]:
    """The typical record of the rows whose actual target is `label` (the Decision Center's typical-row rule)."""
    rows = df[df[target] == label]
    row: dict[str, Any] = {}
    for column in profile.columns:
        if column.name == target or column.semantic_type == "id":  # identifiers are not model inputs
            continue
        values = rows[column.name].dropna()
        if values.empty:
            values = df[column.name].dropna()
        if values.empty:
            continue
        if column.semantic_type == "numeric" and pd.api.types.is_numeric_dtype(values):
            median = float(values.median())
            row[column.name] = int(round(median)) if pd.api.types.is_integer_dtype(values) else round(median, 4)
        elif column.semantic_type == "datetime":
            row[column.name] = values.iloc[int(np.argmax(pd.to_datetime(values, errors="coerce").to_numpy()))]
        else:
            row[column.name] = values.mode().iloc[0]
    return {name: ev._finite(value) for name, value in row.items()}


def replay_what_if(model: TrainedModel, df: pd.DataFrame, profile: DatasetProfile, scenario: Scenario) -> dict[str, Any]:
    label = at_risk_class(model, df)
    row = typical_row(df, profile, model.target, label)
    try:
        notes = simulate(model, row, scenario.changes, profile).warnings  # the product's validation and warnings
    except WhatIfError as exc:
        return {"error": str(exc)}
    base = pd.DataFrame([row]).reindex(columns=model.feature_columns)
    changed, _ = apply_changes(base, scenario.changes, model, profile)
    index = list(model.classes).index(label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = float(model.predict_proba(base)[0, index])
        after = float(model.predict_proba(changed)[0, index])
    return {"class": str(label), "output": f"probability that {model.target} = '{label}'", "base_row": row,
            "changes": scenario.changes, "change_label": scenario.label, "before": before, "after": after,
            "delta": after - before, "warnings": notes}


# --------------------------------------------------------------------------- stability measures


def spearman_top(a: list[str], b: list[str], k: int = TOP_K) -> float | None:
    """Spearman correlation of two driver rankings over the union of their top-k features."""
    union = list(dict.fromkeys(a[:k] + b[:k]))
    if len(union) < 3:
        return None
    ranks_a = [a.index(f) + 1 if f in a else len(a) + 1 for f in union]
    ranks_b = [b.index(f) + 1 if f in b else len(b) + 1 for f in union]
    if len(set(ranks_a)) < 2 or len(set(ranks_b)) < 2:
        return None
    rho = float(spearmanr(ranks_a, ranks_b).statistic)
    return rho if math.isfinite(rho) else None


def jaccard_top(a: list[str], b: list[str], k: int = TOP_OVERLAP) -> float | None:
    first, second = set(a[:k]), set(b[:k])
    union = first | second
    return len(first & second) / len(union) if union else None


def analyse(data: dict[str, Any]) -> dict[str, Any]:
    runs = data["runs"]
    drivers, what_ifs = [], []
    for case in ev.CASES:
        case_runs = sorted((r for r in runs if r["case"] == case.key and r["ranking"]), key=lambda r: r["seed"])
        pairs = []
        for first, second in itertools.combinations(case_runs, 2):
            pairs.append({"seeds": [first["seed"], second["seed"]],
                          "spearman": spearman_top(first["ranking"], second["ranking"]),
                          "top5_jaccard": jaccard_top(first["ranking"], second["ranking"])})
        rho = ev.mean_std([p["spearman"] for p in pairs])
        jac = ev.mean_std([p["top5_jaccard"] for p in pairs])
        top1 = [r["ranking"][0] for r in case_runs]
        n_features = max((r["n_features"] for r in case_runs), default=0)
        top5_sets = {r["seed"]: set(r["ranking"][:TOP_OVERLAP]) for r in case_runs}
        common = set.intersection(*top5_sets.values()) if top5_sets else set()
        varying = {f: [s for s, top in top5_sets.items() if f in top]
                   for f in sorted(set().union(*top5_sets.values()) - common)} if top5_sets else {}
        shares = [d["share"] for r in case_runs for d in r["top10"][:TOP_OVERLAP]]
        drivers.append({
            "dataset": case.key,
            "task": _task(case_runs[0]["task_type"], case_runs[0].get("n_classes")) if case_runs else "–",
            "runs": len(case_runs),
            "n_features": n_features,
            "best_models": [r["best_model"] for r in case_runs],
            "methods": [r["method"] for r in case_runs],
            "pairs": pairs,
            "spearman": rho,
            "top5_jaccard": jac,
            "top1": top1,
            "top1_agree": len(set(top1)) == 1 and len(top1) == len(data["seeds"]),
            "top5_common": sorted(common),
            "top5_varying": varying,
            "top5_share_range": [min(shares), max(shares)] if shares else None,
            "verdict": _verdict(rho[0], jac[0]),
            "small_feature_set": 0 < n_features <= SMALL_FEATURE_SET,
        })
        replays = [r for r in case_runs if r.get("what_if") and "before" in r["what_if"]]
        if case.key in SCENARIOS and replays:
            first = replays[0]["what_if"]
            deltas = [r["what_if"]["delta"] for r in replays]
            what_ifs.append({
                "dataset": case.key,
                "class": first["class"],
                "output": first["output"],
                "base_row": first["base_row"],
                "change_label": first["change_label"],
                "per_seed": [{"seed": r["seed"], "best_model": r["best_model"], "before": r["what_if"]["before"],
                              "after": r["what_if"]["after"], "delta": r["what_if"]["delta"]} for r in replays],
                "before": ev.mean_std([r["what_if"]["before"] for r in replays]),
                "after": ev.mean_std([r["what_if"]["after"] for r in replays]),
                "delta": ev.mean_std(deltas),
                "same_direction": len({math.copysign(1, d) for d in deltas if d != 0}) <= 1,
                "warnings": sorted({w for r in replays for w in r["what_if"].get("warnings", [])}),
            })
    return {"drivers": drivers, "what_if": what_ifs, "interpretation": interpretation(drivers, what_ifs)}


def _task(task_type: str | None, n_classes: int | None) -> str:
    if task_type in ev.TASK_LABELS:
        return ev.TASK_LABELS[task_type]
    if n_classes is None:
        return "classification"
    return "binary classification" if n_classes == 2 else f"{n_classes}-class classification"


def _verdict(rho: float, jaccard: float) -> str:
    if not (math.isfinite(rho) and math.isfinite(jaccard)):
        return "not measured"
    if rho >= STABLE_RHO and jaccard >= STABLE_JACCARD:
        return "stable"
    if rho >= MODERATE_RHO or jaccard >= MODERATE_JACCARD:
        return "moderately stable"
    return "unstable"


def interpretation(drivers: list[dict[str, Any]], what_ifs: list[dict[str, Any]]) -> list[str]:
    """Findings derived from the numbers, so the text always matches the tables."""
    notes: list[str] = []
    measured = [d for d in drivers if d["verdict"] != "not measured"]
    stable = [d["dataset"] for d in measured if d["verdict"] == "stable"]
    weaker = [d for d in measured if d["verdict"] != "stable"]
    agree = [d["dataset"] for d in measured if d["top1_agree"]]
    notes.append(
        f"**Overall.** Driver rankings are stable (mean Spearman ρ ≥ {STABLE_RHO} and top-5 Jaccard ≥ "
        f"{STABLE_JACCARD}) on {len(stable)} of {len(measured)} datasets"
        + (f" ({', '.join(stable)})" if stable else "")
        + f". The top driver is the same under all seeds on {len(agree)} of {len(measured)} datasets"
        + (f" ({', '.join(agree)})." if agree else ".")
    )
    for d in weaker:
        models = sorted(set(d["best_models"]))
        top1 = sorted(set(d["top1"]))
        text = (f"**{d['dataset']}** is {d['verdict']}: ρ = {_pm(d['spearman'])}, top-5 Jaccard = "
                f"{_pm(d['top5_jaccard'])}. The top driver is "
                + (f"{top1[0]} under every seed." if len(top1) == 1 else f"{', '.join(d['top1'])} (per seed)."))
        if len(models) > 1:
            per_seed = ", ".join(f"{m} ({x})" for m, x in zip(d["best_models"], d["methods"], strict=True))
            text += (f" The selected model differs between seeds ({per_seed}), so the explanations describe different "
                     "models; this is the main source of the change.")
        else:
            text += f" The same model ({models[0]}) was selected under every seed."
        if d["top5_varying"]:
            text += " Not in every top 5: " + "; ".join(
                f"{feature} (seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))})"
                for feature, seeds in d["top5_varying"].items()) + "."
        low, high = d["top5_share_range"] or (0.0, 1.0)
        if high - low < CLOSE_SHARES:
            text += (f" The top-5 drivers have nearly equal importance shares ({low:.1%}–{high:.1%}), so small "
                     "differences between the fitted models reorder them while the set stays the same.")
        for context in CONTEXT.get(d["dataset"], []):
            if context["requires"] <= set(d["top5_varying"]):
                text += " " + context["text"]
        notes.append(text)
    small = [d for d in measured if d["small_feature_set"]]
    if small:
        notes.append(
            "**Small feature sets.** " + ", ".join(f"{d['dataset']} ({d['n_features']} drivers)" for d in small)
            + f" have at most {SMALL_FEATURE_SET} drivers, so their top-5 sets overlap almost by construction; the "
            "Spearman correlation is the more informative number there."
        )
    for w in what_ifs:
        before, after, delta = w["before"], w["after"], w["delta"]
        spread = max(p["before"] for p in w["per_seed"]) - min(p["before"] for p in w["per_seed"])
        direction = ("all seeds agree on the direction of the effect" if w["same_direction"]
                     else "the seeds DISAGREE on the direction of the effect")
        text = (f"**What-if, {w['dataset']}** ({w['change_label']} on the typical '{w['class']}' record): the "
                f"probability moves from {_pm(before)} to {_pm(after)}, a change of {_pm(delta, signed=True)}; "
                f"{direction}. The starting probability varies by {spread:.3f} between seeds, so a single run's exact "
                "probability should be read with that margin.")
        counts = Counter(p["best_model"] for p in w["per_seed"])
        if len(counts) > 1:
            rare = min(counts, key=counts.get)
            odd = [p for p in w["per_seed"] if p["best_model"] == rare]
            others = [p for p in w["per_seed"] if p["best_model"] != rare]
            moves = lambda ps: ", ".join(f"{p['before']:.3f} → {p['after']:.3f}" for p in ps)  # noqa: E731
            text += (f" Under seed{'s' if len(odd) > 1 else ''} {', '.join(str(p['seed']) for p in odd)} a different "
                     f"model was selected ({rare}): {moves(odd)}, against {moves(others)} under the other seeds.")
        notes.append(text)
    return notes


# --------------------------------------------------------------------------- formatting


def _pm(pair: tuple[float, float] | list[float], digits: int = 3, *, signed: bool = False) -> str:
    mean, std = pair
    if mean is None or not math.isfinite(mean):
        return "–"
    sign = "+" if signed else ""
    return f"{mean:{sign}.{digits}f} ± {std:.{digits}f}"


def driver_table(drivers: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Task | Drivers | Best model per seed | Spearman ρ (top-10) | Top-5 Jaccard | "
             "Top driver per seed | Verdict |",
             "|---|---|---:|---|---:|---:|---|---|"]
    for d in drivers:
        lines.append(f"| {d['dataset']} | {d['task']} | {d['n_features']} | {', '.join(d['best_models'])} | "
                     f"{_pm(d['spearman'])} | {_pm(d['top5_jaccard'])} | {', '.join(d['top1'])} | {d['verdict']} |")
    return "\n".join(lines)


def what_if_table(what_ifs: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Output | Change | Before | After | Change in probability | Same direction |",
             "|---|---|---|---:|---:|---:|---:|"]
    for w in what_ifs:
        lines.append(f"| {w['dataset']} | P({w['class']}) | {w['change_label']} | {_pm(w['before'])} | "
                     f"{_pm(w['after'])} | {_pm(w['delta'], signed=True)} | "
                     f"{'yes' if w['same_direction'] else 'no'} |")
    return "\n".join(lines)


def per_seed_drivers(data: dict[str, Any]) -> str:
    lines = ["| Dataset | Seed | Model | Method | Top-5 drivers (share of total importance) |", "|---|---:|---|---|---|"]
    for case in ev.CASES:
        for r in sorted((r for r in data["runs"] if r["case"] == case.key), key=lambda r: r["seed"]):
            top = ", ".join(f"{d['feature']} ({d['share']:.1%})" for d in r["top10"][:TOP_OVERLAP]) or "–"
            lines.append(f"| {case.key} | {r['seed']} | {r['best_model'] or 'failed'} | {r['method'] or '–'} | {top} |")
    return "\n".join(lines)


def what_if_seed_table(what_ifs: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Seed | Model | Before | After | Change |", "|---|---:|---|---:|---:|---:|"]
    for w in what_ifs:
        for p in w["per_seed"]:
            lines.append(f"| {w['dataset']} | {p['seed']} | {p['best_model']} | {p['before']:.3f} | {p['after']:.3f} | "
                         f"{p['delta']:+.3f} |")
    return "\n".join(lines)


def base_rows(what_ifs: list[dict[str, Any]]) -> list[str]:
    lines = []
    for w in what_ifs:
        values = ", ".join(f"{k} = {v}" for k, v in w["base_row"].items())
        lines.append(f"- **{w['dataset']}** (typical '{w['class']}' record): {values}.")
    return lines


def build_markdown(data: dict[str, Any], summary: dict[str, Any]) -> str:
    seeds = ", ".join(str(s) for s in data["seeds"])
    env = data["environment"]
    packages = ", ".join(f"{name} {v}" for name, v in env["packages"].items())
    budget = data["budget"]
    parts = [
        "# Explanation stability",
        "",
        f"Generated by `scripts/evaluate_stability.py` on {data['generated']} ({data['duration_s'] / 60:.1f} min). "
        f"Seeds {seeds}; fast budget ({budget['max_candidates']} models × {budget['n_trials_per_model']} trials); "
        f"Python {env['python']} on {env['os']}; {packages}. The datasets are the synthetic, seeded samples in "
        "`data/samples/`. The Step 8 results in `evaluation.md` are not changed by this script.",
        "",
        "**Question.** If the same analysis is run again, do the explanations stay the same? Each seed changes the "
        "train/valid/test split, the hyper-parameter search and the models' random state, as two real runs would.",
        "",
        "## How it was measured",
        "",
        "- The real Orchestrator runs offline (no LLM, as in `evaluate.py`) until the EXPLAIN stage has finished.",
        f"- **Drivers**: the global explanation of the selected model, summed back to the original columns (SHAP, or "
        "permutation importance; for clustering, how strongly each column separates the clusters). The top "
        f"{TOP_K} per seed are saved in `stability.json`.",
        f"- **Spearman ρ (top-{TOP_K})**: rank correlation between two seeds over the union of their top-{TOP_K} "
        "drivers, each ranked by its position in that seed's full list (1 = same order, 0 = unrelated).",
        f"- **Top-{TOP_OVERLAP} Jaccard**: shared drivers ÷ all drivers in the two top-{TOP_OVERLAP} sets "
        "(1 = same five, 0.667 = four shared).",
        "- Both are the mean ± standard deviation over the three seed pairs (42–43, 42–44, 43–44).",
        "- **What-if** (classification): the same change on the same base row under each seed's model. The base row "
        "is the typical record of the at-risk class (median / most common value / latest date), so it does not "
        "depend on any one model. Output = probability of the at-risk class (churn: the positive class; delivery: "
        "the least frequent class).",
        f"- Verdict: *stable* when mean ρ ≥ {STABLE_RHO} and Jaccard ≥ {STABLE_JACCARD}; *unstable* when ρ < "
        f"{MODERATE_RHO} and Jaccard < {MODERATE_JACCARD}; otherwise *moderately stable*.",
        "",
        "## Driver ranking stability",
        "",
        driver_table(summary["drivers"]),
        "",
        "![Explanation stability](figures/stability.png)",
        "",
        "## What-if stability",
        "",
        what_if_table(summary["what_if"]),
        "",
        *base_rows(summary["what_if"]),
        "",
        "## Interpretation",
        "",
        *(f"- {note}" for note in summary["interpretation"]),
        "",
        "## Per-seed detail",
        "",
        per_seed_drivers(data),
        "",
        what_if_seed_table(summary["what_if"]),
        "",
        "## Limitations",
        "",
        "- Three seeds give three pairs per dataset: an indication of variability, not a confidence interval.",
        "- SHAP values for boosted trees are on the log-odds scale and for linear models on the model's own scale, so "
        "importance *shares* are compared between seeds only through ranks, never as raw values across model types.",
        "- The datasets are synthetic with a known signal; real data with correlated features usually gives less "
        "stable rankings (correlated columns can swap places).",
        "- One what-if scenario per classification dataset; other rows or larger changes can vary more.",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- figure


def make_figure(summary: dict[str, Any]) -> Path:
    ev._style()
    drivers = [d for d in summary["drivers"] if d["verdict"] != "not measured"]
    what_ifs = summary["what_if"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.35, 1]})

    series = [
        (f"Spearman ρ of driver ranks (top-{TOP_K})", [d["spearman"][0] for d in drivers],
         [d["spearman"][1] for d in drivers], ev.PURPLE),
        (f"Top-{TOP_OVERLAP} overlap (Jaccard)", [d["top5_jaccard"][0] for d in drivers],
         [d["top5_jaccard"][1] for d in drivers], ev.GREEN),
    ]
    ev._bars(left, [d["dataset"] for d in drivers], series)
    width = min(0.8 / len(series), 0.3)  # as in ev._bars; labels go above the error bar, not across it
    for index, (_, means, stds, _) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        for x, (mean, std) in enumerate(zip(means, stds, strict=True)):
            if math.isfinite(mean):
                left.text(x + offset, mean + (std if math.isfinite(std) else 0) + 0.015, f"{mean:.2f}",
                          ha="center", va="bottom", fontsize=12, color=ev.INK)
    low = min([0.0, *(d["spearman"][0] - d["spearman"][1] for d in drivers)])
    left.set_ylim(low - 0.05 if low < 0 else 0, 1.15)
    if low < 0:
        left.axhline(0, color=ev.INK, linewidth=1.2)
    left.set_ylabel("Agreement between seeds (1 = identical)")
    left.set_title("Driver rankings across seeds 42 / 43 / 44")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    rows = [(w["dataset"], p) for w in what_ifs for p in w["per_seed"]]
    y = np.arange(len(rows))[::-1]
    for pos, (_, p) in zip(y, rows, strict=True):
        right.annotate("", xy=(p["after"], pos), xytext=(p["before"], pos),
                       arrowprops={"arrowstyle": "-|>", "color": ev.MUTED, "linewidth": 1.6, "shrinkA": 7, "shrinkB": 7})
        right.text(p["before"], pos + 0.24, f"{p['before']:.2f}", ha="center", va="bottom", fontsize=12, color=ev.INK)
        right.text(p["after"], pos + 0.24, f"{p['after']:.2f}", ha="center", va="bottom", fontsize=12, color=ev.INK)
    right.scatter([p["before"] for _, p in rows], y, s=120, color=ev.ORANGE, edgecolor="white", linewidth=2,
                  zorder=3, label="Before the change")
    right.scatter([p["after"] for _, p in rows], y, s=120, color=ev.BLUE, edgecolor="white", linewidth=2, zorder=3,
                  label="After the change")
    right.set_yticks(y, [f"{name}, seed {p['seed']}" for name, p in rows])
    right.set_ylim(-0.7, len(rows) - 0.2)
    right.set_xlim(0, 1.05)
    right.grid(axis="y", visible=False)
    right.grid(axis="x", visible=True)
    right.set_xlabel("Probability of the at-risk class")
    right.set_title("Same what-if under each seed's model")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    for index in range(1, len(what_ifs)):  # thin separator between datasets
        right.axhline(len(rows) - 3 * index - 0.5, color=ev.GRID, linewidth=1.2)
    fig.tight_layout(w_pad=4)
    return ev._save(fig, FIGURE_NAME)


# --------------------------------------------------------------------------- main


def _evaluate(seeds: tuple[int, ...]) -> tuple[list[dict[str, Any]], float]:
    ev.ensure_samples()
    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="idss-stability-", ignore_cleanup_errors=True) as tmp:
        ev.isolated_workspace(Path(tmp) / "workspace")
        try:
            total = len(ev.CASES) * len(seeds)
            for number, (case, seed) in enumerate(((c, s) for c in ev.CASES for s in seeds), start=1):
                path = ev.SAMPLES_DIR / case.filename
                df = load_dataset(path)
                profile = profile_dataset(df)
                record = run_seed(case, seed, path, df, profile)
                runs.append(record)
                top = ", ".join(record["ranking"][:3]) or f"FAILED ({record['error']})"
                print(f"[{number}/{total}] {case.key} seed {seed}: {record['best_model']} ({record['method']}) "
                      f"top 3: {top}  {record['wall_time_s']:.0f}s", flush=True)
        finally:
            get_engine().dispose()  # release the SQLite file so the temp dir can be removed
            get_logger().remove()
    return runs, round(time.perf_counter() - started, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explanation stability of IDSS across seeds (offline).")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ev.SEEDS), help="random seeds (default: 42 43 44)")
    parser.add_argument("--from-json", action="store_true", help="rebuild stability.md and the figure from stability.json")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    if len(args.seeds) < 2 and not args.from_json:
        parser.error("stability needs at least two seeds")

    if args.from_json:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    else:
        runs, duration = _evaluate(tuple(args.seeds))
        data = {
            "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
            "seeds": list(args.seeds),
            "budget": BUDGET_PRESETS["fast"].model_dump(),
            "environment": ev.environment(),
            "duration_s": duration,
            "definitions": {
                "spearman": f"Spearman rank correlation between two seeds over the union of their top-{TOP_K} drivers "
                            "(ranks from each seed's full driver list; a missing feature shares the last rank)",
                "top5_jaccard": f"|A ∩ B| / |A ∪ B| of the two seeds' top-{TOP_OVERLAP} driver sets",
                "pairs": "mean ± sample standard deviation over all seed pairs",
                "what_if": "same change on the same base row (typical record of the at-risk class) under each "
                           "seed's model; probability of the at-risk class before/after and the change, mean ± std "
                           "over seeds",
            },
            "runs": runs,
        }

    summary = analyse(data)
    data["summary"] = summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(ev._finite(data), indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    figure = make_figure(summary)
    MD_PATH.write_text(build_markdown(data, summary), encoding="utf-8")

    print("\nDriver ranking stability:\n" + driver_table(summary["drivers"]))
    print("\nWhat-if stability:\n" + what_if_table(summary["what_if"]))
    print("\nInterpretation:", *(f"- {n}" for n in summary["interpretation"]), sep="\n")
    print("\nSaved:", *(p.relative_to(PROJECT_ROOT) for p in (MD_PATH, JSON_PATH, figure)), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

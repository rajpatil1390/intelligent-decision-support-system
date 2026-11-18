"""Step 8: ablation switches in the trainer, the clustering baseline and the evaluation script's helpers."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idss.agents.verifier import Verifier
from idss.core.schemas import TrainingBudget
from idss.ml.model_zoo import ModelContext, get_candidate
from idss.ml.trainer import AutoMLTrainer
from scripts import evaluate as ev
from scripts.make_samples import make_customer_segments, make_delivery_status
from tests.test_automl import classification_df, clustering_df, make_requirement, run_automl, timeseries_df

TWO_TRIALS = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)


# --------------------------------------------------------------------------- trainer switches


def test_without_hyperparameter_tuning_each_model_is_fitted_once_with_defaults(tmp_path: Path) -> None:
    result = run_automl(classification_df(), "tabular_classification", "target", ["random_forest"], tmp_path,
                        budget=TWO_TRIALS, trainer=AutoMLTrainer(tune_hyperparameters=False))
    [entry] = [e for e in result.leaderboard if e.selected]
    assert entry.n_trials == 1 and entry.params == {} and result.best_params == {}


def test_without_threshold_tuning_binary_models_keep_the_default_cutoff(tmp_path: Path) -> None:
    df = classification_df(n=400, minority=0.15)
    tuned = run_automl(df, "tabular_classification", "target", ["logistic_regression"], tmp_path / "on")
    fixed = run_automl(df, "tabular_classification", "target", ["logistic_regression"], tmp_path / "off",
                       trainer=AutoMLTrainer(tune_threshold=False))
    assert tuned.decision_threshold is not None
    assert fixed.decision_threshold is None and fixed.test_metrics_default_threshold == {}


def test_baseline_candidates_can_be_skipped(tmp_path: Path) -> None:
    result = run_automl(timeseries_df(), "time_series_forecasting", "sales", ["ridge"], tmp_path,
                        trainer=AutoMLTrainer(train_baselines=False))
    assert {e.model for e in result.leaderboard} == {"ridge"}
    assert result.baseline is not None and result.baseline.name == "seasonal_naive"  # test-split score kept


def test_clustering_with_ground_truth_is_compared_with_random_assignment(tmp_path: Path) -> None:
    df = clustering_df(n=300)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])
    df["group"] = [int(np.argmin(((centers - p) ** 2).sum(axis=1))) for p in df[["f1", "f2"]].to_numpy()]
    df["group"] = df["group"].map({0: "a", 1: "b", 2: "c"})
    result = run_automl(df, "tabular_clustering", "group", ["kmeans"], tmp_path)
    assert result.primary_metric == "adjusted_rand_index"  # chance-corrected; plain Rand index stays in the metrics
    assert result.baseline is not None and result.baseline.name == "random_assignment"
    assert abs(result.baseline.metrics["adjusted_rand_index"]) < 0.1  # random labels score about 0
    assert result.test_metrics["adjusted_rand_index"] > result.baseline.metrics["adjusted_rand_index"]
    assert "rand_index" in result.test_metrics and "rand_index" in result.baseline.metrics
    report = Verifier(None).verify(make_requirement("tabular_clustering", "group"), result)
    checks = {c.name: c for c in report.checks}
    assert checks["beats_baseline"].passed and "adjusted_rand_index" in checks["beats_baseline"].detail
    assert checks["overfitting"].detail.startswith("adjusted_rand_index:")  # gap on ARI, not silhouette


def test_gaussian_mixture_uses_several_initialisations() -> None:
    ctx = ModelContext(task_type="tabular_clustering", n_rows=1000)
    model = get_candidate("tabular_clustering", "gaussian_mixture").build({"n_components": 4}, ctx)
    assert model.n_init > 1  # one EM start can stop in a poor local optimum and make the refit unreliable


# --------------------------------------------------------------------------- new sample datasets


def test_new_sample_datasets_are_deterministic_and_well_formed() -> None:
    delivery = make_delivery_status(np.random.default_rng(1), n=600)
    again = make_delivery_status(np.random.default_rng(1), n=600)
    pd.testing.assert_frame_equal(delivery, again)
    assert set(delivery["delivery_status"]) == {"On time", "Late", "Very late"}
    assert delivery["delivery_status"].value_counts(normalize=True)["On time"] == pytest.approx(0.55, abs=0.01)
    segments = make_customer_segments(np.random.default_rng(1), n=400)
    assert segments["segment"].nunique() == 4 and segments["customer_id"].is_unique


# --------------------------------------------------------------------------- evaluation helpers


@pytest.mark.parametrize(
    "metric, value, expected",
    [("f1", 0.8, 0.8), ("rand_index", 1.2, 1.0), ("adjusted_rand_index", -0.05, 0.0), ("rmsle", 0.25, 0.8),
     ("rmse", 3.0, 0.25), ("f1", None, 0.0), ("rmse", math.nan, 0.0)],
)
def test_normalized_performance(metric: str, value: float | None, expected: float) -> None:
    assert ev.normalized_performance(metric, value) == pytest.approx(expected)


def test_nps_uses_the_scale_free_error_for_regression() -> None:
    assert ev.nps_metric("f1", {"f1": 0.7}) == "f1"
    assert ev.nps_metric("rmse", {"rmse": 300.0, "rmsle": 0.06}) == "rmsle"
    assert ev.nps_metric("rmse", {"rmse": 300.0}) == "rmse"  # negative targets: no RMSLE available


def test_comprehensive_score_and_improvement() -> None:
    assert ev.comprehensive_score(1.0, 0.6) == pytest.approx(0.8)
    assert ev.improvement_pct(0.75, 0.5, "f1") == pytest.approx(50.0)
    assert ev.improvement_pct(80.0, 100.0, "rmse") == pytest.approx(20.0)  # lower error = positive gain
    assert ev.improvement_pct(0.6, 0.0, "f1") is None
    assert ev.mean_std([1.0, 3.0, None]) == (2.0, pytest.approx(math.sqrt(2)))
    assert ev.mean_std([0.5]) == (0.5, 0.0)


def test_paper_scores_count_failed_runs_as_zero() -> None:
    runs = [ev.Outcome(case="Churn", seed=1, variant="full", trained=True, completed=True, nps=0.6),
            ev.Outcome(case="Churn", seed=2, variant="full", trained=True, completed=False, nps=0.7)]
    scores = ev.paper_scores(runs, [1, 2], end_to_end=True)
    assert scores["Churn"]["sr"][0] == pytest.approx(0.5)
    assert scores["Churn"]["nps"][0] == pytest.approx(0.3)  # the run without a report scores 0
    assert scores["Churn"]["cs"][0] == pytest.approx((0.8 + 0.0) / 2)
    trained_only = ev.paper_scores(runs, [1, 2], end_to_end=False)
    assert trained_only["Churn"]["sr"][0] == pytest.approx(1.0)


def test_perturbations_leave_protected_columns_alone() -> None:
    df = pd.DataFrame({"id": range(200), "x": np.arange(200.0), "c": ["a", "b"] * 100, "target": [0, 1] * 100})
    missing = ev.add_missing_values(df, np.random.default_rng(0), {"id", "target"})
    assert missing[["id", "target"]].notna().all().all()
    assert 0.05 < missing["x"].isna().mean() < 0.15 and missing["c"].isna().any()
    noisy = ev.add_noise_columns(df, np.random.default_rng(0), set())
    assert [c for c in noisy.columns if c.startswith("noise_")] == [f"noise_{i}" for i in range(1, 6)]
    pd.testing.assert_frame_equal(noisy[df.columns], df)

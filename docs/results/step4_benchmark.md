# Step 4 benchmark: AutoML engine

- Date: 2026-10-05 16:00
- Mode: **fast** — up to 3 candidates, 15 Optuna trials each, 60s per model, 180s total
- Plan: default ranking from `ModelAgent(None)` (no LLM); target and task suggested by the profiler
- Data: synthetic samples from `scripts/make_samples.py`; split 70/20/10 (stratified / chronological)
- Selection uses the validation split; the test split is scored once, for the selected model only
- Environment: Python 3.12.0, Windows 11, 16 CPUs; scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, optuna 5.0.0, pandas 3.0.6

## Summary (test split)

| Dataset | Task | Rows | Best model | Primary metric | Test score | Search time (s) | Final fit (s) |
|---|---|---:|---|---|---:|---:|---:|
| customer_churn.csv | tabular_classification | 3,000 | logistic_regression | f1 (higher is better) | 0.5579 | 39.7 | 0.84 |
| customer_churn.csv — after threshold tuning (t = 0.32) | tabular_classification | 3,000 | logistic_regression | f1 (higher is better) | 0.6230 | 38.8 | 0.83 |
| crop_price.csv | tabular_regression | 2,000 | lightgbm | rmse (lower is better) | 332.4751 | 12.3 | 0.65 |
| store_sales.csv | time_series_forecasting | 730 | ridge | rmse (lower is better) | 923.6040 | 6.5 | 0.04 |

## customer_churn.csv — leaderboard (validation f1)

| Rank | Model | Status | Valid score | Inference (ms/row) | Size (KB) | Trials (failed) |
|---:|---|---|---:|---:|---:|---:|
| 1 | logistic_regression (selected) | ok | 0.5965 | 0.0283 | 1.05 | 15 (0) |
| 2 | lightgbm | ok | 0.5762 | 0.0568 | 2029.38 | 15 (0) |
| 3 | random_forest | ok | 0.5754 | 0.1681 | 6113.27 | 15 (0) |

Other test metrics: accuracy = 0.7200, balanced_accuracy = 0.6731, precision = 0.5824, recall = 0.5354, roc_auc = 0.7738, log_loss = 0.5279

## crop_price.csv — leaderboard (validation rmse)

| Rank | Model | Status | Valid score | Inference (ms/row) | Size (KB) | Trials (failed) |
|---:|---|---|---:|---:|---:|---:|
| 1 | lightgbm (selected) | ok | 351.6238 | 0.0496 | 1313.07 | 15 (0) |
| 2 | random_forest | ok | 354.8243 | 0.2901 | 33529.8 | 15 (0) |
| 3 | ridge | ok | 448.3145 | 0.021 | 0.74 | 15 (0) |

Other test metrics: mse = 110539.6754, mae = 248.1788, r2 = 0.9801, mape = 0.0486, rmsle = 0.0597

## store_sales.csv — leaderboard (validation rmse)

| Rank | Model | Status | Valid score | Inference (ms/row) | Size (KB) | Trials (failed) |
|---:|---|---|---:|---:|---:|---:|
| 1 | ridge (selected) | ok | 950.6519 | 0.1653 | 0.75 | 15 (0) |
| 2 | xgboost | ok | 1286.3550 | 0.251 | 310.33 | 15 (0) |
| 3 | lightgbm | ok | 1442.7423 | 0.2704 | 146.86 | 15 (0) |
| 4 | seasonal_naive | ok | 2517.0034 | 0.2409 | 0.1 | 1 (0) |

Other test metrics: mse = 853044.3984, mae = 760.4003, r2 = 0.8885, mape = 0.0515, rmsle = 0.0632

<!-- step5-threshold:start -->
## customer_churn.csv — decision-threshold tuning (Step 5, 2026-10-05)

The probability cut-off for the positive class ('Yes') is tuned on the validation split to maximise f1, then applied unchanged to the test split (read once).

| Model | Threshold | F1 (0.5 → tuned) | Precision (0.5 → tuned) | Recall (0.5 → tuned) | Accuracy (0.5 → tuned) | ROC AUC |
|---|---:|---|---|---|---|---:|
| logistic_regression | 0.32 | 0.5579 → 0.6230 | 0.5824 → 0.5241 | 0.5354 → 0.7677 | 0.7200 → 0.6933 | 0.7738 |

Baseline (majority_class) test f1: 0.0000.

Rule-based verification (no LLM): **PASSED** — 7/7 checks passed; failed: none.

- `metric_targets`: pass — No target values were stated for the metrics.
- `beats_baseline`: pass — logistic_regression f1 = 0.623 vs majority_class baseline 0 (better).
- `constraints`: pass — The selected model meets all user constraints.
- `test_size`: pass — 300 test rows (minimum 30 for reliable metrics).
- `test_set_used_once`: pass — Test split accesses: ['final_evaluation'].
- `no_leakage`: pass — No feature is suspiciously correlated with the target.
- `overfitting`: pass — f1: train 0.649 vs test 0.623; gap 0.026 (limit 0.15).
<!-- step5-threshold:end -->

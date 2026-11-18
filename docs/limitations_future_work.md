# Limitations and future work

## Limitations

### Data and tasks

- **Tabular data only.** IDSS handles one table per run (CSV, Excel, JSON, Parquet up to 50 MB) for
  classification, regression, clustering and single-series forecasting. Images, free text as the main signal,
  graphs, joins across several tables and multi-series (panel) forecasting are out of scope. Free-text columns are
  detected and dropped ("free text is not supported yet").
- **In-memory processing.** Data is loaded with pandas; very large files (hundreds of MB and more) need sampling
  or a different engine.
- **One forecasting setup.** Lag, rolling-window and calendar features with gradient boosting / ridge, a seasonal
  naive baseline and a ±1.96 × validation-RMSE interval. Future values of other columns are assumed to repeat their
  last season; there is no ARIMA/Prophet-style model, hierarchical forecasting or proper probabilistic interval.
- **Fixed model zoo.** Six models for classification and for regression, three for forecasting (plus a
  seasonal-naive baseline) and three for clustering. No deep learning, no
  stacking or ensembling across families, no automated feature engineering beyond preprocessing and time-series
  features.

### Evaluation

- **Synthetic benchmark.** The five evaluation datasets are synthetic with known generating processes. They are
  realistic in shape (missing values, categoricals, imbalance, seasonality, overlapping clusters) but cleaner than
  real business data, and the results should not be read as performance on real data.
- **Small, fast-mode study.** Three seeds, the fast budget and one machine; timings depend on hardware and load.
- **The LLM path is not benchmarked.** The offline evaluation exercises the deterministic fallbacks. The quality of
  requirement parsing, clarifying questions, LLM model ranking and LLM-written briefs depends on the provider and
  model and is covered only by unit tests with a fake LLM and a few optional live tests.
- **Not comparable with the paper's numbers.** SR, NPS and CS follow AutoML-Agent's definitions, but datasets,
  budgets and success criteria differ.

### Modelling and decisions

- **Correlation, not causation.** SHAP values, what-if results and recommendations describe how the *model* responds
  to changed inputs. They are not causal effects: changing a lever in reality may not move the outcome as predicted,
  especially for confounded features. The reports say so, but users may still over-read them.
- **Simple counterfactuals.** The greedy counterfactual search changes at most three features to observed values and
  ignores feasibility and cost of the changes.
- **Selection by validation score.** The model is chosen on one 20% validation split (no cross-validation), so on
  small datasets the choice is noisy; the ablation shows tuned and default models within a few percent of each other
  on near-linear data.
- **Clustering selection uses ground truth when it exists.** If the requirement names a ground-truth column, the
  number of clusters is tuned against it (adjusted Rand index); without one, only internal scores (silhouette) are
  available, which do not always agree with what a business would call a good segmentation.
- **Verification thresholds are fixed.** The overfitting gap (0.15) and the minimum test size (30 rows) are global
  constants, not adapted to the task or metric.

### System

- **Single user, single machine.** One run trains at a time; there is no authentication, no user accounts and no
  multi-tenant isolation. The servers bind to localhost by design.
- **No model monitoring.** Models are not deployed as services, and there is no drift detection or scheduled
  retraining.
- **LLM dependency for free-text questions.** Without an LLM, the question is not parsed: the task and target come
  from the data profile, and *Ask Your Data* is unavailable.
- **Sandbox scope.** The sandbox is a static AST guard plus an isolated, time-limited child process — strong enough
  for pandas analysis code on a personal machine, but not an OS-level container or VM.

## Future work

1. **Real-data case studies.** Evaluate on public datasets (e.g. the tabular sets of the AutoML-Agent benchmark) and
   on a partner organisation's data, with a user study measuring decision quality and time saved.
2. **Benchmark the LLM path.** Run the evaluation with several providers and models, measuring parsing accuracy,
   clarifying-question usefulness and brief faithfulness, and the cost per run.
3. **Stronger AutoML.** Cross-validation for small data, ensembling / stacking, more time-series models and
   probabilistic intervals (e.g. conformal prediction), multi-series forecasting.
4. **Causal and prescriptive decisions.** Uplift modelling or causal inference for interventions, cost-aware
   optimisation of actions (budget constraints), feasibility constraints in counterfactuals.
5. **Multi-user deployment.** Authentication and roles, a job queue for several concurrent runs, a container image
   and a managed database for server deployment.
6. **Model operations.** Export trained models as a prediction endpoint, monitor drift and performance, retrain on a
   schedule.
7. **Richer knowledge base.** More best-practice notes, domain packs (retail, agriculture, telecom), embeddings-based
   retrieval with citations shown in the UI.
8. **Explanations for everyone.** Interactive SHAP plots, natural-language explanations of single predictions, and
   reports in regional languages.

# Chapter 8: Conclusion and Future Scope

## 8.1 Conclusion

We designed, built, tested and evaluated an **Intelligent Decision Support System (IDSS)** that turns a business
question in plain language and a table of data into a verified machine-learning model and explained, actionable
decisions. Inspired by AutoML-Agent [9], we kept its multi-agent decomposition, structured requirements,
retrieval-augmented planning, multi-stage verification and evaluation metrics, and changed the division of labour so
that **LLM agents plan and explain while deterministic, tested code computes every number**.

The objectives of Section 1.3.2 were met:

1. A resumable 13-state Orchestrator coordinates seven agents, each with a deterministic fallback, so the full pipeline
   runs without an LLM.
2. The Requirement Agent checks relevance, parses and repairs the requirement against the data and asks at most two
   rounds of clarifying questions.
3. The AutoML engine tunes a fixed scikit-learn / XGBoost / LightGBM zoo with Optuna within budgets and constraints,
   tunes the decision threshold and reads the test split exactly once, which is audited.
4. Seven verification checks decide whether a result can be trusted, with one re-plan and an explicit *not verified*
   label when the problem persists.
5. SHAP drivers, what-if simulation, risk tiers, counterfactual actions and forecasts are combined in a grounded decision
   brief and PDF / Markdown reports.
6. A 17-endpoint FastAPI backend and an eight-page Streamlit interface with Demo Mode make the system usable from a
   browser, and runs survive restarts.
7. Only aggregates reach the LLM, keys stay in `.env`, LLM-written code runs only in a sandbox and all loops and retries
   are bounded.
8. On five synthetic, seeded datasets the system completed every run (SR = 1.00) with NPS = 0.800 and CS = 0.900, beat
   the baseline everywhere and passed verification in 14 of 15 runs; 449 automated tests (447 offline, all passing)
   cover 93% of the code.

The project also taught us that honest evaluation changes a system: the Rand index proved misleading for clustering, so
we adopted the adjusted Rand index [18], and the same investigation revealed and fixed a real bug; a small
component, threshold tuning, mattered more on imbalanced data than hyperparameter tuning; and a verifier that may say
"not verified" is more useful to a decision maker than one that always passes. IDSS has clear limits — one table at a
time, synthetic and offline evaluation, sensitivity to missing values and model-based rather than causal what-if results
— but within them it shows that LLM agents for language combined with deterministic engines for computation give a
decision support system that is conversational, reproducible, self-checking and safe to run on a laptop.

## 8.2 Future Scope

- **Real-data studies:** public benchmark datasets and a partner organisation's data, with a user study of decision
  quality and time saved.
- **Benchmark the LLM path:** parsing accuracy, usefulness of clarifying questions, faithfulness of briefs and cost per
  run across several providers.
- **Stronger AutoML:** cross-validation for small data, ensembling, better imputation for missing values, more and
  multi-series forecasting models, calibrated prediction intervals.
- **Smarter re-planning:** change the strategy after a failed verification (regularisation, simpler models, more
  cleaning), not only the budget.
- **Causal and prescriptive decisions:** uplift modelling, cost-aware optimisation of actions, feasibility constraints
  in counterfactuals.
- **Multi-user deployment and model operations:** authentication, a job queue, containers, prediction services, drift
  monitoring and scheduled retraining.
- **Richer knowledge and explanations:** domain knowledge packs, embedding-based retrieval with cited passages,
  interactive explanations and reports in regional languages such as Marathi and Hindi.

## 8.3 Summary

IDSS met its eight objectives: it understands a plain-language question, builds and verifies a model with deterministic
code, turns it into explained, actionable decisions and delivers them through a safe, resumable web application that
also works offline. The future work follows from its limitations — real data, evaluation of the LLM path, robustness to
missing data and causal decision support.

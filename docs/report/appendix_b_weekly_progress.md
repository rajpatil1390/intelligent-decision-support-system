# Appendix B: Weekly Progress

This appendix records the work done in each of the eight weeks, in the five-point form of the department's *Project
Weekly Report*. The calendar dates ("From Date – To Date") are to be filled in by the team on each weekly report; the
work is described by week number only. The commits listed are in the project's Git history.

**Project name:** Intelligent Decision Support System Using Generative AI and Machine Learning

**Group members:** Raj Vishwas Patil, Bhavesh Vinod Patil, Chaitany Rajesh Naval, Varun Darshan Patil

**Guide:** Dr. Arjun Jadhaw; **HoD:** Dipak Patil

## Week 1 — Foundation

Work done during week: From ________ To ________

1. Set up the repository, the Python 3.12 virtual environment, pinned dependencies and the project conventions
   (`CLAUDE.md`, `README.md`, `.env.example`).
2. Implemented typed settings read from `.env`, with API keys stored as secret strings that are never logged.
3. Implemented the logging module and the single LLM client: OpenAI-compatible, a timeout on every call, at most 3
   retries with exponential backoff, and structured (Pydantic-validated) outputs.
4. Defined the shared Pydantic schemas (for example `UserRequirement`) and the folder structure for agents, ML engine,
   decision layer, knowledge, sandbox and storage.
5. Set up pytest with a temporary workspace and a mocked LLM; wrote the foundation tests (commits `146b1cf`,
   `417219c`).

Lead this week: Raj Vishwas Patil (LLM client, schemas), Chaitany Rajesh Naval (settings, logging), Varun Darshan
Patil (test harness); Bhavesh Vinod Patil (library selection).

## Week 2 — Data Module

Work done during week: From ________ To ________

1. Built the data loader for CSV (delimiter and encoding detection), Excel, JSON / JSON lines and Parquet.
2. Built the profiler: semantic column types, missing values, distinct counts, suggested target and task, and warnings
   for possible leakage and class imbalance.
3. Implemented cleaning, the 70/20/10 split (stratified for classification, chronological for time series) and the
   preprocessing pipeline (imputation, scaling, capped one-hot encoding).
4. Implemented the Data Agent, which sends only aggregates to the LLM for a plain-English summary and falls back to a
   template.
5. Generated the synthetic sample datasets (`make_samples.py`) and wrote the data-module tests (commit `cd2b765`).

Lead this week: Bhavesh Vinod Patil; Varun Darshan Patil (samples, tests).

## Week 3 — Requirement Agent and Orchestrator

Work done during week: From ________ To ________

1. Wrote the prompt templates and the prompt loader (`## system` / `## user` sections, `$variable` placeholders).
2. Implemented the Requirement Agent: relevance check, parsing into a `UserRequirement`, deterministic repair of target,
   task type and metrics, sufficiency check and clarifying questions.
3. Implemented the Orchestrator state machine (INIT → … → DONE, plus NEEDS_INPUT and FAILED) with one handler per state.
4. Saved the run context atomically to `state.json` after every transition, with resume support and bounded loops.
5. Added a live LLM demo script and the state-machine tests (commit `53ce937`).

Lead this week: Raj Vishwas Patil (agent, prompts), Chaitany Rajesh Naval (persistence); Varun Darshan Patil (tests).

## Week 4 — AutoML Engine

Work done during week: From ________ To ________

1. Built the model zoo: classification, regression, clustering and time-series candidates, each with a build function
   and an Optuna search space.
2. Implemented the tuner (TPE sampler, median pruner, two-stage trials on large data, failed trials pruned) and the
   evaluator (metric registry, complexity measures).
3. Implemented the trainer: selection on the validation split, refit on train + validation, a single audited
   evaluation on the test split, saved model, leaderboard and metrics.
4. Implemented the Model Agent: LLM ranking restricted to zoo names, default ranking as fallback, constraint parsing
   (latency, training time, explainability).
5. Ran the first benchmark on the samples (`docs/results/step4_benchmark.md`) and wrote the AutoML tests (commit
   `9f8d017`).

Lead this week: Bhavesh Vinod Patil (engine), Raj Vishwas Patil (Model Agent); Varun Darshan Patil (tests, benchmark).

## Week 5 — RAG, Verifier and Sandbox

Work done during week: From ________ To ________

1. Wrote ten best-practice notes and built the offline TF-IDF knowledge index with caching, retrieval and the Knowledge
   Agent; optional web search only with an API key.
2. Added decision-threshold tuning on the validation split and trivial baselines on the test split.
3. Implemented the Verifier with seven rule-based checks, an optional LLM review and one re-plan with the full budget.
4. Implemented the sandbox (AST guard, isolated child process, timeout, output limit, no secrets) and the Analysis
   Agent for *Ask Your Data*.
5. Wrote the tests for retrieval, verification, re-planning and sandbox security (commit `eea311a`).

Lead this week: Raj Vishwas Patil (knowledge), Bhavesh Vinod Patil (Verifier, thresholds), Chaitany Rajesh Naval
(sandbox); Varun Darshan Patil (tests).

## Week 6 — Decision Layer

Work done during week: From ________ To ________

1. Implemented SHAP explanations (tree and linear explainers, permutation fallback) mapped back to the original columns.
2. Implemented the what-if simulator for single records and for segments with structured filters and validated changes.
3. Implemented the recommender: actionable columns, risk tiers with stated cut-offs, counterfactual actions, lever
   effects, segment actions, forecasts with an uncertainty band.
4. Implemented the Insight Agent with the number-grounding check and a grounded template fallback.
5. Built the PDF and Markdown report builder and ran two complete offline runs with reports (`demo_step6.py`, commit
   `aa24eed`).

Lead this week: Bhavesh Vinod Patil (explanations, recommender), Raj Vishwas Patil (Insight Agent), Varun Darshan Patil
(reports); Chaitany Rajesh Naval (Orchestrator integration).

## Week 7 — FastAPI Backend and Streamlit UI

Work done during week: From ________ To ________

1. Implemented the SQLite database (SQLAlchemy 2.0) and the repository for datasets, runs and events, mirrored by the
   Orchestrator.
2. Implemented the FastAPI backend with 17 endpoints, upload limits, path checks and a uniform error format.
3. Implemented the background RunWorker (one training at a time) and the recovery of interrupted runs at start-up
   (commit `29007c0`).
4. Built the eight-page Streamlit interface and its HTTP client, including the Run Monitor, Decision Center and Reports
   pages.
5. Wrote the API tests and the UI client and page tests (commit `7f13403`).

Lead this week: Chaitany Rajesh Naval (backend), Varun Darshan Patil (UI); Raj Vishwas Patil and Bhavesh Vinod Patil
(integration of agent and result endpoints).

## Week 8 — Evaluation and Documentation

Work done during week: From ________ To ________

1. Wrote the offline evaluation script and ran the benchmark, SR / NPS / CS, ablation and robustness studies with three
   seeds (`docs/results/evaluation.md`).
2. Switched the clustering metric to the adjusted Rand index and fixed the Gaussian mixture initialisation found during
   the evaluation; re-ran the Segments dataset.
3. Wrote the documentation: architecture, user manual, API reference, testing notes, limitations; generated the diagrams
   (commit `a3aefd2`).
4. Applied the dashboard theme to the UI and captured the screenshots automatically (commit `3965778`).
5. Wrote the content of this project report and prepared the formatted report and the weekly reports.

Lead this week: Bhavesh Vinod Patil (evaluation), Varun Darshan Patil (theme, screenshots, documentation); Raj Vishwas
Patil and Chaitany Rajesh Naval (review, report).

The "Lead this week" lines follow the roles of Table 3.4 and the planned allocation of Table 3.5; all four members
shared testing, review and integration every week.

# Changelog

All notable changes to IDSS (Intelligent Decision Support System). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] – 2026-11-27 (v1.1: production hardening and explanation stability)

Thanks to **external reviewer feedback on LinkedIn**, which prompted this release, including the question "how stable
were the explanations across runs?". v1.1 treats the LLM's output and the uploaded data as untrusted input and measures
how much the explanations move between runs. Details: [README → Security & trust (v1.1)](README.md#security--trust-v11),
[docs/architecture.md § 10.1](docs/architecture.md#101-security-and-trust-v11) and
[docs/results/stability.md](docs/results/stability.md).

### Added

- **Plan contract** (`idss/core/plan_contract.py`): `AnalysisPlan`, the only planner output that can drive execution —
  a strict, frozen Pydantic model with extra keys forbidden. Task type, primary metric and candidate models are closed
  lists built from the metric registry and the model zoo (1–6 models, no duplicates, no baselines); budget mode,
  constraints, actionable features (≤ 20) and the time column are bounded. `parse_plan()` reports one sentence per
  problem.
- **Independent plan validation** (`idss/execution/plan_validator.py`): checks every plan against the data and hard
  limits in the settings without the LLM (target suits the task and is not an ID, text, date, empty or constant
  column; time column is a date; actionable features exist; metric defined; budget, training time, rows and file size
  within limits). A rejected plan is replaced by a fallback plan built from a fresh data profile, never from the LLM's
  text. TRAIN starts only from a plan sealed by the validator in the same process; a plan read back from `state.json`
  is validated again. New settings `MAX_UPLOAD_MB`, `PUBLIC_DEMO_MAX_UPLOAD_MB` and `PLAN_MAX_*` (see `.env.example`).
- **Prompt-injection defence** (`idss/core/sanitize.py`): every dataset-derived string in a prompt is cleaned (control,
  zero-width and bidi characters removed, tag delimiters neutralised, length capped) and wrapped in `<data>…</data>`;
  every system prompt states that such text is data, never instructions. A detector flags instruction-like text in
  headers and cells as a profile warning, shown on the Upload page and in the report under "Security checks".
- **Number provenance** (`idss/decision/facts.py`): a facts registry gives every computed number a key and a source.
  The decision brief is written with `{{key}}` placeholders only (no typed digits); the report prints every number
  through the registry and ends with a **"Sources of numbers"** appendix (number → key → source).
- **Explanation stability evaluation** (`scripts/evaluate_stability.py`): top-10 drivers per dataset and seed
  (42/43/44), Spearman rank correlation and top-5 Jaccard overlap between seed pairs, and the same what-if replayed
  under each seed's model → `docs/results/stability.json`, `stability.md` and `figures/stability.png`. The Step 8
  evaluation results are unchanged.
- API: `GET /runs/{id}` returns `plan_source`, `plan_fallback` and `plan_rejections`; the Run Monitor shows a warning
  with the reasons when a plan was replaced. `GET /runs/{id}/insight` returns `placeholder_text`.
- Data loader checks: empty or header-only files, duplicate column names, ragged rows and binary files are rejected
  with a clear message; UTF-16, cp1252 and latin-1 files are read with a warning; more than 1,000 columns warns.
- Tests: `tests/test_v11_plan_contract.py` and `tests/test_v11_security.py`.
- Deployment (added after the `v1.0-public` tag): embedded API mode for Streamlit Community Cloud (`IDSS_EMBEDDED_API=1`
  starts the API inside the Streamlit process; public demo explains at most 200 rows with SHAP).
- This changelog.

### Changed

- The Model Agent hands over a `PlanProposal`; only the validator turns it into an executable plan.
- The Insight Agent rejects a draft with an unknown key or a typed digit before the grounding check runs; the fallback
  template is itself written with placeholders.
- Report formatting: shares as percentages, thousands separators, signed expected impacts; the LLM reviewer's verdict
  is left out when it contains numbers.
- `POST /runs` refuses unknown or identifier `actionable_features` (400 `invalid_actionable_features`) and more than 20
  (422); New Analysis caps the selection at 20.
- Upload limits come from the settings (`MAX_UPLOAD_MB`, `PUBLIC_DEMO_MAX_UPLOAD_MB`).
- Version 1.1.0 in `/health`, the OpenAPI document and `CITATION.cff`.

### Documentation

- README: new "Security & trust (v1.1)" and "Explanation stability" sections and a current test-suite line;
  `docs/testing.md` stays the record of the test suite at submission.
- `docs/architecture.md`: the new modules, the validated PLAN → TRAIN step and § 10.1 Security and trust.

## [1.0.0] – 2026-11-18 (v1.0: final-year project release)

First public release of the final-year project; the college submission is tagged `v1.0-final`, the public GitHub
release `v1.0-public`.

### Added

- **Agents and orchestration**: Requirement Agent (free text → validated, data-repaired requirement with clarifying
  questions), Data Agent (load, profile, clean, stratified or chronological 70/20/10 split), Knowledge Agent (offline
  TF-IDF retrieval over 10 best-practice notes, optional web search), Model Agent (ranks a fixed model zoo), Verifier
  (seven rule-based checks plus an optional LLM review, one re-plan), Analysis Agent (sandboxed pandas code for
  *Ask Your Data*) and Insight Agent (decision brief with a grounding check); a resumable state machine that saves
  `state.json` after every transition and works fully offline without an LLM.
- **AutoML engine**: scikit-learn, XGBoost and LightGBM models for classification, regression, clustering and
  time-series forecasting; Optuna TPE search with pruning; tuned decision threshold for binary classification;
  trivial baselines; a test split read exactly once.
- **Decision layer**: SHAP drivers summed back to original columns, record and segment what-if simulation, risk tiers
  with stated cut-offs, counterfactual actions, forecasts with intervals, cluster profiles, PDF and Markdown reports.
- **Product**: FastAPI backend with SQLite persistence and one background worker; Streamlit dashboard (8 pages) with
  Demo Mode, Run Monitor, Decision Center and Reports.
- **Public release**: MIT licence, detailed README, public demo mode (small uploads, fast budget only, no code
  execution, automatic clean-up of old runs), `CITATION.cff`.
- **Evaluation and documentation**: offline evaluation on five synthetic datasets with three seeds (SR, NPS, CS,
  ablation, robustness), architecture, API reference, user manual, testing notes, project report, poster and slides.
  At submission: 449 tests, 447 offline passing ([docs/testing.md](docs/testing.md)).

[1.1.0]: https://github.com/<your-account>/<this-repo>/compare/v1.0-public...v1.1-public
[1.0.0]: https://github.com/<your-account>/<this-repo>/releases/tag/v1.0-public

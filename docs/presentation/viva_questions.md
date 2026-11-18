# IDSS – Viva Questions and Answers

Intelligent Decision Support System Using Generative AI and Machine Learning · G H Raisoni College of Engineering and
Management, Jalgaon

Thirty likely viva questions with short, honest answers, grouped by presenter in presenting order. Every fact comes from
our report (`docs/report/`), the evaluation (`docs/results/evaluation.md`) or the testing notes (`docs/testing.md`).
Results are means over three seeds unless they are marked "single demo run (seed 42)".

## Raj Vishwas Patil – problem, objectives and GenAI agents

### 1. What problem does IDSS solve, and for whom?

Business users such as small retailers or logistics teams have data and clear questions but no data scientist. IDSS
takes a plain-language question and one table (CSV, Excel, JSON or Parquet, up to 50 MB) and returns a verified model,
key drivers, what-if results, risk tiers or a forecast, and a PDF report. It runs on one laptop, and everything except
*Ask Your Data* also works without an LLM.

### 2. Why build IDSS instead of using a plain AutoML tool such as Auto-sklearn?

AutoML tools expect a problem specified in code and stop at a trained model. IDSS adds what a decision maker needs
around the model: it understands the question and repairs it against the data, verifies the result with seven
rule-based checks including a trivial baseline, and turns the model into decisions — SHAP drivers, what-if simulation,
risk tiers, counterfactual actions and a grounded brief. A good F1 score alone does not say what to do.

### 3. What did you take from AutoML-Agent, and what did you change?

We kept its multi-agent decomposition, structured requirement parsing, retrieval-augmented planning, multi-stage
verification and the SR / NPS / CS metrics. We changed the division of labour: the LLM only ranks names from a fixed
model zoo and tested code trains everything; retrieval is offline; the only LLM-written code runs in a sandbox; and we
added a decision layer, resumable runs, an API and a UI. We read its code for ideas only and copied nothing.

### 4. What does each agent do, and what happens without an LLM?

Seven agents: Requirement (question → typed requirement), Data (profile, clean, split; the LLM only writes the
summary), Knowledge (best-practice notes), Model (ranks zoo models), Verifier (rule checks plus an optional LLM review),
Analysis (*Ask Your Data* code) and Insight (decision brief). Each has a deterministic fallback — a requirement from the
data profile, a default ranking, a template brief — so the pipeline runs offline; our whole evaluation ran that way.

### 5. What happens if the question is vague, wrong or unrelated to the data?

A relevance check comes first: an unrelated question gets a polite reply and no training. The LLM then parses the
question into a typed requirement, and a deterministic repair fixes it against the data — an unusable target becomes the
profile's suggested target, a conflicting task type follows the data and wrong-family metrics are removed. If something
is still missing, the run asks clarifying questions, at most two rounds, and then stops with a message.

### 6. How do you stop the LLM from inventing numbers in the decision brief?

The Insight Agent only receives a facts object built from computed results. A grounding check then compares every
number in its draft with the facts (±1%, rounding and fraction-to-percent allowed; small counts up to 10 exempt). Any
unsupported number triggers one retry with feedback; if it fails again, a template brief is used, which is complete and
grounded by construction. A unit test covers exactly this case with an invented "37.5%".

### 7. What data is sent to the LLM? Is the user's data private?

Prompts contain column names, types, missing-value percentages, distinct counts, aggregate statistics, metrics and
computed facts — never raw rows, file paths or keys. Keys live only in `.env`, which Git ignores. The API and the UI
listen on localhost, and uploads, runs and the database stay in the local `workspace/` folder. Retrieval is offline, and
web search runs only if a search key is configured.

### 8. How does retrieval-augmented planning work?

Ten best-practice notes that we wrote (`data/knowledge/`) are split into chunks of about 500 characters and indexed with
TF-IDF; the index is cached and rebuilt when the notes change. For each run the Knowledge Agent retrieves the top 4
passages (at most 2 per note) and condenses them into a brief that guides the Model Agent's ranking; without an LLM the
passages are used directly. It is offline, deterministic and easy to audit.

## Chaitany Rajesh Naval – architecture, backend, API, database and testing

### 9. Explain the architecture of IDSS.

Six layers: presentation (Streamlit, eight pages), service (FastAPI, 17 endpoints), orchestration (the 13-state
Orchestrator), agents (seven, each with a fallback), engines (data preparation, AutoML, explanations, what-if, reports,
sandbox) and infrastructure (run folders, `state.json`, SQLite). The UI talks only to the API. The LLM provider and web
search are optional external services, so the whole pipeline can run offline.

### 10. What happens if the server stops in the middle of a run?

After every state transition the Orchestrator saves the run atomically to `workspace/runs/<run_id>/state.json` and
mirrors it as an event in SQLite, so a run can be resumed from its last state. When the API starts, runs that a previous
process left mid-pipeline are marked FAILED, so nothing stays "running" forever. A failing state handler moves the run
to FAILED with a message; it never crashes the server.

### 11. How is the REST API designed?

FastAPI with 17 endpoints: datasets (upload, list, profile), runs (start, list, status with progress and events,
answers to clarifying questions), results (leaderboard, explanations, recommendations, insight), what-if, *Ask Your
Data*, and the report and chart files. One background worker runs one training job at a time; a second request gets
409 `busy`. Every error has the same `{error, detail}` shape, and responses never contain keys or server paths.

### 12. Why SQLite, and what does the database store?

Three tables: `datasets` (file, rows, columns, upload time), `runs` (question, state, task, best model, metric,
verification result, report paths) and `events` (one row per state transition). SQLite needs no server and suits a
single-user application on one machine; we use SQLAlchemy 2.0 with WAL mode and foreign keys. `state.json` remains the
source of truth, so a database error is only logged and never breaks a run.

### 13. How do you make the *Ask Your Data* sandbox safe?

LLM-written code is untrusted input. A static AST check comes first: only pandas, numpy, math, statistics, datetime,
json and re may be imported, and `open`, `eval`, `exec`, `os`, `sys`, private attributes and file I/O are blocked. Only
then does the code run in a separate `python -I` child process in a temporary folder, with restricted built-ins, no
secrets in its environment, a 20-second timeout and output capped at 20 KB.

### 14. How do you validate uploads and other input?

Pydantic validates every request. Uploads are limited to 50 MB — checked from the Content-Length header before reading
and again while streaming — and only CSV, Excel, JSON and Parquet files are accepted. File names are sanitised, run ids
and chart names are pattern-checked, and resolved paths must stay inside the run folder. What-if filters are structured
conditions, never evaluated strings, and CORS allows only the local UI.

### 15. How did you test the system?

We have 449 automated pytest tests: 447 run offline and all pass, and 2 live LLM tests run only on request. The LLM is
replaced by a fake client and the workspace by a temporary folder, so no test touches the network. Statement coverage
is 93%. The API is tested with FastAPI's TestClient and every Streamlit page with AppTest — with data, without data and
with the backend down. `scripts/evaluate.py` is the system-level test.

## Bhavesh Vinod Patil – AutoML engine, algorithms, results and ablation

### 16. How does the AutoML engine choose and tune models?

The LLM only ranks names from a fixed zoo: for example logistic regression, random forest, extra trees, XGBoost,
LightGBM and histogram gradient boosting for classification; ridge and the tree models for regression; k-means, Gaussian
mixture and agglomerative clustering. Optuna's TPE sampler with a median pruner tunes each candidate within a budget
(fast: 3 models × 15 trials, 60 s per model), and the best compliant model is refitted on train + validation.

### 17. How do you prevent data leakage and an over-optimistic test score?

The data is split 70/20/10 — stratified for classification, chronological for time series — and lag and rolling
features use only strictly earlier rows. Model selection and threshold tuning use the validation split only. The test
split sits behind a holdout object that logs every access; the trainer scores it exactly once, and the Verifier checks
that count and the profiler's leakage warnings.

### 18. Why do you tune the decision threshold, and why does recall matter for churn?

Churners are the minority (32.8%), so a 0.5 cut-off misses many of them. We tune the threshold on the validation split
for F1 — not for precision or recall alone, which would pick a degenerate cut-off. Over three seeds churn F1 rises from
0.561 to 0.649. In the single demo run (seed 42) the cut-off 0.32 found about 77% of churners instead of 54%, usually
the right trade-off for a retention campaign: a churner the model misses is never contacted.

### 19. How does verification work, and why was Delivery with missing values "not verified"?

Seven rule checks: metric targets, beating the trivial baseline, constraints, at least 30 test rows, test split used
once, no leakage warnings, and a train–test gap below 0.15 on the primary 0–1 metric. A failure triggers one re-plan
with the full budget. With 10% missing values (seed 42), Delivery chose XGBoost, which fitted the training rows much
better than the test rows; the gap stayed too large after the re-plan, so the run finished but was labelled *not
verified*.

### 20. Why do you use the adjusted Rand index instead of the Rand index?

The Rand index counts record pairs on which the clustering and the truth agree. Most pairs belong to different groups,
so it rewards over-splitting and is not corrected for chance: random clusters already scored RI 0.625, and an untuned
8-cluster k-means once beat the tuned 4-cluster model on RI. ARI is about 0 for random labels and 1 for a perfect match — our
model scores 0.816 against 0.004 for random. We still report RI (0.930).

### 21. What are SR, NPS and CS, and can you compare them with the AutoML-Agent paper?

SR is the share of runs that complete the whole pipeline; NPS is the test score for 0–1 metrics, or 1 / (1 + RMSLE) for
regression and forecasting; CS = 0.5 · SR + 0.5 · NPS. We got SR 1.000, NPS 0.800 and CS 0.900 over five datasets and
three seeds. These cannot be compared with the paper's tables, because datasets, budgets and success criteria differ,
and our SR is high mainly because tested code does the training.

### 22. What did the ablation study show?

We removed one component at a time. Without threshold tuning, churn F1 fell from 0.649 to 0.561 (−13.4%). Without
hyperparameter tuning, CS fell from 0.900 to 0.871, mostly because Segments lost 35.6% — tuning finds the right number
of clusters. Re-planning ran once on clean data and did not change the score. A single default model reached CS 0.854
and passed verification in only 6 of 15 runs (full system: 14 of 15).

### 23. How does a trained model become a decision?

SHAP values on up to 500 training rows, summed back to the original columns, give the key drivers. For binary tasks the
High risk tier starts at the larger of the tuned threshold and the 70th percentile of predicted probability. A greedy
counterfactual search changes at most 3 actionable columns to observed values to bring a risky record below the
threshold, and what-if simulation applies validated changes to a record or a segment. Every recommendation carries its
evidence numbers.

## Varun Darshan Patil – dashboard, live demo, limitations and conclusion

### 24. How is the user interface built, and how does it talk to the backend?

Streamlit with eight pages: Home, Upload Data, New Analysis, Run Monitor, Results, Decision Center, Ask Your Data and
Reports. The UI never imports backend code; it calls the FastAPI backend only through an HTTP client and shows friendly
messages when the backend is down or a result is not ready. The Run Monitor polls every 2 seconds and stops at DONE,
FAILED or NEEDS_INPUT, where it shows the answer form.

### 25. What does Demo Mode do, and what if the live demo fails?

Demo Mode on Home uploads `data/samples/customer_churn.csv` through the API, starts a fast run and opens the Run
Monitor; a fast run takes about 1–3 minutes. We start the servers with `scripts\run_all.ps1` before the talk and do one
rehearsal run. If anything fails, we show the screenshots in `docs/screenshots/`, captured from the real UI with
Playwright, and a hidden backup slide lists the demo steps.

### 26. Why did you use synthetic datasets?

We generated five datasets with `scripts/make_samples.py` from fixed seeds. They describe no real people or companies,
their generating process is known and the results are exactly reproducible. They were designed to be realistic in shape
— missing values, categorical columns, class imbalance, seasonality, overlapping clusters — and to cover all four task
families. The honest cost: these are not results on real business data, which may be noisier and contain drift or
leakage.

### 27. What are the main limitations of IDSS?

One table per analysis (no joins), single-series forecasting, no deep learning and no multi-user accounts. The
evaluation is offline, on synthetic data, with three seeds, the fast budget and one machine, so the LLM path (parsing,
briefs, LLM review) is not benchmarked. Missing values are the main weakness (crop price RMSE 57.1% worse with 10%
blanks), one re-plan did not cure over-fitting on Delivery, and what-if results show correlation, not causation.

### 28. What is the future scope?

Real-data studies with a user study; benchmarking the LLM path (parsing accuracy, faithful briefs, cost per run across
providers); stronger AutoML with cross-validation, ensembling and better imputation; smarter re-planning that changes
the strategy, not only the budget; causal decisions with uplift models and cost-aware actions; multi-user deployment
with authentication and drift monitoring; and reports in regional languages such as Marathi and Hindi.

### 29. COCOMO estimates 28.86 person-months, but you spent about 8. Why the gap?

Basic COCOMO (organic) on our 10.683 KLOC gives *E* = 2.4 × 10.683<sup>1.05</sup> = 28.86 person-months. Our actual
effort was 4 members × about 2 months ≈ 8 person-months, roughly 3.6 times less. COCOMO assumes every line is written
and debugged by hand within an industrial lifecycle; we reused mature libraries (training, tuning, SHAP, the web server,
UI, database and PDF are not our code) and used an AI coding assistant. We report the gap rather than adjust the
estimate.

### 30. What role did Claude Code play in the development?

Claude Code, an AI coding assistant, helped us generate and refactor code, which shortened the time from a design
decision to working, tested code. It did not replace the team's work: the design decisions, integration, testing and
evaluation were carried out and verified by us. We state this openly in Chapters 3 and 5 of the report, and it is one
reason why the actual effort is below the COCOMO estimate.

# IDSS API reference

The backend is a FastAPI app (`api/main.py`, title **IDSS API**, version **1.0.0**). Start it with
`powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1`; it listens on `http://localhost:8000` (this machine
only). Interactive documentation generated from the code is at `http://localhost:8000/docs` (Swagger UI) and
`/redoc`; the OpenAPI schema is at `/openapi.json`.

- All request and response bodies are JSON, except the dataset upload (multipart form) and the report / chart
  downloads (files).
- Responses never contain API keys or server file paths: reports and charts are returned as `/runs/...` URLs.
- CORS allows the Streamlit UI only (`http://localhost:8501`, `http://127.0.0.1:8501`).
- One run trains at a time. Training happens on a background thread; the API answers immediately.

Examples use `curl.exe` (in Windows PowerShell, plain `curl` is an alias for `Invoke-WebRequest`). Values in the
responses are illustrative.

## Endpoints at a glance

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health` | Liveness, LLM configured yes/no, version | 200 |
| POST | `/datasets` | Upload a dataset (CSV, XLSX, JSON, Parquet; ≤ 50 MB) | 201 |
| GET | `/datasets` | List uploaded datasets | 200 |
| GET | `/datasets/{dataset_id}/profile` | Full column-by-column profile | 200 |
| POST | `/runs` | Start an analysis run | 202 |
| GET | `/runs` | List runs (filter by dataset or state) | 200 |
| GET | `/runs/{run_id}` | State, progress, events, metrics, verification, questions | 200 |
| POST | `/runs/{run_id}/answer` | Answer clarifying questions (run in `NEEDS_INPUT`) | 202 |
| GET | `/runs/{run_id}/leaderboard` | Every trained candidate with scores and cost | 200 |
| GET | `/runs/{run_id}/explanations` | Key drivers (SHAP), local examples, clusters / forecast, chart URLs | 200 |
| GET | `/runs/{run_id}/recommendations` | Actions, risk tiers, counterfactual and what-if examples | 200 |
| GET | `/runs/{run_id}/insight` | Plain-language decision brief | 200 |
| POST | `/runs/{run_id}/what-if` | Simulate changes for one record or a segment | 200 |
| POST | `/runs/{run_id}/ask` | Ask a question about the data (needs an LLM) | 200 |
| GET | `/runs/{run_id}/report.pdf` | PDF report | 200 |
| GET | `/runs/{run_id}/report.md` | Markdown report | 200 |
| GET | `/runs/{run_id}/charts/{name}` | One chart PNG of the run | 200 |

## Errors

Every error has the same shape:

```json
{"error": "not_ready", "detail": "Run 20261006-101500-a1b2c3 has no explanations yet (state TRAIN)."}
```

| HTTP | `error` | When |
|---|---|---|
| 400 | `bad_request`, `invalid_dataset`, `empty_file`, `invalid_what_if` | Malformed input, unreadable file, impossible simulation |
| 403 | `public_demo` | Public demo only (`IDSS_PUBLIC_DEMO=1`): `POST /runs` with `mode` `full`, or `POST /runs/{id}/ask` |
| 404 | `not_found` | Unknown or invalid dataset / run id, missing chart or report file |
| 409 | `busy` | Another run is training; try again when it has finished |
| 409 | `not_ready` | The result does not exist yet (e.g. explanations while the run is in `TRAIN`) |
| 409 | `run_in_progress` | What-if / ask on the run that is currently executing |
| 409 | `invalid_state` | Answering a run that is not waiting for an answer |
| 413 | `file_too_large` | Upload over 50 MB, 10 MB in the public demo (checked from `Content-Length` before the body is read, and while streaming) |
| 415 | `unsupported_file_type` | Extension other than `.csv`, `.xlsx`, `.json`, `.parquet` |
| 422 | `validation_error` | Request body fails validation; `detail` is a list of `{loc, msg, type}` |
| 500 | `internal_error` | Unexpected server error (details only in the server log) |
| 503 | `llm_not_configured` | `/ask` without an LLM in `.env` |

## System

### GET /health

```bash
curl.exe http://localhost:8000/health
```

```json
{"status": "ok", "llm_configured": false, "version": "1.0.0", "public_demo": false, "max_upload_mb": 50.0}
```

`llm_configured` is a boolean only; the key is never returned. `public_demo` is true when the server runs with
`IDSS_PUBLIC_DEMO=1`: Ask Your Data is disabled, uploads are capped at 10 MB (`max_upload_mb`), runs use the fast
budget only (also for the re-plan after a failed verification), and at startup runs older than 24 hours are deleted
together with the uploads no remaining run uses. The UI reads this to show its "Public demo – do not upload private
data" banner.

## Datasets

### POST /datasets

Multipart upload with one field, `file`. The file name is sanitised and stored as
`workspace/uploads/<uuid>_<name>`; the profile is computed once and cached.

```bash
curl.exe -F "file=@data/samples/customer_churn.csv" http://localhost:8000/datasets
```

```json
{
  "id": "3f6c2a9d0b8e4c71a5d2e9f04b7c1a66",
  "filename": "customer_churn.csv",
  "rows": 3000,
  "cols": 12,
  "uploaded_at": "2026-10-06T10:14:52.120000Z",
  "profile": {
    "n_rows": 3000,
    "n_cols": 12,
    "columns": [
      {"name": "customer_id", "semantic_type": "id", "missing_pct": 0.0, "n_unique": 3000},
      {"name": "contract_type", "semantic_type": "categorical", "missing_pct": 0.0, "n_unique": 3},
      {"name": "churn", "semantic_type": "categorical", "missing_pct": 0.0, "n_unique": 2}
    ],
    "suggested_target": "churn",
    "suggested_task_type": "tabular_classification",
    "time_column": "signup_date",
    "warnings": []
  }
}
```

Errors: 400 `invalid_dataset` / `empty_file`, 413 `file_too_large`, 415 `unsupported_file_type`.

### GET /datasets

Query: `limit` (1–500, default 100), `offset` (default 0). Newest first.

```bash
curl.exe "http://localhost:8000/datasets?limit=10"
```

```json
[{"id": "3f6c2a9d0b8e4c71a5d2e9f04b7c1a66", "filename": "customer_churn.csv", "rows": 3000, "cols": 12,
  "uploaded_at": "2026-10-06T10:14:52.120000Z"}]
```

### GET /datasets/{dataset_id}/profile

The full `DatasetProfile`: per column the semantic type (`numeric`, `categorical`, `datetime`, `text`, `id`,
`boolean`), missing %, distinct count and statistics, plus duplicate rows, suggested target and task, time column
and warnings (for example possible leakage or class imbalance).

```bash
curl.exe http://localhost:8000/datasets/3f6c2a9d0b8e4c71a5d2e9f04b7c1a66/profile
```

## Runs

### POST /runs

| Field | Type | Notes |
|---|---|---|
| `dataset_id` | string | From `POST /datasets` |
| `question` | string | 1–5,000 characters, the business question |
| `mode` | `"fast"` \| `"full"` | Budget: fast = 3 models × 15 trials; full = 6 models × 40 trials. Default `fast` |
| `actionable_features` | list of strings, optional | Columns the business can change (used for recommendations) |

```bash
curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" ^
  -d "{\"dataset_id\": \"3f6c2a9d0b8e4c71a5d2e9f04b7c1a66\", \"question\": \"Which customers are likely to churn?\", \"mode\": \"fast\"}"
```

```json
{"run_id": "20261006-101530-a1b2c3", "state": "INIT", "status_url": "/runs/20261006-101530-a1b2c3"}
```

Without an LLM the question is not parsed: the run uses the profile's suggested task and target. Errors: 404 unknown
dataset, 403 `public_demo` (mode `full` in the public demo), 409 `busy`, 422 validation.

### GET /runs

Query: `limit`, `offset`, `dataset_id`, `state` (one of the run states, e.g. `DONE`). Newest first.

```json
[{
  "run_id": "20261006-101530-a1b2c3", "dataset_id": "3f6c2a9d0b8e4c71a5d2e9f04b7c1a66",
  "question": "Which customers are likely to churn?", "state": "DONE", "progress": 100, "active": false,
  "task_type": "tabular_classification", "best_model": "logistic_regression", "primary_metric": "f1",
  "metric_value": 0.623, "verification_passed": true,
  "created_at": "2026-10-06T10:15:30Z", "updated_at": "2026-10-06T10:16:02Z",
  "report_pdf_url": "/runs/20261006-101530-a1b2c3/report.pdf",
  "report_md_url": "/runs/20261006-101530-a1b2c3/report.md"
}]
```

### GET /runs/{run_id}

Everything in the summary above plus `message`, `target`, `test_metrics`, `decision_threshold`, `verification`
(checks with pass/fail and detail), `pending_questions`, `answers`, `errors`, the plan review (v1.1: `plan_source`
= `llm` / `default` / `fallback`, `plan_fallback` = the proposed plan was rejected by the plan validator, and
`plan_rejections` = why) and the `events` timeline. Poll it every few seconds while a run is active; `progress` is 0–100 by state (INIT 0 … TRAIN 30, VERIFY 70, EXPLAIN 75,
INSIGHT 90, REPORT 95, DONE / FAILED 100).

```json
{
  "run_id": "20261006-101530-a1b2c3", "state": "DONE", "progress": 100, "active": false,
  "target": "churn", "best_model": "logistic_regression", "primary_metric": "f1", "metric_value": 0.623,
  "decision_threshold": 0.32,
  "test_metrics": {"f1": 0.623, "precision": 0.524, "recall": 0.768, "accuracy": 0.693, "roc_auc": 0.774,
                   "positive_class": "Yes", "n_test_rows": 300},
  "verification": {"passed": true, "checks": [
    {"name": "beats_baseline", "passed": true, "detail": "logistic_regression f1 = 0.623 vs majority_class baseline 0 (better)."},
    {"name": "overfitting", "passed": true, "detail": "f1: train 0.649 vs test 0.623; gap 0.026 (limit 0.15)."}
  ], "llm_verdict": null, "llm_concerns": [], "recommendations": []},
  "pending_questions": [], "answers": [], "errors": [],
  "plan_source": "default", "plan_fallback": false, "plan_rejections": [],
  "message": "About 30% of customers are at high risk ... Full report: /runs/20261006-101530-a1b2c3/report.pdf",
  "events": [
    {"ts": "2026-10-06T10:15:30Z", "from_state": null, "to_state": "INIT", "message": "Run created."},
    {"ts": "2026-10-06T10:15:31Z", "from_state": "INIT", "to_state": "PARSE", "message": "Request is relevant. ..."}
  ]
}
```

### POST /runs/{run_id}/answer

Only for a run in `NEEDS_INPUT`. The answer is merged into the question and the run continues from `PARSE` in the
background (at most two rounds of questions).

```bash
curl.exe -X POST http://localhost:8000/runs/20261006-101530-a1b2c3/answer -H "Content-Type: application/json" ^
  -d "{\"answer\": \"Predict the churn column; F1 matters most.\"}"
```

```json
{"run_id": "20261006-101530-a1b2c3", "state": "PARSE", "status_url": "/runs/20261006-101530-a1b2c3"}
```

Errors: 409 `invalid_state` (not waiting), 409 `busy`.

## Results

All result endpoints return 409 `not_ready` until the run has reached the stage that produces them.

### GET /runs/{run_id}/leaderboard

```json
{
  "run_id": "20261006-101530-a1b2c3", "primary_metric": "f1", "direction": "maximize",
  "best_model": "logistic_regression",
  "entries": [
    {"rank": 1, "model": "logistic_regression", "status": "ok", "selected": true, "params": {"C": 0.41},
     "valid_score": 0.6512, "decision_threshold": 0.32, "valid_score_default_threshold": 0.5965,
     "train_time_s": 0.84, "inference_ms_per_row": 0.0283, "model_size_kb": 1.05, "n_features": 24,
     "n_trials": 15, "n_failed_trials": 0, "violations": []},
    {"rank": 2, "model": "lightgbm", "status": "ok", "selected": false, "valid_score": 0.6301, "...": "..."}
  ]
}
```

### GET /runs/{run_id}/explanations

```json
{
  "run_id": "20261006-101530-a1b2c3",
  "global_explanation": {
    "method": "shap_linear", "output": "probability that churn = 'Yes'", "n_samples": 500,
    "features": [
      {"feature": "contract_type", "importance": 0.081, "share": 0.31, "direction": "mixed",
       "detail": "Month-to-month raises the prediction most"},
      {"feature": "tenure_months", "importance": 0.064, "share": 0.25, "direction": "decreases", "detail": ""}
    ],
    "notes": []
  },
  "local_examples": [{"method": "shap_linear", "prediction": 0.81,
                      "contributions": [{"feature": "contract_type", "value": "Month-to-month", "impact": 0.12}]}],
  "clusters": [],
  "forecast": null,
  "charts": {"feature_importance": "/runs/20261006-101530-a1b2c3/charts/feature_importance.png"}
}
```

For clustering runs `clusters` holds per-segment profiles; for time series `forecast` holds the next periods with
a ±1.96 × validation-RMSE interval.

### GET /runs/{run_id}/recommendations

```json
{
  "run_id": "20261006-101530-a1b2c3",
  "actionable_features": ["contract_type", "payment_method", "support_calls", "monthly_charges"],
  "recommendations": [
    {"action": "Offer one-year contracts to month-to-month customers in the High tier",
     "rationale": "Switching contract_type lowers predicted churn for the riskiest customers.",
     "expected_impact": -0.12, "impact_unit": "change in average churn probability",
     "confidence": "medium", "evidence": {"segment_size": 412, "avg_before": 0.58, "avg_after": 0.46}}
  ],
  "risk_tiers": [
    {"name": "High", "min_probability": 0.47, "max_probability": 0.95, "count": 900, "share": 0.3,
     "avg_probability": 0.62, "top_drivers": ["contract_type", "tenure_months"]}
  ],
  "tier_cutoffs": {"high": 0.47, "medium": 0.32, "percentile": 70, "percentile_value": 0.47},
  "counterfactual_example": {"row_index": 17, "probability_before": 0.71, "probability_after": 0.29,
                             "threshold": 0.32, "success": true,
                             "changes": [{"feature": "contract_type", "before": "Month-to-month", "after": "One year"}]},
  "what_if_example": {"output": "probability that churn = 'Yes'", "before": 0.41, "after": 0.33,
                      "delta": -0.08, "delta_pct": -19.5, "changes": {"support_calls": {"add": -1}}, "warnings": []},
  "notes": []
}
```

### GET /runs/{run_id}/insight

```json
{
  "executive_summary": "About 30% of customers are at high risk of churning ...",
  "key_findings": ["..."], "recommended_actions": ["..."], "risks_and_limitations": ["..."], "next_steps": ["..."],
  "generated_by": "template",
  "grounding": {"passed": true, "attempts": 1, "unsupported_numbers": [], "unknown_keys": []},
  "placeholder_text": {"executive_summary": "About {{risk_tiers.high.share}} of customers are at high risk ...", "...": "..."}
}
```

`generated_by` is `llm` when an LLM wrote the brief and it passed the number checks (v1.1: every number written as a
`{{key}}` placeholder of the run's facts registry and filled in by code, no digit typed outside a placeholder, every
key known, and every resulting number found in the facts); otherwise the deterministic template is used.
`placeholder_text` holds the brief before the numbers were filled in.

### POST /runs/{run_id}/what-if

Give **exactly one** of `base_row` (one record) or `segment` (a list of filters; `[]` = every record), plus
`changes`. A change is a value (`"contract_type": "One year"`) or an operation `{"set": v}`, `{"add": n}`,
`{"pct": n}` (percent). Filters are structured conditions with `op` in `==, !=, <, <=, >, >=, in`; they are never
evaluated as code. Values outside the observed range or unknown categories produce `warnings`.

One record:

```bash
curl.exe -X POST http://localhost:8000/runs/20261006-101530-a1b2c3/what-if -H "Content-Type: application/json" ^
  -d "{\"base_row\": {\"contract_type\": \"Month-to-month\", \"tenure_months\": 5, \"monthly_charges\": 85}, \"changes\": {\"contract_type\": \"One year\"}}"
```

```json
{"output": "probability that churn = 'Yes'", "before": 0.68, "after": 0.47, "delta": -0.21, "delta_pct": -30.9,
 "changes": {"contract_type": "One year"}, "warnings": []}
```

A segment:

```json
{"segment": [{"column": "contract_type", "op": "==", "value": "Month-to-month"},
             {"column": "support_calls", "op": ">=", "value": 3}],
 "changes": {"monthly_charges": {"pct": -10}}}
```

```json
{"output": "probability that churn = 'Yes'", "count": 214, "avg_before": 0.61, "avg_after": 0.57,
 "delta": -0.04, "delta_pct": -6.6, "filters": ["..."], "changes": {"monthly_charges": {"pct": -10}}, "warnings": []}
```

Errors: 400 `invalid_what_if`, 409 `not_ready` / `run_in_progress`, 422 when both or neither of `base_row` and
`segment` are given. Time-series runs simulate with the dataset's history.

### POST /runs/{run_id}/ask

The LLM writes pandas code (it sees only column names and types), which runs in the sandbox (AST guard, isolated
child process, 20 s timeout, no secrets in its environment); one repair attempt if it fails.

```bash
curl.exe -X POST http://localhost:8000/runs/20261006-101530-a1b2c3/ask -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is the churn rate by contract type?\"}"
```

```json
{"question": "What is the churn rate by contract type?", "ok": true,
 "answer": "Month-to-month customers churn most (about 45%) ...",
 "code": "result = df.groupby('contract_type')['churn'].apply(lambda s: (s == 'Yes').mean())",
 "result": {"Month-to-month": 0.45, "One year": 0.21, "Two year": 0.09},
 "stdout": "", "attempts": 1, "error": null, "asked_at": "2026-10-06T10:20:11Z"}
```

Errors: 503 `llm_not_configured`, 409 `run_in_progress`, 403 `public_demo` (always, in the public demo).

## Reports and charts

| Endpoint | Returns |
|---|---|
| `GET /runs/{run_id}/report.pdf` | `application/pdf`, downloaded as `idss_report_<run_id>.pdf` |
| `GET /runs/{run_id}/report.md` | `text/markdown`, downloaded as `idss_report_<run_id>.md` |
| `GET /runs/{run_id}/charts/{name}` | `image/png`; `name` like `feature_importance` or `feature_importance.png` |

```bash
curl.exe -o report.pdf http://localhost:8000/runs/20261006-101530-a1b2c3/report.pdf
```

Chart names must match `[A-Za-z0-9][A-Za-z0-9_-]*(.png)` and resolve to a file directly inside the run's `charts/`
folder; anything else is 400 or 404. Reports before the `REPORT` stage give 409 `not_ready`.

## A complete session in Python

```python
import time

import requests

API = "http://localhost:8000"
with open("data/samples/customer_churn.csv", "rb") as f:
    dataset = requests.post(f"{API}/datasets", files={"file": f}, timeout=60).json()
run = requests.post(f"{API}/runs", json={"dataset_id": dataset["id"],
                                         "question": "Which customers are likely to churn?"}, timeout=30).json()
while (detail := requests.get(f"{API}/runs/{run['run_id']}", timeout=30).json())["state"] not in ("DONE", "FAILED", "NEEDS_INPUT"):
    time.sleep(2)
print(detail["state"], detail["best_model"], detail["metric_value"])
recommendations = requests.get(f"{API}/runs/{run['run_id']}/recommendations", timeout=30).json()
```

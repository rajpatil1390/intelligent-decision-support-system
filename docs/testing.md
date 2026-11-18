# Testing

IDSS has an automated pytest suite that runs fully offline: every LLM call is replaced by a fake client
(`tests/fakes.py`), web search is disabled and the workspace (runs, logs, database, knowledge cache) is redirected to
a temporary folder by `tests/conftest.py`, so tests never touch `workspace/` or the network. Training tests use tiny
synthetic data and budgets of one or two Optuna trials.

## Running the tests

```powershell
.venv\Scripts\python.exe -m pytest -q                                               # offline suite
.venv\Scripts\python.exe -m pytest -q --cov=idss --cov=api --cov=ui --cov=configs   # with coverage
.venv\Scripts\python.exe -m pytest --collect-only -q                                # list the tests
.venv\Scripts\python.exe -m pytest -m live                                          # real LLM (needs .env; costs tokens)
.venv\Scripts\python.exe -m ruff check .                                            # lint
```

`pytest.ini` deselects the `live` marker by default, so the offline command never calls a real LLM.

## Test inventory

Counts from `pytest --collect-only -q` (2026-10-06, after Step 9): **449 tests**, of which **447 run offline** and 2
are `live` tests (real LLM, deselected by default). Last full offline run: **447 passed, 2 deselected** in 206.9 s
(about 3.5 minutes).

| File | Tests | Area |
|---|---:|---|
| `tests/test_foundation.py` | 36 | Settings, schemas, JSON extraction, LLM client |
| `tests/test_data_module.py` | 45 | Loader, profiler, preprocessing, splits, Data Agent |
| `tests/test_requirement_and_manager.py` | 36 | Prompts, Requirement Agent, Orchestrator state machine |
| `tests/test_automl.py` | 46 | Model zoo, tuner, trainer, Model Agent, TRAIN state |
| `tests/test_step5.py` | 55 | Threshold tuning, knowledge retrieval, Verifier, sandbox, Analysis Agent, re-planning |
| `tests/test_step6.py` | 43 | Explanations, what-if, recommendations, grounded insights, reports, offline runs |
| `tests/test_storage.py` | 6 | SQLite repository and the Orchestrator's mirroring |
| `tests/test_api.py` | 39 | Every API endpoint and error case |
| `tests/test_ui_client.py` | 18 | UI HTTP client (mocked `requests`) |
| `tests/test_ui_pages.py` | 106 | Every Streamlit page through `AppTest`, plus the Step 9 theme (hero, footer, KPI strip, Demo Mode, Plotly charts, zebra tables, pipeline stepper) |
| `tests/test_evaluation.py` | 17 | Step 8: ablation switches, ARI baseline and verifier checks, Gaussian-mixture initialisation, sample generators, SR/NPS/CS helpers, perturbations |
| `tests/test_live_llm.py` | 2 | Real LLM smoke tests (`-m live`) |
| **Total** | **449** | |

## Coverage

Measured with pytest-cov 7.1.0 (coverage 7.16.2) over `idss`, `api`, `ui` and `configs`:

```powershell
.venv\Scripts\python.exe -m pytest -q --cov=idss --cov=api --cov=ui --cov=configs
```

**Total: 93%** (6,494 statements, 468 missed; measured 2026-10-06 after Step 9).

| Package | Coverage | Notes |
|---|---:|---|
| `idss/core` | 85–100% | lowest: prompt loader (85%) |
| `idss/agents` | 84–99% | lowest: Analysis Agent (84%) |
| `idss/ml` | 91–99% | |
| `idss/decision` | 92–97% | |
| `idss/knowledge` | 84–100% | lowest: web search providers (network calls are never made in tests) |
| `idss/storage` | 100% | |
| `idss/execution` | 92% / 0% | `sandbox_runner.py` shows 0% only because it runs in a separate child process that coverage does not trace; the sandbox tests do run it |
| `api` | 80–100% | lowest: background worker (80%) |
| `ui` | 75–100% | lowest: Upload Data page (75%) |
| `configs` | 98% | |

## What is tested

| Area | How |
|---|---|
| Settings, schemas, LLM client | Settings from `.env` and environment, schema validation, JSON extraction from model output, structured output retried with validation feedback and abandoned after the retry cap, transient API errors retried with backoff, non-retryable errors raised at once — with a mocked OpenAI client |
| Data module | Loading every supported format, profiling (types, IDs, dates, leakage, imbalance), cleaning, stratified and chronological splits, preprocessing, the Data Agent with and without an LLM |
| Requirement Agent + Orchestrator | Parsing and deterministic repair, clarifying questions, the full state machine (including NEEDS_INPUT, revision limits, FAILED on handler errors, step cap), persistence and resume |
| AutoML engine | Every model-zoo family, Optuna tuning and pruning, budget limits, constraint violations and selection, refit, the test split read exactly once, time-series features and the seasonal-naive baseline, the Model Agent's ranking and fallbacks |
| Knowledge, Verifier, Sandbox | Chunking, index caching and rebuilds, retrieval, web search disabled without keys; each verifier check failing on its own; the VERIFY → PLAN re-plan at most once; the sandbox blocking unsafe code, killing infinite loops, truncating output and hiding secrets |
| Decision layer | SHAP and permutation explanations mapped to original columns, what-if (record and segment) validation, risk tiers and counterfactuals, grounded insights (ungrounded numbers rejected), PDF and Markdown reports, offline end-to-end runs |
| API + persistence | Every endpoint with FastAPI's TestClient: uploads (limits, types, sanitised names), run lifecycle, 409 busy / not_ready / run_in_progress, 404s, path checks for charts and reports; repository CRUD and the Orchestrator's mirroring |
| UI | The HTTP client with mocked `requests`; every Streamlit page through `AppTest` with data, without data and with the backend down, plus the main interactions |
| Step 8 evaluation | The trainer's ablation switches, the clustering baseline, the new sample generators, NPS / CS / improvement formulas and the data perturbations |

## Evaluation beyond unit tests

`scripts/evaluate.py` is the system-level test: it runs the real Orchestrator end to end on five datasets with
three seeds, plus ablation and robustness variants, and writes [results/evaluation.md](results/evaluation.md).

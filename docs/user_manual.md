# IDSS user manual

IDSS (Intelligent Decision Support System) turns a business question and a table of data into a trained, verified
machine-learning model and plain-language decisions: key drivers, recommendations, risk tiers or a forecast,
what-if simulations and a PDF report. This manual covers installation, configuration, running the app on Windows,
each page of the interface, and troubleshooting.

## 1. Requirements

- Windows 10 or 11 (the scripts are PowerShell; the Python code also runs on macOS/Linux).
- **Python 3.12** and Git.
- About 2 GB of free disk space for the virtual environment.
- Optional: an API key for an OpenAI-compatible LLM (OpenAI, Groq, Google Gemini) or a local Ollama model. Without
  one, everything except *Ask Your Data* still works.

## 2. Installation

From a PowerShell window:

```powershell
git clone <your-repo-url> idss
cd idss
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Check the installation by running the test suite (about 5 minutes, no network access needed). The test tools are in
`requirements-dev.txt` (`requirements.txt` holds only what the app needs to run):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Always use `.venv\Scripts\python.exe`, not a system Python.

## 3. Configuration (`.env`)

`.env` holds your settings and secrets; it is ignored by Git and never leaves your machine. `.env.example` documents
every variable.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `groq`, `gemini` or `ollama` |
| `LLM_MODEL` | empty | Model name from your provider's list. Empty = offline mode |
| `LLM_API_KEY` | empty | API key (leave empty for Ollama) |
| `LLM_BASE_URL` | provider default | Only for a custom OpenAI-compatible endpoint |
| `LLM_TIMEOUT` | `60` | Seconds before an LLM call is abandoned (retried at most 3 times) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TAVILY_API_KEY`, `SERPAPI_KEY` | empty | Optional web search for the knowledge agent; empty = local notes only |
| `IDSS_API_URL` | `http://localhost:8000` | Shell variable (not in `.env`) telling the UI where the API is |
| `IDSS_PUBLIC_DEMO` | `0` | `1` for a shared, hosted demo: Ask Your Data off, uploads ≤ 10 MB, fast budget only, runs older than 24 h deleted at API startup, "do not upload private data" banner |

**Offline mode** (no `LLM_MODEL` / `LLM_API_KEY`): the task and target come from the data profile, the model plan
uses a fixed ranking, verification uses its rule-based checks only, and insights come from a grounded template.
**LLM mode**: the agents parse your question, ask clarifying questions, rank models, review the verification and
write the decision brief. Only aggregates (column names and types, metrics, computed facts) are sent to the LLM,
never raw rows. LLM calls cost money with paid providers.

## 4. Starting and stopping

Start the API and the UI together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

This opens the API in a new PowerShell window, waits until it answers, then starts the UI in the current window and
opens `http://localhost:8501` in your browser. To run the two parts separately, use two terminals:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1     # backend on http://localhost:8000 (docs: /docs)
powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1      # UI on http://localhost:8501
```

Options: `run_api.ps1 -Port 8001`, `run_ui.ps1 -Port 8502 -ApiUrl http://localhost:8001 -Headless`,
`run_all.ps1 -ApiPort 8001 -UiPort 8502`. Both servers listen on this computer only.

Stop the UI with **Ctrl+C**; close the API window (or press Ctrl+C in it) to stop the backend. A run that is still
training when the API stops is marked *FAILED* at the next start; start it again.

## 5. Page walkthrough

The sidebar on every page shows whether the API is online, whether an LLM is configured and which run is current.

| # | Page | What you do there |
|---|---|---|
| 1 | **Home** | See what IDSS does, the four-step flow, headline numbers (datasets, runs, successful runs, average 0–1 primary metric) and your most recent runs. **Demo Mode** uploads `data/samples/customer_churn.csv`, starts a fast churn analysis and opens the Run Monitor — one click for a live presentation. |
| 2 | **Upload Data** | Upload a CSV, Excel, JSON or Parquet file (≤ 50 MB) with **Upload and profile**, or pick an earlier upload. The profile lists every column's type, missing values and distinct values, plus the suggested target, the suggested task and any warnings (for example possible leakage or class imbalance). |
| 3 | **New Analysis** | Pick a dataset, describe what you want to know or decide (or click an example), choose the **training budget** — *fast* (3 models × 15 trials, about 1–3 minutes) or *full* (6 models × 40 trials, up to ~18 minutes) — optionally choose the **actionable columns** (the ones your business can change), and press **Start analysis**. |
| 4 | **Run Monitor** | Follow the pipeline stages (done / current / waiting / failed), the progress bar and the event timeline (refreshed every 2 seconds). If IDSS needs more information, the run pauses and shows its questions; type an answer and press **Send answer** (at most two rounds). |
| 5 | **Results** | Headline test metrics and four tabs: **Leaderboard** (every model tried, its validation score, speed and size), **Verification** (each quality check with pass/fail and the reason), **Key drivers** (which columns move the prediction and in which direction) and **Charts**. |
| 6 | **Decision Center** | **Decision brief** — summary, findings, recommended actions, risks and next steps, with risk tiers (classification), the forecast (time series) or segment profiles (clustering). **What-if simulator** — start from a typical record, change inputs and see the prediction before and after. **Segment simulator** — build filters (column, operator, value), choose changes (set / add / percent) and see the average effect on that group. |
| 7 | **Ask Your Data** | Chat about the run's dataset (needs an LLM). IDSS writes pandas code, runs it in a sandbox and answers; the code and printed output are shown in collapsible blocks. |
| 8 | **Reports** | Download the PDF or Markdown report, preview the Markdown with its charts, and browse earlier reports. |

### A first run (about 3 minutes)

Shortcut: press **Demo Mode** on Home to do steps 1–2 in one click.

1. **Upload Data**: upload `data/samples/customer_churn.csv`. The profile suggests `churn` as the target.
2. **New Analysis**: click *Example: Customer churn*, keep **fast**, press **Start analysis**.
3. **Run Monitor**: watch INIT → … → TRAIN → VERIFY → EXPLAIN → INSIGHT → REPORT → DONE.
4. **Results**: the best model, its F1 on held-out test data and the verification checks.
5. **Decision Center**: High / Medium / Low risk tiers, the actions with their expected effect, and a what-if:
   change `contract_type` to *One year* and press **Simulate**.
6. **Reports**: download the PDF.

### Sample datasets

All are synthetic (generated by `scripts/make_samples.py`, no real people or companies).

| File | Rows | Task | Example question |
|---|---:|---|---|
| `customer_churn.csv` | 3,000 | binary classification (`churn`) | Which customers are likely to churn, and what should we do to retain them? |
| `crop_price.csv` | 2,000 | regression (`price`) | Predict the crop price from weather, soil and yield data. |
| `store_sales.csv` | 730 | time-series forecasting (`sales`) | Forecast daily store sales for the next month. |
| `delivery_status.csv` | 2,500 | 3-class classification (`delivery_status`) | Which shipments will arrive late or very late? |
| `customer_segments.csv` | 1,600 | clustering (`segment` is the hidden ground truth) | Group our customers into behavioural segments. |

In offline mode the task is the profile's suggestion; for `customer_segments.csv` that is classification of
`segment`. Clustering needs an LLM-parsed question such as the example above (or an explicit requirement, as
`scripts/evaluate.py` uses).

## 6. Command-line tools

| Command | What it does |
|---|---|
| `.venv\Scripts\python.exe -m pytest -q` | Full offline test suite (live LLM tests are skipped) |
| `.venv\Scripts\python.exe -m pytest -q --cov=idss --cov=api --cov=ui --cov=configs` | Tests with a coverage report |
| `.venv\Scripts\python.exe -m pytest -m live` | Tests that call the real LLM (needs `.env`; costs tokens) |
| `.venv\Scripts\python.exe scripts\evaluate.py` | Offline evaluation: benchmark, SR/NPS/CS, ablation, robustness (about 15 minutes) → `docs/results/` |
| `.venv\Scripts\python.exe scripts\evaluate.py --from-json` | Rebuild the evaluation report and charts from `evaluation.json` |
| `.venv\Scripts\python.exe scripts\evaluate.py --cases Segments --reason "..."` | Re-run only some datasets and merge them into `evaluation.json` |
| `.venv\Scripts\python.exe scripts\make_diagrams.py` | Redraw the architecture and planning diagrams → `docs/diagrams/` |
| `.venv\Scripts\python.exe scripts\make_samples.py` | Regenerate the sample datasets (deterministic) |
| `.venv\Scripts\python.exe scripts\demo_step6.py` | Two complete offline runs from the command line, with reports |
| `.venv\Scripts\python.exe -m ruff check .` | Lint (unused imports, undefined names, syntax problems) |

## 7. Where files go

| Path | Contents |
|---|---|
| `workspace/uploads/` | Uploaded datasets and their cached profiles |
| `workspace/runs/<run_id>/` | `state.json` (the run's full state), data splits, `best_model.joblib`, leaderboard, metrics, test predictions, `charts/`, `report.pdf`, `report.md` |
| `workspace/idss.db` | SQLite index of datasets, runs and events |
| `workspace/logs/idss.log` | Application log (rotates at 10 MB) |

All of `workspace/` except the empty placeholders is ignored by Git. Delete a run's folder to remove it; delete
`idss.db` to reset the lists (it is recreated on the next start).

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| UI shows **"Backend not running"** | Start `scripts\run_api.ps1` (or use `run_all.ps1`). If the API uses another port, start the UI with `-ApiUrl`. |
| `running scripts is disabled on this system` | Run the scripts as shown, with `powershell -ExecutionPolicy Bypass -File ...`. |
| `Virtual environment not found` | Create `.venv` and install the requirements (section 2). |
| Port already in use | Another process uses 8000 / 8501: stop it or pass `-Port` / `-ApiPort` / `-UiPort`. |
| **409 busy** / "Another run is training" | Only one run trains at a time. Wait for it on the Run Monitor page. |
| Upload rejected (413 / 415) | Files must be ≤ 50 MB and `.csv`, `.xlsx`, `.json` or `.parquet`. |
| Upload rejected (400 invalid dataset) | The file could not be read as a table (wrong delimiter, empty file, broken Excel). Re-export it as UTF-8 CSV. |
| Sidebar says **LLM: not configured** | Set `LLM_MODEL` and `LLM_API_KEY` in `.env` and restart the API. Offline mode still works. |
| *Ask Your Data* is disabled | It needs an LLM (see above). |
| LLM errors or timeouts in the log | Check the key, the model name and your provider's quota; raise `LLM_TIMEOUT`. Each call is retried at most 3 times, then the agent falls back to its deterministic path. |
| Run stuck in **NEEDS_INPUT** | It is waiting for your answer on the Run Monitor page. |
| Run **FAILED** | Open Run Monitor → *All errors*; the log in `workspace/logs/idss.log` has the details. Common causes: no usable target column, too few rows (a split needs enough rows per class), a time series without a date column. |
| Results say **"not verified"** | A quality check failed (see Results → Verification), for example a large train-test gap or a model no better than the baseline. The results are shown but should be treated with caution; try the *full* budget or more data. |
| Training is slow | Use *fast* mode; close other heavy programs. *Full* mode can take up to ~18 minutes. |
| Run marked FAILED after a restart | The API stopped while it was training. Start a new run. |
| Tests fail with permission errors on Windows | Another process (an open Excel file, an antivirus scan) holds a file in `workspace/`; close it and run the tests again. |

## 9. Privacy and security

- Data stays on your computer; both servers listen on localhost only.
- With an LLM, only metadata and computed aggregates are sent to the provider — never raw rows, file paths or keys.
- Code written by the LLM (*Ask Your Data*) runs in a sandbox: a static check blocks file, network and system
  access before it runs, then it runs in a separate process with a time limit and without your secrets.
- Keys live only in `.env`, which Git ignores.

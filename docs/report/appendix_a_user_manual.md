# Appendix A: User Manual (Short)

This appendix summarises how to install and use IDSS. The full manual is `docs/user_manual.md` in the repository.

## A.1 Requirements

- Windows 10 or 11, Python 3.12 and Git; about 2 GB of free disk space for the virtual environment.
- Optional: an API key for an OpenAI-compatible LLM provider (OpenAI, Groq, Google Gemini) or a local Ollama model.
  Without one, everything except *Ask Your Data* works in offline mode.

## A.2 Installation

From a PowerShell window in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m pytest -q        # optional check: the offline test suite
```

To use an LLM, set `LLM_PROVIDER`, `LLM_MODEL` and `LLM_API_KEY` in `.env` (the file stays on the computer and is ignored
by Git). Leave them empty for offline mode.

## A.3 Starting and Stopping

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

This starts the API (`http://localhost:8000`, interactive documentation at `/docs`) in a new window and then the user
interface at `http://localhost:8501`. Stop the UI with Ctrl+C and close the API window to stop the backend. Both
servers listen on this computer only.

## A.4 Pages

**Table A.1:** Pages of the user interface

| Page | What the user does |
|---|---|
| Home | Reads the overview, sees headline numbers and recent runs; **Demo Mode** runs the churn example in one click |
| Upload Data | Uploads a CSV, Excel, JSON or Parquet file (≤ 50 MB) and reads its profile and warnings |
| New Analysis | Describes the problem in plain language (or picks an example), chooses the *fast* or *full* budget and, optionally, the actionable columns, and starts the analysis |
| Run Monitor | Follows the stages and the event timeline; answers clarifying questions when the run asks for them |
| Results | Reads the test metrics, the leaderboard, the verification checks, the key drivers and the charts |
| Decision Center | Reads the decision brief, risk tiers or forecast and recommended actions; runs what-if simulations for one record or a segment |
| Ask Your Data | Asks questions about the data (needs an LLM); the generated code and its output are shown |
| Reports | Downloads the PDF or Markdown report and previews it |

## A.5 A First Run (about 3 minutes)

1. On **Home**, press **Demo Mode** (or upload `data/samples/customer_churn.csv` on **Upload Data** and choose the
   churn example on **New Analysis**).
2. On **Run Monitor**, watch the run move from INIT through TRAIN and VERIFY to DONE.
3. On **Results**, check the best model, its F1 score on the held-out test data and the verification checks.
4. In the **Decision Center**, read the High / Medium / Low risk tiers and the recommended actions, then open the
   what-if simulator, change `contract_type` to another contract and press **Simulate**.
5. On **Reports**, download the PDF.

## A.6 Troubleshooting

**Table A.2:** Common problems

| Problem | Solution |
|---|---|
| "Backend not running – start scripts/run_api.ps1" | Start the API (or use `run_all.ps1`) and refresh the page |
| Upload rejected (413 / 415) | Files must be at most 50 MB and of type `.csv`, `.xlsx`, `.json` or `.parquet` |
| "409 busy" / "Another run is training" | Only one run trains at a time; wait for it on the Run Monitor page |
| Run stuck in NEEDS_INPUT | It is waiting for an answer on the Run Monitor page |
| Results say "not verified" | A quality check failed (see Results → Verification); treat the results with caution, try the *full* budget or more data |
| Ask Your Data is disabled | Configure an LLM in `.env` and restart the API |
| A run is marked FAILED after a restart | The API stopped while it was training; start a new run |

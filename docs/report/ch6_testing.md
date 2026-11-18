# Chapter 6: Testing

Testing was part of every weekly increment: no increment was closed until its own tests and the full existing suite
passed. This chapter describes the black-box and white-box techniques, the automated and manual testing with their
statistics and coverage, the identified test cases and the security tests.

## 6.1 Black Box, White Box Testing

**Black-box tests** check what a component does through its public interface. The API is called with FastAPI's
`TestClient` as the UI would call it, Streamlit pages are rendered with `AppTest` and inspected for what a user would
see, and complete offline runs are checked for their final state and outputs. We used **equivalence partitioning**
(supported and unsupported files; relevant and irrelevant questions), **boundary values** (an upload just over the
limit; exactly two clarification rounds; fewer than 30 test rows) and **error guessing** (path-traversal strings,
unknown ids, a stopped backend).

**White-box tests** use knowledge of the code to exercise paths and invariants: a spy confirms that the tuner only ever
sees the training and validation splits, a handler that loops forever tests the step limit, each verification check is
made to fail on its own, and one changed value in a time series proves that a lag feature never sees its own row.
Coverage (Section 6.2.3) measures how much code the tests execute (Table 6.1).

**Table 6.1:** Black-box and white-box testing in IDSS

| Aspect | Black-box | White-box |
|---|---|---|
| Basis | Requirements and public interfaces (API, UI, Orchestrator) | Branches, invariants and internal objects |
| Techniques | Equivalence partitioning, boundary values, error guessing, end-to-end scenarios | Spies and fakes, fault injection, invariant checks, coverage |
| Main files | `test_api.py`, `test_ui_pages.py`, `test_ui_client.py` | `test_automl.py`, `test_step5.py`, `test_requirement_and_manager.py`, `test_data_module.py` |

## 6.2 Automated / Manual Testing

### 6.2.1 Automated Testing

The pytest suite runs fully offline: every LLM call is replaced by a fake client (`tests/fakes.py`) that returns
prepared structured outputs and records the prompts it received; web search is disabled and the workspace is redirected
to a temporary folder (`tests/conftest.py`); training tests use tiny data and one or two Optuna trials. Tests that call
a real LLM are marked `live` and deselected by default. `ruff` lints the whole code base. Two system-level runs complete
the picture: `scripts/evaluate.py` runs the real Orchestrator end to end on five datasets with three seeds (Chapter 7),
and `scripts/make_screenshots.py` drives the running web interface in a real browser (Playwright) through an upload, two
analyses, a what-if simulation and every page.

### 6.2.2 Test Statistics

The 228 test functions expand through parametrisation to **449 test cases**: **447 run offline** and 2 are live LLM
tests. The final offline run gave **447 passed, 2 deselected, 0 failed** in 206.9 s (Table 6.2).

**Table 6.2:** Test inventory

| Test file | Cases | Area |
|---|---:|---|
| `test_foundation.py` | 36 | Settings, schemas, LLM client |
| `test_data_module.py` | 45 | Loader, profiler, preprocessing, splits, Data Agent |
| `test_requirement_and_manager.py` | 36 | Prompts, Requirement Agent, state machine |
| `test_automl.py` | 46 | Model zoo, tuner, trainer, Model Agent |
| `test_step5.py` | 55 | Thresholds, retrieval, Verifier, sandbox, Analysis Agent, re-planning |
| `test_step6.py` | 43 | Explanations, what-if, recommendations, insights, reports, offline runs |
| `test_storage.py` | 6 | Repository and database mirroring |
| `test_api.py` | 39 | Every endpoint and error case |
| `test_ui_client.py` | 18 | UI HTTP client |
| `test_ui_pages.py` | 106 | Every page, the theme and the chart builders |
| `test_evaluation.py` | 17 | Ablation switches, baselines, SR / NPS / CS formulas, perturbations |
| `test_live_llm.py` | 2 | Real LLM smoke tests (only with `-m live`) |
| **Total** | **449** | **447 offline + 2 live** |

### 6.2.3 Code Coverage

Coverage measured with pytest-cov over `idss`, `api`, `ui` and `configs` is **93%** (6,494 statements, 468 not
executed), as shown in Table 6.3.

**Table 6.3:** Code coverage per package

| Package | Statements | Missed | Coverage | Note |
|---|---:|---:|---:|---|
| `idss/core` | 504 | 16 | 96.8% | |
| `idss/agents` | 1,185 | 53 | 95.5% | Lowest: Analysis Agent (84%) |
| `idss/ml` | 1,653 | 87 | 94.7% | |
| `idss/decision` | 668 | 36 | 94.6% | |
| `idss/knowledge` | 160 | 11 | 93.1% | Web search providers 84% (no network in tests) |
| `idss/storage` | 198 | 0 | 100% | |
| `idss/execution` | 168 | 86 | 48.8% | `sandbox.py` 92%; the runner (0%) executes in a child process that coverage does not trace |
| `api` | 498 | 36 | 92.8% | Lowest: background worker (80%) |
| `ui` | 1,408 | 142 | 89.9% | Lowest: Upload Data page (75%) |
| `configs` | 51 | 1 | 98.0% | |
| **Total** | **6,494** | **468** | **93%** | |

### 6.2.4 Manual Testing

Manual testing covered what needs human judgement — whether texts are understandable, charts readable and the flow
natural for a non-programmer. It was carried out during the Week 6–8 testing, as the decision layer, the API and the
user interface became available, and followed the first-run scenario of Appendix A: start the servers, run the churn
example, follow the Run Monitor, read the results and verification, explore the Decision Center and a what-if change,
and download the PDF. The command-line scripts of each increment (`demo_step3.py`, `benchmark_step4.py`,
`demo_step6.py`) served the same purpose for the engine; the saved Step 4 benchmark (`docs/results/step4_benchmark.md`)
is one recorded result.

## 6.3 Test Case Identification and Execution

Table 6.4 lists representative test cases derived from the requirements and risks; each is an automated test whose
function is named in brackets. All passed in the final run.

**Table 6.4:** Test cases

| ID | Module (test function) | Input | Expected output | Actual output | Result |
|---|---|---|---|---|---|
| TC-01 | LLM client (`test_chat_structured_gives_up_after_max_retries`) | LLM always answers "not json at all" | 4 calls (1 + 3 retries), then a validation error with the raw output | As expected | Pass |
| TC-02 | Loader (`test_load_csv_detects_semicolon_delimiter`) | CSV `a;b;c` with two rows | Columns a, b, c; shape (2, 3) | As expected | Pass |
| TC-03 | Split (`test_split_is_stratified_for_classification`) | 1,000 rows, 20% positives | 20% ± 1% positives in every split | As expected | Pass |
| TC-04 | Split (`test_split_time_series_is_chronological`) | 100 ordered rows | 70 / 20 / 10 rows; train before validation before test | As expected | Pass |
| TC-05 | Repair (`test_repair_replaces_unusable_target`, 2 cases) | Target `will_leave` (absent) or `customer_id` (an id) | Target becomes `churn` with an explanatory note | As expected | Pass |
| TC-06 | Orchestrator (`test_irrelevant_question_ends_politely_without_training`) | "Hello! How is your day going?" | INIT → DONE, polite reply, no training | As expected | Pass |
| TC-07 | Orchestrator (`test_revision_limit_fails_the_run`) | Vague question; two vague answers | NEEDS_INPUT, then FAILED after 2 rounds; further answers rejected | As expected | Pass |
| TC-08 | Orchestrator (`test_state_is_persisted_and_resumable`) | Run stopped at PROFILE, resumed by a new instance | `state.json` shows PROFILE; resumed run reaches DONE | As expected | Pass |
| TC-09 | Orchestrator (`test_step_limit_stops_a_looping_run`) | Handler that always returns TRAIN; limit 20 | FAILED with "possible loop" | As expected | Pass |
| TC-10 | Trainer (`test_test_split_is_untouched_until_final_evaluation`) | 240 rows, 2 models | Tuning sees 168 / 48 rows only; test read once; 24 test rows | As expected | Pass |
| TC-11 | Time-series features (`test_time_series_features_use_only_past_values`) | Day-30 sales raised by 1,000 | Features of days ≤ 30 unchanged; day-31 lag-1 equals the new value | As expected | Pass |
| TC-12 | Selection (`test_constraint_violations_are_marked_and_avoided`) | Scores 0.90 at 8 ms/row and 0.80 at 0.5 ms/row; limit 5 ms | Faster model selected; violation recorded | As expected | Pass |
| TC-13 | Model Agent (`test_model_agent_keeps_only_zoo_names`) | LLM ranks "XGBoost", "svm", "Random Forest" | Unknown "svm" dropped with a note; zoo models completed | As expected | Pass |
| TC-14 | Threshold (`test_tune_threshold_finds_a_better_cutoff`) | 15% positives with probabilities mostly below 0.5 | Threshold < 0.5 and higher F1 than at 0.5 | As expected | Pass |
| TC-15 | Verifier (`test_verifier_detects_each_problem`, 8 cases) | One defect at a time (target missed, baseline not beaten, constraint, 20 test rows, test read twice, leakage, over-fitting, …) | Fails with exactly that check and its recommendation | As expected | Pass |
| TC-16 | Re-planning (`test_failed_verification_replans_at_most_once`) | Verifier that always fails | Two PLAN / TRAIN / VERIFY rounds, then DONE flagged not verified; full budget on re-plan | As expected | Pass |
| TC-17 | Explanations (`test_derived_columns_fold_back_to_original_names`) | One-hot and lag / calendar features | Drivers named `region`, `sales (history)`, `date`, `promo` | As expected | Pass |
| TC-18 | What-if (`test_what_if_rejects_invalid_changes`, 6 cases) | Text for a number, % on a category, unknown column, target, empty, bad kind | `WhatIfError` for each | As expected | Pass |
| TC-19 | Counterfactual (`test_counterfactual_lowers_risk_using_actionable_features_only`) | Riskiest record | Below threshold with 1–3 changes to observed values | As expected | Pass |
| TC-20 | Risk tiers (`test_tier_cutoffs_use_threshold_and_percentile`, 2 cases) | 1,000 probabilities; threshold 0.2 or 0.85 | High cut-off 0.6993 or 0.85; counts (300, 500, 200) or (150, 0, 850) | As expected | Pass |
| TC-21 | Grounding (`test_invented_number_triggers_one_regeneration_then_template`) | LLM twice writes "Churn will fall by 37.5% next quarter." | One regeneration naming 37.5%, then the template brief | As expected | Pass |
| TC-22 | Storage (`test_database_failure_never_stops_a_run`) | Repository failing on every call | Run still reaches DONE | As expected | Pass |
| TC-23 | API (`test_second_run_while_busy_returns_409`) | Second run and a what-if during a run | `409 busy`, `409 run_in_progress`; `/health` still answers | As expected | Pass |
| TC-24 | UI (`test_every_page_handles_backend_down`, 8 pages) | Backend stopped | Every page shows "API offline" and the start hint | As expected | Pass |

## 6.4 Security Testing

**Table 6.5:** Security tests (all passed)

| ID | Threat | Test function | Input | Expected and actual result |
|---|---|---|---|---|
| ST-01 | Unsafe LLM-written code | `test_sandbox_blocks_unsafe_code_before_running` (15 cases) | `import os`, `import subprocess`, `open(...)`, `eval(...)`, `__import__('os')`, `().__class__`, `pd.read_csv(...)`, `df.to_csv(...)`, `np.load(...)`, … | Blocked by the static guard in under 1 s, nothing executed |
| ST-02 | Endless loop | `test_sandbox_kills_infinite_loops` | `while True: pass`, 3 s timeout | Process stopped; "Timed out after 3s" |
| ST-03 | Output flooding | `test_sandbox_truncates_huge_output` | `print('x' * 100_000)` | Output cut to 20 KB; result still returned |
| ST-04 | Key leakage to the sandbox | `test_sandbox_child_gets_no_secrets` | `LLM_API_KEY`, `TAVILY_API_KEY` set | Neither present in the child environment |
| ST-05 | Path traversal (charts) | `test_chart_path_traversal_is_blocked` (6 cases) | `charts/..%5C..%5Cstate.json`, `charts/%2E%2E/%2E%2E/state.json`, … | `400` / `404`; `state.json` never returned |
| ST-06 | Path tricks in run ids | `test_run_id_path_tricks_are_rejected` | `..%5C..%5Cworkspace`, `..` | `404` |
| ST-07 | Malicious file names | `test_sanitize_filename` (6 cases) | `../../etc/passwd.csv`, a 300-character name, … | `passwd.csv`; names shortened to 80 characters |
| ST-08 | Bad or oversized uploads | `test_upload_rejects_unsupported_extensions`, `test_upload_rejects_files_over_the_limit` | `data.csv.exe`; files above the limit | `415`; `413` checked before and while reading, partial file deleted |
| ST-09 | Key exposure by the API | `test_health_reports_llm_flag_but_never_the_key` | Secret key configured | `/health` shows only `llm_configured: true` |
| ST-10 | Raw data sent to the LLM | `test_prompts_include_profile_summary_but_never_raw_values` | Churn profile in the parsing prompt | Column summary present; raw values such as `CUST-00001` absent |

## 6.5 Summary

IDSS was tested with black-box techniques through the API and the UI and with white-box techniques inside the engine.
The automated suite has 449 test cases, of which all 447 offline cases passed, covering 93% of 6,494 statements.
Twenty-four representative test cases and ten security tests — sandbox, path traversal, uploads and key leakage — were
documented with their inputs and results, and system-level runs and manual walk-throughs completed the testing.

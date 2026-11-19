"""Smoke tests of the Streamlit pages with streamlit.testing.v1.AppTest and a fake API client (no server).

Each page is opened through ui/app.py (st.navigation), with data, with empty data, with the backend down, and
for runs in every state; a few interactions (examples, start, answer, what-if, ask) check the calls made.
"""

import base64
import copy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from ui import api_client
from ui.api_client import ApiClientError, BackendUnavailable
from ui import components, embedded_api

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"
PAGES = ["pages/1_Home.py", "pages/2_Upload_Data.py", "pages/3_New_Analysis.py", "pages/4_Run_Monitor.py",
         "pages/5_Results.py", "pages/6_Decision_Center.py", "pages/7_Ask_Your_Data.py", "pages/8_Reports.py"]
TIMEOUT_S = 60
TS = "2026-10-05T10:00:00Z"
PNG = base64.b64decode(  # 1x1 transparent PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

# --------------------------------------------------------------------------- canned API responses

PROFILE: dict[str, Any] = {
    "n_rows": 100, "n_cols": 6, "duplicate_rows": 0, "memory_mb": 0.01,
    "suggested_target": "churn", "suggested_task_type": "tabular_classification", "time_column": None,
    "warnings": ["Column 'tenure' has 2.0% missing values."],
    "columns": [
        {"name": "customer_id", "dtype": "str", "semantic_type": "id", "missing_pct": 0.0, "n_unique": 100,
         "sample_values": ["C1", "C2"], "stats": {}},
        {"name": "contract", "dtype": "str", "semantic_type": "categorical", "missing_pct": 0.0, "n_unique": 4,
         "sample_values": ["Monthly"], "stats": {"top_values": {"Monthly": 60, "Yearly": 25, "Two year": 10}}},
        {"name": "senior", "dtype": "int64", "semantic_type": "boolean", "missing_pct": 0.0, "n_unique": 2,
         "sample_values": [0, 1], "stats": {"top_values": {"0": 80, "1": 20}}},
        {"name": "tenure", "dtype": "int64", "semantic_type": "numeric", "missing_pct": 2.0, "n_unique": 50,
         "sample_values": [1, 5], "stats": {"mean": 20.0, "std": 5.0, "min": 0.0, "median": 18.0, "max": 70.0, "integer": True}},
        {"name": "charges", "dtype": "float64", "semantic_type": "numeric", "missing_pct": 0.0, "n_unique": 90,
         "sample_values": [10.5], "stats": {"mean": 50.0, "std": 10.0, "min": 10.0, "median": 49.5, "max": 110.0, "integer": False}},
        {"name": "churn", "dtype": "str", "semantic_type": "boolean", "missing_pct": 0.0, "n_unique": 2,
         "sample_values": ["No", "Yes"], "stats": {"top_values": {"No": 70, "Yes": 30}}},
    ],
}
DATASET = {"id": "ds1", "filename": "churn.csv", "rows": 100, "cols": 6, "uploaded_at": TS}


def summary(run_id: str, state: str, **extra: Any) -> dict[str, Any]:
    finished = state == "DONE"
    base = {
        "run_id": run_id, "dataset_id": "ds1", "question": f"Question for {run_id}?", "state": state,
        "progress": {"DONE": 100, "FAILED": 100, "TRAIN": 30, "NEEDS_INPUT": 10}.get(state, 0),
        "active": state == "TRAIN", "task_type": "tabular_classification" if finished else None,
        "best_model": "logistic_regression" if finished else None, "primary_metric": "f1" if finished else None,
        "metric_value": 0.62 if finished else None, "verification_passed": True if finished else None,
        "created_at": TS, "updated_at": TS,
        "report_pdf_url": f"/runs/{run_id}/report.pdf" if finished else None,
        "report_md_url": f"/runs/{run_id}/report.md" if finished else None,
    }
    base.update(extra)
    return base


def detail(run: dict[str, Any], **extra: Any) -> dict[str, Any]:
    finished = run["state"] == "DONE"
    events = [{"ts": TS, "from_state": None, "to_state": "INIT", "message": "Run created."},
              {"ts": TS, "from_state": "INIT", "to_state": run["state"], "message": f"Now {run['state']} <b>ok</b>."}]
    result = {
        **run, "message": "Run complete." if finished else None, "target": "churn" if finished else None,
        "test_metrics": {"f1": 0.62, "accuracy": 0.81, "confusion_matrix": [[50, 5], [6, 9]]} if finished else {},
        "decision_threshold": 0.32 if finished else None,
        "verification": {"passed": True, "checks": [
            {"name": "metric_targets", "passed": True, "detail": "No targets were set."},
            {"name": "beats_baseline", "passed": True, "detail": "0.62 vs 0.0"},
        ], "llm_verdict": None, "llm_concerns": [], "recommendations": []} if finished else None,
        "pending_questions": [], "answers": [], "errors": [], "events": events,
    }
    result.update(extra)
    return result


LEADERBOARD = {"run_id": "run-done", "primary_metric": "f1", "direction": "maximize", "best_model": "logistic_regression",
               "entries": [
                   {"rank": 1, "model": "logistic_regression", "status": "ok", "valid_score": 0.6, "train_time_s": 0.5,
                    "inference_ms_per_row": 0.01, "model_size_kb": 3.2, "n_trials": 15, "violations": [], "selected": True},
                   {"rank": None, "model": "xgboost", "status": "failed", "valid_score": None, "n_trials": 0,
                    "violations": [], "selected": False, "error": "boom"},
               ]}
GLOBAL = {"method": "shap_linear", "output": "probability that churn = 'Yes'", "n_samples": 100, "notes": ["A note."],
          "features": [{"feature": "contract", "importance": 1.0, "share": 0.5, "direction": "mixed",
                        "detail": "Highest for Monthly."},
                       {"feature": "tenure", "importance": 0.6, "share": 0.3, "direction": "decreases", "detail": ""}]}
EXPLANATION = {"global_explanation": GLOBAL, "local_examples": [], "clusters": [], "forecast": None,
               "charts": {"feature_importance": "/runs/run-done/charts/feature_importance.png",
                          "confusion_matrix": "/runs/run-done/charts/confusion_matrix.png"}}
FORECAST = {"horizon": 3, "interval": "±1.96 × validation RMSE", "assumptions": ["Other columns repeat last season."],
            "points": [{"timestamp": f"2024-01-0{i}", "prediction": 10.0 + i, "lower": 8.0 + i, "upper": 12.0 + i}
                       for i in range(1, 4)]}
CLUSTERS = [{"cluster": 0, "size": 60, "share": 0.6, "summary": {"tenure": 10.5, "contract": "Monthly"},
             "distinguishing": ["Short tenure"]},
            {"cluster": 1, "size": 40, "share": 0.4, "summary": {"tenure": 40.0, "contract": "Two year"},
             "distinguishing": ["Long tenure"]}]
WHAT_IF = {"output": "probability that churn = 'Yes'", "before": 0.9, "after": 0.8, "delta": -0.1, "delta_pct": -11.1,
           "changes": {"contract": "Two year"}, "warnings": []}
RECOMMENDATIONS = {
    "actionable_features": ["contract", "charges"],
    "recommendations": [
        {"action": "Prioritise the 30 high-risk records.", "rationale": "They have the highest probability.",
         "expected_impact": 18.5, "impact_unit": "expected 'Yes' cases", "confidence": "medium",
         "evidence": {"high_tier_count": 30, "threshold": 0.32}},
        {"action": "Move high-risk records to contract = 'Two year'.", "rationale": "Counterfactuals agree.",
         "expected_impact": -0.45, "impact_unit": "change in probability", "confidence": "high", "evidence": {}},
    ],
    "risk_tiers": [
        {"name": "High", "min_probability": 0.46, "max_probability": 1.0, "count": 30, "share": 0.3,
         "avg_probability": 0.65, "top_drivers": ["contract", "tenure"]},
        {"name": "Medium", "min_probability": 0.32, "max_probability": 0.46, "count": 15, "share": 0.15,
         "avg_probability": 0.4, "top_drivers": ["tenure"]},
        {"name": "Low", "min_probability": 0.0, "max_probability": 0.32, "count": 55, "share": 0.55,
         "avg_probability": 0.1, "top_drivers": []},
    ],
    "tier_cutoffs": {"high": 0.46, "medium": 0.32, "percentile": 70, "percentile_value": 0.46},
    "counterfactual_example": {"row_index": 5, "probability_before": 0.9, "probability_after": 0.2, "threshold": 0.32,
                               "success": True, "changes": [{"feature": "contract", "before": "Monthly", "after": "Two year"}]},
    "what_if_example": WHAT_IF, "notes": ["Recommendations assume the model's associations hold."],
}
INSIGHT = {"executive_summary": "Goal: find churners. The best model scored f1 = 0.62.",
           "key_findings": ["contract explains 50%."], "recommended_actions": ["Prioritise high-risk records."],
           "risks_and_limitations": ["Associations, not causes."], "next_steps": ["Run a pilot."],
           "generated_by": "template", "grounding": None}
REPORT_MD = "# IDSS report\n\nSome text.\n\n![feature_importance](charts/feature_importance.png)\n\n## End\n"


# --------------------------------------------------------------------------- fake client


class FakeClient:
    """Implements the ui.api_client.IDSSClient methods the pages use, from in-memory data."""

    base_url = "http://localhost:8000"

    def __init__(self, *, empty: bool = False, down: bool = False, llm: bool = False, demo: bool = False) -> None:
        self.down, self.llm, self.demo = down, llm, demo
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.datasets = [] if empty else [DATASET]
        runs = [] if empty else [
            summary("run-done", "DONE"),
            summary("run-ts", "DONE", task_type="time_series_forecasting", best_model="ridge", primary_metric="rmse",
                    metric_value=12.5),
            summary("run-cluster", "DONE", task_type="tabular_clustering", best_model="kmeans",
                    primary_metric="silhouette", metric_value=0.41),
            summary("run-train", "TRAIN"),
            summary("run-needs", "NEEDS_INPUT"),
            summary("run-failed", "FAILED"),
            summary("run-noresult", "DONE", task_type=None, best_model=None, primary_metric=None, metric_value=None,
                    verification_passed=None, report_pdf_url=None, report_md_url=None),
        ]
        self.runs = runs
        self.details = {run["run_id"]: detail(run) for run in runs}
        if not empty:
            self.details["run-needs"]["pending_questions"] = ["Which column should be predicted?"]
            self.details["run-failed"]["errors"] = ["Training failed: every candidate crashed."]
            self.details["run-noresult"]["message"] = "This question is not about your data."
            self.details["run-ts"].update(target="sales", decision_threshold=None)
            self.details["run-cluster"].update(target=None, decision_threshold=None)

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if self.down:
            raise BackendUnavailable("http://localhost:8000")

    def called(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(args, kwargs) for call, args, kwargs in self.calls if call == name]

    def _run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.details:
            raise ApiClientError(f"Run '{run_id}' was not found.", status_code=404, code="not_found")
        return self.details[run_id]

    def _finished(self, run_id: str, what: str) -> None:
        run = self._run(run_id)
        if not run.get("best_model"):
            raise ApiClientError(f"Run {run_id} has no {what} yet (state {run['state']}).", status_code=409, code="not_ready")

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def health(self) -> dict[str, Any]:
        self._record("health")
        return {"status": "ok", "llm_configured": self.llm, "version": "1.0.0", "public_demo": self.demo,
                "max_upload_mb": 10.0 if self.demo else 50.0}

    def list_datasets(self, limit: int = 100) -> list[dict[str, Any]]:
        self._record("list_datasets")
        return copy.deepcopy(self.datasets)

    def upload_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        self._record("upload_dataset", filename, content)
        return {**DATASET, "filename": filename, "profile": {}}

    def dataset_profile(self, dataset_id: str) -> dict[str, Any]:
        self._record("dataset_profile", dataset_id)
        return copy.deepcopy(PROFILE)

    def create_run(self, dataset_id: str, question: str, mode: str = "fast",
                   actionable_features: list[str] | None = None) -> dict[str, Any]:
        self._record("create_run", dataset_id, question, mode, actionable_features)
        return {"run_id": "run-train", "state": "INIT", "status_url": "/runs/run-train"}

    def list_runs(self, limit: int = 50, dataset_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
        self._record("list_runs")
        return copy.deepcopy(self.runs)

    def get_run(self, run_id: str) -> dict[str, Any]:
        self._record("get_run", run_id)
        return copy.deepcopy(self._run(run_id))

    def answer(self, run_id: str, answer: str) -> dict[str, Any]:
        self._record("answer", run_id, answer)
        return {"run_id": run_id, "state": "PARSE", "status_url": f"/runs/{run_id}"}

    def leaderboard(self, run_id: str) -> dict[str, Any]:
        self._record("leaderboard", run_id)
        self._finished(run_id, "trained models")
        return {**copy.deepcopy(LEADERBOARD), "run_id": run_id}

    def explanations(self, run_id: str) -> dict[str, Any]:
        self._record("explanations", run_id)
        self._finished(run_id, "explanations")
        result = {**copy.deepcopy(EXPLANATION), "run_id": run_id}
        if run_id == "run-ts":
            result["forecast"] = copy.deepcopy(FORECAST)
        if run_id == "run-cluster":
            result["clusters"] = copy.deepcopy(CLUSTERS)
        return result

    def recommendations(self, run_id: str) -> dict[str, Any]:
        self._record("recommendations", run_id)
        self._finished(run_id, "recommendations")
        result = {**copy.deepcopy(RECOMMENDATIONS), "run_id": run_id}
        if run_id != "run-done":
            result.update(risk_tiers=[], tier_cutoffs=None, counterfactual_example=None)
        return result

    def insight(self, run_id: str) -> dict[str, Any]:
        self._record("insight", run_id)
        self._finished(run_id, "decision brief")
        return copy.deepcopy(INSIGHT)

    def what_if(self, run_id: str, changes: dict[str, Any], *, base_row: dict[str, Any] | None = None,
                segment: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self._record("what_if", run_id, changes, base_row=base_row, segment=segment)
        if segment is not None:
            return {"output": WHAT_IF["output"], "filters": segment, "changes": changes, "count": 42,
                    "avg_before": 0.5, "avg_after": 0.4, "delta": -0.1, "delta_pct": -20.0, "warnings": []}
        return {**WHAT_IF, "changes": changes}

    def ask(self, run_id: str, question: str) -> dict[str, Any]:
        self._record("ask", run_id, question)
        return {"question": question, "ok": True, "answer": "There are 100 rows.", "code": "result = len(df)",
                "result": 100, "stdout": "", "attempts": 1, "error": None, "asked_at": TS}

    def get_bytes(self, path: str) -> bytes:
        self._record("get_bytes", path)
        return PNG

    def report_pdf(self, run_id: str) -> bytes:
        self._record("report_pdf", run_id)
        return b"%PDF-1.4 fake"

    def report_markdown(self, run_id: str) -> str:
        self._record("report_markdown", run_id)
        return REPORT_MD

    def chart(self, run_id: str, name: str) -> bytes:
        self._record("chart", run_id, name)
        return PNG


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def use_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeClient]]:
    def install(**kwargs: Any) -> FakeClient:
        fake = FakeClient(**kwargs)
        monkeypatch.setattr(api_client, "get_client", lambda: fake)
        return fake
    yield install


def open_page(page: str, *, run_id: str | None = None, state: dict[str, Any] | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception, at.exception
    if run_id is not None:
        at.session_state["current_run_id"] = run_id
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.switch_page(page).run()
    return at


def texts(at: AppTest) -> str:
    """All visible text of the page and its sidebar (markdown, captions, alerts, metrics) as one string."""
    parts: list[str] = []
    for root in (at, at.sidebar):
        for group in (root.markdown, root.caption, root.info, root.warning, root.error, root.success, root.subheader):
            parts += [str(element.value) for element in group]
        parts += [f"{metric.label} {metric.value}" for metric in root.metric]
    return "\n".join(parts)


def assert_ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------- smoke tests


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_data(use_client, page: str) -> None:
    use_client()
    at = open_page(page, run_id="run-done")
    assert_ok(at)
    assert "API online" in texts(at)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_empty_state(use_client, page: str) -> None:
    use_client(empty=True)
    at = open_page(page)
    assert_ok(at)
    if page != "pages/1_Home.py":
        assert at.info, "an empty page should explain what to do next"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_handles_backend_down(use_client, page: str) -> None:
    use_client(down=True)
    at = open_page(page)
    assert_ok(at)
    page_text = texts(at)
    assert "API offline" in page_text
    assert "Backend not running – start scripts/run_api.ps1" in page_text


@pytest.mark.parametrize("run_id", ["run-train", "run-needs", "run-failed", "run-noresult", "run-ts", "run-cluster"])
@pytest.mark.parametrize("page", PAGES[3:])
def test_run_pages_handle_every_run_state(use_client, page: str, run_id: str) -> None:
    use_client(llm=True)
    at = open_page(page, run_id=run_id)
    assert_ok(at)


# --------------------------------------------------------------------------- page content and interactions


def test_home_lists_recent_runs(use_client) -> None:
    use_client()
    at = open_page("pages/1_Home.py")
    assert_ok(at)
    assert "Upload" in texts(at) or at.markdown
    assert len(at.dataframe) == 1
    assert "run-done" in at.dataframe[0].value["Run"].tolist()


def test_upload_page_shows_profile(use_client) -> None:
    fake = use_client()
    at = open_page("pages/2_Upload_Data.py")
    assert_ok(at)
    page_text = texts(at)
    assert "churn" in page_text and "Classification" in page_text
    assert any("missing values" in w.value for w in at.warning)
    assert fake.called("dataset_profile") == [(("ds1",), {})]
    columns = at.dataframe[0].value
    assert list(columns["Type"]) == ["id", "categorical", "boolean", "numeric", "numeric", "boolean"]


def test_new_analysis_example_and_start(use_client) -> None:
    fake = use_client()
    at = open_page("pages/3_New_Analysis.py")
    assert_ok(at)
    assert any("No LLM is configured" in i.value for i in at.info)
    at.button(key="example_Customer churn").click().run()
    assert "churn" in at.text_area(key="new_question").value
    at.radio(key="new_mode").set_value("full")
    at.multiselect(key="new_actionable").set_value(["contract"])
    at.button(key="start_run").click().run()
    assert_ok(at)
    (args, _), = fake.called("create_run")
    assert args[0] == "ds1" and "churn" in args[1] and args[2] == "full" and args[3] == ["contract"]
    assert at.session_state["current_run_id"] == "run-train"


def test_new_analysis_requires_a_question(use_client) -> None:
    fake = use_client()
    at = open_page("pages/3_New_Analysis.py")
    at.button(key="start_run").click().run()
    assert_ok(at)
    assert any("type a business question" in w.value for w in at.warning)
    assert not fake.called("create_run")


def test_monitor_answers_pending_questions(use_client) -> None:
    fake = use_client()
    at = open_page("pages/4_Run_Monitor.py", run_id="run-needs")
    assert_ok(at)
    assert any("Which column should be predicted?" in i.value for i in at.info)
    at.text_area[0].input("Predict churn")
    next(b for b in at.button if b.label == "Send answer").click().run()
    assert fake.called("answer") == [(("run-needs", "Predict churn"), {})]


def test_monitor_shows_failure_reason_and_escapes_events(use_client) -> None:
    use_client()
    at = open_page("pages/4_Run_Monitor.py", run_id="run-failed")
    assert_ok(at)
    assert any("every candidate crashed" in e.value for e in at.error)
    timeline = next(m.value for m in at.markdown if "idss-timeline" in m.value)
    assert "&lt;b&gt;ok&lt;/b&gt;" in timeline  # event messages are escaped, never rendered as HTML


def test_results_page_tables(use_client) -> None:
    fake = use_client()
    at = open_page("pages/5_Results.py", run_id="run-done")
    assert_ok(at)
    page_text = texts(at)
    assert "logistic regression" in page_text and "✅ Passed" in page_text and "0.32" in page_text
    frames = [frame.value for frame in at.dataframe]
    assert any("Model" in frame.columns and "xgboost" in frame["Model"].tolist() for frame in frames)
    assert any("Check" in frame.columns for frame in frames)
    assert {args[0] for args, _ in fake.called("get_bytes")} == set(EXPLANATION["charts"].values())


def test_results_page_waits_for_training(use_client) -> None:
    fake = use_client()
    at = open_page("pages/5_Results.py", run_id="run-train")
    assert_ok(at)
    assert any("training finishes" in i.value for i in at.info)
    assert not fake.called("leaderboard")


def test_decision_center_brief(use_client) -> None:
    use_client()
    at = open_page("pages/6_Decision_Center.py", run_id="run-done")
    assert_ok(at)
    page_text = texts(at)
    assert "Goal: find churners." in page_text
    assert "Prioritise the 30 high-risk records." in page_text
    assert "medium confidence" in page_text and "high confidence" in page_text
    assert any("Tier" in frame.value.columns for frame in at.dataframe)


def test_decision_center_record_what_if(use_client) -> None:
    fake = use_client()
    at = open_page("pages/6_Decision_Center.py", run_id="run-done")
    at.number_input(key="wi_run-done_charges").set_value(80.0)
    at.selectbox(key="wi_run-done_contract").set_value("Two year")
    simulate = next(b for b in at.button if b.label == "Simulate")
    simulate.click().run()
    assert_ok(at)
    (args, kwargs), = fake.called("what_if")
    assert args[0] == "run-done"
    assert args[1] == {"charges": 80.0, "contract": "Two year"}
    assert kwargs["base_row"]["tenure"] == 18 and kwargs["base_row"]["senior"] == 0
    assert "churn" not in kwargs["base_row"]
    assert any(metric.label == "After" for metric in at.metric)


def test_decision_center_what_if_needs_a_change(use_client) -> None:
    fake = use_client()
    at = open_page("pages/6_Decision_Center.py", run_id="run-done")
    next(b for b in at.button if b.label == "Simulate").click().run()
    assert_ok(at)
    assert any("Change at least one input" in i.value for i in at.info)
    assert not fake.called("what_if")


def test_decision_center_segment_simulator(use_client) -> None:
    fake = use_client()
    at = open_page("pages/6_Decision_Center.py", run_id="run-done")
    at.selectbox(key="seg_col_run-done").set_value("tenure").run()
    at.selectbox(key="seg_op_run-done_num").set_value("<").run()
    at.number_input(key="seg_val_run-done_tenure_<").set_value(12.0)
    at.button(key="seg_add_run-done").click().run()
    assert_ok(at)
    at.multiselect(key="seg_changes_run-done").set_value(["charges"]).run()
    at.button(key="seg_run_run-done").click().run()
    assert_ok(at)
    (args, kwargs), = fake.called("what_if")
    assert kwargs["segment"] == [{"column": "tenure", "op": "<", "value": 12.0}]
    assert args[1] == {"charges": {"pct": 10.0}}
    assert any(metric.label == "Records in segment" and metric.value == "42" for metric in at.metric)


def test_ask_page_chat(use_client) -> None:
    fake = use_client(llm=True)
    at = open_page("pages/7_Ask_Your_Data.py", run_id="run-done")
    assert_ok(at)
    at.chat_input(key="ask_input").set_value("How many rows?").run()
    assert_ok(at)
    assert fake.called("ask") == [(("run-done", "How many rows?"), {})]
    assert "There are 100 rows." in texts(at)
    assert any("result = len(df)" in code.value for code in at.code)
    assert len(at.session_state["ask_history"]["run-done"]) == 2


def test_ask_page_without_llm_is_disabled(use_client) -> None:
    use_client(llm=False)
    at = open_page("pages/7_Ask_Your_Data.py", run_id="run-done")
    assert_ok(at)
    assert any("needs an LLM" in i.value for i in at.info)
    assert at.chat_input(key="ask_input").disabled


# --------------------------------------------------------------------------- public demo (IDSS_PUBLIC_DEMO)

DEMO_BANNER = "Public demo – do not upload private data"


@pytest.mark.parametrize("page", PAGES)
def test_public_demo_banner_on_every_page(use_client, page: str) -> None:
    use_client(demo=True)
    at = open_page(page, run_id="run-done")
    assert_ok(at)
    assert any(DEMO_BANNER in w.value and "10 MB" in w.value for w in at.warning)


def test_no_public_demo_banner_by_default(use_client) -> None:
    use_client()
    at = open_page("pages/1_Home.py")
    assert_ok(at)
    assert DEMO_BANNER not in texts(at)


def test_public_demo_upload_page_shows_the_10_mb_limit(use_client) -> None:
    use_client(demo=True)
    at = open_page("pages/2_Upload_Data.py")
    assert_ok(at)
    assert "max 10 MB" in texts(at) and "max 50 MB" not in texts(at)


def test_public_demo_new_analysis_is_fast_only(use_client) -> None:
    fake = use_client(demo=True)
    at = open_page("pages/3_New_Analysis.py", state={"new_mode": "full"})  # e.g. left over from before
    assert_ok(at)
    assert len(at.radio(key="new_mode").options) == 1 and at.radio(key="new_mode").value == "fast"
    assert any("fast mode only" in c.value for c in at.caption)
    at.button(key="example_Customer churn").click().run()
    at.button(key="start_run").click().run()
    assert_ok(at)
    (args, _), = fake.called("create_run")
    assert args[2] == "fast"


def test_public_demo_disables_ask_your_data(use_client) -> None:
    fake = use_client(llm=True, demo=True)
    at = open_page("pages/7_Ask_Your_Data.py", run_id="run-done")
    assert_ok(at)
    assert any("turned off in the public demo" in i.value for i in at.info)
    assert at.chat_input(key="ask_input").disabled
    assert not fake.called("ask")


def test_reports_page(use_client) -> None:
    fake = use_client()
    at = open_page("pages/8_Reports.py", run_id="run-done")
    assert_ok(at)
    assert fake.called("report_pdf") and fake.called("report_markdown")
    assert fake.called("chart") == [(("run-done", "feature_importance.png"), {})]
    assert "IDSS report" in texts(at)
    frame = at.dataframe[0].value
    assert frame["PDF"].iloc[0].startswith("http://localhost:8000/runs/")
    assert "run-noresult" not in frame["Run"].tolist()


def test_reports_page_embedded_mode_has_no_backend_links(use_client, monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-in backend listens on the server's 127.0.0.1, which a visitor's browser cannot open: the table
    drops its link columns and the download buttons (fetched server side) remain."""
    fake = use_client()
    monkeypatch.setenv(embedded_api.EMBEDDED_ENV, "1")
    monkeypatch.setattr(embedded_api, "ensure_backend", lambda: None)  # the fake client stands in for the server
    at = open_page("pages/8_Reports.py", run_id="run-done")
    assert_ok(at)
    assert fake.called("report_pdf") and fake.called("report_markdown")
    frame = at.dataframe[0].value
    assert "PDF" not in frame.columns and "Markdown" not in frame.columns
    assert "download it with the buttons" in texts(at)


# --------------------------------------------------------------------------- pure helpers


def test_typical_row_and_changes() -> None:
    row = components.typical_row(PROFILE, "churn")
    assert row == {"customer_id": "C1", "contract": "Monthly", "senior": 0, "tenure": 18, "charges": 49.5}
    assert components.diff_changes(row, {**row, "tenure": 18.0, "charges": 60.0}) == {"charges": 60.0}
    senior = PROFILE["columns"][2]
    assert components.coerce_value(senior, "1") == 1
    assert components.category_options(senior) == [0, 1]
    assert [c["name"] for c in components.changeable_columns(PROFILE, "churn")] == ["contract", "senior", "tenure", "charges"]


# --------------------------------------------------------------------------- Step 9: theme, Demo Mode, charts


def plotly_charts(at: AppTest) -> list[dict[str, Any]]:
    """The Plotly figures (as dicts) rendered on the page."""
    import json
    return [json.loads(element.proto.spec) for element in at.get("plotly_chart")]


@pytest.mark.parametrize("mode", [{}, {"empty": True}, {"down": True}])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_has_hero_and_footer(use_client, page: str, mode: dict[str, bool]) -> None:
    use_client(**mode)
    at = open_page(page, run_id=None if mode else "run-done")
    assert_ok(at)
    page_text = texts(at)
    assert 'class="idss-hero"' in page_text
    assert "G H Raisoni College of Engineering and Management, Jalgaon" in page_text
    assert "Intelligent Decision Support System Using Generative AI and Machine Learning" in page_text


def test_home_kpi_strip(use_client) -> None:
    use_client()
    at = open_page("pages/1_Home.py")
    assert_ok(at)
    kpis = {metric.label: metric.value for metric in at.metric}
    # 3 successful runs (run-done f1 0.62, run-ts rmse, run-cluster silhouette 0.41): RMSE is not averaged
    assert kpis == {"Total datasets": "1", "Total runs": "7", "Successful runs": "3", "Avg primary metric": "0.515"}


def test_home_kpi_strip_empty(use_client) -> None:
    use_client(empty=True)
    at = open_page("pages/1_Home.py")
    assert_ok(at)
    assert {metric.label: metric.value for metric in at.metric}["Avg primary metric"] == "—"


def test_home_demo_mode_uploads_sample_and_starts_fast_run(use_client) -> None:
    fake = use_client()
    at = open_page("pages/1_Home.py")
    at.button(key="demo_mode").click().run()
    assert_ok(at)
    (args, _), = fake.called("upload_dataset")
    assert args[0] == "customer_churn.csv" and args[1].startswith(b"customer_id")
    (args, _), = fake.called("create_run")
    assert args[0] == "ds1" and "churn" in args[1] and args[2] == "fast"
    assert at.session_state["current_run_id"] == "run-train"
    assert at.session_state["current_dataset_id"] == "ds1"


def test_home_demo_mode_backend_down(use_client) -> None:
    fake = use_client(down=True)
    at = open_page("pages/1_Home.py")
    at.button(key="demo_mode").click().run()
    assert_ok(at)
    assert not fake.called("create_run")
    assert "current_run_id" not in at.session_state or at.session_state["current_run_id"] is None


def test_results_and_decision_pages_render_plotly_charts(use_client) -> None:
    use_client()
    results = open_page("pages/5_Results.py", run_id="run-done")
    assert_ok(results)
    names = [trace.get("name") for figure in plotly_charts(results) for trace in figure["data"]]
    assert len(plotly_charts(results)) == 2  # leaderboard + key drivers
    assert "Mixed / non-linear effect" in names and "Higher value lowers the output" in names
    decisions = open_page("pages/6_Decision_Center.py", run_id="run-done")
    assert_ok(decisions)
    pies = [trace for figure in plotly_charts(decisions) for trace in figure["data"] if trace["type"] == "pie"]
    assert len(pies) == 1 and pies[0]["values"] == [30, 15, 55]
    assert "High risk" in texts(decisions)  # tier chips


def test_forecast_and_cluster_charts_on_decision_center(use_client) -> None:
    use_client()
    at = open_page("pages/6_Decision_Center.py", run_id="run-ts")
    assert_ok(at)
    traces = [trace for figure in plotly_charts(at) for trace in figure["data"]]
    assert any(trace.get("fill") == "tonexty" for trace in traces)  # shaded interval
    assert any(trace.get("name") == "Forecast" for trace in traces)
    at = open_page("pages/6_Decision_Center.py", run_id="run-cluster")
    assert_ok(at)
    bars = [trace for figure in plotly_charts(at) for trace in figure["data"] if trace["type"] == "bar"]
    assert bars and list(bars[0]["y"]) == [60, 40]


def test_chart_builders() -> None:
    from ui import charts
    from ui.styles import BLUE, GREEN, ORANGE, PURPLE, YELLOW

    board = charts.leaderboard(LEADERBOARD)
    bar = board.data[0]
    assert list(bar.y) == ["★ logistic_regression"]  # the failed candidate has no score
    assert list(bar.marker.color) == [PURPLE]
    minimize = charts.leaderboard({"direction": "minimize", "entries": [
        {"model": "a", "valid_score": 2.0, "selected": False}, {"model": "b", "valid_score": 1.0, "selected": True}]})
    assert list(minimize.data[0].y) == ["a", "★ b"]  # best (lowest) drawn last = on top
    assert list(minimize.data[0].marker.color) == [BLUE, PURPLE]
    assert charts.leaderboard({"entries": []}) is None

    drivers = charts.drivers(GLOBAL["features"])
    colors = {trace.name: trace.marker.color for trace in drivers.data}
    assert colors == {"Higher value lowers the output": GREEN, "Mixed / non-linear effect": PURPLE}
    assert list(drivers.layout.yaxis.categoryarray) == ["tenure", "contract"]  # largest share on top
    assert charts.drivers([]) is None

    donut = charts.tier_donut(RECOMMENDATIONS["risk_tiers"])
    assert list(donut.data[0].marker.colors) == [ORANGE, YELLOW, GREEN]
    assert charts.tier_donut([]) is None

    forecast = charts.forecast(FORECAST["points"])
    assert [trace.name for trace in forecast.data][1:] == ["Uncertainty band", "Forecast"]
    assert charts.forecast([]) is None

    assert list(charts.cluster_sizes(CLUSTERS).data[0].text) == ["60 · 60%", "40 · 40%"]
    assert charts.cluster_sizes([]) is None

    what_if = charts.before_after(0.9, 0.8, "probability that churn = 'Yes'")
    assert list(what_if.data[0].text) == ["90.0%", "80.0%"]
    assert list(charts.before_after(1500.0, 1650.5, "sales").data[0].text) == ["1,500", "1,650"]
    assert charts.before_after("Yes", "No", "class") is None


def test_zebra_table_formats_cells() -> None:
    import pandas as pd

    frame = pd.DataFrame({"Model": ["a", "b"], "Score": [0.6, None], "Missing %": [2.0, 12.25]})
    styler = components.zebra(frame, formats={"Missing %": "{:.1f}%"})
    html_text = styler.to_html()
    assert "0.6<" in html_text and "—" in html_text and "12.2%" in html_text
    assert components.ZEBRA.lower() in html_text.lower()
    assert components.zebra(frame.iloc[0:0]) is not None


@pytest.mark.parametrize(("state", "events", "expected"), [
    ("TRAIN", [], {"Understand": "done", "Plan": "done", "Train": "current", "Verify": ""}),
    ("NEEDS_INPUT", [], {"Understand": "waiting", "Profile": ""}),
    ("DONE", [], {"Understand": "done", "Done": "done"}),
    ("FAILED", [{"from_state": "TRAIN", "to_state": "FAILED"}], {"Plan": "done", "Train": "failed", "Verify": ""}),
])
def test_pipeline_stepper(monkeypatch: pytest.MonkeyPatch, state: str, events: list[dict[str, Any]],
                          expected: dict[str, str]) -> None:
    import re
    captured: list[str] = []
    monkeypatch.setattr(components.st, "markdown", lambda body, **_: captured.append(body))
    components.pipeline(state, events)
    stages = dict((label, kind) for kind, label in re.findall(r'idss-stage ([a-z]*)">. ([A-Za-z]+)<', captured[0]))
    for label, kind in expected.items():
        assert stages[label] == kind, (label, stages)

"""API tests (Step 7A): FastAPI TestClient, a temporary workspace and a fake LLM. No network calls.

One module-wide app uploads a tiny dataset and trains one run (1 model x 2 trials); most tests read that
run. Runs that need a different fake LLM use their own app on the same workspace.
"""

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from api.main import create_app, sanitize_filename
from configs.settings import get_settings
from idss.agents.analysis_agent import AnalysisCode
from idss.agents.manager import Orchestrator, RunState
from idss.agents.requirement_agent import RelevanceCheck
from idss.core.schemas import TrainingBudget, UserRequirement
from idss.storage.db import init_db
from idss.storage.repository import get_repository
from tests.fakes import FakeLLM
from tests.test_automl import classification_df, make_requirement

TINY = TrainingBudget(max_candidates=1, n_trials_per_model=2, timeout_per_model_s=30, total_timeout_s=60)
POLL_TIMEOUT_S = 180
QUESTION = "Which customers will say yes, and what should we do about it?"
FINISHED = ("DONE", "FAILED", "NEEDS_INPUT")


@dataclass
class Env:
    client: TestClient
    workspace: Path
    df: pd.DataFrame
    dataset_id: str
    run_id: str
    llm: FakeLLM


def fake_llm(overrides: dict[type[BaseModel], Any] | None = None) -> FakeLLM:
    responses: dict[type[BaseModel], Any] = {
        RelevanceCheck: RelevanceCheck(relevant=True),
        UserRequirement: make_requirement("tabular_classification", "target"),
        AnalysisCode: AnalysisCode(code="result = int(len(df))", explanation="Number of rows."),
    }
    responses.update(overrides or {})
    return FakeLLM(responses)  # every other schema raises LLMError -> deterministic fallbacks


def make_client(llm: Any = None, **kwargs: Any) -> TestClient:
    factory: Callable[[], Any] = (lambda: llm) if llm is not None else (lambda: None)
    return TestClient(create_app(llm_factory=factory, budget=TINY, **kwargs))


def upload(client: TestClient, df: pd.DataFrame, name: str = "customers.csv") -> dict[str, Any]:
    response = client.post("/datasets", files={"file": (name, df.to_csv(index=False).encode(), "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()


def wait_for(client: TestClient, run_id: str, states: tuple[str, ...] = FINISHED) -> dict[str, Any]:
    """Poll GET /runs/{id} until the run is in one of `states` and its worker has let go of it."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["state"] in states and not body["active"]:
            return body
        time.sleep(0.2)
    raise AssertionError(f"Run {run_id} did not reach {states} in {POLL_TIMEOUT_S}s.")


def strings(value: Any) -> Iterator[str]:
    """Every string inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def assert_error(response: Any, status: int, error: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "detail"} and body["error"] == error, body
    return body


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    settings = get_settings()
    original = settings.workspace_dir
    settings.workspace_dir = tmp_path_factory.mktemp("api_workspace")
    llm = fake_llm()
    try:
        with make_client(llm) as client:
            df = classification_df(400)
            dataset = upload(client, df)
            response = client.post("/runs", json={"dataset_id": dataset["id"], "question": QUESTION, "mode": "fast",
                                                  "actionable_features": ["x1", "x2"]})
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            assert wait_for(client, run_id)["state"] == "DONE"
            yield Env(client, settings.workspace_dir, df, dataset["id"], run_id, llm)
    finally:
        settings.workspace_dir = original


# --------------------------------------------------------------------------- health


def test_health_reports_llm_flag_but_never_the_key(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    secret = "sk-test-SUPER-SECRET-0123456789"
    monkeypatch.setattr(settings, "llm_api_key", SecretStr(secret))
    monkeypatch.setattr(settings, "llm_model", "test-model")
    response = env.client.get("/health")
    assert response.status_code == 200 and secret not in response.text and "sk-" not in response.text
    assert response.json() == {"status": "ok", "llm_configured": True, "version": "1.0.0", "public_demo": False,
                               "max_upload_mb": 50.0}

    monkeypatch.setattr(settings, "llm_api_key", SecretStr(""))
    assert env.client.get("/health").json()["llm_configured"] is False


def test_cors_allows_the_streamlit_ui(env: Env) -> None:
    response = env.client.options("/health", headers={"Origin": "http://localhost:8501",
                                                       "Access-Control-Request-Method": "GET"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


# --------------------------------------------------------------------------- datasets


def test_upload_returns_id_and_profile_and_is_listed(env: Env) -> None:
    datasets = env.client.get("/datasets").json()
    entry = next(d for d in datasets if d["id"] == env.dataset_id)
    assert entry["filename"] == "customers.csv" and entry["rows"] == 400 and entry["cols"] == 4
    assert "path" not in entry  # server paths are never exposed

    profile = env.client.get(f"/datasets/{env.dataset_id}/profile").json()
    assert profile["n_rows"] == 400 and {c["name"] for c in profile["columns"]} == {"x1", "x2", "region", "target"}
    stored = list((env.workspace / "uploads").glob(f"{env.dataset_id}_customers.csv"))
    assert len(stored) == 1


def test_upload_rejects_unsupported_extensions(env: Env) -> None:
    for name in ("notes.txt", "data.csv.exe", "no_extension"):
        response = env.client.post("/datasets", files={"file": (name, b"a,b\n1,2\n", "application/octet-stream")})
        assert_error(response, 415, "unsupported_file_type")


def test_upload_rejects_files_over_the_limit(env: Env) -> None:
    with make_client(max_upload_bytes=1_000) as client:
        before = set((env.workspace / "uploads").iterdir())
        small_but_too_big = pd.DataFrame({"a": range(1_000)}).to_csv(index=False).encode()  # ~4 KB: streamed check
        huge = b"a\n" + b"1\n" * 100_000  # ~200 KB: rejected by Content-Length before the body is read
        for content in (small_but_too_big, huge):
            response = client.post("/datasets", files={"file": ("big.csv", content, "text/csv")})
            assert "1,000 bytes" in assert_error(response, 413, "file_too_large")["detail"]
        assert set((env.workspace / "uploads").iterdir()) == before  # partial files are deleted


def test_upload_rejects_unreadable_and_empty_files(env: Env) -> None:
    assert_error(env.client.post("/datasets", files={"file": ("x.parquet", b"not parquet", "application/octet-stream")}),
                 400, "invalid_dataset")
    assert_error(env.client.post("/datasets", files={"file": ("x.csv", b"", "text/csv")}), 400, "empty_file")


def test_upload_sanitizes_the_filename(env: Env) -> None:
    body = upload(env.client, env.df.head(50), name="..\\..\\My Data (v2)!.CSV")
    assert body["filename"] == "My_Data_v2.csv"
    record = get_repository().get_dataset(body["id"])
    assert Path(record.path).parent == (env.workspace / "uploads") and Path(record.path).name == f"{body['id']}_My_Data_v2.csv"


@pytest.mark.parametrize(
    "raw, clean",
    [("report.csv", "report.csv"), ("../../etc/passwd.csv", "passwd.csv"), ("C:\\temp\\x y.XLSX", "x_y.xlsx"),
     ("ünïcödé.json", "unicode.json"), ("...", "dataset"), ("a" * 300 + ".parquet", "a" * 80 + ".parquet")],
)
def test_sanitize_filename(raw: str, clean: str) -> None:
    assert sanitize_filename(raw) == clean


# --------------------------------------------------------------------------- runs


def test_completed_run_detail(env: Env) -> None:
    body = env.client.get(f"/runs/{env.run_id}").json()
    assert body["state"] == "DONE" and body["progress"] == 100 and body["active"] is False
    assert body["dataset_id"] == env.dataset_id and body["task_type"] == "tabular_classification"
    assert body["target"] == "target" and body["best_model"] and body["primary_metric"] == "f1"
    assert body["metric_value"] == body["test_metrics"]["f1"] and body["verification"]["checks"]
    assert body["pending_questions"] == [] and body["errors"] == []
    states = [e["to_state"] for e in body["events"]]
    assert states[0] == "INIT" and states[-1] == "DONE" and "TRAIN" in states
    assert body["report_pdf_url"] == f"/runs/{env.run_id}/report.pdf"
    assert body["message"].endswith(f"Full report: /runs/{env.run_id}/report.pdf")
    assert not any(str(env.workspace) in text for text in strings(body))  # no server paths in the response


def test_runs_are_listed_and_mirrored_in_the_database(env: Env) -> None:
    runs = env.client.get("/runs", params={"dataset_id": env.dataset_id, "state": "DONE"}).json()
    summary = next(r for r in runs if r["run_id"] == env.run_id)
    assert summary["progress"] == 100 and summary["metric_value"] is not None
    assert summary["verification_passed"] is not None and summary["report_md_url"]
    assert env.client.get("/runs", params={"state": "TRAIN"}).json() == []
    assert_error(env.client.get("/runs", params={"state": "BOGUS"}), 422, "validation_error")

    repository = get_repository()
    record = repository.get_run(env.run_id)
    assert record.state == "DONE" and record.dataset_id == env.dataset_id and record.report_pdf
    events = repository.list_events(env.run_id)
    detail = env.client.get(f"/runs/{env.run_id}").json()
    assert [e.to_state for e in events] == [e["to_state"] for e in detail["events"]]
    assert events[0].from_state is None and events[-1].to_state == "DONE"


def test_result_endpoints(env: Env) -> None:
    client, run_id = env.client, env.run_id
    board = client.get(f"/runs/{run_id}/leaderboard").json()
    assert board["primary_metric"] == "f1" and board["entries"] and any(e["selected"] for e in board["entries"])

    explanation = client.get(f"/runs/{run_id}/explanations").json()
    assert explanation["global_explanation"]["features"]
    assert "feature_importance" in explanation["charts"]
    for url in explanation["charts"].values():
        assert url.startswith(f"/runs/{run_id}/charts/")
        image = client.get(url)
        assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.content[:4] == b"\x89PNG"

    recommendations = client.get(f"/runs/{run_id}/recommendations").json()
    assert recommendations["actionable_features"] == ["x1", "x2"]  # from the request, not inferred
    assert [t["name"] for t in recommendations["risk_tiers"]] == ["High", "Medium", "Low"]
    assert recommendations["tier_cutoffs"]["high"] >= recommendations["tier_cutoffs"]["medium"]

    insight = client.get(f"/runs/{run_id}/insight").json()
    assert insight["executive_summary"] and insight["generated_by"] == "template"


def test_report_downloads(env: Env) -> None:
    pdf = env.client.get(f"/runs/{env.run_id}/report.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content[:4] == b"%PDF"
    assert f"idss_report_{env.run_id}.pdf" in pdf.headers["content-disposition"]
    markdown = env.client.get(f"/runs/{env.run_id}/report.md")
    assert markdown.status_code == 200 and markdown.headers["content-type"].startswith("text/markdown")
    assert "## 1. Executive summary" in markdown.text


def test_what_if_single_row_and_segment(env: Env) -> None:
    row = env.df.iloc[0].to_dict()
    single = env.client.post(f"/runs/{env.run_id}/what-if", json={"base_row": row, "changes": {"x1": row["x1"] + 1}})
    assert single.status_code == 200, single.text
    assert single.json()["after"] >= single.json()["before"] and single.json()["delta"] is not None

    segment = env.client.post(f"/runs/{env.run_id}/what-if", json={
        "segment": [{"column": "region", "op": "==", "value": "north"}], "changes": {"x1": {"pct": 10}}})
    assert segment.status_code == 200 and segment.json()["count"] == int((env.df["region"] == "north").sum())
    saved = env.client.get(f"/runs/{env.run_id}").json()  # what-ifs never change the run itself
    assert saved["state"] == "DONE"


@pytest.mark.parametrize(
    "payload, status, error",
    [
        ({"changes": {"x1": 1.0}}, 422, "validation_error"),  # neither base_row nor segment
        ({"base_row": {"x1": 0.0}, "segment": [], "changes": {"x1": 1.0}}, 422, "validation_error"),  # both
        ({"segment": [], "changes": {}}, 422, "validation_error"),  # nothing to change
        ({"segment": [{"column": "x1", "op": "~=", "value": 1}], "changes": {"x1": 1.0}}, 422, "validation_error"),
        ({"segment": [], "changes": {"x1": 1.0}, "extra": True}, 422, "validation_error"),
        ({"segment": [], "changes": {"not_a_column": 1.0}}, 400, "invalid_what_if"),
        ({"segment": [], "changes": {"x1": "high"}}, 400, "invalid_what_if"),
        ({"segment": [{"column": "nope", "op": "==", "value": 1}], "changes": {"x1": 1.0}}, 400, "invalid_what_if"),
    ],
    ids=["no-target", "both-targets", "no-changes", "bad-op", "unknown-field", "unknown-column", "wrong-type",
         "unknown-filter-column"],
)
def test_what_if_validation_errors(env: Env, payload: dict[str, Any], status: int, error: str) -> None:
    assert_error(env.client.post(f"/runs/{env.run_id}/what-if", json=payload), status, error)


def test_ask_runs_llm_code_in_the_sandbox(env: Env) -> None:
    response = env.client.post(f"/runs/{env.run_id}/ask", json={"question": "How many rows are there?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] and body["result"] == 400 and "400" in body["answer"]
    with make_client() as offline:  # no LLM configured
        assert_error(offline.post(f"/runs/{env.run_id}/ask", json={"question": "How many rows?"}), 503,
                     "llm_not_configured")


def test_unknown_ids_and_bad_requests(env: Env) -> None:
    client = env.client
    assert_error(client.get("/runs/20200101-000000-abcdef"), 404, "not_found")
    assert_error(client.get("/datasets/nope/profile"), 404, "not_found")
    assert_error(client.post("/runs", json={"dataset_id": "nope", "question": QUESTION}), 404, "not_found")
    assert_error(client.post("/runs", json={"dataset_id": env.dataset_id, "question": QUESTION, "mode": "turbo"}),
                 422, "validation_error")
    assert_error(client.post("/runs", json={"dataset_id": env.dataset_id, "question": "   "}), 422, "validation_error")
    assert_error(client.get("/no/such/route"), 404, "not_found")
    assert_error(client.post(f"/runs/{env.run_id}/answer", json={"answer": "yes"}), 409, "invalid_state")


@pytest.mark.parametrize(
    "path",
    ["charts/..%5C..%5Cstate.json", "charts/..%2F..%2Fstate.json", "charts/%2E%2E/%2E%2E/state.json",
     "charts/state.json", "charts/%2E%2E", "charts/missing_chart.png"],
)
def test_chart_path_traversal_is_blocked(env: Env, path: str) -> None:
    response = env.client.get(f"/runs/{env.run_id}/{path}")
    assert response.status_code in (400, 404), response.text
    assert '"question"' not in response.text and "requirement" not in response.text  # state.json never leaks


def test_run_id_path_tricks_are_rejected(env: Env) -> None:
    for run_id in ("..%5C..%5Cworkspace", "..", "a%5C..%5Cb"):
        assert env.client.get(f"/runs/{run_id}/report.pdf").status_code == 404
        assert env.client.get(f"/runs/{run_id}/charts/feature_importance.png").status_code == 404


# --------------------------------------------------------------------------- answer flow and busy worker


def test_needs_input_then_answer_continues_to_done(env: Env) -> None:
    unclear = make_requirement("tabular_classification", "target").model_copy(update={"confidence": 0.1})
    llm = fake_llm({UserRequirement: [unclear, make_requirement("tabular_classification", "target")]})
    with make_client(llm) as client:
        run_id = client.post("/runs", json={"dataset_id": env.dataset_id, "question": "Help with my data."}).json()["run_id"]
        waiting = wait_for(client, run_id)
        assert waiting["state"] == "NEEDS_INPUT" and waiting["pending_questions"] and waiting["progress"] == 10
        assert_error(client.post(f"/runs/{run_id}/answer", json={"answer": "  "}), 422, "validation_error")
        assert_error(client.get(f"/runs/{run_id}/leaderboard"), 409, "not_ready")
        assert_error(client.get(f"/runs/{run_id}/report.pdf"), 409, "not_ready")

        response = client.post(f"/runs/{run_id}/answer", json={"answer": "Predict the target column for each customer."})
        assert response.status_code == 202 and response.json()["state"] == "PARSE"
        done = wait_for(client, run_id)
        assert done["state"] == "DONE" and done["answers"] == ["Predict the target column for each customer."]
        assert "NEEDS_INPUT" in [e["to_state"] for e in done["events"]]


class GateLLM(FakeLLM):
    """Blocks the relevance check until released, so a run stays active while the test probes the API."""

    def __init__(self) -> None:
        super().__init__({RelevanceCheck: RelevanceCheck(relevant=False, reason="Small talk.")})
        self.entered, self.release = threading.Event(), threading.Event()

    def chat_structured(self, messages: list[dict[str, str]], schema: type[BaseModel], temperature: float = 0.0) -> Any:
        if schema is RelevanceCheck:
            self.entered.set()
            assert self.release.wait(POLL_TIMEOUT_S)
        return super().chat_structured(messages, schema, temperature)


def test_second_run_while_busy_returns_409(env: Env) -> None:
    llm = GateLLM()
    with make_client(llm) as client:
        first = client.post("/runs", json={"dataset_id": env.dataset_id, "question": "hello there"}).json()["run_id"]
        assert llm.entered.wait(30)
        try:
            detail = client.get(f"/runs/{first}").json()
            assert detail["active"] is True and detail["state"] == "INIT"
            assert_error(client.post("/runs", json={"dataset_id": env.dataset_id, "question": QUESTION}), 409, "busy")
            assert_error(client.post(f"/runs/{first}/what-if", json={"segment": [], "changes": {"x1": 1.0}}), 409,
                         "run_in_progress")
            assert client.get("/health").status_code == 200  # the API stays responsive while a run trains
        finally:
            llm.release.set()
        finished = wait_for(client, first)
        assert finished["state"] == "DONE" and "help with business questions" in finished["message"]
        assert client.post("/runs", json={"dataset_id": env.dataset_id, "question": "hi"}).status_code == 202
        wait_for(client, client.get("/runs").json()[0]["run_id"])


def test_startup_fails_runs_interrupted_by_a_restart(env: Env) -> None:
    run_id = Orchestrator(None).start(QUESTION, env.workspace / "uploads" / "anything.csv")  # left in INIT
    assert get_repository().get_run(run_id).state == "INIT"
    with make_client() as client:  # entering the client runs the app's startup
        body = client.get(f"/runs/{run_id}").json()
    assert body["state"] == "FAILED" and "restarted" in body["message"]
    assert get_repository().get_run(run_id).state == RunState.FAILED


def test_init_db_is_idempotent(env: Env) -> None:
    before = get_repository().get_run(env.run_id)
    init_db()
    init_db()
    assert get_repository().get_run(env.run_id) == before

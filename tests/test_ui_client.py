"""Tests of ui/api_client.py with mocked HTTP (no server, no network)."""

import json
from typing import Any

import pytest
import requests

from ui import api_client
from ui.api_client import BACKEND_DOWN_MESSAGE, ApiClientError, BackendUnavailable, IDSSClient


class Recorder:
    """Stands in for requests.request: records each call and returns (or raises) the queued outcome."""

    def __init__(self, outcome: requests.Response | Exception) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def response(status: int, body: Any = None, *, content: bytes | None = None, content_type: str = "application/json") -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = content if content is not None else json.dumps(body).encode()
    result.headers["Content-Type"] = content_type
    return result


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch):
    def install(outcome: requests.Response | Exception) -> Recorder:
        recorder = Recorder(outcome)
        monkeypatch.setattr(api_client.requests, "request", recorder)
        return recorder
    return install


# --------------------------------------------------------------------------- configuration


def test_base_url_from_env_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDSS_API_URL", raising=False)
    assert IDSSClient().base_url == "http://localhost:8000"
    monkeypatch.setenv("IDSS_API_URL", "http://example.test:9000/")
    client = IDSSClient()
    assert client.base_url == "http://example.test:9000"
    assert client.url("/runs/abc/report.pdf") == "http://example.test:9000/runs/abc/report.pdf"
    assert IDSSClient("http://other:1/").base_url == "http://other:1"


def test_localhost_connects_over_ipv4(mock_http) -> None:
    recorder = mock_http(response(200, {"status": "ok", "llm_configured": False, "version": "1.0.0"}))
    client = IDSSClient("http://localhost:8000")
    client.health()
    assert recorder.calls[0]["url"] == "http://127.0.0.1:8000/health"  # no ~2 s IPv6 fallback on Windows
    assert client.url("/runs/r1/report.pdf") == "http://localhost:8000/runs/r1/report.pdf"  # browser links keep the name
    IDSSClient("https://api.example.test/base").health()
    assert recorder.calls[1]["url"] == "https://api.example.test/base/health"


# --------------------------------------------------------------------------- success


def test_health_success_uses_short_timeout(mock_http) -> None:
    recorder = mock_http(response(200, {"status": "ok", "llm_configured": False, "version": "1.0.0"}))
    assert IDSSClient("http://api").health() == {"status": "ok", "llm_configured": False, "version": "1.0.0"}
    call = recorder.calls[0]
    assert (call["method"], call["url"]) == ("GET", "http://api/health")
    connect, read = call["timeout"]
    assert connect > 0 and read == api_client.HEALTH_TIMEOUT_S


def test_create_run_body(mock_http) -> None:
    recorder = mock_http(response(202, {"run_id": "r1", "state": "INIT", "status_url": "/runs/r1"}))
    client = IDSSClient("http://api")
    assert client.create_run("ds1", "Who churns?", "full")["run_id"] == "r1"
    assert recorder.calls[0]["json"] == {"dataset_id": "ds1", "question": "Who churns?", "mode": "full"}
    client.create_run("ds1", "Who churns?", actionable_features=["price"])
    assert recorder.calls[1]["json"]["actionable_features"] == ["price"]
    assert recorder.calls[1]["json"]["mode"] == "fast"


def test_upload_sends_multipart_file_with_long_timeout(mock_http) -> None:
    recorder = mock_http(response(201, {"id": "ds1", "filename": "a.csv", "rows": 2, "cols": 1}))
    IDSSClient("http://api").upload_dataset("a.csv", b"x\n1\n2\n")
    call = recorder.calls[0]
    assert (call["method"], call["url"]) == ("POST", "http://api/datasets")
    assert call["files"] == {"file": ("a.csv", b"x\n1\n2\n")}
    assert call["timeout"][1] == api_client.UPLOAD_TIMEOUT_S


def test_what_if_body_has_exactly_one_target(mock_http) -> None:
    recorder = mock_http(response(200, {"output": "y", "before": 1, "after": 2}))
    client = IDSSClient("http://api")
    client.what_if("r1", {"price": {"pct": 10}}, base_row={"price": 5})
    client.what_if("r1", {"price": 3}, segment=[])
    assert recorder.calls[0]["json"] == {"changes": {"price": {"pct": 10}}, "base_row": {"price": 5}}
    assert recorder.calls[1]["json"] == {"changes": {"price": 3}, "segment": []}
    assert recorder.calls[0]["url"] == "http://api/runs/r1/what-if"


def test_files_and_ids_in_paths(mock_http) -> None:
    recorder = mock_http(response(200, content="# Report – ✓".encode(), content_type="text/markdown"))
    client = IDSSClient("http://api")
    assert client.report_markdown("r1") == "# Report – ✓"
    assert client.get_bytes("/runs/r1/charts/a.png") == "# Report – ✓".encode()
    with pytest.raises(ApiClientError):  # the queued response is Markdown, not JSON; only the URL matters here
        client.get_run("../etc/passwd")
    assert recorder.calls[-1]["url"] == "http://api/runs/..%2Fetc%2Fpasswd"  # one path segment, never a traversal


# --------------------------------------------------------------------------- API down / slow


@pytest.mark.parametrize("error", [requests.ConnectionError("refused"), requests.ConnectTimeout("slow connect")])
def test_backend_down_gives_friendly_message(mock_http, error: Exception) -> None:
    mock_http(error)
    with pytest.raises(BackendUnavailable) as caught:
        IDSSClient("http://localhost:8000").list_runs()
    assert BACKEND_DOWN_MESSAGE in str(caught.value)
    assert "scripts/run_api.ps1" in str(caught.value)
    assert caught.value.code == "backend_unavailable"
    assert caught.value.status_code is None


def test_read_timeout_is_not_reported_as_down(mock_http) -> None:
    mock_http(requests.ReadTimeout("slow"))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").get_run("r1")
    assert not isinstance(caught.value, BackendUnavailable)
    assert caught.value.code == "timeout"
    assert "did not answer" in caught.value.message


# --------------------------------------------------------------------------- error responses


@pytest.mark.parametrize(("status", "body", "code", "message"), [
    (404, {"error": "not_found", "detail": "Run 'r9' was not found."}, "not_found", "Run 'r9' was not found."),
    (409, {"error": "busy", "detail": "Run r1 is still training."}, "busy", "Run r1 is still training."),
    (409, {"error": "not_ready", "detail": "Run r1 has no explanations yet (state TRAIN)."}, "not_ready",
     "Run r1 has no explanations yet (state TRAIN)."),
    (415, {"error": "unsupported_file_type", "detail": "'.txt' files are not supported."}, "unsupported_file_type",
     "'.txt' files are not supported."),
    (503, {"error": "llm_not_configured", "detail": "Asking questions needs an LLM."}, "llm_not_configured",
     "Asking questions needs an LLM."),
])
def test_error_responses_carry_backend_code_and_detail(mock_http, status: int, body: dict[str, Any], code: str,
                                                       message: str) -> None:
    mock_http(response(status, body))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").get_run("r1")
    assert (caught.value.status_code, caught.value.code, str(caught.value)) == (status, code, message)


def test_validation_error_list_is_readable(mock_http) -> None:
    detail = [{"loc": ["body", "question"], "msg": "String should have at least 1 character", "type": "string_too_short"}]
    mock_http(response(422, {"error": "validation_error", "detail": detail}))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").create_run("ds1", "")
    assert caught.value.code == "validation_error"
    assert str(caught.value) == "Invalid request – question: String should have at least 1 character"
    assert caught.value.detail == detail


def test_internal_and_non_json_errors(mock_http) -> None:
    mock_http(response(500, {"error": "internal_error", "detail": "Unexpected server error; see the server log."}))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").health()
    assert "unexpected error" in caught.value.message

    mock_http(response(502, content=b"<html>Bad gateway</html>", content_type="text/html"))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").health()
    assert (caught.value.status_code, caught.value.code) == (502, "http_502")
    assert caught.value.message == "The backend returned HTTP 502."


def test_success_with_invalid_json(mock_http) -> None:
    mock_http(response(200, content=b"not json", content_type="text/plain"))
    with pytest.raises(ApiClientError) as caught:
        IDSSClient("http://api").list_datasets()
    assert caught.value.code == "bad_response"

"""Thin HTTP client for the IDSS FastAPI backend: the ONLY way the Streamlit UI reaches the system.

Every method returns plain JSON (dicts/lists) or bytes. Failures raise `ApiClientError` with a message that
can be shown to the user as-is:
- the backend is not reachable -> `BackendUnavailable` ("Backend not running - start scripts/run_api.ps1");
- the backend is too slow       -> `ApiClientError` with code "timeout";
- an error response             -> `ApiClientError` with the backend's {error, detail} code and message.
The base URL comes from the IDSS_API_URL environment variable (default http://localhost:8000).
"""

import os
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL_ENV = "IDSS_API_URL"
CONNECT_TIMEOUT_S = 3.05
DEFAULT_TIMEOUT_S = 30.0
HEALTH_TIMEOUT_S = 5.0
UPLOAD_TIMEOUT_S = 180.0  # profiling a large upload takes a while
WHAT_IF_TIMEOUT_S = 120.0  # loads the model and the dataset
ASK_TIMEOUT_S = 180.0  # LLM call + sandboxed code + one repair attempt
BACKEND_DOWN_MESSAGE = "Backend not running – start scripts/run_api.ps1"


class ApiClientError(Exception):
    """A failed API call; `str(exc)` is a friendly message, `code` the backend's stable error code."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str = "error",
                 detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.detail = detail


class BackendUnavailable(ApiClientError):
    """The backend could not be reached at all (not started, wrong URL, or crashed)."""

    def __init__(self, base_url: str) -> None:
        super().__init__(f"{BACKEND_DOWN_MESSAGE} (could not reach {base_url}).", code="backend_unavailable")


def _seg(value: str) -> str:
    """An id as one URL path segment (ids come from the backend, but never trust them to be URL-safe)."""
    return quote(str(value), safe="")


def _ipv4_loopback(base_url: str) -> str:
    """'localhost' -> '127.0.0.1' for the connection itself. The API binds 127.0.0.1 (scripts/run_api.ps1), and on
    Windows a connection to 'localhost' first tries IPv6 ::1 and waits ~2 s for the refusal, on every request."""
    parts = urlsplit(base_url)
    if parts.hostname != "localhost":
        return base_url
    return urlunsplit(parts._replace(netloc="127.0.0.1" + (f":{parts.port}" if parts.port else "")))


def _format_detail(detail: Any) -> str:
    """The backend's `detail` as one readable sentence (validation errors arrive as a list)."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for problem in detail:
            if isinstance(problem, dict):
                location = ".".join(str(p) for p in problem.get("loc", []) if p not in ("body", "query", "path"))
                message = str(problem.get("msg", "invalid value"))
                parts.append(f"{location}: {message}" if location else message)
            else:
                parts.append(str(problem))
        return "Invalid request – " + "; ".join(parts) if parts else "Invalid request."
    return str(detail) if detail is not None else ""


class IDSSClient:
    """Calls the IDSS API. Stateless (one request per call), so it is safe to share between Streamlit sessions."""

    def __init__(self, base_url: str | None = None, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self._connect_base = _ipv4_loopback(self.base_url)

    def url(self, path: str) -> str:
        """Absolute URL of an API path such as '/runs/<id>/report.pdf' (the API returns paths, not URLs);
        used for links the browser opens."""
        return f"{self.base_url}/{path.lstrip('/')}"

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, *, timeout_s: float | None = None, **kwargs: Any) -> requests.Response:
        read_timeout = timeout_s or self.timeout_s
        try:
            response = requests.request(method, f"{self._connect_base}/{path.lstrip('/')}",
                                        timeout=(CONNECT_TIMEOUT_S, read_timeout), **kwargs)
        except requests.ConnectTimeout:
            raise BackendUnavailable(self.base_url) from None
        except requests.Timeout:
            raise ApiClientError(f"The backend did not answer within {read_timeout:g} s; it may be busy. Try again shortly.",
                                 code="timeout") from None
        except requests.ConnectionError:
            raise BackendUnavailable(self.base_url) from None
        except requests.RequestException as exc:
            raise ApiClientError(f"Request to the backend failed: {exc}", code="request_failed") from None
        if response.status_code >= 400:
            raise self._error(response)
        return response

    @staticmethod
    def _error(response: requests.Response) -> ApiClientError:
        code, detail = f"http_{response.status_code}", None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            code = str(body.get("error") or code)
            detail = body.get("detail")
        message = _format_detail(detail) or f"The backend returned HTTP {response.status_code}."
        if response.status_code >= 500 and code == "internal_error":
            message = "The backend hit an unexpected error; see its log window for details."
        return ApiClientError(message, status_code=response.status_code, code=code, detail=detail)

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError:
            raise ApiClientError("The backend sent a response that is not JSON.", code="bad_response") from None

    # ------------------------------------------------------------------ system and datasets

    def health(self) -> dict[str, Any]:
        """{status, llm_configured, version, public_demo, max_upload_mb}."""
        return self._json("GET", "/health", timeout_s=HEALTH_TIMEOUT_S)

    def list_datasets(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._json("GET", "/datasets", params={"limit": limit})

    def upload_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        """Upload a CSV/XLSX/JSON/Parquet file; returns the dataset record with a profile summary."""
        return self._json("POST", "/datasets", files={"file": (filename, content)}, timeout_s=UPLOAD_TIMEOUT_S)

    def dataset_profile(self, dataset_id: str) -> dict[str, Any]:
        return self._json("GET", f"/datasets/{_seg(dataset_id)}/profile")

    # ------------------------------------------------------------------ runs

    def create_run(self, dataset_id: str, question: str, mode: str = "fast",
                   actionable_features: list[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"dataset_id": dataset_id, "question": question, "mode": mode}
        if actionable_features:
            body["actionable_features"] = actionable_features
        return self._json("POST", "/runs", json=body)

    def list_runs(self, limit: int = 50, dataset_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if dataset_id:
            params["dataset_id"] = dataset_id
        if state:
            params["state"] = state
        return self._json("GET", "/runs", params=params)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{_seg(run_id)}")

    def answer(self, run_id: str, answer: str) -> dict[str, Any]:
        return self._json("POST", f"/runs/{_seg(run_id)}/answer", json={"answer": answer})

    # ------------------------------------------------------------------ results

    def leaderboard(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{_seg(run_id)}/leaderboard")

    def explanations(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{_seg(run_id)}/explanations")

    def recommendations(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{_seg(run_id)}/recommendations")

    def insight(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{_seg(run_id)}/insight")

    def what_if(self, run_id: str, changes: dict[str, Any], *, base_row: dict[str, Any] | None = None,
                segment: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Simulate `changes` for one record (`base_row`) or a filtered segment (`segment`; [] = every record)."""
        body: dict[str, Any] = {"changes": changes}
        if base_row is not None:
            body["base_row"] = base_row
        if segment is not None:
            body["segment"] = segment
        return self._json("POST", f"/runs/{_seg(run_id)}/what-if", json=body, timeout_s=WHAT_IF_TIMEOUT_S)

    def ask(self, run_id: str, question: str) -> dict[str, Any]:
        return self._json("POST", f"/runs/{_seg(run_id)}/ask", json={"question": question}, timeout_s=ASK_TIMEOUT_S)

    # ------------------------------------------------------------------ files

    def get_bytes(self, path: str) -> bytes:
        """Raw bytes of an API path returned by the backend (chart or report URL)."""
        return self._request("GET", path).content

    def report_pdf(self, run_id: str) -> bytes:
        return self.get_bytes(f"/runs/{_seg(run_id)}/report.pdf")

    def report_markdown(self, run_id: str) -> str:
        return self._request("GET", f"/runs/{_seg(run_id)}/report.md").content.decode("utf-8")

    def chart(self, run_id: str, name: str) -> bytes:
        return self.get_bytes(f"/runs/{_seg(run_id)}/charts/{_seg(name)}")


def get_client() -> IDSSClient:
    """The client the UI pages use (tests replace this function with one returning a fake)."""
    return IDSSClient()

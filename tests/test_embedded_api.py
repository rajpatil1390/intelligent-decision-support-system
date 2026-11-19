"""Embedded API mode (IDSS_EMBEDDED_API=1) and the Streamlit Community Cloud setup.

The embedded tests start the real API with uvicorn on a free loopback port (127.0.0.1 only, no outside network)
and stop it afterwards; no LLM is called (only GET /health).
"""

import ast
import os
import socket
import threading
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import PUBLIC_DEMO_MAX_EXPLAIN_ROWS, PUBLIC_DEMO_MAX_UPLOAD_BYTES, ApiConfig, create_app
from configs.settings import get_settings
from idss.storage.runs import ensure_workspace
from ui import api_client, embedded_api
from ui.embedded_api import EmbeddedApiError

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
THREAD_NAME = "idss-embedded-api"
DEV_ONLY = {"pytest", "pytest-cov", "httpx2", "ruff", "playwright"}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def embedded_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == THREAD_NAME and t.is_alive()]


def reset_embedded() -> None:
    if embedded_api._running is not None:
        embedded_api._running.stop()
    embedded_api._running = None
    embedded_api._shared_backend.clear()


@pytest.fixture(autouse=True)
def _no_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets come only from what a test sets (never from a secrets.toml on the machine)."""
    monkeypatch.setattr(embedded_api.st, "secrets", {})


@pytest.fixture
def embedded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[int]]:
    """Embedded mode on a free port with a fresh workspace; yields the list of API apps created (one per start)."""
    monkeypatch.setattr(get_settings(), "workspace_dir", tmp_path / "workspace")
    monkeypatch.setenv(embedded_api.EMBEDDED_ENV, "1")
    monkeypatch.setenv(embedded_api.API_URL_ENV, "http://unused.invalid")  # restored afterwards; the start overwrites it
    monkeypatch.setattr(embedded_api, "PORT", free_port())
    starts: list[int] = []
    real_factory = embedded_api.create_backend_app

    def counting_factory() -> Any:
        starts.append(1)
        return real_factory()

    monkeypatch.setattr(embedded_api, "create_backend_app", counting_factory)
    reset_embedded()
    yield starts
    reset_embedded()


# --------------------------------------------------------------------------- embedded mode


def test_embedded_backend_starts_once_answers_health_and_is_reused(embedded: list[int]) -> None:
    first = embedded_api.ensure_backend()
    assert first is not None and first.alive and embedded == [1]
    assert first.base_url == f"http://127.0.0.1:{embedded_api.PORT}"
    assert first.thread.daemon and first.thread.name == THREAD_NAME

    health = requests.get(f"{first.base_url}/health", timeout=5)
    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert os.environ[embedded_api.API_URL_ENV] == first.base_url
    assert api_client.IDSSClient().health()["status"] == "ok"  # the UI's client now reaches the built-in API

    assert embedded_api.ensure_backend() is first  # every later script run reuses it
    embedded_api._shared_backend.clear()  # even after Streamlit's resource cache is cleared
    assert embedded_api.ensure_backend() is first
    assert embedded == [1] and len(embedded_threads()) == 1

    settings = get_settings()  # the API's startup created the workspace folders
    assert all(folder.is_dir() for folder in (settings.runs_dir, settings.logs_dir, settings.uploads_dir))


def test_concurrent_first_calls_start_a_single_server(embedded: list[int]) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        backends = list(pool.map(lambda _: embedded_api.get_or_start(port=embedded_api.PORT), range(4)))
    assert len({id(backend) for backend in backends}) == 1
    assert embedded == [1] and len(embedded_threads()) == 1


def test_off_by_default_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("embedded mode is off")

    monkeypatch.setattr(embedded_api, "start_backend", must_not_start)
    assert not embedded_api.enabled()
    assert embedded_api.ensure_backend() is None


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), (" YES ", True), ("on", True),
                                                 ("0", False), ("false", False), ("no", False), ("", False)])
def test_flag_from_the_environment(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(embedded_api.EMBEDDED_ENV, value)
    assert embedded_api.enabled() is expected


def test_settings_come_from_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedded_api.st, "secrets", {"IDSS_EMBEDDED_API": True, "IDSS_PUBLIC_DEMO": 1,
                                                     "section": {"IDSS_EMBEDDED_API": "1"}})
    assert embedded_api.enabled()
    assert embedded_api.flag("IDSS_PUBLIC_DEMO") is True
    assert embedded_api.config_value("section") is None  # only root-level values count
    assert embedded_api.flag("NOT_SET") is None
    monkeypatch.setenv(embedded_api.EMBEDDED_ENV, "0")  # the environment wins over secrets
    assert not embedded_api.enabled()


def test_missing_secrets_file_means_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoSecrets:
        def get(self, name: str, default: Any = None) -> Any:
            raise FileNotFoundError("No secrets found.")  # what st.secrets does without a secrets.toml

    monkeypatch.setattr(embedded_api.st, "secrets", NoSecrets())
    assert embedded_api.config_value(embedded_api.EMBEDDED_ENV) is None
    assert not embedded_api.enabled()


def test_public_demo_secret_reaches_the_embedded_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedded_api.st, "secrets", {"IDSS_PUBLIC_DEMO": "1"})
    config = embedded_api.create_backend_app().state.config
    assert config.public_demo is True and config.max_upload_bytes == PUBLIC_DEMO_MAX_UPLOAD_BYTES
    monkeypatch.setattr(embedded_api.st, "secrets", {})
    assert embedded_api.create_backend_app().state.config.public_demo is False  # unset: the API's setting (off)


def test_port_in_use_fails_before_the_api_starts() -> None:
    created: list[int] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(EmbeddedApiError, match="already in use"):
            embedded_api.start_backend(lambda: created.append(1), port=busy.getsockname()[1])
    assert created == []  # the API (and its startup clean-up) never ran


def test_startup_times_out_without_a_healthy_answer() -> None:
    def unhealthy_app() -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        def health() -> JSONResponse:
            return JSONResponse({"status": "starting"}, status_code=503)

        return app

    started = time.monotonic()
    with pytest.raises(EmbeddedApiError, match=r"within 0\.5 s"):
        embedded_api.start_backend(unhealthy_app, port=free_port(), timeout_s=0.5)
    assert time.monotonic() - started < 15
    assert embedded_threads() == []  # the half-started server was stopped


def test_startup_error_is_reported() -> None:
    @asynccontextmanager
    async def failing_lifespan(app: FastAPI) -> Any:
        raise RuntimeError("database unavailable")
        yield  # never reached; makes this an async generator

    with pytest.raises(EmbeddedApiError, match="stopped during startup"):
        embedded_api.start_backend(lambda: FastAPI(lifespan=failing_lifespan), port=free_port(), timeout_s=20)
    assert embedded_threads() == []


def test_failed_start_is_shown_on_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(embedded_api.EMBEDDED_ENV, "1")
    shown: list[str] = []

    def failing(host: str, port: int) -> None:
        raise EmbeddedApiError("port 8000 on 127.0.0.1 is already in use")

    monkeypatch.setattr(embedded_api, "_shared_backend", failing)
    monkeypatch.setattr(embedded_api.st, "error", lambda message, **kwargs: shown.append(message))
    assert embedded_api.ensure_backend() is None
    assert len(shown) == 1 and "could not start" in shown[0] and "already in use" in shown[0]


def _first_call(path: Path) -> str:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    calls = [node for node in body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    return ast.unparse(calls[0]) if calls else ""


@pytest.mark.parametrize("path", [UI_DIR / "app.py", *sorted((UI_DIR / "pages").glob("*.py"))], ids=lambda p: p.name)
def test_app_and_every_page_start_the_embedded_api_first(path: Path) -> None:
    assert _first_call(path) == "embedded_api.ensure_backend()"


def test_only_the_embedded_module_imports_the_backend() -> None:
    """The UI still reaches the system only over HTTP; embedded_api.py alone imports api.main (lazily)."""
    offenders = []
    for path in UI_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".")[0] in ("api", "idss", "configs") for name in names):
                offenders.append((path.name, names))
    assert offenders == [("embedded_api.py", ["api.main"])]


# --------------------------------------------------------------------------- Streamlit Community Cloud setup


def test_streamlit_config_leaves_address_and_port_to_the_host() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert "address" not in config.get("server", {}) and "port" not in config.get("server", {})
    assert config["theme"]["primaryColor"] == "#6C5CE7"  # the theme stays
    script = (ROOT / "scripts" / "run_ui.ps1").read_text(encoding="utf-8")
    assert '[string]$Address = "localhost"' in script and '"--server.address", $Address' in script


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-r"):
            assert "==" in line, f"{path.name}: unpinned requirement {line!r}"
            names.add(line.split("==", 1)[0].strip().lower())
    return names


def test_runtime_requirements_have_no_dev_tools() -> None:
    runtime = _requirement_names(ROOT / "requirements.txt")
    dev_file = ROOT / "requirements-dev.txt"
    assert not runtime & DEV_ONLY
    assert {"streamlit", "fastapi", "uvicorn", "requests", "shap", "lightgbm"} <= runtime
    assert DEV_ONLY <= _requirement_names(dev_file)
    assert "-r requirements.txt" in dev_file.read_text(encoding="utf-8").splitlines()


def test_workspace_folders_are_created_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(get_settings(), "workspace_dir", fresh)
    assert ensure_workspace() == [fresh, fresh / "runs", fresh / "logs", fresh / "uploads"]
    assert all(folder.is_dir() for folder in ensure_workspace())  # idempotent

    empty = tmp_path / "ephemeral"
    monkeypatch.setattr(get_settings(), "workspace_dir", empty)
    with TestClient(create_app(llm_factory=lambda: None)) as client:
        assert client.get("/health").status_code == 200
    assert all((empty / name).is_dir() for name in ("runs", "logs", "uploads"))


def test_public_demo_orchestrator_is_fast_only_with_a_small_shap_sample() -> None:
    def no_llm() -> None:
        return None

    demo = api_main._orchestrator(ApiConfig(llm_factory=no_llm, public_demo=True), None)
    assert demo.fast_only and demo.max_explain_rows == PUBLIC_DEMO_MAX_EXPLAIN_ROWS <= 200
    local = api_main._orchestrator(ApiConfig(llm_factory=no_llm), None)
    assert not local.fast_only and local.max_explain_rows is None  # the default sample (MAX_EXPLAIN_ROWS)

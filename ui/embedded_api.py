"""Embedded API mode: the Streamlit UI starts the FastAPI backend itself, inside the same process.

Switched on by IDSS_EMBEDDED_API=1, read from the environment or from Streamlit secrets. Hosting that runs ONE
Streamlit app and nothing else (Streamlit Community Cloud) then needs no second service: the first script run
starts uvicorn on 127.0.0.1:8000 in a daemon thread, exactly once per process (st.cache_resource, plus a
process-wide guard in case that cache is cleared), and waits until GET /health answers (at most 60 s). After that
the pages talk to the backend over HTTP through ui/api_client.py, exactly as they do with a separate API process.

Default off: locally scripts/run_api.ps1 + scripts/run_ui.ps1 run two processes, as before.

ui/app.py and every page call `ensure_backend()` first. This is the only UI module that touches the backend
package, and only inside the embedded start (a lazy import of `api.main`).
"""

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

EMBEDDED_ENV = "IDSS_EMBEDDED_API"
PUBLIC_DEMO_ENV = "IDSS_PUBLIC_DEMO"
API_URL_ENV = "IDSS_API_URL"  # what ui/api_client.py reads
HOST = "127.0.0.1"  # never reachable from outside the machine/container
PORT = 8000
STARTUP_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 0.2
HEALTH_TIMEOUT_S = 2.0
STOP_TIMEOUT_S = 10.0
_TRUE = frozenset({"1", "true", "yes", "on"})

AppFactory = Callable[[], Any]  # returns the ASGI app to serve


class EmbeddedApiError(RuntimeError):
    """The embedded backend could not start: port taken, startup error, or no /health answer in time."""


@dataclass
class EmbeddedBackend:
    """A backend served by uvicorn on a daemon thread of this process."""

    server: Any  # uvicorn.Server
    thread: threading.Thread
    base_url: str

    @property
    def alive(self) -> bool:
        return self.thread.is_alive() and not self.server.should_exit

    def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> None:
        """Ask uvicorn to shut down and wait for its thread (tests; the app itself never stops it)."""
        self.server.should_exit = True
        self.thread.join(timeout_s)


_lock = threading.Lock()
_running: EmbeddedBackend | None = None


# --------------------------------------------------------------------------- configuration


def config_value(name: str) -> str | None:
    """`name` from the environment, else from the root level of Streamlit secrets, else None (blank = unset)."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        secret = st.secrets.get(name)
    except Exception:  # no secrets.toml (Streamlit raises a FileNotFoundError) or an unreadable one: unset
        return None
    if secret is None or isinstance(secret, Mapping):
        return None
    if isinstance(secret, bool):  # TOML `IDSS_EMBEDDED_API = true`
        return "1" if secret else "0"
    return str(secret).strip() or None


def flag(name: str) -> bool | None:
    """A boolean setting from the environment or Streamlit secrets; None when it is not set at all."""
    value = config_value(name)
    return None if value is None else value.lower() in _TRUE


def enabled() -> bool:
    """True when the UI should run the API itself (IDSS_EMBEDDED_API=1)."""
    return bool(flag(EMBEDDED_ENV))


# --------------------------------------------------------------------------- start / reuse


def ensure_backend() -> EmbeddedBackend | None:
    """Embedded mode: start the backend on the first call, reuse it afterwards. Otherwise a no-op (None).

    A failed start is shown on the page and retried on the next rerun; meanwhile the pages show their usual
    "backend offline" state.
    """
    if not enabled():
        return None
    try:
        return _shared_backend(HOST, PORT)
    except EmbeddedApiError as exc:
        st.error(f"The built-in backend could not start: {exc}", icon=":material/error:")
        return None


@st.cache_resource(show_spinner="Starting the IDSS backend (first visit only) ...")
def _shared_backend(host: str, port: int) -> EmbeddedBackend:
    """One backend per process; Streamlit computes a cached resource once, even for concurrent sessions."""
    return get_or_start(host=host, port=port)


def get_or_start(app_factory: AppFactory | None = None, *, host: str = HOST, port: int = PORT,
                 timeout_s: float = STARTUP_TIMEOUT_S) -> EmbeddedBackend:
    """The running embedded backend, or a new one (also points ui/api_client at it via IDSS_API_URL)."""
    global _running
    with _lock:
        if _running is None or not _running.alive:
            _running = start_backend(app_factory, host=host, port=port, timeout_s=timeout_s)
        os.environ[API_URL_ENV] = _running.base_url
        return _running


def start_backend(app_factory: AppFactory | None = None, *, host: str = HOST, port: int = PORT,
                  timeout_s: float = STARTUP_TIMEOUT_S) -> EmbeddedBackend:
    """Serve the API with uvicorn on a daemon thread and wait until GET /health answers."""
    import uvicorn

    if not _port_free(host, port):  # checked first, so the API's startup never runs against a foreign server
        raise EmbeddedApiError(f"port {port} on {host} is already in use (is another API running?)")
    app = (app_factory or create_backend_app)()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False,
                                           lifespan="on"))
    thread = threading.Thread(target=_serve, args=(server,), name="idss-embedded-api", daemon=True)
    thread.start()  # uvicorn skips its signal handlers off the main thread
    backend = EmbeddedBackend(server=server, thread=thread, base_url=f"http://{host}:{port}")
    try:
        wait_for_health(backend, timeout_s)
    except EmbeddedApiError:
        backend.stop(timeout_s=5.0)
        raise
    return backend


def create_backend_app() -> Any:
    """The real IDSS API. IDSS_PUBLIC_DEMO may come from Streamlit secrets, so it is passed in explicitly
    (unset: the API's own setting decides, i.e. the environment or .env)."""
    from api.main import create_app  # the backend shares this process only in embedded mode

    return create_app(public_demo=flag(PUBLIC_DEMO_ENV))


def wait_for_health(backend: EmbeddedBackend, timeout_s: float = STARTUP_TIMEOUT_S) -> None:
    """Poll GET /health until it answers 200; fail early if the server thread dies."""
    deadline = time.monotonic() + timeout_s
    while True:
        if not backend.thread.is_alive():
            raise EmbeddedApiError("the API server stopped during startup (see the log for the error)")
        if backend.server.started:
            try:
                if requests.get(f"{backend.base_url}/health", timeout=HEALTH_TIMEOUT_S).status_code == 200:
                    return
            except requests.RequestException:
                pass
        if time.monotonic() >= deadline:
            raise EmbeddedApiError(f"no answer from {backend.base_url}/health within {timeout_s:g} s")
        time.sleep(POLL_INTERVAL_S)


def _serve(server: Any) -> None:
    """Thread target: run uvicorn until it is told to stop."""
    try:
        server.run()
    except SystemExit:  # uvicorn exits on a startup failure; it logged why, and wait_for_health reports it
        pass


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True

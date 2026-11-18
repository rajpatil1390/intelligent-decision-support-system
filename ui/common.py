"""Shared shell of the Streamlit pages: page list, styling, sidebar status, session state, pickers, formatting.

Pages reach the backend only through `client()` (ui.api_client); nothing in ui/ imports idss.* or api.*.
"""

import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from ui import api_client, styles
from ui.api_client import ApiClientError, BackendUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Demo Mode (Home): the bundled churn sample, uploaded through the API and analysed in fast mode.
DEMO_DATASET = PROJECT_ROOT / "data" / "samples" / "customer_churn.csv"
DEMO_QUESTION = "Which customers are likely to churn, and what should we do to retain them?"

# (file relative to ui/app.py, title, icon); the first page is the default.
PAGES: list[tuple[str, str, str]] = [
    ("pages/1_Home.py", "Home", ":material/home:"),
    ("pages/2_Upload_Data.py", "Upload Data", ":material/upload_file:"),
    ("pages/3_New_Analysis.py", "New Analysis", ":material/add_chart:"),
    ("pages/4_Run_Monitor.py", "Run Monitor", ":material/monitor_heart:"),
    ("pages/5_Results.py", "Results", ":material/leaderboard:"),
    ("pages/6_Decision_Center.py", "Decision Center", ":material/insights:"),
    ("pages/7_Ask_Your_Data.py", "Ask Your Data", ":material/chat:"),
    ("pages/8_Reports.py", "Reports", ":material/description:"),
]
PAGE_HOME, PAGE_UPLOAD, PAGE_NEW, PAGE_MONITOR, PAGE_RESULTS, PAGE_DECISIONS, PAGE_ASK, PAGE_REPORTS = (
    path for path, _, _ in PAGES
)

FINISHED_STATES = ("DONE", "FAILED")
STOP_POLLING_STATES = ("DONE", "FAILED", "NEEDS_INPUT")
STATE_LABELS: dict[str, str] = {
    "INIT": "Starting up",
    "PARSE": "Understanding the question",
    "VALIDATE": "Checking the requirement",
    "NEEDS_INPUT": "Waiting for your answer",
    "PROFILE": "Profiling and preparing the data",
    "PLAN": "Planning which models to try",
    "TRAIN": "Training and tuning models",
    "VERIFY": "Verifying the results",
    "EXPLAIN": "Explaining the model and simulating decisions",
    "INSIGHT": "Writing the decision brief",
    "REPORT": "Building the report",
    "DONE": "Finished",
    "FAILED": "Failed",
}
STATE_ICONS: dict[str, str] = {"DONE": "✅", "FAILED": "❌", "NEEDS_INPUT": "❓"}
TASK_LABELS: dict[str, str] = {
    "tabular_classification": "Classification",
    "tabular_regression": "Regression",
    "tabular_clustering": "Clustering",
    "time_series_forecasting": "Time-series forecast",
    "text_classification": "Text classification",
}
MODE_HELP = {"fast": "Fast – 3 models × 15 tuning trials (about 1–3 min)",
             "full": "Full – 6 models × 40 tuning trials (up to ~18 min)"}
DEFAULT_UPLOAD_MB = 50  # the backend's limit when /health does not say (older backend)
PUBLIC_DEMO_BANNER = ("**Public demo – do not upload private data.** Uploads are limited to {limit} MB, runs use the "
                      "fast budget, Ask Your Data is turned off, and runs older than 24 hours are deleted when the "
                      "server restarts.")

_RUN_KEY, _DATASET_KEY, _HEALTH_KEY = "current_run_id", "current_dataset_id", "health"


def client() -> api_client.IDSSClient:
    """The backend client (looked up on every call so tests can swap `api_client.get_client`)."""
    return api_client.get_client()


# --------------------------------------------------------------------------- session state


def current_run_id() -> str | None:
    return st.session_state.get(_RUN_KEY)


def set_current_run(run_id: str | None) -> None:
    st.session_state[_RUN_KEY] = run_id


def current_dataset_id() -> str | None:
    return st.session_state.get(_DATASET_KEY)


def set_current_dataset(dataset_id: str | None) -> None:
    st.session_state[_DATASET_KEY] = dataset_id


def health() -> dict[str, Any] | None:
    """Backend health from this script run's sidebar check (None if the backend is down)."""
    return st.session_state.get(_HEALTH_KEY)


def llm_configured() -> bool:
    status = health()
    return bool(status and status.get("llm_configured"))


def public_demo() -> bool:
    """True when the backend runs as the public demo (IDSS_PUBLIC_DEMO=1)."""
    status = health()
    return bool(status and status.get("public_demo"))


def max_upload_mb() -> float:
    status = health()
    value = status.get("max_upload_mb") if status else None
    return float(value) if isinstance(value, int | float) and value > 0 else DEFAULT_UPLOAD_MB


# --------------------------------------------------------------------------- layout


def page_config() -> None:
    st.set_page_config(page_title="IDSS – Decision Support", page_icon=":material/insights:", layout="wide",
                       initial_sidebar_state="expanded")


def shell() -> None:
    """The frame every page starts with: wide layout, styling, footer and sidebar status.

    Pages call this themselves (rather than relying on ui/app.py) because Streamlit runs page files directly,
    without the entry script, while it is in its legacy `pages/` folder mode (e.g. the first request after a
    server restart lands on a page URL, and in AppTest). The footer is created here, before any `st.stop()`,
    and the stylesheet moves it to the end of the page.
    """
    page_config()
    styles.inject_css()
    styles.footer()
    render_sidebar()
    if public_demo():
        st.warning(PUBLIC_DEMO_BANNER.format(limit=fmt_num(max_upload_mb())), icon=":material/public:")


def page_title(title: str, caption: str = "", icon: str = "✨") -> None:
    """The page's gradient hero banner."""
    styles.hero(title, caption, icon)


def render_sidebar() -> None:
    """API health, whether an LLM is configured, and the current run/dataset."""
    status: dict[str, Any] | None = None
    problem = ""
    try:
        status = client().health()
    except ApiClientError as exc:
        problem = exc.message
    st.session_state[_HEALTH_KEY] = status

    with st.sidebar:
        st.markdown('<div class="idss-side-title">System status</div>', unsafe_allow_html=True)
        if status is not None:
            llm = bool(status.get("llm_configured"))
            st.markdown(
                '<div class="idss-status">'
                f'<span class="idss-dot on"></span>API online <span class="idss-muted">v{status.get("version", "?")}</span><br>'
                f'<span class="idss-dot {"on" if llm else "warn"}"></span>LLM configured: <b>{"Yes" if llm else "No"}</b>'
                '</div>', unsafe_allow_html=True)
            if not llm:
                st.caption("Offline mode: the task is taken from the data profile and insights use templates.")
        else:
            st.markdown('<div class="idss-status"><span class="idss-dot off"></span>API offline</div>',
                        unsafe_allow_html=True)
            st.caption(problem or api_client.BACKEND_DOWN_MESSAGE)

        st.markdown('<div class="idss-side-title">Current run</div>', unsafe_allow_html=True)
        run_id = current_run_id()
        if not run_id:
            st.caption("None selected yet.")
        else:
            st.code(run_id, language=None)
            if status is not None:
                try:
                    run = client().get_run(run_id)
                    state = run.get("state", "?")
                    st.caption(f"{STATE_ICONS.get(state, '⏳')} {state} · {run.get('progress', 0)}%")
                except ApiClientError as exc:
                    if exc.status_code == 404:
                        set_current_run(None)
                    st.caption(exc.message)
            if st.button("Clear current run", key="sidebar_clear_run", width="stretch"):
                set_current_run(None)
                st.rerun()


def show_error(exc: ApiClientError) -> None:
    """A friendly message for a failed call (backend down, still running, not ready, invalid input...)."""
    if isinstance(exc, BackendUnavailable):
        st.error(exc.message, icon=":material/cloud_off:")
    elif exc.status_code == 409:
        st.warning(exc.message, icon=":material/hourglass_top:")
    else:
        st.error(exc.message, icon=":material/error:")


def fetch[T](fn: Callable[..., T], *args: Any, not_ready: str | None = None, **kwargs: Any) -> T | None:
    """Call the backend; on failure show a friendly message and return None.

    `not_ready`: message shown (as info) when the result is not computed yet (409 not_ready).
    When this script run's health check already found the backend down, no call is made (on Windows every
    refused connection costs 1-2 s).
    """
    if st.session_state.get(_HEALTH_KEY, ...) is None:
        show_error(BackendUnavailable(client().base_url))
        return None
    try:
        return fn(*args, **kwargs)
    except ApiClientError as exc:
        if exc.code == "not_ready" and not_ready:
            st.info(not_ready, icon=":material/hourglass_top:")
        else:
            show_error(exc)
        return None


def empty_state(message: str, page: str | None = None, label: str = "", icon: str = ":material/info:") -> None:
    st.info(message, icon=icon)
    if page:
        st.page_link(page, label=label or "Go", icon=":material/arrow_forward:")


# --------------------------------------------------------------------------- pickers


def run_label(run: dict[str, Any]) -> str:
    question = (run.get("question") or "").strip().replace("\n", " ")
    if len(question) > 60:
        question = question[:57] + "…"
    return f"{STATE_ICONS.get(run.get('state', ''), '⏳')} {run['run_id']} · {question}"


def _sync(widget_key: str, setter: Callable[[str | None], None]) -> None:
    setter(st.session_state.get(widget_key))


def run_picker(runs: list[dict[str, Any]], *, key: str, label: str = "Run") -> str | None:
    """Select box of runs that keeps the app-wide current run in sync. None if there are no runs."""
    if not runs:
        return None
    ids = [run["run_id"] for run in runs]
    labels = {run["run_id"]: run_label(run) for run in runs}
    current = current_run_id()
    st.session_state[key] = current if current in ids else ids[0]
    st.selectbox(label, ids, key=key, format_func=lambda rid: labels.get(rid, rid),
                 on_change=_sync, args=(key, set_current_run))
    selected = st.session_state[key]
    if selected != current:  # defaulted to the newest run: rerun once so the sidebar shows it too
        set_current_run(selected)
        st.rerun()
    return selected


def dataset_label(dataset: dict[str, Any]) -> str:
    return f"{dataset['filename']} · {dataset['rows']:,} rows × {dataset['cols']} cols · {fmt_time(dataset.get('uploaded_at'))}"


def dataset_picker(datasets: list[dict[str, Any]], *, key: str, label: str = "Dataset") -> str | None:
    """Select box of uploaded datasets that keeps the app-wide current dataset in sync."""
    if not datasets:
        return None
    ids = [dataset["id"] for dataset in datasets]
    labels = {dataset["id"]: dataset_label(dataset) for dataset in datasets}
    current = current_dataset_id()
    st.session_state[key] = current if current in ids else ids[0]
    st.selectbox(label, ids, key=key, format_func=lambda did: labels.get(did, did),
                 on_change=_sync, args=(key, set_current_dataset))
    selected = st.session_state[key]
    if selected != current:
        set_current_dataset(selected)
    return selected


# --------------------------------------------------------------------------- formatting


def fmt_num(value: Any, digits: int = 3) -> str:
    """Compact, readable number: 0.623, 12.346, 1,234.5, 98,765; '—' for missing."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    return str(value)


def fmt_pct(share: float | None, digits: int = 1) -> str:
    return "—" if share is None else f"{share * 100:.{digits}f}%"


def fmt_time(value: str | None, with_seconds: bool = False) -> str:
    """ISO timestamp from the API -> local time text."""
    if not value:
        return "—"
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone()
    except ValueError:
        return str(value)
    return moment.strftime("%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M")


def humanize(name: str) -> str:
    """'feature_importance' -> 'Feature importance'."""
    text = name.replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:]


def model_label(name: str | None) -> str:
    """'logistic_regression' -> 'logistic regression' (wraps at the space in a KPI card)."""
    return name.replace("_", " ") if name else "—"


def task_label(task_type: str | None) -> str:
    return TASK_LABELS.get(task_type or "", humanize(task_type) if task_type else "—")


def verified_mark(passed: bool | None) -> str:
    return "—" if passed is None else ("✅ Passed" if passed else "❌ Failed")

"""Run Monitor: progress, current state and live event timeline of a run; answer clarifying questions.

While the run is working, a fragment re-polls GET /runs/{id} every 2 seconds. Each state change reruns the
whole page (so the sidebar stays in step); polling stops at DONE, FAILED or NEEDS_INPUT.
"""

from typing import Any

import streamlit as st

from ui import common
from ui import components as ui
from ui import styles
from ui import embedded_api

embedded_api.ensure_backend()  # IDSS_EMBEDDED_API=1: start the in-process API once (no-op otherwise)

POLL_SECONDS = 2


def render_status(run: dict[str, Any]) -> None:
    state = run.get("state", "?")
    progress = int(run.get("progress", 0))
    ui.pipeline(state, run.get("events") or [])
    st.progress(progress / 100, text=f"**{state}** – {common.STATE_LABELS.get(state, '')} ({progress}%)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Task", common.task_label(run.get("task_type")) if run.get("task_type") else "—")
    col2.metric("Target", run.get("target") or "—")
    col3.metric("Best model", common.model_label(run.get("best_model")))
    metric, value = run.get("primary_metric"), run.get("metric_value")
    col4.metric(f"Test {metric}" if metric else "Test score", common.fmt_num(value) if value is not None else "—")
    if run.get("active"):
        st.caption(f"⏳ Working… this page refreshes every {POLL_SECONDS} s.")
    styles.section("Event timeline")
    ui.timeline(run.get("events") or [])


common.shell()
common.page_title("Run monitor", "Follow a run live: its progress, current step and every state change.", icon="🛰️")

api = common.client()
runs = common.fetch(api.list_runs, limit=100)
if runs is None:
    st.stop()
if not runs:
    common.empty_state("No runs yet. Start one from New Analysis.", common.PAGE_NEW, "New analysis")
    st.stop()

run_id = common.run_picker(runs, key="monitor_run_picker")
detail = common.fetch(api.get_run, run_id)
if detail is None:
    st.stop()
st.markdown(f"**❓ Question:** {detail.get('question', '')}")

polling = detail["state"] not in common.STOP_POLLING_STATES
first_render: dict[str, dict[str, Any]] = {"run": detail}  # reuse this fetch for the first render only


@st.fragment(run_every=POLL_SECONDS if polling else None)
def live_status() -> None:
    run = first_render.pop("run", None) or common.fetch(api.get_run, run_id)
    if run is None:
        return
    render_status(run)
    if polling and run["state"] != detail["state"]:
        st.rerun()  # whole page: sidebar status, and at the end the answer box / results links


live_status()

state = detail["state"]
if state == "NEEDS_INPUT":
    styles.section("The agents need a little more information")
    for question in detail.get("pending_questions") or []:
        st.info(question, icon=":material/help:")
    if detail.get("answers"):
        with st.expander("Your previous answers"):
            for previous in detail["answers"]:
                st.markdown(f"- {previous}")
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area("Your answer", placeholder="e.g. The target column is 'churn'; optimise F1.")
        submitted = st.form_submit_button("Send answer", type="primary")
    if submitted:
        if not answer.strip():
            st.warning("Please type an answer first.")
        elif common.fetch(api.answer, run_id, answer.strip()) is not None:
            st.toast("Answer sent – the run continues.")
            st.rerun()
elif state == "FAILED":
    reasons = detail.get("errors") or []
    st.error("**The run failed.** " + (reasons[-1] if reasons else (detail.get("message") or "No reason was recorded.")),
             icon=":material/error:")
    if len(reasons) > 1:
        with st.expander("All errors"):
            for reason in reasons:
                st.markdown(f"- {reason}")
    st.page_link(common.PAGE_NEW, label="Start a new analysis", icon=":material/refresh:")
elif state == "DONE":
    if detail.get("best_model"):
        st.success("The run finished. Explore the results, decisions and report:", icon=":material/check_circle:")
        col1, col2, col3, col4 = st.columns(4)
        col1.page_link(common.PAGE_RESULTS, label="Results", icon=":material/leaderboard:")
        col2.page_link(common.PAGE_DECISIONS, label="Decision center", icon=":material/insights:")
        col3.page_link(common.PAGE_ASK, label="Ask your data", icon=":material/chat:")
        col4.page_link(common.PAGE_REPORTS, label="Reports", icon=":material/description:")
    else:
        st.info(detail.get("message") or "The run finished without training a model.", icon=":material/info:")

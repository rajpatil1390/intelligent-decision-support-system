"""Ask Your Data: chat about a run's dataset (POST /runs/{id}/ask). The backend has the LLM write pandas code,
runs it in a sandbox and answers; the generated code is shown in a collapsible block. History is kept per run
in session state."""

import json
from typing import Any

import streamlit as st

from ui import common
from ui import components as ui
from ui.api_client import ApiClientError

HISTORY_KEY = "ask_history"
EXAMPLE_QUESTIONS = [
    "How many rows does the dataset have?",
    "What is the average of the target for each category of the most important feature?",
    "Which 5 records have the highest values of the target?",
]


def history(run_id: str) -> list[dict[str, Any]]:
    return st.session_state.setdefault(HISTORY_KEY, {}).setdefault(run_id, [])


def show_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"], icon=":material/error:")
        else:
            st.markdown(message["content"])
        if message.get("code"):
            with st.expander("Generated code"):
                st.code(message["code"], language="python")
        if message.get("stdout"):
            with st.expander("Printed output"):
                st.code(message["stdout"], language=None)


def assistant_message(answer: dict[str, Any]) -> dict[str, Any]:
    content = answer.get("answer") or ""
    result = answer.get("result")
    if not content and result is not None:
        content = f"```\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
    return {"role": "assistant", "content": content or "(no answer)", "code": answer.get("code") or "",
            "stdout": answer.get("stdout") or "", "error": not answer.get("ok", True)}


common.shell()
common.page_title("Ask your data", "Ask questions about a run's dataset in plain language. "
                                   "An LLM writes the analysis code, which runs in a locked-down sandbox.", icon="🤖")

with st.expander("How it works", expanded=not common.llm_configured()):
    ui.flow_diagram([
        ("Ask", "Type a question about the run's data; only column names and types are sent to the LLM."),
        ("Write code", "The LLM writes a short pandas analysis; one automatic repair if it fails."),
        ("Sandbox", "An AST guard and a separate, locked-down Python process run it with a time limit."),
        ("Answer", "You get the answer, the printed output and the generated code to inspect."),
    ], icons=("💬", "🧑‍💻", "🛡️", "✅"))

api = common.client()
runs = common.fetch(api.list_runs, limit=100)
if runs is None:
    st.stop()
if not runs:
    common.empty_state("No runs yet. Start one from New Analysis; questions are asked about a run's dataset.",
                       common.PAGE_NEW, "New analysis")
    st.stop()

run_id = common.run_picker(runs, key="ask_run_picker")
run = next((r for r in runs if r["run_id"] == run_id), {})
demo = common.public_demo()
llm_ready = common.llm_configured() and not demo
busy = bool(run.get("active"))
if demo:
    st.info("Ask Your Data is turned off in the public demo, because it runs generated code on the server. "
            "Run IDSS on your own computer to use it; every other page works here.", icon=":material/block:")
elif not llm_ready:
    st.info("Asking questions needs an LLM. Set LLM_MODEL and LLM_API_KEY in .env and restart the backend; "
            "every other page works offline.", icon=":material/cloud_off:")
elif busy:
    st.info("This run is still working; you can ask questions once it has finished.", icon=":material/hourglass_top:")

messages = history(run_id)
if not messages:
    st.caption("Try for example: " + " · ".join(f"“{q}”" for q in EXAMPLE_QUESTIONS))
for message in messages:
    show_message(message)

if messages and st.button("Clear conversation", key=f"ask_clear_{run_id}"):
    messages.clear()
    st.rerun()

prompt = st.chat_input("Ask a question about this run's data", disabled=not llm_ready or busy, key="ask_input")
if prompt:
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    show_message(user_message)
    try:
        with st.spinner("Writing and running the analysis…"):
            reply = assistant_message(api.ask(run_id, prompt))
    except ApiClientError as exc:
        reply = {"role": "assistant", "content": exc.message, "error": True}
    messages.append(reply)
    show_message(reply)

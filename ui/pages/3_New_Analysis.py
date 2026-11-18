"""New Analysis: pick a dataset, ask a business question, choose the budget, and start a run (POST /runs)."""

import streamlit as st

from ui import common, styles

QUESTION_KEY = "new_question"
EXAMPLES: dict[str, str] = {
    "Customer churn": common.DEMO_QUESTION,
    "Crop price": "Predict the crop price from weather, soil and yield data, and tell me which factors matter most.",
    "Sales forecast": "Forecast daily store sales for the next month and show what drives sales "
                      "(promotions, holidays, temperature).",
}


def use_example(text: str) -> None:
    st.session_state[QUESTION_KEY] = text


common.shell()
common.page_title("New analysis", "Ask a business question about one of your datasets. "
                                  "The agents turn it into a machine-learning plan and run it.", icon="💬")

api = common.client()
datasets = common.fetch(api.list_datasets)
if datasets is None:
    st.stop()
if not datasets:
    common.empty_state("Upload a dataset first.", common.PAGE_UPLOAD, "Upload data")
    st.stop()

dataset_id = common.dataset_picker(datasets, key="new_dataset_picker", label="Dataset")
profile = common.fetch(api.dataset_profile, dataset_id) if dataset_id else None
if profile is None:
    st.stop()
target = profile.get("suggested_target")
st.caption(f"{profile['n_rows']:,} rows × {profile['n_cols']} columns · suggested target: "
           f"{target or '—'} · suggested task: {common.task_label(profile.get('suggested_task_type'))}")

styles.section("Business question")
example_columns = st.columns(len(EXAMPLES))
for column, (label, text) in zip(example_columns, EXAMPLES.items(), strict=True):
    column.button(f"Example: {label}", key=f"example_{label}", on_click=use_example, args=(text,), width="stretch")
question = st.text_area("Describe what you want to know or decide", key=QUESTION_KEY, height=110,
                        placeholder="e.g. Which customers are likely to churn, and what should we do about it?",
                        label_visibility="collapsed")

styles.section("Options")
left, right = st.columns([1, 2])
with left:
    modes = ["fast"] if common.public_demo() else ["fast", "full"]
    if st.session_state.get("new_mode") not in modes:
        st.session_state.pop("new_mode", None)
    mode = st.radio("Training budget", modes, format_func=lambda m: common.MODE_HELP[m], key="new_mode")
    if common.public_demo():
        st.caption("The public demo runs in fast mode only.")
with right:
    candidates = [c["name"] for c in profile.get("columns", [])
                  if c["name"] != target and c.get("semantic_type") not in ("id", "text")]
    actionable = st.multiselect(
        "Actionable features (optional)", candidates, key="new_actionable",
        help="Columns your business can actually change, such as price or contract type. "
             "Recommendations and what-if levers focus on these. Leave empty to let IDSS infer them.")

if not common.llm_configured():
    st.info(f"No LLM is configured, so IDSS runs offline: the question is kept for the report, and the task and "
            f"target come from the data profile (target **{target or '—'}**, "
            f"{common.task_label(profile.get('suggested_task_type'))}). Insights use templates.",
            icon=":material/cloud_off:")

if st.button("Start analysis", type="primary", key="start_run", icon=":material/rocket_launch:"):
    if not (question or "").strip():
        st.warning("Please type a business question first (or click one of the examples).")
    else:
        created = common.fetch(api.create_run, dataset_id, question.strip(), mode, actionable or None)
        if created is not None:
            common.set_current_run(created["run_id"])
            st.switch_page(common.PAGE_MONITOR)

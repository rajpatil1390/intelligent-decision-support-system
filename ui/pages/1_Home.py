"""Home: what the system does, Demo Mode, the 4-step flow, headline numbers and the most recent runs."""

from typing import Any

import streamlit as st

from ui import common
from ui import components as ui
from ui import styles

RECENT_RUNS = 10
# Primary metrics on a bounded, higher-is-better scale: only these are averaged on the KPI strip
# (error metrics such as RMSE have the target's units and cannot be averaged with them).
SCORE_METRICS = {"accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "r2",
                 "silhouette", "rand_index", "adjusted_rand_index", "nmi"}


def start_demo() -> None:
    """Upload the churn sample through the API, start a fast run on it and open the Run Monitor."""
    try:
        content = common.DEMO_DATASET.read_bytes()
    except OSError:
        st.error("The demo dataset data/samples/customer_churn.csv is missing; "
                 "recreate it with scripts/make_samples.py.", icon=":material/error:")
        return
    api = common.client()
    with st.spinner("Uploading the customer churn sample and starting a fast run…"):
        dataset = common.fetch(api.upload_dataset, common.DEMO_DATASET.name, content)
        if dataset is None:
            return
        common.set_current_dataset(dataset["id"])
        created = common.fetch(api.create_run, dataset["id"], common.DEMO_QUESTION, "fast")
    if created is not None:
        common.set_current_run(created["run_id"])
        st.switch_page(common.PAGE_MONITOR)


def kpi_strip(datasets: list[dict[str, Any]] | None, runs: list[dict[str, Any]] | None) -> None:
    successful = [r for r in runs or [] if r.get("state") == "DONE" and r.get("best_model")]
    verified = sum(1 for r in successful if r.get("verification_passed"))
    scores = [float(r["metric_value"]) for r in successful
              if r.get("primary_metric") in SCORE_METRICS and isinstance(r.get("metric_value"), (int, float))]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total datasets", f"{len(datasets):,}" if datasets is not None else "—")
    col2.metric("Total runs", f"{len(runs):,}" if runs is not None else "—")
    col3.metric("Successful runs", f"{len(successful):,}" if runs is not None else "—",
                help=f"Finished with a trained model; {verified} passed every verification check.")
    col4.metric("Avg primary metric", common.fmt_num(sum(scores) / len(scores)) if scores else "—",
                help="Mean test score of successful runs whose primary metric is a 0–1 score (F1, accuracy, "
                     "R², ARI, …). Error metrics such as RMSE are in the target's units and are left out.")


common.shell()
common.page_title(
    "Turn a business question into a verified, explained decision",
    "Describe your problem in plain language and upload your data. A team of AI agents plans the analysis, "
    "an AutoML engine trains and verifies models, and the decision layer explains the results and recommends actions.",
    icon="🧭",
)

with st.container(border=True):
    text_col, button_col = st.columns([3, 1], vertical_alignment="center")
    text_col.markdown("**🚀 Live demo** — upload the customer churn sample, ask *“Which customers are likely to churn, "
                      "and what should we do?”* and watch the agents work in fast mode (about 1–3 minutes).")
    if button_col.button("Demo Mode", type="primary", key="demo_mode", icon=":material/rocket_launch:",
                         width="stretch"):
        start_demo()

ui.flow_diagram([
    ("Upload", "Add a CSV, Excel, JSON or Parquet file; IDSS profiles every column."),
    ("Ask", "Type your business question, e.g. “Which customers will churn and why?”"),
    ("Analyse", "Agents pick the task and target, tune several models and verify the best one."),
    ("Decide", "See key drivers, risk tiers or forecasts, run what-if scenarios and download the report."),
])

api = common.client()
all_datasets = common.fetch(api.list_datasets, limit=500)
all_runs = common.fetch(api.list_runs, limit=500) if all_datasets is not None else None
kpi_strip(all_datasets, all_runs)

link1, link2, link3, link4 = st.columns(4)
link1.page_link(common.PAGE_UPLOAD, label="Upload data", icon=":material/upload_file:")
link2.page_link(common.PAGE_NEW, label="New analysis", icon=":material/add_chart:")
link3.page_link(common.PAGE_DECISIONS, label="Decision center", icon=":material/insights:")
link4.page_link(common.PAGE_REPORTS, label="Reports", icon=":material/description:")

styles.section("Recent runs")
if all_runs is None:
    st.caption("Start the backend to see your runs.")
elif not all_runs:
    common.empty_state("No runs yet. Click Demo Mode, or upload a dataset and ask your first question.",
                       common.PAGE_UPLOAD, "Upload a dataset")
else:
    ui.runs_table(all_runs[:RECENT_RUNS])
    st.caption("Select a run on the Run Monitor, Results or Decision Center pages to open it.")

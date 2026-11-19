"""Results: headline metrics, model leaderboard, verification checks, key drivers and the run's charts."""

from typing import Any

import streamlit as st

from ui import charts, common, styles
from ui import components as ui
from ui.api_client import ApiClientError
from ui import embedded_api

embedded_api.ensure_backend()  # IDSS_EMBEDDED_API=1: start the in-process API once (no-op otherwise)


def show_charts(charts: dict[str, str]) -> None:
    if not charts:
        st.caption("This run produced no charts.")
        return
    api = common.client()
    columns = st.columns(2)
    for index, (name, url) in enumerate(charts.items()):
        with columns[index % 2]:
            st.markdown(f"**{common.humanize(name)}**")
            try:
                st.image(api.get_bytes(url), caption=ui.CHART_CAPTIONS.get(name, ""), width="stretch")
            except ApiClientError as exc:
                st.caption(f"Chart unavailable: {exc.message}")


def show_verification(verification: dict[str, Any] | None) -> None:
    if not verification:
        st.info("This run has not been verified yet.", icon=":material/hourglass_top:")
        return
    if verification.get("passed"):
        st.success("All verification checks passed.", icon=":material/verified:")
    else:
        failed = [common.humanize(c["name"]) for c in verification.get("checks", []) if not c.get("passed")]
        st.error(f"Verification failed: {', '.join(failed) or 'see the checks below'}. "
                 "Treat the results with caution.", icon=":material/gpp_bad:")
    ui.table(ui.checks_frame(verification), column_config={"Detail": st.column_config.TextColumn(width="large")})
    if verification.get("llm_verdict"):
        st.markdown(f"**Reviewer verdict:** {verification['llm_verdict']}")
    for title, key in (("Concerns", "llm_concerns"), ("Recommendations", "recommendations")):
        if verification.get(key):
            st.markdown(f"**{title}**")
            for item in verification[key]:
                st.markdown(f"- {item}")


common.shell()
common.page_title("Results", "How well the models performed, whether the best one passed verification, "
                             "and what drives it.", icon="🏆")

api = common.client()
runs = common.fetch(api.list_runs, limit=100)
if runs is None:
    st.stop()
if not runs:
    common.empty_state("No runs yet. Start one from New Analysis.", common.PAGE_NEW, "New analysis")
    st.stop()

run_id = common.run_picker(runs, key="results_run_picker")
detail = common.fetch(api.get_run, run_id)
if detail is None:
    st.stop()
if not detail.get("best_model"):
    if detail["state"] in common.FINISHED_STATES:
        common.empty_state(detail.get("message") or "This run finished without training a model.",
                           common.PAGE_MONITOR, "See the run's timeline")
    else:
        common.empty_state(f"Results appear when training finishes (the run is at {detail['state']}).",
                           common.PAGE_MONITOR, "Follow it on the Run Monitor", icon=":material/hourglass_top:")
    st.stop()

metric = detail.get("primary_metric") or "score"
threshold = detail.get("decision_threshold")
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Test {metric}", common.fmt_num(detail.get("metric_value")))
col2.metric("Best model", common.model_label(detail["best_model"]))
col3.metric("Verification", common.verified_mark(detail.get("verification_passed")))
col4.metric("Decision threshold", common.fmt_num(threshold) if threshold is not None else "n/a",
            help="Probability cut-off tuned on the validation split (binary classification only).")
st.caption(f"{common.task_label(detail.get('task_type'))} · target: {detail.get('target') or '—'} · "
           f"state: {detail['state']} · started {common.fmt_time(detail.get('created_at'))}")
if detail["state"] not in common.FINISHED_STATES:
    st.info("The run is still working; explanations and charts appear when it reaches EXPLAIN.",
            icon=":material/hourglass_top:")

tab_board, tab_verify, tab_drivers, tab_charts = st.tabs(
    ["Leaderboard", "Verification", "Key drivers", "Charts"])

with tab_board:
    test_metrics = detail.get("test_metrics") or {}
    scalar_metrics = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if scalar_metrics:
        styles.section("Test-set metrics of the selected model")
        ui.chips([ui.chip(f"{name}: {common.fmt_num(value)}") for name, value in scalar_metrics.items()])
    board = common.fetch(api.leaderboard, run_id, not_ready="The leaderboard appears when training finishes.")
    if board is not None:
        direction = "higher is better" if board.get("direction") == "maximize" else "lower is better"
        styles.section("Model leaderboard")
        st.caption(f"Validation {board.get('primary_metric')} of every candidate ({direction}); "
                   "★ = selected model (purple).")
        figure = charts.leaderboard(board)
        if figure is not None:
            charts.show(figure, alt="Validation score of every candidate model", key="leaderboard_chart")
        ui.table(ui.leaderboard_frame(board), column_config={"Notes": st.column_config.TextColumn(width="large")})

with tab_verify:
    show_verification(detail.get("verification"))

explanation = None
if detail["state"] in ("INSIGHT", "REPORT", "DONE"):
    explanation = common.fetch(api.explanations, run_id, not_ready="Explanations are not ready yet.")

with tab_drivers:
    if explanation is None:
        st.info("Key drivers appear once the model has been explained.", icon=":material/hourglass_top:")
    else:
        global_explanation = explanation.get("global_explanation") or {}
        features = global_explanation.get("features") or []
        if features:
            styles.section(f"What drives the {global_explanation.get('output', 'prediction')}")
            charts.show(charts.drivers(features), alt="Share of the explanation per input column", key="drivers_chart")
            st.caption(f"Method: {global_explanation.get('method', '—')} on "
                       f"{global_explanation.get('n_samples', 0):,} training rows. ▲ a higher value raises the "
                       "output, ▼ lowers it, ◆ mixed effect. Hover a bar for details.")
            for feature in features[:5]:
                if feature.get("detail"):
                    st.markdown(f"- **{feature['feature']}** ({common.fmt_pct(feature.get('share'))}): {feature['detail']}")
        else:
            st.info("This model has no feature-level explanation.", icon=":material/info:")
        for note in global_explanation.get("notes") or []:
            st.caption(note)

with tab_charts:
    if explanation is None:
        st.info("Charts appear once the model has been explained.", icon=":material/hourglass_top:")
    else:
        show_charts(explanation.get("charts") or {})

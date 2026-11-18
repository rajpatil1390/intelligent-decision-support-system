"""Decision Center: the decision brief, recommendations, risk tiers / forecast / clusters, and two simulators
(one typical record, or a filtered segment) backed by POST /runs/{id}/what-if."""

from typing import Any

import pandas as pd
import streamlit as st

from ui import charts, common, styles
from ui import components as ui

INPUT_COLUMNS = 3


# --------------------------------------------------------------------------- decision brief


def show_insight(insight: dict[str, Any]) -> None:
    if insight.get("generated_by") == "llm":
        grounding = insight.get("grounding") or {}
        label = "Written by the LLM · numbers verified" if grounding.get("passed") else "Written by the LLM"
        source = ui.badge(label, "info")
    else:
        source = ui.badge("Template (offline) · built from computed facts", "low")
    st.markdown(f'<div class="idss-section">Executive summary {source}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="idss-summary">{ui.esc(insight.get("executive_summary", ""))}</div>',
                unsafe_allow_html=True)
    if insight.get("key_findings"):
        styles.section("Key findings")
        for finding in insight["key_findings"]:
            st.markdown(f"- {finding}")


def show_tiers(recs: dict[str, Any]) -> None:
    tiers = recs.get("risk_tiers") or []
    ui.tier_chips(tiers)
    left, right = st.columns([1, 1.5], vertical_alignment="center")
    with left:
        charts.show(charts.tier_donut(tiers), alt="Records per risk tier", key="tier_donut")
    with right:
        ui.table(ui.tiers_frame(tiers), column_config={"Top drivers": st.column_config.TextColumn(width="medium")})
        cutoffs = recs.get("tier_cutoffs")
        if cutoffs:
            st.caption(
                f"High: probability ≥ {common.fmt_num(cutoffs['high'], 4)} (the higher of the decision threshold "
                f"{common.fmt_num(cutoffs['medium'], 4)} and the {cutoffs.get('percentile', 70)}th percentile "
                f"{common.fmt_num(cutoffs.get('percentile_value'), 4)}) · Medium: ≥ {common.fmt_num(cutoffs['medium'], 4)}"
                f" · Low: below {common.fmt_num(cutoffs['medium'], 4)}.")


def show_forecast(forecast: dict[str, Any]) -> None:
    figure = charts.forecast(forecast.get("points", []))
    if figure is not None:
        charts.show(figure, alt="Forecast with its uncertainty band", key="forecast_chart")
    st.caption(f"{forecast.get('horizon', len(forecast.get('points', [])))} steps ahead · band: {forecast.get('interval', '')}")
    for assumption in forecast.get("assumptions") or []:
        st.caption(f"• {assumption}")


def show_clusters(clusters: list[dict[str, Any]]) -> None:
    left, right = st.columns([1, 1.6], vertical_alignment="center")
    with left:
        charts.show(charts.cluster_sizes(clusters), alt="Records per cluster", key="cluster_sizes")
    with right:
        ui.table(ui.clusters_frame(clusters))
    for cluster in clusters:
        if cluster.get("distinguishing"):
            st.markdown(f"- **Cluster {cluster['cluster']}:** " + "; ".join(cluster["distinguishing"]))


def show_visual(recs: dict[str, Any] | None, explanation: dict[str, Any] | None) -> None:
    if recs and recs.get("risk_tiers"):
        styles.section("Risk tiers")
        show_tiers(recs)
    elif explanation and explanation.get("forecast"):
        styles.section("Forecast")
        show_forecast(explanation["forecast"])
    elif explanation and explanation.get("clusters"):
        styles.section("Cluster profiles")
        show_clusters(explanation["clusters"])
    elif explanation and (explanation.get("global_explanation") or {}).get("features"):
        styles.section("Key drivers")
        charts.show(charts.drivers(explanation["global_explanation"]["features"]),
                    alt="Share of the explanation per input column", key="decision_drivers")


def show_counterfactual(example: dict[str, Any]) -> None:
    changes = ", ".join(f"<b>{ui.esc(c['feature'])}</b>: {ui.esc(c['before'])} → {ui.esc(c['after'])}"
                        for c in example.get("changes") or [])
    outcome = "drops below" if example.get("success") else "does not drop below"
    ui.card("🔁 Example: smallest change that lowers one high-risk record",
            f"<p>Record {ui.esc(example.get('row_index', '—'))}: probability "
            f"{ui.esc(common.fmt_num(example.get('probability_before')))} → "
            f"{ui.esc(common.fmt_num(example.get('probability_after')))}, which {outcome} the decision threshold "
            f"{ui.esc(common.fmt_num(example.get('threshold')))}.</p><p>Changes: {changes or 'none'}</p>")


# --------------------------------------------------------------------------- simulators


def input_grid(columns: list[dict[str, Any]], base: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    grid = st.columns(INPUT_COLUMNS)
    for index, column in enumerate(columns):
        with grid[index % INPUT_COLUMNS]:
            values[column["name"]] = ui.value_input(column, base.get(column["name"]), key=f"{key_prefix}{column['name']}")
    return values


def record_simulator(run_id: str, profile: dict[str, Any], target: str | None, actionable: list[str],
                     example: dict[str, Any] | None) -> None:
    base = ui.typical_row(profile, target)
    changeable = ui.changeable_columns(profile, target)
    if not changeable:
        st.info("This dataset has no numeric or categorical inputs to change.", icon=":material/info:")
        return
    levers = [c for c in changeable if c["name"] in actionable] or changeable[:INPUT_COLUMNS]
    others = [c for c in changeable if c not in levers]
    key_prefix = f"wi_{run_id}_"

    st.markdown("Start from a **typical record** (median of every number, most common category) and change any "
                "inputs to see how the model's output moves.")
    with st.expander("Typical record used as the starting point"):
        ui.table(pd.DataFrame([{"Column": k, "Value": str(v)} for k, v in base.items()]))

    with st.form(f"what_if_form_{run_id}"):
        st.markdown("**Levers**" + (" (actionable features)" if actionable else ""))
        scenario = input_grid(levers, base, key_prefix)
        if others:
            with st.expander("Other inputs"):
                scenario.update(input_grid(others, base, key_prefix))
        submitted = st.form_submit_button("Simulate", type="primary", icon=":material/science:")

    def reset() -> None:
        for column in changeable:
            st.session_state.pop(f"{key_prefix}{column['name']}", None)
        st.session_state.pop(f"what_if_result_{run_id}", None)

    st.button("Reset to the typical record", key=f"what_if_reset_{run_id}", on_click=reset)

    result_key = f"what_if_result_{run_id}"
    if submitted:
        changes = ui.diff_changes(base, scenario)
        if not changes:
            st.info("Change at least one input from its typical value, then simulate.", icon=":material/info:")
        else:
            with st.spinner("Simulating…"):
                result = common.fetch(common.client().what_if, run_id, changes, base_row=base)
            if result is not None:
                st.session_state[result_key] = result
    if st.session_state.get(result_key):
        styles.section("Result")
        ui.what_if_result(st.session_state[result_key])
    elif example:
        styles.section("Example computed during the run")
        ui.what_if_result(example)


def _filter_value_input(column: dict[str, Any], op: str, key: str) -> Any:
    if column.get("semantic_type") == "numeric":
        median = (column.get("stats") or {}).get("median") or 0.0
        return st.number_input("Value", value=float(median), key=key)
    options = ui.category_options(column)
    if op == "in":
        return st.multiselect("Values", options, default=options[:1], key=key, format_func=str)
    return ui.coerce_value(column, st.selectbox("Value", options, key=key, format_func=str,
                                                accept_new_options=int(column.get("n_unique", 0)) > len(options)))


def segment_simulator(run_id: str, profile: dict[str, Any], target: str | None) -> None:
    changeable = ui.changeable_columns(profile, target)
    if not changeable:
        st.info("This dataset has no numeric or categorical inputs to change.", icon=":material/info:")
        return
    by_name = {c["name"]: c for c in changeable}
    filters_key, result_key = f"segment_filters_{run_id}", f"segment_result_{run_id}"
    filters: list[dict[str, Any]] = st.session_state.setdefault(filters_key, [])

    st.markdown("Choose **which records** to change with filters (none = every record), then **what to change**. "
                "IDSS reports the average model output before and after.")
    st.markdown("**1. Segment filters**")
    filterable = [c for c in profile.get("columns", []) if c.get("semantic_type") in ui.CHANGEABLE_TYPES]
    with st.container(border=True):
        col_a, col_b, col_c, col_d = st.columns([2, 1, 2, 1], vertical_alignment="bottom")
        with col_a:
            name = st.selectbox("Column", [c["name"] for c in filterable], key=f"seg_col_{run_id}")
        column = next(c for c in filterable if c["name"] == name)
        numeric = column.get("semantic_type") == "numeric"
        with col_b:
            op = st.selectbox("Operator", ui.FILTER_OPS_NUMERIC if numeric else ui.FILTER_OPS_CATEGORICAL,
                              key=f"seg_op_{run_id}_{'num' if numeric else 'cat'}")
        with col_c:
            value = _filter_value_input(column, op, key=f"seg_val_{run_id}_{name}_{op}")
        with col_d:
            if st.button("Add filter", key=f"seg_add_{run_id}", icon=":material/add:"):
                if op == "in" and not value:
                    st.toast("Pick at least one value for 'in'.")
                else:
                    filters.append({"column": name, "op": op, "value": value})
                    st.session_state.pop(result_key, None)
                    st.rerun()
    if filters:
        for index, condition in enumerate(filters):
            text_col, remove_col = st.columns([6, 1], vertical_alignment="center")
            text_col.markdown(f"`{condition['column']} {condition['op']} {condition['value']!r}`")
            if remove_col.button("Remove", key=f"seg_remove_{run_id}_{index}"):
                filters.pop(index)
                st.session_state.pop(result_key, None)
                st.rerun()
    else:
        st.caption("No filters: the whole dataset is the segment.")

    st.markdown("**2. Changes**")
    chosen = st.multiselect("Inputs to change", list(by_name), key=f"seg_changes_{run_id}")
    changes: dict[str, Any] = {}
    for name in chosen:
        column = by_name[name]
        with st.container(border=True):
            if column.get("semantic_type") == "numeric":
                kind_col, value_col = st.columns([1, 2])
                with kind_col:
                    kind = st.radio(name, ["pct", "add", "set"], horizontal=True, key=f"seg_kind_{run_id}_{name}",
                                    format_func={"pct": "% change", "add": "add", "set": "set to"}.get)
                with value_col:
                    if kind == "set":
                        changes[name] = ui.value_input(column, ui.typical_value(column), key=f"seg_set_{run_id}_{name}",
                                                       label="New value")
                    else:
                        amount = st.number_input("Percent" if kind == "pct" else "Amount", value=10.0 if kind == "pct" else 1.0,
                                                 key=f"seg_amount_{run_id}_{name}_{kind}")
                        changes[name] = {kind: amount}
            else:
                changes[name] = ui.value_input(column, ui.typical_value(column), key=f"seg_set_{run_id}_{name}",
                                               label=f"{name}: set to")

    if st.button("Simulate segment", type="primary", key=f"seg_run_{run_id}", icon=":material/groups:"):
        if not changes:
            st.warning("Choose at least one input to change.")
        else:
            with st.spinner("Simulating the segment…"):
                result = common.fetch(common.client().what_if, run_id, changes, segment=filters)
            if result is not None:
                st.session_state[result_key] = result
    if st.session_state.get(result_key):
        styles.section("Result")
        ui.segment_result(st.session_state[result_key])


# --------------------------------------------------------------------------- page

common.shell()
common.page_title("Decision center", "The decision brief, recommended actions and what-if simulation for a "
                                     "finished run.", icon="🎯")

api = common.client()
runs = common.fetch(api.list_runs, limit=100)
if runs is None:
    st.stop()
if not runs:
    common.empty_state("No runs yet. Start one from New Analysis.", common.PAGE_NEW, "New analysis")
    st.stop()

run_id = common.run_picker(runs, key="decisions_run_picker")
detail = common.fetch(api.get_run, run_id)
if detail is None:
    st.stop()
if detail["state"] != "DONE" or not detail.get("best_model"):
    if detail["state"] == "DONE":
        message = detail.get("message") or "This run finished without training a model."
    elif detail["state"] == "FAILED":
        message = "This run failed, so there are no decisions to show."
    else:
        message = f"Decisions appear when the run has finished (it is at {detail['state']})."
    common.empty_state(message, common.PAGE_MONITOR, "Open the Run Monitor")
    st.stop()

insight = common.fetch(api.insight, run_id, not_ready="The decision brief is not ready yet.")
recs = common.fetch(api.recommendations, run_id, not_ready="Recommendations are not ready yet.")
explanation = common.fetch(api.explanations, run_id, not_ready="Explanations are not ready yet.")

tab_brief, tab_record, tab_segment = st.tabs(["Decision brief", "What-if simulator", "Segment simulator"])

with tab_brief:
    if insight:
        show_insight(insight)
    show_visual(recs, explanation)
    if recs and recs.get("recommendations"):
        styles.section("Recommended actions")
        for rec in recs["recommendations"]:
            ui.recommendation_card(rec)
    elif recs is not None:
        st.info("No recommendations were produced for this run.", icon=":material/info:")
    if recs and recs.get("counterfactual_example"):
        show_counterfactual(recs["counterfactual_example"])
    if insight:
        for title, key in (("Risks and limitations", "risks_and_limitations"), ("Next steps", "next_steps")):
            if insight.get(key):
                with st.expander(title):
                    for item in insight[key]:
                        st.markdown(f"- {item}")
    if recs and recs.get("notes"):
        with st.expander("Notes"):
            for note in recs["notes"]:
                st.markdown(f"- {note}")

profile = None
if detail.get("dataset_id"):
    profile = common.fetch(api.dataset_profile, detail["dataset_id"])
actionable = (recs or {}).get("actionable_features") or []

with tab_record:
    if profile is None:
        st.info("The what-if simulator needs the run's uploaded dataset (runs started outside the UI have none).",
                icon=":material/info:")
    else:
        record_simulator(run_id, profile, detail.get("target"), actionable, (recs or {}).get("what_if_example"))

with tab_segment:
    if profile is None:
        st.info("The segment simulator needs the run's uploaded dataset.", icon=":material/info:")
    else:
        segment_simulator(run_id, profile, detail.get("target"))

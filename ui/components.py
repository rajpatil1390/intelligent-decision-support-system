"""Reusable visual pieces of the pages (cards, chips, zebra tables, what-if inputs and results) and the pure
helpers behind them (typical row, value coercion, change detection). Inputs are the backend's JSON responses.
Charts live in ui/charts.py."""

import html
import math
from typing import Any

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from ui import charts
from ui.common import (
    STATE_LABELS,
    fmt_num,
    fmt_pct,
    fmt_time,
    humanize,
    task_label,
    verified_mark,
)

CONFIDENCE_KINDS = ("high", "medium", "low")
TIER_KINDS = {"High": "high", "Medium": "medium", "Low": "low"}
ZEBRA = "#F5F3FF"
# Pipeline stages shown on the Run Monitor; INIT/VALIDATE/NEEDS_INPUT belong to "Understand".
PIPELINE: list[tuple[str, str]] = [
    ("PARSE", "Understand"), ("PROFILE", "Profile"), ("PLAN", "Plan"), ("TRAIN", "Train"), ("VERIFY", "Verify"),
    ("EXPLAIN", "Explain"), ("INSIGHT", "Insight"), ("REPORT", "Report"), ("DONE", "Done"),
]
STAGE_OF = {"INIT": "PARSE", "VALIDATE": "PARSE", "NEEDS_INPUT": "PARSE"}
CHANGEABLE_TYPES = ("numeric", "categorical", "boolean")
FILTER_OPS_NUMERIC = ["==", "!=", "<", "<=", ">", ">="]
FILTER_OPS_CATEGORICAL = ["==", "!=", "in"]
CHART_CAPTIONS = {
    "feature_importance": "Which inputs matter most overall",
    "shap_summary": "How each input pushes individual predictions up or down",
    "confusion_matrix": "Test-set predictions vs actual outcomes",
    "forecast": "Forecast with its uncertainty band",
    "actual_vs_pred": "Test-set actual vs predicted values",
    "cluster_sizes": "How many records fall in each cluster",
}


def esc(value: Any) -> str:
    return html.escape(str(value))


def badge(label: str, kind: str = "info") -> str:
    return f'<span class="idss-badge {esc(kind)}">{esc(label)}</span>'


def card(title: str, body_html: str = "", css_class: str = "") -> None:
    """A bordered card; `title` is escaped, `body_html` must already be safe HTML."""
    st.markdown(f'<div class="idss-card {esc(css_class)}"><h4>{esc(title)}</h4>{body_html}</div>',
                unsafe_allow_html=True)


def chip(label: str, kind: str = "neutral", sub: str = "") -> str:
    """A rounded chip with a coloured dot (risk tiers: high / medium / low)."""
    extra = f' <span class="idss-chip-sub">{esc(sub)}</span>' if sub else ""
    return f'<span class="idss-chip {esc(kind)}">{esc(label)}{extra}</span>'


def chips(items: list[str]) -> None:
    st.markdown(f'<div class="idss-chips">{"".join(items)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- tables


def _display(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return fmt_num(value) if isinstance(value, (int, float)) else str(value)


def zebra(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> pd.DataFrame | Styler:
    """Alternating row shading for st.dataframe (a canvas, so rows are tinted through a pandas Styler).

    Streamlit shows the Styler's display strings, so every cell is formatted here: numbers as `fmt_num`,
    missing as '—', and `formats` ({column: "{:.1f}%"}) for columns whose column_config sets a format.
    """
    if frame.empty:
        return frame
    styler = frame.style.format(_display)
    for column, pattern in (formats or {}).items():
        if column in frame.columns:
            styler = styler.format(pattern, subset=[column], na_rep="—")
    return styler.apply(lambda col: [f"background-color: {ZEBRA}" if i % 2 else "" for i in range(len(col))], axis=0)


def table(frame: pd.DataFrame, *, formats: dict[str, str] | None = None, **kwargs: Any) -> None:
    """st.dataframe with zebra rows (index hidden)."""
    st.dataframe(zebra(frame, formats), hide_index=True, **kwargs)


# --------------------------------------------------------------------------- home


STEP_ICONS = ("📤", "💬", "🤖", "🎯")


def flow_diagram(steps: list[tuple[str, str]], icons: tuple[str, ...] = STEP_ICONS) -> None:
    """Colourful numbered step cards joined by arrows (e.g. Upload -> Ask -> Analyse -> Decide)."""
    parts = []
    for number, (title, text) in enumerate(steps, start=1):
        if number > 1:
            parts.append('<div class="idss-arrow">→</div>')
        icon = icons[(number - 1) % len(icons)]
        parts.append(f'<div class="idss-step c{(number - 1) % 4 + 1}"><div class="idss-step-head">'
                     f'<div class="idss-step-icon">{icon}</div><div><div class="idss-step-num">Step {number}</div>'
                     f'<div class="idss-step-title">{esc(title)}</div></div></div>'
                     f'<div class="idss-step-text">{esc(text)}</div></div>')
    st.markdown(f'<div class="idss-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def runs_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metric = run.get("primary_metric")
        value = run.get("metric_value")
        rows.append({
            "Run": run["run_id"],
            "Question": run.get("question", ""),
            "State": run.get("state", ""),
            "Progress": run.get("progress", 0),
            "Task": task_label(run.get("task_type")) if run.get("task_type") else "—",
            "Best model": run.get("best_model") or "—",
            "Score": f"{metric} = {fmt_num(value)}" if metric and value is not None else "—",
            "Verified": verified_mark(run.get("verification_passed")),
            "Created": fmt_time(run.get("created_at")),
        })
    return pd.DataFrame(rows)


def runs_table(runs: list[dict[str, Any]], height: int | str = "auto") -> None:
    table(
        runs_frame(runs), formats={"Progress": "{:d}%"}, height=height,
        column_config={
            "Question": st.column_config.TextColumn(width="large"),
            "Progress": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
        },
    )


# --------------------------------------------------------------------------- datasets


def column_stats_text(column: dict[str, Any]) -> str:
    stats = column.get("stats") or {}
    semantic = column.get("semantic_type")
    if semantic == "numeric" and "min" in stats:
        return f"{fmt_num(stats.get('min'))} – {fmt_num(stats.get('max'))} (median {fmt_num(stats.get('median'))})"
    if stats.get("top_values"):
        return ", ".join(f"{value} ({count:,})" for value, count in list(stats["top_values"].items())[:3])
    if semantic == "datetime" and "min" in stats:
        return f"{str(stats['min'])[:10]} → {str(stats['max'])[:10]}"
    if semantic == "text" and "avg_length" in stats:
        return f"avg {fmt_num(stats['avg_length'])} chars"
    samples = column.get("sample_values") or []
    return ", ".join(str(value) for value in samples[:3])


def profile_columns_frame(profile: dict[str, Any]) -> pd.DataFrame:
    target = profile.get("suggested_target")
    return pd.DataFrame([{
        "Column": column["name"] + ("  🎯" if column["name"] == target else ""),
        "Type": column.get("semantic_type", ""),
        "Missing %": float(column.get("missing_pct", 0.0)),
        "Unique": int(column.get("n_unique", 0)),
        "Values": column_stats_text(column),
    } for column in profile.get("columns", [])])


def profile_table(profile: dict[str, Any]) -> None:
    table(
        profile_columns_frame(profile), formats={"Missing %": "{:.1f}%", "Unique": "{:,}"},
        column_config={
            "Missing %": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f%%"),
            "Values": st.column_config.TextColumn(width="large"),
        },
    )


# --------------------------------------------------------------------------- run monitor


def pipeline(state: str, events: list[dict[str, Any]] | None = None) -> None:
    """Stage chips: finished stages green, the current one highlighted, waiting yellow, a failed stage orange."""
    keys = [key for key, _ in PIPELINE]
    current = STAGE_OF.get(state, state)
    if state == "FAILED":  # the stage the run failed in: the source of its FAILED event
        source = next((e.get("from_state") for e in reversed(events or []) if e.get("to_state") == "FAILED"), "")
        current = STAGE_OF.get(source or "", source or "")
    index = keys.index(current) if current in keys else 0
    parts = []
    for position, (_, label) in enumerate(PIPELINE):
        if state == "DONE" or position < index:
            kind, mark = "done", "✓"
        elif position == index:
            kind, mark = {"FAILED": ("failed", "✕"), "NEEDS_INPUT": ("waiting", "?")}.get(state, ("current", "●"))
        else:
            kind, mark = "", "○"
        parts.append(f'<span class="idss-stage {kind}">{mark} {esc(label)}</span>')
    st.markdown(f'<div class="idss-pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


def timeline(events: list[dict[str, Any]]) -> None:
    """Vertical event timeline, oldest first."""
    if not events:
        st.caption("No events yet.")
        return
    items = []
    for event in events:
        state = event.get("to_state", "")
        kind = {"FAILED": "failed", "DONE": "done", "NEEDS_INPUT": "waiting"}.get(state, "")
        source = event.get("from_state")
        head = f"{esc(source)} → {esc(state)}" if source else esc(state)
        label = STATE_LABELS.get(state, "")
        items.append(
            f'<div class="idss-event {kind}"><div class="idss-event-head">{head}'
            f'<span class="idss-event-time">{esc(fmt_time(event.get("ts"), with_seconds=True))}'
            f'{" · " + esc(label) if label else ""}</span></div>'
            f'<div class="idss-event-msg">{esc(event.get("message", ""))}</div></div>'
        )
    st.markdown(f'<div class="idss-timeline">{"".join(items)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- results


def leaderboard_frame(board: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in board.get("entries", []):
        rows.append({
            "": "★" if entry.get("selected") else "",
            "Rank": entry.get("rank"),
            "Model": entry.get("model", ""),
            "Status": entry.get("status", ""),
            f"Valid {board.get('primary_metric', 'score')}": entry.get("valid_score"),
            "Train time (s)": entry.get("train_time_s"),
            "Inference (ms/row)": entry.get("inference_ms_per_row"),
            "Size (KB)": entry.get("model_size_kb"),
            "Trials": entry.get("n_trials", 0),
            "Notes": "; ".join(entry.get("violations") or []) or (entry.get("error") or ""),
        })
    return pd.DataFrame(rows)


def checks_frame(verification: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Result": "✅" if check.get("passed") else "❌",
        "Check": humanize(check.get("name", "")),
        "Detail": check.get("detail", ""),
    } for check in verification.get("checks", [])])


# --------------------------------------------------------------------------- decision center


def confidence_badge(confidence: str) -> str:
    kind = confidence if confidence in CONFIDENCE_KINDS else "low"
    return badge(f"{kind} confidence", kind)


def recommendation_card(rec: dict[str, Any]) -> None:
    confidence = rec.get("confidence", "low")
    kind = confidence if confidence in CONFIDENCE_KINDS else "low"
    impact = rec.get("expected_impact")
    impact_html = (f'<div class="idss-impact">📈 <b>Expected impact:</b> {esc(fmt_num(impact))} '
                   f'{esc(rec.get("impact_unit", ""))}</div>') if impact is not None else ""
    st.markdown(
        f'<div class="idss-card idss-rec {kind}"><h4><span>{esc(rec.get("action", ""))}</span>'
        f'{confidence_badge(confidence)}</h4><p>{esc(rec.get("rationale", ""))}</p>{impact_html}</div>',
        unsafe_allow_html=True)
    evidence = rec.get("evidence") or {}
    if evidence:
        with st.expander("Evidence", expanded=False):
            table(pd.DataFrame([{"Measure": humanize(k), "Value": fmt_num(v)} for k, v in evidence.items()]))


def tiers_frame(tiers: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Tier": tier["name"],
        "Records": int(tier.get("count", 0)),
        "Share": fmt_pct(tier.get("share")),
        "Probability range": f"{fmt_num(tier.get('min_probability'))} – {fmt_num(tier.get('max_probability'))}",
        "Average probability": fmt_num(tier.get("avg_probability")),
        "Top drivers": ", ".join(tier.get("top_drivers") or []),
    } for tier in tiers])


def tier_chips(tiers: list[dict[str, Any]]) -> None:
    """One chip per risk tier: High orange/red, Medium yellow, Low green."""
    chips([chip(f"{tier['name']} risk", TIER_KINDS.get(tier["name"], "neutral"),
                f"{int(tier.get('count', 0)):,} records · {fmt_pct(tier.get('share'))}") for tier in tiers])


def clusters_frame(clusters: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        row: dict[str, Any] = {"Cluster": cluster["cluster"], "Records": cluster.get("size", 0),
                               "Share": fmt_pct(cluster.get("share"))}
        row.update({k: fmt_num(v) if isinstance(v, float) else v for k, v in (cluster.get("summary") or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- what-if helpers (pure)


def coerce_value(column: dict[str, Any], value: Any) -> Any:
    """Convert a value shown as text (profile top values, typed input) back to the column's own type."""
    if not isinstance(value, str):
        return value
    dtype = str(column.get("dtype", "")).lower()
    try:
        if dtype.startswith(("int", "uint")):
            return int(float(value))
        if dtype.startswith("float"):
            return float(value)
        if dtype in ("bool", "boolean"):
            return value.strip().lower() in ("true", "1", "yes")
    except ValueError:
        return value
    return value


def category_options(column: dict[str, Any]) -> list[Any]:
    top = (column.get("stats") or {}).get("top_values") or {}
    values = list(top) or [str(v) for v in column.get("sample_values") or []]
    return [coerce_value(column, value) for value in values]


def typical_value(column: dict[str, Any], time_column: str | None = None) -> Any:
    """Median for numbers, most common category, latest date for the time column, else a sample value."""
    stats = column.get("stats") or {}
    semantic = column.get("semantic_type")
    samples = column.get("sample_values") or []
    if semantic == "numeric" and stats.get("median") is not None:
        median = float(stats["median"])
        return int(round(median)) if stats.get("integer") else median
    if semantic in ("categorical", "boolean"):
        options = category_options(column)
        if options:
            return options[0]
    if semantic == "datetime" and column["name"] == time_column and stats.get("max"):
        latest = str(stats["max"])
        if latest.endswith("T00:00:00") and samples and "T" not in str(samples[0]) and ":" not in str(samples[0]):
            latest = latest[:10]
        return latest
    return samples[0] if samples else None


def typical_row(profile: dict[str, Any], target: str | None) -> dict[str, Any]:
    """A representative record built from the profile (all columns except the target)."""
    time_column = profile.get("time_column")
    row = {}
    for column in profile.get("columns", []):
        if column["name"] == target:
            continue
        value = typical_value(column, time_column)
        if value is not None:
            row[column["name"]] = value
    return row


def changeable_columns(profile: dict[str, Any], target: str | None) -> list[dict[str, Any]]:
    """Columns a what-if may change: numeric, categorical or boolean inputs (not the target)."""
    return [c for c in profile.get("columns", []) if c["name"] != target and c.get("semantic_type") in CHANGEABLE_TYPES]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def diff_changes(base: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """The inputs whose scenario value differs from the base record."""
    return {name: value for name, value in scenario.items() if not _same(base.get(name), value)}


def number_bounds(column: dict[str, Any]) -> tuple[float | int | None, float | int | None, bool]:
    stats = column.get("stats") or {}
    integer = bool(stats.get("integer"))
    low, high = stats.get("min"), stats.get("max")
    if integer:
        return (int(math.floor(low)) if low is not None else None,
                int(math.ceil(high)) if high is not None else None, True)
    return (float(low) if low is not None else None, float(high) if high is not None else None, False)


# --------------------------------------------------------------------------- what-if widgets


def value_input(column: dict[str, Any], value: Any, *, key: str, label: str | None = None) -> Any:
    """Input widget for one column: number input within the observed range, or a select box of categories."""
    name = column["name"]
    label = label or name
    if column.get("semantic_type") == "numeric":
        low, high, integer = number_bounds(column)
        if integer:
            current = int(round(float(value))) if value is not None else (low or 0)
            if low is not None and high is not None:
                current = min(max(current, low), high)
            return st.number_input(label, min_value=low, max_value=high, value=current, step=1, key=key,
                                   help=f"Observed range {fmt_num(low)} – {fmt_num(high)}")
        span = (high - low) if low is not None and high is not None else 1.0
        decimals = 2 if span >= 1 else 4
        current_f = float(value) if value is not None else float(low or 0.0)
        if low is not None and high is not None:
            current_f = min(max(current_f, low), high)
        return st.number_input(label, min_value=low, max_value=high, value=current_f, step=10 ** -decimals,
                               format=f"%.{decimals}f", key=key,
                               help=f"Observed range {fmt_num(low)} – {fmt_num(high)}")
    options = category_options(column)
    if value is not None and value not in options:
        options = [value, *options]
    more = int(column.get("n_unique", 0)) > len(options)
    chosen = st.selectbox(label, options, index=options.index(value) if value in options else 0, key=key,
                          format_func=str, accept_new_options=more,
                          help="Most common values; type to enter another." if more else None)
    return coerce_value(column, chosen)


def is_probability(output: str) -> bool:
    return "probability" in (output or "").lower()


def fmt_output(value: Any, output: str) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and is_probability(output):
        return fmt_pct(float(value))
    return fmt_num(value) if isinstance(value, (int, float)) else str(value)


def what_if_result(result: dict[str, Any]) -> None:
    """Before / after / delta of a single-record what-if, with a before/after bar chart."""
    output = result.get("output", "")
    st.caption(f"Model output: {output}")
    before, after, delta = result.get("before"), result.get("after"), result.get("delta")
    col1, col2, col3 = st.columns(3)
    col1.metric("Before", fmt_output(before, output))
    delta_text = None
    if delta is not None:
        delta_text = (f"{delta * 100:+.1f} pts" if is_probability(output) else f"{delta:+,.3f}")
    col2.metric("After", fmt_output(after, output), delta=delta_text, delta_color="off")  # neutral: up is not always good
    pct = result.get("delta_pct")
    col3.metric("Change", f"{pct:+.1f}%" if pct is not None else ("changed" if before != after else "no change"))
    changes = result.get("changes") or {}
    figure = charts.before_after(before, after, output)
    left, right = st.columns([1.3, 1], vertical_alignment="center")
    if figure is not None:
        with left:
            charts.show(figure, alt="Model output before and after the change")
    with right if figure is not None else left:
        if changes:
            st.markdown("**Changes simulated**")
            chips([chip(f"{k} → {v}") for k, v in changes.items()])
    for warning in result.get("warnings") or []:
        st.warning(warning, icon=":material/warning:")


def segment_result(result: dict[str, Any]) -> None:
    """Average output of a segment before and after the changes."""
    output = result.get("output", "")
    st.caption(f"Model output: average {output}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records in segment", f"{int(result.get('count', 0)):,}")
    col2.metric("Average before", fmt_output(result.get("avg_before"), output))
    delta = result.get("delta")
    delta_text = None
    if delta is not None:
        delta_text = f"{delta * 100:+.1f} pts" if is_probability(output) else f"{delta:+,.3f}"
    col3.metric("Average after", fmt_output(result.get("avg_after"), output), delta=delta_text, delta_color="off")
    pct = result.get("delta_pct")
    col4.metric("Change", f"{pct:+.1f}%" if pct is not None else "—")
    figure = charts.before_after(result.get("avg_before"), result.get("avg_after"), output,
                                 labels=("Average before", "Average after"))
    if figure is not None:
        left, _ = st.columns([1.3, 1])
        with left:
            charts.show(figure, alt="Average model output of the segment before and after the change")
    for warning in result.get("warnings") or []:
        st.warning(warning, icon=":material/warning:")

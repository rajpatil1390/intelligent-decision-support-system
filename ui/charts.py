"""Plotly charts of the UI in the IDSS palette: leaderboard, key drivers, risk-tier donut, forecast band, cluster
sizes and what-if before/after. Builders are pure (backend JSON -> go.Figure); `show()` renders one.

The palette fails colour-vision checks for some pairs (pink/green, yellow on white), so identity is never carried
by colour alone: every bar and slice is directly labelled and multi-series charts have a legend.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.styles import BLUE, GREEN, MUTED, ORANGE, PALETTE, PURPLE, TEXT, YELLOW

FONT = "'Source Sans', 'Source Sans Pro', 'Segoe UI', sans-serif"
GRID = "#ECEBFA"
TIER_COLORS = {"High": ORANGE, "Medium": YELLOW, "Low": GREEN}
# Effect of a higher value of the feature on the model output.
DIRECTION_STYLE: dict[str, tuple[str, str, str]] = {
    "increases": (ORANGE, "▲", "Higher value raises the output"),
    "decreases": (GREEN, "▼", "Higher value lowers the output"),
    "mixed": (PURPLE, "◆", "Mixed / non-linear effect"),
}
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _base(fig: go.Figure, height: int, *, legend: bool = False) -> go.Figure:
    """Clean shared look: transparent background, no grid clutter, readable 14px labels, rounded bars."""
    fig.update_layout(
        template="plotly_white", height=height, margin={"l": 8, "r": 28, "t": 36 if legend else 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 14, "color": TEXT},
        hoverlabel={"bgcolor": "white", "bordercolor": PURPLE, "font": {"family": FONT, "size": 14, "color": TEXT}},
        showlegend=legend, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None,
                                   "font": {"size": 13}},
        bargap=0.32, barcornerradius=6,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": MUTED}, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": TEXT}, automargin=True)
    return fig


def show(fig: go.Figure, *, alt: str, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, alt=alt, key=key)


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


# --------------------------------------------------------------------------- results


def leaderboard(board: dict[str, Any]) -> go.Figure | None:
    """Horizontal bars of every scored candidate's validation score; the selected model is purple, best on top."""
    entries = [e for e in board.get("entries", []) if isinstance(e.get("valid_score"), (int, float))]
    if not entries:
        return None
    maximize = board.get("direction", "maximize") == "maximize"
    entries.sort(key=lambda e: float(e["valid_score"]), reverse=not maximize)  # plotly draws the first bar at the bottom
    names = [("★ " if e.get("selected") else "") + str(e.get("model", "")) for e in entries]
    scores = [float(e["valid_score"]) for e in entries]
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h", marker={"color": [PURPLE if e.get("selected") else BLUE for e in entries]},
        text=[_fmt(s) for s in scores], textposition="outside", textfont={"color": TEXT, "size": 14},
        cliponaxis=False, hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    metric = board.get("primary_metric", "score")
    fig.update_xaxes(title={"text": f"Validation {metric} ({'higher' if maximize else 'lower'} is better)",
                            "font": {"color": MUTED, "size": 13}})
    fig = _base(fig, max(220, 52 * len(entries) + 100))
    fig.update_layout(margin_b=56)  # room for the axis title
    return fig


def drivers(features: list[dict[str, Any]], top: int = 10) -> go.Figure | None:
    """Share of the explanation per original column, coloured (and marked ▲/▼/◆) by the direction of the effect."""
    items = [f for f in features[:top] if f.get("feature") is not None]
    if not items:
        return None
    order = [str(f["feature"]) for f in reversed(items)]  # largest share on top
    fig = go.Figure()
    for direction, (color, mark, label) in DIRECTION_STYLE.items():
        group = [f for f in items if (f.get("direction") or "mixed") == direction]
        if not group:
            continue
        fig.add_trace(go.Bar(
            name=label, orientation="h", marker={"color": color},
            x=[float(f.get("share", 0.0)) for f in group], y=[str(f["feature"]) for f in group],
            text=[f"{mark} {float(f.get('share', 0.0)):.1%}" for f in group], textposition="outside",
            textfont={"color": TEXT, "size": 14}, cliponaxis=False,
            customdata=[f.get("detail") or label for f in group],
            hovertemplate="<b>%{y}</b>: %{x:.1%} of the explanation<br>%{customdata}<extra></extra>",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=order)
    fig.update_xaxes(tickformat=".0%", title={"text": "Share of the explanation", "font": {"color": MUTED, "size": 13}})
    fig = _base(fig, max(240, 38 * len(items) + 120), legend=True)
    fig.update_layout(margin_b=56)  # room for the axis title
    return fig


# --------------------------------------------------------------------------- decision center


def tier_donut(tiers: list[dict[str, Any]]) -> go.Figure | None:
    """Donut of records per risk tier (High orange, Medium yellow, Low green) with the total in the middle."""
    rows = [t for t in tiers if t.get("name")]
    if not rows:
        return None
    names = [str(t["name"]) for t in rows]
    counts = [int(t.get("count", 0)) for t in rows]
    fig = go.Figure(go.Pie(
        labels=[f"{name} risk" for name in names], values=counts, hole=0.62, sort=False, direction="clockwise",
        marker={"colors": [TIER_COLORS.get(name, PURPLE) for name in names], "line": {"color": "#ffffff", "width": 3}},
        textinfo="label+percent", textposition="outside", textfont={"size": 14, "color": TEXT},
        hovertemplate="<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{sum(counts):,}</b><br><span style='font-size:13px;color:{MUTED}'>records</span>",
                       showarrow=False, font={"size": 26, "color": TEXT})
    fig = _base(fig, 330)
    fig.update_layout(margin={"l": 30, "r": 30, "t": 30, "b": 30})
    return fig


def forecast(points: list[dict[str, Any]]) -> go.Figure | None:
    """Forecast line with its shaded uncertainty band (the page caption states how the band is computed)."""
    frame = pd.DataFrame(points)
    if frame.empty or "prediction" not in frame:
        return None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    fig = go.Figure()
    if {"lower", "upper"} <= set(frame.columns):
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["upper"], mode="lines", line={"width": 0},
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["lower"], mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor="rgba(9, 132, 227, 0.16)",
                                 name="Uncertainty band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=frame["timestamp"], y=frame["prediction"], mode="lines+markers", name="Forecast",
        line={"color": PURPLE, "width": 3},
        marker={"size": 8, "color": PURPLE, "line": {"color": "#ffffff", "width": 2}},
        customdata=frame[["lower", "upper"]].to_numpy() if {"lower", "upper"} <= set(frame.columns) else None,
        hovertemplate=("<b>%{x|%d %b %Y}</b><br>Forecast %{y:,.2f}"
                       + ("<br>Band %{customdata[0]:,.2f} – %{customdata[1]:,.2f}" if "lower" in frame else "")
                       + "<extra></extra>"),
    ))
    fig = _base(fig, 360, legend=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, tickfont={"color": MUTED})
    fig.update_layout(hovermode="x unified")
    return fig


def cluster_sizes(clusters: list[dict[str, Any]]) -> go.Figure | None:
    """Records per cluster, one palette colour per cluster (fixed order), labelled with count and share."""
    if not clusters:
        return None
    names = [f"Cluster {c['cluster']}" for c in clusters]
    sizes = [int(c.get("size", 0)) for c in clusters]
    shares = [float(c.get("share", 0.0)) for c in clusters]
    fig = go.Figure(go.Bar(
        x=names, y=sizes, marker={"color": [PALETTE[i % len(PALETTE)] for i in range(len(clusters))]},
        text=[f"{size:,} · {share:.0%}" for size, share in zip(sizes, shares, strict=True)], textposition="outside",
        textfont={"color": TEXT, "size": 14}, cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,} records<extra></extra>",
    ))
    fig.update_yaxes(showticklabels=False)
    return _base(fig, 300)


def before_after(before: Any, after: Any, output: str, *, labels: tuple[str, str] = ("Before", "After")) -> go.Figure | None:
    """Two bars: the model output before and after a what-if change (neutral colours: up is not always good)."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before, after)):
        return None
    probability = "probability" in (output or "").lower()
    values = [float(before), float(after)]
    text = [f"{v:.1%}" if probability else _fmt(v) for v in values]
    fig = go.Figure(go.Bar(
        x=list(labels), y=values, marker={"color": [BLUE, PURPLE]}, text=text, textposition="outside",
        textfont={"color": TEXT, "size": 16}, cliponaxis=False, width=0.5,
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    if probability:
        fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, tickfont={"color": MUTED})
    fig.update_xaxes(tickfont={"color": TEXT, "size": 15})
    return _base(fig, 280)

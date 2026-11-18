"""Draw the report / poster / slide diagrams with matplotlib only (no external tools), 300 dpi PNG.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\make_diagrams.py
Writes docs/diagrams/{system_architecture,agent_workflow,sequence_one_run,data_flow,er_diagram,use_case,
eight_week_timeline,dfd_level0,class_diagram,activity_diagram,deployment_diagram}.png. The Mermaid sources of the
first seven diagrams are in docs/architecture.md. MEMBERS / TIMELINE below hold the team and the task split shown in
the Gantt chart.
"""

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "diagrams"
DPI = 300

# Same palette and order as scripts/evaluate.py (neighbours stay distinct for colour-blind readers).
PURPLE, GREEN, ORANGE, BLUE, PINK, YELLOW = "#6C5CE7", "#00B894", "#E17055", "#0984E3", "#FD79A8", "#FDCB6E"
INK, MUTED, LINE, PAPER = "#2D3436", "#636E72", "#B2BEC3", "#FFFFFF"

# Gantt chart: (name, focus) per member, and (task, first week, weeks, member index or None = whole team).
MEMBERS = [
    ("Raj Patil", "GenAI / agents lead"),
    ("Bhavesh Patil", "ML engine"),
    ("Chaitany Naval", "backend, API, database"),
    ("Varun Patil", "UI, testing, documentation"),
]
MEMBER_COLORS = [PURPLE, GREEN, ORANGE, BLUE]
TIMELINE = [
    ("Foundation: LLM client, schemas, prompt loader", 1, 1, 0),
    ("Foundation: settings, logging, workspace layout", 1, 1, 2),
    ("Data module: loading, profiling, cleaning, splits", 2, 1, 1),
    ("Requirement Agent + Orchestrator state machine", 3, 1, 0),
    ("AutoML engine: model zoo, Optuna tuner, trainer", 4, 1, 1),
    ("Knowledge base (offline RAG) + Knowledge Agent", 5, 1, 0),
    ("Verifier + threshold tuning", 5, 1, 1),
    ("Sandbox + Analysis Agent", 5, 1, 2),
    ("Decision layer: SHAP, what-if, recommendations", 6, 1, 1),
    ("Insight Agent + grounding check", 6, 1, 0),
    ("PDF / Markdown report builder", 6, 1, 3),
    ("FastAPI backend + SQLite persistence", 7, 1, 2),
    ("Streamlit UI (8 pages)", 7, 1, 3),
    ("Evaluation: benchmark, ablation, robustness", 8, 1, 1),
    ("Dashboard theme, screenshots, documentation", 8, 1, 3),
    ("Testing, code review, integration", 1, 8, None),
]
MILESTONES = ["Foundation", "Data module", "Requirement +\nManager", "AutoML engine", "RAG + Verifier\n+ Sandbox",
              "Decision layer", "API + UI", "Evaluation\n+ docs"]


# --------------------------------------------------------------------------- drawing helpers


def _luminance(color: str) -> float:
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def text_on(color: str) -> str:
    """White text where it has at least 3:1 contrast on `color`, dark ink otherwise."""
    return PAPER if 1.05 / (_luminance(color) + 0.05) >= 3.0 else INK


def tint(color: str, amount: float) -> str:
    """Mix `color` with white; amount 0 = color, 1 = white."""
    channels = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02X}" for c in channels)


Box = tuple[float, float, float, float]  # centre x, centre y, width, height


def canvas(width: float, height: float, title: str, subtitle: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    """A figure whose data coordinates are inches, so shapes and arrows keep their proportions."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(width / 2, height - 0.45, title, ha="center", va="center", fontsize=24, fontweight="bold", color=INK)
    if subtitle:
        ax.text(width / 2, height - 0.95, subtitle, ha="center", va="center", fontsize=14, color=MUTED)
    return fig, ax


def box(ax: plt.Axes, cx: float, cy: float, w: float, h: float, title: str, color: str, *, sub: str | None = None,
        filled: bool = True, size: float = 14, sub_size: float = 11, radius: float = 0.14, lw: float = 2.0) -> Box:
    face = color if filled else tint(color, 0.86)
    ink = text_on(face) if filled else INK
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=color, linewidth=lw, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center", fontsize=size, fontweight="bold", color=ink,
                zorder=4, linespacing=1.15)
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center", fontsize=sub_size, color=ink, zorder=4,
                linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=size, fontweight="bold", color=ink, zorder=4,
                linespacing=1.15)
    return cx, cy, w, h


def band(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, label: str, color: str) -> None:
    """A layer: a light tinted strip with its name on the left."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.2",
                                facecolor=tint(color, 0.93), edgecolor=tint(color, 0.55), linewidth=1.5, zorder=1))
    ax.text(x0 + 0.25, (y0 + y1) / 2, label, ha="left", va="center", fontsize=14, fontweight="bold", color=color,
            zorder=2)


def left(b: Box, dy: float = 0.0) -> tuple[float, float]:
    return b[0] - b[2] / 2, b[1] + dy


def right(b: Box, dy: float = 0.0) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + dy


def top(b: Box, dx: float = 0.0) -> tuple[float, float]:
    return b[0] + dx, b[1] + b[3] / 2


def bottom(b: Box, dx: float = 0.0) -> tuple[float, float]:
    return b[0] + dx, b[1] - b[3] / 2


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = MUTED,
          label: str | None = None, rad: float = 0.0, dashed: bool = False, lw: float = 2.0, size: float = 11,
          offset: tuple[float, float] = (0.0, 0.0), label_color: str = INK, both: bool = False) -> None:
    style = "<|-|>" if both else "-|>"
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=20, color=color, linewidth=lw,
                                 linestyle=(0, (5, 4)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1, zorder=2))
    if label:
        (x1, y1), (x2, y2) = start, end
        # Midpoint of matplotlib's arc3 Bezier curve: chord midpoint + rad/2 * (dy, -dx).
        mx, my = (x1 + x2) / 2 + rad * (y2 - y1) / 2, (y1 + y2) / 2 - rad * (x2 - x1) / 2
        ax.text(mx + offset[0], my + offset[1], label, ha="center", va="center", fontsize=size, color=label_color,
                zorder=5, linespacing=1.15,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.95})


def legend(ax: plt.Axes, items: Sequence[tuple[str, str]], x: float, y: float, *, step: float = 2.9,
           size: float = 13) -> None:
    """A one-row legend of colour swatches starting at (x, y)."""
    for index, (label, color) in enumerate(items):
        left_x = x + index * step
        ax.add_patch(FancyBboxPatch((left_x, y - 0.13), 0.42, 0.26, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=color, edgecolor=color, zorder=3))
        ax.text(left_x + 0.55, y, label, ha="left", va="center", fontsize=size, color=INK)


def save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 1. system architecture


def system_architecture() -> Path:
    fig, ax = canvas(17, 12, "IDSS — system architecture",
                     "LLM agents plan and explain; deterministic engines compute; every layer works offline")
    x0, x1, inner0 = 0.3, 13.9, 2.45
    width = x1 - 0.2 - inner0

    def row(y: float, h: float, items: list[tuple[str, str, str]], filled: bool = True, gap: float = 0.22,
            size: float = 13, sub_size: float = 10.5) -> list[Box]:
        w = (width - gap * (len(items) - 1)) / len(items)
        return [box(ax, inner0 + w / 2 + i * (w + gap), y, w, h, title, color, sub=sub, filled=filled, size=size,
                    sub_size=sub_size) for i, (title, sub, color) in enumerate(items)]

    band(ax, x0, 9.35, x1, 10.65, "Presentation", PURPLE)
    ui, docs = row(10.0, 1.0, [("Streamlit UI", "8 pages · ui/ · talks only to the API", PURPLE),
                               ("REST clients", "Swagger UI at /docs · scripts", PURPLE)], filled=False)
    band(ax, x0, 7.85, x1, 9.15, "Service", BLUE)
    api, worker, repo = row(8.5, 1.0, [("FastAPI backend", "api/main.py · upload, runs, results", BLUE),
                                       ("RunWorker", "one background run thread", BLUE),
                                       ("Repository", "SQLAlchemy 2.0 CRUD", BLUE)])
    band(ax, x0, 6.55, x1, 7.65, "Orchestration", PURPLE)
    orch = box(ax, inner0 + width / 2, 7.1, width, 0.85, "Orchestrator",
               PURPLE, sub="resumable state machine · state.json after every transition · bounded loops", size=14)
    band(ax, x0, 4.75, x1, 6.35, "Agents", GREEN)
    row(5.55, 1.35, [
        ("Requirement", "free text →\nrequirement", GREEN), ("Data", "profile, clean,\nsplit", GREEN),
        ("Knowledge", "retrieved best\npractices", GREEN), ("Model", "ranks zoo\nmodels", GREEN),
        ("Verifier", "7 checks +\nLLM review", GREEN), ("Analysis", "pandas code\nfor questions", GREEN),
        ("Insight", "grounded\nbrief", GREEN)], filled=False, gap=0.14, size=12.5, sub_size=10.5)
    band(ax, x0, 2.95, x1, 4.55, "Engines", ORANGE)
    automl, decision, sandbox, rag = row(3.75, 1.3, [
        ("AutoML engine", "model zoo · Optuna tuner\nevaluator · trainer", ORANGE),
        ("Decision layer", "SHAP · what-if\nrecommender · report", ORANGE),
        ("Sandbox", "AST guard + isolated\nchild process", ORANGE),
        ("Knowledge base", "TF-IDF index over\n10 best-practice notes", ORANGE)])
    band(ax, x0, 1.15, x1, 2.75, "Infrastructure", BLUE)
    storage, config, llm = row(1.95, 1.3, [
        ("Storage", "SQLite idss.db · workspace/runs\nstate.json · models · charts", BLUE),
        ("Settings & logging", ".env (secrets) · pydantic\nsettings · loguru", BLUE),
        ("LLM client", "used by every agent (optional)\ntimeout · ≤ 3 retries", BLUE)], filled=False)

    ax.text(15.45, 10.95, "External", ha="center", va="center", fontsize=14, fontweight="bold", color=MUTED)
    user = box(ax, 15.45, 10.0, 2.4, 1.0, "Business user", PINK, sub="web browser")
    provider = box(ax, 15.45, 1.95, 2.4, 1.3, "LLM provider", PINK, sub="OpenAI · Groq\nGemini · Ollama")
    search = box(ax, 15.45, 3.75, 2.4, 1.3, "Web search", PINK, sub="optional\nTavily · SerpAPI", filled=False)

    arrow(ax, left(user), right(docs), color=INK)
    arrow(ax, bottom(ui), top(api, -0.9), color=INK, label="HTTP / JSON", offset=(0.75, 0))
    arrow(ax, right(api), left(worker), color=INK)
    arrow(ax, right(worker), left(repo), color=INK)
    arrow(ax, bottom(worker), top(orch, worker[0] - orch[0]), color=INK, label="run_until_blocked()",
          offset=(1.2, 0))
    arrow(ax, bottom(orch), (orch[0], 6.25), color=INK, label="one handler per state", offset=(1.5, 0))
    arrow(ax, (orch[0], 4.85), (orch[0], 4.45), color=INK)
    arrow(ax, right(llm), left(provider), color=PINK, both=True)
    ax.text(provider[0], provider[1] - provider[3] / 2 - 0.3, "aggregates only,\nnever raw rows", ha="center",
            va="top", fontsize=11, color=MUTED)
    arrow(ax, right(rag), left(search), color=PINK, dashed=True)
    ax.text(x1 / 2 + 0.6, 0.55, "Arrows show the main call path; every agent falls back to a deterministic path "
            "when no LLM is configured.", ha="center", va="center", fontsize=12, color=MUTED)
    return save(fig, "system_architecture.png")


# --------------------------------------------------------------------------- 2. agent workflow


def agent_workflow() -> Path:
    fig, ax = canvas(16.8, 10.5, "Orchestrator state machine (one run)",
                     "Each state has one handler that returns (next state, message); the run is saved after every step")
    w, h, xs = 2.15, 1.25, [1.75, 4.4, 7.05, 9.7, 12.35, 15.0]
    y1, y2 = 5.7, 2.75
    groups = {"requirement": PURPLE, "data": BLUE, "model": GREEN, "decision": ORANGE}

    def state(x: float, y: float, name: str, sub: str, color: str, filled: bool = True) -> Box:
        return box(ax, x, y, w, h, name, color, sub=sub, size=14, sub_size=10.5, filled=filled)

    init = state(xs[0], y1, "INIT", "relevance check\n+ data profile", groups["requirement"])
    parse = state(xs[1], y1, "PARSE", "Requirement Agent\n→ UserRequirement", groups["requirement"])
    validate = state(xs[2], y1, "VALIDATE", "sufficiency rules\n(+ LLM review)", groups["requirement"])
    profile = state(xs[3], y1, "PROFILE", "Data Agent + \nknowledge brief", groups["data"])
    plan = state(xs[4], y1, "PLAN", "Model Agent ranks\nzoo models", groups["data"])
    train = state(xs[5], y1, "TRAIN", "AutoML search,\ntest split once", groups["model"])
    verify = state(xs[5], y2, "VERIFY", "7 rule checks\n(+ LLM review)", groups["model"])
    explain = state(xs[4], y2, "EXPLAIN", "SHAP, actions,\nwhat-if example", groups["decision"])
    insight = state(xs[3], y2, "INSIGHT", "grounded\ndecision brief", groups["decision"])
    report = state(xs[2], y2, "REPORT", "report.pdf +\nreport.md", groups["decision"])
    done = state(xs[1], y2, "DONE", "results ready", INK)
    needs = state((xs[1] + xs[2]) / 2, 8.05, "NEEDS_INPUT", "clarifying questions", YELLOW)
    failed = state(xs[3] + 0.6, 8.05, "FAILED", "error recorded;\nnever crashes", PINK)

    ax.add_patch(Circle((0.3, y1), 0.16, color=INK, zorder=3))
    arrow(ax, (0.46, y1), left(init), color=INK)
    for a, b in ((init, parse), (parse, validate), (validate, profile), (profile, plan), (plan, train)):
        arrow(ax, right(a), left(b), color=INK)
    arrow(ax, bottom(train), top(verify), color=INK, label="TrainingResult", offset=(0.95, 0), size=10.5)
    arrow(ax, left(verify), right(explain), color=INK)  # checks passed
    for a, b in ((explain, insight), (insight, report), (report, done)):
        arrow(ax, left(a), right(b), color=INK)
    arrow(ax, top(verify, -0.6), bottom(plan, 0.6), color=GREEN, rad=-0.35, lw=2.4,
          label="checks failed: re-plan once\nwith the full budget", offset=(-0.15, 0), size=10.5)
    arrow(ax, (verify[0] - 0.3, y2 - h / 2), (explain[0] + 0.3, y2 - h / 2), color=GREEN, rad=-0.45,
          label="still failing → continue, flagged 'not verified'", offset=(0, -0.2), size=10.5)
    arrow(ax, top(validate, -0.5), right(needs, -0.25), color=MUTED, rad=0.25, label="missing info", size=10.5,
          offset=(0.35, 0.1))
    arrow(ax, left(needs, -0.25), top(parse, -0.5), color=MUTED, rad=0.25, size=10.5,
          label="provide_answer()\n≤ 2 revisions", offset=(-0.75, 0.05))
    arrow(ax, top(validate, 0.6), left(failed), color=PINK, dashed=True, rad=-0.2, size=10.5,
          label="still incomplete", offset=(0.1, 0.25))
    arrow(ax, bottom(init), top(done, -0.6), color=MUTED, dashed=True, label="not a data\nquestion", size=10.5,
          offset=(-0.55, 0.2))
    ax.text(failed[0] + w / 2 + 0.25, failed[1], "any state → FAILED on an\nunhandled error (logged, saved)",
            ha="left", va="center", fontsize=11, color=MUTED)
    legend(ax, [("Requirement", PURPLE), ("Data & planning", BLUE), ("Training & verification", GREEN),
                ("Decision support", ORANGE), ("Waiting for user", YELLOW), ("Failure", PINK)], 0.4, 0.75,
           step=2.75, size=12)
    return save(fig, "agent_workflow.png")


# --------------------------------------------------------------------------- 3. sequence diagram


def sequence_one_run() -> Path:
    width, height = 18.0, 14.8
    fig, ax = canvas(width, height, "One analysis run — sequence",
                     "Upload → start → background training → polling → results, report and what-if")
    names = [("User", PINK), ("Streamlit UI", PURPLE), ("FastAPI", BLUE), ("RunWorker", BLUE),
             ("Orchestrator", PURPLE), ("Agents", GREEN), ("AutoML engine", ORANGE), ("Storage", BLUE)]
    xs = [1.15 + i * 2.25 for i in range(len(names))]
    head_y, end_y = height - 1.75, 0.45
    for x, (name, color) in zip(xs, names, strict=True):
        box(ax, x, head_y, 1.95, 0.7, name, color, size=13)
        ax.plot([x, x], [head_y - 0.35, end_y], color=LINE, linewidth=1.5, linestyle=(0, (4, 4)), zorder=1)
    col = {name: x for x, (name, _) in zip(xs, names, strict=True)}

    messages: list[tuple[str, str, str, bool]] = [
        ("User", "Streamlit UI", "upload CSV", False),
        ("Streamlit UI", "FastAPI", "POST /datasets", False),
        ("FastAPI", "Storage", "save file, cache profile, insert dataset", False),
        ("FastAPI", "Streamlit UI", "201 · dataset_id + profile", True),
        ("User", "Streamlit UI", "question + Start analysis", False),
        ("Streamlit UI", "FastAPI", "POST /runs", False),
        ("FastAPI", "RunWorker", "submit(run)", False),
        ("FastAPI", "Streamlit UI", "202 · run_id", True),
        ("RunWorker", "Orchestrator", "start() · run_until_blocked()", False),
        ("Orchestrator", "Agents", "parse → validate → profile → plan", False),
        ("Orchestrator", "AutoML engine", "train: Optuna search, refit, test once", False),
        ("AutoML engine", "Orchestrator", "TrainingResult", True),
        ("Orchestrator", "Agents", "verify → explain → insight → report", False),
        ("Orchestrator", "Storage", "state.json + event after every transition", False),
        ("Streamlit UI", "FastAPI", "GET /runs/{id}", False),
        ("FastAPI", "Storage", "read state.json", False),
        ("FastAPI", "Streamlit UI", "state · progress % · events", True),
        ("User", "Streamlit UI", "open Results / Decision Center", False),
        ("Streamlit UI", "FastAPI", "GET leaderboard · explanations · report.pdf", False),
        ("FastAPI", "Streamlit UI", "JSON + PDF + chart URLs", True),
        ("User", "Streamlit UI", "what-if: change inputs", False),
        ("Streamlit UI", "FastAPI", "POST /runs/{id}/what-if", False),
        ("FastAPI", "Orchestrator", "what_if(): trained model", False),
        ("FastAPI", "Streamlit UI", "prediction before → after", True),
    ]
    step, gap = 0.47, 0.2
    y = head_y - 0.85
    poll_top = poll_bottom = 0.0
    number = 0
    for src, dst, label, reply in messages:
        if src == "Streamlit UI" and label.startswith("GET /runs/{id}"):
            y -= gap  # room for the loop frame's top edge
            poll_top = y + 0.5
        x1, x2 = col[src], col[dst]
        color = MUTED if reply else INK
        arrow(ax, (x1, y), (x2, y), color=color, dashed=reply, lw=1.6)
        if not reply:
            number += 1
        text = label if reply else f"{number}. {label}"
        ax.text((x1 + x2) / 2, y + 0.14, text, ha="center", va="bottom", fontsize=11, color=INK, zorder=5,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": PAPER, "edgecolor": "none"})
        if label.startswith("state · progress"):
            poll_bottom = y - 0.17
            y -= gap
        y -= step
    x_left, x_right = col["Streamlit UI"] - 0.75, col["Storage"] + 0.9
    ax.add_patch(Rectangle((x_left, poll_bottom), x_right - x_left, poll_top - poll_bottom, fill=False,
                           edgecolor=PURPLE, linewidth=1.6, zorder=1))
    ax.add_patch(Rectangle((x_right - 2.75, poll_top - 0.32), 2.75, 0.32, facecolor=PURPLE, edgecolor=PURPLE,
                           zorder=2))
    ax.text(x_right - 2.65, poll_top - 0.16, "loop · every 2 s until DONE", ha="left", va="center", fontsize=10.5,
            color=PAPER, fontweight="bold", zorder=3)
    ax.text(width / 2, 0.2, "Solid arrows: requests and calls · dashed arrows: responses", ha="center",
            va="center", fontsize=12, color=MUTED)
    return save(fig, "sequence_one_run.png")


# --------------------------------------------------------------------------- 4. data flow


def data_flow() -> Path:
    fig, ax = canvas(17.5, 10.2, "Data flow through IDSS",
                     "From the uploaded file to decisions; the test split is read once, the LLM sees aggregates only")
    w, h = 2.85, 1.35
    xs = [1.75 + i * 3.5 for i in range(5)]
    y1, y2, y3 = 7.35, 4.85, 2.35
    ingest = [
        box(ax, xs[0], y1, w, h, "Upload", PURPLE, sub="CSV · Excel · JSON\nParquet (≤ 50 MB)"),
        box(ax, xs[1], y1, w, h, "Load & profile", PURPLE, sub="types, missing %,\nsuggested target & task"),
        box(ax, xs[2], y1, w, h, "Clean", PURPLE, sub="duplicates, rows without\ntarget, time order"),
        box(ax, xs[3], y1, w, h, "Split 70 / 20 / 10", PURPLE, sub="stratified or\nchronological"),
        box(ax, xs[4], y1, w, h, "Preprocess", PURPLE, sub="impute · scale · one-hot\nlags, rolling, calendar"),
    ]
    model = [
        box(ax, xs[4], y2, w, h, "Tune candidates", GREEN, sub="Optuna TPE + pruning\non train → valid"),
        box(ax, xs[3], y2, w, h, "Select", GREEN, sub="best valid score within\nconstraints; threshold"),
        box(ax, xs[2], y2, w, h, "Refit + test once", GREEN, sub="train+valid → model;\nscore vs baseline"),
        box(ax, xs[1], y2, w, h, "Verify", GREEN, sub="7 rule-based checks\n(+ optional LLM review)"),
        box(ax, xs[0], y2, w, h, "Explain", ORANGE, sub="SHAP drivers, forecast\nor cluster profiles"),
    ]
    decide = [
        box(ax, xs[0], y3, w, h, "Decide", ORANGE, sub="risk tiers, actions,\nwhat-if simulation"),
        box(ax, xs[1], y3, w, h, "Insight", ORANGE, sub="brief from facts only;\nnumbers grounded"),
        box(ax, xs[2], y3, w, h, "Report", ORANGE, sub="report.pdf + report.md\n+ charts"),
    ]
    for a, b in zip(ingest, ingest[1:]):
        arrow(ax, right(a), left(b), color=INK)
    arrow(ax, bottom(ingest[-1]), top(model[0]), color=INK, label="train / valid", offset=(0.85, 0), size=10.5)
    for a, b in zip(model, model[1:]):
        arrow(ax, left(a), right(b), color=INK)
    arrow(ax, bottom(model[-1]), top(decide[0]), color=INK)
    for a, b in zip(decide, decide[1:]):
        arrow(ax, right(a), left(b), color=INK)
    arrow(ax, bottom(ingest[3], -0.6), top(model[2], 0.6), color=GREEN, dashed=True, rad=0.15,
          label="test split\n(held out)", size=10.5, offset=(0.1, 0.05))

    def store(cx: float, cy: float, sw: float, title: str, sub: str) -> Box:
        sh = 1.15
        ax.add_patch(Rectangle((cx - sw / 2, cy - sh / 2), sw, sh, facecolor=tint(BLUE, 0.9), edgecolor="none",
                               zorder=2))
        ax.plot([cx - sw / 2, cx + sw / 2], [cy + sh / 2] * 2, color=BLUE, linewidth=2.2, zorder=3)
        ax.plot([cx - sw / 2, cx + sw / 2], [cy - sh / 2] * 2, color=BLUE, linewidth=2.2, zorder=3)
        ax.add_patch(Rectangle((cx - sw / 2, cy - sh / 2), 0.16, sh, facecolor=BLUE, edgecolor="none", zorder=3))
        ax.text(cx + 0.08, cy + 0.2, title, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(cx + 0.08, cy - 0.23, sub, ha="center", va="center", fontsize=10.5, color=INK)
        return cx, cy, sw, sh

    runs = store(xs[3] + 0.05, y3 + 0.05, 3.4, "workspace/runs/<run_id>",
                 "state.json · splits\nbest_model.joblib · charts")
    db = store(xs[4] + 0.35, y3 + 0.05, 2.45, "SQLite idss.db", "datasets · runs · events")
    llm = box(ax, xs[1], 0.75, 3.4, 0.85, "LLM (optional)", PINK, sub="profile summary, metrics, facts", size=12.5,
              sub_size=10.5)
    arrow(ax, right(decide[-1]), left(runs), color=BLUE, dashed=True)
    arrow(ax, bottom(model[2], 0.8), top(runs, -0.8), color=BLUE, dashed=True, label="model, leaderboard,\nmetrics",
          size=10.5, offset=(0.2, 0.1))
    arrow(ax, right(runs), left(db), color=BLUE, dashed=True, label="index", size=10.5, offset=(0, 0.25))
    arrow(ax, bottom(decide[1]), top(llm), color=PINK, both=True, dashed=True)
    ax.text(xs[2] + 0.45, 0.75, "Never sent to the LLM: raw rows, file paths, API keys.\n"
            "LLM-written analysis code runs only in the sandbox.", ha="left", va="center", fontsize=11.5,
            color=MUTED)
    legend(ax, [("Data preparation", PURPLE), ("Modelling & verification", GREEN), ("Decision support", ORANGE),
                ("Data store", BLUE)], 2.9, 8.55, step=3.3, size=12)
    return save(fig, "data_flow.png")


# --------------------------------------------------------------------------- 5. ER diagram


def er_diagram() -> Path:
    fig, ax = canvas(17.6, 9.4, "Database schema (SQLite, SQLAlchemy 2.0)",
                     "workspace/idss.db indexes datasets, runs and their events; state.json stays the source of truth")
    row_h = 0.44

    def entity(x: float, y_top: float, w: float, name: str, color: str,
               fields: list[tuple[str, str, str]]) -> dict[str, tuple[float, float]]:
        """Draw a table; return the left/right anchor of each field row."""
        height = 0.62 + row_h * len(fields)
        ax.add_patch(FancyBboxPatch((x, y_top - height), w, height, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=PAPER, edgecolor=color, linewidth=2.2, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y_top - 0.62), w, 0.62, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=color, edgecolor=color, linewidth=2.2, zorder=3))
        ax.text(x + w / 2, y_top - 0.31, name, ha="center", va="center", fontsize=15, fontweight="bold",
                color=text_on(color), zorder=4)
        anchors = {}
        for index, (field, kind, key) in enumerate(fields):
            cy = y_top - 0.62 - row_h * (index + 0.5)
            if index % 2:
                ax.add_patch(Rectangle((x + 0.03, cy - row_h / 2), w - 0.06, row_h, facecolor=tint(color, 0.93),
                                       edgecolor="none", zorder=2))
            ax.text(x + 0.18, cy, key, ha="left", va="center", fontsize=10, fontweight="bold", color=color,
                    zorder=4)
            ax.text(x + 0.72, cy, field, ha="left", va="center", fontsize=12.5, color=INK, zorder=4,
                    fontweight="bold" if key == "PK" else "normal")
            ax.text(x + w - 0.15, cy, kind, ha="right", va="center", fontsize=11, color=MUTED, zorder=4)
            anchors[field] = (x, cy), (x + w, cy)
        return anchors

    datasets = entity(0.5, 8.3, 4.4, "datasets", PURPLE, [
        ("id", "string(32)", "PK"), ("filename", "string(255)", ""), ("path", "text", ""), ("rows", "integer", ""),
        ("cols", "integer", ""), ("uploaded_at", "datetime (UTC)", "")])
    runs = entity(6.6, 8.3, 4.6, "runs", GREEN, [
        ("run_id", "string(64)", "PK"), ("dataset_id", "→ datasets.id", "FK"), ("question", "text", ""),
        ("state", "string(20) · index", ""), ("task_type", "string(40)", ""), ("best_model", "string(64)", ""),
        ("primary_metric", "string(40)", ""), ("metric_value", "float", ""), ("verification_passed", "boolean", ""),
        ("created_at", "datetime · index", ""), ("updated_at", "datetime", ""), ("report_pdf", "text", ""),
        ("report_md", "text", "")])
    events = entity(12.9, 8.3, 4.2, "events", ORANGE, [
        ("id", "integer · auto", "PK"), ("run_id", "→ runs.run_id", "FK"), ("ts", "datetime (UTC)", ""),
        ("from_state", "string(20)", ""), ("to_state", "string(20)", ""), ("message", "text", "")])

    def relation(start: tuple[float, float], end: tuple[float, float], label: str, note: str) -> None:
        (x1, y1), (x2, y2) = start, end
        mid = (x1 + x2) / 2
        ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2], color=INK, linewidth=1.8, zorder=1)
        ax.plot([x1 + 0.12, x1 + 0.12], [y1 - 0.13, y1 + 0.13], color=INK, linewidth=1.8)  # "one" bar
        ax.plot([x1 + 0.22, x1 + 0.22], [y1 - 0.13, y1 + 0.13], color=INK, linewidth=1.8)
        for dy in (-0.16, 0.0, 0.16):  # crow's foot: "many"
            ax.plot([x2 - 0.3, x2], [y2, y2 + dy], color=INK, linewidth=1.8)
        ax.add_patch(Circle((x2 - 0.42, y2), 0.08, facecolor=PAPER, edgecolor=INK, linewidth=1.6, zorder=3))
        ax.text(mid, min(y1, y2) - 0.3, label, ha="center", va="top", fontsize=12, fontweight="bold", color=INK)
        ax.text(mid, min(y1, y2) - 0.62, note, ha="center", va="top", fontsize=10.5, color=MUTED)

    relation(datasets["id"][1], runs["dataset_id"][0], "1 : 0..*", "on delete: SET NULL")
    relation(runs["run_id"][1], events["run_id"][0], "1 : 0..*", "on delete: CASCADE")
    ax.text(0.5, 3.0, "Notes", fontsize=14, fontweight="bold", color=INK)
    notes = [
        "WAL journal mode, foreign keys on; tables created on first use.",
        "Datetimes stored as naive UTC, returned timezone-aware.",
        "The Orchestrator mirrors every transition into runs + events;",
        "a database error is only logged — state.json is the source of truth.",
    ]
    for index, note in enumerate(notes):
        ax.text(0.5, 2.55 - index * 0.42, "• " + note, fontsize=12, color=INK)
    return save(fig, "er_diagram.png")


# --------------------------------------------------------------------------- 6. use cases


def _actor(ax: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    ax.add_patch(Circle((x, y + 0.75), 0.24, facecolor=PAPER, edgecolor=color, linewidth=2.4, zorder=3))
    for xs, ys in (([x, x], [y + 0.51, y - 0.15]), ([x - 0.45, x + 0.45], [y + 0.3, y + 0.3]),
                   ([x, x - 0.35], [y - 0.15, y - 0.75]), ([x, x + 0.35], [y - 0.15, y - 0.75])):
        ax.plot(xs, ys, color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
    ax.text(x, y - 1.05, label, ha="center", va="top", fontsize=13, fontweight="bold", color=INK, linespacing=1.15)


def use_case() -> Path:
    fig, ax = canvas(17.0, 11.0, "Use cases", "What each actor does with IDSS")
    ax.add_patch(FancyBboxPatch((3.1, 0.55), 10.4, 9.15, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=tint(PURPLE, 0.96), edgecolor=PURPLE, linewidth=2, zorder=0))
    ax.text(3.35, 9.45, "IDSS", ha="left", va="center", fontsize=16, fontweight="bold", color=PURPLE)
    main = [
        "Upload a dataset and view its profile",
        "Describe the problem in plain language",
        "Answer clarifying questions",
        "Monitor run progress",
        "View model results and verification",
        "Explore key drivers (SHAP)",
        "Review insights, recommendations, risk tiers",
        "Simulate what-if scenarios and segments",
        "Ask questions about the data",
        "Download the PDF / Markdown report",
    ]
    cx, ew, eh = 6.95, 5.2, 0.68
    ys = [8.75 - i * 0.84 for i in range(len(main))]
    for label, y in zip(main, ys, strict=True):
        ax.add_patch(Ellipse((cx, y), ew, eh, facecolor=PAPER, edgecolor=GREEN, linewidth=2, zorder=2))
        ax.text(cx, y, label, ha="center", va="center", fontsize=12, color=INK, zorder=3)
    team = ["Run the offline evaluation", "Run the test suite", "Regenerate samples and diagrams"]
    tx, tw, tys = 11.8, 3.25, [3.2, 2.2, 1.2]
    for label, y in zip(team, tys, strict=True):
        ax.add_patch(Ellipse((tx, y), tw, eh, facecolor=PAPER, edgecolor=ORANGE, linewidth=2, zorder=2))
        ax.text(tx, y, label, ha="center", va="center", fontsize=11, color=INK, zorder=3)

    _actor(ax, 1.5, 5.3, "Business user", PURPLE)
    for y in ys:
        ax.plot([1.95, cx - ew / 2], [5.6, y], color=LINE, linewidth=1.4, zorder=1)
    _actor(ax, 15.6, 2.4, "Team member /\nevaluator", ORANGE)
    for y in tys:
        ax.plot([15.15, tx + tw / 2], [2.7, y], color=LINE, linewidth=1.4, zorder=1)
    provider = box(ax, 15.6, 7.3, 2.5, 1.2, "LLM provider", PINK, sub="«external system»\noptional", size=13)
    for index in (1, 6, 8):
        ax.plot([provider[0] - provider[2] / 2, cx + ew / 2], [provider[1], ys[index]], color=PINK, linewidth=1.6,
                linestyle=(0, (5, 4)), zorder=1)
    extend_x = cx + ew / 2 - 0.6  # «extend»: Answer clarifying questions → Describe the problem
    arrow(ax, (extend_x, ys[2] + 0.22), (extend_x, ys[1] - 0.22), color=INK, dashed=True, lw=1.8)
    ax.text(extend_x + 0.2, (ys[1] + ys[2]) / 2, "«extend»", ha="left", va="center", fontsize=11.5, color=INK)
    ax.text(10.4, 0.25, "Dashed pink: an LLM is used when configured; otherwise deterministic fallbacks "
            "(except Ask Your Data).", ha="center", va="center", fontsize=11.5, color=MUTED)
    return save(fig, "use_case.png")


# --------------------------------------------------------------------------- 7. Gantt chart


def eight_week_timeline() -> Path:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(17, 9.2))
    rows = len(TIMELINE)
    for index, (task, start, weeks, member) in enumerate(TIMELINE):
        y = rows - 1 - index
        color = MEMBER_COLORS[member] if member is not None else LINE
        ax.barh(y, weeks - 0.12, left=start - 1 + 0.06, height=0.62, color=color, edgecolor=PAPER, linewidth=1.5,
                zorder=3)
        owner = MEMBERS[member][0].split()[0] if member is not None else "whole team"  # first name fits the bar
        ax.text(start - 1 + 0.14, y, owner, ha="left", va="center", fontsize=11, fontweight="bold",
                color=text_on(color), zorder=4)
    ax.set_yticks(range(rows), [task for task, *_ in reversed(TIMELINE)], fontsize=13, color=INK)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.7, rows + 0.9)
    ax.set_xticks([w + 0.5 for w in range(8)], [f"Week {w + 1}" for w in range(8)], fontsize=13, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    for week in range(9):
        ax.axvline(week, color="#DFE6E9", linewidth=1, zorder=1)
    for week, milestone in enumerate(MILESTONES):
        ax.text(week + 0.5, rows + 0.05, milestone, ha="center", va="bottom", fontsize=10.5, color=MUTED,
                linespacing=1.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("8-week project plan", fontsize=24, fontweight="bold", color=INK, pad=60)
    handles = [Patch(facecolor=color, label=f"{name} — {focus}")
               for (name, focus), color in zip(MEMBERS, MEMBER_COLORS, strict=True)]
    handles.append(Patch(facecolor=LINE, label="Whole team"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.02), ncol=3, frameon=False, fontsize=12.5)
    fig.tight_layout()
    return save(fig, "eight_week_timeline.png")


# --------------------------------------------------------------------------- 8. DFD level 0 (context diagram)


def dfd_level0() -> Path:
    fig, ax = canvas(16.5, 9.6, "Data flow diagram — Level 0 (context)",
                     "IDSS as one process; external entities exchange these data flows with it")
    centre, radius = (8.25, 4.35), 1.85
    ax.add_patch(Circle(centre, radius, facecolor=PURPLE, edgecolor=PURPLE, linewidth=2.4, zorder=3))
    ax.text(centre[0], centre[1] + 0.85, "0", ha="center", va="center", fontsize=15, fontweight="bold",
            color=PAPER, zorder=4)
    ax.text(centre[0], centre[1] - 0.05, "IDSS", ha="center", va="center", fontsize=24, fontweight="bold",
            color=PAPER, zorder=4)
    ax.text(centre[0], centre[1] - 0.8, "Intelligent Decision\nSupport System", ha="center", va="center",
            fontsize=12.5, color=PAPER, zorder=4, linespacing=1.15)
    user = box(ax, 1.75, 4.35, 2.7, 1.3, "Business user", PINK, sub="analyst / manager", size=14)
    llm = box(ax, 14.6, 6.85, 3.3, 1.3, "LLM provider", BLUE, sub="optional · OpenAI, Groq,\nGemini or local Ollama",
              size=14)
    web = box(ax, 14.6, 1.85, 3.3, 1.3, "Web search", BLUE, sub="optional · Tavily, SerpAPI", size=14,
              filled=False)

    def on_circle(degrees: float) -> tuple[float, float]:
        angle = math.radians(degrees)
        return centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)

    arrow(ax, right(user, 0.35), on_circle(165), color=INK, rad=-0.25, size=11.5, offset=(0, 0.5),
          label="dataset (CSV, Excel, JSON, Parquet)\nquestion · answers · what-if changes")
    arrow(ax, on_circle(195), right(user, -0.35), color=INK, rad=-0.25, size=11.5, offset=(0, -0.55),
          label="profile · progress · clarifying questions\nresults · drivers · actions · PDF report")
    arrow(ax, on_circle(40), left(llm, -0.1), color=BLUE, rad=-0.2, size=11, offset=(-0.35, 0.55),
          label="prompts with aggregates only\n(names, types, metrics, facts)")
    arrow(ax, left(llm, -0.45), on_circle(20), color=BLUE, rad=-0.2, size=11, offset=(0.45, -0.5),
          label="structured JSON: requirement,\nranking, review, brief, code")
    arrow(ax, on_circle(-20), left(web, 0.3), color=BLUE, dashed=True, rad=-0.2, size=11, offset=(0.3, 0.3),
          label="search query")
    arrow(ax, left(web, -0.3), on_circle(-50), color=BLUE, dashed=True, rad=-0.2, size=11, offset=(-0.3, -0.3),
          label="result snippets")
    ax.text(8.25, 0.45, "Raw rows, file paths and API keys never leave the machine; without an LLM every flow to "
            "the right is skipped and deterministic fallbacks are used.", ha="center", va="center", fontsize=11.5,
            color=MUTED)
    return save(fig, "dfd_level0.png")


# --------------------------------------------------------------------------- 9. class diagram


ClassAnchors = tuple[float, float, float, float]  # x0, y_top, width, height


def uml_class(ax: plt.Axes, x0: float, y_top: float, w: float, name: str, color: str, attributes: Sequence[str],
              methods: Sequence[str], *, stereotype: str | None = None, size: float = 10.5) -> ClassAnchors:
    """A UML class: name compartment (filled), attributes, then operations."""
    line, head = 0.27, 0.62 if stereotype else 0.48
    attr_h = line * len(attributes) + 0.14 if attributes else 0.16
    meth_h = line * len(methods) + 0.14 if methods else 0.16
    height = head + attr_h + meth_h
    ax.add_patch(Rectangle((x0, y_top - height), w, height, facecolor=PAPER, edgecolor=color, linewidth=1.8,
                           zorder=3))
    ax.add_patch(Rectangle((x0, y_top - head), w, head, facecolor=color, edgecolor=color, linewidth=1.8, zorder=3))
    ink = text_on(color)
    if stereotype:
        ax.text(x0 + w / 2, y_top - 0.17, f"«{stereotype}»", ha="center", va="center", fontsize=size - 1.5,
                color=ink, zorder=4)
    ax.text(x0 + w / 2, y_top - head + 0.22, name, ha="center", va="center", fontsize=size + 1.5,
            fontweight="bold", color=ink, zorder=4)
    ax.plot([x0, x0 + w], [y_top - head - attr_h] * 2, color=color, linewidth=1.2, zorder=4)
    for index, text in enumerate(attributes):
        ax.text(x0 + 0.1, y_top - head - 0.07 - line * (index + 0.5), text, ha="left", va="center",
                fontsize=size - 0.5, color=INK, zorder=4)
    for index, text in enumerate(methods):
        ax.text(x0 + 0.1, y_top - head - attr_h - 0.07 - line * (index + 0.5), text, ha="left", va="center",
                fontsize=size - 0.5, color=INK, zorder=4)
    return x0, y_top, w, height


def _polyline(ax: plt.Axes, points: Sequence[tuple[float, float]], *, color: str = INK, dashed: bool = False,
              head: bool = True, lw: float = 1.6) -> None:
    """Orthogonal connector: plain segments, with an arrow head on the last one."""
    style = (0, (5, 4)) if dashed else "-"
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, zorder=2, solid_capstyle="round")
    start, end = points[-2], points[-1]
    if head:
        arrow(ax, start, end, color=color, dashed=dashed, lw=lw)
    else:
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, linestyle=style, zorder=2)


def _diamond(ax: plt.Axes, x: float, y: float, *, filled: bool = True, size: float = 0.17) -> None:
    """UML composition (filled) / aggregation (hollow) diamond, pointing up or down along a vertical line."""
    ax.add_patch(Polygon([(x, y), (x + size * 0.7, y - size), (x, y - 2 * size), (x - size * 0.7, y - size)],
                         closed=True, facecolor=INK if filled else PAPER, edgecolor=INK, linewidth=1.4, zorder=5))


def class_diagram() -> Path:
    fig, ax = canvas(21.0, 13.6, "Class diagram — main classes",
                     "Orchestration, agents, AutoML engine and infrastructure (Pydantic models shown as types)")
    worker = uml_class(ax, 0.4, 11.75, 3.6, "RunWorker", BLUE, ["- active_run: str | None"],
                       ["+ submit(prepare): str", "+ run_lock(run_id): Lock", "+ busy(): bool"], stereotype="api")
    orch = uml_class(ax, 4.9, 11.75, 4.9, "Orchestrator", PURPLE, ["- context: RunContext", "- budget_mode: str"],
                     ["+ start(question, path): str", "+ step(): RunState", "+ run_until_blocked(): RunContext",
                      "+ provide_answer(run_id, answer)", "+ resume(run_id): RunContext",
                      "+ what_if(run_id, changes)", "+ ask(run_id, question)"], stereotype="control")
    ctx = uml_class(ax, 10.7, 11.75, 4.6, "RunContext", PURPLE,
                    ["+ run_id: str", "+ state: RunState", "+ requirement: UserRequirement",
                     "+ profile: DatasetProfile", "+ plan: ModelPlan", "+ training: TrainingResult",
                     "+ verification: VerificationReport", "+ history: list[RunEvent]"], [], stereotype="entity")
    repo = uml_class(ax, 16.2, 11.75, 4.4, "Repository", BLUE, [],
                     ["+ add_dataset(...): DatasetRecord", "+ save_run(run): RunRecord", "+ list_runs(...)",
                      "+ record_transition(run, event)"], stereotype="storage")

    agents_spec = [
        ("RequirementAgent", ["+ parse(text, profile)", "+ repair(req, profile)", "+ check_sufficiency(req)",
                              "+ summarize(req): str"]),
        ("DataAgent", ["+ run(path, requirement)", "  : DataAgentResult"]),
        ("KnowledgeAgent", ["+ brief(req, profile)", "  : KnowledgeBrief"]),
        ("ModelAgent", ["+ plan(req, profile,", "  knowledge): ModelPlan"]),
        ("Verifier", ["+ verify(req, training,", "  profile): Verification-", "  Report"]),
        ("AnalysisAgent", ["+ ask(question, profile,", "  df_path): Analysis-", "  Answer"]),
        ("InsightAgent", ["+ generate(facts)", "  : DecisionInsight"]),
    ]
    agent_w, agent_gap, agent_x0, agent_top = 2.22, 0.13, 0.4, 7.35
    agents = [uml_class(ax, agent_x0 + i * (agent_w + agent_gap), agent_top, agent_w, name, GREEN,
                        ["- llm_client: LLMClient?"], methods, stereotype="agent", size=9.8)
              for i, (name, methods) in enumerate(agents_spec)]
    trainer = uml_class(ax, 17.15, agent_top, 3.45, "AutoMLTrainer", ORANGE,
                        ["- budget: TrainingBudget"], ["+ run(df, requirement, profile,", "  run_dir, budget, plan)",
                                                       "  : TrainingResult"], stereotype="engine")
    llm = uml_class(ax, 0.4, 3.45, 3.4, "LLMClient", BLUE, ["- timeout_s: float", "- max_retries = 3"],
                    ["+ chat(messages): str", "+ chat_structured(messages,", "  schema): BaseModel"],
                    stereotype="infrastructure")
    index = uml_class(ax, 4.6, 3.45, 3.3, "KnowledgeIndex", ORANGE, ["- chunks (TF-IDF)"],
                      ["+ ensure_fresh(): bool", "+ search(query, k=4)"], stereotype="engine")
    explainer = uml_class(ax, 9.5, 3.45, 3.5, "ModelExplainer", ORANGE, ["- method: shap | permutation"],
                          ["+ global_explanation()", "+ local(position)"], stereotype="engine")
    model = uml_class(ax, 13.65, 3.45, 3.35, "TrainedModel", ORANGE,
                      ["- pipeline: Pipeline", "- decision_threshold"], ["+ predict(X)", "+ predict_proba(X)"],
                      stereotype="engine")
    holdout = uml_class(ax, 17.6, 3.45, 3.0, "Holdout", ORANGE, ["- access_log: list[str]"],
                        ["+ reveal(purpose)", "  : DataFrame"], stereotype="engine")

    def mid_y(c: ClassAnchors) -> float:
        return c[1] - c[3] / 2

    # RunWorker -> Orchestrator, Orchestrator <>- RunContext, Orchestrator -> Repository (above row 1).
    arrow(ax, (worker[0] + worker[2], 10.6), (orch[0], 10.6), color=INK, label="drives", size=10.5,
          offset=(0, 0.22))
    ax.plot([orch[0] + orch[2], ctx[0]], [10.6, 10.6], color=INK, linewidth=1.6, zorder=2)
    ax.add_patch(Polygon([(orch[0] + orch[2], 10.6), (orch[0] + orch[2] + 0.2, 10.72),
                          (orch[0] + orch[2] + 0.4, 10.6), (orch[0] + orch[2] + 0.2, 10.48)], closed=True,
                         facecolor=INK, edgecolor=INK, zorder=5))
    ax.text((orch[0] + orch[2] + ctx[0]) / 2 + 0.1, 10.82, "1   1", ha="center", va="center", fontsize=10.5,
            color=INK)
    lane_top = 12.05
    _polyline(ax, [(orch[0] + orch[2] - 0.6, orch[1]), (orch[0] + orch[2] - 0.6, lane_top),
                   (repo[0] + repo[2] / 2, lane_top), (repo[0] + repo[2] / 2, repo[1])], dashed=True)
    ax.text((ctx[0] + ctx[2] / 2), lane_top + 0.18, "«use» mirrors every transition into SQLite", ha="center",
            va="center", fontsize=10.5, color=INK, bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER,
                                                         "edgecolor": "none"})

    # Orchestrator <>- each agent (composition bus).
    bus_y = 7.75
    orch_bottom = orch[1] - orch[3]
    _diamond(ax, orch[0] + 1.2, orch_bottom)
    ax.plot([orch[0] + 1.2, orch[0] + 1.2], [orch_bottom - 0.34, bus_y], color=INK, linewidth=1.6, zorder=2)
    centres = [a[0] + a[2] / 2 for a in agents]
    ax.plot([centres[0], centres[-1]], [bus_y, bus_y], color=INK, linewidth=1.6, zorder=2)
    for x in centres:
        ax.plot([x, x], [bus_y, agent_top], color=INK, linewidth=1.6, zorder=2)
    ax.text(orch[0] + 1.35, (orch_bottom + bus_y) / 2 - 0.1, "owns one of each agent (1 : 1)", ha="left",
            va="center", fontsize=10.5, color=INK)
    # Orchestrator -> AutoMLTrainer (TRAIN state).
    lane2 = 8.1
    _polyline(ax, [(orch[0] + orch[2] - 0.8, orch_bottom), (orch[0] + orch[2] - 0.8, lane2),
                   (trainer[0] + trainer[2] / 2, lane2), (trainer[0] + trainer[2] / 2, agent_top)], dashed=True)
    ax.text(trainer[0] - 0.9, lane2 + 0.2, "«use» TRAIN state", ha="center", va="center", fontsize=10.5, color=INK,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none"})

    # Every agent -> LLMClient (dashed bus), KnowledgeAgent -> KnowledgeIndex.
    llm_bus = 4.0
    bottoms = [a[1] - a[3] for a in agents]
    for x, bottom_y in zip(centres, bottoms, strict=True):
        ax.plot([x, x], [bottom_y, llm_bus], color=BLUE, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    ax.plot([min(centres[0], llm[0] + llm[2] / 2), centres[-1]], [llm_bus, llm_bus], color=BLUE, linewidth=1.5,
            linestyle=(0, (5, 4)), zorder=2)
    arrow(ax, (llm[0] + llm[2] / 2, llm_bus), (llm[0] + llm[2] / 2, llm[1]), color=BLUE, dashed=True, lw=1.5)
    ax.text(centres[4] + 0.2, llm_bus + 0.2, "«use» optional LLM client — each agent has a deterministic fallback",
            ha="center", va="center", fontsize=10.5, color=INK,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none"})
    knowledge = agents[2]
    arrow(ax, (knowledge[0] + 0.45, knowledge[1] - knowledge[3]), (index[0] + index[2] / 2 + 0.3, index[1]),
          color=ORANGE, lw=1.6, label="retrieves", size=10, offset=(-0.45, 0.0))
    # Engine relations.
    trainer_bottom = trainer[1] - trainer[3]
    arrow(ax, (trainer[0] + 0.5, trainer_bottom), (model[0] + model[2] / 2 + 0.6, model[1]), color=ORANGE, lw=1.6,
          label="creates", size=10, offset=(0.45, 0.1))
    arrow(ax, (trainer[0] + trainer[2] - 0.6, trainer_bottom), (holdout[0] + holdout[2] / 2 + 0.4, holdout[1]),
          color=ORANGE, lw=1.6, label="reads test split once", size=10, offset=(0.1, 0.1))
    arrow(ax, (explainer[0] + explainer[2], mid_y(explainer) + 0.2), (model[0], mid_y(model) + 0.2),
          color=ORANGE, lw=1.6, label="explains", size=10, offset=(0, -0.32))
    legend(ax, [("Control / entity", PURPLE), ("Agent", GREEN), ("Engine", ORANGE), ("API / storage / LLM", BLUE)],
           1.2, 0.4, step=4.4, size=12)
    ax.text(19.9, 0.4, "◆ composition   - - ▷ «use»", ha="right", va="center", fontsize=12, color=MUTED)
    return save(fig, "class_diagram.png")


# --------------------------------------------------------------------------- 10. activity diagram


def activity_diagram() -> Path:
    width, height = 14.2, 18.4
    fig, ax = canvas(width, height, "Activity diagram — one analysis",
                     "Swimlanes show who performs each activity; diamonds are decisions")
    lanes = [("Business user", PINK), ("UI + API", BLUE), ("Orchestrator + agents", PURPLE),
             ("AutoML + decision engines", ORANGE)]
    lane_w, x_start, lane_top, lane_bottom = 3.3, 0.5, 16.95, 0.2
    centres = [x_start + lane_w * (i + 0.5) for i in range(len(lanes))]
    for i, (name, color) in enumerate(lanes):
        x0 = x_start + lane_w * i
        ax.add_patch(Rectangle((x0, lane_bottom), lane_w, lane_top - lane_bottom, facecolor=tint(color, 0.95),
                               edgecolor=tint(color, 0.5), linewidth=1.3, zorder=0))
        ax.add_patch(Rectangle((x0, lane_top - 0.55), lane_w, 0.55, facecolor=color, edgecolor=color, zorder=1))
        ax.text(x0 + lane_w / 2, lane_top - 0.275, name, ha="center", va="center", fontsize=13, fontweight="bold",
                color=text_on(color), zorder=2)
    w, h = 2.7, 0.66
    lane_color = [color for _, color in lanes]

    def act(lane: int, y: float, text: str) -> Box:
        return box(ax, centres[lane], y, w, h, text, lane_color[lane], filled=False, size=11, radius=0.2, lw=1.8)

    def decision(lane: int, y: float, text: str) -> Box:
        x, dw, dh = centres[lane], 1.9, 0.9
        ax.add_patch(Polygon([(x, y + dh / 2), (x + dw / 2, y), (x, y - dh / 2), (x - dw / 2, y)], closed=True,
                             facecolor=PAPER, edgecolor=YELLOW, linewidth=2.2, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=4,
                linespacing=1.05)
        return x, y, dw, dh

    def end(x: float, y: float) -> None:
        ax.add_patch(Circle((x, y), 0.2, facecolor=PAPER, edgecolor=INK, linewidth=2, zorder=3))
        ax.add_patch(Circle((x, y), 0.12, facecolor=INK, zorder=4))

    ax.add_patch(Circle((centres[0], 16.05), 0.17, facecolor=INK, zorder=3))
    upload = act(0, 15.25, "Upload a dataset")
    store = act(1, 14.4, "Check type and size,\nsave and profile")
    ask = act(0, 13.55, "Describe the problem,\npick budget, start")
    create = act(1, 12.7, "Create run (202),\nhand to RunWorker")
    relevance = act(2, 11.85, "Relevance check")
    relevant = decision(2, 11.0, "relevant?")
    polite = act(3, 11.0, "Polite reply → DONE")
    parse = act(2, 10.05, "Parse + repair the\nrequirement")
    enough = decision(2, 9.1, "enough\ninformation?")
    questions = act(1, 9.1, "Show clarifying\nquestions")
    answer = act(0, 9.1, "Answer\n(≤ 2 rounds)")
    profile = act(2, 8.15, "Profile data +\nknowledge brief")
    plan = act(2, 7.3, "Plan: rank models\nfrom the zoo")
    tune = act(3, 6.45, "Tune candidates\n(Optuna, train → valid)")
    select = act(3, 5.6, "Select, tune threshold,\nrefit, test once")
    verify = act(2, 4.75, "Verify: 7 checks")
    passed = decision(2, 3.85, "passed?")
    explain = act(3, 3.0, "Explain (SHAP), tiers,\nactions, what-if")
    insight = act(2, 2.15, "Grounded decision brief")
    report = act(3, 2.15, "Build PDF +\nMarkdown report")
    show = act(1, 1.3, "Show results and\nDecision Center")
    review = act(0, 1.3, "Review, simulate,\ndownload report")

    arrow(ax, (centres[0], 15.88), top(upload), color=INK)
    flow = [(upload, store), (store, ask), (ask, create), (create, relevance)]
    for a, b in flow:
        _polyline(ax, [bottom(a), (a[0], b[1]), left(b) if b[0] > a[0] else right(b)])
    arrow(ax, bottom(relevance), top(relevant), color=INK)
    arrow(ax, right(relevant), left(polite), color=INK, label="no", size=10.5, offset=(0, 0.2))
    end(polite[0], 10.35)
    arrow(ax, bottom(polite), (polite[0], 10.55), color=INK)
    arrow(ax, bottom(relevant), top(parse), color=INK, label="yes", size=10.5, offset=(0.3, 0))
    arrow(ax, bottom(parse), top(enough), color=INK)
    arrow(ax, left(enough), right(questions), color=INK, label="no", size=10.5, offset=(0, 0.2))
    arrow(ax, left(questions), right(answer), color=INK)
    _polyline(ax, [top(answer), (answer[0], parse[1]), left(parse)])
    arrow(ax, bottom(enough), top(profile), color=INK, label="yes", size=10.5, offset=(0.3, 0))
    arrow(ax, bottom(profile), top(plan), color=INK)
    _polyline(ax, [right(plan), (tune[0], plan[1]), top(tune)])
    arrow(ax, bottom(tune), top(select), color=INK)
    _polyline(ax, [bottom(select), (select[0], verify[1]), right(verify)])
    arrow(ax, bottom(verify), top(passed), color=INK)
    loop_x = centres[2] - lane_w / 2 + 0.12
    _polyline(ax, [left(passed), (loop_x, passed[1]), (loop_x, plan[1]), left(plan)], color=GREEN)
    ax.text(loop_x - 0.15, (passed[1] + plan[1]) / 2, "no, first time:\nre-plan with\nthe full budget",
            ha="right", va="center", fontsize=10.5, color=GREEN, fontweight="bold")
    _polyline(ax, [right(passed), (explain[0], passed[1]), top(explain)])
    ax.text(passed[0] + 1.05, passed[1] + 0.2, "yes, or still failing\n(flagged)", ha="left", va="bottom",
            fontsize=10, color=INK)
    _polyline(ax, [bottom(explain), (explain[0], 2.62), (insight[0], 2.62), top(insight)])
    arrow(ax, right(insight), left(report), color=INK)
    _polyline(ax, [bottom(report), (report[0], show[1]), right(show)])
    arrow(ax, left(show), right(review), color=INK)
    end(centres[0], 0.5)
    arrow(ax, bottom(review), (centres[0], 0.7), color=INK)
    return save(fig, "activity_diagram.png")


# --------------------------------------------------------------------------- 11. deployment diagram


def _node(ax: plt.Axes, x0: float, y0: float, w: float, h: float, title: str, color: str, *,
          depth: float = 0.28, stereotype: str = "device") -> None:
    """A UML node: a box with a 3-D top and right side."""
    front = Rectangle((x0, y0), w, h, facecolor=tint(color, 0.94), edgecolor=color, linewidth=2.2, zorder=1)
    ax.add_patch(front)
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + depth, y0 + h + depth), (x0 + w + depth, y0 + h + depth),
                          (x0 + w, y0 + h)], closed=True, facecolor=tint(color, 0.75), edgecolor=color,
                         linewidth=2.2, zorder=1))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + depth, y0 + depth), (x0 + w + depth, y0 + h + depth),
                          (x0 + w, y0 + h)], closed=True, facecolor=tint(color, 0.6), edgecolor=color,
                         linewidth=2.2, zorder=1))
    ax.text(x0 + 0.2, y0 + h - 0.25, f"«{stereotype}»", ha="left", va="center", fontsize=11, color=color, zorder=2)
    ax.text(x0 + 0.2, y0 + h - 0.62, title, ha="left", va="center", fontsize=14, fontweight="bold", color=INK,
            zorder=2)


def _component(ax: plt.Axes, cx: float, cy: float, w: float, h: float, title: str, sub: str, color: str) -> Box:
    """A UML component: a box with the two-tab component icon in its top-right corner."""
    b = box(ax, cx, cy, w, h, title, color, sub=sub, filled=False, size=12.5, sub_size=10.5, radius=0.08)
    ix, iy = cx + w / 2 - 0.42, cy + h / 2 - 0.33
    ax.add_patch(Rectangle((ix, iy - 0.08), 0.26, 0.24, facecolor=PAPER, edgecolor=color, linewidth=1.3, zorder=5))
    for dy in (0.0, 0.1):
        ax.add_patch(Rectangle((ix - 0.06, iy - 0.04 + dy), 0.12, 0.05, facecolor=PAPER, edgecolor=color,
                               linewidth=1.1, zorder=6))
    return b


def deployment_diagram() -> Path:
    fig, ax = canvas(18.0, 11.4, "Component and deployment diagram",
                     "Everything runs on one Windows PC and listens on localhost only; the LLM is an optional "
                     "external service")
    _node(ax, 0.4, 0.9, 13.4, 8.75, "User's PC — Windows 10 / 11, Python 3.12 virtual environment", PURPLE)
    _node(ax, 0.75, 6.0, 3.0, 2.6, "Web browser", PINK, stereotype="execution environment")
    _component(ax, 2.25, 6.85, 2.6, 1.0, "Dashboard", "HTML / JS from\nStreamlit", PINK)
    _node(ax, 4.25, 6.0, 3.6, 2.6, "Streamlit server :8501", PURPLE, stereotype="process")
    _component(ax, 6.05, 6.85, 3.2, 1.0, "ui/ (8 pages)", "IDSSClient →\n127.0.0.1:8000", PURPLE)
    _node(ax, 8.35, 2.0, 5.0, 6.6, "Uvicorn + FastAPI :8000", BLUE, stereotype="process")
    api = _component(ax, 10.85, 6.85, 4.6, 1.0, "api/ — REST routes", "17 endpoints · {error, detail}", BLUE)
    worker = _component(ax, 10.85, 5.55, 4.6, 1.0, "RunWorker thread", "Orchestrator + 7 agents", BLUE)
    engines = _component(ax, 10.85, 4.25, 4.6, 1.0, "Engines", "AutoML · decision layer · RAG", BLUE)
    storage = _component(ax, 10.85, 2.95, 4.6, 1.0, "Storage", "SQLAlchemy repository", BLUE)
    _node(ax, 0.75, 1.3, 3.4, 3.6, "Sandbox child", ORANGE, stereotype="process")
    sandbox = _component(ax, 2.45, 2.65, 2.9, 1.25, "sandbox_runner.py", "python -I · temp dir\nno secrets · timeout",
                         ORANGE)
    _node(ax, 4.6, 1.3, 3.3, 3.6, "workspace/", GREEN, stereotype="file system")
    files = box(ax, 6.25, 2.65, 2.9, 1.65, "Run files", GREEN, filled=False, size=12.5, sub_size=10.5,
                sub="uploads/ · runs/<run_id>/\nstate.json · model · charts\nreport.pdf · idss.db · logs")
    _node(ax, 14.6, 6.4, 3.0, 2.2, "LLM provider", PINK, stereotype="external service")
    ax.text(16.1, 6.95, "OpenAI · Groq · Gemini\n(or Ollama on the PC)", ha="center", va="center", fontsize=10.5,
            color=INK)
    _node(ax, 14.6, 2.6, 3.0, 2.0, "Web search", PINK, stereotype="external service")
    ax.text(16.1, 3.1, "Tavily · SerpAPI\n(optional)", ha="center", va="center", fontsize=10.5, color=INK)

    arrow(ax, (3.55, 6.85), (4.45, 6.85), color=INK, both=True, label="HTTP :8501", size=10, offset=(0, 0.3))
    arrow(ax, (7.65, 6.85), (8.55, 6.85), color=INK, both=True, label="REST/JSON", size=10, offset=(0, 0.3))
    arrow(ax, bottom(api), top(worker), color=INK)
    arrow(ax, bottom(worker), top(engines), color=INK)
    arrow(ax, bottom(engines), top(storage), color=INK)
    arrow(ax, left(storage), right(files, storage[1] - files[1]), color=GREEN, label="files ·\nSQLite", size=9.5)
    _polyline(ax, [left(worker), (3.2, worker[1]), (3.2, sandbox[1] + sandbox[3] / 2)], color=ORANGE, dashed=True)
    ax.text(5.9, worker[1] + 0.2, "spawn: config.json in, output.json out", ha="center", va="center", fontsize=10,
            color=ORANGE)
    arrow(ax, (13.35 + 0.28, 5.55), (14.6, 7.0), color=PINK, both=True, label="HTTPS\n(optional)", size=10,
          offset=(-0.05, 0.1))
    arrow(ax, (13.35 + 0.28, 4.25), (14.6, 3.5), color=PINK, dashed=True, label="HTTPS", size=10,
          offset=(0, -0.25))
    ax.text(9.0, 0.4, ".env (keys) is read only by the API process; the browser, the sandbox and the logs never "
            "see it.", ha="center", va="center", fontsize=11.5, color=MUTED)
    return save(fig, "deployment_diagram.png")


DIAGRAMS = (system_architecture, agent_workflow, sequence_one_run, data_flow, er_diagram, use_case,
            eight_week_timeline, dfd_level0, class_diagram, activity_diagram, deployment_diagram)


def main() -> int:
    for draw in DIAGRAMS:
        path = draw()
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

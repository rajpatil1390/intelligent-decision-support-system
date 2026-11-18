"""Prompt templates, stored as Markdown files in this package.

Each file has a `## system` and a `## user` section. Placeholders use `$name` syntax
(string.Template), so JSON braces in prompts need no escaping; write a literal dollar as `$$`.
Dataset context must come from format_profile(): aggregates only, never raw rows.
"""

import re
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import NamedTuple

from idss.core.schemas import DatasetProfile

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_NAMES = (
    "relevance_check",
    "requirement_parse",
    "sufficiency_check",
    "clarifying_questions",
    "requirement_summary",
    "model_selection",
    "knowledge_brief",
    "verification_review",
    "analysis_code",
    "analysis_repair",
    "decision_insight",
)
_SECTION = re.compile(r"^## (system|user)[ \t]*$", re.MULTILINE)


class PromptTemplate(NamedTuple):
    system: Template
    user: Template


@lru_cache(maxsize=None)
def load_prompt(name: str) -> PromptTemplate:
    """Load and parse `<name>.md` from this package."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise KeyError(f"Unknown prompt '{name}'. Available: {', '.join(PROMPT_NAMES)}.")
    parts = _SECTION.split(path.read_text(encoding="utf-8"))
    sections = dict(zip(parts[1::2], (part.strip() for part in parts[2::2]), strict=True))
    if set(sections) != {"system", "user"}:
        raise ValueError(f"Prompt file {path.name} must contain exactly one '## system' and one '## user' section.")
    return PromptTemplate(system=Template(sections["system"]), user=Template(sections["user"]))


def render_prompt(name: str, **variables: str) -> list[dict[str, str]]:
    """Render a prompt into chat messages. Raises KeyError if a placeholder has no value."""
    prompt = load_prompt(name)
    return [
        {"role": "system", "content": prompt.system.substitute(variables)},
        {"role": "user", "content": prompt.user.substitute(variables)},
    ]


def format_columns(profile: DatasetProfile | None) -> str:
    """Column names and semantic types only — the minimum needed to write code against the data."""
    if profile is None:
        return "No dataset profile is available."
    return "\n".join(f"- {col.name} ({col.semantic_type}, pandas dtype {col.dtype})" for col in profile.columns)


def format_profile(profile: DatasetProfile | None) -> str:
    """Describe a dataset for an LLM using names, types and aggregate statistics only (no values or rows)."""
    if profile is None:
        return "No dataset has been provided."
    lines = [
        f"Rows: {profile.n_rows:,}. Columns: {profile.n_cols}.",
        "Columns (name | type | % missing | distinct values):",
        *(f"- {col.name} | {col.semantic_type} | {col.missing_pct:g}% | {col.n_unique}" for col in profile.columns),
        f"Suggested target column: {profile.suggested_target or 'none'}.",
        f"Suggested task type: {profile.suggested_task_type or 'none'}.",
    ]
    if profile.time_column:
        lines.append(f"Time column: {profile.time_column}.")
    if profile.warnings:
        lines.append("Data-quality warnings:")
        lines += [f"- {warning}" for warning in profile.warnings]
    return "\n".join(lines)

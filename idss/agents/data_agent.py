"""Data Agent: load, profile, clean and split a dataset, fit the preprocessor, and explain it for business users.

All data decisions are deterministic. The LLM is used for one thing only: a short plain-English summary of
the profile. If that call fails for any reason, a template summary is used instead.
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.sanitize import DATA_RULE, data_json
from idss.core.schemas import TASK_DESCRIPTIONS, DatasetProfile, TaskType, UserRequirement
from idss.ml.data_loader import load_dataset
from idss.ml.preprocess import build_preprocessor, clean_dataset, plan_columns, save_preprocessor, split_data
from idss.ml.profiler import profile_dataset
from idss.storage.runs import new_run_id, run_dir

logger = get_logger()

_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst explaining a dataset to a business user with no technical background. "
    "Write 3 to 5 short sentences of plain English: what the data contains, what will be predicted, "
    "and the most important data-quality issues. No jargon, no markdown, no bullet points. " + DATA_RULE
)


class SplitSizes(BaseModel):
    train: int
    valid: int
    test: int


class DataAgentResult(BaseModel):
    run_id: str
    dataset_path: str
    target: str | None
    task_type: TaskType
    profile: DatasetProfile
    cleaning_steps: list[str]
    split_sizes: SplitSizes
    artifacts: dict[str, str] = Field(default_factory=dict, description="Artifact name -> file path.")
    llm_summary: str
    summary_source: Literal["llm", "template"]


class DataAgent:
    """Prepares a dataset for the AutoML engine."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        runs_dir: Path | None = None,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        self.llm_client = llm_client
        self.runs_dir = runs_dir
        self.max_rows = max_rows
        self.seed = seed

    def run(
        self, path: str | Path, user_requirement: UserRequirement | None = None, *, run_id: str | None = None
    ) -> DataAgentResult:
        run_id = run_id or new_run_id()
        df = load_dataset(path, max_rows=self.max_rows, seed=self.seed)
        steps: list[str] = []

        requested_target, requested_task = _requested_target_and_task(user_requirement)
        profile = self._profile(df, requested_target, requested_task, steps)
        if all(col.n_unique <= 1 for col in profile.columns):
            raise ValueError("No column varies: every column has a single value (or none), so there is nothing to learn.")
        task_type = requested_task or profile.suggested_task_type
        target = None if task_type == "tabular_clustering" else profile.suggested_target
        if task_type is None or (target is None and task_type != "tabular_clustering"):
            raise ValueError(
                "Could not determine a usable target column and task type; specify the target column explicitly."
            )

        df, cleaning = clean_dataset(df, target, task_type, profile)
        steps += cleaning
        train, valid, test = split_data(df, target, task_type, seed=self.seed)
        steps.append(f"Split into train/valid/test = {len(train):,}/{len(valid):,}/{len(test):,} rows.")

        preprocessor = build_preprocessor(profile, target)
        preprocessor.fit(train.drop(columns=[target]) if target else train)
        steps += plan_columns(profile, target).describe()
        steps.append(f"Preprocessing produces {len(preprocessor.get_feature_names_out())} model features.")

        out_dir = run_dir(run_id, self.runs_dir)
        artifacts = {"preprocessor": str(save_preprocessor(preprocessor, run_id, self.runs_dir))}
        for name, split in (("train", train), ("valid", valid), ("test", test)):
            split_path = out_dir / f"{name}.parquet"
            split.to_parquet(split_path)
            artifacts[name] = str(split_path)

        summary, source = self._summarize(profile, target, task_type)
        logger.info("DataAgent run {} finished: target={}, task={}.", run_id, target, task_type)
        return DataAgentResult(
            run_id=run_id,
            dataset_path=str(path),
            target=target,
            task_type=task_type,
            profile=profile,
            cleaning_steps=steps,
            split_sizes=SplitSizes(train=len(train), valid=len(valid), test=len(test)),
            artifacts=artifacts,
            llm_summary=summary,
            summary_source=source,
        )

    def _profile(
        self, df: pd.DataFrame, requested_target: str | None, requested_task: TaskType | None, steps: list[str]
    ) -> DatasetProfile:
        if requested_task == "tabular_clustering":
            return profile_dataset(df, target=None, suggest_target=False)
        if requested_target is not None:
            try:
                return profile_dataset(df, target=requested_target)
            except ValueError:
                steps.append(f"Requested target '{requested_target}' was not found; using the suggested target instead.")
                logger.warning("Requested target '{}' not in dataset; falling back to suggestion.", requested_target)
        return profile_dataset(df)

    def _summarize(
        self, profile: DatasetProfile, target: str | None, task_type: TaskType
    ) -> tuple[str, Literal["llm", "template"]]:
        if self.llm_client is not None:
            messages = [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": data_json(_summary_facts(profile, target, task_type))},
            ]
            try:
                summary = self.llm_client.chat(messages, temperature=0.3).strip()
                if summary:
                    return summary, "llm"
                logger.warning("LLM returned an empty data summary; using the template summary.")
            except Exception as exc:  # the summary is optional; never let it break the pipeline
                logger.warning("LLM data summary failed ({}: {}); using the template summary.", type(exc).__name__, exc)
        return template_summary(profile, target, task_type), "template"


def template_summary(profile: DatasetProfile, target: str | None, task_type: TaskType) -> str:
    """Deterministic plain-English summary used when the LLM is unavailable."""
    type_counts = Counter(col.semantic_type for col in profile.columns)
    type_text = ", ".join(f"{count} {semantic}" for semantic, count in type_counts.most_common())
    parts = [f"The dataset has {profile.n_rows:,} rows and {profile.n_cols} columns ({type_text})."]
    if target:
        parts.append(f"The goal is to predict '{target}', which is {TASK_DESCRIPTIONS[task_type]}.")
    else:
        parts.append(f"No target column is used, so this is {TASK_DESCRIPTIONS[task_type]}.")
    missing = [col.name for col in profile.columns if col.missing_pct > 0]
    if missing:
        parts.append(f"{len(missing)} column(s) have missing values, which will be filled in automatically.")
    if profile.warnings:
        shown = "; ".join(profile.warnings[:3])
        more = f" (and {len(profile.warnings) - 3} more)" if len(profile.warnings) > 3 else ""
        parts.append(f"Points to review: {shown}{more}.")
    else:
        parts.append("No major data-quality issues were found.")
    return " ".join(parts)


def _summary_facts(profile: DatasetProfile, target: str | None, task_type: TaskType) -> dict[str, object]:
    """Aggregate facts for the LLM. Deliberately excludes sample values so no raw records leave the system."""
    facts: dict[str, object] = {
        "rows": profile.n_rows,
        "columns": profile.n_cols,
        "column_types": dict(Counter(col.semantic_type for col in profile.columns)),
        "column_names": [col.name for col in profile.columns],
        "target": target,
        "task": TASK_DESCRIPTIONS[task_type],
        "columns_with_missing_values_pct": {col.name: col.missing_pct for col in profile.columns if col.missing_pct > 0},
        "duplicate_rows": profile.duplicate_rows,
        "warnings": profile.warnings,
    }
    if target:
        facts["target_distribution"] = profile.column(target).stats
    return facts


def _requested_target_and_task(requirement: UserRequirement | None) -> tuple[str | None, TaskType | None]:
    if requirement is None:
        return None, None
    return requirement.target, requirement.problem.task_type

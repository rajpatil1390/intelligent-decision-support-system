"""Request and response models of the IDSS API.

Responses never contain secrets or server file paths: files are exposed through their API URLs only.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from idss.core.plan_contract import MAX_ACTIONABLE_FEATURES
from idss.core.schemas import (
    BudgetMode,
    ClusterProfile,
    Counterfactual,
    DatasetProfile,
    FilterCondition,
    Forecast,
    GlobalExplanation,
    LeaderboardEntry,
    LocalExplanation,
    Recommendation,
    RiskTier,
    RiskTierCutoffs,
    VerificationReport,
    WhatIfResult,
)

MAX_QUESTION_CHARS = 5_000


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ErrorResponse(BaseModel):
    """Every error has this shape. `error` is a stable code; `detail` is a message (or a list for validation)."""

    error: str
    detail: Any


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_configured: bool
    version: str
    public_demo: bool = False  # IDSS_PUBLIC_DEMO: no Ask Your Data, fast mode only, short retention
    max_upload_mb: float  # largest accepted upload


# --------------------------------------------------------------------------- datasets


class ColumnSummary(BaseModel):
    name: str
    semantic_type: str
    missing_pct: float
    n_unique: int


class ProfileSummary(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnSummary]
    suggested_target: str | None = None
    suggested_task_type: str | None = None
    time_column: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> "ProfileSummary":
        return cls(
            n_rows=profile.n_rows, n_cols=profile.n_cols,
            columns=[ColumnSummary(name=c.name, semantic_type=c.semantic_type, missing_pct=c.missing_pct,
                                   n_unique=c.n_unique) for c in profile.columns],
            suggested_target=profile.suggested_target, suggested_task_type=profile.suggested_task_type,
            time_column=profile.time_column, warnings=profile.warnings,
        )


class DatasetOut(BaseModel):
    id: str
    filename: str
    rows: int
    cols: int
    uploaded_at: datetime


class DatasetUploaded(DatasetOut):
    profile: ProfileSummary


# --------------------------------------------------------------------------- runs


class RunCreate(_Request):
    dataset_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: BudgetMode = "fast"
    actionable_features: list[str] | None = Field(default=None, max_length=MAX_ACTIONABLE_FEATURES,
                                                  description="Columns the business can change (existing, non-ID).")


class RunCreated(BaseModel):
    run_id: str
    state: str
    status_url: str


class RunEventOut(BaseModel):
    ts: datetime
    from_state: str | None = None
    to_state: str
    message: str


class RunSummary(BaseModel):
    run_id: str
    dataset_id: str | None = None
    question: str
    state: str
    progress: int = Field(ge=0, le=100)
    active: bool = Field(description="A worker thread is executing this run right now.")
    task_type: str | None = None
    best_model: str | None = None
    primary_metric: str | None = None
    metric_value: float | None = None
    verification_passed: bool | None = None
    created_at: datetime
    updated_at: datetime
    report_pdf_url: str | None = None
    report_md_url: str | None = None


class RunDetail(RunSummary):
    message: str | None = None
    target: str | None = None
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    decision_threshold: float | None = None
    verification: VerificationReport | None = None
    pending_questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    plan_source: str | None = Field(default=None, description="llm, default or fallback, once a plan is validated.")
    plan_fallback: bool = Field(default=False, description="The proposed plan was rejected; a fallback plan was used.")
    plan_rejections: list[str] = Field(default_factory=list, description="Why the proposed plan was rejected.")
    events: list[RunEventOut] = Field(default_factory=list)


class AnswerRequest(_Request):
    answer: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class LeaderboardResponse(BaseModel):
    run_id: str
    primary_metric: str
    direction: str
    best_model: str
    entries: list[LeaderboardEntry]


class ExplanationResponse(BaseModel):
    run_id: str
    global_explanation: GlobalExplanation
    local_examples: list[LocalExplanation] = Field(default_factory=list)
    clusters: list[ClusterProfile] = Field(default_factory=list)
    forecast: Forecast | None = None
    charts: dict[str, str] = Field(default_factory=dict, description="Chart name -> API URL of the PNG.")


class RecommendationsResponse(BaseModel):
    run_id: str
    actionable_features: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    risk_tiers: list[RiskTier] = Field(default_factory=list)
    tier_cutoffs: RiskTierCutoffs | None = None
    counterfactual_example: Counterfactual | None = None
    what_if_example: WhatIfResult | None = None
    notes: list[str] = Field(default_factory=list)


class WhatIfRequest(_Request):
    """Simulate `changes` for one record (`base_row`) or for a filtered segment (`segment`), never both."""

    base_row: dict[str, Any] | None = None
    segment: list[FilterCondition] | None = Field(default=None, description="Filters; [] means every record.")
    changes: dict[str, Any] = Field(min_length=1, description="column -> value | {set|add|pct: number}")

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "WhatIfRequest":
        if (self.base_row is None) == (self.segment is None):
            raise ValueError("Give exactly one of base_row or segment.")
        return self


class AskRequest(_Request):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

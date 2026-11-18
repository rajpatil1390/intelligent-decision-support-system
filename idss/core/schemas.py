"""Core Pydantic schemas for a parsed user requirement.

The Requirement Agent fills a UserRequirement from free-text user input; downstream agents
(Data, AutoML, Verifier, Decision) read it. Field descriptions are sent to the LLM as part of
the JSON Schema, so keep them short and unambiguous.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

TaskType = Literal[
    "tabular_classification",
    "tabular_regression",
    "tabular_clustering",
    "time_series_forecasting",
    "text_classification",
]
TASK_DESCRIPTIONS: dict[str, str] = {
    "tabular_classification": "a classification problem (predicting a category)",
    "tabular_regression": "a regression problem (predicting a number)",
    "tabular_clustering": "a clustering problem (grouping similar records)",
    "time_series_forecasting": "a time-series forecasting problem (predicting future values over time)",
    "text_classification": "a text classification problem (predicting a category from text)",
}
UserIntent = Literal["build", "consult", "inquire", "unclear"]
ExpertiseLevel = Literal["low", "medium", "high"]
DatasetSource = Literal["upload", "url", "sample", "unspecified"]
ModelType = Literal["classical_ml", "ensemble", "neural_network", "statistical", "any"]
SemanticType = Literal["numeric", "categorical", "datetime", "text", "id", "boolean"]


class _Schema(BaseModel):
    # LLM output often carries extra keys or stray whitespace; tolerate both.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class UserInfo(_Schema):
    intent: UserIntent = Field(default="build", description="What the user wants: build a model, get advice, ask a question, or unclear.")
    expertise: ExpertiseLevel = Field(default="medium", description="User's ML expertise level.")


class PerformanceMetric(_Schema):
    name: str = Field(min_length=1, description="Metric name, e.g. accuracy, f1, rmse, mae.")
    target: float | None = Field(default=None, description="Desired value for the metric, if the user stated one.")


class DatasetSpec(_Schema):
    name: str = Field(default="", description="Dataset or file name, if given.")
    description: str = Field(default="", description="What the data contains.")
    source: DatasetSource = Field(default="unspecified", description="Where the data comes from.")
    path: str | None = Field(default=None, description="File path or URL, if given.")
    target_column: str | None = Field(default=None, description="Column to predict, if known.")
    preprocessing: list[str] = Field(default_factory=list, description="Preprocessing steps the user asked for.")


class ModelSpec(_Schema):
    name: str = Field(default="", description="Specific model the user asked for, e.g. XGBoost. Empty if none.")
    model_type: ModelType = Field(default="any", description="Model family preference.")
    description: str = Field(default="", description="Any other model expectations.")


class ProblemSpec(_Schema):
    task_type: TaskType = Field(description="ML task category.")
    description: str = Field(min_length=1, description="The problem to solve, in one or two sentences.")
    domain: str = Field(default="", description="Application domain, e.g. healthcare, retail, finance.")
    performance_metrics: list[PerformanceMetric] = Field(default_factory=list, description="Metrics and targets the user cares about.")
    constraints: list[str] = Field(default_factory=list, description="Other requirements, e.g. 'must be explainable', 'inference < 100 ms'.")


class UserRequirement(_Schema):
    user: UserInfo = Field(default_factory=UserInfo)
    problem: ProblemSpec
    datasets: list[DatasetSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="How complete and correct this parse is, from 0 to 1.")
    missing_info: list[str] = Field(default_factory=list, description="Important details the user did not provide.")
    actionable_features: list[str] = Field(
        default_factory=list,
        description="Columns the business can actually change (e.g. price, contract type). Leave empty if unsure.",
    )
    # Corrections made by RequirementAgent.repair; hidden from the JSON Schema the LLM sees.
    system_notes: SkipJsonSchema[list[str]] = Field(default_factory=list)

    @property
    def target(self) -> str | None:
        """The first target column named in `datasets`, if any."""
        return next((ds.target_column for ds in self.datasets if ds.target_column), None)


BudgetMode = Literal["fast", "full"]


class TrainingBudget(_Schema):
    """How much search the AutoML engine may do."""

    max_candidates: int = Field(ge=1)
    n_trials_per_model: int = Field(ge=1)
    timeout_per_model_s: float = Field(gt=0)
    total_timeout_s: float = Field(gt=0)


BUDGET_PRESETS: dict[str, TrainingBudget] = {
    "fast": TrainingBudget(max_candidates=3, n_trials_per_model=15, timeout_per_model_s=60, total_timeout_s=180),
    "full": TrainingBudget(max_candidates=6, n_trials_per_model=40, timeout_per_model_s=180, total_timeout_s=1080),
}


class ModelConstraints(_Schema):
    """User constraints that limit which trained models are acceptable."""

    max_inference_ms_per_row: float | None = Field(default=None, gt=0)
    max_train_time_s: float | None = Field(default=None, gt=0)
    prefer_interpretable: bool = False


class RankedCandidate(_Schema):
    name: str = Field(description="A model name from the allowed list.")
    reason: str = Field(default="", description="One sentence on why this model suits the problem.")


class ModelPlan(_Schema):
    """What the AutoML engine should train (produced by ModelAgent in the PLAN state)."""

    task_type: TaskType
    target: str | None
    candidates: list[RankedCandidate] = Field(min_length=1)
    primary_metric: str
    budget_mode: BudgetMode = "fast"
    budget: TrainingBudget
    constraints: ModelConstraints = Field(default_factory=ModelConstraints)
    notes: list[str] = Field(default_factory=list)
    source: Literal["llm", "default"] = "default"


class LeaderboardEntry(_Schema):
    """One tuned candidate. Scores are on the validation split; only the selected model has test metrics."""

    rank: int | None = None
    model: str
    status: Literal["ok", "failed", "skipped"]
    params: dict[str, Any] = Field(default_factory=dict)
    valid_score: float | None = None
    valid_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    train_time_s: float | None = None
    inference_ms_per_row: float | None = None
    model_size_kb: float | None = None
    n_features: int | None = None
    n_trials: int = 0
    n_failed_trials: int = 0
    tuning_time_s: float = 0.0
    decision_threshold: float | None = Field(
        default=None, description="Binary classification: probability cut-off tuned on the validation split."
    )
    valid_score_default_threshold: float | None = Field(default=None, description="Validation score at threshold 0.5.")
    violations: list[str] = Field(default_factory=list)
    selected: bool = False
    error: str | None = None


class Baseline(_Schema):
    """A trivial reference model (majority class / mean / seasonal naive) scored on the test split."""

    name: str
    metrics: dict[str, Any]


class TrainingResult(_Schema):
    """Outcome of AutoMLTrainer.run; also persisted in the run's state.json."""

    task_type: TaskType
    target: str | None
    primary_metric: str
    direction: Literal["maximize", "minimize"]
    best_model: str
    best_params: dict[str, Any]
    valid_score: float | None
    test_metrics: dict[str, Any]
    decision_threshold: float | None = None
    test_metrics_default_threshold: dict[str, Any] = Field(
        default_factory=dict, description="Binary classification: test metrics at threshold 0.5, for comparison."
    )
    train_metrics: dict[str, Any] = Field(default_factory=dict, description="Final model scored on its own training data.")
    baseline: Baseline | None = None
    complexity: dict[str, Any]
    leaderboard: list[LeaderboardEntry]
    budget: TrainingBudget
    total_time_s: float
    model_path: str
    leaderboard_path: str
    metrics_path: str
    predictions_path: str | None = Field(default=None, description="Test-split actual vs predicted values (CSV).")
    test_set_accesses: list[str] = Field(description="Audit trail: every read of the test split, by purpose.")
    warnings: list[str] = Field(default_factory=list)


class KnowledgeChunk(_Schema):
    """A passage retrieved from the knowledge base (or the web)."""

    source: str
    text: str
    score: float = 0.0


class KnowledgeBrief(_Schema):
    """Best-practice insights retrieved for a run, used when planning models."""

    query: str
    insights: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    generated_by: Literal["llm", "retrieval", "none"] = "none"

    def as_prompt_text(self) -> str:
        if not self.insights:
            return "None available."
        return "\n".join(f"- {insight}" for insight in self.insights) + f"\n(Sources: {', '.join(self.sources)})"


class Check(_Schema):
    name: str
    passed: bool
    detail: str


class VerificationReport(_Schema):
    """Outcome of the Verifier: rule-based checks (always) plus an optional LLM review."""

    passed: bool
    checks: list[Check]
    llm_verdict: str | None = None
    llm_concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]


# --------------------------------------------------------------------------- decision layer (Step 6)


class FeatureImpact(_Schema):
    """Importance of one ORIGINAL column (one-hot / derived columns are summed back into it)."""

    feature: str
    importance: float
    share: float = Field(description="Importance as a share of the total, 0-1.")
    direction: Literal["increases", "decreases", "mixed"] = "mixed"
    detail: str = ""


class GlobalExplanation(_Schema):
    method: Literal["shap_tree", "shap_linear", "permutation", "cluster_profile", "none"]
    output: str = Field(description="What the explained number is, e.g. \"probability that churn = 'Yes'\".")
    features: list[FeatureImpact] = Field(default_factory=list)
    n_samples: int = 0
    notes: list[str] = Field(default_factory=list)


class LocalContribution(_Schema):
    feature: str
    value: Any = None
    impact: float


class LocalExplanation(_Schema):
    method: str
    prediction: Any = None
    contributions: list[LocalContribution] = Field(default_factory=list)


class ClusterProfile(_Schema):
    cluster: int
    size: int
    share: float
    summary: dict[str, Any] = Field(default_factory=dict, description="Mean (numeric) or most common value per column.")
    distinguishing: list[str] = Field(default_factory=list)


class ForecastPoint(_Schema):
    timestamp: str
    prediction: float
    lower: float
    upper: float


class Forecast(_Schema):
    horizon: int
    points: list[ForecastPoint]
    interval: str
    assumptions: list[str] = Field(default_factory=list)


class ExplanationResult(_Schema):
    global_explanation: GlobalExplanation
    local_examples: list[LocalExplanation] = Field(default_factory=list)
    clusters: list[ClusterProfile] = Field(default_factory=list)
    forecast: Forecast | None = None
    charts: dict[str, str] = Field(default_factory=dict, description="Chart name -> PNG path.")


class FilterCondition(_Schema):
    """Structured row filter for what-if segments (no free-text expressions are ever evaluated)."""

    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in"]
    value: Any


class WhatIfResult(_Schema):
    output: str
    before: Any
    after: Any
    delta: float | None = None
    delta_pct: float | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SegmentImpact(_Schema):
    output: str
    filters: list[FilterCondition] = Field(default_factory=list)
    changes: dict[str, Any] = Field(default_factory=dict)
    count: int
    avg_before: float | None = None
    avg_after: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    warnings: list[str] = Field(default_factory=list)


class RiskTierCutoffs(_Schema):
    """Probability cut-offs: High >= `high`; Medium >= `medium` (the decision threshold) and < `high`; Low below."""

    high: float = Field(description="max(decision threshold, `percentile`-th percentile of predicted probabilities).")
    medium: float = Field(description="The tuned decision threshold.")
    percentile: int = 70
    percentile_value: float = Field(description="The `percentile`-th percentile of the predicted probabilities.")


class RiskTier(_Schema):
    name: Literal["High", "Medium", "Low"]
    min_probability: float
    max_probability: float
    count: int
    share: float
    avg_probability: float | None = None
    top_drivers: list[str] = Field(default_factory=list)


class CounterfactualChange(_Schema):
    feature: str
    before: Any
    after: Any


class Counterfactual(_Schema):
    row_index: Any = None
    probability_before: float
    probability_after: float
    threshold: float
    success: bool
    changes: list[CounterfactualChange] = Field(default_factory=list)


class Recommendation(_Schema):
    action: str
    rationale: str
    expected_impact: float
    impact_unit: str
    confidence: Literal["high", "medium", "low"]
    evidence: dict[str, Any] = Field(default_factory=dict, description="Named computed numbers behind the claim.")


class RecommendationResult(_Schema):
    actionable_features: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    risk_tiers: list[RiskTier] = Field(default_factory=list)
    tier_cutoffs: RiskTierCutoffs | None = None
    counterfactual_example: Counterfactual | None = None
    notes: list[str] = Field(default_factory=list)


class GroundingCheck(_Schema):
    passed: bool
    attempts: int
    unsupported_numbers: list[str] = Field(default_factory=list)


class DecisionInsight(_Schema):
    executive_summary: str
    key_findings: list[str]
    recommended_actions: list[str]
    risks_and_limitations: list[str]
    next_steps: list[str]
    generated_by: Literal["llm", "template"] = "template"
    grounding: GroundingCheck | None = None


class ColumnProfile(_Schema):
    """Profile of one dataset column. All values are JSON-serialisable."""

    name: str
    dtype: str = Field(description="pandas dtype as loaded.")
    semantic_type: SemanticType
    missing_pct: float = Field(ge=0.0, le=100.0, description="Percentage of missing values, 0-100.")
    n_unique: int = Field(ge=0, description="Distinct non-missing values.")
    sample_values: list[Any] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict, description="Type-specific statistics.")


class DatasetProfile(_Schema):
    """Profile of a whole dataset, produced by idss.ml.profiler.profile_dataset."""

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    columns: list[ColumnProfile]
    duplicate_rows: int = Field(ge=0)
    memory_mb: float = Field(ge=0.0)
    suggested_target: str | None = None
    suggested_task_type: TaskType | None = None
    time_column: str | None = Field(default=None, description="Complete, unique, increasing datetime column, if any.")
    warnings: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile:
        """Return the profile of column `name` (KeyError if absent)."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def columns_of(self, *semantic_types: SemanticType) -> list[str]:
        """Names of columns with any of the given semantic types, in dataset order."""
        return [col.name for col in self.columns if col.semantic_type in semantic_types]

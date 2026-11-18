"""Knowledge Agent: retrieval-augmented planning.

Retrieves best-practice notes relevant to the run from the local TF-IDF index (plus web results only if a
search API key is configured) and condenses them into a short KnowledgeBrief with the LLM. If the LLM is
unavailable, the brief is simply the top retrieved passages. Never raises.
"""

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient
from idss.core.logger import get_logger
from idss.core.prompts import format_profile, render_prompt
from idss.core.schemas import KnowledgeBrief, KnowledgeChunk, DatasetProfile, UserRequirement
from idss.knowledge.retriever import retrieve
from idss.knowledge.vector_store import KnowledgeIndex
from idss.knowledge.web_search import web_search

logger = get_logger()

MAX_INSIGHTS = 6
FALLBACK_INSIGHT_CHARS = 300
SMALL_DATA_ROWS = 1_000
LARGE_DATA_ROWS = 200_000


class BriefInsights(BaseModel):
    insights: list[str] = Field(min_length=1)


class KnowledgeAgent:
    def __init__(self, llm_client: LLMClient | None, *, index: KnowledgeIndex | None = None, k: int = 4,
                 use_web: bool = True) -> None:
        self.llm_client = llm_client
        self.index = index
        self.k = k
        self.use_web = use_web

    def brief(self, requirement: UserRequirement | None, profile: DatasetProfile | None) -> KnowledgeBrief:
        query = build_query(requirement, profile)
        chunks = self._retrieve(query)
        if not chunks:
            return KnowledgeBrief(query=query)
        sources = list(dict.fromkeys(chunk.source for chunk in chunks))

        if self.llm_client is not None:
            try:
                messages = render_prompt(
                    "knowledge_brief",
                    requirement_summary=_requirement_summary(requirement),
                    dataset_context=format_profile(profile),
                    notes="\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks),
                )
                result = self.llm_client.chat_structured(messages, BriefInsights)
                insights = [text.strip() for text in result.insights if text.strip()][:MAX_INSIGHTS]
                if insights:
                    return KnowledgeBrief(query=query, insights=insights, sources=sources, chunks=chunks, generated_by="llm")
            except Exception as exc:  # the brief is advisory; fall back to the raw passages
                logger.warning("Knowledge brief by LLM failed ({}: {}); using retrieved passages.", type(exc).__name__, exc)

        insights = [_shorten(chunk.text) for chunk in chunks]
        return KnowledgeBrief(query=query, insights=insights, sources=sources, chunks=chunks, generated_by="retrieval")

    def _retrieve(self, query: str) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        try:
            chunks = retrieve(query, k=self.k, index=self.index)
        except Exception as exc:  # e.g. unreadable notes directory
            logger.warning("Knowledge retrieval failed ({}: {}).", type(exc).__name__, exc)
        if self.use_web:
            chunks += web_search(query)
        return chunks


def build_query(requirement: UserRequirement | None, profile: DatasetProfile | None) -> str:
    """A keyword-rich query describing the task, goal, constraints and notable data properties."""
    parts: list[str] = []
    if requirement is not None:
        problem = requirement.problem
        parts += [problem.task_type.replace("_", " "), problem.description, problem.domain, *problem.constraints]
        parts += [metric.name for metric in problem.performance_metrics]
        if requirement.target:
            parts.append(f"predict {requirement.target}")
    if profile is not None:
        size = "small" if profile.n_rows < SMALL_DATA_ROWS else "large" if profile.n_rows > LARGE_DATA_ROWS else "medium"
        parts.append(f"{size} dataset")
        warnings = " ".join(profile.warnings).lower()
        if "imbalanced" in warnings:
            parts.append("class imbalance minority class weighting threshold")
        if "leakage" in warnings:
            parts.append("feature leakage")
        if profile.time_column:
            parts.append("time series forecasting seasonality lag features")
    return " ".join(part for part in parts if part).strip()


def _requirement_summary(requirement: UserRequirement | None) -> str:
    if requirement is None:
        return "No requirement available."
    problem = requirement.problem
    lines = [
        f"Task: {problem.task_type}", f"Goal: {problem.description}", f"Target: {requirement.target or 'none'}",
        f"Domain: {problem.domain or 'unspecified'}",
        f"Metrics: {', '.join(m.name for m in problem.performance_metrics) or 'default'}",
        f"Constraints: {'; '.join(problem.constraints) or 'none'}",
    ]
    return "\n".join(lines)


def _shorten(text: str, limit: int = FALLBACK_INSIGHT_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(". ", 0, limit)
    return text[: cut + 1] if cut > limit // 3 else text[:limit].rsplit(" ", 1)[0] + "..."

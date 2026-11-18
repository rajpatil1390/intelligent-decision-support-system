"""Analysis Agent: answer a custom question about the user's data with LLM-written pandas code.

The LLM sees column names and types only (never values) and writes code; the code runs in the sandbox.
If it fails, the LLM gets exactly one repair attempt with the error message. The answer text is built
deterministically from the sandbox output, so raw result values are not sent back to the LLM.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from idss.core.llm_client import LLMClient, LLMError
from idss.core.logger import get_logger
from idss.core.prompts import format_columns, render_prompt
from idss.core.schemas import DatasetProfile
from idss.execution.sandbox import DEFAULT_TIMEOUT_S, SandboxResult, run_code

logger = get_logger()

MAX_ANSWER_ITEMS = 20
MAX_STDOUT_IN_ANSWER = 1_000


class AnalysisCode(BaseModel):
    code: str = Field(min_length=1)
    explanation: str = ""


class AnalysisAnswer(BaseModel):
    question: str
    ok: bool
    answer: str
    code: str = ""
    result: Any = None
    stdout: str = ""
    attempts: int = 0
    error: str | None = None
    asked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalysisAgent:
    def __init__(self, llm_client: LLMClient | None, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.llm_client = llm_client
        self.timeout_s = timeout_s

    def ask(self, question: str, profile: DatasetProfile | None, df_path: str | Path) -> AnalysisAnswer:
        """Generate code, run it in the sandbox, repair once on failure. Never raises."""
        columns = format_columns(profile)
        try:
            generated = self._generate("analysis_code", question=question, columns=columns)
        except LLMError as exc:
            return _failed(question, f"The assistant could not write analysis code: {exc}")

        run = run_code(generated.code, df_path, self.timeout_s)
        attempts = 1
        if not run.ok:
            logger.info("Analysis code failed ({}); asking the LLM for one repair.", run.error)
            try:
                generated = self._generate("analysis_repair", question=question, columns=columns,
                                           code=generated.code, error=run.error or "unknown error")
            except LLMError as exc:
                return _failed(question, f"The analysis failed ({run.error}) and could not be repaired: {exc}",
                               generated.code, attempts)
            run = run_code(generated.code, df_path, self.timeout_s)
            attempts = 2

        if not run.ok:
            return _failed(question, f"Sorry, I could not compute an answer: {run.error}", generated.code, attempts, run)
        return AnalysisAnswer(
            question=question,
            ok=True,
            answer=format_answer(generated.explanation, run),
            code=generated.code,
            result=run.result,
            stdout=run.stdout,
            attempts=attempts,
        )

    def _generate(self, prompt: str, **variables: str) -> AnalysisCode:
        if self.llm_client is None:
            raise LLMError("No LLM client is configured.")
        return self.llm_client.chat_structured(render_prompt(prompt, **variables), AnalysisCode)


def format_answer(explanation: str, run: SandboxResult) -> str:
    parts = [explanation.strip()] if explanation.strip() else []
    if run.stdout.strip():
        parts.append(run.stdout.strip()[:MAX_STDOUT_IN_ANSWER])
    if run.result is not None:
        parts.append(render_result(run.result))
    return "\n".join(parts) or "The code ran but produced no result."


def render_result(value: Any) -> str:
    """Readable text for a JSON-like result: scalars, dicts, lists and lists of records."""
    if isinstance(value, dict):
        lines = [f"- {key}: {_fmt(item)}" for key, item in list(value.items())[:MAX_ANSWER_ITEMS]]
        return "Result:\n" + "\n".join(lines) + _more(len(value))
    if isinstance(value, list):
        if value and all(isinstance(row, dict) for row in value):
            header = list(value[0])
            rows = [" | ".join(_fmt(row.get(col)) for col in header) for row in value[:MAX_ANSWER_ITEMS]]
            return "Result:\n" + " | ".join(header) + "\n" + "\n".join(rows) + _more(len(value))
        return "Result: " + ", ".join(_fmt(item) for item in value[:MAX_ANSWER_ITEMS]) + _more(len(value))
    return f"Result: {_fmt(value)}"


def _fmt(value: Any) -> str:
    return f"{value:,.4g}" if isinstance(value, float) else str(value)


def _more(count: int) -> str:
    return f"\n(showing {MAX_ANSWER_ITEMS} of {count})" if count > MAX_ANSWER_ITEMS else ""


def _failed(question: str, message: str, code: str = "", attempts: int = 0, run: SandboxResult | None = None) -> AnalysisAnswer:
    return AnalysisAnswer(question=question, ok=False, answer=message, code=code, attempts=attempts,
                          stdout=run.stdout if run else "", error=(run.error if run else message))

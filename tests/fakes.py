"""Test doubles shared by several test modules."""

from typing import Any

from pydantic import BaseModel

from idss.core.llm_client import LLMError
from idss.core.schemas import VerificationReport


class PassingVerifier:
    """Verifier stand-in for state-machine tests that are not about verification."""

    def verify(self, *args: Any, **kwargs: Any) -> VerificationReport:
        return VerificationReport(passed=True, checks=[])


class FakeLLM:
    """Stands in for LLMClient. Returns queued responses per schema (the last one repeats) and records calls.

    A schema with no response raises LLMError, which is how tests simulate an unavailable LLM.
    """

    def __init__(self, responses: dict[type[BaseModel], Any], chat_reply: str | Exception = "LLM data summary.") -> None:
        self.responses = {schema: list(value) if isinstance(value, list) else value for schema, value in responses.items()}
        self.chat_reply = chat_reply
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat_structured(self, messages: list[dict[str, str]], schema: type[BaseModel], temperature: float = 0.0) -> Any:
        self.calls.append((schema.__name__, messages))
        if schema not in self.responses:
            raise LLMError(f"No fake response for {schema.__name__}.")
        value = self.responses[schema]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Exception):
            raise value
        return value

    def chat(self, messages: list[dict[str, str]], temperature: float | None = 0.3) -> str:
        self.calls.append(("chat", messages))
        if isinstance(self.chat_reply, Exception):
            raise self.chat_reply
        return self.chat_reply

    def count(self, name: str) -> int:
        return sum(1 for called, _ in self.calls if called == name)

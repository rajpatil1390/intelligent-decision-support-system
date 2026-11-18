"""LLM client for any OpenAI-compatible endpoint (OpenAI, Groq, Gemini, Ollama).

Every call has a timeout, retries are bounded (MAX_RETRIES) with exponential backoff, and token
usage + latency of each API call is appended to LLMClient.usage_log.
"""

import json
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from configs.settings import Settings, get_settings
from idss.core.logger import get_logger

logger = get_logger()

MAX_RETRIES = 3  # retries after the first call, so at most MAX_RETRIES + 1 API calls
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 8.0
_MAX_FEEDBACK_CHARS = 2000

# Network failures, timeouts, rate limits and 5xx are worth retrying; auth/bad-request errors are not.
_RETRYABLE_API_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_BLOCK = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)
_JSON_START = re.compile(r"[{\[]")

Message = dict[str, str]


class LLMError(Exception):
    """An LLM call failed: configuration, network, timeout or API error."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMValidationError(LLMError):
    """The LLM never produced output matching the requested schema within the retry budget."""

    def __init__(self, message: str, *, raw_output: str = "", errors: str = "") -> None:
        super().__init__(message, retryable=False)
        self.raw_output = raw_output
        self.errors = errors


class UsageRecord(BaseModel):
    """Token usage and latency of one API call."""

    timestamp: datetime
    provider: str
    model: str
    operation: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    success: bool
    error: str | None = None


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Extract the first JSON object or array from LLM output.

    Handles bare JSON, ```json fenced blocks, surrounding prose and <think>...</think> reasoning blocks.
    Raises ValueError if no JSON object or array can be decoded.
    """
    if not text or not text.strip():
        raise ValueError("LLM output is empty; no JSON to extract.")
    cleaned = _THINK_BLOCK.sub("", text).strip()

    candidates = [cleaned, *(block.strip() for block in _FENCED_BLOCK.findall(cleaned))]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict | list):
            return value

    # Fall back to scanning for the first position where a complete JSON value decodes.
    decoder = json.JSONDecoder()
    for match in _JSON_START.finditer(cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("No valid JSON object or array found in LLM output.")


class LLMClient:
    """Chat client for the provider configured in settings."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        self.model = self.settings.llm_model.strip()
        self.timeout = self.settings.llm_timeout
        if not self.model:
            raise LLMError("LLM_MODEL is not set. Add it to your .env file.")
        self._client = client if client is not None else self._build_client()
        self._sleep = sleep
        self.usage_log: list[UsageRecord] = []

    def chat(self, messages: list[Message], temperature: float | None = 0.3) -> str:
        """Return the model's reply text. Pass temperature=None for models that reject the parameter."""
        last_error: LLMError | None = None
        for attempt in range(MAX_RETRIES + 1):
            if last_error is not None:
                self._backoff(attempt, last_error)
            try:
                return self._complete(messages, temperature=temperature, operation="chat")
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
        raise LLMError(f"LLM call failed after {MAX_RETRIES} retries: {last_error}", retryable=True) from last_error

    def chat_structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        temperature: float | None = 0.0,
    ) -> T:
        """Return the model's reply parsed and validated as `schema`.

        Invalid JSON or schema errors are fed back to the model so it can correct itself; transient API
        errors are retried with backoff. Both share one budget of MAX_RETRIES retries.
        """
        conversation = _with_schema_instruction(messages, schema)
        last_error: LLMError | None = None
        for attempt in range(MAX_RETRIES + 1):
            if last_error is not None and last_error.retryable:
                self._backoff(attempt, last_error)
            try:
                raw = self._complete(conversation, temperature=temperature, operation="chat_structured", json_mode=True)
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                continue

            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                feedback = _describe_error(exc)
                logger.warning("{} output invalid (attempt {}/{}): {}", schema.__name__, attempt + 1, MAX_RETRIES + 1, feedback)
                last_error = LLMValidationError(
                    f"LLM output did not match {schema.__name__} after {MAX_RETRIES + 1} attempts. Last error: {feedback}",
                    raw_output=raw,
                    errors=feedback,
                )
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": raw[:_MAX_FEEDBACK_CHARS]},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid.\n{feedback[:_MAX_FEEDBACK_CHARS]}\n"
                            "Reply again with ONLY a corrected JSON object that matches the schema."
                        ),
                    },
                ]

        if isinstance(last_error, LLMValidationError):
            raise last_error
        raise LLMError(f"LLM call failed after {MAX_RETRIES} retries: {last_error}", retryable=True) from last_error

    def _build_client(self) -> OpenAI:
        api_key = self.settings.llm_api_key.get_secret_value().strip()
        if not api_key:
            if self.provider != "ollama":
                raise LLMError(f"LLM_API_KEY is empty for provider '{self.provider}'. Add it to your .env file.")
            api_key = "ollama"  # Ollama ignores the key, but the SDK requires a non-empty value.
        # max_retries=0: retries are handled here so the total stays within MAX_RETRIES.
        return OpenAI(api_key=api_key, base_url=self.settings.llm_base_url, timeout=self.timeout, max_retries=0)

    def _complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None,
        operation: str,
        json_mode: bool = False,
    ) -> str:
        """Make exactly one API call, record its usage, and return the reply text."""
        request: dict[str, Any] = {"model": self.model, "messages": messages, "timeout": self.timeout}
        if temperature is not None:
            request["temperature"] = temperature
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            error = LLMError(f"{type(exc).__name__}: {exc}", retryable=isinstance(exc, _RETRYABLE_API_ERRORS))
            self._record(operation, time.perf_counter() - start, response=None, error=str(error))
            raise error from exc

        latency = time.perf_counter() - start
        choices = getattr(response, "choices", None) or []
        content = (choices[0].message.content or "").strip() if choices else ""
        if not content:
            error = LLMError("LLM returned an empty response.", retryable=True)
            self._record(operation, latency, response=response, error=str(error))
            raise error
        self._record(operation, latency, response=response)
        return content

    def _record(self, operation: str, latency_s: float, *, response: Any, error: str | None = None) -> None:
        usage = getattr(response, "usage", None)
        record = UsageRecord(
            timestamp=datetime.now(UTC),
            provider=self.provider,
            model=self.model,
            operation=operation,
            latency_s=round(latency_s, 4),
            prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
            completion_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
            total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
            success=error is None,
            error=error,
        )
        self.usage_log.append(record)
        logger.debug(
            "LLM {} {}/{}: {:.2f}s, tokens={}, success={}",
            operation, self.provider, self.model, record.latency_s, record.total_tokens, record.success,
        )

    def _backoff(self, attempt: int, error: LLMError) -> None:
        delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS) + random.uniform(0, 0.25)
        logger.warning("LLM call failed ({}); retry {}/{} in {:.1f}s", error, attempt, MAX_RETRIES, delay)
        self._sleep(delay)


def _with_schema_instruction(messages: list[Message], schema: type[BaseModel]) -> list[Message]:
    """Copy `messages` and add a system instruction describing the required JSON schema."""
    instruction = (
        "Respond with a single JSON object only, with no markdown fences and no extra text. "
        "It must conform to this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    conversation = [dict(message) for message in messages]
    if conversation and conversation[0].get("role") == "system":
        conversation[0]["content"] = f"{conversation[0]['content']}\n\n{instruction}"
    else:
        conversation.insert(0, {"role": "system", "content": instruction})
    return conversation


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        lines = [
            f"- {'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors(include_url=False)
        ]
        return "Schema validation failed:\n" + "\n".join(lines)
    return f"Could not parse JSON: {exc}"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None

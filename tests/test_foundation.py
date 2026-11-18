"""Foundation tests: settings, schemas, JSON extraction and LLMClient (mocked; no network calls)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx2
import openai
import pytest
from pydantic import BaseModel, Field, ValidationError

from configs.settings import DEFAULT_BASE_URLS, Settings
from idss.core.llm_client import MAX_RETRIES, LLMClient, LLMError, LLMValidationError, extract_json
from idss.core.schemas import UserRequirement

ENV_VARS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_TIMEOUT", "LOG_LEVEL", "WORKSPACE_DIR")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove IDSS variables from the process environment so tests don't depend on the developer's shell."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def make_settings(**overrides: Any) -> Settings:
    values = {"llm_provider": "openai", "llm_model": "test-model", "llm_api_key": "test-key"} | overrides
    return Settings(_env_file=None, **values)


# --------------------------------------------------------------------------- settings


def test_settings_defaults(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == DEFAULT_BASE_URLS["openai"]
    assert settings.llm_timeout == 60
    assert settings.log_level == "INFO"
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.logs_dir == settings.workspace_dir / "logs"


def test_settings_load_from_env_file(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=Groq\nLLM_MODEL=some-model\nLLM_API_KEY=secret-123\nLLM_BASE_URL=\nLLM_TIMEOUT=30\nLOG_LEVEL=debug\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "some-model"
    assert settings.llm_base_url == DEFAULT_BASE_URLS["groq"]
    assert settings.llm_timeout == 30.0
    assert settings.log_level == "DEBUG"
    assert "secret-123" not in repr(settings)


def test_settings_env_var_and_explicit_base_url(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("LLM_BASE_URL", "http://192.168.1.10:11434/v1")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.llm_base_url == "http://192.168.1.10:11434/v1"


@pytest.mark.parametrize("field, value", [("llm_provider", "anthropic"), ("llm_timeout", 0), ("log_level", "LOUD")])
def test_settings_reject_invalid_values(clean_env: pytest.MonkeyPatch, field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


# --------------------------------------------------------------------------- schemas

VALID_REQUIREMENT: dict[str, Any] = {
    "user": {"intent": "build", "expertise": "low"},
    "problem": {
        "task_type": "tabular_classification",
        "description": "  Predict which customers will churn next month.  ",
        "domain": "telecom",
        "performance_metrics": [{"name": "f1", "target": 0.8}],
        "constraints": ["must be explainable"],
    },
    "datasets": [{"name": "churn.csv", "source": "upload", "target_column": "churned"}],
    "models": [{"name": "XGBoost", "model_type": "ensemble"}],
    "confidence": 0.85,
    "missing_info": [],
    "unexpected_key": "ignored",
}


def test_user_requirement_valid() -> None:
    req = UserRequirement.model_validate(VALID_REQUIREMENT)
    assert req.problem.task_type == "tabular_classification"
    assert req.problem.description == "Predict which customers will churn next month."
    assert req.problem.performance_metrics[0].target == 0.8
    assert req.datasets[0].target_column == "churned"
    assert req.models[0].model_type == "ensemble"


def test_user_requirement_minimal_uses_defaults() -> None:
    req = UserRequirement.model_validate(
        {"problem": {"task_type": "tabular_regression", "description": "Forecast house prices."}, "confidence": 0.4}
    )
    assert req.user.intent == "build"
    assert req.datasets == [] and req.models == [] and req.missing_info == []


@pytest.mark.parametrize(
    "patch",
    [
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"problem": {"task_type": "image_classification", "description": "x"}},
        {"problem": {"task_type": "tabular_regression", "description": ""}},
        {"user": {"intent": "build", "expertise": "guru"}},
        {"datasets": [{"source": "ftp"}]},
    ],
    ids=["confidence>1", "confidence<0", "bad-task-type", "empty-description", "bad-expertise", "bad-source"],
)
def test_user_requirement_invalid(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        UserRequirement.model_validate(VALID_REQUIREMENT | patch)


def test_user_requirement_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        UserRequirement.model_validate({"confidence": 0.5})
    with pytest.raises(ValidationError):
        UserRequirement.model_validate({"problem": VALID_REQUIREMENT["problem"]})


# --------------------------------------------------------------------------- extract_json


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1, "b": [1, 2]}\n```', {"a": 1, "b": [1, 2]}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here is the result:\n{"a": {"b": "x}y"}}\nHope this helps.', {"a": {"b": "x}y"}}),
        ('Some text ```json\n{"a": 2}\n``` trailing', {"a": 2}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('<think>maybe {"draft": true}</think>\n{"final": 1}', {"final": 1}),
        ('Bad first {not json} then good {"ok": true}', {"ok": True}),
    ],
    ids=["bare", "fenced-json", "fenced-plain", "prose-and-braces-in-string", "inline-fence", "array", "think-block", "skip-invalid"],
)
def test_extract_json(text: str, expected: Any) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no json here", '{"a": 1', "42", '"just a string"'])
def test_extract_json_raises(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json(text)


# --------------------------------------------------------------------------- LLMClient, mocked


class Verdict(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


def fake_response(content: str, prompt_tokens: int = 12, completion_tokens: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens
        ),
    )


def make_client(*outcomes: Any) -> tuple[LLMClient, MagicMock, list[float]]:
    """LLMClient backed by a mock whose create() returns/raises `outcomes` in order. Sleeps are recorded, not slept."""
    fake = MagicMock()
    fake.chat.completions.create.side_effect = list(outcomes)
    sleeps: list[float] = []
    return LLMClient(make_settings(), client=fake, sleep=sleeps.append), fake, sleeps


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://llm.test/v1/chat/completions"))


def test_chat_returns_text_and_tracks_usage() -> None:
    client, fake, _ = make_client(fake_response("  Hello!  "))
    assert client.chat([{"role": "user", "content": "Hi"}]) == "Hello!"
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["timeout"] == 60
    assert kwargs["temperature"] == 0.3
    [record] = client.usage_log
    assert record.success and record.total_tokens == 20 and record.latency_s >= 0


def test_chat_structured_success() -> None:
    client, fake, sleeps = make_client(fake_response('```json\n{"label": "approve", "score": 0.9}\n```'))
    result = client.chat_structured([{"role": "user", "content": "Decide."}], Verdict)
    assert result == Verdict(label="approve", score=0.9)
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][0]["role"] == "system"
    assert '"score"' in kwargs["messages"][0]["content"]  # schema is in the system prompt
    assert len(client.usage_log) == 1 and sleeps == []


def test_chat_structured_retries_with_validation_feedback() -> None:
    client, fake, sleeps = make_client(
        fake_response('{"label": "approve", "score": 7}'),
        fake_response('{"label": "approve", "score": 0.7}'),
    )
    messages = [{"role": "system", "content": "You are a judge."}, {"role": "user", "content": "Decide."}]
    result = client.chat_structured(messages, Verdict)

    assert result.score == 0.7
    assert fake.chat.completions.create.call_count == 2
    retry_messages = fake.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"label": "approve", "score": 7}'}
    assert "score" in retry_messages[-1]["content"] and "less than or equal to 1" in retry_messages[-1]["content"]
    assert messages[0]["content"] == "You are a judge."  # caller's list is not mutated
    assert len(client.usage_log) == 2 and sleeps == []


def test_chat_structured_gives_up_after_max_retries() -> None:
    bad = [fake_response("not json at all") for _ in range(MAX_RETRIES + 1)]
    client, fake, _ = make_client(*bad)
    with pytest.raises(LLMValidationError) as exc_info:
        client.chat_structured([{"role": "user", "content": "Decide."}], Verdict)
    assert fake.chat.completions.create.call_count == MAX_RETRIES + 1
    assert exc_info.value.raw_output == "not json at all"


def test_transient_api_error_is_retried_with_backoff() -> None:
    client, fake, sleeps = make_client(connection_error(), fake_response('{"label": "ok", "score": 0.5}'))
    result = client.chat_structured([{"role": "user", "content": "Decide."}], Verdict)
    assert result.label == "ok"
    assert len(sleeps) == 1 and sleeps[0] >= 1.0
    assert [r.success for r in client.usage_log] == [False, True]


def test_non_retryable_api_error_raises_immediately() -> None:
    request = httpx2.Request("POST", "https://llm.test/v1/chat/completions")
    auth_error = openai.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)
    client, fake, sleeps = make_client(auth_error)
    with pytest.raises(LLMError) as exc_info:
        client.chat([{"role": "user", "content": "Hi"}])
    assert not exc_info.value.retryable
    assert fake.chat.completions.create.call_count == 1 and sleeps == []


def test_client_config_errors() -> None:
    with pytest.raises(LLMError, match="LLM_MODEL"):
        LLMClient(make_settings(llm_model=""))
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient(make_settings(llm_api_key=""))
    # Ollama needs no key; building the client makes no network call.
    assert LLMClient(make_settings(llm_provider="ollama", llm_api_key="")).provider == "ollama"

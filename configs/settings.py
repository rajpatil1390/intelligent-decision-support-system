"""Application settings loaded from environment variables and the project .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["openai", "groq", "gemini", "ollama"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
}


class Settings(BaseSettings):
    """Typed settings. Environment variables override values from .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: LLMProvider = "openai"
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = ""
    llm_timeout: float = Field(default=60.0, gt=0)
    log_level: LogLevel = "INFO"

    # Optional web search for the knowledge agent; empty keys disable it.
    tavily_api_key: SecretStr = SecretStr("")
    serpapi_key: SecretStr = SecretStr("")

    workspace_dir: Path = PROJECT_ROOT / "workspace"

    # Public demo (env IDSS_PUBLIC_DEMO=1): no Ask Your Data (code execution), 10 MB uploads, fast budget only,
    # runs older than 24 h deleted at API startup, and a "do not upload private data" banner in the UI.
    public_demo: bool = Field(default=False, validation_alias="IDSS_PUBLIC_DEMO")

    @field_validator("public_demo", mode="before")
    @classmethod
    def _blank_is_false(cls, value: object) -> object:
        return False if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_provider", "log_level", mode="before")
    @classmethod
    def _normalize_case(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value.upper() if info.field_name == "log_level" else value.lower()

    @model_validator(mode="after")
    def _default_base_url(self) -> "Settings":
        if not self.llm_base_url.strip():
            self.llm_base_url = DEFAULT_BASE_URLS[self.llm_provider]
        return self

    @property
    def logs_dir(self) -> Path:
        return self.workspace_dir / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_dir / "runs"

    @property
    def uploads_dir(self) -> Path:
        return self.workspace_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / "idss.db"

    @property
    def llm_configured(self) -> bool:
        """True when an LLM can be called: a model name plus an API key (Ollama needs no key)."""
        has_key = bool(self.llm_api_key.get_secret_value().strip()) or self.llm_provider == "ollama"
        return bool(self.llm_model.strip()) and has_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached). Call get_settings.cache_clear() to reload."""
    return Settings()

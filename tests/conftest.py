"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from configs.settings import get_settings
from idss.core.logger import setup_logging


@pytest.fixture(autouse=True, scope="session")
def _isolated_workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the workspace (logs, runs, knowledge index cache) at a temp dir so tests never touch workspace/,
    disable optional web search so tests never make network calls even if .env has search keys, and switch the
    public demo off (tests that need it turn it on explicitly) even if the environment sets IDSS_PUBLIC_DEMO."""
    settings = get_settings()
    original = (settings.workspace_dir, settings.tavily_api_key, settings.serpapi_key, settings.public_demo)
    settings.workspace_dir = tmp_path_factory.mktemp("workspace")
    settings.tavily_api_key = settings.serpapi_key = SecretStr("")
    settings.public_demo = False
    setup_logging(level="DEBUG", log_dir=settings.logs_dir, force=True)
    yield
    settings.workspace_dir, settings.tavily_api_key, settings.serpapi_key, settings.public_demo = original

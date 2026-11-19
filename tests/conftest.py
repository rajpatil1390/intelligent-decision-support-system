"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from configs.settings import get_settings
from idss.core.logger import setup_logging

# Read by the UI (ui/embedded_api.py) straight from the environment: unset for the whole session, so no UI test
# starts a real server; tests/test_embedded_api.py switches embedded mode on explicitly.
_UI_ONLY_ENV = ("IDSS_EMBEDDED_API", "IDSS_PUBLIC_DEMO", "IDSS_API_URL")


@pytest.fixture(autouse=True, scope="session")
def _isolated_workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the workspace (logs, runs, knowledge index cache) at a temp dir so tests never touch workspace/,
    disable optional web search so tests never make network calls even if .env has search keys, and switch the
    public demo and the embedded API off (tests that need them turn them on explicitly) even if the environment
    sets IDSS_PUBLIC_DEMO / IDSS_EMBEDDED_API."""
    settings = get_settings()
    original = (settings.workspace_dir, settings.tavily_api_key, settings.serpapi_key, settings.public_demo)
    settings.workspace_dir = tmp_path_factory.mktemp("workspace")
    settings.tavily_api_key = settings.serpapi_key = SecretStr("")
    settings.public_demo = False
    setup_logging(level="DEBUG", log_dir=settings.logs_dir, force=True)
    with pytest.MonkeyPatch.context() as env:
        for name in _UI_ONLY_ENV:
            env.delenv(name, raising=False)
        yield
    settings.workspace_dir, settings.tavily_api_key, settings.serpapi_key, settings.public_demo = original

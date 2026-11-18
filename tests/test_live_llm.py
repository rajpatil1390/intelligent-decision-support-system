"""Smoke tests against the real LLM configured in .env.

Deselected by default (see pytest.ini). Run with:  .venv\\Scripts\\python.exe -m pytest -m live
Skipped automatically when no model / API key is configured. These make paid API calls.
"""

from pathlib import Path

import pytest

from configs.settings import Settings
from idss.agents.requirement_agent import RequirementAgent
from idss.core.llm_client import LLMClient
from idss.ml.data_loader import load_dataset
from idss.ml.profiler import profile_dataset

pytestmark = pytest.mark.live

CHURN_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "customer_churn.csv"


@pytest.fixture(scope="module")
def live_agent() -> RequirementAgent:
    settings = Settings()  # loads .env at runtime
    has_key = settings.llm_provider == "ollama" or bool(settings.llm_api_key.get_secret_value().strip())
    if not settings.llm_model.strip() or not has_key:
        pytest.skip("No LLM configured: set LLM_MODEL and LLM_API_KEY in .env to run live tests.")
    return RequirementAgent(LLMClient(settings))


def test_live_relevance(live_agent: RequirementAgent) -> None:
    assert live_agent.is_relevant("Which customers are likely to churn next quarter?") is True
    assert live_agent.is_relevant("Tell me a joke about cats.") is False


def test_live_parse_churn_request(live_agent: RequirementAgent) -> None:
    profile = profile_dataset(load_dataset(CHURN_CSV))
    req = live_agent.parse("Which customers are likely to churn and what should we do to retain them?", profile)
    assert req.target == "churn"
    assert req.problem.task_type == "tabular_classification"
    assert req.problem.performance_metrics

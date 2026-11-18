"""Optional web search (Tavily or SerpAPI). Used only when an API key is configured; otherwise returns [].

Never required: any error (no key, network, quota, unexpected response) yields an empty list. When a
key is set, the query text (task description and domain, no data rows) is sent to that provider.
"""

import json
import urllib.parse
import urllib.request
from typing import Any

from configs.settings import Settings, get_settings
from idss.core.logger import get_logger
from idss.core.schemas import KnowledgeChunk

logger = get_logger()

TAVILY_URL = "https://api.tavily.com/search"
SERPAPI_URL = "https://serpapi.com/search.json"
TIMEOUT_S = 10
MAX_SNIPPET_CHARS = 600


def web_search(query: str, max_results: int = 3, *, settings: Settings | None = None) -> list[KnowledgeChunk]:
    settings = settings or get_settings()
    tavily_key = settings.tavily_api_key.get_secret_value().strip()
    serpapi_key = settings.serpapi_key.get_secret_value().strip()
    if not query.strip() or not (tavily_key or serpapi_key):
        return []
    try:
        if tavily_key:
            return _tavily(query, max_results, tavily_key)
        return _serpapi(query, max_results, serpapi_key)
    except Exception as exc:  # optional feature: never let it break planning
        logger.warning("Web search failed ({}: {}); continuing without it.", type(exc).__name__, exc)
        return []


def _tavily(query: str, max_results: int, key: str) -> list[KnowledgeChunk]:
    body = json.dumps({"query": query, "max_results": max_results}).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    payload = _fetch_json(request)
    return [
        KnowledgeChunk(source=item.get("url", "web"), text=str(item.get("content", ""))[:MAX_SNIPPET_CHARS],
                       score=float(item.get("score") or 0.0))
        for item in payload.get("results", [])[:max_results]
        if item.get("content")
    ]


def _serpapi(query: str, max_results: int, key: str) -> list[KnowledgeChunk]:
    params = urllib.parse.urlencode({"q": query, "engine": "google", "num": max_results, "api_key": key})
    payload = _fetch_json(urllib.request.Request(f"{SERPAPI_URL}?{params}", method="GET"))
    return [
        KnowledgeChunk(source=item.get("link", "web"), text=str(item.get("snippet", ""))[:MAX_SNIPPET_CHARS])
        for item in payload.get("organic_results", [])[:max_results]
        if item.get("snippet")
    ]


def _fetch_json(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 (fixed https URLs)
        return json.loads(response.read().decode("utf-8"))

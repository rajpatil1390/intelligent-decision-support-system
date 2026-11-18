"""Retrieve best-practice passages from the local knowledge base."""

from pathlib import Path

from configs.settings import get_settings
from idss.core.schemas import KnowledgeChunk
from idss.knowledge.vector_store import INDEX_FILENAME, KnowledgeIndex

_indexes: dict[Path, KnowledgeIndex] = {}


def get_default_index() -> KnowledgeIndex:
    """The shared index over data/knowledge, cached per index location (workspace may change in tests)."""
    path = get_settings().workspace_dir / INDEX_FILENAME
    if path not in _indexes:
        _indexes[path] = KnowledgeIndex(index_path=path)
    return _indexes[path]


def retrieve(query: str, k: int = 4, *, index: KnowledgeIndex | None = None) -> list[KnowledgeChunk]:
    """Top-`k` knowledge chunks for `query` (rebuilds the index first if any note changed)."""
    return (index or get_default_index()).search(query, k=k)

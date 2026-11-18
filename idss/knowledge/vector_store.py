"""Offline TF-IDF index over the Markdown notes in data/knowledge/.

No model downloads and no GPU: scikit-learn's TfidfVectorizer turns ~500-character chunks into sparse
vectors, and cosine similarity ranks them for a query. The fitted index is cached with joblib and rebuilt
automatically when any note is added, removed or edited (content-hash check).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from configs.settings import PROJECT_ROOT, get_settings
from idss.core.logger import get_logger
from idss.core.schemas import KnowledgeChunk

logger = get_logger()

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
INDEX_FILENAME = "knowledge_index.joblib"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
_INDEX_FORMAT = 1  # bump when the stored structure changes, forcing a rebuild


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~`size`-character chunks that overlap by ~`overlap` characters, cutting at spaces."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start  # begin the next chunk on a word boundary
    return [chunk for chunk in chunks if chunk]


@dataclass
class _Chunk:
    source: str
    title: str
    text: str


class KnowledgeIndex:
    """TF-IDF index over `knowledge_dir/*.md`, cached at `index_path`."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR, index_path: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir
        self.index_path = index_path if index_path is not None else get_settings().workspace_dir / INDEX_FILENAME
        self._fingerprint: str | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None
        self._chunks: list[_Chunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def ensure_fresh(self) -> bool:
        """Load or (re)build the index so it matches the notes on disk. Returns True if it was rebuilt."""
        current = self._current_fingerprint()
        if self._fingerprint == current:
            return False
        if self._load(current):
            return False
        self._build(current)
        return True

    def search(self, query: str, k: int = 4, *, max_per_source: int = 2) -> list[KnowledgeChunk]:
        """Top-`k` chunks by cosine similarity (at most `max_per_source` per note, score > 0)."""
        self.ensure_fresh()
        if not self._chunks or not query.strip():
            return []
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        results: list[KnowledgeChunk] = []
        per_source: dict[str, int] = {}
        for index in np.argsort(-scores, kind="stable"):
            if scores[index] <= 0 or len(results) == k:
                break
            chunk = self._chunks[index]
            if per_source.get(chunk.source, 0) >= max_per_source:
                continue
            per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
            results.append(KnowledgeChunk(source=chunk.source, text=chunk.text, score=round(float(scores[index]), 4)))
        return results

    # ------------------------------------------------------------------ internals

    def _files(self) -> list[Path]:
        return sorted(self.knowledge_dir.glob("*.md")) if self.knowledge_dir.is_dir() else []

    def _current_fingerprint(self) -> str:
        digest = hashlib.sha256(f"{_INDEX_FORMAT}:{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
        for path in self._files():
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def _load(self, fingerprint: str) -> bool:
        if not self.index_path.is_file():
            return False
        try:
            stored = joblib.load(self.index_path)  # our own cache file, written by _build
        except Exception as exc:  # a corrupt cache is simply rebuilt
            logger.warning("Ignoring unreadable knowledge index {} ({}).", self.index_path, exc)
            return False
        if stored.get("fingerprint") != fingerprint:
            return False
        self._fingerprint, self._vectorizer = fingerprint, stored["vectorizer"]
        self._matrix, self._chunks = stored["matrix"], stored["chunks"]
        logger.debug("Loaded knowledge index ({} chunks) from {}.", len(self._chunks), self.index_path)
        return True

    def _build(self, fingerprint: str) -> None:
        chunks: list[_Chunk] = []
        for path in self._files():
            text = path.read_text(encoding="utf-8")
            title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.stem)
            body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
            chunks += [_Chunk(path.name, title, chunk) for chunk in chunk_text(body)]

        self._fingerprint, self._chunks = fingerprint, chunks
        if chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            # The note title is indexed with every chunk so short chunks still match their topic.
            self._matrix = self._vectorizer.fit_transform([f"{c.title}. {c.text}" for c in chunks])
        else:
            self._vectorizer, self._matrix = None, None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"fingerprint": fingerprint, "vectorizer": self._vectorizer, "matrix": self._matrix,
                     "chunks": self._chunks}, self.index_path)
        logger.info("Built knowledge index: {} chunks from {} notes.", len(chunks), len(self._files()))

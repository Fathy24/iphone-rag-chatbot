"""Sparse lexical retrieval (BM25) for hybrid search.

Dense embeddings capture semantics but miss exact lexical signals — model
names, settings paths ("Settings > General"), button labels, and rare terms
that a user manual is full of. We complement the vector store with a classic
BM25 index (the ``BM25Plus`` variant, mirroring the production backend's
``bm25plus`` configuration) and fuse the two rankings (see
:mod:`app.rag.fusion`).

The corpus (chunk text + citation metadata) is dumped to disk during ingestion
and the BM25 index is rebuilt in-memory at serve time. For a single user guide
this is a few hundred short chunks, so the rebuild is effectively instant and
keeps the design backend-agnostic (works identically whether the dense vectors
live in Qdrant or a local FAISS file).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

from app.logging_config import get_logger

logger = get_logger(__name__)

# Minimal, dependency-free tokenizer: lowercase alphanumeric tokens. Good enough
# for English instructional prose without pulling in a heavy NLP stack.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenization (drops single-character tokens)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


class SparseIndex:
    """In-memory BM25Plus index over the chunk corpus."""

    def __init__(self, records: list[dict]) -> None:
        """Build the index from corpus records.

        Args:
            records: One dict per chunk with at least a ``text`` key plus the
                citation metadata (``chunk_id``, ``page``, ``section``,
                ``source``).
        """
        self._records = records or []
        tokenized = [_tokenize(r.get("text", "")) for r in self._records]
        # BM25Plus needs a non-empty corpus; guard the degenerate case.
        self._bm25 = BM25Plus(tokenized) if tokenized else None

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._records)

    def query(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Return the ``top_k`` highest-scoring records for ``query``.

        Args:
            query: Raw user query.
            top_k: Maximum number of records to return.

        Returns:
            ``(record, bm25_score)`` pairs sorted by descending score; only
            records with a positive score are included.
        """
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[dict, float]] = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                break
            out.append((self._records[i], float(scores[i])))
        return out


def save_corpus(records: list[dict], path: str) -> None:
    """Persist the BM25 corpus to disk as JSON.

    Raises:
        RuntimeError: If the corpus cannot be written.
    """
    try:
        Path(path).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved BM25 corpus (%d records) to '%s'", len(records), path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not write BM25 corpus to '{path}': {exc}") from exc


def load_corpus_records(path: str) -> list[dict] | None:
    """Load the raw corpus records (chunk text + metadata) from disk.

    Shared by the sparse index and the parent-document expander. Returns
    ``None`` when the corpus file is missing.
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        return None
    try:
        return json.loads(corpus_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read corpus '%s': %s", path, exc)
        return None


def load_sparse_index(path: str) -> SparseIndex | None:
    """Load the corpus from disk and build a :class:`SparseIndex`.

    Returns:
        A ready index, or ``None`` if the corpus file is missing (callers then
        gracefully fall back to dense-only retrieval).
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        logger.warning("BM25 corpus '%s' not found; hybrid search disabled.", path)
        return None
    try:
        records = json.loads(corpus_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read BM25 corpus '%s': %s", path, exc)
        return None
    logger.info("Loaded BM25 corpus (%d records) from '%s'", len(records), path)
    return SparseIndex(records)

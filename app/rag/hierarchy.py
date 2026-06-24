"""Coarse level of the hierarchical (coarse-to-fine) retriever.

The guide is organised into clear chapters/sections. Instead of searching every
chunk flatly, we first pick the most relevant *sections*, then search chunks
only inside them. The coarse level represents each section by a **centroid** —
the mean of its child-chunk embeddings — so it needs **no extra LLM or
embedding calls** at ingestion (we already embed every chunk) and is fully
deterministic.

At serve time :class:`SectionRouter` ranks sections by cosine similarity between
the query embedding and the section centroids and returns the top-N section
ids, which the fine stage uses to constrain chunk retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero vectors)."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_section_records(
    chunks: list, vectors: list[list[float]]
) -> list[dict]:
    """Group chunk embeddings by ``section_id`` into section centroid records.

    Args:
        chunks: Chunk ``Document`` objects (carrying ``section_id``/``section``/
            ``page`` metadata), aligned 1:1 with ``vectors``.
        vectors: The embedding for each chunk, in the same order.

    Returns:
        One record per section: ``{section_id, title, pages, n_chunks,
        centroid}`` where ``centroid`` is the L2-normalised mean embedding.
    """
    return build_section_records_from_metas(
        [(c.metadata or {}) for c in chunks], vectors
    )


def build_section_records_from_metas(
    metas: list[dict], vectors: list[list[float]]
) -> list[dict]:
    """Build section centroid records from raw metadata dicts + vectors.

    Same as :func:`build_section_records` but takes metadata dicts directly
    (e.g. Qdrant payloads streamed at serve time) instead of ``Document``
    objects, so the coarse index can be rebuilt from the cloud collection with
    no on-disk section file.

    Args:
        metas: Per-chunk metadata dicts (``section_id``/``section``/``page``),
            aligned 1:1 with ``vectors``.
        vectors: The embedding for each chunk, in the same order.
    """
    buckets: dict[str, dict] = {}
    for meta, vector in zip(metas, vectors):
        meta = meta or {}
        section_id = str(meta.get("section_id", "general"))
        bucket = buckets.setdefault(
            section_id,
            {
                "section_id": section_id,
                "title": str(meta.get("section", "General")),
                "pages": set(),
                "vectors": [],
            },
        )
        bucket["pages"].add(int(meta.get("page", 0)))
        bucket["vectors"].append(vector)

    records: list[dict] = []
    for bucket in buckets.values():
        mat = np.asarray(bucket["vectors"], dtype=np.float32)
        centroid = _l2_normalize(mat.mean(axis=0, keepdims=True))[0]
        records.append(
            {
                "section_id": bucket["section_id"],
                "title": bucket["title"],
                "pages": sorted(p for p in bucket["pages"] if p),
                "n_chunks": len(bucket["vectors"]),
                "centroid": centroid.astype(float).tolist(),
            }
        )
    records.sort(key=lambda r: (r["pages"][0] if r["pages"] else 0))
    return records


def save_section_index(records: list[dict], path: str) -> None:
    """Persist section centroid records to disk as JSON.

    Raises:
        RuntimeError: If the file cannot be written.
    """
    try:
        Path(path).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved section index (%d sections) to '%s'", len(records), path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not write section index to '{path}': {exc}") from exc


class SectionRouter:
    """Ranks sections against a query embedding (the coarse retrieval stage)."""

    def __init__(self, records: list[dict]) -> None:
        self._ids = [r["section_id"] for r in records]
        self._titles = {r["section_id"]: r.get("title", r["section_id"]) for r in records}
        if records:
            self._centroids = _l2_normalize(
                np.asarray([r["centroid"] for r in records], dtype=np.float32)
            )
        else:
            self._centroids = np.empty((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        """Number of indexed sections."""
        return len(self._ids)

    def title_of(self, section_id: str) -> str:
        """Human-readable title for a section id (falls back to the id)."""
        return self._titles.get(section_id, section_id)

    def rank_sections(self, query_vector: list[float]) -> list[tuple[str, str, float]]:
        """Return ALL sections ranked by cosine similarity to the query.

        Used as the dense channel of the hybrid coarse stage.
        """
        if self._centroids.size == 0 or not query_vector:
            return []
        query = _l2_normalize(np.asarray([query_vector], dtype=np.float32))[0]
        sims = self._centroids @ query
        order = np.argsort(-sims)
        return [(self._ids[i], self._titles[self._ids[i]], float(sims[i])) for i in order]

    def top_sections(
        self, query_vector: list[float], n: int
    ) -> list[tuple[str, str, float]]:
        """Return the ``n`` most similar sections to the query embedding.

        Args:
            query_vector: The query's embedding.
            n: How many sections to return.

        Returns:
            ``(section_id, title, cosine_similarity)`` tuples, best first.
        """
        return self.rank_sections(query_vector)[: max(1, n)]


def load_section_router(path: str) -> SectionRouter | None:
    """Load the section index from disk and build a :class:`SectionRouter`.

    Returns:
        A ready router, or ``None`` if the index file is missing (callers then
        fall back to flat, non-hierarchical retrieval).
    """
    index_path = Path(path)
    if not index_path.exists():
        logger.warning("Section index '%s' not found; hierarchical retrieval off.", path)
        return None
    try:
        records = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read section index '%s': %s", path, exc)
        return None
    logger.info("Loaded section index (%d sections) from '%s'", len(records), path)
    return SectionRouter(records)

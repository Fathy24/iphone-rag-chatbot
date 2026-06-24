"""Vector store access.

Two interchangeable backends, both exposing the same cosine-similarity
retrieval interface (higher score = more relevant) so the rest of the pipeline
is backend-agnostic:

* "qdrant" — Qdrant Cloud (managed). Embeds the query with the same OpenAI
  model used at ingestion and runs a native-filtered cosine search.
* "faiss"  — local on-disk FAISS index (no signup/network); also used as the
  automatic fallback when Qdrant is unreachable at startup
  (``CLOUD_FALLBACK_TO_FAISS``).

Vectors are OpenAI ``text-embedding-3-large`` embeddings; per-vector metadata
carries the ``page``/``section``/``section_id``/``source`` citation fields.

This module is the single integration point for both ingestion and serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.config import Settings, get_settings
from app.llm.clients import build_embeddings
from app.logging_config import get_logger

if TYPE_CHECKING:  # heavy/optional deps are imported lazily inside functions
    from langchain_core.documents import Document

logger = get_logger(__name__)

# Metadata key under which the chunk text is stored alongside citation metadata.
TEXT_KEY = "text"


def _faiss_fallback(settings: Settings, why: str):
    """Fall back to the local FAISS index when Qdrant is unreachable.

    Honours ``CLOUD_FALLBACK_TO_FAISS``. Re-raises the original error if the
    fallback is disabled or no local index has been built.
    """
    if not settings.cloud_fallback_to_faiss:
        raise RuntimeError(why)
    try:
        store = load_local_store(settings)
    except Exception:  # noqa: BLE001 - surface the *cloud* failure, not this one
        raise RuntimeError(
            f"{why} No local FAISS fallback is available either "
            f"(expected an index at '{settings.faiss_index_path}'). "
            "Run ingestion with VECTOR_BACKEND=faiss to build a fallback."
        )
    logger.warning("%s Falling back to the local FAISS index.", why)
    return store


def get_vector_store(settings: Settings | None = None):
    """Return a serving-time vector store for the configured backend.

    Dispatches to Qdrant (cloud) or a local FAISS index depending on
    ``VECTOR_BACKEND``. The returned object always exposes
    ``similarity_search_with_score(query, k)`` returning ``(Document, score)``
    pairs where the score is a cosine similarity (higher = more relevant).
    Qdrant falls back to FAISS when it's unreachable at startup and
    ``CLOUD_FALLBACK_TO_FAISS`` is set and a local index exists.

    Raises:
        RuntimeError: If the index has not been populated (run ingestion first).
    """
    settings = settings or get_settings()
    if settings.vector_backend == "faiss":
        return load_local_store(settings)
    return get_dense_store(settings)


def get_dense_store(settings: Settings | None = None):
    """Return the fine-stage dense store with a uniform ``dense_search`` API.

    Every backend exposes ``dense_search(query, k, section_ids)`` returning
    ``(Document, cosine_similarity)`` pairs, optionally constrained to the given
    set of ``section_id``s (the fine stage of coarse-to-fine retrieval):

    * Qdrant applies a native ``section_id`` keyword filter.
    * FAISS over-fetches and post-filters in memory (no native filtering).

    Qdrant falls back to the local FAISS index when it can't be reached at
    startup and ``CLOUD_FALLBACK_TO_FAISS`` is on.
    """
    settings = settings or get_settings()
    if settings.vector_backend == "faiss":
        return load_local_store(settings)

    try:
        return QdrantDenseStore.connect(settings)
    except Exception as exc:  # noqa: BLE001
        return _faiss_fallback(settings, f"Qdrant is unavailable ({exc}).")


# --- Local FAISS backend (offline testing) -----------------------------------


def _section_of(doc: "Document") -> str:
    """Return a document's ``section_id`` metadata (empty string if absent)."""
    return str((doc.metadata or {}).get("section_id", ""))


class LocalVectorStore:
    """Thin adapter over a langchain FAISS index.

    Normalises FAISS' interface to match Qdrant's: ``similarity_search_with_score``
    returns cosine *similarity* (higher = better), so the retriever's cosine
    threshold gating behaves identically across backends.
    """

    def __init__(self, faiss_index) -> None:
        self._faiss = faiss_index

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple["Document", float]]:
        """Return ``(Document, cosine_similarity)`` pairs, best first.

        Under the COSINE strategy, langchain's FAISS returns the squared L2
        distance over L2-normalised vectors, i.e. ``d = 2 - 2*cos``. We convert
        back to a cosine similarity in ``[-1, 1]`` so the score is directly
        comparable to Qdrant's (and to ``SCORE_THRESHOLD``).
        """
        pairs = self._faiss.similarity_search_with_score(query, k=k)
        return [(doc, 1.0 - float(dist) / 2.0) for doc, dist in pairs]

    def dense_search(
        self, query: str, k: int, section_ids: set[str] | None = None
    ) -> list[tuple["Document", float]]:
        """Cosine search, optionally restricted to a set of sections.

        FAISS has no native metadata filtering, so when ``section_ids`` is
        given we over-fetch a larger pool and keep only chunks from those
        sections (exact, since the whole guide is small).
        """
        if not section_ids:
            return self.similarity_search_with_score(query, k=k)
        pool = self.similarity_search_with_score(query, k=max(k * 6, 60))
        filtered = [(doc, score) for doc, score in pool if _section_of(doc) in section_ids]
        return filtered[:k]


# --- Qdrant Cloud backend ----------------------------------------------------


def build_qdrant_client(settings: Settings | None = None):
    """Create a Qdrant client from settings.

    Raises:
        RuntimeError: If the client cannot be initialised.
    """
    settings = settings or get_settings()
    try:
        from qdrant_client import QdrantClient

        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not initialise Qdrant client: {exc}") from exc


def load_corpus_from_qdrant(
    settings: Settings | None = None,
) -> tuple[list[dict], list[list[float] | None]]:
    """Stream the full chunk corpus (payloads + dense vectors) from Qdrant.

    This is what makes the cloud collection the *single source of truth*: at
    serve time the retriever rebuilds the BM25 sparse index, the section
    centroids (coarse stage) and the parent-document expander **in memory from
    Qdrant**, so no ``.bm25_corpus.json`` / ``.sections.json`` files need to be
    bundled. One ``scroll`` pass over a small guide is effectively instant.

    Returns:
        ``(records, vectors)`` where ``records[i]`` is a chunk payload dict
        (carrying the citation metadata and the chunk ``text``) and
        ``vectors[i]`` is its dense embedding (or ``None`` if absent). On any
        failure returns ``([], [])`` so retrieval degrades gracefully to
        dense-only instead of failing the app.
    """
    settings = settings or get_settings()
    try:
        client = build_qdrant_client(settings)
        name = settings.qdrant_collection
        records: list[dict] = []
        vectors: list[list[float] | None] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=256,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            for point in points:
                vector = point.vector
                # Named-vector collections return a dict; pick the dense vector.
                if isinstance(vector, dict):
                    vector = vector.get("dense") or next(iter(vector.values()), None)
                records.append(dict(point.payload or {}))
                vectors.append(vector)
            if offset is None:
                break
        logger.info(
            "Loaded corpus from Qdrant (%d chunks) for hybrid + hierarchical retrieval.",
            len(records),
        )
        return records, vectors
    except Exception as exc:  # noqa: BLE001 - degrade to dense-only retrieval
        logger.warning(
            "Could not load corpus from Qdrant (%s); hybrid/hierarchical "
            "disabled this session, falling back to dense-only.",
            exc,
        )
        return [], []


def ensure_qdrant_collection(
    client,
    settings: Settings,
    *,
    recreate: bool = False,
) -> None:
    """Ensure a cosine collection of the right dimension exists (ingestion).

    Also creates a keyword payload index on ``section_id`` so the fine stage's
    section filter is served natively.
    """
    from qdrant_client import models

    name = settings.qdrant_collection
    exists = client.collection_exists(name)
    if exists and recreate:
        logger.warning("Deleting existing Qdrant collection '%s'", name)
        client.delete_collection(name)
        exists = False

    if not exists:
        logger.info(
            "Creating Qdrant collection '%s' (dim=%d, cosine)",
            name,
            settings.embedding_dim,
        )
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        client.create_payload_index(
            collection_name=name,
            field_name="section_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def upsert_qdrant_embeddings(
    client,
    texts: list[str],
    vectors: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
    settings: Settings | None = None,
    *,
    batch_size: int = 128,
) -> None:
    """Upsert precomputed embeddings (+ payload) into the Qdrant collection.

    The chunk text is stored in the payload under :data:`TEXT_KEY` alongside the
    citation metadata, mirroring how the other backends round-trip a Document.
    """
    from qdrant_client import models

    settings = settings or get_settings()
    name = settings.qdrant_collection
    total = len(ids)
    for start in range(0, total, batch_size):
        sl = slice(start, start + batch_size)
        points = [
            models.PointStruct(
                id=pid,
                vector=vec,
                payload={**(meta or {}), TEXT_KEY: text},
            )
            for pid, vec, meta, text in zip(
                ids[sl], vectors[sl], metadatas[sl], texts[sl]
            )
        ]
        client.upsert(collection_name=name, points=points)
        logger.info("Upserted %d/%d chunks", min(start + len(points), total), total)


class QdrantDenseStore:
    """Adapter exposing the uniform dense-search API over Qdrant Cloud.

    Embeds the query with the same OpenAI model used at ingestion, then runs a
    cosine search (optionally filtered to a set of ``section_id``s for the fine
    stage). Returns ``(Document, cosine_similarity)`` pairs so it is a drop-in
    replacement for the local FAISS store.
    """

    def __init__(self, client, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._embeddings = build_embeddings(settings)

    @classmethod
    def connect(cls, settings: Settings) -> "QdrantDenseStore":
        """Build the store and verify the collection is reachable.

        Raises:
            RuntimeError: If the cluster is unreachable or the collection is
                missing (so the caller can fall back to FAISS).
        """
        client = build_qdrant_client(settings)
        try:
            if not client.collection_exists(settings.qdrant_collection):
                raise RuntimeError(
                    f"Qdrant collection '{settings.qdrant_collection}' does not "
                    "exist. Run ingestion with VECTOR_BACKEND=qdrant first."
                )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - network / auth failure
            raise RuntimeError(f"could not reach Qdrant: {exc}") from exc
        logger.info(
            "Qdrant store ready (collection='%s').", settings.qdrant_collection
        )
        return cls(client, settings)

    def _search(self, query: str, k: int, flt) -> list[tuple["Document", float]]:
        from langchain_core.documents import Document

        vector = self._embeddings.embed_query(query)
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=k,
            query_filter=flt,
            with_payload=True,
        )
        pairs: list[tuple[Document, float]] = []
        for point in result.points:
            payload = point.payload or {}
            text = payload.get(TEXT_KEY, "")
            meta = {key: val for key, val in payload.items() if key != TEXT_KEY}
            pairs.append((Document(page_content=text, metadata=meta), float(point.score)))
        return pairs

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple["Document", float]]:
        return self._search(query, k, None)

    def dense_search(
        self, query: str, k: int, section_ids: set[str] | None = None
    ) -> list[tuple["Document", float]]:
        """Cosine search with a native ``section_id`` keyword filter."""
        flt = None
        if section_ids:
            from qdrant_client import models

            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="section_id",
                        match=models.MatchAny(any=list(section_ids)),
                    )
                ]
            )
        return self._search(query, k, flt)


def _build_faiss(documents: list["Document"], settings: Settings):
    """Create a cosine-similarity FAISS index from documents."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    return FAISS.from_documents(
        documents,
        embedding=build_embeddings(settings),
        distance_strategy=DistanceStrategy.COSINE,
    )


def build_local_store_from_embeddings(
    texts: list[str],
    vectors: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
    settings: Settings | None = None,
) -> None:
    """Build a local FAISS index from precomputed embeddings and persist it.

    Used by ingestion so chunks are embedded exactly once (the same vectors
    feed both the FAISS index and the section centroids), avoiding a redundant
    embedding pass.

    Raises:
        RuntimeError: If the index cannot be built or saved.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    settings = settings or get_settings()
    try:
        path = Path(settings.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, vectors)),
            embedding=build_embeddings(settings),
            metadatas=metadatas,
            ids=ids,
            distance_strategy=DistanceStrategy.COSINE,
        )
        index.save_local(str(path))
        logger.info("Saved local FAISS index (%d vectors) to '%s'", len(ids), path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not build local FAISS index: {exc}") from exc


def build_local_store_from_documents(
    documents: list["Document"],
    ids: list[str],
    settings: Settings | None = None,
) -> None:
    """Build a local FAISS index from documents and persist it to disk.

    Used by the ingestion script when ``VECTOR_BACKEND=faiss``.

    Raises:
        RuntimeError: If the index cannot be built or saved.
    """
    settings = settings or get_settings()
    try:
        path = Path(settings.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        index = _build_faiss(documents, settings)
        index.save_local(str(path))
        logger.info(
            "Saved local FAISS index (%d vectors) to '%s'", len(ids), path
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not build local FAISS index: {exc}") from exc


def load_local_store(settings: Settings | None = None) -> LocalVectorStore:
    """Load the persisted local FAISS index for serving.

    Raises:
        RuntimeError: If the index directory is missing (run ingestion first).
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    settings = settings or get_settings()
    path = Path(settings.faiss_index_path)
    if not (path / "index.faiss").exists():
        raise RuntimeError(
            f"Local FAISS index not found at '{path}'. Run the ingestion script "
            "with VECTOR_BACKEND=faiss to build it first."
        )
    try:
        index = FAISS.load_local(
            str(path),
            build_embeddings(settings),
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load local FAISS index: {exc}") from exc
    return LocalVectorStore(index)

"""One-shot ingestion: load the PDF, chunk it, embed, and upsert to Qdrant.

This script is run **once before submission** to populate the Qdrant Cloud
collection (or a local FAISS index when ``VECTOR_BACKEND=faiss``). The reviewer
does NOT run it — the live chatbot queries the already-populated collection. It
is written to be idempotent: re-running with ``--recreate`` rebuilds the
collection cleanly, and chunk IDs are deterministic so a normal re-run upserts
in place rather than duplicating.

Usage::

    python -m ingest.ingest                 # create-if-missing, then upsert
    python -m ingest.ingest --recreate      # drop and rebuild the collection
    python -m ingest.ingest --pdf data/iphone_user_guide.pdf --batch-size 128
"""

from __future__ import annotations

import argparse
import sys
import uuid

from app.config import get_settings
from app.llm.clients import build_embeddings
from app.logging_config import configure_logging, get_logger
from app.rag.bm25 import save_corpus
from app.rag.chunking import build_chunks
from app.rag.hierarchy import build_section_records, save_section_index
from app.rag.loader import load_pdf_pages
from app.rag.sections import build_section_index
from app.rag.vector_store import (
    build_local_store_from_embeddings,
    build_qdrant_client,
    ensure_qdrant_collection,
    upsert_qdrant_embeddings,
)

logger = get_logger(__name__)

# Stable namespace so the same chunk always maps to the same vector point id.
_ID_NAMESPACE = uuid.UUID("7c2a1e2e-9f1a-4c5e-9b3d-1a2b3c4d5e6f")


def _deterministic_id(chunk_id: str) -> str:
    """Derive a stable UUID point id from a chunk's logical id."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant / FAISS.")
    parser.add_argument("--pdf", default=settings.pdf_path, help="Path to the source PDF.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of chunks to embed/upsert per batch.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """Execute the full ingestion pipeline.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args(argv)

    # Validate the secrets needed for ingestion up front. Qdrant credentials
    # are only required for the cloud backend.
    required = [("OPENAI_API_KEY", settings.openai_api_key)]
    if settings.vector_backend == "qdrant":
        required.append(("QDRANT_URL", settings.qdrant_url))
        required.append(("QDRANT_API_KEY", settings.qdrant_api_key))
    missing = [name for name, value in required if not value]
    if missing:
        logger.error("Missing required env vars for ingestion: %s", ", ".join(missing))
        return 2

    try:
        logger.info("Loading PDF: %s", args.pdf)
        pages = load_pdf_pages(args.pdf)
        section_index = build_section_index(args.pdf)

        # The semantic strategy needs an embedding function; reuse the same model
        # configured for storage so chunking and retrieval share a space.
        embeddings = build_embeddings(settings)
        chunks = build_chunks(
            pages,
            section_index,
            settings,
            embed_fn=embeddings.embed_documents,
        )
        if not chunks:
            logger.error("No chunks produced; aborting.")
            return 1

        ids = [_deterministic_id(c.metadata["chunk_id"]) for c in chunks]
        total = len(chunks)

        # Embed every chunk exactly once. The same vectors feed both the dense
        # index and the section centroids (coarse retrieval level), so there is
        # no redundant embedding pass.
        logger.info("Embedding %d chunks...", total)
        texts = [c.page_content for c in chunks]
        vectors = embeddings.embed_documents(texts)

        # Build and persist the section index (centroids) for coarse-to-fine
        # retrieval. Sections are grouped by the chunks' ``section_id``.
        section_records = build_section_records(chunks, vectors)
        save_section_index(section_records, settings.sections_path)

        # Persist the BM25 corpus (chunk text + citation metadata) so the sparse
        # half of hybrid retrieval can be rebuilt at serve time, independent of
        # which dense backend (Pinecone/FAISS) stores the vectors.
        corpus = [
            {
                "chunk_id": c.metadata["chunk_id"],
                "text": c.page_content,
                "page": int(c.metadata.get("page", 0)),
                "section": str(c.metadata.get("section", "General")),
                "section_id": str(c.metadata.get("section_id", "general")),
                "chunk_index": int(c.metadata.get("chunk_index", 0)),
                "source": str(c.metadata.get("source", "document")),
            }
            for c in chunks
        ]
        save_corpus(corpus, settings.bm25_corpus_path)

        metadatas = [c.metadata for c in chunks]

        if settings.vector_backend == "faiss":
            logger.info("Building local FAISS index (%d chunks)...", total)
            build_local_store_from_embeddings(texts, vectors, metadatas, ids, settings)
            logger.info(
                "Ingestion complete: %d chunks, %d sections in local index '%s'.",
                total,
                len(section_records),
                settings.faiss_index_path,
            )
            return 0

        logger.info("Upserting %d chunks into Qdrant...", total)
        client = build_qdrant_client(settings)
        ensure_qdrant_collection(client, settings, recreate=args.recreate)
        upsert_qdrant_embeddings(
            client, texts, vectors, metadatas, ids, settings,
            batch_size=args.batch_size,
        )
        logger.info(
            "Ingestion complete: %d chunks, %d sections in Qdrant collection '%s'.",
            total,
            len(section_records),
            settings.qdrant_collection,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(run())

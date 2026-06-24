"""Chunking strategy for the iPhone User Guide.

Design rationale (be ready to defend this in the interview):

* **Token-based, recursive splitting.** We split with a recursive character
  splitter calibrated by a *tiktoken* encoder, so chunk sizes are measured in
  the same tokens the embedding/chat models use — not characters. This keeps
  every chunk safely within the embedding context and makes cost predictable.
* **~512 tokens, ~80 token overlap (~15%).** A user manual is written in short,
  self-contained task sections ("Set up Wi-Fi", "Use AirDrop"). ~512 tokens is
  large enough to hold a complete procedure with its heading, yet small enough
  to keep retrieval precise and citations specific to a page. The overlap
  preserves continuity for steps that straddle a chunk boundary.
* **Page-preserving.** We split *per page* so a chunk never spans two pages and
  its ``page`` citation is always exact. Section titles are attached from the
  section index.
* **Separator hierarchy.** We prefer to break on paragraph, then line, then
  sentence, then word boundaries to avoid cutting mid-sentence.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.logging_config import get_logger
from app.rag.sections import section_slug
from app.rag.semantic_chunker import SemanticChunker

logger = get_logger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


def build_chunks(
    pages: list[Document],
    section_index: dict[int, str],
    settings: Settings,
    embed_fn: EmbedFn | None = None,
) -> list[Document]:
    """Chunk pages using the configured strategy.

    Dispatches to the semantic chunker when ``CHUNK_STRATEGY=semantic`` and an
    embedding function is available; otherwise uses token-based splitting. Both
    strategies operate per page so page citations remain exact.

    Args:
        pages: One document per page.
        section_index: ``{page: section_title}`` for citations.
        settings: Application settings.
        embed_fn: Embedding callable required by the semantic strategy.

    Returns:
        Chunk documents with citation metadata.
    """
    if settings.chunk_strategy == "semantic" and embed_fn is not None:
        return chunk_pages_semantic(
            pages,
            section_index,
            embed_fn,
            window_size=settings.semantic_window_size,
            threshold_percentile=settings.semantic_threshold_percentile,
            min_tokens=settings.semantic_min_tokens,
            max_tokens=settings.semantic_max_tokens,
        )
    if settings.chunk_strategy == "semantic":
        logger.warning("Semantic strategy requested but no embed_fn; using tokens.")
    return chunk_pages(
        pages,
        section_index,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


def _to_chunk_documents(
    page_doc: Document, pieces: list[str], section: str
) -> list[Document]:
    """Wrap split text pieces into Documents with citation metadata."""
    page_no = int(page_doc.metadata.get("page", 0))
    source = page_doc.metadata.get("source", "document")
    out: list[Document] = []
    for local_idx, piece in enumerate(pieces):
        text = piece.strip()
        if not text:
            continue
        out.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_no,
                    "section": section,
                    "section_id": section_slug(section),
                    "chunk_id": f"{source}:p{page_no}:c{local_idx}",
                    "chunk_index": local_idx,
                },
            )
        )
    return out


def chunk_pages_semantic(
    pages: list[Document],
    section_index: dict[int, str],
    embed_fn: EmbedFn,
    *,
    window_size: int = 2,
    threshold_percentile: int = 30,
    min_tokens: int = 120,
    max_tokens: int = 512,
) -> list[Document]:
    """Chunk each page semantically (embedding-similarity splits).

    See :class:`app.rag.semantic_chunker.SemanticChunker` for the algorithm.
    """
    chunker = SemanticChunker(
        embed_fn,
        window_size=window_size,
        threshold_percentile=threshold_percentile,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    chunks: list[Document] = []
    for page_doc in pages:
        page_no = int(page_doc.metadata.get("page", 0))
        section = section_index.get(page_no, "General")
        pieces = chunker.split_text(page_doc.page_content)
        chunks.extend(_to_chunk_documents(page_doc, pieces, section))

    logger.info(
        "Produced %d semantic chunks from %d pages (window=%d, pct=%d, %d-%d tok)",
        len(chunks),
        len(pages),
        window_size,
        threshold_percentile,
        min_tokens,
        max_tokens,
    )
    return chunks


def chunk_pages(
    pages: list[Document],
    section_index: dict[int, str],
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
    encoding_name: str = "cl100k_base",
) -> list[Document]:
    """Split page documents into retrieval chunks with rich citation metadata.

    Args:
        pages: One document per page (from :func:`app.rag.loader.load_pdf_pages`).
        section_index: ``{page: section_title}`` map for citations.
        chunk_size_tokens: Target chunk size in tokens.
        chunk_overlap_tokens: Overlap between consecutive chunks, in tokens.
        encoding_name: tiktoken encoding used to measure token lengths.

    Returns:
        A list of chunk ``Document`` objects. Each carries metadata:
        ``source``, ``page``, ``section``, ``chunk_id``, and ``chunk_index``.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )

    chunks: list[Document] = []
    for page_doc in pages:
        page_no = int(page_doc.metadata.get("page", 0))
        section = section_index.get(page_no, "General")
        source = page_doc.metadata.get("source", "document")

        for local_idx, piece in enumerate(splitter.split_text(page_doc.page_content)):
            text = piece.strip()
            if not text:
                continue
            chunk_id = f"{source}:p{page_no}:c{local_idx}"
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "page": page_no,
                        "section": section,
                        "section_id": section_slug(section),
                        "chunk_id": chunk_id,
                        "chunk_index": local_idx,
                    },
                )
            )

    logger.info(
        "Produced %d chunks from %d pages (size=%d, overlap=%d tokens)",
        len(chunks),
        len(pages),
        chunk_size_tokens,
        chunk_overlap_tokens,
    )
    return chunks

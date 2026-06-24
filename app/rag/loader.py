"""PDF loading with page-accurate metadata.

We use PyMuPDF (``fitz``) because it is fast, dependency-light, and — crucially
for this assessment — gives us the exact 1-based page index for every block of
text. Accurate page numbers are mandatory: the chatbot must cite the page where
each answer was found.
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF
from langchain_core.documents import Document

from app.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf_pages(pdf_path: str) -> list[Document]:
    """Load a PDF into one :class:`Document` per page.

    Args:
        pdf_path: Filesystem path to the source PDF.

    Returns:
        A list of LangChain ``Document`` objects, one per non-empty page, each
        carrying ``{"source", "page", "total_pages"}`` metadata. ``page`` is the
        1-based physical page index.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    source_name = os.path.basename(pdf_path)
    documents: list[Document] = []

    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            logger.info("Loading '%s' (%d pages)", source_name, total_pages)

            for index in range(total_pages):
                page = doc.load_page(index)
                # "text" preserves reading order; good enough for a user guide.
                raw_text = page.get_text("text") or ""
                cleaned = _normalise_whitespace(raw_text)
                if not cleaned:
                    continue

                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source": source_name,
                            "page": index + 1,
                            "total_pages": total_pages,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001 - surface a clear ingestion error
        raise RuntimeError(f"Failed to parse PDF '{pdf_path}': {exc}") from exc

    if not documents:
        raise RuntimeError(f"No extractable text found in PDF '{pdf_path}'.")

    logger.info("Extracted text from %d non-empty pages", len(documents))
    return documents


def _normalise_whitespace(text: str) -> str:
    """Collapse excessive whitespace while keeping paragraph boundaries.

    PDF extraction often yields ragged spacing and stray blank lines. We trim
    each line and drop runs of blank lines so downstream chunking is stable.
    """
    lines = [line.strip() for line in text.splitlines()]
    out_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            out_lines.append(line)
            blank_run = 0
        else:
            blank_run += 1
            if blank_run <= 1:
                out_lines.append("")
    return "\n".join(out_lines).strip()

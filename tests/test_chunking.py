"""Tests for the chunking pipeline and citation metadata."""

from __future__ import annotations

from langchain_core.documents import Document

from app.rag.chunking import chunk_pages


def _make_pages() -> list[Document]:
    long_text = (
        "Set up Wi-Fi. Go to Settings then Wi-Fi. Choose a network and enter "
        "the password. " * 40
    )
    return [
        Document(page_content=long_text, metadata={"source": "guide.pdf", "page": 15}),
        Document(page_content="Short page about Siri.", metadata={"source": "guide.pdf", "page": 41}),
    ]


def test_chunks_carry_citation_metadata() -> None:
    section_index = {15: "Chapter 2: Getting Started", 41: "Chapter 4: Siri"}
    chunks = chunk_pages(
        _make_pages(),
        section_index,
        chunk_size_tokens=64,
        chunk_overlap_tokens=10,
    )

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.metadata["source"] == "guide.pdf"
        assert chunk.metadata["page"] in (15, 41)
        assert chunk.metadata["section"] in section_index.values()
        assert chunk.metadata["chunk_id"].startswith("guide.pdf:p")


def test_long_page_is_split_into_multiple_chunks() -> None:
    section_index = {15: "Chapter 2: Getting Started"}
    pages = [_make_pages()[0]]
    chunks = chunk_pages(pages, section_index, chunk_size_tokens=64, chunk_overlap_tokens=10)
    assert len(chunks) > 1


def test_missing_section_defaults_to_general() -> None:
    chunks = chunk_pages(
        _make_pages(), section_index={}, chunk_size_tokens=64, chunk_overlap_tokens=10
    )
    assert all(c.metadata["section"] == "General" for c in chunks)

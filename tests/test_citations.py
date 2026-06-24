"""Tests for citation enforcement and context formatting."""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.nodes import _ensure_citations
from app.graph.prompts import NOT_FOUND_MESSAGE, format_context


@dataclass
class FakeChunk:
    text: str
    page: int
    section: str


def test_sources_line_is_appended_when_missing() -> None:
    chunks = [
        FakeChunk("Go to Settings > Wi-Fi.", 15, "Chapter 2: Getting Started"),
        FakeChunk("Tap a network.", 16, "Chapter 2: Getting Started"),
    ]
    answer = "Open Settings, tap Wi-Fi, then choose your network."
    enforced = _ensure_citations(answer, chunks)
    assert "Sources:" in enforced
    assert "p. 15 — Chapter 2: Getting Started" in enforced


def test_existing_sources_line_is_preserved() -> None:
    chunks = [FakeChunk("...", 41, "Chapter 4: Siri")]
    answer = "Hold the Home button.\n\nSources: p. 41 — Chapter 4: Siri"
    assert _ensure_citations(answer, chunks) == answer


def test_not_found_message_is_not_decorated() -> None:
    chunks = [FakeChunk("irrelevant", 1, "General")]
    assert _ensure_citations(NOT_FOUND_MESSAGE, chunks) == NOT_FOUND_MESSAGE


def test_format_context_tags_each_passage_with_citation() -> None:
    chunks = [
        FakeChunk("Connect to Wi-Fi here.", 15, "Chapter 2: Getting Started"),
        FakeChunk("Use Siri like this.", 41, "Chapter 4: Siri"),
    ]
    rendered = format_context(chunks)
    assert "p. 15 — Chapter 2: Getting Started" in rendered
    assert "p. 41 — Chapter 4: Siri" in rendered
    assert "Passage 1" in rendered and "Passage 2" in rendered


def test_format_context_handles_empty() -> None:
    assert "no relevant passages" in format_context([])

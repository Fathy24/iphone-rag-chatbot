"""Tests for the BM25 sparse index."""

from __future__ import annotations

from app.rag.bm25 import SparseIndex


def _records() -> list[dict]:
    return [
        {"chunk_id": "p1", "text": "Take a screenshot by pressing Sleep/Wake and Home.", "page": 1, "section": "Camera", "source": "g"},
        {"chunk_id": "p2", "text": "The weather app shows a six-day forecast.", "page": 2, "section": "Weather", "source": "g"},
        {"chunk_id": "p3", "text": "Use AirDrop to share photos with nearby devices.", "page": 3, "section": "Sharing", "source": "g"},
    ]


def test_bm25_ranks_lexical_match_first() -> None:
    index = SparseIndex(_records())
    results = index.query("how do I take a screenshot", top_k=3)
    assert results, "expected at least one match"
    assert results[0][0]["chunk_id"] == "p1"


def test_bm25_returns_nothing_for_no_overlap() -> None:
    index = SparseIndex(_records())
    assert index.query("quantum chromodynamics", top_k=3) == []


def test_bm25_empty_corpus_is_safe() -> None:
    index = SparseIndex([])
    assert index.size == 0
    assert index.query("anything", top_k=5) == []

"""Offline tests for hierarchical retrieval, citations, memory windowing, and
section-scoped parent expansion (no network)."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.nodes import (
    _clean_queries,
    _ensure_citations,
    _is_refusal,
    _merge_chunks,
    _run_parallel,
    _tag_origin,
    _window_messages,
)
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.hierarchy import SectionRouter, build_section_records
from app.rag.parent import ParentExpander
from app.rag.sections import section_slug


@dataclass
class _FakeChunk:
    page: int
    section: str
    text: str
    score: float = 0.0


# --- section ids -------------------------------------------------------------


def test_section_slug_is_stable_and_safe() -> None:
    assert section_slug("Chapter 4: Siri") == "chapter-4-siri"
    assert section_slug("  Appendix A — Accessibility ") == "appendix-a-accessibility"
    assert section_slug("") == "general"


# --- coarse stage: section centroids + router --------------------------------


def _doc(section_id: str, section: str, page: int) -> Document:
    return Document(
        page_content="x",
        metadata={"section_id": section_id, "section": section, "page": page},
    )


def test_build_section_records_groups_and_normalises() -> None:
    chunks = [
        _doc("a", "A", 1),
        _doc("a", "A", 2),
        _doc("b", "B", 3),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    records = build_section_records(chunks, vectors)
    by_id = {r["section_id"]: r for r in records}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"]["n_chunks"] == 2
    # Centroid of two identical unit vectors is the same unit vector.
    assert round(by_id["a"]["centroid"][0], 5) == 1.0


def test_section_router_ranks_by_similarity() -> None:
    records = [
        {"section_id": "a", "title": "A", "centroid": [1.0, 0.0]},
        {"section_id": "b", "title": "B", "centroid": [0.0, 1.0]},
    ]
    router = SectionRouter(records)
    top = router.top_sections([0.9, 0.1], n=1)
    assert top and top[0][0] == "a"


def test_section_router_empty_is_safe() -> None:
    assert SectionRouter([]).top_sections([1.0, 0.0], n=3) == []


def test_section_router_rank_returns_all_sorted() -> None:
    records = [
        {"section_id": "a", "title": "A", "centroid": [1.0, 0.0]},
        {"section_id": "b", "title": "B", "centroid": [0.0, 1.0]},
        {"section_id": "c", "title": "C", "centroid": [0.7, 0.7]},
    ]
    router = SectionRouter(records)
    ranked = router.rank_sections([1.0, 0.0])
    assert [r[0] for r in ranked] == ["a", "c", "b"]
    assert router.title_of("b") == "B"


# --- coarse stage: hybrid RRF fusion of dense + sparse section ranks ----------


def test_coarse_hybrid_promotes_keyword_section() -> None:
    # Dense ranks "a" first; sparse (BM25) ranks "b" first. Equal-weight RRF
    # should fuse them so the keyword-strong section "b" is competitive.
    dense_ranks = {"a": 1, "b": 2, "c": 3}
    sparse_ranks = {"b": 1, "a": 2, "c": 3}
    fused = reciprocal_rank_fusion(dense_ranks, sparse_ranks, k_constant=60)
    order = [sid for sid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
    # a and b tie at the top (symmetric), c stays last; the keyword section is lifted.
    assert set(order[:2]) == {"a", "b"} and order[-1] == "c"


# --- section-scoped parent expansion -----------------------------------------


def _records() -> list[dict]:
    # Same section spanning two pages; ordered by (page, within-page index).
    return [
        {"chunk_id": "g:p10:c0", "text": "First.", "page": 10, "section_id": "s", "chunk_index": 0},
        {"chunk_id": "g:p10:c1", "text": "Second.", "page": 10, "section_id": "s", "chunk_index": 1},
        {"chunk_id": "g:p11:c0", "text": "Third.", "page": 11, "section_id": "s", "chunk_index": 0},
    ]


def test_parent_expands_across_pages_within_section() -> None:
    exp = ParentExpander(_records())
    out = exp.expand("g:p10:c1", "Second.", window=1)
    assert "First." in out and "Third." in out


def test_parent_window_zero_is_noop() -> None:
    exp = ParentExpander(_records())
    assert exp.expand("g:p10:c1", "Second.", window=0) == "Second."


def test_parent_unknown_chunk_falls_back() -> None:
    exp = ParentExpander(_records())
    assert exp.expand("g:p99:c0", "fallback", window=2) == "fallback"


# --- agent helpers -----------------------------------------------------------


def test_tag_origin_keeps_first_query() -> None:
    a = _FakeChunk(80, "Camera", "Press the buttons.")
    _tag_origin([a], "take a screenshot")
    _tag_origin([a], "something else")  # already tagged -> first query wins
    assert a.origin_query == "take a screenshot"


def test_merge_chunks_dedupes_across_tool_calls() -> None:
    shared = _FakeChunk(80, "Camera", "Press the buttons to take a screenshot.")
    other = _FakeChunk(33, "Basics", "Turn on AirDrop in Control Center.")
    merged = _merge_chunks([shared], [other, shared])
    assert len(merged) == 2


def test_ensure_citations_appends_sources() -> None:
    chunks = [_FakeChunk(80, "Chapter 5: Camera", "...")]
    out = _ensure_citations("Press the buttons.", chunks)
    assert "Sources:" in out and "p. 80" in out


def test_ensure_citations_skips_refusal() -> None:
    chunks = [_FakeChunk(80, "Camera", "...")]
    out = _ensure_citations("I couldn't find that in the iPhone User Guide.", chunks)
    assert "Sources:" not in out


def test_refusal_detection() -> None:
    assert _is_refusal("I couldn't find that in the iPhone User Guide.")
    assert not _is_refusal("Press the Sleep/Wake button. Sources: p. 80 — Camera")


# --- parallel multi-query tool -----------------------------------------------


def test_clean_queries_dedupes_caps_and_coerces() -> None:
    assert _clean_queries(["a", "a", " b ", "", "c"], cap=2) == ["a", "b"]
    # A single string is wrapped into a one-element list.
    assert _clean_queries("just one", cap=5) == ["just one"]
    assert _clean_queries(None, cap=5) == []


@dataclass
class _FakeInfo:
    sections: list = None  # type: ignore[assignment]
    hierarchical: bool = True
    coarse_hybrid: bool = True
    fell_back: bool = False

    def __post_init__(self) -> None:
        if self.sections is None:
            self.sections = []


class _FakeSettings:
    retrieval_max_workers = 4
    max_parallel_queries = 6


def test_run_parallel_groups_results_and_dedupes_chunks() -> None:
    shot = _FakeChunk(28, "Camera", "Press the buttons to take a screenshot.")
    hotspot = _FakeChunk(33, "Basics", "Turn on Personal Hotspot in Settings.")

    def fake_search(query, top_k, use_reranker, coarse_n):
        if "screenshot" in query:
            return [shot], _FakeInfo()
        if "hotspot" in query:
            return [hotspot, shot], _FakeInfo()  # shot repeats -> must dedupe
        return [], _FakeInfo()

    content, chunks, runs = _run_parallel(
        fake_search,
        ["take a screenshot", "set up a hotspot"],
        None, None, None, [], _FakeSettings(),
    )
    assert "Sub-query 1:" in content and "Sub-query 2:" in content
    assert len(chunks) == 2  # shot deduped across sub-queries
    assert len(runs) == 2 and all(r["batch"] for r in runs)


def test_run_parallel_empty_queries_is_safe() -> None:
    content, chunks, runs = _run_parallel(
        lambda *a: ([], None), [], None, None, None, [], _FakeSettings()
    )
    assert "NO RESULTS" in content and runs == []


def test_window_messages_starts_at_user_boundary() -> None:
    msgs = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "search_guide", "args": {"query": "x"}, "id": "1"}]),
        ToolMessage(content="passages", tool_call_id="1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]
    windowed = _window_messages(msgs, max_turns=1)
    # Keeps only the last user turn, starting exactly at the HumanMessage.
    assert isinstance(windowed[0], HumanMessage) and windowed[0].content == "q2"
    assert len(windowed) == 2

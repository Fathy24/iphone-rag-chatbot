"""Tests for the percentile-based semantic chunker (offline, stub embeddings)."""

from __future__ import annotations

from app.rag.semantic_chunker import SemanticChunker


def _topic_embed(texts: list[str]) -> list[list[float]]:
    """Stub embedder: Wi-Fi windows point one way, Siri windows the other."""
    vectors = []
    for t in texts:
        low = t.lower()
        if "wi-fi" in low or "wifi" in low or "network" in low:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


def test_splits_at_topic_boundary() -> None:
    text = (
        "Open Settings to begin. Tap Wi-Fi in the list. Choose your network name. "
        "Enter the network password. Now ask Siri a question. Hold the Home button. "
        "Speak your request to Siri. Siri responds aloud."
    )
    chunker = SemanticChunker(
        _topic_embed,
        window_size=2,
        threshold_percentile=40,
        min_tokens=5,   # keep the two topics as separate chunks
        max_tokens=512,
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    joined = " ".join(chunks).lower()
    assert "wi-fi" in joined and "siri" in joined


def test_short_text_is_single_chunk() -> None:
    chunker = SemanticChunker(_topic_embed, window_size=2)
    assert chunker.split_text("Just one short sentence.") == ["Just one short sentence."]


def test_empty_text_yields_nothing() -> None:
    chunker = SemanticChunker(_topic_embed)
    assert chunker.split_text("   ") == []


def test_embedding_failure_falls_back_to_tokens() -> None:
    def boom(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service down")

    text = "Sentence one here. Sentence two here. Sentence three here. Sentence four."
    chunker = SemanticChunker(boom, window_size=2, min_tokens=5, max_tokens=512)
    chunks = chunker.split_text(text)
    # Fallback returns the text (token-limited), never raising or losing content.
    assert chunks and "Sentence one" in chunks[0]

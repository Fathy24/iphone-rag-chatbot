"""Percentile-based semantic chunker.

Ported and trimmed from the production backend's ``EnglishSemanticChunker``
(``AgentsRag/semantic_final.py``). Instead of splitting on a fixed token count,
we split where the *meaning* shifts: we embed sliding windows of sentences,
measure cosine similarity between consecutive windows, and cut where similarity
drops into the bottom ``threshold_percentile`` of the document. Segments are
then merged/limited to respect token bounds.

Why this matters for the iPhone User Guide: a page often bundles several short
procedures ("Connect to Wi-Fi", "Forget a network"). Semantic splitting keeps
each procedure together while avoiding mega-chunks, which improves both
retrieval precision and the specificity of citations.

The chunker is embedding-provider agnostic: it takes an ``embed_fn`` callable so
it can be unit-tested offline with a stub.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import tiktoken

from app.logging_config import get_logger

logger = get_logger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

# Sentence splitter: break after ., !, ? followed by whitespace. Good enough for
# instructional prose; avoids pulling in a heavy NLP dependency.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Split text into semantically coherent chunks using embedding similarity."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        *,
        window_size: int = 2,
        threshold_percentile: int = 30,
        min_tokens: int = 120,
        max_tokens: int = 512,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialise the chunker.

        Args:
            embed_fn: Function mapping a list of texts to a list of vectors.
            window_size: Number of sentences per similarity window.
            threshold_percentile: Split where similarity is in this bottom
                percentile (lower => fewer, larger chunks).
            min_tokens: Merge adjacent segments until they reach this size.
            max_tokens: Hard cap; oversized segments are split on token count.
            encoding_name: tiktoken encoding used for token counting.
        """
        self._embed = embed_fn
        self._window = max(1, window_size)
        self._percentile = max(5, min(95, threshold_percentile))
        self._min_tokens = max(20, min_tokens)
        self._max_tokens = max(self._min_tokens + 50, max_tokens)
        self._encoder = tiktoken.get_encoding(encoding_name)

    def split_text(self, text: str) -> list[str]:
        """Split a single block of text into semantic chunks.

        Falls back to a single chunk for short inputs, and to token-based
        splitting if anything goes wrong, so ingestion never loses content.
        """
        text = (text or "").strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= self._window:
            return self._enforce_max(text)

        try:
            windows = self._build_windows(sentences)
            embeddings = [np.asarray(v, dtype=np.float32) for v in self._embed(windows)]
            split_after = self._find_split_indices(sentences, embeddings)
            segments = self._segments_from_splits(sentences, split_after)
            return self._merge_and_limit(segments)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic split failed (%s); falling back to tokens.", exc)
            return self._enforce_max(text)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    def _build_windows(self, sentences: list[str]) -> list[str]:
        return [
            " ".join(sentences[i : i + self._window])
            for i in range(len(sentences) - self._window + 1)
        ]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _find_split_indices(
        self, sentences: list[str], embeddings: list[np.ndarray]
    ) -> list[int]:
        """Return sentence indices after which a split should occur."""
        if len(embeddings) < 2:
            return []
        sims: list[tuple[int, float]] = []
        for i in range(len(embeddings) - 1):
            split_idx = i + self._window - 1
            if split_idx < len(sentences) - 1:
                sims.append((split_idx, self._cosine(embeddings[i], embeddings[i + 1])))
        if not sims:
            return []
        threshold = float(np.percentile([s for _, s in sims], self._percentile))
        return [idx for idx, sim in sims if sim <= threshold]

    @staticmethod
    def _segments_from_splits(sentences: list[str], split_after: list[int]) -> list[str]:
        """Group sentences into segments at the chosen split points."""
        cut = set(split_after)
        segments: list[str] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in cut:
                segments.append(" ".join(current))
                current = []
        if current:
            segments.append(" ".join(current))
        return segments

    def _merge_and_limit(self, segments: list[str]) -> list[str]:
        """Merge sub-minimum segments and hard-split over-maximum ones."""
        merged: list[str] = []
        buffer = ""
        for seg in segments:
            candidate = f"{buffer} {seg}".strip() if buffer else seg
            if self._count_tokens(candidate) < self._min_tokens:
                buffer = candidate
                continue
            buffer = ""
            merged.extend(self._enforce_max(candidate))
        if buffer:
            merged.extend(self._enforce_max(buffer))
        return [m for m in merged if m.strip()]

    def _enforce_max(self, text: str) -> list[str]:
        """Split ``text`` into <= max_tokens pieces on token boundaries."""
        tokens = self._encoder.encode(text)
        if len(tokens) <= self._max_tokens:
            return [text]
        pieces: list[str] = []
        for start in range(0, len(tokens), self._max_tokens):
            piece = self._encoder.decode(tokens[start : start + self._max_tokens]).strip()
            if piece:
                pieces.append(piece)
        return pieces

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

"""Parent-document (small-to-big) context expansion.

This is NOT chapter-level hierarchical retrieval. Retrieval is done over flat,
page-preserving semantic chunks; this module only *widens the context* of an
already-matched chunk before it is shown to the model.

Small chunks are great for *precise retrieval and citation* but can be too
narrow for the model to reason over (e.g. a numbered procedure split
mid-list). This is the classic small-to-big / parent-document trade-off: we
keep the matched chunk as the citation unit, but expand the text passed to the
model to include its immediate neighbours on the same page (same section), in
reading order. Citations stay chunk-precise; the model gets coherent
surrounding text.
"""

from __future__ import annotations

import re

from app.rag.bm25 import load_corpus_records
from app.logging_config import get_logger

logger = get_logger(__name__)

_CHUNK_IDX_RE = re.compile(r":c(\d+)$")


def _within_page_index(record: dict) -> int:
    """Recover a chunk's within-page order from its id (fallback: metadata)."""
    match = _CHUNK_IDX_RE.search(str(record.get("chunk_id", "")))
    if match:
        return int(match.group(1))
    return int(record.get("chunk_index", 0) or 0)


def _reading_order(record: dict) -> tuple[int, int]:
    """Sort key putting a section's chunks in reading order (page, then index)."""
    return (int(record.get("page", 0)), _within_page_index(record))


class ParentExpander:
    """Expands a chunk with its neighbours in the SAME section (reading order).

    Sections can span multiple pages, so neighbours are ordered by ``(page,
    within-page index)`` across the whole section rather than within one page.
    """

    def __init__(self, records: list[dict]) -> None:
        self._by_section: dict[str, list[dict]] = {}
        for record in records:
            key = str(record.get("section_id", "general"))
            self._by_section.setdefault(key, []).append(record)
        # chunk_id -> (section_id, position within the section's ordered list)
        self._pos: dict[str, tuple] = {}
        for key, siblings in self._by_section.items():
            siblings.sort(key=_reading_order)
            for i, record in enumerate(siblings):
                self._pos[str(record.get("chunk_id"))] = (key, i)

    def expand(self, chunk_id: str, fallback_text: str, window: int) -> str:
        """Return the chunk's text joined with up to ``window`` neighbours/side.

        Neighbours are drawn from the same section in reading order. Falls back
        to ``fallback_text`` if the chunk is unknown or has no useful neighbours.
        """
        if window <= 0:
            return fallback_text
        loc = self._pos.get(str(chunk_id))
        if loc is None:
            return fallback_text
        key, pos = loc
        siblings = self._by_section[key]
        lo = max(0, pos - window)
        hi = min(len(siblings), pos + window + 1)
        if hi - lo <= 1:
            return fallback_text
        return "\n".join(s.get("text", "") for s in siblings[lo:hi])


def load_parent_expander(path: str) -> ParentExpander | None:
    """Build a :class:`ParentExpander` from the on-disk corpus, or ``None``."""
    records = load_corpus_records(path)
    if not records:
        logger.warning("Corpus '%s' unavailable; parent expansion disabled.", path)
        return None
    logger.info("Parent-document expander ready (%d chunks).", len(records))
    return ParentExpander(records)

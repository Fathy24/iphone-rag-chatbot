"""Map every page to its enclosing section/chapter title.

The assessment requires citing a *page number and/or section*. We enrich each
chunk with a human-readable section label so answers can say e.g.
"(p. 42, Chapter 4: Siri)".

Strategies, in order of preference:

1. The PDF's embedded outline / table of contents (bookmarks) — the most
   reliable source of section titles when present.
2. The **running header** that the iPhone User Guide prints on every content
   page (e.g. "Chapter 1   iPhone at a Glance"). This gives a direct,
   per-page chapter label and is robust even though this PDF has no outline.

Any pages still without a label inherit the most recent known section.
"""

from __future__ import annotations

import re
import unicodedata

import fitz  # PyMuPDF

from app.logging_config import get_logger

logger = get_logger(__name__)

# Matches both the table-of-contents form ("Chapter 4: Siri") and the running
# header form ("Chapter 1   iPhone at a Glance"), capturing number and title.
_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[:.\s]\s*(.+)$", re.IGNORECASE)

# The guide's back matter uses "Appendix A  Accessibility" / "Appendix B
# International Keyboards" headers instead of "Chapter N". Without this, those
# pages would inherit the last chapter ("Podcasts") and cite the wrong section.
_APPENDIX_RE = re.compile(r"^Appendix\s+([A-Za-z])\s*[:.\s]\s*(.+)$", re.IGNORECASE)

# Exotic spaces frequently present in PDF outlines (thin / narrow no-break /
# non-breaking) that would otherwise show up as artefacts in citations.
_ODD_SPACES = re.compile(r"[\u00a0\u2009\u202f\u200a\u2007\u2060]+")


def _clean_title(title: str) -> str:
    """Normalise a section title so citations render with plain spaces."""
    title = unicodedata.normalize("NFKC", title)
    title = _ODD_SPACES.sub(" ", title)
    return re.sub(r"\s+", " ", title).strip()


def section_slug(title: str) -> str:
    """Derive a stable, filename-safe section id from a section title.

    Chunks sharing a title (e.g. every page of "Chapter 4: Siri") map to the
    same ``section_id``, which is the grouping key for coarse-to-fine retrieval
    and section-scoped context expansion.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "general").lower()).strip("-")
    return slug or "general"


def build_section_index(pdf_path: str) -> dict[int, str]:
    """Build a ``{page_number: section_title}`` lookup for the whole document.

    Every page is assigned the most recent section heading at or before it, so
    pages between headings inherit the correct chapter title.

    Args:
        pdf_path: Filesystem path to the source PDF.

    Returns:
        A mapping from 1-based page number to its section title. Pages with no
        resolvable section map to ``"General"``.
    """
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            anchors = _outline_anchors(doc)
            if not anchors:
                logger.info("No usable PDF outline; using running-header scan.")
                anchors = _running_header_anchors(doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Section detection failed (%s); using 'General'.", exc)
        return {}

    return _fill_forward(anchors, total_pages)


def _outline_anchors(doc: "fitz.Document") -> dict[int, str]:
    """Extract ``{page: title}`` anchors from the PDF's table of contents."""
    anchors: dict[int, str] = {}
    try:
        toc = doc.get_toc(simple=True)  # list of [level, title, page]
    except Exception:  # noqa: BLE001
        return anchors
    for entry in toc or []:
        try:
            _level, title, page = entry[0], entry[1], entry[2]
        except (IndexError, TypeError):
            continue
        title = _clean_title(title or "")
        if title and isinstance(page, int) and page >= 1:
            # Keep the first (top-level) title seen for a given page.
            anchors.setdefault(page, title)
    if anchors:
        logger.info("Found %d section anchors from PDF outline.", len(anchors))
    return anchors


def _running_header_anchors(doc: "fitz.Document") -> dict[int, str]:
    """Assign each page its chapter from the printed running header.

    The guide repeats a "Chapter N  <Title>" header on every content page, so
    this yields a direct, accurate page-to-section map. For a given page we take
    the first chapter-shaped line found (the header sits at the top of the page).
    """
    anchors: dict[int, str] = {}
    for index in range(doc.page_count):
        text = doc.load_page(index).get_text("text") or ""
        for raw_line in text.splitlines():
            line = _clean_title(raw_line)
            chapter = _CHAPTER_RE.match(line)
            if chapter:
                number, title = chapter.group(1), chapter.group(2).strip(" .:-\t")
                if title:
                    anchors[index + 1] = f"Chapter {number}: {title}"
                    break
            appendix = _APPENDIX_RE.match(line)
            if appendix:
                letter, title = appendix.group(1).upper(), appendix.group(2).strip(" .:-\t")
                if title:
                    anchors[index + 1] = f"Appendix {letter}: {title}"
                    break
    logger.info("Found %d section anchors from running headers.", len(anchors))
    return anchors


def _fill_forward(anchors: dict[int, str], total_pages: int) -> dict[int, str]:
    """Propagate each section title forward until the next anchor."""
    index: dict[int, str] = {}
    current = "General"
    for page in range(1, max(total_pages, 1) + 1):
        if page in anchors:
            current = anchors[page]
        index[page] = current
    return index

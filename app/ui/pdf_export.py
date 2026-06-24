"""Styled PDF export for answers and cited passages.

Mirrors the "professional" document style used by the AI-Agentic backend
(``MCP_Clients/local_function_client.py``): A4 with 72pt margins, dark-blue
titles/headings, justified body text, and a grey footer with a timestamp and
page number. We keep the dependency surface tiny (only ``reportlab``) and do a
lightweight Markdown -> flowables conversion so the exported answer keeps its
headings, bold labels, and step lists.

These helpers are used by the Chainlit download actions so a reviewer can save
an answer or any cited passage as a clean, branded PDF.
"""

from __future__ import annotations

import html
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# A muted "PwC-ish" professional palette to match the backend exports.
_BRAND = colors.HexColor("#2D2D72")  # deep indigo for titles/headings
_ACCENT = colors.HexColor("#D04A02")  # warm accent for source headers
_MUTED = colors.HexColor("#6B7280")  # grey metadata/footer


def _styles() -> dict:
    """Return the shared professional paragraph styles."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Heading1"],
            fontSize=18,
            leading=22,
            spaceAfter=6,
            textColor=_BRAND,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontSize=9,
            textColor=_MUTED,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "DocH2",
            parent=base["Heading2"],
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=_BRAND,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "DocBullet",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            spaceAfter=4,
            leftIndent=16,
            bulletIndent=4,
            alignment=TA_LEFT,
        ),
        "source_head": ParagraphStyle(
            "SourceHead",
            parent=base["Heading3"],
            fontSize=11.5,
            leading=14,
            spaceBefore=12,
            spaceAfter=2,
            textColor=_ACCENT,
        ),
        "meta": ParagraphStyle(
            "SourceMeta",
            parent=base["Normal"],
            fontSize=8.5,
            textColor=_MUTED,
            spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "SourceQuote",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=12,
            spaceAfter=8,
            textColor=colors.HexColor("#1F2937"),
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=_MUTED,
            alignment=TA_CENTER,
        ),
    }


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Convert a subset of inline Markdown to reportlab markup (``<b>`` etc.)."""
    escaped = html.escape(text)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _CODE_RE.sub(r"<font face='Courier'>\1</font>", escaped)
    return escaped


def _markdown_flowables(text: str, styles: dict) -> List:
    """Lightweight Markdown -> flowables (headings, bullets, numbered, prose)."""
    flow: List = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:]), styles["h2"]))
        elif stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:]), styles["h2"]))
        elif stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:]), styles["h2"]))
        elif re.match(r"^[-*]\s+", stripped):
            body = _inline(re.sub(r"^[-*]\s+", "", stripped))
            flow.append(Paragraph(body, styles["bullet"], bulletText="•"))
        elif re.match(r"^\d+[.)]\s+", stripped):
            num = re.match(r"^(\d+)[.)]\s+", stripped).group(1)
            body = _inline(re.sub(r"^\d+[.)]\s+", "", stripped))
            flow.append(Paragraph(body, styles["bullet"], bulletText=f"{num}."))
        else:
            flow.append(Paragraph(_inline(stripped), styles["body"]))
    return flow


def _page_furniture(canvas, doc, title: str) -> None:
    """Draw the footer (and a thin header rule) on every page."""
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(_MUTED)
    canvas.setLineWidth(0.4)
    canvas.line(72, A4[1] - 60, width - 72, A4[1] - 60)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_MUTED)
    canvas.drawString(72, A4[1] - 52, "iPhone Guide Assistant")
    footer = f"Generated by iPhone Guide Assistant  ·  Page {doc.page}"
    canvas.drawCentredString(width / 2.0, 36, footer)
    canvas.restoreState()


def _build(story: Iterable, title: str) -> str:
    """Render the story to a temp PDF and return its path."""
    out = Path(tempfile.gettempdir()) / f"iphone-guide-{uuid.uuid4().hex[:8]}.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        title=title,
        leftMargin=72,
        rightMargin=72,
        topMargin=78,
        bottomMargin=54,
    )
    styles = _styles()
    head = [
        Paragraph(html.escape(title), styles["title"]),
        Paragraph(
            "Grounded in the official iPhone User Guide  ·  "
            + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["subtitle"],
        ),
        HRFlowable(width="100%", thickness=0.6, color=_MUTED, spaceAfter=10),
    ]
    doc.build(
        head + list(story),
        onFirstPage=lambda c, d: _page_furniture(c, d, title),
        onLaterPages=lambda c, d: _page_furniture(c, d, title),
    )
    return str(out)


def build_answer_pdf(answer: str, title: str = "iPhone Guide — Answer") -> str:
    """Export an assistant answer (Markdown) to a styled PDF; return its path."""
    styles = _styles()
    story = _markdown_flowables(answer, styles)
    if not story:
        story = [Paragraph("(empty answer)", styles["body"])]
    return _build(story, title)


def _chunk_flowables(chunk, index: int, styles: dict) -> List:
    """Flowables for a single cited passage (header, scores, quote, context)."""
    flow: List = [
        Paragraph(
            f"[{index}] Page {getattr(chunk, 'page', '?')} · "
            f"{html.escape(str(getattr(chunk, 'section', 'General')))}",
            styles["source_head"],
        )
    ]
    scores = []
    for label, attr, fmt in (
        ("rerank", "rerank_score", "{:.2f}"),
        ("dense", "dense_score", "{:.2f}"),
        ("bm25", "sparse_score", "{:.1f}"),
    ):
        val = getattr(chunk, attr, None)
        if val is not None:
            scores.append(f"{label} {fmt.format(val)}")
    fused = getattr(chunk, "fused_score", None)
    prerank = getattr(chunk, "prerank_rank", None)
    if fused is not None:
        scores.append(f"RRF {fused:.4f}")
    if prerank is not None:
        scores.append(f"pre-rerank #{prerank}")
    if scores:
        flow.append(Paragraph("  ·  ".join(scores), styles["meta"]))

    text = (getattr(chunk, "text", "") or "").strip()
    flow.append(Paragraph(_inline(text).replace("\n", "<br/>"), styles["quote"]))

    parent = (getattr(chunk, "parent_text", "") or "").strip()
    if parent and parent != text:
        flow.append(Paragraph("Surrounding context (same section):", styles["meta"]))
        flow.append(Paragraph(_inline(parent).replace("\n", "<br/>"), styles["quote"]))
    flow.append(Spacer(1, 6))
    return flow


def build_chunk_pdf(chunk, index: int, title: str | None = None) -> str:
    """Export a single cited passage to a styled PDF; return its path."""
    styles = _styles()
    page = getattr(chunk, "page", "?")
    title = title or f"iPhone Guide — Source (p. {page})"
    return _build(_chunk_flowables(chunk, index, styles), title)


def build_chunks_pdf(chunks: list, title: str = "iPhone Guide — Cited Sources") -> str:
    """Export all cited passages to a single styled PDF; return its path."""
    styles = _styles()
    story: List = []
    for i, chunk in enumerate(chunks, start=1):
        story.extend(_chunk_flowables(chunk, i, styles))
    if not story:
        story = [Paragraph("No sources were cited for this answer.", styles["body"])]
    return _build(story, title)

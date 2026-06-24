"""Typed state for the tool-calling RAG agent.

``messages`` uses LangGraph's ``add_messages`` reducer so the conversation grows
correctly across turns when a checkpointer is attached (this is our in-session
short-term memory) and so tool calls / tool results thread through the
``agent <-> tools`` loop. The remaining keys are per-turn scratch space.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RagState(TypedDict, total=False):
    """State threaded through the agent graph."""

    # Full conversation history (persisted per session by the checkpointer),
    # including the agent's tool-call messages and the tools' results.
    messages: Annotated[list, add_messages]

    # The current user question (raw text of the latest human turn).
    question: str

    # Chunks retrieved across all `search_guide` calls this turn
    # (``RetrievedChunk`` objects) — used for citations/source panels and the
    # faithfulness check.
    chunks: list[Any]

    # Per-tool-call telemetry for the UI: one dict per `search_guide` call with
    # the query, the coarse sections chosen, and how many passages it grounded.
    tool_runs: list[Any]

    # Post-generation grounding verdict: ``{"score", "label", "unsupported"}``.
    faithfulness: dict

    # Set by the input guard; routes the turn to a safe refusal.
    blocked: bool
    block_reason: str

    # Rolling natural-language summary of older turns (long-term memory). The
    # checkpointer persists it per session so context survives beyond the recent
    # message window without unbounded prompt growth.
    summary: str

    # Index into the conversation up to which messages have already been folded
    # into ``summary`` — lets the memory node summarise each turn exactly once.
    summarized_upto: int

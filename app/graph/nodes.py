"""Graph node implementations for the tool-calling RAG agent.

Each node is a function ``(state[, config]) -> partial_state``; they are wired
together in :mod:`app.graph.build`. Nodes capture their heavy dependencies
(chat model, retriever) via closures created at build time, so the graph itself
stays cheap to construct and easy to test.

The core is an ``agent <-> tools`` loop: the model decides, each turn, whether
to call the ``search_guide`` tool (genuine, intent-driven tooling). Greetings
and out-of-scope requests are answered directly with no retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import Settings
from app.graph.guard import assess_input
from app.graph.prompts import (
    AGENT_SYSTEM_PROMPT,
    FAITHFULNESS_PROMPT,
    NOT_FOUND_MESSAGE,
    SUMMARY_PROMPT,
    format_context,
    format_tool_result,
    render_memory_block,
)
from app.graph.state import RagState
from app.logging_config import get_logger

if TYPE_CHECKING:  # avoid importing the retriever (heavy deps) at module load
    from app.rag.retriever import Retriever

logger = get_logger(__name__)


@tool
def search_guide(query: str) -> str:
    """Search the official iPhone User Guide and return the most relevant
    passages, each tagged with its page number and section.

    Use this for a SINGLE iPhone topic — setup, a feature, a setting, a how-to,
    or troubleshooting. Do NOT use it for greetings, small-talk, or clearly
    off-topic requests. For several distinct topics at once, use
    `search_guide_parallel` instead.

    Args:
        query: A focused, self-contained search query for one topic.
    """
    # Executed by the tools node (which captures structured results into state);
    # this body is only the schema/description bound to the model.
    return ""


@tool
def search_guide_parallel(queries: list[str]) -> str:
    """Search the iPhone User Guide for SEVERAL distinct topics at once.

    Use this when the user's message bundles multiple unrelated questions (e.g.
    "how do I take a screenshot, set up a hotspot, and save battery?"). Pass one
    focused, self-contained query per topic; they are retrieved IN PARALLEL and
    the passages are returned grouped per sub-query so you can synthesise one
    cohesive, cited answer. Prefer this over many sequential `search_guide`
    calls when the topics are known up front.

    Args:
        queries: A list of focused search queries, one per distinct topic.
    """
    # Executed by the tools node; body is only the schema bound to the model.
    return ""


# The toolset bound to the model.
TOOLS = [search_guide, search_guide_parallel]


def _latest_user_text(state: RagState) -> str:
    """Return the text of the most recent human message in the state."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def make_guard_node() -> Callable[[RagState], dict]:
    """Build the input-guard node (cheap, pre-LLM injection screen)."""

    def guard_node(state: RagState) -> dict:
        text = _latest_user_text(state)
        result = assess_input(text)
        if not result.allowed:
            logger.info("Input blocked by guard: %s", result.reason)
        return {
            "question": text,
            "blocked": not result.allowed,
            "block_reason": result.reason,
        }

    return guard_node


# -- Agent (decides whether to call the search tool) -------------------------


def make_agent_node(model_with_tools, settings: Settings) -> Callable[[RagState], dict]:
    """Build the agent node: the model reasons and may emit tool calls.

    The model receives the system prompt (safety contract + tool protocol), the
    long-term memory summary, and the recent message window (including this
    turn's tool calls/results), then either calls ``search_guide`` or produces
    the final answer.
    """

    def agent_node(state: RagState) -> dict:
        history = _window_messages(state.get("messages", []), settings.max_history_messages)
        memory_block = render_memory_block(state.get("summary", ""))
        system = AGENT_SYSTEM_PROMPT
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        messages = [SystemMessage(content=system), *history]
        try:
            response = model_with_tools.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent step failed: %s", exc)
            return {
                "messages": [
                    AIMessage(content="I hit an error while thinking. Please try again.")
                ]
            }

        # Final answer (no tool calls): enforce the mandatory citations.
        if not getattr(response, "tool_calls", None):
            content = response.content if isinstance(response.content, str) else str(response.content)
            content = _ensure_citations(content, state.get("chunks") or [])
            return {"messages": [AIMessage(content=content)]}
        return {"messages": [response]}

    return agent_node


def make_tools_node(
    retriever: "Retriever", settings: Settings
) -> Callable[[RagState, dict], dict]:
    """Build the tool-executor node for the search tools.

    Handles both ``search_guide`` (one query) and ``search_guide_parallel``
    (many queries, retrieved concurrently with a bounded thread pool). Returns a
    ``ToolMessage`` per tool call for the model and captures the structured
    chunks and per-call telemetry into state (for citations and the UI).
    """

    def _search(query: str, top_k, use_reranker, coarse_n) -> tuple[list, object]:
        return retriever.retrieve_with_info(
            query.strip(), top_k=top_k, use_reranker=use_reranker, coarse_n=coarse_n
        )

    def tools_node(state: RagState, config: dict | None = None) -> dict:
        overrides = (config or {}).get("configurable", {})
        top_k = overrides.get("top_k")
        use_reranker = overrides.get("use_reranker")
        coarse_n = overrides.get("coarse_sections_n")

        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        out: list = []
        chunks_acc = list(state.get("chunks") or [])
        runs = list(state.get("tool_runs") or [])

        for call in calls:
            name = call.get("name")
            call_id = call.get("id")
            args = call.get("args") or {}

            if name == "search_guide":
                query = str(args.get("query", "")).strip()
                content, chunks_acc, run = _run_single(
                    _search, query, top_k, use_reranker, coarse_n, chunks_acc
                )
                out.append(ToolMessage(content=content, tool_call_id=call_id))
                runs.append(run)

            elif name == "search_guide_parallel":
                queries = _clean_queries(args.get("queries"), settings.max_parallel_queries)
                content, chunks_acc, batch_runs = _run_parallel(
                    _search, queries, top_k, use_reranker, coarse_n, chunks_acc, settings
                )
                out.append(ToolMessage(content=content, tool_call_id=call_id))
                runs.extend(batch_runs)

            else:
                out.append(ToolMessage(content="Unknown tool.", tool_call_id=call_id))

        return {"messages": out, "chunks": chunks_acc, "tool_runs": runs}

    return tools_node


def _clean_queries(raw, cap: int) -> list[str]:
    """Normalise the parallel tool's ``queries`` arg into a bounded list."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    seen: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return seen[:cap]


def _run_single(search, query, top_k, use_reranker, coarse_n, chunks_acc):
    """Execute one search; return (tool_content, updated_chunks, telemetry)."""
    if not query:
        return "NO RESULTS: empty query.", chunks_acc, {"query": query, "n": 0}
    try:
        chunks, info = search(query, top_k, use_reranker, coarse_n)
    except Exception as exc:  # noqa: BLE001 - isolate retrieval failures
        logger.error("search_guide failed for %r: %s", query, exc)
        return ("The search failed; treat as no results.", chunks_acc,
                {"query": query, "error": True})
    _tag_origin(chunks, query)
    chunks_acc = _merge_chunks(chunks_acc, chunks)
    logger.info("search_guide(%r) -> %d passages", query[:60], len(chunks))
    return format_tool_result(chunks), chunks_acc, _run_telemetry(query, chunks, info)


def _run_parallel(search, queries, top_k, use_reranker, coarse_n, chunks_acc, settings):
    """Execute several searches concurrently; group results per sub-query."""
    if not queries:
        return "NO RESULTS: no sub-queries provided.", chunks_acc, []
    workers = max(1, min(settings.retrieval_max_workers, len(queries)))

    def _one(q: str):
        try:
            return q, search(q, top_k, use_reranker, coarse_n)
        except Exception as exc:  # noqa: BLE001 - isolate per-branch failures
            logger.warning("Parallel search failed for %r: %s", q, exc)
            return q, ([], None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, queries))

    blocks: list[str] = []
    runs: list[dict] = []
    for i, (q, (chunks, info)) in enumerate(results, start=1):
        _tag_origin(chunks, q)
        chunks_acc = _merge_chunks(chunks_acc, chunks)
        body = format_tool_result(chunks)
        blocks.append(f"### Sub-query {i}: {q}\n{body}")
        runs.append(_run_telemetry(q, chunks, info, batch=True))
    logger.info("search_guide_parallel(%d queries) -> %d unique passages",
                len(queries), len(chunks_acc))
    header = (
        "PARALLEL RESULTS — one block per sub-query (use ONLY these; cite each "
        "passage's page and section, and address every sub-query):\n\n"
    )
    return header + "\n\n".join(blocks), chunks_acc, runs


def _run_telemetry(query: str, chunks: list, info, *, batch: bool = False) -> dict:
    """Build the per-tool-call telemetry dict surfaced in the UI."""
    return {
        "query": query,
        "sections": list(info.sections) if info else [],
        "hierarchical": bool(info and info.hierarchical),
        "coarse_hybrid": bool(info and getattr(info, "coarse_hybrid", False)),
        "fell_back": bool(info and info.fell_back),
        "n": len(chunks),
        "batch": batch,
    }


def route_after_agent(state: RagState) -> str:
    """Conditional edge: run requested tools, else verify the final answer."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "verify"


def _tag_origin(chunks: list, query: str) -> None:
    """Record which search query first surfaced each chunk (first query wins)."""
    for chunk in chunks:
        if not getattr(chunk, "origin_query", ""):
            chunk.origin_query = query


def _merge_chunks(existing: list, new: list) -> list:
    """Union of retrieved chunks across tool calls, de-duplicated, order-stable."""
    seen = {(c.page, c.section, c.text[:80]) for c in existing}
    merged = list(existing)
    for chunk in new:
        key = (chunk.page, chunk.section, chunk.text[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
    return merged


# -- Faithfulness / grounding verification -----------------------------------


def _is_refusal(answer: str) -> bool:
    """True for the standard "not in the guide" refusal (skip verification)."""
    text = (answer or "").strip().lower()
    return text == NOT_FOUND_MESSAGE.lower() or "couldn't find that in the iphone user guide" in text


def make_faithfulness_node(chat_model, settings: Settings) -> Callable[[RagState], dict]:
    """Build the post-generation grounding-verification node.

    Verifies that the answer's claims are supported by the retrieved passages
    and records a faithfulness verdict (score + any unsupported claims). It is
    advisory (surfaced as a UI badge) and never rewrites the answer, so a flaky
    check can't corrupt a good response. Skipped for small-talk/refusals (no
    chunks) — which is also why a greeting never triggers it.
    """

    def faithfulness_node(state: RagState) -> dict:
        if not settings.enable_faithfulness_check:
            return {}
        chunks = state.get("chunks") or []
        messages = state.get("messages") or []
        if not chunks or not messages:
            return {}
        answer = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
        if _is_refusal(answer):
            return {}
        try:
            score, unsupported = _check_faithfulness(chat_model, answer, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Faithfulness check failed (%s); skipping.", exc)
            return {}
        label = "grounded" if score >= settings.faithfulness_min_score else "partial"
        logger.info("Faithfulness: %.2f (%s)", score, label)
        return {"faithfulness": {"score": score, "label": label, "unsupported": unsupported}}

    return faithfulness_node


_SCORE_RE = re.compile(r"SCORE:\s*(\d+)", re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"UNSUPPORTED:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _check_faithfulness(chat_model, answer: str, chunks: list) -> tuple[float, list[str]]:
    """Return a 0-1 grounding score and any unsupported claim phrases."""
    context = format_context(chunks)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = chat_model.invoke([HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)

    score_match = _SCORE_RE.search(text)
    score = (int(score_match.group(1)) / 100.0) if score_match else 1.0
    score = max(0.0, min(1.0, score))

    unsupported: list[str] = []
    unsup_match = _UNSUPPORTED_RE.search(text)
    if unsup_match:
        raw = unsup_match.group(1).strip()
        if raw.lower() not in ("none", "none.", "n/a", ""):
            unsupported = [p.strip(" -•\t") for p in raw.split(",") if p.strip(" -•\t")]
    return score, unsupported[:5]


# -- Guard refusal + long-term memory ----------------------------------------


def make_refusal_node() -> Callable[[RagState], dict]:
    """Build the node that responds to guard-blocked input safely."""

    def refusal_node(state: RagState) -> dict:
        message = (
            "I can only answer questions about the iPhone User Guide, and I "
            "can't follow instructions that try to change that. Ask me about "
            "using your iPhone and I'll help."
        )
        return {"messages": [AIMessage(content=message)]}

    return refusal_node


def make_memory_node(chat_model, settings: Settings) -> Callable[[RagState], dict]:
    """Build the long-term-memory node (rolling conversation summary).

    Once the conversation grows beyond the recent-message window, the turns that
    have scrolled out are folded into a compact running summary (each turn
    summarised exactly once via ``summarized_upto``). This keeps long sessions
    coherent without letting the prompt grow unbounded.
    """

    def memory_node(state: RagState) -> dict:
        if not settings.enable_summary_memory:
            return {}
        convo = [
            m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
        ]
        # Count only human/AI turns; ignore tool-call/tool-result messages.
        convo = [m for m in convo if not getattr(m, "tool_calls", None)]
        if len(convo) <= settings.summary_after_messages:
            return {}

        window = settings.max_history_messages
        boundary = max(0, len(convo) - window) if window > 0 else len(convo)
        already = int(state.get("summarized_upto", 0))
        pending = convo[already:boundary]
        if not pending:
            return {}

        transcript = _format_transcript(pending)
        prompt = SUMMARY_PROMPT.format(
            max_tokens=settings.summary_max_tokens,
            summary=state.get("summary", "") or "(none)",
            transcript=transcript,
        )
        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
            summary = (response.content or "").strip()
            if summary:
                logger.info("Updated long-term summary (folded %d msgs)", len(pending))
                return {"summary": summary, "summarized_upto": boundary}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summary update failed (%s); keeping prior summary.", exc)
        return {}

    return memory_node


def route_after_guard(state: RagState) -> str:
    """Conditional edge: send blocked input to refusal, else to the agent."""
    return "refuse" if state.get("blocked") else "agent"


def _format_transcript(messages: list) -> str:
    """Render messages as a compact ``Role: text`` transcript for summarising."""
    lines: list[str] = []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _window_messages(messages: list, max_turns: int) -> list:
    """Return a tail window of messages that starts at a user-turn boundary.

    Keeps the last ``max_turns`` user turns plus everything after the earliest
    kept user message. Starting at a ``HumanMessage`` guarantees we never split
    an assistant tool-call from its matching tool result (which the chat API
    requires to stay paired).
    """
    if max_turns <= 0 or not messages:
        return list(messages)
    human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_positions) <= max_turns:
        return list(messages)
    start = human_positions[-max_turns]
    return list(messages[start:])


def _ensure_citations(answer: str, chunks: list) -> str:
    """Guarantee a ``Sources:`` line on substantive answers (deterministic net).

    The agent is instructed to cite, but we enforce it so the mandatory, tested
    requirement never silently fails. Refusal-style answers never get a sources
    line, even if chunks were retrieved earlier in the turn.
    """
    normalised = (answer or "").strip().lower()
    if not chunks:
        return answer
    if "couldn't find that in the iphone user guide" in normalised:
        return answer
    if normalised == NOT_FOUND_MESSAGE.lower():
        return answer
    if "sources:" in normalised:
        return answer
    seen: list[str] = []
    for chunk in chunks:
        citation = f"p. {chunk.page} — {chunk.section}"
        if citation not in seen:
            seen.append(citation)
    if not seen:
        return answer
    return f"{answer.rstrip()}\n\nSources: " + "; ".join(seen)

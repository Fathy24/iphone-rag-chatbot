"""Chainlit entrypoint for the iPhone User Guide RAG agent.

Run with::

    chainlit run app/ui/chainlit_app.py --host 0.0.0.0 --port $PORT

The compiled LangGraph is built once per process and shared across sessions.
Each Chainlit session is mapped to a unique LangGraph ``thread_id`` so the
checkpointer keeps every conversation's memory isolated.

The UI surfaces the agent's reasoning across TWO messages per turn so the logs
sit above the answer (Chainlit pins a turn's ``cl.Step`` trace *below* its
message, so we avoid steps entirely):

1. An ``ExecutionTrace`` custom element (sent first, updated live) renders the
   pipeline stages — guard, every search tool call with its coarse sections,
   faithfulness, memory, turn metrics — as collapsible rows.
2. The grounded answer streams into a second message underneath, carrying an
   ``AnswerTools`` custom element: a compact action bar (copy answer, download
   the answer / cited sources as a styled PDF) plus reference chips grouped by
   topic. Nothing opens on its own — a passage viewer with an organised
   scoreboard (final rank, hybrid-fusion rank, raw signals) appears inline only
   when the user clicks a chip, and closes via its X button.

A settings panel exposes live retrieval controls.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

# Ensure the project root is importable when Chainlit loads this file by path
# (``chainlit run app/ui/chainlit_app.py``) regardless of the current PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
from chainlit.input_widget import Slider, Switch
from langchain_core.messages import AIMessage, HumanMessage

from app.config import get_settings
from app.graph.build import build_rag_graph
from app.logging_config import configure_logging, get_logger
from app.observability import TokenCounter
from app.ui.pdf_export import build_answer_pdf, build_chunk_pdf, build_chunks_pdf

configure_logging(get_settings().log_level)
logger = get_logger(__name__)

# Process-wide singletons: build the graph lazily and remember any startup error
# so we can surface a clear message in the UI instead of crashing silently.
_GRAPH = None
_GRAPH_ERROR: str | None = None

# Approximate context-window budget for the chat model (gpt-4o family ~128K),
# used only to scale the UI context-fill gauge. Not a hard limit.
_MODEL_CONTEXT_TOKENS = 128_000

_WELCOME = (
    "### 👋 iPhone Guide Assistant\n\n"
    "I answer **only** from the official **iPhone User Guide** — and I always "
    "cite the page and section. If something isn't in the guide, I'll tell you.\n\n"
    "**Try asking:**\n"
    "- How do I take a screenshot?\n"
    "- How do I check my storage?\n"
    "- Set up Face ID and change my wallpaper\n\n"
    "Expand the **steps** under any answer to see how I found it · open the "
    "**Live dashboard** (top-right) to watch retrieval & memory · tune things in ⚙️."
)


def _get_graph():
    """Return the compiled graph, building it once and caching failures."""
    global _GRAPH, _GRAPH_ERROR
    if _GRAPH is None and _GRAPH_ERROR is None:
        try:
            _GRAPH = build_rag_graph()
        except Exception as exc:  # noqa: BLE001
            _GRAPH_ERROR = str(exc)
            logger.error("Failed to build RAG graph: %s", exc)
    return _GRAPH


@cl.set_starters
async def starters() -> list[cl.Starter]:
    """Suggested prompts shown on the welcome screen."""
    return [
        cl.Starter(
            label="Take a screenshot",
            message="How do I take a screenshot on my iPhone?",
        ),
        cl.Starter(
            label="Set up a Personal Hotspot",
            message="How do I set up a Personal Hotspot?",
        ),
        cl.Starter(
            label="3 topics at once (parallel)",
            message=(
                "Three quick things: how do I take a screenshot, set up a "
                "Personal Hotspot, and extend my battery life?"
            ),
        ),
        cl.Starter(
            label="What can you do?",
            message="What can you do?",
        ),
    ]


def _default_settings() -> dict:
    s = get_settings()
    return {
        "use_reranker": s.use_reranker,
        "top_k": s.retrieval_top_k,
        "coarse_sections_n": s.coarse_sections_n,
    }


def _count_convo_turns(messages: list) -> int:
    """Count human/assistant turns currently in the live message window.

    Tool-call and tool-result messages are ignored — only the conversational
    turns the memory node tracks for its fold threshold are counted.
    """
    return sum(
        1
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    )


def _context_props(
    state: dict,
    counter: TokenCounter,
    *,
    running: bool,
    just_summarized: bool,
) -> dict:
    """Build the ``context`` props for the Dashboard element from the turn state.

    Surfaces (a) how full the context window got — the *peak* single-prompt
    token count vs the model budget — and (b) the rolling-summary memory state
    (live window vs fold threshold, turns already folded), so the user can see
    exactly when long-term summarisation kicks in.
    """
    s = get_settings()
    messages = (state or {}).get("messages") or []
    summary = (state or {}).get("summary", "") or ""
    peak = getattr(counter, "peak_input_tokens", 0) or 0
    pct = (peak / _MODEL_CONTEXT_TOKENS * 100) if _MODEL_CONTEXT_TOKENS else 0.0
    return {
        "running": running,
        "pct": round(pct, 1),
        "peakTokens": peak,
        "budget": _MODEL_CONTEXT_TOKENS,
        "turnTokens": {
            "input": counter.input_tokens,
            "output": counter.output_tokens,
            "total": counter.total_tokens,
            "calls": counter.llm_calls,
        },
        "convoTurns": _count_convo_turns(messages),
        "foldThreshold": s.summary_after_messages,
        "windowSize": s.max_history_messages,
        "foldedTurns": int((state or {}).get("summarized_upto", 0) or 0),
        "summaryActive": bool(summary),
        "summaryChars": len(summary),
        "summaryEnabled": s.enable_summary_memory,
        "justSummarized": just_summarized,
    }


async def _render_sidebar() -> None:
    """Refresh the live dashboard element (context meter + session panel).

    The dashboard is a single inline custom element. A header "Live dashboard"
    link (wired in ``public/custom.js``) slides it into a docked side panel on
    click — Chainlit 2.11's native ``ElementSidebar`` does not reliably mount
    JSX custom elements, so we render it as a normal element and reposition it
    in the DOM instead. We keep a handle to the element and just push fresh
    props + ``update()`` — ``to_dict`` ships ``props`` over the socket, so the
    UI re-renders in place whether it's inline or already in the side panel.
    """
    ctx = cl.user_session.get("context_props")
    side = cl.user_session.get("side_props")
    dash_el = cl.user_session.get("dash_el")
    if dash_el is not None and (ctx is not None or side is not None):
        dash_el.props = {"context": ctx or {}, "panel": side or {}}
        await dash_el.update()


async def _safe_render_sidebar() -> None:
    """Update the dashboard, swallowing any error (it's non-essential)."""
    try:
        await _render_sidebar()
    except Exception as exc:  # noqa: BLE001 - panels must never break a turn
        logger.debug("Dashboard update skipped: %s", exc)


# --- Session panel: catalog, accumulators, and prop builders -----------------

# gpt-4o public list price (USD per 1M tokens) — used only for a rough,
# clearly-approximate session cost readout in the UI.
_PRICE_IN_PER_M = 2.5
_PRICE_OUT_PER_M = 10.0

_SECTION_CATALOG: list[dict] | None = None


def _section_catalog() -> list[dict]:
    """Load (once) the guide's section list for the Guide-map panel.

    Reads the same ``.sections.json`` the coarse retriever uses, keeping only
    the lightweight title/page-range fields (the centroids are ignored here).
    """
    global _SECTION_CATALOG
    if _SECTION_CATALOG is not None:
        return _SECTION_CATALOG
    catalog: list[dict] = []
    try:
        import json

        path = Path(get_settings().sections_path)
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            for r in records:
                pages = r.get("pages") or []
                catalog.append(
                    {
                        "title": str(r.get("title", r.get("section_id", "General"))),
                        "first_page": pages[0] if pages else 0,
                        "last_page": pages[-1] if pages else 0,
                        "n_chunks": int(r.get("n_chunks", 0)),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - panel is non-essential
        logger.debug("Could not load section catalog: %s", exc)
    _SECTION_CATALOG = catalog
    return catalog


def _init_session_panel() -> None:
    """Seed the per-session accumulators backing the side panel."""
    cl.user_session.set(
        "stats",
        {"turns": 0, "input": 0, "output": 0, "calls": 0, "latency": 0.0},
    )
    cl.user_session.set("cited_sections", {})
    cl.user_session.set("cited_pages", {})
    cl.user_session.set("guard_blocks", 0)
    cl.user_session.set("last_guard", {"blocked": False, "reason": ""})


def _update_session_panel(
    final_state: dict,
    counter: TokenCounter,
    latency: float,
    guard: dict,
) -> set[str]:
    """Fold this turn's outcome into the session accumulators.

    Returns the set of section titles cited this turn (for the Guide-map "hit"
    highlight).
    """
    stats = cl.user_session.get("stats") or {}
    stats["turns"] = int(stats.get("turns", 0)) + 1
    stats["input"] = int(stats.get("input", 0)) + counter.input_tokens
    stats["output"] = int(stats.get("output", 0)) + counter.output_tokens
    stats["calls"] = int(stats.get("calls", 0)) + counter.llm_calls
    stats["latency"] = float(stats.get("latency", 0.0)) + latency
    cl.user_session.set("stats", stats)

    cited_sections = cl.user_session.get("cited_sections") or {}
    cited_pages = cl.user_session.get("cited_pages") or {}
    this_turn: set[str] = set()
    seen_here: set[str] = set()
    for chunk in (final_state or {}).get("chunks") or []:
        section = str(getattr(chunk, "section", "General") or "General")
        page = getattr(chunk, "page", "?")
        this_turn.add(section)
        key = f"{page}|{section}"
        if key not in seen_here:  # count each (page, section) once per turn
            seen_here.add(key)
            cited_pages[key] = int(cited_pages.get(key, 0)) + 1
            cited_sections[section] = int(cited_sections.get(section, 0)) + 1
    cl.user_session.set("cited_sections", cited_sections)
    cl.user_session.set("cited_pages", cited_pages)

    if guard.get("blocked"):
        cl.user_session.set("guard_blocks", int(cl.user_session.get("guard_blocks") or 0) + 1)
    cl.user_session.set(
        "last_guard",
        {"blocked": bool(guard.get("blocked")), "reason": str(guard.get("reason", ""))},
    )
    return this_turn


def _build_side_props(final_state: dict, this_turn_sections: set[str]) -> dict:
    """Assemble the Dashboard ``panel`` props from the session accumulators + state."""
    s = get_settings()
    settings = cl.user_session.get("settings") or _default_settings()
    stats = cl.user_session.get("stats") or {}
    turns = int(stats.get("turns", 0)) or 0
    inp = int(stats.get("input", 0))
    out = int(stats.get("output", 0))
    cost = inp / 1_000_000 * _PRICE_IN_PER_M + out / 1_000_000 * _PRICE_OUT_PER_M
    avg_latency = (float(stats.get("latency", 0.0)) / turns) if turns else 0.0

    summary = (final_state or {}).get("summary", "") or ""
    folded = int((final_state or {}).get("summarized_upto", 0) or 0)

    cited_sections = cl.user_session.get("cited_sections") or {}
    guide = [
        {
            "title": c["title"],
            "pages": _page_range(c["first_page"], c["last_page"]),
            "cites": int(cited_sections.get(c["title"], 0)),
            "hit": c["title"] in this_turn_sections,
        }
        for c in _section_catalog()
    ]

    cited_pages = cl.user_session.get("cited_pages") or {}
    citations = []
    for key, count in cited_pages.items():
        page, _, section = key.partition("|")
        citations.append({"page": page, "section": section, "count": count})
    citations.sort(key=lambda x: (-x["count"], str(x["section"])))

    last_guard = cl.user_session.get("last_guard") or {}
    return {
        "stats": {
            "turns": turns,
            "input": inp,
            "output": out,
            "total": inp + out,
            "calls": int(stats.get("calls", 0)),
            "avgLatency": round(avg_latency, 1),
            "cost": round(cost, 4),
        },
        "config": {
            "backend": s.vector_backend,
            "mode": s.retrieval_mode,
            "chatModel": s.chat_model,
            "embedModel": s.embedding_model,
            "reranker": bool(settings.get("use_reranker", s.use_reranker)),
            "coarseN": int(settings.get("coarse_sections_n", s.coarse_sections_n)),
            "topK": int(settings.get("top_k", s.retrieval_top_k)),
            "sectionsTotal": len(_section_catalog()),
        },
        "summary": {
            "text": summary,
            "active": bool(summary),
            "foldedTurns": folded,
        },
        "guideMap": {
            "sections": guide,
            "totalCites": sum(cited_sections.values()),
        },
        "citations": {"items": citations, "total": len(citations)},
        "guard": {
            "lastBlocked": bool(last_guard.get("blocked")),
            "lastReason": str(last_guard.get("reason", "")),
            "blockedCount": int(cl.user_session.get("guard_blocks") or 0),
        },
    }


def _page_range(first: int, last: int) -> str:
    """Render a section's page span (e.g. ``p.80`` or ``p.80–95``)."""
    if not first:
        return ""
    if not last or last == first:
        return f"p.{first}"
    return f"p.{first}\u2013{last}"


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise a session: thread id, settings panel, and greeting."""
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    cl.user_session.set("settings", _default_settings())

    if _get_graph() is None:
        await cl.Message(
            content=(
                "The assistant failed to start. Please check the server "
                f"configuration.\n\n```\n{_GRAPH_ERROR}\n```"
            )
        ).send()
        return

    defaults = _default_settings()
    try:
        await cl.ChatSettings(
            [
                Switch(
                    id="use_reranker",
                    label="Cohere reranking (cross-encoder)",
                    initial=defaults["use_reranker"],
                ),
                Slider(
                    id="coarse_sections_n",
                    label="Coarse stage — top sections searched (of 35)",
                    initial=float(defaults["coarse_sections_n"]),
                    min=1,
                    max=15,
                    step=1,
                ),
                Slider(
                    id="top_k",
                    label="Fine stage — child passages returned (top-k)",
                    initial=float(defaults["top_k"]),
                    min=1,
                    max=12,
                    step=1,
                ),
            ]
        ).send()
    except Exception as exc:  # noqa: BLE001 - settings panel is non-essential
        logger.warning("Could not render settings panel: %s", exc)

    await cl.Message(content=_WELCOME).send()

    # A single live-dashboard custom element. It renders inline, but a header
    # "Live dashboard" link (wired in public/custom.js) slides it into a docked
    # side panel on click — keeping it out of the chat flow until the user wants
    # it, while still updating live. We hold a handle and refresh it in place
    # (while a turn runs and after it finishes) so the user can watch the
    # context-window fill, memory folding, retrieval config, guide-map hits,
    # citations and guardrail status. If custom.js fails to load, the element
    # simply stays visible inline (graceful degradation — never hidden).
    try:
        cl.user_session.set(
            "context_props",
            _context_props({}, TokenCounter(), running=False, just_summarized=False),
        )
        cl.user_session.set("folded_turns", 0)
        _init_session_panel()
        cl.user_session.set("side_props", _build_side_props({}, set()))
        dash_el = cl.CustomElement(
            name="Dashboard",
            props={
                "context": cl.user_session.get("context_props"),
                "panel": cl.user_session.get("side_props"),
            },
        )
        cl.user_session.set("dash_el", dash_el)
        # Empty content: the dashboard renders only the element, and custom.js
        # hides this origin bubble entirely (relocating the live element into a
        # header-toggled side panel). If JS is unavailable it stays inline.
        await cl.Message(content="", elements=[dash_el]).send()
    except Exception as exc:  # noqa: BLE001 - the dashboard is non-essential
        logger.warning("Could not render session dashboard: %s", exc)

    # A static "Test cases" panel: copy-and-paste prompts grouped per feature so
    # a reviewer can exercise every capability (hierarchical/hybrid retrieval,
    # parallel multi-query, memory, refusal, guardrails). Rendered inline, but a
    # header "Test cases" link (wired in public/custom.js) slides it into a
    # docked panel on click. All content lives in TestCases.jsx — no props.
    try:
        tests_el = cl.CustomElement(name="TestCases", props={})
        await cl.Message(content="", elements=[tests_el]).send()
    except Exception as exc:  # noqa: BLE001 - the test panel is non-essential
        logger.warning("Could not render test-cases panel: %s", exc)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Persist live retrieval settings for the current session."""
    current = cl.user_session.get("settings") or _default_settings()
    current.update(
        {
            "use_reranker": bool(settings.get("use_reranker", current["use_reranker"])),
            "top_k": int(settings.get("top_k", current["top_k"])),
            "coarse_sections_n": int(
                settings.get("coarse_sections_n", current["coarse_sections_n"])
            ),
        }
    )
    cl.user_session.set("settings", current)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle a user turn: visualise the agent, stream the grounded answer."""
    graph = _get_graph()
    if graph is None:
        await cl.Message(
            content=f"The assistant is unavailable.\n\n```\n{_GRAPH_ERROR}\n```"
        ).send()
        return

    thread_id = cl.user_session.get("thread_id") or str(uuid.uuid4())
    settings = cl.user_session.get("settings") or _default_settings()
    app_settings = get_settings()
    counter = TokenCounter()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "use_reranker": settings["use_reranker"],
            "top_k": settings["top_k"],
            "coarse_sections_n": settings["coarse_sections_n"],
        },
        "callbacks": [counter],
        # Bound the agent<->tools loop so a misbehaving turn can't run away.
        "recursion_limit": app_settings.agent_max_tool_calls * 2 + 4,
    }

    started = time.time()
    final_state: dict | None = None
    ctx = {"runs": 0}  # how many tool runs have been folded into the trace

    # The execution trace lives in its OWN message (a custom element), sent
    # FIRST and updated live as the graph runs. The grounded answer is a SECOND
    # message sent afterwards. Two separate messages render in send order, so
    # the logs reliably sit ABOVE the response (Chainlit pins a turn's `cl.Step`
    # trace *below* its message, which is why the previous single-message
    # approach couldn't flip them).
    trace_steps: list[dict] = []
    trace_el = cl.CustomElement(
        name="ExecutionTrace", props={"steps": trace_steps, "running": True}
    )
    trace_msg = cl.Message(content="", elements=[trace_el])
    await trace_msg.send()

    # Flip the inline dashboard's context meter into its "running" (spinner)
    # state while keeping its last-known values, so it animates in place without
    # flickering. The dashboard is a pinned inline element, so it never
    # disappears — we just push fresh props and update() it.
    ctx_props = cl.user_session.get("context_props")
    if ctx_props is not None:
        ctx_props["running"] = True
        await _safe_render_sidebar()

    turn_guard = {"blocked": False, "reason": ""}

    async def _push(new_steps: list[tuple[str, str]]) -> None:
        if not new_steps:
            return
        trace_steps.extend({"title": t, "body": b} for t, b in new_steps)
        trace_el.props["steps"] = list(trace_steps)
        await trace_el.update()

    try:
        async for stream_mode, data in graph.astream(
            # Seed empties for the per-turn scratch channels so retrieval
            # telemetry, citations, and the faithfulness check are scoped to THIS
            # turn only. The checkpointer persists state across turns, so without
            # this reset these plain (reducer-less) channels would accumulate
            # every prior turn's searches/chunks. `messages` (add_messages) and
            # the long-term `summary`/`summarized_upto` deliberately persist.
            {
                "messages": [HumanMessage(content=message.content)],
                "chunks": [],
                "tool_runs": [],
                "faithfulness": {},
            },
            config,
            stream_mode=["updates", "values"],
        ):
            if stream_mode == "updates":
                for node, payload in (data or {}).items():
                    if node == "guard" and (payload or {}).get("blocked"):
                        turn_guard = {
                            "blocked": True,
                            "reason": str((payload or {}).get("block_reason", "unsafe")),
                        }
                    await _push(_steps_for(node, payload or {}, ctx))
            elif stream_mode == "values":
                final_state = data
    except Exception as exc:  # noqa: BLE001
        logger.error("Turn failed: %s", exc)
        trace_el.props["running"] = False
        await trace_el.update()
        if ctx_props is not None:
            ctx_props["running"] = False
            await _safe_render_sidebar()
        await cl.Message(
            content=(
                "Sorry, something went wrong while answering. Please try again.\n\n"
                f"```\n{exc}\n```"
            )
        ).send()
        return

    # Fold the turn-metrics into the trace, then mark it finished.
    await _push(_metrics_steps(started, counter, final_state or {}))
    trace_el.props["running"] = False
    await trace_el.update()

    # Refresh the whole sidebar with this turn's real numbers: the context meter
    # (window fill + fold pulse) and the session dashboard (stats, citations,
    # guide-map hits, guardrail, rolling summary).
    prev_folded = int(cl.user_session.get("folded_turns") or 0)
    now_folded = int((final_state or {}).get("summarized_upto", 0) or 0)
    just_summarized = now_folded > prev_folded
    cl.user_session.set(
        "context_props",
        _context_props(
            final_state or {}, counter,
            running=False, just_summarized=just_summarized,
        ),
    )
    cl.user_session.set("folded_turns", now_folded)
    this_turn_sections = _update_session_panel(
        final_state or {}, counter, time.time() - started, turn_guard
    )
    cl.user_session.set("side_props", _build_side_props(final_state or {}, this_turn_sections))
    await _safe_render_sidebar()

    # Now stream the grounded answer in a SEPARATE message, which lands beneath
    # the trace message. It carries the AnswerTools element (copy / download /
    # clickable, closable source viewer).
    messages = (final_state or {}).get("messages") or []
    answer = messages[-1].content if messages else ""
    chunks = (final_state or {}).get("chunks") or []
    display = _strip_sources_line(answer) or answer or "_(no answer)_"

    turn_id = uuid.uuid4().hex
    _store_turn(turn_id, display, chunks)

    element = cl.CustomElement(
        name="AnswerTools",
        props=_answer_tools_props(display, chunks, turn_id),
    )
    reply = cl.Message(content="", elements=[element])
    await reply.send()
    await _stream_text(reply, display)
    reply.content = display
    await reply.update()


async def _stream_text(msg: cl.Message, text: str, chunk_size: int = 10) -> None:
    """Replay-stream a finished answer into a message for a smooth typing feel.

    The answer is generated while the steps render, but the message bubble is
    created afterwards (so it sits below the logs). We stream the text in small
    slices to preserve the live-typing effect the user expects.
    """
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        await msg.stream_token(text[i : i + chunk_size])
        await asyncio.sleep(0.012)


def _steps_for(node: str, payload: dict, ctx: dict) -> list[tuple[str, str]]:
    """Map a node's output to zero or more (step title, markdown body) pairs."""
    if node == "guard":
        if payload.get("blocked"):
            return [(
                "🛡️ Guard — blocked",
                f"Input flagged as **{payload.get('block_reason', 'unsafe')}** and "
                "routed to a safe refusal.",
            )]
        return [("🛡️ Guard — passed", "No injection patterns detected.")]
    if node == "agent":
        last = (payload.get("messages") or [None])[-1]
        calls = getattr(last, "tool_calls", None) or []
        steps: list[tuple[str, str]] = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            if name == "search_guide_parallel":
                qs = args.get("queries") or []
                listing = "\n".join(f"{i}. `{q}`" for i, q in enumerate(qs, start=1))
                steps.append((
                    "🧰 Agent — parallel multi-topic search",
                    f"Detected **{len(qs)}** distinct topics; searching them in "
                    f"parallel:\n\n{listing}",
                ))
            else:
                q = args.get("query", "")
                steps.append(("🧠 Agent — calling search_guide", f"Looking up: `{q}`"))
        return steps  # the final answer (no tool calls) streams into the message
    if node == "tools":
        runs = payload.get("tool_runs") or []
        new = runs[ctx.get("runs", 0):]
        ctx["runs"] = len(runs)
        return [("🔎 " + _tool_title(run), _tool_body(run)) for run in new]
    if node == "verify":
        return _verify_steps(payload.get("faithfulness") or {})
    if node == "memory":
        if payload.get("summary"):
            return [(
                "🧠 Long-term memory updated",
                "Folded older turns into the rolling summary:\n\n> "
                + payload["summary"].strip(),
            )]
        return []
    return []


def _tool_title(run: dict) -> str:
    prefix = "↳ " if run.get("batch") else ""
    return f"{prefix}search_guide(\"{run.get('query', '')}\")"


def _tool_body(run: dict) -> str:
    if run.get("error"):
        return "The search failed; treated as no results."
    lines: list[str] = []
    sections = run.get("sections") or []
    if run.get("hierarchical") and sections:
        channel = "hybrid (dense + BM25 → RRF)" if run.get("coarse_hybrid") else "dense"
        listing = ", ".join(f"{title} ({sim:.2f})" for _id, title, sim in sections)
        lines.append(f"**Coarse ({channel}) →** top sections: {listing}")
    else:
        lines.append("**Flat search** (section index unavailable).")
    if run.get("fell_back"):
        lines.append("_No chunks within those sections; fell back to a full search._")
    n = run.get("n", 0)
    if n:
        lines.append(f"**Fine (hybrid + rerank) →** {n} grounded passage(s).")
    else:
        lines.append("No passage cleared the grounding threshold.")
    return "\n\n".join(lines)


def _verify_steps(fth: dict) -> list[tuple[str, str]]:
    if not fth:
        return []
    score = fth.get("score", 0.0)
    if fth.get("label") == "grounded":
        return [("✅ Faithfulness check", f"Answer is grounded in the cited passages (score {score:.2f}).")]
    unsupported = fth.get("unsupported") or []
    detail = ("\n\nPossibly unsupported: " + "; ".join(unsupported)) if unsupported else ""
    return [(
        "⚠️ Faithfulness check",
        f"Some claims may not be fully supported (score {score:.2f}).{detail}",
    )]


def _strip_sources_line(answer: str) -> str:
    """Remove every "Sources:" line from the visible answer.

    Citations are mandatory, but rendering them as raw text reads like part of
    the answer. We surface them instead as compact reference chips (the
    ``Sources`` custom element) that open the passage on click, grouped by
    topic. Multi-topic answers may carry one "Sources:" line per section, so we
    strip them wherever they appear (not just the trailing one) and collapse the
    blank lines that are left behind. Inline citations like "(p. 80)" within the
    prose are left untouched.
    """
    kept: list[str] = []
    for line in (answer or "").splitlines():
        if line.strip().lower().startswith("sources:"):
            continue
        kept.append(line)
    # Collapse 3+ consecutive blank lines (left by removed source lines) into one.
    cleaned: list[str] = []
    blanks = 0
    for line in kept:
        if line.strip():
            blanks = 0
            cleaned.append(line)
        else:
            blanks += 1
            if blanks <= 1:
                cleaned.append(line)
    return "\n".join(cleaned).rstrip()


def _short_section(section: str, limit: int = 32) -> str:
    """Trim a section title for a compact reference chip label."""
    section = (section or "General").strip()
    return section if len(section) <= limit else section[: limit - 1].rstrip() + "…"


def _chunk_scoring(chunk, final_rank: int) -> dict:
    """Structured, labelled relevance metrics for one passage.

    Organised into the three things a reviewer actually wants to read:
    - the **final rank** (position after Cohere rerank) + its rerank score,
    - the **hybrid-fusion rank** (position after dense+BM25 RRF, before rerank),
    - the raw **signals** (dense cosine, BM25 lexical score).
    """
    def fmt(val, spec):
        return format(val, spec) if val is not None else None

    fused = getattr(chunk, "fused_score", None)
    prerank = getattr(chunk, "prerank_rank", None)
    rerank = chunk.rerank_score
    dense = chunk.dense_score
    bm25 = chunk.sparse_score
    return {
        "finalRank": final_rank,
        "rerank": fmt(rerank, ".2f"),
        "preRank": prerank,
        "rrf": fmt(fused, ".4f"),
        "dense": fmt(dense, ".2f"),
        "bm25": fmt(bm25, ".1f"),
        # Fallback single score when no individual signal is available.
        "score": fmt(chunk.score, ".2f")
        if (rerank is None and dense is None and bm25 is None)
        else None,
    }


def _answer_tools_props(answer: str, chunks: list, turn_id: str) -> dict:
    """Build the props for the ``AnswerTools`` custom element.

    Citations are grouped by the originating search query (so a multi-topic
    answer shows one cluster of chips per topic). Each item carries everything
    the in-message passage viewer needs — text, same-section context, scores —
    so nothing has to round-trip to the server to open a passage.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        topic = (getattr(chunk, "origin_query", "") or "").strip()
        if topic not in groups:
            groups[topic] = {"topic": topic, "items": []}
            order.append(topic)
        parent = (getattr(chunk, "parent_text", "") or "").strip()
        text = (chunk.text or "").strip()
        groups[topic]["items"].append(
            {
                "index": i,
                "page": chunk.page,
                "section": _short_section(chunk.section),
                "scoring": _chunk_scoring(chunk, i),
                "text": text,
                "parent": parent if parent and parent != text else None,
            }
        )
    return {
        "answer": answer,
        "turnId": turn_id,
        "hasSources": bool(chunks),
        "groups": [groups[t] for t in order],
    }


# --- Per-turn store + PDF export action -------------------------------------

_MAX_STORED_TURNS = 12


def _store_turn(turn_id: str, answer: str, chunks: list) -> None:
    """Remember a turn's answer + chunks so download actions can rebuild PDFs."""
    turns = cl.user_session.get("answer_turns") or {}
    turns[turn_id] = {"answer": answer, "chunks": chunks}
    # Keep only the most recent turns so a long session can't grow unbounded.
    if len(turns) > _MAX_STORED_TURNS:
        for stale in list(turns)[: len(turns) - _MAX_STORED_TURNS]:
            turns.pop(stale, None)
    cl.user_session.set("answer_turns", turns)


@cl.action_callback("export_pdf")
async def export_pdf(action: cl.Action) -> None:
    """Build a styled PDF for the answer / all sources / a single passage.

    Triggered from the ``AnswerTools`` element via ``callAction``. The payload
    carries ``kind`` ("answer" | "sources" | "chunk"), the ``turnId`` to look
    up, and (for "chunk") the 1-based passage ``index``.
    """
    payload = action.payload or {}
    kind = payload.get("kind")
    turn = (cl.user_session.get("answer_turns") or {}).get(payload.get("turnId"))
    if not turn:
        await cl.Message(content="That answer is no longer available to export.").send()
        return
    try:
        if kind == "answer":
            path, label = build_answer_pdf(turn["answer"]), "answer"
        elif kind == "sources":
            path, label = build_chunks_pdf(turn["chunks"]), "cited sources"
        elif kind == "chunk":
            idx = int(payload.get("index") or 0)
            chunks = turn["chunks"]
            if not (1 <= idx <= len(chunks)):
                await cl.Message(content="That passage is no longer available.").send()
                return
            path = build_chunk_pdf(chunks[idx - 1], idx)
            label = f"passage [{idx}]"
        else:
            return
        await cl.Message(
            content=f"Here is the {label} as a PDF.",
            elements=[cl.File(name=Path(path).name, path=path, display="inline")],
        ).send()
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF export failed: %s", exc)
        await cl.Message(content=f"Sorry, the PDF export failed.\n\n```\n{exc}\n```").send()


def _metrics_steps(started: float, counter: TokenCounter, state: dict) -> list[tuple[str, str]]:
    """Build the per-turn metrics entry (latency, tokens, grounding) for the trace."""
    if not get_settings().show_metrics:
        return []
    latency = time.time() - started
    parts = [f"⏱️ {latency:.1f}s", f"🧮 {counter.llm_calls} LLM calls"]
    if counter.total_tokens:
        parts.append(
            f"🔢 {counter.total_tokens:,} tokens "
            f"({counter.input_tokens:,} in / {counter.output_tokens:,} out)"
        )
    fth = state.get("faithfulness") or {}
    if fth:
        icon = "✅" if fth.get("label") == "grounded" else "⚠️"
        parts.append(f"{icon} grounding {fth.get('score', 0.0):.2f}")
    return [("📊 Turn metrics", "\n".join(parts))]

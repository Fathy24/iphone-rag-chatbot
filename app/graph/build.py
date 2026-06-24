"""Assemble and compile the tool-calling RAG agent graph.

Flow::

    START
      -> guard                (block obvious injection attempts)
        -> refuse -> END           (if blocked: safe, in-scope refusal)
        -> agent                   (LLM picks a tool: reply / search / parallel search)
            -> tools -> agent      (run the chosen search tool; loop back with results)
            -> verify              (faithfulness check on the final answer)
                -> memory -> END   (fold older turns into the rolling summary)

Retrieval is a genuine **tool the model chooses to call**. Greetings, small-talk
and out-of-scope requests are answered directly (no tool call, no retrieval); a
single topic uses ``search_guide``; several distinct topics use
``search_guide_parallel`` (retrieved concurrently, grouped); and weak results are
handled by the agent re-querying — so there are no separate decompose/CRAG/HyDE
stages.

Memory is two-tier: an in-memory checkpointer keyed by a per-session
``thread_id`` keeps the recent message window (short-term), and the ``memory``
node maintains a rolling summary of older turns (long-term) — so context
survives long conversations without unbounded prompt growth and without any
external database.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.graph.nodes import (
    TOOLS,
    make_agent_node,
    make_faithfulness_node,
    make_guard_node,
    make_memory_node,
    make_refusal_node,
    make_tools_node,
    route_after_agent,
    route_after_guard,
)
from app.graph.state import RagState
from app.llm.clients import build_chat_model
from app.logging_config import get_logger
from app.observability import configure_tracing
from app.rag.retriever import Retriever

logger = get_logger(__name__)


def build_rag_graph(settings: Settings | None = None):
    """Construct and compile the LangGraph agent application.

    Heavy dependencies (chat model, hierarchical retriever, reranker) are created
    once here and shared by all turns. Raises if the index/credentials are not
    usable, so the container fails fast at startup rather than per request.

    Returns:
        A compiled LangGraph runnable with an in-memory checkpointer.
    """
    settings = settings or get_settings()
    settings.require_runtime_secrets()
    configure_tracing(settings)

    chat_model = build_chat_model(settings)
    model_with_tools = chat_model.bind_tools(TOOLS)
    retriever = Retriever(settings)

    graph = StateGraph(RagState)
    graph.add_node("guard", make_guard_node())
    graph.add_node("agent", make_agent_node(model_with_tools, settings))
    graph.add_node("tools", make_tools_node(retriever, settings))
    graph.add_node("verify", make_faithfulness_node(chat_model, settings))
    graph.add_node("memory", make_memory_node(chat_model, settings))
    graph.add_node("refuse", make_refusal_node())

    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"refuse": "refuse", "agent": "agent"},
    )
    # Agent loop: either run the requested tools (then re-enter the agent) or
    # finish and verify/remember.
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "verify": "verify"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("verify", "memory")
    graph.add_edge("memory", END)
    graph.add_edge("refuse", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    logger.info("RAG agent graph compiled and ready.")
    return compiled

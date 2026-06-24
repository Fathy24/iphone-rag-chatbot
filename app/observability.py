"""Observability: LangSmith tracing wiring and per-turn token accounting.

Two lightweight, dependency-free pieces:

* :func:`configure_tracing` opts the process into LangSmith tracing by exporting
  the standard ``LANGCHAIN_*`` environment variables when enabled. LangChain /
  LangGraph then trace every LLM, retriever, and node call automatically — no
  code changes in the pipeline. It is a no-op (and never raises) when disabled or
  unconfigured.
* :class:`TokenCounter` is a callback handler that accumulates token usage and
  call counts across *all* LLM calls in a single graph turn (contextualize,
  plan, branch summaries, grading, generation, faithfulness, memory), so the UI
  can show a per-turn cost/latency badge.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_TRACING_CONFIGURED = False


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing via environment variables when configured.

    Safe to call multiple times; only acts once. Never raises — observability
    must never break the application.
    """
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return
    _TRACING_CONFIGURED = True
    try:
        if not settings.langchain_tracing or not settings.langchain_api_key:
            return
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info("LangSmith tracing enabled (project=%s).", settings.langchain_project)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not configure tracing: %s", exc)


class TokenCounter(BaseCallbackHandler):
    """Accumulates token usage and LLM call counts for one turn.

    Besides the running totals (summed across every LLM call in the turn), we
    also track ``peak_input_tokens`` — the largest single prompt sent to the
    model. That peak is the best proxy for how *full* the context window got on
    this turn (the totals over-count because a tool-calling turn makes several
    calls), so the UI's context meter uses it.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.peak_input_tokens = 0
        self.llm_calls = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Pull token usage from whichever field the provider populated."""
        self.llm_calls += 1
        try:
            call_input = 0
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            if usage:
                call_input = int(usage.get("prompt_tokens", 0) or 0)
                self.input_tokens += call_input
                self.output_tokens += int(usage.get("completion_tokens", 0) or 0)
            else:
                # Fallback: aggregate per-generation usage_metadata (newer clients).
                for batch in getattr(response, "generations", []) or []:
                    for gen in batch:
                        message = getattr(gen, "message", None)
                        meta = getattr(message, "usage_metadata", None) or {}
                        call_input += int(meta.get("input_tokens", 0) or 0)
                        self.output_tokens += int(meta.get("output_tokens", 0) or 0)
                self.input_tokens += call_input
            if call_input > self.peak_input_tokens:
                self.peak_input_tokens = call_input
        except Exception:  # noqa: BLE001 - accounting must never break a turn
            pass

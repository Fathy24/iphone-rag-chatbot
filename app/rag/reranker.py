"""Cross-encoder reranking via the gateway's Cohere Rerank endpoint.

After fusion we have a strong *candidate* set, but lexical/semantic rankers only
score query-vs-document independently. A cross-encoder reranker jointly attends
to the query and each passage, giving a far more accurate final ordering — and a
calibrated relevance score we use as the grounding gate (refuse when even the
best passage scores low).

We call the same OpenAI-compatible gateway used for chat/embeddings (Cohere
Rerank 3.5, ``bedrock.cohere.rerank-3-5``), so no extra credentials are needed.
If no gateway is configured (e.g. a reviewer using the public OpenAI API) the
reranker disables itself and retrieval falls back to fused/dense ordering.
"""

from __future__ import annotations

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Transient errors worth retrying (network blips, gateway 5xx/429).
_RETRYABLE = (requests.Timeout, requests.ConnectionError, requests.HTTPError)


class RerankerUnavailable(Exception):
    """Permanent failure (e.g. no rerank endpoint / auth denied) — do not retry."""


class Reranker:
    """Thin client over the gateway's Cohere-style ``/rerank`` endpoint.

    The reranker is *best-effort*: we don't know whether the deployment running
    this app actually exposes a rerank endpoint (a reviewer may point us at the
    public OpenAI API, which has none). So if a call fails, we catch it, latch
    the reranker **off for the rest of the session**, and continue exactly as if
    reranking were disabled — retrieval simply keeps the fused/dense ordering
    and the grounding gate falls back to the dense-similarity threshold. This
    avoids paying the retry/latency cost on every subsequent turn.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        base = (self._settings.openai_base_url or "").rstrip("/")
        self._url = f"{base}{self._settings.rerank_endpoint_path}" if base else ""
        # Set once a call fails; from then on we behave as if disabled.
        self._disabled = False

    @property
    def available(self) -> bool:
        """True when reranking is enabled, configured, and not latched off."""
        return bool(self._settings.use_reranker and self._url and not self._disabled)

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Rerank ``documents`` against ``query``.

        Args:
            query: The user query.
            documents: Candidate passage texts (already retrieved/fused).

        Returns:
            ``(original_index, relevance_score)`` pairs sorted by descending
            relevance. Returns an empty list if reranking is unavailable or the
            call fails (callers then keep the pre-rerank order).
        """
        if not self.available or not documents:
            return []
        payload = {
            "model": self._settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        try:
            results = self._post_with_retry(payload)
            ranked = [
                (int(item["index"]), float(item["relevance_score"]))
                for item in results
                if "index" in item and "relevance_score" in item
            ]
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            logger.info("Reranked %d candidates via %s", len(ranked), self._settings.rerank_model)
            return ranked
        except Exception as exc:  # noqa: BLE001 - never let reranking break a turn
            # Latch off for the rest of the session: no rerank endpoint here, or
            # it's failing. Subsequent turns skip reranking entirely (treated as
            # disabled) instead of retrying and adding latency every turn.
            self._disabled = True
            logger.warning(
                "Reranker disabled for this session (%s); continuing without "
                "reranking — keeping fused order and using the dense grounding "
                "threshold.",
                exc,
            )
            return []

    def _post_with_retry(self, payload: dict) -> list[dict]:
        """POST to the rerank endpoint with exponential-backoff retries.

        Wrapped in a per-call tenacity policy so the retry budget tracks the
        current settings. Raises after the budget is exhausted (the caller
        catches and falls back to the fused order).
        """
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        @retry(
            stop=stop_after_attempt(max(1, self._settings.rerank_max_retries)),
            wait=wait_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        def _call() -> list[dict]:
            response = requests.post(self._url, headers=headers, json=payload, timeout=30)
            # Permanent client errors (no rerank endpoint / bad auth / wrong
            # path) shouldn't be retried — fail fast so we latch off immediately
            # rather than burning the retry budget on every turn.
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                raise RerankerUnavailable(
                    f"rerank endpoint returned HTTP {response.status_code}"
                )
            response.raise_for_status()
            return response.json().get("results", [])

        return _call()

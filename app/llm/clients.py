"""Factories for the chat and embedding models.

Both the ingestion pipeline and the serving path build their model clients here
so that credential resolution and model configuration live in exactly one place
(mirroring the single ``resolve_credentials`` pattern used in the main backend).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def build_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """Construct the chat model used to generate grounded answers.

    Temperature defaults to 0 for deterministic, faithful question answering.

    Args:
        settings: Optional settings override (defaults to the cached instance).

    Returns:
        A configured :class:`~langchain_openai.ChatOpenAI` client.
    """
    settings = settings or get_settings()
    logger.info("Initialising chat model: %s", settings.chat_model)
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Construct the embedding model shared by ingestion and retrieval.

    The same model and dimensionality MUST be used at ingestion and query time,
    otherwise vectors are not comparable. ``embedding_dim`` is passed explicitly
    so the value is asserted against the configured vector index on startup.

    Args:
        settings: Optional settings override (defaults to the cached instance).

    Returns:
        A configured :class:`~langchain_openai.OpenAIEmbeddings` client.
    """
    settings = settings or get_settings()
    logger.info(
        "Initialising embeddings: %s (dim=%d)",
        settings.embedding_model,
        settings.embedding_dim,
    )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )

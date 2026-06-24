"""Centralised, validated application configuration.

All runtime configuration is sourced from environment variables (loaded from a
local ``.env`` during development, or injected via ``--env-file`` in Docker).
Using a single typed settings object means the rest of the codebase never reads
``os.environ`` directly, which keeps secret handling auditable and testable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read (in order of precedence) from real environment variables,
    then from a ``.env`` file. Unknown variables are ignored so the same file
    can be shared with tooling that needs extra keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider --------------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI (or compatible) API key.")
    openai_base_url: str | None = Field(
        default=None, description="Optional OpenAI-compatible base URL (Azure/gateway)."
    )
    chat_model: str = Field(default="gpt-4o")
    chat_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dim: int = Field(default=3072, gt=0)

    # --- Vector backend selection --------------------------------------------
    # "qdrant" = Qdrant Cloud (managed); auto-falls back to local FAISS if the
    #            cluster is unreachable at startup.
    # "faiss"  = local on-disk index (no signup/network; for offline testing and
    #            as the cloud fallback).
    vector_backend: str = Field(default="qdrant")
    # Directory where the local FAISS index is saved/loaded (faiss backend, and
    # the automatic fallback when Qdrant is unreachable).
    faiss_index_path: str = Field(default=".faiss_index")
    # If Qdrant can't be reached at startup, fall back to the local FAISS index
    # when one is present instead of failing the app.
    cloud_fallback_to_faiss: bool = Field(default=True)

    # --- Vector database (Qdrant Cloud) --------------------------------------
    # Cluster REST endpoint, e.g. https://<id>.<region>.gcp.cloud.qdrant.io:6333
    qdrant_url: str = Field(default="", description="Qdrant cluster URL.")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key.")
    qdrant_collection: str = Field(default="iphone-user-guide")

    # --- Retrieval (dense, OpenAI embeddings) --------------------------------
    retrieval_top_k: int = Field(default=10, gt=0)
    retrieval_fetch_k: int = Field(default=40, gt=0)
    # Minimum cosine similarity for a chunk to count as relevant (used as the
    # grounding gate in dense-only mode, or when the reranker is unavailable).
    score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # --- Hierarchical (coarse-to-fine) retrieval -----------------------------
    # The guide has clear chapters/sections. We retrieve in two stages: first
    # pick the top-N most relevant SECTIONS (coarse, via section centroids),
    # then run hybrid chunk retrieval restricted to those sections (fine). This
    # locks onto the right topic before precise chunk matching, which sharply
    # reduces cross-chapter false positives.
    enable_hierarchical: bool = Field(default=True)
    coarse_sections_n: int = Field(default=5, ge=1, le=20)
    # Sparse channel of the coarse stage: rank each section by the SUM of its
    # top-K child BM25 scores (robust "MaxP+" aggregation). 1 = rank by the
    # single best child only (pure MaxP); higher aggregates a few strong hits
    # without diluting across the whole chapter.
    coarse_sparse_topk: int = Field(default=3, ge=1, le=20)
    # On-disk section index (centroids + titles) built during ingestion.
    sections_path: str = Field(default=".sections.json")

    # --- Hybrid retrieval (dense + BM25 sparse, fused with weighted RRF) ------
    # "hybrid" = dense + BM25 fused; "dense" = embeddings only.
    retrieval_mode: str = Field(default="hybrid")
    # Reciprocal Rank Fusion constant and per-channel weights (mirrors backend).
    rrf_k: int = Field(default=60, gt=0)
    rrf_weight_dense: float = Field(default=1.0, ge=0.0)
    rrf_weight_sparse: float = Field(default=1.0, ge=0.0)
    # On-disk corpus the BM25 index is rebuilt from at serve time.
    bm25_corpus_path: str = Field(default=".bm25_corpus.json")

    # --- Reranking (Cohere Rerank via the OpenAI-compatible gateway) ---------
    use_reranker: bool = Field(default=True)
    rerank_model: str = Field(default="bedrock.cohere.rerank-3-5")
    rerank_endpoint_path: str = Field(default="/rerank")
    # Over-fetch this multiple of top_k before reranking down to top_k.
    rerank_top_n_multiplier: float = Field(default=3.0, ge=1.0)
    # Minimum reranker relevance for a chunk to be kept (grounding gate when the
    # reranker is active). Below this for all candidates -> honest refusal.
    rerank_score_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # --- Tool-calling agent --------------------------------------------------
    # The model decides, per turn, which tool to use:
    #   * search_guide(query)            -> one focused topic
    #   * search_guide_parallel(queries) -> several distinct topics at once
    # It greets / refuses out-of-scope directly (no retrieval). This cap bounds
    # the agent<->tools loop so a misbehaving turn can't run away.
    agent_max_tool_calls: int = Field(default=6, ge=1, le=20)
    # Thread-pool size for the parallel multi-query tool (bounded fan-out).
    retrieval_max_workers: int = Field(default=4, ge=1, le=16)
    # Hard cap on how many sub-queries a single parallel call will run.
    max_parallel_queries: int = Field(default=6, ge=1, le=12)

    # --- Faithfulness / grounding verification ------------------------------
    # After generation, verify the answer's claims are supported by the cited
    # context (the "trustworthy AI" check). Surfaced as a UI badge.
    enable_faithfulness_check: bool = Field(default=True)
    faithfulness_min_score: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- Parent-document (small-to-big) context expansion -------------------
    # Expand each retrieved chunk with its neighbours in the SAME section (in
    # reading order) so the model sees coherent surrounding context. Citations
    # stay chunk-precise; only the context shown to the model grows.
    enable_parent_expansion: bool = Field(default=True)
    parent_window: int = Field(default=1, ge=0, le=4)

    # --- Observability (LangSmith tracing + per-turn metrics) ---------------
    langchain_tracing: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="iphone-rag-chatbot")
    langchain_endpoint: str = Field(default="")
    # Show a per-turn latency/token badge in the UI.
    show_metrics: bool = Field(default=True)

    # --- LLM client robustness ----------------------------------------------
    llm_timeout: int = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    # Retry budget for the reranker HTTP call (transient network/5xx errors).
    rerank_max_retries: int = Field(default=3, ge=0)

    # --- Long-term memory (rolling conversation summary) ---------------------
    enable_summary_memory: bool = Field(default=True)
    # Start folding older turns into a running summary once the conversation
    # exceeds this many messages (keeps prompts bounded while retaining context).
    summary_after_messages: int = Field(default=8, ge=2)
    summary_max_tokens: int = Field(default=400, gt=0)

    # --- Ingestion -----------------------------------------------------------
    pdf_path: str = Field(default="data/iphone_user_guide.pdf")
    # Chunking strategy: "semantic" (embedding-similarity splits) or "token".
    chunk_strategy: str = Field(default="semantic")
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    # Semantic chunker parameters (see app/rag/semantic_chunker.py).
    semantic_window_size: int = Field(default=2, ge=1)
    semantic_threshold_percentile: int = Field(default=30, ge=5, le=95)
    semantic_min_tokens: int = Field(default=120, gt=0)
    semantic_max_tokens: int = Field(default=512, gt=0)

    # --- Runtime -------------------------------------------------------------
    port: int = Field(default=8000, gt=0, lt=65536)
    # Recent conversational turns kept verbatim in the prompt. Older turns are
    # folded into the rolling summary (see enable_summary_memory).
    max_history_messages: int = Field(default=8, ge=0)
    log_level: str = Field(default="INFO")

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _blank_base_url_is_none(cls, value: object) -> object:
        """Treat an empty ``OPENAI_BASE_URL`` as "use the default endpoint"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("chunk_strategy", "vector_backend", "retrieval_mode", mode="before")
    @classmethod
    def _normalise_choice(cls, value: object) -> object:
        """Lower-case and trim choice-style settings for robust comparisons."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def require_runtime_secrets(self) -> None:
        """Validate that secrets needed to *serve* requests are present.

        Called at application startup so that a misconfigured container fails
        fast with a clear message instead of producing confusing errors on the
        first user query.

        Raises:
            RuntimeError: If any mandatory runtime secret is missing.
        """
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        # Qdrant credentials are only required for the cloud backend.
        if self.vector_backend == "qdrant":
            if not self.qdrant_url:
                missing.append("QDRANT_URL")
            if not self.qdrant_api_key:
                missing.append("QDRANT_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". See .env.example for the full list."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings()

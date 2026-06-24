"""Hierarchical (coarse-to-fine) hybrid retrieval.

The iPhone User Guide is organised into clear chapters/sections, so we retrieve
in two stages instead of searching every chunk flatly:

1. **Coarse — pick the right sections (hybrid).** Two signals are fused:
   * dense: query embedding vs. per-section *centroids* (mean of each section's
     chunk embeddings, :mod:`app.rag.hierarchy`);
   * sparse: sections ranked by their best BM25 chunk hit;
   combined with weighted RRF, and the top-N sections are kept. This locks onto
   the correct topic first and avoids pulling a lexically-similar chunk from an
   unrelated chapter, while still surfacing sections with strong keyword hits.
2. **Fine — precise chunk retrieval inside those sections.** Restricted to the
   chosen sections, we run hybrid search:
   * dense similarity over OpenAI embeddings (Qdrant filter / FAISS post-filter);
   * BM25 sparse search over the chunk corpus;
   then fuse with weighted Reciprocal Rank Fusion (:mod:`app.rag.fusion`),
   rerank the top fused candidates with a Cohere cross-encoder, and apply a
   relevance **grounding gate** (below threshold → honest refusal). Survivors
   are widened with same-section neighbours for coherent context while citations
   stay chunk-precise.

If the section index is unavailable (or hierarchical mode is off) the pipeline
degrades gracefully to flat hybrid retrieval over all chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.llm.clients import build_embeddings
from app.logging_config import get_logger
from app.rag.bm25 import SparseIndex, load_sparse_index
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.hierarchy import (
    SectionRouter,
    build_section_records_from_metas,
    load_section_router,
)
from app.rag.parent import ParentExpander, load_parent_expander
from app.rag.reranker import Reranker
from app.rag.vector_store import get_dense_store, load_corpus_from_qdrant

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk plus its relevance scores and citation fields."""

    text: str
    page: int
    section: str
    source: str
    score: float
    chunk_id: str = ""
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    # Fused RRF score and 1-based pre-rerank position (the "overall rank" formed
    # by combining dense + BM25 before the cross-encoder reorders the pool).
    fused_score: float | None = None
    prerank_rank: int | None = None
    # Same-section neighbour context for the model (parent-document expansion);
    # the citation still points at this chunk's own page/section.
    parent_text: str | None = None
    # The search-tool query that first surfaced this chunk (set by the tools
    # node) so the UI can group sources by topic. First query wins.
    origin_query: str = ""

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. ``"p. 42 — Chapter 4: Siri"``."""
        return f"p. {self.page} — {self.section}"

    @property
    def context_text(self) -> str:
        """Text to feed the model: expanded parent context when available."""
        return self.parent_text or self.text


@dataclass
class RetrievalInfo:
    """Telemetry for one retrieval call, surfaced in the UI step view."""

    sections: list[tuple[str, str, float]] = field(default_factory=list)
    hierarchical: bool = False
    coarse_hybrid: bool = False
    n_candidates: int = 0
    n_grounded: int = 0
    fell_back: bool = False


@dataclass
class _Candidate:
    """Internal accumulator for a chunk seen across channels."""

    chunk_id: str
    text: str
    page: int
    section: str
    source: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    # 1-based position in the fused (pre-rerank) ordering.
    prerank_rank: int | None = None


class Retriever:
    """Coarse-to-fine hybrid retrieval with RRF fusion and reranking."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._dense = get_dense_store(self._settings)
        self._embeddings = build_embeddings(self._settings)
        self._reranker = Reranker(self._settings)
        self._sparse: SparseIndex | None = None
        self._router: SectionRouter | None = None
        self._parent: ParentExpander | None = None
        # The sparse index, section centroids and parent-expander are built from
        # the chunk corpus. With the Qdrant backend that corpus is streamed from
        # the cloud collection at startup (the single source of truth — no local
        # index files); with the local FAISS backend it is read from the on-disk
        # files written by ingestion.
        if self._settings.vector_backend == "qdrant":
            self._init_corpus_from_cloud()
        else:
            self._init_corpus_from_files()
        logger.info(
            "Retriever ready (mode=%s, source=%s, hierarchical=%s, sparse=%s, reranker=%s, parent=%s)",
            self._settings.retrieval_mode,
            "qdrant" if self._settings.vector_backend == "qdrant" else "files",
            self._router is not None,
            self._sparse is not None,
            self._reranker.available,
            self._parent is not None,
        )

    def _init_corpus_from_files(self) -> None:
        """Build sparse/coarse/parent helpers from on-disk files (FAISS dev)."""
        s = self._settings
        if s.retrieval_mode == "hybrid":
            self._sparse = load_sparse_index(s.bm25_corpus_path)
        if s.enable_hierarchical:
            self._router = load_section_router(s.sections_path)
        if s.enable_parent_expansion:
            self._parent = load_parent_expander(s.bm25_corpus_path)

    def _init_corpus_from_cloud(self) -> None:
        """Rebuild sparse/coarse/parent helpers from the Qdrant collection.

        One ``scroll`` over the (small) collection yields every chunk's payload
        and dense vector, from which we rebuild the BM25 index, the section
        centroids and the parent-expander in memory. If the scroll fails the
        retriever degrades to dense-only (helpers stay ``None``).
        """
        s = self._settings
        records, vectors = load_corpus_from_qdrant(s)
        if not records:
            return
        if s.retrieval_mode == "hybrid":
            self._sparse = SparseIndex(records)
        if s.enable_hierarchical:
            paired = [(r, v) for r, v in zip(records, vectors) if v is not None]
            if paired:
                section_records = build_section_records_from_metas(
                    [r for r, _ in paired], [v for _, v in paired]
                )
                self._router = SectionRouter(section_records)
        if s.enable_parent_expansion:
            self._parent = ParentExpander(records)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_reranker: bool | None = None,
        coarse_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for ``query`` (chunks only)."""
        chunks, _info = self.retrieve_with_info(
            query, top_k=top_k, use_reranker=use_reranker, coarse_n=coarse_n
        )
        return chunks

    def retrieve_with_info(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_reranker: bool | None = None,
        coarse_n: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalInfo]:
        """Retrieve chunks and return retrieval telemetry alongside them.

        Args:
            query: The search query (one focused topic).
            top_k: Override the number of child chunks returned (UI slider).
            use_reranker: Override whether to apply the Cohere reranker.
            coarse_n: Override how many top sections the coarse stage keeps.

        Returns:
            ``(chunks, info)`` — chunks is empty when nothing clears the
            grounding gate (which the agent turns into an honest refusal).
        """
        query = (query or "").strip()
        info = RetrievalInfo()
        if not query:
            return [], info

        final_k = top_k or self._settings.retrieval_top_k
        do_rerank = self._settings.use_reranker if use_reranker is None else use_reranker

        # Coarse stage: choose the most relevant sections (hybrid).
        section_ids = self._coarse_sections(query, info, coarse_n)

        candidates, dense_ranks, sparse_ranks = self._gather(query, section_ids)
        # If section filtering starved retrieval, retry flat (resilience).
        if not candidates and section_ids:
            logger.info("No candidates within top sections; retrying flat.")
            info.fell_back = True
            candidates, dense_ranks, sparse_ranks = self._gather(query, None)
        if not candidates:
            return [], info

        ordered = self._order_candidates(candidates, dense_ranks, sparse_ranks)
        # Record the fused (pre-rerank) position so the UI can show the overall
        # rank that dense + BM25 produced before the cross-encoder reorders.
        for position, cand in enumerate(ordered, start=1):
            cand.prerank_rank = position
        pool = ordered[: self._rerank_pool_size()]
        if do_rerank:
            pool = self._apply_reranker(query, pool)
        gated = self._apply_grounding_gate(pool)

        info.n_candidates = len(candidates)
        info.n_grounded = len(gated)
        logger.info(
            "Retrieval: %d candidates -> %d pooled -> %d grounded (hier=%s)",
            len(candidates),
            len(pool),
            len(gated),
            info.hierarchical,
        )
        results = [self._to_chunk(c) for c in gated[:final_k]]
        self._attach_parent_context(results)
        return results, info

    # -- coarse stage (hybrid: dense centroids + aggregated BM25) --------------

    def _coarse_sections(
        self, query: str, info: RetrievalInfo, coarse_n: int | None
    ) -> set[str] | None:
        """Return the top-N section ids for the query (or ``None`` if flat).

        Dense channel = query-vs-section-centroid cosine. Sparse channel =
        sections ranked by their best BM25 chunk hit. The two rankings are fused
        with weighted RRF, mirroring the fine stage, so a section with strong
        keyword hits is selected even if its centroid is only moderately similar.
        """
        if self._router is None or self._router.size == 0:
            return None
        n = coarse_n or self._settings.coarse_sections_n
        try:
            query_vector = self._embeddings.embed_query(query)
            dense_ranked = self._router.rank_sections(query_vector)
        except Exception as exc:  # noqa: BLE001 - degrade to flat retrieval
            logger.warning("Coarse section routing failed (%s); using flat search.", exc)
            return None
        if not dense_ranked:
            return None

        sims = {sid: sim for sid, _title, sim in dense_ranked}
        titles = {sid: title for sid, title, _sim in dense_ranked}
        dense_ranks = {sid: rank for rank, (sid, _t, _s) in enumerate(dense_ranked, start=1)}

        if self._sparse is not None and self._settings.retrieval_mode == "hybrid":
            sparse_ranks = self._sparse_section_ranks(query)
            fused = reciprocal_rank_fusion(
                dense_ranks,
                sparse_ranks,
                k_constant=self._settings.rrf_k,
                weight_dense=self._settings.rrf_weight_dense,
                weight_sparse=self._settings.rrf_weight_sparse,
            )
            order = [sid for sid, _score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
            info.coarse_hybrid = True
        else:
            order = [sid for sid, _t, _s in dense_ranked]

        chosen = order[: max(1, n)]
        info.sections = [(sid, titles.get(sid, sid), sims.get(sid, 0.0)) for sid in chosen]
        info.hierarchical = True
        return set(chosen)

    def _sparse_section_ranks(self, query: str) -> dict[str, int]:
        """Rank sections by the SUM of their top-K child BM25 scores.

        Instead of ranking a section by its single best chunk (pure MaxP), we
        aggregate each section's strongest few child hits
        (``COARSE_SPARSE_TOPK``) and rank by that sum. This is more robust than a
        lone chunk while avoiding the dilution/length-penalty of scoring the
        whole concatenated chapter as one BM25 document. With ``topk=1`` it
        reduces to the previous best-child behaviour.

        Returns a ``{section_id: rank}`` map (1-based, best first) so it slots
        straight into the coarse RRF fusion.
        """
        try:
            raw = self._sparse.query(query, self._settings.retrieval_fetch_k * 4)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Sparse section ranking failed (%s); dense-only coarse.", exc)
            return {}
        topk = max(1, self._settings.coarse_sparse_topk)
        per_section: dict[str, list[float]] = {}
        for record, score in raw:
            section_id = str(record.get("section_id", ""))
            if not section_id:
                continue
            hits = per_section.setdefault(section_id, [])
            if len(hits) < topk:
                hits.append(float(score))
        # Aggregate the top-K hits per section, then rank by descending sum.
        aggregated = {sid: sum(scores) for sid, scores in per_section.items()}
        ordered = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
        return {sid: rank for rank, (sid, _agg) in enumerate(ordered, start=1)}

    # -- fine stage: candidate generation -------------------------------------

    def _gather(
        self, query: str, section_ids: set[str] | None
    ) -> tuple[dict[str, _Candidate], dict[str, int], dict[str, int]]:
        """Run dense (+ sparse) search, restricted to ``section_ids``."""
        candidates: dict[str, _Candidate] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        fetch_k = self._settings.retrieval_fetch_k

        self._gather_dense(query, fetch_k, section_ids, candidates, dense_ranks)
        if self._sparse is not None:
            self._gather_sparse(query, fetch_k, section_ids, candidates, sparse_ranks)
        return candidates, dense_ranks, sparse_ranks

    def _gather_dense(
        self,
        query: str,
        fetch_k: int,
        section_ids: set[str] | None,
        candidates: dict[str, _Candidate],
        dense_ranks: dict[str, int],
    ) -> None:
        try:
            pairs = self._dense.dense_search(query, k=fetch_k, section_ids=section_ids)
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense search failed for %r: %s", query, exc)
            raise RuntimeError(f"Vector search failed: {exc}") from exc

        for rank, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or doc.page_content[:80])
            cand = candidates.get(chunk_id)
            if cand is None:
                cand = _Candidate(
                    chunk_id=chunk_id,
                    text=doc.page_content,
                    page=int(meta.get("page", 0)),
                    section=str(meta.get("section", "General")),
                    source=str(meta.get("source", "document")),
                )
                candidates[chunk_id] = cand
            cand.dense_score = float(score)
            if chunk_id not in dense_ranks or rank < dense_ranks[chunk_id]:
                dense_ranks[chunk_id] = rank

    def _gather_sparse(
        self,
        query: str,
        fetch_k: int,
        section_ids: set[str] | None,
        candidates: dict[str, _Candidate],
        sparse_ranks: dict[str, int],
    ) -> None:
        # Over-fetch then restrict to the selected sections so the sparse channel
        # respects the coarse stage too.
        raw = self._sparse.query(query, fetch_k * 3 if section_ids else fetch_k)
        rank = 0
        for record, bm25 in raw:
            if section_ids and str(record.get("section_id", "")) not in section_ids:
                continue
            rank += 1
            if rank > fetch_k:
                break
            chunk_id = str(record.get("chunk_id") or record.get("text", "")[:80])
            cand = candidates.get(chunk_id)
            if cand is None:
                cand = _Candidate(
                    chunk_id=chunk_id,
                    text=record.get("text", ""),
                    page=int(record.get("page", 0)),
                    section=str(record.get("section", "General")),
                    source=str(record.get("source", "document")),
                )
                candidates[chunk_id] = cand
            cand.sparse_score = max(cand.sparse_score or 0.0, float(bm25))
            if chunk_id not in sparse_ranks or rank < sparse_ranks[chunk_id]:
                sparse_ranks[chunk_id] = rank

    # -- fusion / ordering ----------------------------------------------------

    def _order_candidates(
        self,
        candidates: dict[str, _Candidate],
        dense_ranks: dict[str, int],
        sparse_ranks: dict[str, int],
    ) -> list[_Candidate]:
        """Order candidates by fused score (hybrid) or dense cosine (dense)."""
        if self._sparse is not None and self._settings.retrieval_mode == "hybrid":
            fused = reciprocal_rank_fusion(
                dense_ranks,
                sparse_ranks,
                k_constant=self._settings.rrf_k,
                weight_dense=self._settings.rrf_weight_dense,
                weight_sparse=self._settings.rrf_weight_sparse,
            )
            for chunk_id, score in fused.items():
                if chunk_id in candidates:
                    candidates[chunk_id].fused_score = score
            return sorted(
                candidates.values(),
                key=lambda c: c.fused_score or 0.0,
                reverse=True,
            )
        return sorted(
            candidates.values(),
            key=lambda c: c.dense_score if c.dense_score is not None else -math.inf,
            reverse=True,
        )

    def _rerank_pool_size(self) -> int:
        """Number of fused candidates to send to the reranker (3x over-fetch)."""
        return max(
            self._settings.retrieval_top_k,
            int(self._settings.retrieval_top_k * self._settings.rerank_top_n_multiplier),
        )

    # -- reranking / grounding ------------------------------------------------

    def _apply_reranker(self, query: str, pool: list[_Candidate]) -> list[_Candidate]:
        """Rerank the pool with the cross-encoder; keep order if unavailable."""
        if not pool or not self._reranker.available:
            return pool
        ranked = self._reranker.rerank(query, [c.text for c in pool])
        if not ranked:
            return pool
        reordered: list[_Candidate] = []
        for index, score in ranked:
            if 0 <= index < len(pool):
                pool[index].rerank_score = score
                reordered.append(pool[index])
        return reordered or pool

    def _apply_grounding_gate(self, pool: list[_Candidate]) -> list[_Candidate]:
        """Drop weak candidates so off-topic queries yield an honest refusal.

        Uses the reranker relevance score when available (the strongest signal),
        otherwise the dense cosine similarity.
        """
        reranked = any(c.rerank_score is not None for c in pool)
        if reranked:
            threshold = self._settings.rerank_score_threshold
            return [c for c in pool if (c.rerank_score or 0.0) >= threshold]
        threshold = self._settings.score_threshold
        return [
            c
            for c in pool
            if c.dense_score is not None and c.dense_score >= threshold
        ]

    def _attach_parent_context(self, chunks: list[RetrievedChunk]) -> None:
        """Expand each result with same-section neighbours (parent-document)."""
        if self._parent is None:
            return
        window = self._settings.parent_window
        for chunk in chunks:
            try:
                chunk.parent_text = self._parent.expand(chunk.chunk_id, chunk.text, window)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Parent expansion skipped for %s: %s", chunk.chunk_id, exc)

    @staticmethod
    def _to_chunk(cand: _Candidate) -> RetrievedChunk:
        # Final relevance: reranker score if present, else dense cosine, else fused.
        if cand.rerank_score is not None:
            score = cand.rerank_score
        elif cand.dense_score is not None:
            score = cand.dense_score
        else:
            score = cand.fused_score or 0.0
        return RetrievedChunk(
            text=cand.text,
            page=cand.page,
            section=cand.section,
            source=cand.source,
            score=float(score),
            chunk_id=cand.chunk_id,
            dense_score=cand.dense_score,
            sparse_score=cand.sparse_score,
            rerank_score=cand.rerank_score,
            fused_score=cand.fused_score,
            prerank_rank=cand.prerank_rank,
        )

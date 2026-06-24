"""Reciprocal Rank Fusion (RRF) for combining dense and sparse rankings.

RRF fuses ranked lists by *rank position* rather than raw score, which makes it
robust to the wildly different score scales of cosine similarity and BM25 (no
fragile min-max normalisation or hand-tuned ``alpha`` blend). This mirrors the
production backend's ``rrf_fuse``::

    score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

A document present in only one list still contributes that list's term, so
hybrid search degrades gracefully to whichever channel found the document.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    dense_ranks: dict[str, int],
    sparse_ranks: dict[str, int],
    *,
    k_constant: int = 60,
    weight_dense: float = 1.0,
    weight_sparse: float = 1.0,
) -> dict[str, float]:
    """Fuse two rank maps into a single fused-score map.

    Args:
        dense_ranks: ``{doc_id: rank}`` from dense retrieval (rank is 1-based).
        sparse_ranks: ``{doc_id: rank}`` from sparse (BM25) retrieval.
        k_constant: RRF damping constant (larger = flatter contribution curve).
        weight_dense: Weight applied to the dense channel.
        weight_sparse: Weight applied to the sparse channel.

    Returns:
        ``{doc_id: fused_score}`` over the union of both inputs (higher = better).
    """
    fused: dict[str, float] = {}
    for doc_id, rank in dense_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + weight_dense / (k_constant + rank)
    for doc_id, rank in sparse_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + weight_sparse / (k_constant + rank)
    return fused

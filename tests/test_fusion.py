"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from app.rag.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_channels() -> None:
    # Doc "a" is top in both channels; "b"/"c" each top one channel only.
    dense = {"a": 1, "b": 2, "c": 3}
    sparse = {"a": 1, "c": 2, "b": 3}
    fused = reciprocal_rank_fusion(dense, sparse, k_constant=60)
    ranked = sorted(fused, key=lambda d: fused[d], reverse=True)
    assert ranked[0] == "a"


def test_rrf_includes_single_channel_docs() -> None:
    dense = {"a": 1}
    sparse = {"b": 1}
    fused = reciprocal_rank_fusion(dense, sparse)
    assert set(fused) == {"a", "b"}


def test_rrf_weights_bias_channels() -> None:
    dense = {"d": 1}
    sparse = {"s": 1}
    fused = reciprocal_rank_fusion(dense, sparse, weight_dense=2.0, weight_sparse=1.0)
    assert fused["d"] > fused["s"]

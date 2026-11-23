"""Reciprocal Rank Fusion (RRF) for combining dense and lexical (BM25) retrieved candidates."""

import logging
from typing import Dict, List, Optional
from app.modules.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_chunks: List[RetrievedChunk],
    bm25_chunks: List[RetrievedChunk],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Combine dense and BM25 ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score = sum(1.0 / (k + rank_i)) for each ranked list i

    Args:
        dense_chunks: Validated chunks from vector search, ordered by similarity descending.
        bm25_chunks: Validated chunks from BM25 full-text search, ordered by BM25 score descending.
        k: Smoothing constant (default 60). Mitigates the impact of high ranks.
        top_k: Maximum number of fused candidates to return. If None, returns all unique candidates.

    Returns:
        List of unique RetrievedChunk objects ordered by RRF_score descending, with debugging
        ranks and scores preserved.
    """
    if not dense_chunks and not bm25_chunks:
        return []

    if not dense_chunks:
        # Only BM25 candidates available
        fused: List[RetrievedChunk] = []
        for rank_idx, chunk in enumerate(bm25_chunks, start=1):
            rrf_score = round(1.0 / (k + rank_idx), 6)
            c = chunk.model_copy()
            c.bm25_rank = rank_idx
            c.rrf_score = rrf_score
            c.retrieval_mode = "hybrid"
            fused.append(c)
        return fused[:top_k] if top_k else fused

    if not bm25_chunks:
        # Only dense candidates available
        fused: List[RetrievedChunk] = []
        for rank_idx, chunk in enumerate(dense_chunks, start=1):
            rrf_score = round(1.0 / (k + rank_idx), 6)
            c = chunk.model_copy()
            c.dense_rank = rank_idx
            c.rrf_score = rrf_score
            c.retrieval_mode = "hybrid"
            fused.append(c)
        return fused[:top_k] if top_k else fused

    # Map candidate by chunk_id
    chunks_map: Dict[str, RetrievedChunk] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    for rank_idx, chunk in enumerate(dense_chunks, start=1):
        chunks_map[chunk.chunk_id] = chunk.model_copy()
        dense_ranks[chunk.chunk_id] = rank_idx

    for rank_idx, chunk in enumerate(bm25_chunks, start=1):
        if chunk.chunk_id not in chunks_map:
            chunks_map[chunk.chunk_id] = chunk.model_copy()
        else:
            # Preserve BM25 specific scores onto existing chunk copy
            existing = chunks_map[chunk.chunk_id]
            if chunk.bm25_score is not None:
                existing.bm25_score = chunk.bm25_score
        bm25_ranks[chunk.chunk_id] = rank_idx

    # Compute RRF score for all unique chunks
    fused_list: List[RetrievedChunk] = []
    for chunk_id, chunk in chunks_map.items():
        d_rank = dense_ranks.get(chunk_id)
        b_rank = bm25_ranks.get(chunk_id)

        rrf_d = (1.0 / (k + d_rank)) if d_rank else 0.0
        rrf_b = (1.0 / (k + b_rank)) if b_rank else 0.0
        total_rrf = round(rrf_d + rrf_b, 6)

        chunk.dense_rank = d_rank
        if d_rank and chunk.dense_score is None:
            chunk.dense_score = chunk.similarity_score
        chunk.bm25_rank = b_rank
        chunk.rrf_score = total_rrf
        chunk.retrieval_mode = "hybrid"

        fused_list.append(chunk)

    # Sort descending by RRF score, breaking ties by dense similarity then BM25 score
    fused_list.sort(
        key=lambda c: (
            -(c.rrf_score or 0.0),
            -(c.similarity_score or 0.0),
            -(c.bm25_score or 0.0),
        )
    )

    result = fused_list[:top_k] if top_k else fused_list
    logger.info(
        "RRF fusion completed: %d dense + %d BM25 candidates -> %d unique fused chunks (k=%d)",
        len(dense_chunks),
        len(bm25_chunks),
        len(result),
        k,
    )
    return result

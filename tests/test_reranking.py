"""Unit and integration tests for Phase 7 Reranking Module."""

import pytest
from app.modules.reranking.base import BaseReranker
from app.modules.reranking.reranker import DeterministicV1Reranker
from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.schemas import RetrievedChunk


def _create_sample_chunk(
    chunk_id: str,
    content: str,
    similarity_score: float,
    heading: str = "General Provisions",
    heading_path: list = None,
    doc_name: str = "HR Manual",
    page_start: int = 1,
    page_end: int = 1,
) -> RetrievedChunk:
    """Helper to instantiate mock RetrievedChunk objects."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-test-123",
        document_name=doc_name,
        document_type="policy",
        version=1,
        content=content,
        heading=heading,
        heading_path=heading_path or [heading],
        page_start=page_start,
        page_end=page_end,
        similarity_score=similarity_score,
        school_id="sch-001",
        group_id="grp-001",
        branch_id="br-001",
    )


def test_base_reranker_cannot_be_instantiated():
    """Verify BaseReranker is an abstract base class."""
    with pytest.raises(TypeError):
        BaseReranker()


def test_reranker_empty_chunks():
    """Verify reranker handles empty chunk lists gracefully."""
    reranker = DeterministicV1Reranker()
    results = reranker.rerank(query="maternity leave policy", chunks=[])
    assert results == []


def test_reranker_weights_normalization():
    """Verify custom weights are normalized to sum to 1.0."""
    reranker = DeterministicV1Reranker(semantic_weight=2.0, lexical_weight=1.0, heading_weight=1.0)
    assert reranker.w_sem == pytest.approx(0.5)
    assert reranker.w_lex == pytest.approx(0.25)
    assert reranker.w_head == pytest.approx(0.25)


def test_reranker_heading_and_lexical_boost_reorders_candidates():
    """Verify that a candidate with matching heading and keywords is ranked higher than a pure semantic match."""
    reranker = DeterministicV1Reranker(semantic_weight=0.5, lexical_weight=0.3, heading_weight=0.2)

    # Chunk A: High semantic score from embedding, but generic content about staff general conduct
    chunk_a = _create_sample_chunk(
        chunk_id="chk-general-conduct",
        content="All employees must maintain professionalism, integrity, and arrive punctually at work premises.",
        similarity_score=0.85,
        heading="Staff Code of Conduct",
        heading_path=["General Policies", "Staff Code of Conduct"],
    )

    # Chunk B: Slightly lower semantic score, but exact heading match and rich keyword overlap for maternity leave
    chunk_b = _create_sample_chunk(
        chunk_id="chk-maternity-leave",
        content="Female teaching and non-teaching staff are eligible for 26 weeks of fully paid maternity leave.",
        similarity_score=0.75,
        heading="Maternity Leave Benefits",
        heading_path=["Leave Regulations", "Maternity Leave Benefits"],
    )

    ranked = reranker.rerank(
        query="maternity leave benefits and duration",
        chunks=[chunk_a, chunk_b],
        keywords=["maternity leave", "paid leave", "benefits"],
    )

    assert len(ranked) == 2
    # Chunk B should be boosted to rank 1 due to heading alignment and lexical density
    assert ranked[0].chunk_id == "chk-maternity-leave"
    assert ranked[0].rerank_rank == 1
    assert ranked[0].original_rank == 2
    assert ranked[0].rerank_score > ranked[1].rerank_score
    assert ranked[0].heading_score > ranked[1].heading_score
    assert ranked[0].lexical_score > ranked[1].lexical_score

    # Chunk A should now be rank 2
    assert ranked[1].chunk_id == "chk-general-conduct"
    assert ranked[1].rerank_rank == 2
    assert ranked[1].original_rank == 1


def test_reranker_scores_bounded_and_non_probabilistic():
    """Verify all scores are strictly in [0.0, 1.0] and explanation documents non-probabilistic heuristic."""
    reranker = DeterministicV1Reranker()
    chunk = _create_sample_chunk(
        chunk_id="chk-probation",
        content="Probation period shall be 6 months for all newly hired academic faculty.",
        similarity_score=0.90,
        heading="Probation Period",
    )

    results = reranker.rerank(
        query="probation period faculty",
        chunks=[chunk],
        keywords=["probation", "academic faculty"],
    )

    item = results[0]
    assert 0.0 <= item.rerank_score <= 1.0
    assert 0.0 <= item.lexical_score <= 1.0
    assert 0.0 <= item.heading_score <= 1.0
    assert 0.0 <= item.similarity_score <= 1.0

    # Ensure explanation contains the non-probability disclaimer
    assert "not confidence probability" in item.score_explanation
    assert "ranking heuristic" in item.score_explanation


def test_reranker_preserves_original_scores_and_metadata():
    """Verify original chunk fields, tenant IDs, and dates are preserved exactly."""
    reranker = DeterministicV1Reranker()
    chunk = _create_sample_chunk(
        chunk_id="chk-sample-meta",
        content="Teachers are entitled to 12 days casual leave annually.",
        similarity_score=0.72,
        heading="Casual Leave",
        doc_name="Staff Policy 2026",
        page_start=4,
        page_end=5,
    )

    ranked = reranker.rerank(query="casual leave", chunks=[chunk])
    assert len(ranked) == 1
    res = ranked[0]
    assert res.chunk_id == "chk-sample-meta"
    assert res.document_id == "doc-test-123"
    assert res.document_name == "Staff Policy 2026"
    assert res.page_start == 4
    assert res.page_end == 5
    assert res.similarity_score == 0.72
    assert res.school_id == "sch-001"
    assert res.group_id == "grp-001"
    assert res.branch_id == "br-001"

"""Comprehensive tests for Hybrid Retrieval (BM25 + Dense + RRF) and SQLite FTS5 Index Provider."""

import asyncio
from datetime import datetime, timezone
import logging
import os
import tempfile
from typing import List
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.api.dependencies import SchoolContext
from app.core.exceptions import AppException
from app.core.logging_utils import clear_request_id, set_request_id
from app.modules.bm25.schemas import BM25ChunkRecord, BM25SearchResult
from app.modules.bm25.sqlite_provider import SQLiteFTSBM25IndexProvider
from app.modules.chunking.schemas import DocumentChunk
from app.modules.context.service import ContextService
from app.modules.generation.schemas import AnswerRequest, LLMAnswerRaw
from app.modules.generation.service import AnswerService
from app.modules.indexing.schemas import DocumentIndexingRecord, IndexingStageStatus
from app.modules.indexing.service import IndexingService
from app.modules.reranking.reranker import DeterministicV1Reranker
from app.modules.retrieval.filtering import PolicyScopeFilter
from app.modules.retrieval.fusion import reciprocal_rank_fusion
from app.modules.retrieval.schemas import EmployeeContext, RetrievalRequest, RetrievedChunk
from app.modules.retrieval.service import RetrievalService


# ==============================================================================
# 1. SQLite FTS5 BM25 Index Provider Unit Tests
# ==============================================================================


def test_sqlite_fts5_provider_lifecycle():
    """Verify SQLite FTS5 initialization, memory mode, health check, and schema creation."""
    provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    health = provider.health_check()

    assert health["status"] == "healthy"
    assert health["provider"] == "sqlite_fts5"
    assert health["fts5_available"] is True
    assert health["total_chunks"] == 0
    assert health["total_documents"] == 0

    provider.close()


def test_sqlite_fts5_file_persistence():
    """Verify SQLite FTS5 file persistence across provider re-initialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_bm25.db")
        provider1 = SQLiteFTSBM25IndexProvider(db_path=db_path)

        chunk = BM25ChunkRecord(
            chunk_id="chunk-persistent-01",
            document_id="doc-persist",
            generation=1,
            content="Maternity leave benefit is 26 weeks of fully paid leave.",
            document_name="Leave Policy",
            school_id="SCH001",
        )
        provider1.index_chunks([chunk])
        assert provider1.count() == 1
        provider1.close()

        # Reopen same database file
        provider2 = SQLiteFTSBM25IndexProvider(db_path=db_path)
        assert provider2.count() == 1
        results = provider2.search("maternity leave 26 weeks", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-persistent-01"
        provider2.close()


def test_sqlite_fts5_query_sanitization_and_matching():
    """Verify complex natural language queries with quotes, contractions, and punctuation parse safely."""
    provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    chunks = [
        BM25ChunkRecord(
            chunk_id="chk-01",
            document_id="doc-01",
            generation=1,
            content="Teacher's probationary period is 6 months; subject to performance review.",
            heading="Probationary Period",
            document_name="Staff Policy",
            school_id="SCH001",
        ),
        BM25ChunkRecord(
            chunk_id="chk-02",
            document_id="doc-02",
            generation=1,
            content="Emergency medical leave requires a certificate within 48 hours.",
            heading="Medical Leave",
            document_name="Medical Guidelines",
            school_id="SCH001",
        ),
    ]
    provider.index_chunks(chunks)

    # Contractions, apostrophes, question marks, parenthesis
    results = provider.search("What is the teacher's probation (in months)?")
    assert len(results) > 0
    assert results[0].chunk_id == "chk-01"
    assert results[0].bm25_rank == 1
    assert results[0].bm25_score > 0

    # Quoted phrase search
    results_phrase = provider.search('"emergency medical leave" certificate')
    assert len(results_phrase) > 0
    assert results_phrase[0].chunk_id == "chk-02"

    provider.close()


def test_sqlite_fts5_idempotent_upsert():
    """Verify re-indexing the same chunk_id replaces content rather than creating duplicates."""
    provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    chunk_v1 = BM25ChunkRecord(
        chunk_id="chk-upsert",
        document_id="doc-01",
        generation=1,
        content="Initial policy drafted version.",
        document_name="Draft Policy",
    )
    provider.index_chunks([chunk_v1])
    assert provider.count() == 1

    chunk_v2 = BM25ChunkRecord(
        chunk_id="chk-upsert",
        document_id="doc-01",
        generation=1,
        content="Revised finalized policy content with updated leave limits.",
        document_name="Final Policy",
    )
    provider.index_chunks([chunk_v2])
    assert provider.count() == 1

    results = provider.search("finalized policy updated leave")
    assert len(results) == 1
    assert "Revised finalized policy" in results[0].content

    provider.close()


def test_sqlite_fts5_generation_pruning():
    """Verify delete_generation and delete_previous_generations operate strictly on generations."""
    provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    gen1_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-g1-{i}",
            document_id="doc-multi-gen",
            generation=1,
            content=f"Generation 1 content clause {i}",
            document_name="Policy",
        )
        for i in range(3)
    ]
    gen2_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-g2-{i}",
            document_id="doc-multi-gen",
            generation=2,
            content=f"Generation 2 updated content clause {i}",
            document_name="Policy",
        )
        for i in range(3)
    ]
    provider.index_chunks(gen1_chunks + gen2_chunks)
    assert provider.count() == 6
    assert provider.count_generation_chunks("doc-multi-gen", 1) == 3
    assert provider.count_generation_chunks("doc-multi-gen", 2) == 3

    # Delete previous generation (keeping gen 2)
    deleted = provider.delete_previous_generations("doc-multi-gen", active_generation=2)
    assert deleted == 3
    assert provider.count() == 3
    assert provider.count_generation_chunks("doc-multi-gen", 1) == 0
    assert provider.count_generation_chunks("doc-multi-gen", 2) == 3

    # Abort/purge gen 2
    deleted_g2 = provider.delete_generation("doc-multi-gen", generation=2)
    assert deleted_g2 == 3
    assert provider.count() == 0

    provider.close()


# ==============================================================================
# 2. Reciprocal Rank Fusion (RRF) Unit Tests
# ==============================================================================


def _make_dummy_chunk(chunk_id: str, similarity: float = 0.8, bm25_score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-01",
        document_name="Test Policy",
        document_type="policy",
        version=1,
        content=f"Content for {chunk_id}",
        heading="Section 1",
        page_start=1,
        page_end=1,
        similarity_score=similarity,
        bm25_score=bm25_score,
        school_id="SCH001",
    )


def test_rrf_both_streams_overlapping():
    """Verify overlapping chunks receive reciprocal rank contributions from both dense and BM25."""
    # Chunk A: Dense rank 1, BM25 rank 2
    # Chunk B: Dense rank 2, not in BM25
    # Chunk C: BM25 rank 1, not in Dense
    dense = [_make_dummy_chunk("A", similarity=0.95), _make_dummy_chunk("B", similarity=0.85)]
    bm25 = [_make_dummy_chunk("C", bm25_score=2.5), _make_dummy_chunk("A", bm25_score=1.8)]

    fused = reciprocal_rank_fusion(dense, bm25, k=60)

    # Chunk A should be #1 because it has rank 1 in dense and rank 2 in BM25
    assert len(fused) == 3
    assert fused[0].chunk_id == "A"
    expected_rrf_a = round((1.0 / (60 + 1)) + (1.0 / (60 + 2)), 6)
    assert fused[0].rrf_score == expected_rrf_a
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[0].dense_score == 0.95
    assert fused[0].retrieval_mode == "hybrid"

    # Chunk C: only BM25 rank 1 -> 1 / (60 + 1) = 0.016393
    # Chunk B: only Dense rank 2 -> 1 / (60 + 2) = 0.016129
    assert fused[1].chunk_id == "C"
    assert fused[1].dense_rank is None
    assert fused[1].bm25_rank == 1

    assert fused[2].chunk_id == "B"
    assert fused[2].dense_rank == 2
    assert fused[2].bm25_rank is None


def test_rrf_deduplication():
    """Verify RRF deduplicates identical chunk IDs and merges metadata."""
    chunk1 = _make_dummy_chunk("C1", similarity=0.9)
    chunk2 = _make_dummy_chunk("C1", bm25_score=3.2)

    fused = reciprocal_rank_fusion([chunk1], [chunk2], k=60)
    assert len(fused) == 1
    assert fused[0].chunk_id == "C1"
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 1
    assert fused[0].bm25_score == 3.2


def test_rrf_empty_dense_stream():
    """Verify RRF safely handles empty dense list (BM25 only)."""
    bm25 = [_make_dummy_chunk("BM-1", bm25_score=2.0), _make_dummy_chunk("BM-2", bm25_score=1.5)]
    fused = reciprocal_rank_fusion([], bm25, k=60)
    assert len(fused) == 2
    assert fused[0].chunk_id == "BM-1"
    assert fused[0].bm25_rank == 1
    assert fused[0].rrf_score == round(1.0 / (60 + 1), 6)


def test_rrf_empty_bm25_stream():
    """Verify RRF safely handles empty BM25 list (Dense only)."""
    dense = [_make_dummy_chunk("D-1", similarity=0.92)]
    fused = reciprocal_rank_fusion(dense, [], k=60)
    assert len(fused) == 1
    assert fused[0].chunk_id == "D-1"
    assert fused[0].dense_rank == 1
    assert fused[0].rrf_score == round(1.0 / (60 + 1), 6)


def test_rrf_both_empty():
    """Verify RRF with two empty lists returns empty list."""
    fused = reciprocal_rank_fusion([], [], k=60)
    assert fused == []


# ==============================================================================
# 3. Document Indexing with BM25 & Generation Rollback Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_indexing_service_indexes_bm25_and_activates_generation():
    """Verify IndexingService writes to both ChromaDB and BM25 under target generation and activates."""
    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-idx-01",
        "document_name": "Leave Policy.pdf",
        "processing_status": "chunked",
    }
    mock_chunk_repo = AsyncMock()
    mock_chunk_repo.get_all_by_document_id.return_value = [
        {
            "chunk_id": "chk-01",
            "document_id": "doc-idx-01",
            "chunk_index": 0,
            "content": "Employees get 15 days annual leave.",
            "document_name": "Leave Policy",
            "document_type": "policy",
            "version": 1,
            "school_id": "SCH001",
            "page_start": 1,
            "page_end": 1,
            "token_count": 10,
        }
    ]
    mock_indexing_repo = AsyncMock()
    mock_indexing_repo.get_tracking_record.return_value = None
    mock_indexing_repo.get_chunk_embedding.return_value = None

    mock_vector_repo = MagicMock()
    mock_vector_repo.count_generation_vectors.return_value = 1

    mock_embedding_provider = MagicMock()
    mock_embedding_provider.get_model_name.return_value = "text-embedding-3-small"
    mock_embedding_provider.embed_texts = AsyncMock(return_value=[[0.1] * 1536])

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    service = IndexingService(
        document_repo=mock_doc_repo,
        chunk_repo=mock_chunk_repo,
        indexing_repo=mock_indexing_repo,
        vector_repo=mock_vector_repo,
        embedding_provider=mock_embedding_provider,
        bm25_provider=bm25_provider,
    )

    res = await service.index_document("doc-idx-01")
    assert res.indexing_status == "indexed"
    assert res.active_generation == 1
    assert bm25_provider.count() == 1
    assert bm25_provider.count_generation_chunks("doc-idx-01", 1) == 1

    bm25_provider.close()


@pytest.mark.asyncio
async def test_indexing_service_rollback_on_bm25_verification_failure():
    """Verify rollback prunes target generation from both ChromaDB and BM25 on count mismatch."""
    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-fail-01",
        "document_name": "Policy.pdf",
        "processing_status": "chunked",
    }
    mock_chunk_repo = AsyncMock()
    mock_chunk_repo.get_all_by_document_id.return_value = [
        {
            "chunk_id": "chk-fail-1",
            "document_id": "doc-fail-01",
            "chunk_index": 0,
            "content": "Clause 1",
            "document_name": "Policy",
            "document_type": "policy",
            "version": 1,
            "school_id": "SCH001",
            "page_start": 1,
            "page_end": 1,
            "token_count": 5,
        }
    ]
    mock_indexing_repo = AsyncMock()
    mock_indexing_repo.get_tracking_record.return_value = {"active_generation": 1}
    mock_indexing_repo.get_chunk_embedding.return_value = {"embedding": [0.1] * 1536}

    mock_vector_repo = MagicMock()
    # Simulate vector write count mismatch
    mock_vector_repo.count_generation_vectors.return_value = 0

    mock_embedding_provider = MagicMock()
    mock_embedding_provider.get_model_name.return_value = "text-embedding-3-small"
    mock_embedding_provider.embed_texts = AsyncMock(return_value=[[0.1] * 1536])

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    service = IndexingService(
        document_repo=mock_doc_repo,
        chunk_repo=mock_chunk_repo,
        indexing_repo=mock_indexing_repo,
        vector_repo=mock_vector_repo,
        embedding_provider=mock_embedding_provider,
        bm25_provider=bm25_provider,
    )

    with pytest.raises(AppException) as exc_info:
        await service.index_document("doc-fail-01")

    assert "Generation verification failed" in str(exc_info.value)
    # Target generation 2 must have been pruned from BM25
    assert bm25_provider.count_generation_chunks("doc-fail-01", 2) == 0
    mock_vector_repo.delete_generation.assert_called_with("doc-fail-01", 2)

    bm25_provider.close()


@pytest.mark.asyncio
async def test_indexing_service_rebuild_bm25_index():
    """Verify rebuild_bm25_index reads active chunks from MongoDB and populates BM25."""
    mock_doc_repo = AsyncMock()
    mock_chunk_repo = AsyncMock()
    mock_chunk_repo.get_all_by_document_id.return_value = [
        {
            "chunk_id": "chk-rebuild-01",
            "document_id": "doc-reb",
            "chunk_index": 0,
            "content": "Rebuilt content from authoritative MongoDB chunks.",
            "document_name": "Policy",
            "document_type": "policy",
            "version": 1,
            "school_id": "SCH001",
            "page_start": 1,
            "page_end": 1,
            "token_count": 10,
        }
    ]
    mock_indexing_repo = AsyncMock()
    mock_indexing_repo.get_tracking_record.return_value = {"document_id": "doc-reb", "active_generation": 2}
    mock_indexing_repo.get_all_active_tracking_records.return_value = [
        {"document_id": "doc-reb", "active_generation": 2}
    ]

    mock_vector_repo = MagicMock()
    mock_embedding_provider = AsyncMock()

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    service = IndexingService(
        document_repo=mock_doc_repo,
        chunk_repo=mock_chunk_repo,
        indexing_repo=mock_indexing_repo,
        vector_repo=mock_vector_repo,
        embedding_provider=mock_embedding_provider,
        bm25_provider=bm25_provider,
    )

    count = await service.rebuild_bm25_index(document_id="doc-reb")
    assert count == 1
    assert bm25_provider.count() == 1
    results = bm25_provider.search("rebuilt content")
    assert len(results) == 1
    assert results[0].chunk_id == "chk-rebuild-01"

    # Test full rebuild across all documents
    count_all = await service.rebuild_bm25_index(document_id=None)
    assert count_all == 1
    assert bm25_provider.count() == 1

    bm25_provider.close()


# ==============================================================================
# 4. Hybrid Retrieval Pipeline Tests
# ==============================================================================


@pytest.fixture
def mock_pipeline_repos():
    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-01",
        "document_name": "Leave Policy",
        "document_type": "policy",
        "version": 1,
        "policy_lifecycle_status": "approved",
        "status": "ready",
        "school_id": "SCH001",
        "group_id": "GRP001",
    }
    mock_indexing_repo = AsyncMock()
    mock_indexing_repo.get_tracking_record.return_value = {
        "document_id": "doc-01",
        "active_generation": 1,
        "indexing_status": "indexed",
    }
    mock_vector_repo = MagicMock()
    mock_embedding_provider = AsyncMock()
    mock_embedding_provider.embed_texts.return_value = [[0.1] * 1536]

    return mock_doc_repo, mock_indexing_repo, mock_vector_repo, mock_embedding_provider


@pytest.mark.asyncio
async def test_hybrid_retrieval_exact_keyword_policy_code(mock_pipeline_repos):
    """Verify BM25 surfaces exact keyword match (e.g. policy clause code) even when dense score is lower."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    # Dense returns an unrelated chunk with moderate similarity
    vector_repo.query_vectors.return_value = {
        "ids": [["chk-generic"]],
        "documents": [["Generic leave guidelines for all staff."]],
        "metadatas": [[{
            "chunk_id": "chk-generic",
            "document_id": "doc-01",
            "generation": 1,
            "school_id": "SCH001",
            "group_id": "GRP001",
        }]],
        "distances": [[0.3]],
    }
    vector_repo.count.return_value = 1

    # BM25 has an exact policy clause code "CLAUSE-SEC-802"
    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    bm25_provider.index_chunks([
        BM25ChunkRecord(
            chunk_id="chk-exact-clause",
            document_id="doc-01",
            generation=1,
            content="Under CLAUSE-SEC-802, emergency casual leave is granted up to 3 days.",
            document_name="Leave Policy",
            school_id="SCH001",
            group_id="GRP001",
        )
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_rrf_k=60,
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="What does CLAUSE-SEC-802 state?"),
        school_context=SchoolContext(school_id="SCH001", group_id="GRP001"),
    )

    assert res.retrieval_mode == "hybrid"
    assert res.total_results >= 1
    # The exact keyword match from BM25 must be present in the retrieved candidates
    exact_match = next((c for c in res.chunks if c.chunk_id == "chk-exact-clause"), None)
    assert exact_match is not None
    assert exact_match.bm25_rank == 1
    assert "CLAUSE-SEC-802" in exact_match.content

    bm25_provider.close()


@pytest.mark.asyncio
async def test_hybrid_retrieval_semantic_only_match(mock_pipeline_repos):
    """Verify dense retrieval surfaces semantic match when keyword terms are completely absent in BM25."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    # Query: "taking time off for newborn child"
    # Document content: "Paternity and adoption leave entitlements"
    vector_repo.query_vectors.return_value = {
        "ids": [["chk-semantic"]],
        "documents": [["Paternity and parental entitlements allow up to 4 weeks of paid time off."]],
        "metadatas": [[{
            "chunk_id": "chk-semantic",
            "document_id": "doc-01",
            "generation": 1,
            "school_id": "SCH001",
            "group_id": "GRP001",
        }]],
        "distances": [[0.15]],
    }
    vector_repo.count.return_value = 1

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    # Index unrelated text into BM25
    bm25_provider.index_chunks([
        BM25ChunkRecord(
            chunk_id="chk-unrelated",
            document_id="doc-01",
            generation=1,
            content="Cafeteria and lunch break timings are from 12:30 to 1:30 PM.",
            document_name="Leave Policy",
            school_id="SCH001",
            group_id="GRP001",
        )
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_rrf_k=60,
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="taking time off for newborn child"),
        school_context=SchoolContext(school_id="SCH001", group_id="GRP001"),
    )

    assert res.retrieval_mode == "hybrid"
    assert len(res.chunks) >= 1
    semantic_match = next((c for c in res.chunks if c.chunk_id == "chk-semantic"), None)
    assert semantic_match is not None
    assert semantic_match.dense_rank == 1

    bm25_provider.close()


@pytest.mark.asyncio
async def test_hybrid_retrieval_overlapping_boost(mock_pipeline_repos):
    """Verify chunk matching both dense and BM25 gets boosted to #1 above single-modality chunks."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    # Dense returns Chunk A (rank 2) and Chunk B (rank 1)
    vector_repo.query_vectors.return_value = {
        "ids": [["chk-B", "chk-A"]],
        "documents": [
            ["General annual leave information."],
            ["Maternity leave is 26 weeks for female employees."],
        ],
        "metadatas": [[
            {"chunk_id": "chk-B", "document_id": "doc-01", "generation": 1, "school_id": "SCH001", "group_id": "GRP001"},
            {"chunk_id": "chk-A", "document_id": "doc-01", "generation": 1, "school_id": "SCH001", "group_id": "GRP001"},
        ]],
        "distances": [[0.1, 0.2]],
    }
    vector_repo.count.return_value = 2

    # BM25 returns Chunk A (rank 1) and Chunk C (rank 2)
    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    bm25_provider.index_chunks([
        BM25ChunkRecord(
            chunk_id="chk-A",
            document_id="doc-01",
            generation=1,
            content="Maternity leave is 26 weeks for female employees.",
            document_name="Leave Policy",
            school_id="SCH001",
            group_id="GRP001",
        ),
        BM25ChunkRecord(
            chunk_id="chk-C",
            document_id="doc-01",
            generation=1,
            content="Maternity hospital medical benefits allowance.",
            document_name="Leave Policy",
            school_id="SCH001",
            group_id="GRP001",
        ),
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_rrf_k=60,
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="maternity leave weeks"),
        school_context=SchoolContext(school_id="SCH001", group_id="GRP001"),
    )

    # Chunk A was rank 2 in Dense and rank 1 in BM25 -> Highest RRF score
    assert res.chunks[0].chunk_id == "chk-A"
    assert res.chunks[0].dense_rank == 2
    assert res.chunks[0].bm25_rank == 1

    bm25_provider.close()


@pytest.mark.asyncio
async def test_hybrid_retrieval_scope_and_applicability_filtering(mock_pipeline_repos):
    """Verify BM25 candidates are strictly filtered by tenant scope, employee category, and dates."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    bm25_provider.index_chunks([
        # Chunk 1: Wrong school
        BM25ChunkRecord(
            chunk_id="chk-wrong-school",
            document_id="doc-01",
            generation=1,
            content="Probation period policy for other school.",
            document_name="Leave Policy",
            school_id="SCH999",
            staff_category="teaching",
        ),
        # Chunk 2: Wrong staff category
        BM25ChunkRecord(
            chunk_id="chk-wrong-staff",
            document_id="doc-01",
            generation=1,
            content="Probation period policy for administrative staff.",
            document_name="Leave Policy",
            school_id="SCH001",
            staff_category="administrative",
        ),
        # Chunk 3: Valid matching chunk
        BM25ChunkRecord(
            chunk_id="chk-valid-teaching",
            document_id="doc-01",
            generation=1,
            content="Probation period for teaching staff is 6 months.",
            document_name="Leave Policy",
            school_id="SCH001",
            staff_category="teaching",
        ),
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(
            question="probation period",
            employee_context=EmployeeContext(staff_category="teaching"),
        ),
        school_context=SchoolContext(school_id="SCH001"),
    )

    # Only chk-valid-teaching should pass validation
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "chk-valid-teaching"

    bm25_provider.close()


@pytest.mark.asyncio
async def test_hybrid_retrieval_missing_bm25_provider_explicit_failure(mock_pipeline_repos):
    """Verify AppException is raised when hybrid mode is requested without BM25 provider (no silent fallback)."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=None,
        retrieval_mode="hybrid",
    )

    with pytest.raises(AppException) as exc_info:
        await retrieval_service.retrieve(
            request=RetrievalRequest(question="maternity leave"),
            school_context=SchoolContext(school_id="SCH001"),
        )

    assert exc_info.value.error_code == "BM25_PROVIDER_UNAVAILABLE"


@pytest.mark.asyncio
async def test_dense_mode_backward_compatibility(mock_pipeline_repos):
    """Verify default dense mode does not invoke BM25 and returns dense response."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    vector_repo.query_vectors.return_value = {
        "ids": [["chk-dense-01"]],
        "documents": [["Annual leave is 15 days."]],
        "metadatas": [[{
            "chunk_id": "chk-dense-01",
            "document_id": "doc-01",
            "generation": 1,
            "school_id": "SCH001",
        }]],
        "distances": [[0.1]],
    }
    vector_repo.count.return_value = 1

    bm25_mock = MagicMock()

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_mock,
        retrieval_mode="dense",
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="How many days annual leave?"),
        school_context=SchoolContext(school_id="SCH001"),
    )

    assert res.retrieval_mode == "dense"
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "chk-dense-01"
    # BM25 provider must NOT have been called in dense mode
    bm25_mock.search.assert_not_called()


# ==============================================================================
# 5. End-to-End Context & Answering Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_context_and_answer_service_with_hybrid_retrieval(mock_pipeline_repos):
    """Verify full question answering pipeline functions correctly with hybrid retrieval."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    vector_repo.query_vectors.return_value = {
        "ids": [["chk-ans-01"]],
        "documents": [["Probationary period for teaching faculty is strictly 6 months."]],
        "metadatas": [[{
            "chunk_id": "chk-ans-01",
            "document_id": "doc-01",
            "generation": 1,
            "school_id": "SCH001",
            "heading": "Probation Terms",
        }]],
        "distances": [[0.12]],
    }
    vector_repo.count.return_value = 1

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    bm25_provider.index_chunks([
        BM25ChunkRecord(
            chunk_id="chk-ans-01",
            document_id="doc-01",
            generation=1,
            content="Probationary period for teaching faculty is strictly 6 months.",
            document_name="Faculty Policy",
            school_id="SCH001",
            heading="Probation Terms",
        )
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
    )

    context_service = ContextService(
        retrieval_service=retrieval_service,
        reranker=DeterministicV1Reranker(),
    )

    import json

    mock_llm_provider = MagicMock()
    mock_llm_provider.get_model_name.return_value = "gpt-4o"
    mock_llm_provider.generate = AsyncMock(
        return_value=json.dumps({
            "answer": "Teaching faculty have a 6-month probationary period [Source 1].",
            "citations": ["[Source 1]"],
            "insufficient_evidence": False,
            "conflicting_evidence": False,
        })
    )

    answer_service = AnswerService(
        context_service=context_service,
        llm_provider=mock_llm_provider,
    )

    response = await answer_service.generate_answer(
        request=AnswerRequest(question="What is the probationary period for teaching staff?"),
        school_context=SchoolContext(school_id="SCH001"),
    )

    assert response.status == "success"
    assert "6-month" in response.answer
    assert len(response.citations) >= 1
    assert response.citations[0].document_id == "doc-01"
    assert response.citations[0].citation_label == "[Source 1]"

    bm25_provider.close()


# ==============================================================================
# 6. Structured Logging Timing Tests for Hybrid Retrieval
# ==============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_structured_timing_logs(caplog, mock_pipeline_repos):
    """Verify structured INFO logs with duration_ms for all hybrid retrieval phases."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    vector_repo.query_vectors.return_value = {
        "ids": [["chk-log-01"]],
        "documents": [["Staff leave policy."]],
        "metadatas": [[{"chunk_id": "chk-log-01", "document_id": "doc-01", "generation": 1, "school_id": "SCH001"}]],
        "distances": [[0.2]],
    }
    vector_repo.count.return_value = 1

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")
    bm25_provider.index_chunks([
        BM25ChunkRecord(
            chunk_id="chk-log-01",
            document_id="doc-01",
            generation=1,
            content="Staff leave policy.",
            document_name="Leave Policy",
            school_id="SCH001",
        )
    ])

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
    )

    set_request_id("hybrid-timing-test")
    with caplog.at_level(logging.INFO):
        await retrieval_service.retrieve(
            request=RetrievalRequest(question="leave policy rules"),
            school_context=SchoolContext(school_id="SCH001"),
        )
    clear_request_id()

    all_msgs = [rec.message for rec in caplog.records]

    # Verify structured logs for hybrid phases
    assert any("module=total_retrieval" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=dense_retrieval" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=bm25_retrieval" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=mongo_validation" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=rrf_fusion" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)

    bm25_provider.close()


# ==============================================================================
# 7. Regression & Audit Tests: Starvation, Reranker/Context Integration & No Double-Counting
# ==============================================================================


@pytest.mark.asyncio
async def test_bm25_candidate_starvation_prevention_across_schools(mock_pipeline_repos):
    """Verify metadata filtering and controlled overfetching prevent tenant starvation in BM25 search.

    Scenario: School 'SCH_OTHER' has 25 chunks heavily matching query keywords.
    School 'SCH_TARGET' has only 2 chunks.
    Without tenant filtering in BM25, the 20 pool slots would be 100% dominated by SCH_OTHER,
    causing all candidates to be rejected by tenant validation and returning 0 chunks for SCH_TARGET.
    With metadata filtering, SCH_OTHER chunks are excluded at query time, and SCH_TARGET chunks are retrieved.
    """
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos

    # Dense returns no hits (isolate BM25 starvation behavior)
    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    # 25 chunks for dominating other school SCH_OTHER
    other_school_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-other-{i}",
            document_id="doc-other",
            generation=1,
            content=f"Probationary period for teaching staff terms and conditions clause {i}.",
            document_name="Other School Policy",
            school_id="SCH_OTHER",
            group_id="GRP_OTHER",
        )
        for i in range(25)
    ]

    # 2 chunks for requesting target school SCH_TARGET
    target_school_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-target-{i}",
            document_id="doc-target",
            generation=1,
            content=f"Probationary period for teaching staff terms and conditions for our school {i}.",
            document_name="Target School Policy",
            school_id="SCH_TARGET",
            group_id="GRP_TARGET",
        )
        for i in range(2)
    ]

    bm25_provider.index_chunks(other_school_chunks + target_school_chunks)
    assert bm25_provider.count() == 27

    # Configure MongoDB mock to recognize doc-target for SCH_TARGET
    async def mock_get_by_id(doc_id: str):
        if doc_id == "doc-target":
            return {
                "document_id": "doc-target",
                "document_name": "Target School Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "school_id": "SCH_TARGET",
                "group_id": "GRP_TARGET",
            }
        return {
            "document_id": doc_id,
            "document_name": "Other School Policy",
            "document_type": "policy",
            "version": 1,
            "policy_lifecycle_status": "approved",
            "status": "ready",
            "school_id": "SCH_OTHER",
            "group_id": "GRP_OTHER",
        }

    doc_repo.get_by_id.side_effect = mock_get_by_id
    indexing_repo.get_tracking_record.return_value = {"active_generation": 1, "indexing_status": "indexed"}

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_bm25_top_k=20,
    )

    # Request retrieval for SCH_TARGET
    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="probationary period teaching staff", top_k=2),
        school_context=SchoolContext(school_id="SCH_TARGET", group_id="GRP_TARGET"),
    )

    # Verify that SCH_TARGET chunks were successfully retrieved and not starved out by SCH_OTHER
    assert res.retrieval_mode == "hybrid"
    assert len(res.chunks) == 2
    for chunk in res.chunks:
        assert chunk.school_id == "SCH_TARGET"
        assert chunk.chunk_id.startswith("chk-target-")

    bm25_provider.close()


def test_reranker_and_context_builder_with_all_three_candidate_types():
    """Verify that dense-only, BM25-only and overlapping RRF candidates all pass correctly

    into the existing Phase 7 reranker and context builder, preserving all telemetry fields.
    """
    from app.modules.context.builder import ContextBuilder

    dense_only = RetrievedChunk(
        chunk_id="chk-dense-only",
        document_id="doc-01",
        document_name="Leave Policy",
        document_type="policy",
        version=1,
        content="Employees are entitled to 15 days of annual leave.",
        heading="Annual Leave",
        heading_path=["Leave", "Annual Leave"],
        page_start=1,
        page_end=1,
        similarity_score=0.88,
        dense_score=0.88,
        dense_rank=1,
        bm25_score=None,
        bm25_rank=None,
        rrf_score=0.016393,
        retrieval_mode="hybrid",
        school_id="SCH001",
    )

    bm25_only = RetrievedChunk(
        chunk_id="chk-bm25-only",
        document_id="doc-02",
        document_name="Emergency Leave Policy",
        document_type="policy",
        version=1,
        content="Emergency casual leave clause CLAUSE-802 provides up to 3 days.",
        heading="Emergency Leave",
        heading_path=["Leave", "Emergency Leave"],
        page_start=2,
        page_end=2,
        similarity_score=0.0,
        dense_score=None,
        dense_rank=None,
        bm25_score=3.45,
        bm25_rank=1,
        rrf_score=0.016393,
        retrieval_mode="hybrid",
        school_id="SCH001",
    )

    overlapping = RetrievedChunk(
        chunk_id="chk-overlapping",
        document_id="doc-01",
        document_name="Leave Policy",
        document_type="policy",
        version=1,
        content="Sick leave accrues at 1 day per completed month of service.",
        heading="Sick Leave",
        heading_path=["Leave", "Sick Leave"],
        page_start=1,
        page_end=1,
        similarity_score=0.84,
        dense_score=0.84,
        dense_rank=2,
        bm25_score=2.80,
        bm25_rank=2,
        rrf_score=0.032258,
        retrieval_mode="hybrid",
        school_id="SCH001",
    )

    chunks = [dense_only, bm25_only, overlapping]

    # 1. Phase 7 Deterministic Reranking
    reranker = DeterministicV1Reranker(semantic_weight=0.5, lexical_weight=0.3, heading_weight=0.2)
    ranked = reranker.rerank(
        query="annual leave emergency casual",
        chunks=chunks,
        keywords=["annual", "leave", "emergency"],
    )

    assert len(ranked) == 3

    # Verify all telemetry attributes are preserved on RankedChunk
    for rc in ranked:
        assert rc.retrieval_mode == "hybrid"
        assert rc.rrf_score is not None
        assert rc.rerank_score > 0.0
        assert rc.score_explanation is not None

    # Check dense-only chunk has dense_score and None bm25_score
    r_dense = next(c for c in ranked if c.chunk_id == "chk-dense-only")
    assert r_dense.dense_score == 0.88
    assert r_dense.bm25_score is None

    # Check bm25-only chunk has bm25_score and None dense_score
    r_bm25 = next(c for c in ranked if c.chunk_id == "chk-bm25-only")
    assert r_bm25.bm25_score == 3.45
    assert r_bm25.dense_score is None

    # Check overlapping chunk has both scores
    r_overlap = next(c for c in ranked if c.chunk_id == "chk-overlapping")
    assert r_overlap.dense_score == 0.84
    assert r_overlap.bm25_score == 2.80

    # 2. Phase 7 Context Builder
    builder = ContextBuilder(default_token_budget=2000)
    build_result = builder.build_context(chunks=ranked, token_budget=2000)

    assert build_result.selected_chunks_count == 3
    assert build_result.skipped_chunks_count == 0
    assert len(build_result.source_manifest) == 3
    assert "[Source 1:" in build_result.assembled_context
    assert "[Source 2:" in build_result.assembled_context
    assert "[Source 3:" in build_result.assembled_context


def test_reranker_formula_no_double_counting():
    """Verify reranker formula strictly isolates semantic vector similarity (s_sem) from lexical score (s_lex).

    RRF combines rank positions at the retrieval stage; DeterministicV1Reranker evaluates:
        composite = w_sem * s_sem + w_lex * s_lex + w_head * s_head
    where s_sem is strictly chunk.similarity_score and does NOT ingest rrf_score or bm25_score.
    """
    reranker = DeterministicV1Reranker(semantic_weight=0.5, lexical_weight=0.3, heading_weight=0.2)

    # Chunk with high RRF score but zero vector similarity (BM25 only)
    bm25_chunk = RetrievedChunk(
        chunk_id="chk-bm-pure",
        document_id="doc-01",
        document_name="Policy",
        document_type="policy",
        version=1,
        content="Casual leave is strictly limited to 10 days per academic year.",
        heading="Casual Leave Rules",
        heading_path=["Leave", "Casual Leave"],
        page_start=1,
        page_end=1,
        similarity_score=0.0,
        dense_score=None,
        bm25_score=4.5,
        bm25_rank=1,
        rrf_score=0.016393,
        retrieval_mode="hybrid",
    )

    ranked = reranker.rerank(
        query="casual leave days",
        chunks=[bm25_chunk],
        keywords=["casual", "leave", "days"],
    )

    assert len(ranked) == 1
    r = ranked[0]

    # Verify s_sem was 0.0 (not inflated by rrf_score)
    expected_lexical = r.lexical_score
    expected_heading = r.heading_score
    expected_composite = round((0.5 * 0.0) + (0.3 * expected_lexical) + (0.2 * expected_heading), 4)

    assert r.rerank_score == expected_composite
    assert r.similarity_score == 0.0
    assert "sem=0.0000*0.50" in r.score_explanation


@pytest.mark.asyncio
async def test_bm25_tenant_filtering_same_state_unrelated_schools(mock_pipeline_repos):
    """Verify unrelated schools in the same state are excluded by the SQLite pre-filter.

    Even if School B (SCH002) is in 'Karnataka' with 25 chunks matching the query,
    a search for School A (SCH001) in 'Karnataka' must exclude SCH002's chunks
    so they do not crowd out SCH001's chunks.
    """
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos
    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    # 25 chunks for unrelated school SCH002 in Karnataka
    sch002_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-sch002-{i}",
            document_id="doc-sch002",
            generation=1,
            content=f"Annual leave policy Karnataka state regulations clause {i}.",
            document_name="School 2 Policy",
            school_id="SCH002",
            state="Karnataka",
        )
        for i in range(25)
    ]

    # 2 chunks for target school SCH001 in Karnataka
    sch001_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-sch001-{i}",
            document_id="doc-sch001",
            generation=1,
            content=f"Annual leave policy Karnataka state regulations for school 1 clause {i}.",
            document_name="School 1 Policy",
            school_id="SCH001",
            state="Karnataka",
        )
        for i in range(2)
    ]

    bm25_provider.index_chunks(sch002_chunks + sch001_chunks)

    async def mock_get_by_id(doc_id: str):
        if doc_id == "doc-sch001":
            return {
                "document_id": "doc-sch001",
                "document_name": "School 1 Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "school_id": "SCH001",
                "state": "Karnataka",
            }
        return {
            "document_id": doc_id,
            "document_name": "School 2 Policy",
            "document_type": "policy",
            "version": 1,
            "policy_lifecycle_status": "approved",
            "status": "ready",
            "school_id": "SCH002",
            "state": "Karnataka",
        }

    doc_repo.get_by_id.side_effect = mock_get_by_id
    indexing_repo.get_tracking_record.return_value = {"active_generation": 1, "indexing_status": "indexed"}

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_bm25_top_k=20,
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="annual leave Karnataka", top_k=2),
        school_context=SchoolContext(school_id="SCH001", state="Karnataka"),
    )

    assert len(res.chunks) == 2
    for chunk in res.chunks:
        assert chunk.school_id == "SCH001"
        assert chunk.chunk_id.startswith("chk-sch001-")

    bm25_provider.close()


@pytest.mark.asyncio
async def test_bm25_tenant_filtering_shared_branch_id_unrelated_schools(mock_pipeline_repos):
    """Verify different schools sharing a generic branch ID (e.g. 'Main') are isolated in BM25."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos
    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    # 25 chunks from School 2 Branch 'Main'
    sch002_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-branch-sch002-{i}",
            document_id="doc-sch002-branch",
            generation=1,
            content=f"Working hours Main campus guidelines clause {i}.",
            document_name="School 2 Branch Policy",
            school_id="SCH002",
            branch_id="Main",
        )
        for i in range(25)
    ]

    # 2 chunks from School 1 Branch 'Main'
    sch001_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-branch-sch001-{i}",
            document_id="doc-sch001-branch",
            generation=1,
            content=f"Working hours Main campus guidelines for School 1 clause {i}.",
            document_name="School 1 Branch Policy",
            school_id="SCH001",
            branch_id="Main",
        )
        for i in range(2)
    ]

    bm25_provider.index_chunks(sch002_chunks + sch001_chunks)

    async def mock_get_by_id(doc_id: str):
        if doc_id == "doc-sch001-branch":
            return {
                "document_id": "doc-sch001-branch",
                "document_name": "School 1 Branch Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "school_id": "SCH001",
                "branch_id": "Main",
            }
        return {
            "document_id": doc_id,
            "document_name": "School 2 Branch Policy",
            "document_type": "policy",
            "version": 1,
            "policy_lifecycle_status": "approved",
            "status": "ready",
            "school_id": "SCH002",
            "branch_id": "Main",
        }

    doc_repo.get_by_id.side_effect = mock_get_by_id
    indexing_repo.get_tracking_record.return_value = {"active_generation": 1, "indexing_status": "indexed"}

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_bm25_top_k=20,
    )

    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="working hours Main campus", top_k=2),
        school_context=SchoolContext(school_id="SCH001", branch_id="Main"),
    )

    assert len(res.chunks) == 2
    for chunk in res.chunks:
        assert chunk.school_id == "SCH001"
        assert chunk.chunk_id.startswith("chk-branch-sch001-")

    bm25_provider.close()


@pytest.mark.asyncio
async def test_bm25_tenant_filtering_group_and_global_policies(mock_pipeline_repos):
    """Verify legitimate group-wide and global policies are retrieved while sibling schools are excluded."""
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos
    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    # 1. Group-wide policy (school_id is None)
    group_chunk = BM25ChunkRecord(
        chunk_id="chk-group-wide",
        document_id="doc-group",
        generation=1,
        content="Group trust maternity policy for all institutions in the network.",
        document_name="Trust Policy",
        group_id="GRP001",
    )
    # 2. Global policy (school_id is None, is_global=True)
    global_chunk = BM25ChunkRecord(
        chunk_id="chk-global-policy",
        document_id="doc-global",
        generation=1,
        content="Global code of conduct policy applicable across all organizations.",
        document_name="Global Conduct Policy",
        is_global=True,
    )
    # 3. Sibling school policy in same group (school_id='SCH002', group_id='GRP001')
    sibling_chunk = BM25ChunkRecord(
        chunk_id="chk-sibling-school",
        document_id="doc-sibling",
        generation=1,
        content="School 2 specific trust maternity policy supplement.",
        document_name="School 2 Supplement",
        school_id="SCH002",
        group_id="GRP001",
    )

    bm25_provider.index_chunks([group_chunk, global_chunk, sibling_chunk])

    async def mock_get_by_id(doc_id: str):
        if doc_id == "doc-group":
            return {
                "document_id": "doc-group",
                "document_name": "Trust Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "group_id": "GRP001",
            }
        elif doc_id == "doc-global":
            return {
                "document_id": "doc-global",
                "document_name": "Global Conduct Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "is_global": True,
            }
        return {
            "document_id": doc_id,
            "document_name": "School 2 Supplement",
            "document_type": "policy",
            "version": 1,
            "policy_lifecycle_status": "approved",
            "status": "ready",
            "school_id": "SCH002",
            "group_id": "GRP001",
        }

    doc_repo.get_by_id.side_effect = mock_get_by_id
    indexing_repo.get_tracking_record.return_value = {"active_generation": 1, "indexing_status": "indexed"}

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_bm25_top_k=20,
    )

    # Query for School 1 in Group 1
    res = await retrieval_service.retrieve(
        request=RetrievalRequest(question="trust maternity conduct policy", top_k=5),
        school_context=SchoolContext(school_id="SCH001", group_id="GRP001"),
    )

    retrieved_ids = [c.chunk_id for c in res.chunks]
    # Group policy and global policy should be retrieved
    assert "chk-group-wide" in retrieved_ids
    assert "chk-global-policy" in retrieved_ids
    # Sibling school policy MUST NOT be retrieved for School 1
    assert "chk-sibling-school" not in retrieved_ids

    bm25_provider.close()


@pytest.mark.asyncio
async def test_controlled_overfetching_retrieves_when_candidates_fail_mongodb_validation(mock_pipeline_repos):
    """Verify controlled overfetching expands candidate pool when initial pool candidates fail MongoDB validation.

    Scenario: The top 20 BM25 candidates for SCH001 belong to 'administrative' staff.
    The user is querying for 'teaching' staff.
    The matching 'teaching' chunks are located at ranks 21-22 in SQLite.
    Controlled overfetching re-queries with expanded pool size (up to 100), retrieving the valid teaching chunks.
    """
    doc_repo, indexing_repo, vector_repo, emb_provider = mock_pipeline_repos
    vector_repo.query_vectors.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    vector_repo.count.return_value = 0

    bm25_provider = SQLiteFTSBM25IndexProvider(db_path=":memory:")

    # 20 administrative staff chunks (will be rejected by employee applicability validation)
    admin_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-admin-{i}",
            document_id="doc-admin",
            generation=1,
            content=f"Probation period policy for administrative personnel clause {i}.",
            document_name="Admin Policy",
            school_id="SCH001",
            staff_category="administrative",
        )
        for i in range(20)
    ]

    # 2 teaching staff chunks (ranked 21 and 22 in BM25)
    teaching_chunks = [
        BM25ChunkRecord(
            chunk_id=f"chk-teach-{i}",
            document_id="doc-teach",
            generation=1,
            content=f"Probation period policy for teaching faculty clause {i}.",
            document_name="Faculty Policy",
            school_id="SCH001",
            staff_category="teaching",
        )
        for i in range(2)
    ]

    bm25_provider.index_chunks(admin_chunks + teaching_chunks)
    assert bm25_provider.count() == 22

    async def mock_get_by_id(doc_id: str):
        if doc_id == "doc-teach":
            return {
                "document_id": "doc-teach",
                "document_name": "Faculty Policy",
                "document_type": "policy",
                "version": 1,
                "policy_lifecycle_status": "approved",
                "status": "ready",
                "school_id": "SCH001",
            }
        return {
            "document_id": "doc-admin",
            "document_name": "Admin Policy",
            "document_type": "policy",
            "version": 1,
            "policy_lifecycle_status": "approved",
            "status": "ready",
            "school_id": "SCH001",
        }

    doc_repo.get_by_id.side_effect = mock_get_by_id
    indexing_repo.get_tracking_record.return_value = {"active_generation": 1, "indexing_status": "indexed"}

    retrieval_service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=emb_provider,
        bm25_provider=bm25_provider,
        retrieval_mode="hybrid",
        hybrid_bm25_top_k=20,  # Initial pool size is 20
    )

    # Query for 'teaching' staff with top_k=2
    res = await retrieval_service.retrieve(
        request=RetrievalRequest(
            question="probation period policy",
            employee_context=EmployeeContext(staff_category="teaching"),
            top_k=2,
        ),
        school_context=SchoolContext(school_id="SCH001"),
    )

    # Overfetching expanded pool from 20 to 60 and retrieved the teaching chunks
    assert len(res.chunks) == 2
    for chunk in res.chunks:
        assert chunk.chunk_id.startswith("chk-teach-")

    bm25_provider.close()


def test_rrf_and_final_reranking_tracing():
    """Verify trace of RRF scoring vs Phase 7 composite reranking.

    Confirms:
    1. In retrieval, candidates are sorted by RRF score descending.
    2. RRF rank becomes the 'original_rank' in DeterministicV1Reranker.
    3. Final rerank order is computed via composite score (0.5 sem + 0.3 lex + 0.2 head).
    4. RRF rank acts as a deterministic tie-breaker when rerank and similarity scores tie.
    """
    # Create two chunks with identical content and heading, but different RRF rank
    chunk_high_rrf = RetrievedChunk(
        chunk_id="chk-high-rrf",
        document_id="doc-01",
        document_name="Policy",
        document_type="policy",
        version=1,
        content="Standard code of conduct applies to all staff.",
        heading="Conduct Code",
        heading_path=["Conduct"],
        page_start=1,
        page_end=1,
        similarity_score=0.85,
        dense_score=0.85,
        dense_rank=1,
        bm25_score=2.0,
        bm25_rank=1,
        rrf_score=0.032787,  # Rank 1 in RRF
        retrieval_mode="hybrid",
    )

    chunk_low_rrf = RetrievedChunk(
        chunk_id="chk-low-rrf",
        document_id="doc-02",
        document_name="Policy",
        document_type="policy",
        version=1,
        content="Standard code of conduct applies to all staff.",
        heading="Conduct Code",
        heading_path=["Conduct"],
        page_start=1,
        page_end=1,
        similarity_score=0.85,
        dense_score=0.85,
        dense_rank=2,
        bm25_score=1.5,
        bm25_rank=2,
        rrf_score=0.032258,  # Rank 2 in RRF
        retrieval_mode="hybrid",
    )

    # Pass in order of RRF score (high_rrf first)
    chunks = [chunk_high_rrf, chunk_low_rrf]

    reranker = DeterministicV1Reranker(semantic_weight=0.5, lexical_weight=0.3, heading_weight=0.2)
    ranked = reranker.rerank(
        query="standard code conduct",
        chunks=chunks,
        keywords=["conduct"],
    )

    # Both chunks have identical similarity, lexical, and heading scores
    assert ranked[0].rerank_score == ranked[1].rerank_score
    assert ranked[0].similarity_score == ranked[1].similarity_score

    # Verify RRF rank (original_rank) broke the tie
    assert ranked[0].chunk_id == "chk-high-rrf"
    assert ranked[0].original_rank == 1
    assert ranked[0].rerank_rank == 1

    assert ranked[1].chunk_id == "chk-low-rrf"
    assert ranked[1].original_rank == 2
    assert ranked[1].rerank_rank == 2



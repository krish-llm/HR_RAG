"""Unit and integration tests for Phase 7 Context Building and API."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.api.dependencies import SchoolContext
from app.main import app
from app.modules.context.builder import ContextBuilder
from app.modules.context.router import get_context_service
from app.modules.context.schemas import ContextRequest, ContextResponse
from app.modules.context.service import ContextService
from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.router import get_retrieval_service
from app.modules.retrieval.schemas import (
    QueryUnderstandingResult,
    RetrievalResponse,
    RetrievedChunk,
)
from app.modules.retrieval.service import RetrievalService


def _create_sample_ranked_chunk(
    chunk_id: str,
    content: str,
    rerank_score: float = 0.8,
    similarity_score: float = 0.75,
    heading: str = "Leave Rules",
    heading_path: list = None,
    doc_name: str = "HR Policy Manual",
    page_start: int = 2,
    page_end: int = 3,
    version: int = 1,
) -> RankedChunk:
    """Helper to instantiate mock RankedChunk objects."""
    return RankedChunk(
        chunk_id=chunk_id,
        document_id="doc-uuid-999",
        document_name=doc_name,
        document_type="policy",
        version=version,
        content=content,
        heading=heading,
        heading_path=heading_path or [heading],
        page_start=page_start,
        page_end=page_end,
        similarity_score=similarity_score,
        school_id="sch-delhi-01",
        group_id="grp-national",
        branch_id="br-main",
        effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        effective_to=datetime(2026, 12, 31, tzinfo=timezone.utc),
        rerank_score=rerank_score,
        lexical_score=0.7,
        heading_score=0.8,
        original_rank=1,
        rerank_rank=1,
        score_explanation="sample explanation",
    )


def test_context_builder_token_counting():
    """Verify ContextBuilder correctly tokenizes text using tiktoken."""
    builder = ContextBuilder(default_token_budget=1000)
    count = builder.count_tokens("The quick brown fox jumps over the lazy dog.")
    assert count > 0
    assert builder.count_tokens("") == 0


def test_context_builder_exact_and_near_deduplication():
    """Verify identical or near-duplicate chunks are pruned without dropping distinct clauses."""
    builder = ContextBuilder(default_token_budget=3000)

    # Chunk 1: Primary clause
    chunk1 = _create_sample_ranked_chunk(
        chunk_id="chk-1",
        content="Teaching faculty are entitled to 15 days of earned leave annually upon completion of one year of service.",
        heading="Earned Leave",
        heading_path=["Leave Policy", "Earned Leave"],
    )

    # Chunk 2: Exact duplicate content (e.g. from duplicate ingestion or overlapping segment)
    chunk2 = _create_sample_ranked_chunk(
        chunk_id="chk-2",
        content="  teaching faculty are entitled to 15 days of earned leave annually upon completion of one year of service.  ",
        heading="Earned Leave",
        heading_path=["Leave Policy", "Earned Leave"],
    )

    # Chunk 3: Near duplicate with same heading and 95% word overlap
    chunk3 = _create_sample_ranked_chunk(
        chunk_id="chk-3",
        content="Teaching faculty are entitled to 15 days of earned leave annually upon full completion of one year of service.",
        heading="Earned Leave",
        heading_path=["Leave Policy", "Earned Leave"],
    )

    # Chunk 4: Distinct policy clause with different heading, must NOT be pruned
    chunk4 = _create_sample_ranked_chunk(
        chunk_id="chk-4",
        content="Non-teaching staff are entitled to 20 days of earned leave annually.",
        heading="Non-Teaching Earned Leave",
        heading_path=["Leave Policy", "Non-Teaching Earned Leave"],
    )

    res = builder.build_context(chunks=[chunk1, chunk2, chunk3, chunk4])

    # Should keep chunk1 and chunk4, pruning chunk2 (exact) and chunk3 (near)
    assert res.selected_chunks_count == 2
    assert res.skipped_chunks_count == 2
    assert len(res.source_manifest) == 2
    assert "Teaching faculty are entitled to 15 days" in res.assembled_context
    assert "Non-teaching staff are entitled to 20 days" in res.assembled_context


def test_context_builder_budget_enforcement_skips_rather_than_truncates():
    """Verify that complete chunks are preserved and oversized chunks are skipped rather than truncated."""
    # Set a small token budget that only fits the first chunk
    chunk1 = _create_sample_ranked_chunk(
        chunk_id="chk-fit",
        content="Staff members must notify their supervisor 24 hours prior to taking casual leave.",
        heading="Casual Leave Notification",
    )

    # Very large chunk that will exceed budget
    large_table_content = "\n".join([f"| Clause {i} | Specific detailed regulation description number {i} |" for i in range(150)])
    chunk2 = _create_sample_ranked_chunk(
        chunk_id="chk-oversized",
        content=large_table_content,
        heading="Comprehensive Clause Schedule",
    )

    # Budget big enough for chunk1 (~70 tokens) but not chunk1 + chunk2 (~600+ tokens)
    builder = ContextBuilder(default_token_budget=150)
    res = builder.build_context(chunks=[chunk1, chunk2], token_budget=150)

    assert res.selected_chunks_count == 1
    assert res.skipped_chunks_count == 1
    assert res.total_tokens <= 150
    assert "Staff members must notify their supervisor" in res.assembled_context
    assert "Clause Schedule" not in res.assembled_context


def test_context_builder_structured_source_manifest():
    """Verify generated source citations and manifest follow required format."""
    builder = ContextBuilder(default_token_budget=2000)
    chunk = _create_sample_ranked_chunk(
        chunk_id="chk-source-test",
        content="Maternity leave of 26 weeks is provided with full pay.",
        heading="Maternity Leave",
        heading_path=["Leave Policy", "Maternity Leave"],
        doc_name="Staff Policy 2026",
        page_start=5,
        page_end=7,
        version=2,
    )

    res = builder.build_context([chunk])
    assert len(res.source_manifest) == 1
    item = res.source_manifest[0]

    assert item.citation_index == 1
    assert item.citation_label == "[Source 1]"
    assert item.citation_reference == "[Source 1: Staff Policy 2026, pages 5–7, Leave Policy > Maternity Leave]"
    assert item.document_id == "doc-uuid-999"
    assert item.document_name == "Staff Policy 2026"
    assert item.version == 2
    assert item.page_start == 5
    assert item.page_end == 7
    assert item.school_id == "sch-delhi-01"

    # Verify context text formatting
    assert "### [Source 1: Staff Policy 2026, pages 5–7, Leave Policy > Maternity Leave]" in res.assembled_context
    assert "Maternity leave of 26 weeks is provided" in res.assembled_context


def test_context_builder_empty_chunks():
    """Verify empty input chunks result in empty context."""
    builder = ContextBuilder(default_token_budget=1000)
    res = builder.build_context([])
    assert res.assembled_context == ""
    assert res.source_manifest == []
    assert res.total_tokens == 0
    assert res.selected_chunks_count == 0
    assert res.skipped_chunks_count == 0


@pytest.mark.asyncio
async def test_context_service_orchestration_success():
    """Verify ContextService coordinates retrieval, reranking, and context building."""
    mock_retrieval_service = MagicMock(spec=RetrievalService)

    sample_retrieved = RetrievedChunk(
        chunk_id="chk-retrieved-1",
        document_id="doc-uuid-1",
        document_name="Leave Guidelines",
        document_type="policy",
        version=1,
        content="Sick leave requires a medical certificate if extending beyond 3 consecutive days.",
        heading="Sick Leave Requirements",
        heading_path=["Leave Policy", "Sick Leave Requirements"],
        page_start=3,
        page_end=3,
        similarity_score=0.82,
        school_id="sch-01",
    )

    mock_retrieval_service.retrieve = AsyncMock(
        return_value=RetrievalResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="sick leave medical certificate rules",
                normalized_query="sick leave medical certificate rules",
                intent="leave",
                extracted_keywords=["sick leave", "medical certificate"],
                requested_scope={"school_id": "sch-01"},
            ),
            chunks=[sample_retrieved],
            total_candidates=1,
            total_results=1,
            retrieval_status="success",
        )
    )

    context_service = ContextService(
        retrieval_service=mock_retrieval_service,
        default_top_k=5,
        default_token_budget=2000,
    )

    req = ContextRequest(
        question="sick leave medical certificate rules",
        token_budget=1500,
    )
    school_ctx = SchoolContext(school_id="sch-01")

    res = await context_service.build_context(req, school_ctx)

    assert res.status == "success"
    assert res.selected_chunks_count == 1
    assert len(res.ranked_chunks) == 1
    assert len(res.source_manifest) == 1
    assert res.token_budget == 1500
    assert "medical certificate" in res.assembled_context


@pytest.mark.asyncio
async def test_context_service_orchestration_empty_results():
    """Verify ContextService handles empty retrieval results gracefully."""
    mock_retrieval_service = MagicMock(spec=RetrievalService)
    mock_retrieval_service.retrieve = AsyncMock(
        return_value=RetrievalResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="unrelated query",
                normalized_query="unrelated query",
                intent="other",
                extracted_keywords=[],
            ),
            chunks=[],
            total_candidates=0,
            total_results=0,
            retrieval_status="empty",
        )
    )

    context_service = ContextService(
        retrieval_service=mock_retrieval_service,
        default_top_k=5,
        default_token_budget=2000,
    )

    req = ContextRequest(question="unrelated query")
    school_ctx = SchoolContext(school_id="sch-01")

    res = await context_service.build_context(req, school_ctx)

    assert res.status == "empty"
    assert res.selected_chunks_count == 0
    assert res.assembled_context == ""
    assert res.source_manifest == []


def test_api_context_endpoint_success():
    """Verify POST /api/v1/query/context endpoint works with dependency overrides."""
    mock_service = MagicMock(spec=ContextService)
    mock_service.build_context = AsyncMock(
        return_value=ContextResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="What is the casual leave policy?",
                normalized_query="casual leave policy",
                intent="leave",
                extracted_keywords=["casual leave"],
                requested_scope={"school_id": "sch-01"},
            ),
            ranked_chunks=[],
            assembled_context="### [Source 1: HR Policy, page 2, Leave]\nCasual leave is 12 days.",
            source_manifest=[],
            total_tokens=25,
            token_budget=2000,
            selected_chunks_count=1,
            skipped_chunks_count=0,
            status="success",
        )
    )

    app.dependency_overrides[get_context_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/query/context",
            headers={"X-School-ID": "sch-01"},
            json={
                "question": "What is the casual leave policy?",
                "top_k": 5,
                "token_budget": 2000,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "success"
        assert "Casual leave is 12 days." in data["data"]["assembled_context"]
    finally:
        app.dependency_overrides.pop(get_context_service, None)


def test_api_context_endpoint_empty_question_validation_error():
    """Verify POST /api/v1/query/context fails with 422 on empty question string."""
    mock_service = MagicMock(spec=ContextService)
    app.dependency_overrides[get_context_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/query/context",
            headers={"X-School-ID": "sch-01"},
            json={"question": ""},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_context_service, None)


def test_both_retrieve_and_context_endpoints_accessible():
    """Verify both /retrieve and /context endpoints remain accessible and functional."""
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="test",
                normalized_query="test",
                intent="other",
                extracted_keywords=[],
            ),
            chunks=[],
            total_candidates=0,
            total_results=0,
            retrieval_status="empty",
        )
    )

    mock_context_svc = MagicMock(spec=ContextService)
    mock_context_svc.build_context = AsyncMock(
        return_value=ContextResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="test",
                normalized_query="test",
                intent="other",
                extracted_keywords=[],
            ),
            ranked_chunks=[],
            assembled_context="",
            source_manifest=[],
            total_tokens=0,
            token_budget=2000,
            selected_chunks_count=0,
            skipped_chunks_count=0,
            status="empty",
        )
    )

    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_svc
    app.dependency_overrides[get_context_service] = lambda: mock_context_svc
    client = TestClient(app)

    try:
        res_retrieve = client.post(
            "/api/v1/query/retrieve",
            headers={"X-School-ID": "sch-01"},
            json={"question": "test"},
        )
        assert res_retrieve.status_code == 200

        res_context = client.post(
            "/api/v1/query/context",
            headers={"X-School-ID": "sch-01"},
            json={"question": "test"},
        )
        assert res_context.status_code == 200
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)
        app.dependency_overrides.pop(get_context_service, None)

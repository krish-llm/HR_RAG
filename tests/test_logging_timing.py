"""Tests for structured logging, execution timing, and request ID propagation."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.api.dependencies import SchoolContext
from app.core.exceptions import AppException, ExtractionException
from app.core.logging_utils import (
    TimedBlock,
    clear_request_id,
    format_log_msg,
    get_request_id,
    log_operation_completed,
    log_operation_failed,
    log_operation_start,
    set_request_id,
)
from app.main import app
from app.modules.context.builder import ContextBuilder
from app.modules.context.schemas import ContextRequest
from app.modules.context.service import ContextService
from app.modules.extraction.schemas import DocumentExtractionResult, ExtractedPage
from app.modules.extraction.service import ExtractionService
from app.modules.generation.schemas import AnswerRequest, LLMAnswerRaw
from app.modules.generation.service import AnswerService
from app.modules.reranking.reranker import DeterministicV1Reranker
from app.modules.retrieval.schemas import QueryUnderstandingResult, RetrievalResponse, RetrievedChunk
from app.modules.retrieval.service import RetrievalService


# ==============================================================================
# 1. Unit Tests for ContextVar & Log Formatter
# ==============================================================================


def test_request_id_context():
    """Verify request_id context variable set, get, auto-generate, and clear."""
    clear_request_id()
    assert get_request_id() == "-"

    # Set custom ID
    custom_id = set_request_id("custom-req-123")
    assert custom_id == "custom-req-123"
    assert get_request_id() == "custom-req-123"

    # Auto-generate 8-character ID
    auto_id = set_request_id()
    assert len(auto_id) == 8
    assert get_request_id() == auto_id

    clear_request_id()
    assert get_request_id() == "-"


def test_format_log_msg():
    """Verify standard format: request_id=... | module=... | status=... | duration_ms=..."""
    set_request_id("test-req-456")

    msg = format_log_msg(
        module="query_embedding",
        status="completed",
        duration_ms=420.45,
        candidates=20,
    )
    clear_request_id()

    assert "request_id=test-req-456" in msg
    assert "module=query_embedding" in msg
    assert "status=completed" in msg
    assert "duration_ms=420" in msg
    assert "candidates=20" in msg


def test_format_log_msg_override_request_id():
    """Verify explicit request_id can override the context variable."""
    set_request_id("global-ctx-id")

    msg = format_log_msg(
        module="vector_search",
        status="completed",
        duration_ms=12.0,
        request_id="explicit-id-789",
    )
    clear_request_id()

    assert "request_id=explicit-id-789" in msg
    assert "global-ctx-id" not in msg


# ==============================================================================
# 2. Unit Tests for TimedBlock (Sync & Async)
# ==============================================================================


def test_timed_block_sync_success(caplog):
    """Verify TimedBlock sync execution logs started and completed with duration."""
    set_request_id("sync-test-id")
    test_logger = logging.getLogger("test.sync")

    with caplog.at_level(logging.INFO):
        with TimedBlock(test_logger, "test_sync_module") as timer:
            timer.set_extra(items=5)

    clear_request_id()

    messages = [rec.message for rec in caplog.records if "test_sync_module" in rec.message]
    assert len(messages) == 2
    assert "status=started" in messages[0]
    assert "request_id=sync-test-id" in messages[0]
    assert "status=completed" in messages[1]
    assert "duration_ms=" in messages[1]
    assert "items=5" in messages[1]


def test_timed_block_sync_failure(caplog):
    """Verify TimedBlock sync records status=failed and error on unhandled exception."""
    set_request_id("fail-test-id")
    test_logger = logging.getLogger("test.failure")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="Operation error"):
            with TimedBlock(test_logger, "test_failing_module") as timer:
                timer.set_extra(detail="some_context")
                raise ValueError("Operation error")

    clear_request_id()

    messages = [rec.message for rec in caplog.records if "test_failing_module" in rec.message]
    assert len(messages) == 2
    assert "status=started" in messages[0]
    assert "status=failed" in messages[1]
    assert "duration_ms=" in messages[1]
    assert "error=Operation error" in messages[1]
    assert "detail=some_context" in messages[1]


@pytest.mark.asyncio
async def test_timed_block_async_success(caplog):
    """Verify TimedBlock async execution logs started and completed with duration."""
    set_request_id("async-test-id")
    test_logger = logging.getLogger("test.async")

    with caplog.at_level(logging.INFO):
        async with TimedBlock(test_logger, "test_async_module") as timer:
            await asyncio.sleep(0.01)
            timer.set_extra(tokens=150)

    clear_request_id()

    messages = [rec.message for rec in caplog.records if "test_async_module" in rec.message]
    assert len(messages) == 2
    assert "status=started" in messages[0]
    assert "status=completed" in messages[1]
    assert "duration_ms=" in messages[1]
    assert "tokens=150" in messages[1]


# ==============================================================================
# 3. HTTP Request ID Middleware Tests
# ==============================================================================


def test_middleware_propagates_incoming_request_id():
    """Verify X-Request-ID header in request is preserved in response header."""
    client = TestClient(app)
    response = client.get("/health", headers={"X-Request-ID": "client-req-999"})

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "client-req-999"


def test_middleware_generates_request_id_if_missing():
    """Verify X-Request-ID is generated and returned when not provided by client."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    generated_id = response.headers.get("X-Request-ID")
    assert generated_id is not None
    assert len(generated_id) == 8


# ==============================================================================
# 4. Instrumented Pipeline Execution & Log Output Verification
# ==============================================================================


@pytest.mark.asyncio
async def test_context_service_instrumentation(caplog):
    """Verify ContextService logs 'reranking' and 'context_builder' timing."""
    set_request_id("ctx-service-test")

    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query_understanding=QueryUnderstandingResult(
            original_query="What is the leave policy?",
            normalized_query="leave policy",
            intent="leave",
            extracted_keywords=["leave"],
        ),
        chunks=[
            RetrievedChunk(
                chunk_id="chunk-01",
                document_id="doc-01",
                document_name="Leave Policy",
                document_type="policy",
                version=1,
                content="Employees are entitled to 15 days of annual leave.",
                heading="Annual Leave",
                page_start=1,
                page_end=1,
                similarity_score=0.92,
            )
        ],
        total_candidates=1,
        total_results=1,
        retrieval_status="success",
    )

    context_service = ContextService(
        retrieval_service=mock_retrieval,
        reranker=DeterministicV1Reranker(),
        context_builder=ContextBuilder(default_token_budget=1000),
    )

    with caplog.at_level(logging.INFO):
        response = await context_service.build_context(
            request=ContextRequest(question="What is the leave policy?"),
            school_context=SchoolContext(school_id="SCH001"),
        )

    clear_request_id()

    assert response.status == "success"
    all_msgs = [rec.message for rec in caplog.records]

    # Verify reranking logged
    rerank_logs = [m for m in all_msgs if "module=reranking" in m]
    assert any("status=started" in m for m in rerank_logs)
    assert any("status=completed" in m and "duration_ms=" in m for m in rerank_logs)

    # Verify context_builder logged
    builder_logs = [m for m in all_msgs if "module=context_builder" in m]
    assert any("status=started" in m for m in builder_logs)
    assert any("status=completed" in m and "duration_ms=" in m for m in builder_logs)


@pytest.mark.asyncio
async def test_extraction_service_failure_logging(caplog):
    """Verify ExtractionService logs status=failed and error details on failure."""
    set_request_id("extract-fail-test")

    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-fail",
        "stored_file_path": "non_existent.pdf",
        "document_name": "Policy.pdf",
    }
    mock_extractor = MagicMock()
    mock_extractor.extract.side_effect = RuntimeError("PDF corrupted or unreadable")

    service = ExtractionService(
        extraction_repo=AsyncMock(),
        document_repo=mock_doc_repo,
        extractor=mock_extractor,
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(ExtractionException):
            await service.extract_document("doc-fail")

    clear_request_id()

    all_msgs = [rec.message for rec in caplog.records]
    extract_logs = [m for m in all_msgs if "module=pdf_extraction" in m]

    assert any("status=started" in m for m in extract_logs)
    failure_log = next((m for m in extract_logs if "status=failed" in m), None)
    assert failure_log is not None
    assert "duration_ms=" in failure_log
    assert "error=PDF corrupted or unreadable" in failure_log


@pytest.mark.asyncio
async def test_full_qa_pipeline_logs_sample(caplog):
    """Execute end-to-end question answering pipeline and verify structured logs for all modules."""
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query_understanding=QueryUnderstandingResult(
            original_query="What is the probationary period for teaching staff?",
            normalized_query="probationary period teaching staff",
            intent="probation",
            extracted_keywords=["probationary", "teaching"],
        ),
        chunks=[
            RetrievedChunk(
                chunk_id="chk-prob-01",
                document_id="doc-prob-01",
                document_name="Staff Policy 2026",
                document_type="policy",
                version=1,
                content="Teaching staff probation period is 6 months from date of joining.",
                heading="Probation Period",
                page_start=2,
                page_end=2,
                similarity_score=0.95,
            )
        ],
        total_candidates=1,
        total_results=1,
        retrieval_status="success",
    )

    mock_llm_provider = MagicMock()
    mock_llm_provider.get_model_name.return_value = "gpt-4o"
    mock_llm_provider.generate = AsyncMock(return_value=(
        '{"answer": "Teaching staff have a 6-month probation period [Source 1].", '
        '"citations": ["[Source 1]"], "insufficient_evidence": false, '
        '"conflicting_evidence": false, "limitations": null}'
    ))

    context_service = ContextService(
        retrieval_service=mock_retrieval,
        reranker=DeterministicV1Reranker(),
        context_builder=ContextBuilder(default_token_budget=1000),
    )
    answer_service = AnswerService(
        context_service=context_service,
        llm_provider=mock_llm_provider,
    )

    set_request_id("qa-sample-req-001")
    with caplog.at_level(logging.INFO):
        response = await answer_service.generate_answer(
            request=AnswerRequest(question="What is the probationary period for teaching staff?"),
            school_context=SchoolContext(school_id="SCH001"),
        )
    clear_request_id()

    assert response.status == "success"
    all_msgs = [rec.message for rec in caplog.records]

    # Verify structured logs for each module in QA pipeline
    assert any("module=reranking" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=context_builder" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=llm_generation" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)


@pytest.mark.asyncio
async def test_retrieval_service_modules_logging(caplog):
    """Verify RetrievalService logs query_understanding, query_embedding, vector_search, and mongo_validation."""
    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-01",
        "document_name": "Leave Policy",
        "document_type": "policy",
        "version": 1,
        "policy_lifecycle_status": "approved",
        "status": "ready",
        "school_id": "SCH001",
    }

    mock_indexing_repo = AsyncMock()
    mock_indexing_repo.get_tracking_record.return_value = {
        "document_id": "doc-01",
        "active_generation": 1,
    }

    mock_vector_repo = MagicMock()
    mock_vector_repo.query_vectors.return_value = {
        "ids": [["chunk-01"]],
        "documents": [["Annual leave is 15 days."]],
        "metadatas": [[{
            "chunk_id": "chunk-01",
            "document_id": "doc-01",
            "generation": 1,
            "school_id": "SCH001",
        }]],
        "distances": [[0.1]],
    }
    mock_vector_repo.count.return_value = 1

    mock_embedding_provider = AsyncMock()
    mock_embedding_provider.embed_texts.return_value = [[0.1] * 1536]

    retrieval_service = RetrievalService(
        document_repo=mock_doc_repo,
        indexing_repo=mock_indexing_repo,
        vector_repo=mock_vector_repo,
        embedding_provider=mock_embedding_provider,
    )

    from app.modules.retrieval.schemas import RetrievalRequest

    set_request_id("retrieval-test-req-002")
    with caplog.at_level(logging.INFO):
        res = await retrieval_service.retrieve(
            request=RetrievalRequest(question="How many days annual leave?"),
            school_context=SchoolContext(school_id="SCH001"),
        )
    clear_request_id()

    assert res.retrieval_status in ("success", "empty")
    all_msgs = [rec.message for rec in caplog.records]

    assert any("module=query_understanding" in m and "status=completed" in m for m in all_msgs)
    assert any("module=query_embedding" in m and "status=completed" in m and "duration_ms=" in m for m in all_msgs)
    assert any("module=vector_search" in m and "status=completed" in m and "candidates=1" in m for m in all_msgs)
    assert any("module=mongo_validation" in m and "status=completed" in m and "candidates=1" in m for m in all_msgs)



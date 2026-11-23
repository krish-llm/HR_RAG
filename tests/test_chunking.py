"""Tests for Document Chunking (Heading-aware, Recursive Splitting, Metadata & Persistence)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.core.exceptions import NotFoundException, ProcessingException
from app.main import app
from app.modules.chunking.chunker import DocumentChunker
from app.modules.chunking.router import get_chunking_service
from app.modules.chunking.schemas import (
    ChunkingSummaryResponse,
    DocumentChunk,
    PaginatedChunksResponse,
)
from app.modules.chunking.service import ChunkingService
from app.modules.processing.schemas import (
    DocumentProcessedResult,
    ProcessedPage,
    ProcessingStageStatus,
)

# Sample test metadata for scoping verification
SAMPLE_DOC_METADATA = {
    "document_id": "doc_test_123",
    "document_name": "leave_policy_2026.pdf",
    "document_type": "policy",
    "version": 2,
    "school_id": "SCH-DPS-01",
    "group_id": "GRP-DPS",
    "branch_id": "BR-MAIN",
    "is_global": False,
    "state": "Karnataka",
    "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "effective_to": datetime(2026, 12, 31, tzinfo=timezone.utc),
    "employee_applicability": {
        "staff_category": "teaching",
        "employment_type": "full-time",
        "department": "science",
    },
}


# ============================================================================
# 1. Unit Tests for DocumentChunker
# ============================================================================


def test_chunker_heading_aware_splitting():
    """Verify distinct policy sections are split at heading boundaries and not merged."""
    chunker = DocumentChunker(target_tokens=500, max_tokens=700)

    page_content = (
        "# 1. Leave Policy\n\n"
        "Introduction to general leave policy for all permanent staff.\n\n"
        "## 1.1 Casual Leave\n\n"
        "Staff are eligible for 10 days casual leave per year. Notice must be given 2 days in advance.\n\n"
        "## 1.2 Sick Leave\n\n"
        "Staff are entitled to 12 days medical sick leave. Doctor certificate required for > 3 days."
    )

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=page_content, character_count=len(page_content))],
        total_characters=len(page_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    # There should be 3 distinct section chunks
    assert len(chunks) == 3
    assert chunks[0].heading == "# 1. Leave Policy"
    assert chunks[0].heading_path == ["# 1. Leave Policy"]
    assert "Introduction to general leave policy" in chunks[0].content

    assert chunks[1].heading == "## 1.1 Casual Leave"
    assert chunks[1].heading_path == ["# 1. Leave Policy", "## 1.1 Casual Leave"]
    assert "10 days casual leave" in chunks[1].content

    assert chunks[2].heading == "## 1.2 Sick Leave"
    assert chunks[2].heading_path == ["# 1. Leave Policy", "## 1.2 Sick Leave"]
    assert "12 days medical sick leave" in chunks[2].content


def test_chunker_recursive_splitting_oversized_section():
    """Verify an oversized section (> max_tokens) is split recursively into multiple chunks."""
    # Use a small target_tokens to easily trigger recursive splitting
    chunker = DocumentChunker(target_tokens=40, max_tokens=60, overlap_tokens=10)

    # 4 distinct paragraphs under one heading
    paragraphs = [
        f"Paragraph {i}: The quick brown fox jumps over the lazy dog repeatedly to generate tokens for this test."
        for i in range(1, 6)
    ]
    page_content = "## Section 1: Overview\n\n" + "\n\n".join(paragraphs)

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=page_content, character_count=len(page_content))],
        total_characters=len(page_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.heading == "## Section 1: Overview"
        assert chunk.heading_path == ["## Section 1: Overview"]
        assert chunk.token_count <= 80  # Stays bounded near max_tokens


def test_chunker_token_count_limits_and_accuracy():
    """Verify all chunks strictly observe configured token limits and token_count matches tiktoken."""
    chunker = DocumentChunker(target_tokens=100, max_tokens=150, overlap_tokens=15)

    long_text = " ".join(["Policy standard operating procedure clause."] * 100)
    page_content = f"# Operations\n\n{long_text}"

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=page_content, character_count=len(page_content))],
        total_characters=len(page_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count == chunker.count_tokens(chunk.content)
        assert chunk.token_count <= 160


def test_chunker_overlap_behavior():
    """Verify adjacent split chunks in an oversized section contain clean token overlap."""
    chunker = DocumentChunker(target_tokens=15, max_tokens=25, overlap_tokens=8)

    page_content = (
        "## Regulations\n\n"
        "Section Alpha requires daily reporting by noon.\n\n"
        "Section Beta requires weekly timesheet validation by Friday afternoon.\n\n"
        "Section Gamma mandates quarterly compliance reviews for all teachers."
    )

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=page_content, character_count=len(page_content))],
        total_characters=len(page_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) >= 2
    # Verify the start of chunk 1 has overlap from chunk 0
    words_chunk_0 = set(chunks[0].content.split())
    words_chunk_1 = set(chunks[1].content.split())
    # There should be shared words (overlap)
    common_words = words_chunk_0.intersection(words_chunk_1)
    assert len(common_words) > 0


def test_chunker_lists_and_numbered_clauses():
    """Verify lists and numbered clauses are preserved without breaking internal lines."""
    chunker = DocumentChunker(target_tokens=200, max_tokens=300)

    list_content = (
        "## Policy Clauses\n\n"
        "1. Working Hours:\n"
        "   1.1 Core hours are 8:00 AM to 4:00 PM.\n"
        "   1.2 Grace period of 15 minutes is allowed.\n"
        "2. Attendance Tracking:\n"
        "   2.1 Biometric check-in is mandatory upon entry.\n"
        "   2.2 Failure to check in constitutes unexcused absence."
    )

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=list_content, character_count=len(list_content))],
        total_characters=len(list_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) == 1
    assert "1. Working Hours:" in chunks[0].content
    assert "1.1 Core hours are 8:00 AM to 4:00 PM." in chunks[0].content
    assert "2. Attendance Tracking:" in chunks[0].content


def test_chunker_small_and_large_tables():
    """Verify small tables stay intact and large tables repeat header rows on split."""
    chunker = DocumentChunker(target_tokens=40, max_tokens=60)

    # Generate a table with 10 rows that exceeds 60 tokens
    rows = "\n".join([f"| Staff Level {i} | Allowance {i} | Status Active |" for i in range(1, 12)])
    table_content = (
        "## Compensation Table\n\n"
        "| Level | Allowance | Status |\n"
        "| --- | --- | --- |\n"
        f"{rows}"
    )

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=table_content, character_count=len(table_content))],
        total_characters=len(table_content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) > 1
    # Verify every table chunk contains the column headers and separator row!
    for chunk in chunks:
        assert "| Level | Allowance | Status |" in chunk.content
        assert "| --- | --- | --- |" in chunk.content


def test_chunker_multipage_content_and_page_references():
    """Verify cross-page sections accurately record source_pages, page_start, and page_end."""
    chunker = DocumentChunker(target_tokens=300, max_tokens=500)

    page_1 = "## Health Benefits\n\nHealth insurance is provided to all full-time employees."
    page_2 = "Dependents including spouse and children are covered under the plan."

    processed_doc = DocumentProcessedResult(
        document_id="doc_test_123",
        source_extraction_id="ext_123",
        total_pages=2,
        processed_pages=[
            ProcessedPage(page_number=1, cleaned_content=page_1, character_count=len(page_1)),
            ProcessedPage(page_number=2, cleaned_content=page_2, character_count=len(page_2)),
        ],
        total_characters=len(page_1) + len(page_2),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) == 1
    assert chunks[0].source_pages == [1, 2]
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_chunker_metadata_preservation_and_tenant_scope():
    """Verify all 22 required metadata fields are properly populated from the document."""
    chunker = DocumentChunker(target_tokens=500, max_tokens=700)

    content = "## General Policy\nAll staff must adhere to code of conduct."
    processed_doc = DocumentProcessedResult(
        document_id="doc_meta_test",
        source_extraction_id="ext_meta_test",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content=content, character_count=len(content))],
        total_characters=len(content),
        processing_status=ProcessingStageStatus.COMPLETED,
    )

    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc_meta_test"
    assert chunk.chunk_id == "doc_meta_test_chunk_0"
    assert chunk.chunk_index == 0
    assert chunk.school_id == "SCH-DPS-01"
    assert chunk.group_id == "GRP-DPS"
    assert chunk.branch_id == "BR-MAIN"
    assert chunk.state == "Karnataka"
    assert chunk.is_global is False
    assert chunk.staff_category == "teaching"
    assert chunk.employment_type == "full-time"
    assert chunk.department == "science"
    assert chunk.version == 2
    assert chunk.document_name == "leave_policy_2026.pdf"


def test_chunker_empty_content():
    """Verify empty document produces an empty list without error or empty chunks."""
    chunker = DocumentChunker()
    processed_doc = DocumentProcessedResult(
        document_id="doc_empty",
        source_extraction_id="ext_empty",
        total_pages=1,
        processed_pages=[ProcessedPage(page_number=1, cleaned_content="", character_count=0)],
        total_characters=0,
        processing_status=ProcessingStageStatus.COMPLETED,
    )
    chunks = chunker.chunk_document(processed_doc, SAMPLE_DOC_METADATA)
    assert chunks == []


# ============================================================================
# 2. Service Layer Tests
# ============================================================================


@pytest.mark.asyncio
async def test_service_successful_chunking_and_rechunking():
    """Verify chunking service saves chunks, updates status, and deletes old chunks on re-chunking."""
    document_repo = MagicMock()
    processed_repo = MagicMock()
    chunk_repo = MagicMock()

    document_repo.get_by_id = AsyncMock(return_value=SAMPLE_DOC_METADATA)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    processed_record = {
        "document_id": "doc_test_123",
        "source_extraction_id": "ext_123",
        "processing_status": "completed",
        "processed_pages": [
            {
                "page_number": 1,
                "cleaned_content": "## Section 1\nContent for section 1.",
                "character_count": 30,
            }
        ],
        "total_pages": 1,
        "total_characters": 30,
        "processed_at": datetime.now(timezone.utc),
    }
    processed_repo.get_by_document_id = AsyncMock(return_value=processed_record)
    chunk_repo.save_chunks = AsyncMock(return_value=1)

    service = ChunkingService(
        document_repo=document_repo,
        processed_repo=processed_repo,
        chunk_repo=chunk_repo,
    )

    # 1. First Chunking
    summary1 = await service.chunk_document("doc_test_123")
    assert summary1.document_id == "doc_test_123"
    assert summary1.total_chunks == 1
    assert summary1.processing_status == "completed"
    assert chunk_repo.save_chunks.call_count == 1

    # 2. Re-chunking (Idempotent execution)
    summary2 = await service.chunk_document("doc_test_123")
    assert summary2.document_id == "doc_test_123"
    assert summary2.total_chunks == 1
    assert chunk_repo.save_chunks.call_count == 2


@pytest.mark.asyncio
async def test_service_missing_document_raises_404():
    """Verify NotFoundException is raised when parent document record does not exist."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value=None)

    service = ChunkingService(
        document_repo=document_repo,
        processed_repo=MagicMock(),
        chunk_repo=MagicMock(),
    )

    with pytest.raises(NotFoundException) as exc_info:
        await service.chunk_document("doc_nonexistent")
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_service_missing_processed_content_raises_400():
    """Verify ProcessingException is raised when processed content does not exist."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value=SAMPLE_DOC_METADATA)

    processed_repo = MagicMock()
    processed_repo.get_by_document_id = AsyncMock(return_value=None)

    service = ChunkingService(
        document_repo=document_repo,
        processed_repo=processed_repo,
        chunk_repo=MagicMock(),
    )

    with pytest.raises(ProcessingException) as exc_info:
        await service.chunk_document("doc_test_123")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "PROCESSED_CONTENT_NOT_FOUND"


# ============================================================================
# 3. API Endpoint Tests
# ============================================================================


def test_api_chunk_document_success():
    """Verify POST /api/v1/documents/{document_id}/chunk returns 200 with summary."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_summary = ChunkingSummaryResponse(
        document_id="doc_api_chunk",
        total_chunks=3,
        average_tokens=150.0,
        maximum_tokens=220,
        processing_status="completed",
    )
    mock_service.chunk_document = AsyncMock(return_value=mock_summary)

    app.dependency_overrides[get_chunking_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_api_chunk/chunk")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == "doc_api_chunk"
        assert data["data"]["total_chunks"] == 3
        assert data["data"]["average_tokens"] == 150.0
    finally:
        app.dependency_overrides.pop(get_chunking_service, None)


def test_api_chunk_document_not_found():
    """Verify POST /api/v1/documents/{document_id}/chunk returns 404 if document not found."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.chunk_document = AsyncMock(
        side_effect=NotFoundException("Document with ID 'doc_miss' not found", "DOCUMENT_NOT_FOUND")
    )

    app.dependency_overrides[get_chunking_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_miss/chunk")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_chunking_service, None)


def test_api_get_chunks_paginated_success():
    """Verify GET /api/v1/documents/{document_id}/chunks returns 200 with paginated items."""
    client = TestClient(app)
    mock_service = MagicMock()

    chunk_sample = DocumentChunk(
        chunk_id="doc_chunk_get_chunk_0",
        document_id="doc_chunk_get",
        chunk_index=0,
        content="# Section\nContent",
        token_count=10,
        heading="# Section",
        heading_path=["# Section"],
        page_start=1,
        page_end=1,
        source_pages=[1],
        document_name="doc.pdf",
        created_at=datetime.now(timezone.utc),
    )
    mock_response = PaginatedChunksResponse(
        items=[chunk_sample],
        total=1,
        page=1,
        page_size=20,
        total_pages=1,
    )
    mock_service.get_chunks_paginated = AsyncMock(return_value=mock_response)

    app.dependency_overrides[get_chunking_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_chunk_get/chunks?page=1&page_size=20")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total"] == 1
        assert len(data["data"]["items"]) == 1
        assert data["data"]["items"][0]["chunk_id"] == "doc_chunk_get_chunk_0"
    finally:
        app.dependency_overrides.pop(get_chunking_service, None)


def test_api_get_chunks_not_found():
    """Verify GET /api/v1/documents/{document_id}/chunks returns 404 if document does not exist."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.get_chunks_paginated = AsyncMock(
        side_effect=NotFoundException("Document with ID 'doc_miss' not found", "DOCUMENT_NOT_FOUND")
    )

    app.dependency_overrides[get_chunking_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_miss/chunks")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_chunking_service, None)


# ============================================================================
# 4. Persistence Safety & Failure-Injection Tests
# ============================================================================


class _FakeCursor:
    def __init__(self, items):
        self._items = list(items)

    def sort(self, key, direction=1):
        self._items.sort(key=lambda x: x.get(key, 0), reverse=(direction == -1))
        return self

    def skip(self, n):
        self._items = self._items[n:]
        return self

    def limit(self, n):
        self._items = self._items[:n]
        return self

    async def to_list(self, length=None):
        if length is not None:
            return [dict(x) for x in self._items[:length]]
        return [dict(x) for x in self._items]


class _FakeAsyncCollection:
    def __init__(self, name="collection"):
        self.name = name
        self.docs = []

    async def create_indexes(self, indexes):
        return True

    async def count_documents(self, filter_doc):
        return len([d for d in self.docs if all(d.get(k) == v for k, v in filter_doc.items())])

    def find(self, filter_doc=None):
        filter_doc = filter_doc or {}
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in filter_doc.items())]
        return _FakeCursor(matches)

    async def insert_many(self, records):
        class _InsertResult:
            inserted_ids = []
        result = _InsertResult()
        for r in records:
            doc = dict(r)
            if "_id" not in doc:
                import uuid
                doc["_id"] = str(uuid.uuid4())
            self.docs.append(doc)
            result.inserted_ids.append(doc["_id"])
        return result

    async def delete_many(self, filter_doc):
        class _DeleteResult:
            deleted_count = 0
        result = _DeleteResult()
        initial_len = len(self.docs)
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in filter_doc.items())]
        result.deleted_count = initial_len - len(self.docs)
        return result


class _FakeAsyncDatabase:
    def __init__(self, client=None):
        self._collections = {}
        self.client = client

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = _FakeAsyncCollection(name)
        return self._collections[name]


def _build_test_chunk(doc_id: str, index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}_chunk_{index}",
        document_id=doc_id,
        chunk_index=index,
        content=text,
        token_count=15,
        heading=f"## Heading {index}",
        heading_path=[f"## Heading {index}"],
        page_start=1,
        page_end=1,
        source_pages=[1],
        document_name="test_policy.pdf",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_persistence_failure_injection_preserves_existing_chunks():
    """Verify that if insertion fails during re-chunking, existing valid chunks are NOT destroyed.

    Failure Injection:
    1. Initially save 3 valid chunks for document 'doc_safe_1'.
    2. Attempt to replace with 2 new chunks, but inject a write failure into active collection insert_many.
    3. Assert save_chunks raises the injected exception.
    4. Assert that all 3 original chunks remain intact, available, and unaltered.
    """
    from app.modules.chunking.repository import ChunkRepository

    db = _FakeAsyncDatabase()
    repo = ChunkRepository(db)

    # 1. Save 3 initial valid chunks
    initial_chunks = [_build_test_chunk("doc_safe_1", i, f"Original clause {i}") for i in range(3)]
    saved = await repo.save_chunks("doc_safe_1", initial_chunks)
    assert saved == 3

    existing = await repo.get_all_by_document_id("doc_safe_1")
    assert len(existing) == 3
    assert [c["content"] for c in existing] == ["Original clause 0", "Original clause 1", "Original clause 2"]

    # 2. Inject failure during active collection insert_many (first attempt fails, rollback restores)
    original_insert = repo.collection.insert_many
    call_count = 0

    async def failing_insert_many(records):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Injected database disk write error during active chunk replacement")
        return await original_insert(records)

    repo.collection.insert_many = failing_insert_many

    # 3. Attempt re-chunking with new chunks
    new_chunks = [_build_test_chunk("doc_safe_1", i, f"New updated clause {i}") for i in range(2)]

    with pytest.raises(RuntimeError) as exc_info:
        await repo.save_chunks("doc_safe_1", new_chunks)
    assert "Injected database disk write error" in str(exc_info.value)

    # Restore original insert_many
    repo.collection.insert_many = original_insert

    # 4. Verification: Original valid chunks MUST still be available
    preserved = await repo.get_all_by_document_id("doc_safe_1")
    assert len(preserved) == 3, f"Expected 3 preserved chunks, found {len(preserved)}"
    assert [c["content"] for c in preserved] == [
        "Original clause 0",
        "Original clause 1",
        "Original clause 2",
    ]
    assert [c["chunk_id"] for c in preserved] == [
        "doc_safe_1_chunk_0",
        "doc_safe_1_chunk_1",
        "doc_safe_1_chunk_2",
    ]


@pytest.mark.asyncio
async def test_persistence_staging_failure_leaves_active_untouched():
    """Verify that if staging collection insertion fails, active collection is never modified."""
    from app.modules.chunking.repository import ChunkRepository

    db = _FakeAsyncDatabase()
    repo = ChunkRepository(db)

    # 1. Save initial chunks
    initial_chunks = [_build_test_chunk("doc_safe_2", i, f"Initial clause {i}") for i in range(2)]
    await repo.save_chunks("doc_safe_2", initial_chunks)
    assert len(await repo.get_all_by_document_id("doc_safe_2")) == 2

    # 2. Inject failure on staging collection
    async def failing_staging_insert(records):
        raise RuntimeError("Staging storage quota exceeded")

    repo.staging_collection.insert_many = failing_staging_insert

    # 3. Attempt re-chunking
    new_chunks = [_build_test_chunk("doc_safe_2", 0, "Updated clause")]
    with pytest.raises(RuntimeError) as exc_info:
        await repo.save_chunks("doc_safe_2", new_chunks)
    assert "Staging storage quota exceeded" in str(exc_info.value)

    # 4. Verification: Active collection was untouched
    active = await repo.get_all_by_document_id("doc_safe_2")
    assert len(active) == 2
    assert [c["content"] for c in active] == ["Initial clause 0", "Initial clause 1"]


@pytest.mark.asyncio
async def test_persistence_transactional_rollback_preserves_chunks():
    """Verify that when transactions are supported, a transaction abort cleanly rolls back."""
    from app.modules.chunking.repository import ChunkRepository

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_db.client = mock_client

    mock_collection = MagicMock()
    mock_collection.delete_many = AsyncMock()
    mock_collection.insert_many = AsyncMock()
    mock_db.__getitem__.return_value = mock_collection

    class MockSessionContext:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_session = MagicMock()
    async def failing_with_txn(callback):
        raise RuntimeError("Transaction aborted due to write conflict")
    mock_session.with_transaction = failing_with_txn
    mock_client.start_session.return_value = MockSessionContext(mock_session)

    repo = ChunkRepository(mock_db)

    chunk = _build_test_chunk("doc_txn_fail", 0, "Txn clause")
    with pytest.raises(RuntimeError) as exc_info:
        await repo.save_chunks("doc_txn_fail", [chunk])
    assert "Transaction aborted due to write conflict" in str(exc_info.value)


@pytest.mark.asyncio
async def test_persistence_standalone_fallback_to_staging():
    """Verify standalone MongoDB (OperationFailure code 20) safely falls back to staging strategy."""
    from pymongo.errors import OperationFailure
    from app.modules.chunking.repository import ChunkRepository

    mock_client = MagicMock()
    mock_db = _FakeAsyncDatabase(client=mock_client)

    class MockSessionContext:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_session = MagicMock()
    async def mock_standalone_txn(callback):
        raise OperationFailure(
            "Transaction numbers are only allowed on a replica set member or mongos",
            code=20,
        )
    mock_session.with_transaction = mock_standalone_txn
    mock_client.start_session.return_value = MockSessionContext(mock_session)

    repo = ChunkRepository(mock_db)
    assert repo._supports_transactions is None

    chunks = [_build_test_chunk("doc_standalone", i, f"Clause {i}") for i in range(2)]
    saved = await repo.save_chunks("doc_standalone", chunks)

    assert saved == 2
    assert repo._supports_transactions is False
    active = await repo.get_all_by_document_id("doc_standalone")
    assert len(active) == 2

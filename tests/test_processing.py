"""Tests for Document Processing (Cleaning, Normalization & Structure Preservation)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.core.exceptions import NotFoundException, ProcessingException
from app.main import app
from app.modules.extraction.schemas import (
    DocumentExtractionResult,
    ExtractedPage,
    ExtractionStatus,
)
from app.modules.processing.cleaner import DocumentCleaner
from app.modules.processing.router import get_processing_service
from app.modules.processing.schemas import (
    DocumentProcessedResult,
    ProcessedPage,
    ProcessingStageStatus,
)
from app.modules.processing.service import ProcessingService


# ============================================================================
# Unit Tests for DocumentCleaner
# ============================================================================


def test_cleaner_normal_text_and_whitespace():
    """Verify cleaner normalizes paragraph spacing, collapses spaces and strips trailing whitespace."""
    cleaner = DocumentCleaner()
    raw = (
        "  First paragraph with   multiple   spaces.  \n"
        "Still in first paragraph.  \n"
        "\n\n\n\n"
        "Second paragraph after excessive newlines.  \n"
    )
    cleaned = cleaner.clean_page(raw)

    assert "First paragraph with multiple spaces." in cleaned
    assert "Still in first paragraph." in cleaned
    assert "Second paragraph after excessive newlines." in cleaned
    # Ensure excessive newlines were collapsed to \n\n
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("First")
    assert cleaned.endswith("newlines.")


def test_cleaner_repeated_headers_and_footers():
    """Verify repeated headers and template footers are identified and stripped across pages."""
    cleaner = DocumentCleaner()

    page_1 = (
        "Apex Trust HR Policy Manual\n"
        "# Section 1: Overview\n"
        "This is the policy overview.\n"
        "Confidential - Page 1 of 3\n"
    )
    page_2 = (
        "Apex Trust HR Policy Manual\n"
        "# Section 2: Eligibility\n"
        "All permanent employees are eligible.\n"
        "Confidential - Page 2 of 3\n"
    )
    page_3 = (
        "Apex Trust HR Policy Manual\n"
        "# Section 3: Benefits\n"
        "Standard benefits apply.\n"
        "Confidential - Page 3 of 3\n"
    )

    extraction = DocumentExtractionResult(
        document_id="doc_header_test",
        document_name="policy.pdf",
        total_pages=3,
        extracted_pages=[
            ExtractedPage(page_number=1, content=page_1, character_count=len(page_1)),
            ExtractedPage(page_number=2, content=page_2, character_count=len(page_2)),
            ExtractedPage(page_number=3, content=page_3, character_count=len(page_3)),
        ],
        total_characters=len(page_1) + len(page_2) + len(page_3),
        extraction_status=ExtractionStatus.COMPLETED,
    )

    result = cleaner.clean_document(extraction)

    assert result.total_pages == 3
    for p in result.processed_pages:
        # Running header and running footer should be removed
        assert "Apex Trust HR Policy Manual" not in p.cleaned_content
        assert "Confidential - Page" not in p.cleaned_content

    # Content should remain
    assert "# Section 1: Overview" in result.processed_pages[0].cleaned_content
    assert "# Section 2: Eligibility" in result.processed_pages[1].cleaned_content
    assert "# Section 3: Benefits" in result.processed_pages[2].cleaned_content


def test_cleaner_structure_preservation_headings_and_lists():
    """Verify headings, bulleted lists, and numbered lists are preserved and reliably detected."""
    cleaner = DocumentCleaner()
    raw = (
        "# Leave Policy\n"
        "General introduction to leave.\n"
        "## Casual Leave\n"
        "- Entitlement: 10 days\n"
        "- Notice period: 2 days\n"
        "## Sick Leave\n"
        "1. Medical certificate required for > 3 days\n"
        "2. Contact HR department immediately\n"
    )
    cleaned = cleaner.clean_page(raw)
    structure = cleaner.detect_structure(cleaned)

    assert structure.has_headings is True
    assert structure.has_lists is True
    assert structure.has_tables is False
    assert "# Leave Policy" in structure.headings
    assert "## Casual Leave" in structure.headings
    assert "## Sick Leave" in structure.headings
    assert structure.lists_count >= 2
    assert structure.paragraphs_count >= 1

    # Verify content preserved
    assert "- Entitlement: 10 days" in cleaned
    assert "1. Medical certificate required for > 3 days" in cleaned


def test_cleaner_tables_preservation():
    """Verify Markdown tables, pipe delimiters, and alignment rows are preserved intact."""
    cleaner = DocumentCleaner()
    table_text = (
        "# Allowance Table\n\n"
        "| Category | Entitlement | Minimum Service |\n"
        "| --- | --- | --- |\n"
        "| Full-Time Staff | $50,000 | 1 Year |\n"
        "| Part-Time Staff | $25,000 | 2 Years |\n\n"
        "Applies to all branches.\n"
    )
    cleaned = cleaner.clean_page(table_text)
    structure = cleaner.detect_structure(cleaned)

    assert structure.has_tables is True
    assert structure.tables_count == 1
    assert "| Category | Entitlement | Minimum Service |" in cleaned
    assert "| --- | --- | --- |" in cleaned
    assert "| Full-Time Staff | $50,000 | 1 Year |" in cleaned


def test_cleaner_empty_page():
    """Verify empty pages result in empty cleaned text and zeroed structure stats."""
    cleaner = DocumentCleaner()
    cleaned = cleaner.clean_page("")
    assert cleaned == ""

    structure = cleaner.detect_structure(cleaned)
    assert structure.has_headings is False
    assert structure.has_tables is False
    assert structure.has_lists is False
    assert structure.headings == []
    assert structure.tables_count == 0
    assert structure.lists_count == 0
    assert structure.paragraphs_count == 0


def test_cleaner_artifacts_page_noise_and_value_preservation():
    """Verify extraction artifacts and standalone page numbers are removed while preserving dates, currency and policy codes."""
    cleaner = DocumentCleaner()
    raw = (
        "\x00\x0c\ufeff"  # Form feed, null byte, BOM
        "Page 1\n"  # Standalone page number
        "# Policy Ref: HR-POL-2026.01\n"
        "Effective from 15/04/2026.\n"
        "Base stipend is ₹45,000 or $550.00.\n"
        "Zero\u200bwidth and soft\xadhyphen artifacts.\n"
        "- 1 -\n"  # Bottom page number artifact
    )
    cleaned = cleaner.clean_page(raw)

    # Noise removed
    assert "\x0c" not in cleaned
    assert "\u200b" not in cleaned
    assert "\xad" not in cleaned
    assert "Page 1" not in cleaned
    assert "- 1 -" not in cleaned

    # Values, currency, dates, punctuation preserved
    assert "# Policy Ref: HR-POL-2026.01" in cleaned
    assert "15/04/2026" in cleaned
    assert "₹45,000" in cleaned
    assert "$550.00" in cleaned
    assert "HR-POL-2026.01" in cleaned


def test_cleaner_single_page_document_no_false_header_removal():
    """Verify single page documents do not mistakenly remove top titles as repeated headers."""
    cleaner = DocumentCleaner()
    single_page = (
        "# Standalone Policy Document\n"
        "This policy applies exclusively to the main campus.\n"
    )
    extraction = DocumentExtractionResult(
        document_id="doc_single",
        document_name="single.pdf",
        total_pages=1,
        extracted_pages=[ExtractedPage(page_number=1, content=single_page, character_count=len(single_page))],
        total_characters=len(single_page),
        extraction_status=ExtractionStatus.COMPLETED,
    )
    result = cleaner.clean_document(extraction)

    assert result.total_pages == 1
    assert "# Standalone Policy Document" in result.processed_pages[0].cleaned_content


# ============================================================================
# Service Layer Tests
# ============================================================================


@pytest.mark.asyncio
async def test_service_successful_processing_and_reprocessing_upsert():
    """Verify processing service fetches extraction, cleans content, updates status, and supports reprocessing."""
    document_repo = MagicMock()
    extraction_repo = MagicMock()
    processed_repo = MagicMock()

    doc_meta = {
        "document_id": "doc_123",
        "document_name": "handbook.pdf",
        "processing_status": "completed",
    }
    document_repo.get_by_id = AsyncMock(return_value=doc_meta)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    page_content = "# Welcome\nWelcome to our school network.\nPage 1\n"
    extraction_record = {
        "document_id": "doc_123",
        "document_name": "handbook.pdf",
        "total_pages": 1,
        "extracted_pages": [
            {"page_number": 1, "content": page_content, "character_count": len(page_content)}
        ],
        "total_characters": len(page_content),
        "extraction_status": "completed",
        "extracted_at": datetime.now(timezone.utc),
    }
    extraction_repo.get_by_document_id = AsyncMock(return_value=extraction_record)
    processed_repo.save_processed = AsyncMock(return_value={"document_id": "doc_123"})

    service = ProcessingService(
        document_repo=document_repo,
        extraction_repo=extraction_repo,
        processed_repo=processed_repo,
    )

    # First run
    result1 = await service.process_document("doc_123")
    assert result1.document_id == "doc_123"
    assert result1.total_pages == 1
    assert result1.processing_status == ProcessingStageStatus.COMPLETED
    assert processed_repo.save_processed.call_count == 1

    # Second run (Reprocessing / Upsert)
    result2 = await service.process_document("doc_123")
    assert result2.document_id == "doc_123"
    assert processed_repo.save_processed.call_count == 2
    # Extraction was NOT modified
    extraction_repo.save_extraction.assert_not_called() if hasattr(extraction_repo, "save_extraction") else None


@pytest.mark.asyncio
async def test_service_missing_document_raises_404():
    """Verify NotFoundException is raised when the document record does not exist."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value=None)

    service = ProcessingService(
        document_repo=document_repo,
        extraction_repo=MagicMock(),
        processed_repo=MagicMock(),
    )

    with pytest.raises(NotFoundException) as exc_info:
        await service.process_document("doc_nonexistent")
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_service_missing_extraction_raises_400():
    """Verify ProcessingException is raised when the document has not yet been extracted."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_no_ext"})

    extraction_repo = MagicMock()
    extraction_repo.get_by_document_id = AsyncMock(return_value=None)

    service = ProcessingService(
        document_repo=document_repo,
        extraction_repo=extraction_repo,
        processed_repo=MagicMock(),
    )

    with pytest.raises(ProcessingException) as exc_info:
        await service.process_document("doc_no_ext")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "EXTRACTION_NOT_FOUND"


@pytest.mark.asyncio
async def test_service_failed_extraction_state_raises_400():
    """Verify ProcessingException is raised if the extraction status was failed."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_failed_ext"})

    extraction_repo = MagicMock()
    extraction_repo.get_by_document_id = AsyncMock(
        return_value={
            "document_id": "doc_failed_ext",
            "document_name": "corrupt.pdf",
            "total_pages": 0,
            "extracted_pages": [],
            "total_characters": 0,
            "extraction_status": "failed",
            "extracted_at": datetime.now(timezone.utc),
            "error_message": "Corrupted PDF file",
        }
    )

    service = ProcessingService(
        document_repo=document_repo,
        extraction_repo=extraction_repo,
        processed_repo=MagicMock(),
    )

    with pytest.raises(ProcessingException) as exc_info:
        await service.process_document("doc_failed_ext")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "EXTRACTION_FAILED_STATE"


@pytest.mark.asyncio
async def test_service_processing_failure_updates_status():
    """Verify processing failure marks document as failed and leaves extraction untouched."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_err"})
    document_repo.update_processing_status = AsyncMock(return_value=True)

    extraction_repo = MagicMock()
    extraction_repo.get_by_document_id = AsyncMock(
        return_value={
            "document_id": "doc_err",
            "document_name": "doc.pdf",
            "total_pages": 1,
            "extracted_pages": [{"page_number": 1, "content": "test", "character_count": 4}],
            "total_characters": 4,
            "extraction_status": "completed",
            "extracted_at": datetime.now(timezone.utc),
        }
    )

    cleaner = MagicMock()
    cleaner.clean_document = MagicMock(side_effect=RuntimeError("Unexpected cleaner error"))

    service = ProcessingService(
        document_repo=document_repo,
        extraction_repo=extraction_repo,
        processed_repo=MagicMock(),
        cleaner=cleaner,
    )

    with pytest.raises(ProcessingException):
        await service.process_document("doc_err")

    # Document processing_status should be marked as failed
    last_call = document_repo.update_processing_status.call_args_list[-1]
    assert last_call.kwargs["processing_status"] == "failed"
    assert "Unexpected cleaner error" in last_call.kwargs["error_message"]


# ============================================================================
# API Endpoint Tests
# ============================================================================


def test_api_process_document_success():
    """Verify POST /api/v1/documents/{document_id}/process triggers processing and returns 200."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_result = DocumentProcessedResult(
        document_id="doc_api_1",
        source_extraction_id="ext_doc_api_1_123456",
        total_pages=2,
        processed_pages=[
            ProcessedPage(page_number=1, cleaned_content="Page 1 cleaned", character_count=14),
            ProcessedPage(page_number=2, cleaned_content="Page 2 cleaned", character_count=14),
        ],
        total_characters=28,
        processing_status=ProcessingStageStatus.COMPLETED,
        processed_at=datetime.now(timezone.utc),
    )
    mock_service.process_document = AsyncMock(return_value=mock_result)

    app.dependency_overrides[get_processing_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_api_1/process")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == "doc_api_1"
        assert data["data"]["total_pages"] == 2
        assert data["data"]["processed_pages_count"] == 2
        assert data["data"]["total_characters"] == 28
        assert data["data"]["processing_status"] == "completed"
    finally:
        app.dependency_overrides.pop(get_processing_service, None)


def test_api_process_document_not_found():
    """Verify POST /api/v1/documents/{document_id}/process returns 404 if document does not exist."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.process_document = AsyncMock(
        side_effect=NotFoundException("Document with ID 'doc_miss' not found", "DOCUMENT_NOT_FOUND")
    )

    app.dependency_overrides[get_processing_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_miss/process")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_processing_service, None)


def test_api_get_processed_success():
    """Verify GET /api/v1/documents/{document_id}/processed returns 200 with full processed content."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_result = DocumentProcessedResult(
        document_id="doc_api_get",
        source_extraction_id="ext_doc_api_get_123456",
        total_pages=1,
        processed_pages=[
            ProcessedPage(
                page_number=1,
                cleaned_content="# Cleaned Page\nAll noise stripped.",
                character_count=35,
            )
        ],
        total_characters=35,
        processing_status=ProcessingStageStatus.COMPLETED,
        processed_at=datetime.now(timezone.utc),
    )
    mock_service.get_processed = AsyncMock(return_value=mock_result)

    app.dependency_overrides[get_processing_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_api_get/processed")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == "doc_api_get"
        assert len(data["data"]["processed_pages"]) == 1
        assert data["data"]["processed_pages"][0]["page_number"] == 1
        assert "# Cleaned Page" in data["data"]["processed_pages"][0]["cleaned_content"]
    finally:
        app.dependency_overrides.pop(get_processing_service, None)


def test_api_get_processed_not_found():
    """Verify GET /api/v1/documents/{document_id}/processed returns 404 if content does not exist."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.get_processed = AsyncMock(
        side_effect=NotFoundException("Processed content not found", "PROCESSED_CONTENT_NOT_FOUND")
    )

    app.dependency_overrides[get_processing_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_missing/processed")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "PROCESSED_CONTENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_processing_service, None)

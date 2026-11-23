"""Tests for Phase 2 PDF Extraction Module."""

from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.core.exceptions import ExtractionException, NotFoundException
from app.main import app
from app.modules.extraction.extractor import PDFExtractor
from app.modules.extraction.repository import ExtractionRepository
from app.modules.extraction.router import get_extraction_service
from app.modules.extraction.schemas import (
    DocumentExtractionResult,
    ExtractedPage,
    ExtractionStatus,
)
from app.modules.extraction.service import ExtractionService
from app.modules.ingestion.repository import DocumentRepository


def generate_multipage_pdf(file_path: Path) -> None:
    """Generate a 3-page PDF with headings, a markdown table, and an empty page."""
    doc = pymupdf.open()

    # Page 1: Heading & Paragraphs
    p1 = doc.new_page()
    p1.insert_text((50, 72), "# HR Policy 2026: Leave Regulations\n\nAll staff are eligible for standard leave.", fontsize=14)

    # Page 2: Table Data
    p2 = doc.new_page()
    table_text = (
        "## Policy Schedule\n\n"
        "| Leave Type | Annual Allowance | Encashment |\n"
        "| Casual Leave | 12 Days | Not Permitted |\n"
        "| Sick Leave | 15 Days | Permitted |\n"
    )
    p2.insert_text((50, 72), table_text, fontsize=12)

    # Page 3: Completely empty page
    doc.new_page()

    doc.save(str(file_path))
    doc.close()


# ==============================================================================
# 1. PDFExtractor Unit Tests
# ==============================================================================

def test_extractor_multipage_heading_table_and_empty_page():
    """Verify PyMuPDF4LLM extraction preserves headings, tables, boundaries, and handles empty pages."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        pdf_path = Path(temp_dir) / "test_policy.pdf"
        generate_multipage_pdf(pdf_path)

        extractor = PDFExtractor()
        result = extractor.extract(
            file_path_str=str(pdf_path),
            document_id="doc-test-123",
            document_name="test_policy.pdf",
        )

        assert result.document_id == "doc-test-123"
        assert result.document_name == "test_policy.pdf"
        assert result.total_pages == 3
        assert len(result.extracted_pages) == 3
        assert result.extraction_status == ExtractionStatus.COMPLETED

        # Page 1: Heading preservation & correct numbering
        p1 = result.extracted_pages[0]
        assert p1.page_number == 1
        assert "Leave Regulations" in p1.content
        assert p1.character_count > 0

        # Page 2: Table structure preservation
        p2 = result.extracted_pages[1]
        assert p2.page_number == 2
        assert "Leave Type" in p2.content or "Casual Leave" in p2.content
        assert p2.character_count > 0

        # Page 3: Empty page handling
        p3 = result.extracted_pages[2]
        assert p3.page_number == 3
        assert p3.content.strip() == ""
        assert p3.character_count == 0

        # Total characters
        expected_chars = sum(p.character_count for p in result.extracted_pages)
        assert result.total_characters == expected_chars


def test_extractor_missing_file_raises():
    """Verify missing PDF file raises 404 ExtractionException."""
    extractor = PDFExtractor()
    with pytest.raises(ExtractionException) as exc_info:
        extractor.extract(
            file_path_str="./nonexistent_file.pdf",
            document_id="doc-missing",
            document_name="nonexistent.pdf",
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "FILE_NOT_FOUND"


def test_extractor_corrupted_pdf_raises():
    """Verify corrupted PDF file raises 400 ExtractionException."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        corrupted_path = Path(temp_dir) / "corrupt.pdf"
        corrupted_path.write_bytes(b"%PDF-1.4\nCorrupted garbage stream data that fails parser\n")

        extractor = PDFExtractor()
        with pytest.raises(ExtractionException) as exc_info:
            extractor.extract(
                file_path_str=str(corrupted_path),
                document_id="doc-corrupt",
                document_name="corrupt.pdf",
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "CORRUPTED_PDF"


# ==============================================================================
# 2. ExtractionService Unit Tests
# ==============================================================================

@pytest.fixture
def mock_doc_repo():
    """Mock ingestion DocumentRepository."""
    repo = MagicMock(spec=DocumentRepository)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.update_processing_status = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_extract_repo():
    """Mock ExtractionRepository."""
    repo = MagicMock(spec=ExtractionRepository)
    repo.save_extraction = AsyncMock(side_effect=lambda x: x.model_dump(mode="python"))
    repo.get_by_document_id = AsyncMock(return_value=None)
    return repo


@pytest.mark.asyncio
async def test_service_successful_extraction_and_reextraction(mock_doc_repo, mock_extract_repo):
    """Verify successful extraction updates processing status to 'completed' and supports re-extraction."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        pdf_path = Path(temp_dir) / "policy.pdf"
        generate_multipage_pdf(pdf_path)

        mock_doc_repo.get_by_id.return_value = {
            "document_id": "doc-extract-1",
            "document_name": "policy.pdf",
            "stored_file_path": str(pdf_path),
            "processing_status": "pending",
            "status": "uploaded",
        }

        service = ExtractionService(
            extraction_repo=mock_extract_repo,
            document_repo=mock_doc_repo,
            extractor=PDFExtractor(),
        )

        # 1. First extraction
        result1 = await service.extract_document("doc-extract-1")
        assert result1.document_id == "doc-extract-1"
        assert result1.total_pages == 3
        assert result1.extraction_status == ExtractionStatus.COMPLETED

        # Check status updates in MongoDB: processing -> completed/ready
        mock_doc_repo.update_processing_status.assert_any_await(
            document_id="doc-extract-1",
            processing_status="processing",
            error_message=None,
        )
        mock_doc_repo.update_processing_status.assert_any_await(
            document_id="doc-extract-1",
            processing_status="completed",
            error_message=None,
            status="ready",
        )
        mock_extract_repo.save_extraction.assert_awaited_once()

        # 2. Re-extraction replaces result
        result2 = await service.extract_document("doc-extract-1")
        assert result2.document_id == "doc-extract-1"
        assert mock_extract_repo.save_extraction.await_count == 2


@pytest.mark.asyncio
async def test_service_extraction_missing_document(mock_doc_repo, mock_extract_repo):
    """Verify NotFoundException when document does not exist in ingestion repository."""
    mock_doc_repo.get_by_id.return_value = None
    service = ExtractionService(
        extraction_repo=mock_extract_repo,
        document_repo=mock_doc_repo,
    )

    with pytest.raises(NotFoundException):
        await service.extract_document("non-existent-doc")


@pytest.mark.asyncio
async def test_service_extraction_failure_updates_status(mock_doc_repo, mock_extract_repo):
    """Verify failed extraction marks document processing_status as 'failed' and records error message."""
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-missing-file",
        "document_name": "ghost.pdf",
        "stored_file_path": "/missing/path/ghost.pdf",
        "processing_status": "pending",
        "status": "uploaded",
    }

    service = ExtractionService(
        extraction_repo=mock_extract_repo,
        document_repo=mock_doc_repo,
        extractor=PDFExtractor(),
    )

    with pytest.raises(ExtractionException):
        await service.extract_document("doc-missing-file")

    # Document must be updated to status="failed" and processing_status="failed"
    last_call = mock_doc_repo.update_processing_status.call_args
    assert last_call is not None
    assert last_call.kwargs["document_id"] == "doc-missing-file"
    assert last_call.kwargs["processing_status"] == "failed"
    assert last_call.kwargs["status"] == "failed"
    assert "not found on disk" in last_call.kwargs["error_message"]


@pytest.mark.asyncio
async def test_service_get_extraction(mock_doc_repo, mock_extract_repo):
    """Verify get_extraction retrieves stored extraction."""
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-stored-1",
        "document_name": "policy.pdf",
        "processing_status": "completed",
    }
    mock_extract_repo.get_by_document_id.return_value = {
        "document_id": "doc-stored-1",
        "document_name": "policy.pdf",
        "total_pages": 1,
        "extracted_pages": [{"page_number": 1, "content": "Text", "character_count": 4}],
        "total_characters": 4,
        "extraction_status": "completed",
        "extracted_at": "2026-11-18T10:00:00Z",
        "error_message": None,
    }

    service = ExtractionService(
        extraction_repo=mock_extract_repo,
        document_repo=mock_doc_repo,
    )

    extraction = await service.get_extraction("doc-stored-1")
    assert extraction.document_id == "doc-stored-1"
    assert len(extraction.extracted_pages) == 1
    assert extraction.extracted_pages[0].content == "Text"


# ==============================================================================
# 3. FastAPI Route Integration Tests
# ==============================================================================

@pytest.fixture
def override_extraction_service(mock_doc_repo, mock_extract_repo):
    """Dependency override fixture for FastAPI TestClient."""
    test_service = ExtractionService(
        extraction_repo=mock_extract_repo,
        document_repo=mock_doc_repo,
        extractor=PDFExtractor(),
    )
    app.dependency_overrides[get_extraction_service] = lambda: test_service
    yield test_service
    app.dependency_overrides.pop(get_extraction_service, None)


def test_api_extract_document_success(client: TestClient, override_extraction_service, mock_doc_repo):
    """POST /api/v1/documents/{document_id}/extract: Trigger extraction successfully."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        pdf_path = Path(temp_dir) / "sample_policy.pdf"
        generate_multipage_pdf(pdf_path)

        mock_doc_repo.get_by_id.return_value = {
            "document_id": "doc-api-123",
            "document_name": "sample_policy.pdf",
            "stored_file_path": str(pdf_path),
            "processing_status": "pending",
            "status": "uploaded",
        }

        response = client.post("/api/v1/documents/doc-api-123/extract")
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["success"] is True
        summary = res_json["data"]
        assert summary["document_id"] == "doc-api-123"
        assert summary["total_pages"] == 3
        assert summary["extracted_pages_count"] == 3
        assert summary["total_characters"] > 0
        assert summary["extraction_status"] == "completed"


def test_api_extract_document_not_found(client: TestClient, override_extraction_service, mock_doc_repo):
    """POST /api/v1/documents/{document_id}/extract: 404 for missing document."""
    mock_doc_repo.get_by_id.return_value = None

    response = client.post("/api/v1/documents/missing-doc-id/extract")
    assert response.status_code == 404
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "DOCUMENT_NOT_FOUND"


def test_api_get_extraction_success(client: TestClient, override_extraction_service, mock_doc_repo, mock_extract_repo):
    """GET /api/v1/documents/{document_id}/extraction: Retrieve extraction details."""
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-api-123",
        "processing_status": "completed",
    }
    mock_extract_repo.get_by_document_id.return_value = {
        "document_id": "doc-api-123",
        "document_name": "sample_policy.pdf",
        "total_pages": 2,
        "extracted_pages": [
            {"page_number": 1, "content": "# Heading\nText", "character_count": 14},
            {"page_number": 2, "content": "| A | B |", "character_count": 9},
        ],
        "total_characters": 23,
        "extraction_status": "completed",
        "extracted_at": "2026-11-18T10:00:00Z",
        "error_message": None,
    }

    response = client.get("/api/v1/documents/doc-api-123/extraction")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    data = res_json["data"]
    assert data["document_id"] == "doc-api-123"
    assert data["total_pages"] == 2
    assert len(data["extracted_pages"]) == 2
    assert data["extracted_pages"][0]["content"] == "# Heading\nText"
    assert data["extracted_pages"][1]["content"] == "| A | B |"


def test_api_get_extraction_not_found(client: TestClient, override_extraction_service, mock_doc_repo, mock_extract_repo):
    """GET /api/v1/documents/{document_id}/extraction: 404 when extraction record not present."""
    mock_doc_repo.get_by_id.return_value = {
        "document_id": "doc-api-123",
        "processing_status": "pending",
    }
    mock_extract_repo.get_by_document_id.return_value = None

    response = client.get("/api/v1/documents/doc-api-123/extraction")
    assert response.status_code == 404
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "EXTRACTION_NOT_FOUND"

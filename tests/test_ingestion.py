"""Tests for Phase 1 Document Ingestion Module."""

import io
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    DuplicateDocumentException,
    FileValidationException,
    NotFoundException,
)
from app.main import app
from app.modules.ingestion.repository import DocumentRepository
from app.modules.ingestion.router import get_ingestion_service
from app.modules.ingestion.schemas import (
    DocumentCreate,
    DocumentFilterParams,
    DocumentResponse,
    DocumentStatus,
    ProcessingStatus,
)
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import LocalStorageService

# A standard minimal valid PDF header and trailer
MINIMAL_VALID_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f\n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n149\n%%EOF"
)


# ==============================================================================
# 1. LocalStorageService Tests
# ==============================================================================

def test_storage_valid_pdf_validation():
    """Verify validation passes for valid PDF content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir, max_file_size_bytes=1024 * 1024)
        size, file_hash = storage.validate_pdf_content(MINIMAL_VALID_PDF, "sample.pdf")
        assert size == len(MINIMAL_VALID_PDF)
        assert len(file_hash) == 64  # SHA-256 hex string


def test_storage_invalid_file_extension():
    """Verify rejection of non-PDF extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir)
        with pytest.raises(FileValidationException) as exc:
            storage.validate_pdf_content(MINIMAL_VALID_PDF, "sample.docx")
        assert exc.value.error_code == "INVALID_FILE_TYPE"


def test_storage_empty_file():
    """Verify rejection of empty 0-byte file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir)
        with pytest.raises(FileValidationException) as exc:
            storage.validate_pdf_content(b"", "empty.pdf")
        assert exc.value.error_code == "EMPTY_FILE"


def test_storage_oversized_file():
    """Verify rejection of file exceeding max size limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set max size to 100 bytes
        storage = LocalStorageService(upload_dir=temp_dir, max_file_size_bytes=100)
        oversized = MINIMAL_VALID_PDF * 5
        with pytest.raises(FileValidationException) as exc:
            storage.validate_pdf_content(oversized, "big.pdf")
        assert exc.value.error_code == "FILE_TOO_LARGE"


def test_storage_corrupted_signature():
    """Verify rejection of files missing standard %PDF- magic bytes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir)
        fake_content = b"NOT_A_PDF_FILE_HEADER_TEXT_DATA_FOR_TESTING"
        with pytest.raises(FileValidationException) as exc:
            storage.validate_pdf_content(fake_content, "fake.pdf")
        assert exc.value.error_code == "INVALID_PDF_SIGNATURE"


def test_storage_path_traversal_prevention():
    """Verify path traversal filenames are safely sanitized and kept in upload_dir."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir)
        saved_path_str = storage.save_file(
            document_id="doc-123",
            original_filename="../../../../../etc/passwd.pdf",
            content=MINIMAL_VALID_PDF,
        )
        saved_path = Path(saved_path_str)
        assert saved_path.exists()
        # Verify it stays strictly within the temp_dir
        assert saved_path.is_relative_to(Path(temp_dir).resolve())
        assert "passwd.pdf" in saved_path.name
        assert ".." not in saved_path.name


def test_storage_deletion():
    """Verify safe deletion of stored file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageService(upload_dir=temp_dir)
        saved_path_str = storage.save_file("doc-999", "policy.pdf", MINIMAL_VALID_PDF)
        assert Path(saved_path_str).exists()

        deleted = storage.delete_file(saved_path_str)
        assert deleted is True
        assert not Path(saved_path_str).exists()

        # Deleting non-existent returns False
        assert storage.delete_file(saved_path_str) is False


# ==============================================================================
# 2. IngestionService Tests
# ==============================================================================

@pytest.fixture
def mock_repo():
    """Create a mock DocumentRepository."""
    repo = MagicMock(spec=DocumentRepository)
    repo.find_duplicate = AsyncMock(return_value=None)
    repo.insert_document = AsyncMock(side_effect=lambda doc: doc.model_dump(mode="python"))
    repo.get_by_id = AsyncMock(return_value=None)
    repo.list_documents = AsyncMock(return_value=([], 0))
    repo.delete_by_id = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def temp_storage():
    """Create a real LocalStorageService in a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield LocalStorageService(upload_dir=temp_dir)


@pytest.mark.asyncio
async def test_service_successful_upload(mock_repo, temp_storage):
    """Verify service successfully validates, stores, and registers a PDF."""
    service = IngestionService(repository=mock_repo, storage=temp_storage)

    mock_file = MagicMock()
    mock_file.filename = "leave_policy_2026.pdf"
    mock_file.read = AsyncMock(return_value=MINIMAL_VALID_PDF)

    doc_resp = await service.upload_document(
        file=mock_file,
        school_id="SCH-001",
        group_id="GRP-100",
        branch_id="BR-MAIN",
        document_type="leave_policy",
        version=1,
        staff_category="teaching",
        employment_type="full-time",
        department="science",
    )

    assert doc_resp.document_id is not None
    assert doc_resp.school_id == "SCH-001"
    assert doc_resp.group_id == "GRP-100"
    assert doc_resp.status == DocumentStatus.UPLOADED
    assert doc_resp.processing_status == ProcessingStatus.PENDING
    assert doc_resp.employee_applicability.staff_category == "teaching"
    assert Path(doc_resp.stored_file_path).exists()
    mock_repo.insert_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_missing_scope_rejection(mock_repo, temp_storage):
    """Verify upload without any scoping dimension is rejected."""
    service = IngestionService(repository=mock_repo, storage=temp_storage)
    mock_file = MagicMock()
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=MINIMAL_VALID_PDF)

    with pytest.raises(FileValidationException) as exc:
        await service.upload_document(
            file=mock_file,
            is_global=False,
            school_id=None,
            group_id=None,
            state=None,
        )
    assert exc.value.error_code == "INVALID_SCHOOL_SCOPE"


@pytest.mark.asyncio
async def test_service_duplicate_detection(mock_repo, temp_storage):
    """Verify duplicate upload is rejected with DuplicateDocumentException."""
    mock_repo.find_duplicate.return_value = {
        "document_id": "existing-doc-uuid",
        "document_name": "leave_policy.pdf",
    }
    service = IngestionService(repository=mock_repo, storage=temp_storage)

    mock_file = MagicMock()
    mock_file.filename = "leave_policy.pdf"
    mock_file.read = AsyncMock(return_value=MINIMAL_VALID_PDF)

    with pytest.raises(DuplicateDocumentException):
        await service.upload_document(
            file=mock_file,
            school_id="SCH-001",
        )


@pytest.mark.asyncio
async def test_service_get_and_delete_document(mock_repo, temp_storage):
    """Verify document retrieval and deletion through service."""
    fake_record = {
        "document_id": "test-doc-id",
        "document_name": "rules.pdf",
        "stored_file_path": temp_storage.save_file("test-doc-id", "rules.pdf", MINIMAL_VALID_PDF),
        "file_size_bytes": len(MINIMAL_VALID_PDF),
        "file_hash": "hash123",
        "school_id": "SCH-001",
        "group_id": None,
        "branch_id": None,
        "is_global": False,
        "state": None,
        "document_type": "policy",
        "version": 1,
        "status": DocumentStatus.UPLOADED,
        "effective_from": None,
        "effective_to": None,
        "employee_applicability": {"staff_category": None, "employment_type": None, "department": None},
        "uploaded_at": "2026-11-18T10:00:00Z",
        "processing_status": ProcessingStatus.PENDING,
        "error_message": None,
        "metadata": {},
    }
    mock_repo.get_by_id.return_value = fake_record

    service = IngestionService(repository=mock_repo, storage=temp_storage)

    # 1. Retrieval
    retrieved = await service.get_document("test-doc-id")
    assert retrieved.document_id == "test-doc-id"

    # 2. Deletion
    deleted = await service.delete_document("test-doc-id")
    assert deleted is True
    assert not Path(fake_record["stored_file_path"]).exists()


# ==============================================================================
# 3. FastAPI Route Integration Tests
# ==============================================================================

@pytest.fixture
def override_ingestion_service(temp_storage, mock_repo):
    """Dependency override fixture for FastAPI TestClient."""
    test_service = IngestionService(repository=mock_repo, storage=temp_storage)
    app.dependency_overrides[get_ingestion_service] = lambda: test_service
    yield test_service
    app.dependency_overrides.pop(get_ingestion_service, None)


def test_api_upload_successful(client: TestClient, override_ingestion_service):
    """POST /api/v1/documents/upload: Successful PDF upload."""
    files = {"file": ("code_of_conduct.pdf", MINIMAL_VALID_PDF, "application/pdf")}
    data = {
        "school_id": "SCH-DPS-01",
        "group_id": "GRP-DPS",
        "branch_id": "BR-NORTH",
        "document_type": "code_of_conduct",
        "version": 1,
        "staff_category": "teaching",
        "employment_type": "full-time",
        "department": "all",
    }

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 201
    res_json = response.json()
    assert res_json["success"] is True
    doc_data = res_json["data"]
    assert doc_data["school_id"] == "SCH-DPS-01"
    assert doc_data["status"] == "uploaded"
    assert doc_data["processing_status"] == "pending"


def test_api_upload_with_custom_document_name(client: TestClient, override_ingestion_service):
    """POST /api/v1/documents/upload: Custom document_name without .pdf suffix does not cause validation failure."""
    files = {"file": ("hr_policy.pdf", MINIMAL_VALID_PDF, "application/pdf")}
    data = {
        "school_id": "SCH-DPS-01",
        "document_name": "Annual Leave Policy 2026",  # Custom name without .pdf extension
    }

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 201
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["data"]["document_name"] == "Annual Leave Policy 2026"
    assert res_json["data"]["stored_file_path"].endswith(".pdf")


def test_api_upload_invalid_file_type(client: TestClient, override_ingestion_service):
    """POST /api/v1/documents/upload: Rejects non-PDF extension."""
    files = {"file": ("notes.txt", b"plain text content", "text/plain")}
    data = {"school_id": "SCH-DPS-01"}

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "INVALID_FILE_TYPE"


def test_api_upload_corrupted_pdf_signature(client: TestClient, override_ingestion_service):
    """POST /api/v1/documents/upload: Rejects fake PDF with non-%PDF- header."""
    files = {"file": ("corrupted.pdf", b"NOT_A_VALID_PDF_HEADER_DATA_STREAM", "application/pdf")}
    data = {"school_id": "SCH-DPS-01"}

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "INVALID_PDF_SIGNATURE"


def test_api_upload_missing_scope(client: TestClient, override_ingestion_service):
    """POST /api/v1/documents/upload: Rejects upload with no scoping dimensions."""
    files = {"file": ("policy.pdf", MINIMAL_VALID_PDF, "application/pdf")}
    data = {}  # No school_id, group_id, state, or is_global

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "INVALID_SCHOOL_SCOPE"


def test_api_get_document_success(client: TestClient, override_ingestion_service, mock_repo):
    """GET /api/v1/documents/{document_id}: Retrieve document details."""
    mock_repo.get_by_id.return_value = {
        "document_id": "doc-abc-123",
        "document_name": "leave_policy.pdf",
        "stored_file_path": "/data/uploads/doc-abc-123_leave_policy.pdf",
        "file_size_bytes": 1024,
        "file_hash": "hashabc",
        "school_id": "SCH-DPS-01",
        "group_id": None,
        "branch_id": None,
        "is_global": False,
        "state": None,
        "document_type": "leave_policy",
        "version": 1,
        "status": DocumentStatus.UPLOADED,
        "effective_from": None,
        "effective_to": None,
        "employee_applicability": {"staff_category": "teaching", "employment_type": None, "department": None},
        "uploaded_at": "2026-11-18T10:00:00Z",
        "processing_status": ProcessingStatus.PENDING,
        "error_message": None,
        "metadata": {},
    }

    response = client.get("/api/v1/documents/doc-abc-123")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["data"]["document_id"] == "doc-abc-123"


def test_api_get_document_not_found(client: TestClient, override_ingestion_service, mock_repo):
    """GET /api/v1/documents/{document_id}: 404 for missing document."""
    mock_repo.get_by_id.return_value = None

    response = client.get("/api/v1/documents/non-existent-id")
    assert response.status_code == 404
    res_json = response.json()
    assert res_json["success"] is False
    assert res_json["error_code"] == "DOCUMENT_NOT_FOUND"


def test_api_list_documents(client: TestClient, override_ingestion_service, mock_repo):
    """GET /api/v1/documents: List documents with pagination and filtering."""
    item = {
        "document_id": "doc-1",
        "document_name": "doc1.pdf",
        "stored_file_path": "/data/uploads/doc1.pdf",
        "file_size_bytes": 500,
        "file_hash": "hash1",
        "school_id": "SCH-1",
        "group_id": None,
        "branch_id": None,
        "is_global": False,
        "state": None,
        "document_type": "policy",
        "version": 1,
        "status": DocumentStatus.UPLOADED,
        "effective_from": None,
        "effective_to": None,
        "employee_applicability": {"staff_category": None, "employment_type": None, "department": None},
        "uploaded_at": "2026-11-18T10:00:00Z",
        "processing_status": ProcessingStatus.PENDING,
        "error_message": None,
        "metadata": {},
    }
    mock_repo.list_documents.return_value = ([item], 1)

    response = client.get("/api/v1/documents?school_id=SCH-1&page=1&page_size=10")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["data"]["total"] == 1
    assert len(res_json["data"]["items"]) == 1
    assert res_json["data"]["items"][0]["document_id"] == "doc-1"


def test_api_delete_document_success(client: TestClient, override_ingestion_service, mock_repo, temp_storage):
    """DELETE /api/v1/documents/{document_id}: Safely delete document and file."""
    stored_path = temp_storage.save_file("del-id-123", "policy.pdf", MINIMAL_VALID_PDF)
    assert Path(stored_path).exists()

    mock_repo.get_by_id.return_value = {
        "document_id": "del-id-123",
        "stored_file_path": stored_path,
    }
    mock_repo.delete_by_id.return_value = True

    response = client.delete("/api/v1/documents/del-id-123")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["data"]["deleted"] is True
    # Physical file should be deleted
    assert not Path(stored_path).exists()


# ==============================================================================
# 4. Targeted Review Tests: Hash Computation & Deletion Consistency
# ==============================================================================

@pytest.mark.asyncio
async def test_file_hash_uses_actual_bytes_and_powers_duplicate_detection(mock_repo, temp_storage):
    """Point 1: Verify file_hash is calculated on actual uploaded PDF bytes, not empty bytes,

    and verify that duplicate detection receives and utilizes this exact hash.
    """
    import hashlib

    custom_pdf_bytes = MINIMAL_VALID_PDF + b"\n% Custom policy revision comment"
    expected_hash = hashlib.sha256(custom_pdf_bytes).hexdigest()
    empty_bytes_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # 1. Ensure expected hash is calculated from the real bytes and is not empty hash
    assert expected_hash != empty_bytes_hash

    service = IngestionService(repository=mock_repo, storage=temp_storage)

    mock_file = MagicMock()
    mock_file.filename = "anti_harassment_policy.pdf"
    mock_file.read = AsyncMock(return_value=custom_pdf_bytes)
    mock_file.seek = AsyncMock()

    doc_resp = await service.upload_document(
        file=mock_file,
        school_id="SCH-500",
        group_id="GRP-100",
        version=2,
    )

    # 2. Verify returned and stored hash matches the actual SHA-256 of uploaded bytes
    assert doc_resp.file_hash == expected_hash
    assert doc_resp.file_hash != empty_bytes_hash
    assert doc_resp.file_size_bytes == len(custom_pdf_bytes)

    # 3. Verify repository.find_duplicate was invoked with this exact non-empty hash
    mock_repo.find_duplicate.assert_awaited_once_with(
        file_hash=expected_hash,
        school_id="SCH-500",
        group_id="GRP-100",
        branch_id=None,
        is_global=False,
        state=None,
        version=2,
    )


@pytest.mark.asyncio
async def test_safe_deletion_rollback_on_database_failure(mock_repo, temp_storage):
    """Point 2: Verify consistency when MongoDB deletion fails:

    The staged physical file must be restored back to its original location,
    preventing inconsistent records pointing to missing files.
    """
    stored_path = temp_storage.save_file("rollback-doc-1", "policy_active.pdf", MINIMAL_VALID_PDF)
    assert Path(stored_path).exists()

    mock_repo.get_by_id.return_value = {
        "document_id": "rollback-doc-1",
        "stored_file_path": stored_path,
    }
    # Simulate database failure during delete_by_id
    mock_repo.delete_by_id = AsyncMock(side_effect=Exception("Database network timeout"))

    service = IngestionService(repository=mock_repo, storage=temp_storage)

    with pytest.raises(Exception) as exc_info:
        await service.delete_document("rollback-doc-1")
    assert "Database network timeout" in str(exc_info.value)

    # Consistency check: File MUST be restored at stored_path, NOT left missing or orphaned
    assert Path(stored_path).exists()


@pytest.mark.asyncio
async def test_safe_deletion_rollback_on_concurrent_database_deletion(mock_repo, temp_storage):
    """Point 2: Verify consistency when MongoDB record is not found during delete_by_id:

    The staged physical file must be restored, and NotFoundException raised.
    """
    stored_path = temp_storage.save_file("rollback-doc-2", "policy_active.pdf", MINIMAL_VALID_PDF)
    assert Path(stored_path).exists()

    mock_repo.get_by_id.return_value = {
        "document_id": "rollback-doc-2",
        "stored_file_path": stored_path,
    }
    # delete_by_id returns False (e.g. concurrently deleted)
    mock_repo.delete_by_id = AsyncMock(return_value=False)

    service = IngestionService(repository=mock_repo, storage=temp_storage)

    with pytest.raises(NotFoundException):
        await service.delete_document("rollback-doc-2")

    # Consistency check: File MUST be restored at stored_path
    assert Path(stored_path).exists()


@pytest.mark.asyncio
async def test_safe_deletion_when_physical_file_already_missing(mock_repo, temp_storage):
    """Point 2: Verify deletion handles dangling database records where physical file was missing."""
    missing_path = str((temp_storage.upload_dir / "already_deleted.pdf").resolve())
    assert not Path(missing_path).exists()

    mock_repo.get_by_id.return_value = {
        "document_id": "dangling-doc-id",
        "stored_file_path": missing_path,
    }
    mock_repo.delete_by_id = AsyncMock(return_value=True)

    service = IngestionService(repository=mock_repo, storage=temp_storage)

    deleted = await service.delete_document("dangling-doc-id")
    assert deleted is True
    mock_repo.delete_by_id.assert_awaited_once_with("dangling-doc-id")


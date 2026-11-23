"""Ingestion service implementing document upload, retrieval, listing, and deletion."""

import inspect
import math
from typing import Any, Dict, List, Optional
import uuid
from fastapi import UploadFile

from app.core.exceptions import (
    DuplicateDocumentException,
    FileValidationException,
    NotFoundException,
)
from app.modules.ingestion.repository import DocumentRepository
from app.modules.ingestion.schemas import (
    DocumentCreate,
    DocumentFilterParams,
    DocumentResponse,
    DocumentStatus,
    EmployeeApplicability,
    ProcessingStatus,
)
from app.modules.ingestion.storage import LocalStorageService
from app.schemas.common import PaginatedData


class IngestionService:
    """Orchestrates document ingestion, file validation, storage, and persistence."""

    def __init__(self, repository: DocumentRepository, storage: LocalStorageService) -> None:
        self.repository = repository
        self.storage = storage

    async def upload_document(
        self,
        file: UploadFile,
        school_id: Optional[str] = None,
        group_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        is_global: bool = False,
        state: Optional[str] = None,
        document_type: str = "policy",
        version: int = 1,
        effective_from: Optional[Any] = None,
        effective_to: Optional[Any] = None,
        staff_category: Optional[str] = None,
        employment_type: Optional[str] = None,
        department: Optional[str] = None,
        document_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentResponse:
        """Process and register an uploaded policy PDF document.

        Steps:
        1. Validate school scope constraints.
        2. Read file bytes and validate PDF content, size, and signature.
        3. Check for duplicates (matching hash, scope, and version).
        4. Persist PDF to local storage with safe path traversal checks.
        5. Register metadata in MongoDB with initial status='uploaded' and processing_status='pending'.
        """
        # 1. School scope validation
        if not is_global and not school_id and not group_id and not state:
            raise FileValidationException(
                message="Document must specify at least one scoping dimension: "
                "school_id, group_id, state, or is_global=True.",
                error_code="INVALID_SCHOOL_SCOPE",
            )

        # 2. Read and validate file content
        if hasattr(file, "seek"):
            seek_res = file.seek(0)
            if inspect.isawaitable(seek_res):
                await seek_res
        content = await file.read()
        uploaded_filename = file.filename or "document.pdf"
        file_size, file_hash = self.storage.validate_pdf_content(
            content=content,
            original_filename=uploaded_filename,
        )

        # Document display title: use custom document_name if provided, else uploaded filename
        doc_title = document_name.strip() if document_name and document_name.strip() else uploaded_filename

        # 3. Duplicate detection
        duplicate = await self.repository.find_duplicate(
            file_hash=file_hash,
            school_id=school_id,
            group_id=group_id,
            branch_id=branch_id,
            is_global=is_global,
            state=state,
            version=version,
        )
        if duplicate:
            raise DuplicateDocumentException(
                message=f"A document with identical content and scoping already exists (document_id: {duplicate['document_id']})"
            )

        # 4. Generate unique document ID and save physical file
        document_id = str(uuid.uuid4())
        stored_path = self.storage.save_file(
            document_id=document_id,
            original_filename=uploaded_filename,
            content=content,
        )

        # 5. Build record and persist in MongoDB
        doc_create = DocumentCreate(
            document_id=document_id,
            document_name=doc_title,
            stored_file_path=stored_path,
            file_size_bytes=file_size,
            file_hash=file_hash,
            school_id=school_id,
            group_id=group_id,
            branch_id=branch_id,
            is_global=is_global,
            state=state,
            document_type=document_type,
            version=version,
            status=DocumentStatus.UPLOADED,
            effective_from=effective_from,
            effective_to=effective_to,
            employee_applicability=EmployeeApplicability(
                staff_category=staff_category,
                employment_type=employment_type,
                department=department,
            ),
            processing_status=ProcessingStatus.PENDING,
            error_message=None,
            metadata=metadata or {},
        )

        try:
            record = await self.repository.insert_document(doc_create)
            return DocumentResponse.model_validate(record)
        except Exception:
            # Rollback: Clean up physical file if database registration fails
            self.storage.delete_file(stored_path)
            raise

    async def get_document(self, document_id: str) -> DocumentResponse:
        """Fetch document details by document_id."""
        record = await self.repository.get_by_id(document_id)
        if not record:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )
        return DocumentResponse.model_validate(record)

    async def list_documents(self, filters: DocumentFilterParams) -> PaginatedData[DocumentResponse]:
        """List documents matching filter criteria with pagination."""
        records, total = await self.repository.list_documents(filters)
        items = [DocumentResponse.model_validate(rec) for rec in records]
        total_pages = math.ceil(total / filters.page_size) if total > 0 else 0

        return PaginatedData[DocumentResponse](
            items=items,
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            total_pages=total_pages,
        )

    async def delete_document(self, document_id: str) -> bool:
        """Safely delete document record from MongoDB and remove physical file from disk.

        Ensures consistency between storage and database:
        1. Verifies document exists in database.
        2. Stages physical file for deletion (isolated in temporary trash path).
        3. Deletes document record from MongoDB.
        4. If MongoDB deletion fails or raises an error, rolls back and restores physical file.
        5. If MongoDB deletion succeeds, permanently cleans up the staged physical file.
        """
        record = await self.repository.get_by_id(document_id)
        if not record:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        stored_path = record.get("stored_file_path")
        staged_trash_path: Optional[str] = None

        # 1. Stage physical file if it exists on disk
        if stored_path:
            staged_trash_path = self.storage.stage_file_for_deletion(stored_path)

        # 2. Delete database record with rollback protection
        try:
            deleted = await self.repository.delete_by_id(document_id)
            if not deleted:
                # Concurrent deletion or record unexpectedly missing
                if staged_trash_path and stored_path:
                    self.storage.restore_staged_file(staged_trash_path, stored_path)
                raise NotFoundException(
                    message=f"Document with ID '{document_id}' not found or already deleted",
                    error_code="DOCUMENT_NOT_FOUND",
                )
        except Exception:
            # Rollback: restore physical file so we do not leave orphaned/inconsistent states
            if staged_trash_path and stored_path:
                self.storage.restore_staged_file(staged_trash_path, stored_path)
            raise

        # 3. Finalize physical deletion now that database record is safely removed
        if staged_trash_path:
            self.storage.finalize_staged_deletion(staged_trash_path)

        return True

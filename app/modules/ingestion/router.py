"""FastAPI routes for Document Ingestion and management."""

from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from app.api.dependencies import AsyncDatabase, Settings, get_mongo_db, get_settings
from app.modules.ingestion.repository import DocumentRepository
from app.modules.ingestion.schemas import (
    DocumentFilterParams,
    DocumentResponse,
    DocumentStatus,
    ProcessingStatus,
)
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import LocalStorageService
from app.schemas.common import ApiResponse, PaginatedData

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_ingestion_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    """Dependency provider for IngestionService."""
    repository = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
    storage = LocalStorageService(
        upload_dir=settings.UPLOAD_DIR,
        max_file_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
    )
    return IngestionService(repository=repository, storage=storage)


@router.post(
    "/upload",
    response_model=ApiResponse[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest a Policy PDF",
    description="Uploads a PDF file, validates format/integrity, stores file, and registers metadata in MongoDB.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload"),
    school_id: Optional[str] = Form(default=None, description="School identifier"),
    group_id: Optional[str] = Form(default=None, description="Educational group/trust identifier"),
    branch_id: Optional[str] = Form(default=None, description="Campus/branch identifier"),
    is_global: bool = Form(default=False, description="Global group-wide policy flag"),
    state: Optional[str] = Form(default=None, description="Geographical jurisdiction"),
    document_type: str = Form(default="policy", description="Document type/classification"),
    version: int = Form(default=1, ge=1, description="Policy version number"),
    effective_from: Optional[datetime] = Form(default=None, description="Effective start date"),
    effective_to: Optional[datetime] = Form(default=None, description="Effective end date"),
    staff_category: Optional[str] = Form(default=None, description="Target staff category"),
    employment_type: Optional[str] = Form(default=None, description="Employment type applicability"),
    department: Optional[str] = Form(default=None, description="Department applicability"),
    document_name: Optional[str] = Form(default=None, description="Custom document title"),
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentResponse]:
    """Handle PDF upload, validation, physical persistence, and database registration."""
    document = await service.upload_document(
        file=file,
        school_id=school_id,
        group_id=group_id,
        branch_id=branch_id,
        is_global=is_global,
        state=state,
        document_type=document_type,
        version=version,
        effective_from=effective_from,
        effective_to=effective_to,
        staff_category=staff_category,
        employment_type=employment_type,
        department=department,
        document_name=document_name,
    )
    return ApiResponse[DocumentResponse](
        message="Document uploaded and registered successfully",
        data=document,
    )


@router.get(
    "/{document_id}",
    response_model=ApiResponse[DocumentResponse],
    summary="Get Document Details",
    description="Retrieves metadata and status of an ingested document by document_id.",
)
async def get_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentResponse]:
    """Retrieve document metadata by document_id."""
    document = await service.get_document(document_id)
    return ApiResponse[DocumentResponse](
        message="Document retrieved successfully",
        data=document,
    )


@router.get(
    "",
    response_model=ApiResponse[PaginatedData[DocumentResponse]],
    summary="List Documents with Filtering and Pagination",
    description="Query policy documents by school scope, type, and lifecycle status.",
)
async def list_documents(
    school_id: Optional[str] = Query(default=None, description="Filter by school ID"),
    group_id: Optional[str] = Query(default=None, description="Filter by group ID"),
    branch_id: Optional[str] = Query(default=None, description="Filter by branch ID"),
    is_global: Optional[bool] = Query(default=None, description="Filter by global scope"),
    state: Optional[str] = Query(default=None, description="Filter by state jurisdiction"),
    document_type: Optional[str] = Query(default=None, description="Filter by document type"),
    status: Optional[DocumentStatus] = Query(default=None, description="Filter by status"),
    processing_status: Optional[ProcessingStatus] = Query(default=None, description="Filter by processing status"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[PaginatedData[DocumentResponse]]:
    """List documents matching filters."""
    filters = DocumentFilterParams(
        school_id=school_id,
        group_id=group_id,
        branch_id=branch_id,
        is_global=is_global,
        state=state,
        document_type=document_type,
        status=status,
        processing_status=processing_status,
        page=page,
        page_size=page_size,
    )
    result = await service.list_documents(filters)
    return ApiResponse[PaginatedData[DocumentResponse]](
        message="Documents retrieved successfully",
        data=result,
    )


@router.delete(
    "/{document_id}",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Delete Document",
    description="Deletes document record from MongoDB and removes the physical PDF file safely.",
)
async def delete_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[Dict[str, Any]]:
    """Delete document record and associated physical file."""
    await service.delete_document(document_id)
    return ApiResponse[Dict[str, Any]](
        message=f"Document '{document_id}' and stored file successfully deleted",
        data={"document_id": document_id, "deleted": True},
    )

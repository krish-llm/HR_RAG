"""FastAPI routes for Document Processing (Cleaning & Normalization) operations."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import AsyncDatabase, Settings, get_mongo_db, get_settings
from app.modules.extraction.repository import ExtractionRepository
from app.modules.ingestion.repository import DocumentRepository
from app.modules.processing.cleaner import DocumentCleaner
from app.modules.processing.repository import ProcessedRepository
from app.modules.processing.schemas import (
    DocumentProcessedResult,
    ProcessingSummaryResponse,
)
from app.modules.processing.service import ProcessingService
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/documents", tags=["Document Processing"])


def get_processing_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    settings: Settings = Depends(get_settings),
) -> ProcessingService:
    """Dependency provider for ProcessingService."""
    document_repo = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
    extraction_repo = ExtractionRepository(db, collection_name=settings.EXTRACTIONS_COLLECTION_NAME)
    processed_repo = ProcessedRepository(db, collection_name=settings.PROCESSED_COLLECTION_NAME)
    cleaner = DocumentCleaner()
    return ProcessingService(
        document_repo=document_repo,
        extraction_repo=extraction_repo,
        processed_repo=processed_repo,
        cleaner=cleaner,
    )


@router.post(
    "/{document_id}/process",
    response_model=ApiResponse[ProcessingSummaryResponse],
    status_code=status.HTTP_200_OK,
    summary="Process and Clean Document Content",
    description="Synchronously cleans and normalizes extracted Markdown content, removing repeated headers/footers and noise while preserving tables and structure.",
)
async def process_document(
    document_id: str,
    service: ProcessingService = Depends(get_processing_service),
) -> ApiResponse[ProcessingSummaryResponse]:
    """Trigger synchronous processing for an extracted PDF document."""
    result = await service.process_document(document_id)
    summary = ProcessingSummaryResponse(
        document_id=result.document_id,
        source_extraction_id=result.source_extraction_id,
        total_pages=result.total_pages,
        processed_pages_count=len(result.processed_pages),
        total_characters=result.total_characters,
        processing_status=result.processing_status,
        processed_at=result.processed_at,
    )
    return ApiResponse[ProcessingSummaryResponse](
        message="Document content successfully cleaned and normalized",
        data=summary,
    )


@router.get(
    "/{document_id}/processed",
    response_model=ApiResponse[DocumentProcessedResult],
    summary="Get Processed Document Content",
    description="Retrieves the page-level cleaned and normalized document text along with detected structure metadata.",
)
async def get_processed(
    document_id: str,
    service: ProcessingService = Depends(get_processing_service),
) -> ApiResponse[DocumentProcessedResult]:
    """Retrieve full page-by-page cleaned and normalized document content."""
    result = await service.get_processed(document_id)
    return ApiResponse[DocumentProcessedResult](
        message="Processed document content retrieved successfully",
        data=result,
    )

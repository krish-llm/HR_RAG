"""FastAPI routes for PDF Extraction operations."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import AsyncDatabase, Settings, get_mongo_db, get_settings
from app.modules.extraction.extractor import PDFExtractor
from app.modules.extraction.repository import ExtractionRepository
from app.modules.extraction.schemas import (
    DocumentExtractionResult,
    ExtractionSummaryResponse,
)
from app.modules.extraction.service import ExtractionService
from app.modules.ingestion.repository import DocumentRepository
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/documents", tags=["Document Extraction"])


def get_extraction_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    settings: Settings = Depends(get_settings),
) -> ExtractionService:
    """Dependency provider for ExtractionService."""
    extraction_repo = ExtractionRepository(db, collection_name=settings.EXTRACTIONS_COLLECTION_NAME)
    document_repo = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
    extractor = PDFExtractor()
    return ExtractionService(
        extraction_repo=extraction_repo,
        document_repo=document_repo,
        extractor=extractor,
    )


@router.post(
    "/{document_id}/extract",
    response_model=ApiResponse[ExtractionSummaryResponse],
    status_code=status.HTTP_200_OK,
    summary="Extract Document Content",
    description="Synchronously extracts Markdown content and page boundaries from an uploaded PDF using PyMuPDF4LLM.",
)
async def extract_document(
    document_id: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> ApiResponse[ExtractionSummaryResponse]:
    """Trigger synchronous extraction for a registered PDF document."""
    result = await service.extract_document(document_id)
    summary = ExtractionSummaryResponse(
        document_id=result.document_id,
        document_name=result.document_name,
        total_pages=result.total_pages,
        extracted_pages_count=len(result.extracted_pages),
        total_characters=result.total_characters,
        extraction_status=result.extraction_status,
        extracted_at=result.extracted_at,
    )
    return ApiResponse[ExtractionSummaryResponse](
        message="PDF content successfully extracted",
        data=summary,
    )


@router.get(
    "/{document_id}/extraction",
    response_model=ApiResponse[DocumentExtractionResult],
    summary="Get Extracted Document Content",
    description="Retrieves the page-level extracted Markdown content and statistics for a document.",
)
async def get_extraction(
    document_id: str,
    service: ExtractionService = Depends(get_extraction_service),
) -> ApiResponse[DocumentExtractionResult]:
    """Retrieve full page-by-page extracted Markdown content."""
    result = await service.get_extraction(document_id)
    return ApiResponse[DocumentExtractionResult](
        message="Extraction content retrieved successfully",
        data=result,
    )

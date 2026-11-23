"""FastAPI routes for Document Chunking operations."""

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import AsyncDatabase, Settings, get_mongo_db, get_settings
from app.modules.chunking.chunker import DocumentChunker
from app.modules.chunking.repository import ChunkRepository
from app.modules.chunking.schemas import (
    ChunkingSummaryResponse,
    PaginatedChunksResponse,
)
from app.modules.chunking.service import ChunkingService
from app.modules.ingestion.repository import DocumentRepository
from app.modules.processing.repository import ProcessedRepository
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/documents", tags=["Document Chunking"])


def get_chunking_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    settings: Settings = Depends(get_settings),
) -> ChunkingService:
    """Dependency provider for ChunkingService."""
    document_repo = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
    processed_repo = ProcessedRepository(db, collection_name=settings.PROCESSED_COLLECTION_NAME)
    chunk_repo = ChunkRepository(db, collection_name=settings.CHUNKS_COLLECTION_NAME)
    chunker = DocumentChunker(
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        encoding_name=settings.TOKENIZER_ENCODING,
    )
    return ChunkingService(
        document_repo=document_repo,
        processed_repo=processed_repo,
        chunk_repo=chunk_repo,
        chunker=chunker,
    )


@router.post(
    "/{document_id}/chunk",
    response_model=ApiResponse[ChunkingSummaryResponse],
    status_code=status.HTTP_200_OK,
    summary="Chunk Processed Document Content",
    description="Synchronously splits cleaned document content into heading-aware semantic chunks with tiktoken token counting.",
)
async def chunk_document(
    document_id: str,
    service: ChunkingService = Depends(get_chunking_service),
) -> ApiResponse[ChunkingSummaryResponse]:
    """Trigger synchronous chunking for a processed policy document."""
    summary = await service.chunk_document(document_id)
    return ApiResponse[ChunkingSummaryResponse](
        message="Document content successfully chunked",
        data=summary,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ApiResponse[PaginatedChunksResponse],
    summary="Get Paginated Document Chunks",
    description="Retrieves paginated chunks with rich metadata, source page numbers, and breadcrumbs for a document.",
)
async def get_chunks(
    document_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    service: ChunkingService = Depends(get_chunking_service),
) -> ApiResponse[PaginatedChunksResponse]:
    """Retrieve paginated document chunks for an ingested document."""
    result = await service.get_chunks_paginated(
        document_id=document_id,
        page=page,
        page_size=page_size,
    )
    return ApiResponse[PaginatedChunksResponse](
        message="Document chunks retrieved successfully",
        data=result,
    )

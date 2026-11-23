"""Chunking service orchestrating document chunking, metadata attachment, and persistence."""

import logging
import math
from typing import Any, Dict, List, Optional

from app.core.exceptions import (
    AppException,
    ChunkingException,
    NotFoundException,
    ProcessingException,
)
from app.core.logging_utils import TimedBlock
from app.modules.chunking.chunker import DocumentChunker
from app.modules.chunking.repository import ChunkRepository
from app.modules.chunking.schemas import (
    ChunkingSummaryResponse,
    DocumentChunk,
    PaginatedChunksResponse,
)
from app.modules.ingestion.repository import DocumentRepository
from app.modules.processing.repository import ProcessedRepository
from app.modules.processing.schemas import (
    DocumentProcessedResult,
    ProcessingStageStatus,
)

logger = logging.getLogger(__name__)


class ChunkingService:
    """Orchestrates document chunking, metadata scoping, and persistence in MongoDB."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        processed_repo: ProcessedRepository,
        chunk_repo: ChunkRepository,
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.document_repo = document_repo
        self.processed_repo = processed_repo
        self.chunk_repo = chunk_repo
        self.chunker = chunker or DocumentChunker()

    async def chunk_document(self, document_id: str) -> ChunkingSummaryResponse:
        """Trigger synchronous chunking for a processed document.

        Steps:
        1. Verify registered document exists in MongoDB.
        2. Fetch cleaned and normalized content from document_processed_content.
        3. Validate processing was successful.
        4. Mark document processing_status as 'processing'.
        5. Execute heading-aware recursive chunking with tiktoken token counts.
        6. On failure: Update document processing_status to 'failed' and re-raise.
        7. On success: Persist chunks via ChunkRepository (purging any stale chunks)
           and update document processing_status to 'completed'.
        """
        # 1. Verify registered document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        # 2. Fetch processed content
        processed_record = await self.processed_repo.get_by_document_id(document_id)
        if not processed_record:
            raise ProcessingException(
                message=f"Document with ID '{document_id}' has not been processed yet. Run processing before chunking.",
                error_code="PROCESSED_CONTENT_NOT_FOUND",
                status_code=400,
            )

        try:
            processed_doc = DocumentProcessedResult.model_validate(processed_record)
        except Exception as exc:
            raise ProcessingException(
                message=f"Invalid processed document data format for document '{document_id}': {exc}",
                error_code="INVALID_PROCESSED_DATA",
                status_code=500,
            ) from exc

        # 3. Validate processing stage status
        if processed_doc.processing_status == ProcessingStageStatus.FAILED:
            raise ProcessingException(
                message=f"Cannot chunk document '{document_id}' because document processing failed: {processed_doc.error_message}",
                error_code="PROCESSING_FAILED_STATE",
                status_code=400,
            )

        # 4. Mark document processing_status as 'processing'
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="processing",
            error_message=None,
        )

        # 5. Execute chunking
        try:
            with TimedBlock(logger, "chunking") as timer:
                chunks = self.chunker.chunk_document(
                    processed_doc=processed_doc,
                    doc_metadata=doc,
                )
                timer.set_extra(chunks=len(chunks), tokens=sum(c.token_count for c in chunks))
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Chunking failed for document_id=%s: %s", document_id, error_msg)
            await self.document_repo.update_processing_status(
                document_id=document_id,
                processing_status="failed",
                error_message=error_msg,
            )
            if isinstance(exc, AppException):
                raise
            raise ChunkingException(
                message=f"Document chunking failed: {error_msg}",
                error_code="CHUNKING_FAILED",
                status_code=500,
            ) from exc

        # 6. Persist chunks in MongoDB (purges previous chunks on re-chunking)
        await self.chunk_repo.save_chunks(document_id=document_id, chunks=chunks)

        # 7. Update document processing_status to 'completed'
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="completed",
            error_message=None,
        )

        # Compute summary statistics
        total_chunks = len(chunks)
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = round(total_tokens / total_chunks, 2) if total_chunks > 0 else 0.0
        max_tokens = max((c.token_count for c in chunks), default=0)

        logger.info(
            "Completed chunking for document_id=%s (chunks=%d, avg_tokens=%.1f, max_tokens=%d)",
            document_id,
            total_chunks,
            avg_tokens,
            max_tokens,
        )

        return ChunkingSummaryResponse(
            document_id=document_id,
            total_chunks=total_chunks,
            average_tokens=avg_tokens,
            maximum_tokens=max_tokens,
            processing_status="completed",
        )

    async def get_chunks_paginated(
        self,
        document_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedChunksResponse:
        """Retrieve paginated document chunks for a document."""
        # Verify parent document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        records, total = await self.chunk_repo.get_chunks_paginated(
            document_id=document_id,
            page=page,
            page_size=page_size,
        )

        items = [DocumentChunk.model_validate(r) for r in records]
        total_pages = math.ceil(total / page_size) if total > 0 else 1

        return PaginatedChunksResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

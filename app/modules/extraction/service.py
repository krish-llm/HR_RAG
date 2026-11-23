"""Extraction service orchestrating document retrieval, PyMuPDF4LLM extraction, and persistence."""

import logging
from typing import Optional

from app.core.exceptions import (
    AppException,
    ExtractionException,
    NotFoundException,
)
from app.core.logging_utils import TimedBlock
from app.modules.extraction.extractor import PDFExtractor
from app.modules.extraction.repository import ExtractionRepository
from app.modules.extraction.schemas import DocumentExtractionResult
from app.modules.ingestion.repository import DocumentRepository

logger = logging.getLogger(__name__)


class ExtractionService:
    """Orchestrates PDF text extraction, document status lifecycle, and extraction storage."""

    def __init__(
        self,
        extraction_repo: ExtractionRepository,
        document_repo: DocumentRepository,
        extractor: Optional[PDFExtractor] = None,
    ) -> None:
        self.extraction_repo = extraction_repo
        self.document_repo = document_repo
        self.extractor = extractor or PDFExtractor()

    async def extract_document(self, document_id: str) -> DocumentExtractionResult:
        """Trigger synchronous page-by-page extraction for an ingested document.

        Steps:
        1. Retrieve document metadata from MongoDB.
        2. Set processing_status to 'processing'.
        3. Extract Markdown page content and statistics via PyMuPDF4LLM.
        4. On failure: Update processing_status to 'failed' with error explanation and re-raise.
        5. On success: Persist extraction result, update document processing_status to 'completed',
           and set status to 'ready'.
        """
        # 1. Retrieve registered document metadata
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        stored_file_path = doc.get("stored_file_path")
        document_name = doc.get("document_name", "document.pdf")

        # 2. Mark processing_status as 'processing'
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="processing",
            error_message=None,
        )

        # 3. Perform page-level extraction
        try:
            with TimedBlock(logger, "pdf_extraction") as timer:
                result = self.extractor.extract(
                    file_path_str=stored_file_path,
                    document_id=document_id,
                    document_name=document_name,
                )
                timer.set_extra(pages=result.total_pages)
        except Exception as exc:
            # 4. Handle failure: Update document processing_status to 'failed'
            error_msg = str(exc)
            logger.error("Extraction failed for document_id=%s: %s", document_id, error_msg)
            await self.document_repo.update_processing_status(
                document_id=document_id,
                processing_status="failed",
                error_message=error_msg,
                status="failed",
            )
            if isinstance(exc, AppException):
                raise
            raise ExtractionException(
                message=f"PDF extraction failed: {error_msg}",
                error_code="EXTRACTION_FAILED",
                status_code=500,
            ) from exc

        # 5. Persist extraction result in MongoDB (supports safe re-extraction via upsert)
        await self.extraction_repo.save_extraction(result)

        # 6. Update document metadata status in MongoDB
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="completed",
            error_message=None,
            status="ready",
        )

        return result

    async def get_extraction(self, document_id: str) -> DocumentExtractionResult:
        """Retrieve previously stored extraction result for a document."""
        # 1. Verify parent document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        # 2. Retrieve extraction document
        record = await self.extraction_repo.get_by_document_id(document_id)
        if not record:
            current_status = doc.get("processing_status", "unknown")
            raise NotFoundException(
                message=f"Extraction content for document '{document_id}' not found. Current status: '{current_status}'",
                error_code="EXTRACTION_NOT_FOUND",
            )

        return DocumentExtractionResult.model_validate(record)

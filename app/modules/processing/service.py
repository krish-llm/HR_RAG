"""Document processing service orchestrating cleaning, normalization, and persistence."""

import logging
from typing import Optional

from app.core.exceptions import (
    AppException,
    NotFoundException,
    ProcessingException,
)
from app.core.logging_utils import TimedBlock
from app.modules.extraction.repository import ExtractionRepository
from app.modules.extraction.schemas import DocumentExtractionResult, ExtractionStatus
from app.modules.ingestion.repository import DocumentRepository
from app.modules.processing.cleaner import DocumentCleaner
from app.modules.processing.repository import ProcessedRepository
from app.modules.processing.schemas import DocumentProcessedResult

logger = logging.getLogger(__name__)


class ProcessingService:
    """Orchestrates document content cleaning, normalization, and MongoDB persistence.

    Consumes existing extraction results without re-reading or re-extracting the original PDF.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        extraction_repo: ExtractionRepository,
        processed_repo: ProcessedRepository,
        cleaner: Optional[DocumentCleaner] = None,
    ) -> None:
        self.document_repo = document_repo
        self.extraction_repo = extraction_repo
        self.processed_repo = processed_repo
        self.cleaner = cleaner or DocumentCleaner()

    async def process_document(self, document_id: str) -> DocumentProcessedResult:
        """Trigger synchronous cleaning and normalization for an extracted document.

        Steps:
        1. Verify the document exists in the documents collection.
        2. Fetch the existing extraction result from MongoDB.
        3. Validate extraction exists and completed successfully.
        4. Mark document processing_status as 'processing'.
        5. Clean and normalize content page-by-page preserving document structure.
        6. On failure: Update document processing_status to 'failed' and record error.
           (Extraction record remains unchanged).
        7. On success: Persist processed content in MongoDB and update document
           processing_status to 'completed'.
        """
        # 1. Verify registered document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        # 2. Fetch existing extraction result from MongoDB
        extraction_record = await self.extraction_repo.get_by_document_id(document_id)
        if not extraction_record:
            raise ProcessingException(
                message=f"Document with ID '{document_id}' has not been extracted yet. Please extract the document before processing.",
                error_code="EXTRACTION_NOT_FOUND",
                status_code=400,
            )

        try:
            extraction = DocumentExtractionResult.model_validate(extraction_record)
        except Exception as exc:
            raise ProcessingException(
                message=f"Invalid extraction data format for document '{document_id}': {exc}",
                error_code="INVALID_EXTRACTION_DATA",
                status_code=500,
            ) from exc

        # 3. Validate extraction state
        if extraction.extraction_status == ExtractionStatus.FAILED:
            raise ProcessingException(
                message=f"Cannot process document '{document_id}' because extraction failed: {extraction.error_message}",
                error_code="EXTRACTION_FAILED_STATE",
                status_code=400,
            )

        # 4. Update document processing_status to 'processing'
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="processing",
            error_message=None,
        )

        # 5. Clean, normalize, and detect structure
        try:
            with TimedBlock(logger, "cleaning_normalization") as timer:
                result = self.cleaner.clean_document(extraction)
                timer.set_extra(pages=result.total_pages, characters=result.total_characters)
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Processing failed for document_id=%s: %s", document_id, error_msg)
            # Update document processing_status to 'failed' without altering extraction result
            await self.document_repo.update_processing_status(
                document_id=document_id,
                processing_status="failed",
                error_message=error_msg,
            )
            if isinstance(exc, AppException):
                raise
            raise ProcessingException(
                message=f"Document processing failed: {error_msg}",
                error_code="PROCESSING_FAILED",
                status_code=500,
            ) from exc

        # 6. Persist processed content in MongoDB (supports safe reprocessing via upsert)
        await self.processed_repo.save_processed(result)

        # 7. Update document processing_status to 'completed'
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="completed",
            error_message=None,
        )

        logger.info(
            "Document processing completed for document_id=%s (pages=%d, chars=%d)",
            document_id,
            result.total_pages,
            result.total_characters,
        )
        return result

    async def get_processed(self, document_id: str) -> DocumentProcessedResult:
        """Retrieve previously stored processed content for a document."""
        # Check if parent document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        record = await self.processed_repo.get_by_document_id(document_id)
        if not record:
            raise NotFoundException(
                message=f"Processed content not found for document ID '{document_id}'",
                error_code="PROCESSED_CONTENT_NOT_FOUND",
            )

        return DocumentProcessedResult.model_validate(record)

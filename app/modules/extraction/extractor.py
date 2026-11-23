"""PDF Extraction engine wrapping PyMuPDF and PyMuPDF4LLM."""

import logging
from pathlib import Path
from typing import List, Optional
import pymupdf
import pymupdf4llm

from app.core.exceptions import ExtractionException
from app.modules.extraction.schemas import (
    DocumentExtractionResult,
    ExtractedPage,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts Markdown content, page structure, and metadata from PDF files using PyMuPDF4LLM."""

    def extract(self, file_path_str: str, document_id: str, document_name: str) -> DocumentExtractionResult:
        """Extract page-level Markdown content from a PDF file.

        Args:
            file_path_str: Absolute or relative filesystem path to the PDF
            document_id: Ingested document UUID
            document_name: Document display name

        Returns:
            DocumentExtractionResult with separate page-level contents and statistics.
        """
        path = Path(file_path_str).resolve()
        if not path.exists() or not path.is_file():
            logger.error("PDF file not found for extraction: %s", path)
            raise ExtractionException(
                message=f"PDF file for document '{document_id}' not found on disk at {path}",
                error_code="FILE_NOT_FOUND",
                status_code=404,
            )

        doc: Optional[pymupdf.Document] = None
        try:
            try:
                pdf_bytes = path.read_bytes()
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            except pymupdf.FileDataError as exc:
                logger.error("Failed to open PDF %s: Corrupted or unreadable: %s", path, str(exc))
                raise ExtractionException(
                    message="Failed to read PDF file: File is corrupted or has an invalid structure",
                    error_code="CORRUPTED_PDF",
                    status_code=400,
                ) from exc
            except Exception as exc:
                logger.error("Unexpected error opening PDF %s: %s", path, str(exc))
                raise ExtractionException(
                    message=f"Unable to open PDF file: {exc}",
                    error_code="PDF_OPEN_ERROR",
                    status_code=500,
                ) from exc

            total_pages = len(doc)
            logger.info("Extracting %d pages from %s (document_id=%s)", total_pages, path.name, document_id)

            if total_pages == 0:
                return DocumentExtractionResult(
                    document_id=document_id,
                    document_name=document_name,
                    total_pages=0,
                    extracted_pages=[],
                    total_characters=0,
                    extraction_status=ExtractionStatus.COMPLETED,
                    error_message="Document contains 0 pages",
                )

            # Use PyMuPDF4LLM to convert page by page while preserving Markdown tables and headings
            try:
                page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
            except Exception as exc:
                logger.error("PyMuPDF4LLM extraction failed for %s: %s", path, str(exc))
                raise ExtractionException(
                    message=f"PyMuPDF4LLM content extraction failed: {exc}",
                    error_code="EXTRACTION_PROCESSING_ERROR",
                    status_code=500,
                ) from exc

            extracted_pages: List[ExtractedPage] = []
            total_chars = 0

            # Iterate through each page chunk preserving page boundaries and order
            for i in range(total_pages):
                page_num = i + 1
                page_text = ""

                if page_chunks and i < len(page_chunks):
                    chunk = page_chunks[i]
                    if isinstance(chunk, dict):
                        page_text = chunk.get("text", "") or ""
                    elif isinstance(chunk, str):
                        page_text = chunk

                char_count = len(page_text)
                total_chars += char_count

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        content=page_text,
                        character_count=char_count,
                    )
                )

            logger.info(
                "Successfully extracted %d pages (%d total characters) for document_id=%s",
                total_pages,
                total_chars,
                document_id,
            )

            return DocumentExtractionResult(
                document_id=document_id,
                document_name=document_name,
                total_pages=total_pages,
                extracted_pages=extracted_pages,
                total_characters=total_chars,
                extraction_status=ExtractionStatus.COMPLETED,
                error_message=None,
            )

        finally:
            # Ensure document handle is closed immediately on Windows
            if doc is not None:
                try:
                    doc.close()
                except Exception as exc:
                    logger.warning("Error closing PyMuPDF document handle: %s", str(exc))

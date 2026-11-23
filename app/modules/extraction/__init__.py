"""Document Extraction Module.

Responsible for parsing PDF policies into page-separated Markdown content
using PyMuPDF and PyMuPDF4LLM, maintaining page boundaries and table structure.
"""

from app.modules.extraction.extractor import PDFExtractor
from app.modules.extraction.repository import ExtractionRepository
from app.modules.extraction.router import router as extraction_router
from app.modules.extraction.schemas import (
    DocumentExtractionResult,
    ExtractedPage,
    ExtractionStatus,
    ExtractionSummaryResponse,
)
from app.modules.extraction.service import ExtractionService

__all__ = [
    "extraction_router",
    "PDFExtractor",
    "ExtractionRepository",
    "ExtractionService",
    "ExtractedPage",
    "DocumentExtractionResult",
    "ExtractionStatus",
    "ExtractionSummaryResponse",
]

"""Pydantic schemas and contracts for the PDF Extraction module."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class ExtractionStatus(str, Enum):
    """Lifecycle status of the PDF content extraction."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class ExtractedPage(BaseModel):
    """Structured representation of a single extracted PDF page."""

    model_config = ConfigDict(populate_by_name=True)

    page_number: int = Field(ge=1, description="1-indexed page number")
    content: str = Field(description="Markdown content extracted for this page")
    character_count: int = Field(ge=0, description="Total characters in this page's content")


class DocumentExtractionResult(BaseModel):
    """Full extraction result record stored in MongoDB and returned via API."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(description="Reference to the ingested document UUID")
    document_name: str = Field(description="Sanitized document title or filename")
    total_pages: int = Field(ge=0, description="Total pages present in the PDF")
    extracted_pages: List[ExtractedPage] = Field(
        default_factory=list,
        description="List of individual page-level extractions preserving boundaries",
    )
    total_characters: int = Field(ge=0, description="Sum of character counts across all pages")
    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.COMPLETED,
        description="Status of the extraction operation",
    )
    extracted_at: datetime = Field(
        default_factory=current_utc_time,
        description="UTC timestamp of extraction completion",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error description if extraction encountered issues",
    )


class ExtractionSummaryResponse(BaseModel):
    """Summary response returned immediately upon extraction trigger."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    document_name: str
    total_pages: int
    extracted_pages_count: int
    total_characters: int
    extraction_status: ExtractionStatus
    extracted_at: datetime

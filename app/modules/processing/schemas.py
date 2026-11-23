"""Pydantic schemas and contracts for the Document Processing (Cleaning & Normalization) module."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class ProcessingStageStatus(str, Enum):
    """Lifecycle status of the document processing/cleaning pipeline."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"


class DetectedStructure(BaseModel):
    """Reliably identifiable structural elements on a processed page.

    Only populated when structural features are confidently and deterministically detected.
    """

    model_config = ConfigDict(populate_by_name=True)

    has_headings: bool = Field(default=False, description="Whether markdown headings were detected")
    has_tables: bool = Field(default=False, description="Whether markdown tables were detected")
    has_lists: bool = Field(default=False, description="Whether bullet or numbered lists were detected")
    headings: List[str] = Field(
        default_factory=list,
        description="Reliably identified heading titles (e.g. '# Policy Scope')",
    )
    tables_count: int = Field(default=0, ge=0, description="Count of identified markdown tables")
    lists_count: int = Field(default=0, ge=0, description="Count of identified list blocks")
    paragraphs_count: int = Field(default=0, ge=0, description="Count of identified prose paragraph blocks")


class ProcessedPage(BaseModel):
    """A single normalized and cleaned page of policy text."""

    model_config = ConfigDict(populate_by_name=True)

    page_number: int = Field(ge=1, description="1-indexed page number matching source extraction")
    cleaned_content: str = Field(description="Normalized and cleaned page text in format-neutral Markdown")
    detected_structure: DetectedStructure = Field(
        default_factory=DetectedStructure,
        description="Reliably identifiable structure metadata",
    )
    character_count: int = Field(ge=0, description="Total characters in cleaned_content")


class DocumentProcessedResult(BaseModel):
    """Format-neutral contract for processed and normalized document content."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(description="Reference to the ingested document UUID")
    source_extraction_id: str = Field(description="Reference or timestamp identifier of source extraction")
    processing_status: ProcessingStageStatus = Field(
        default=ProcessingStageStatus.COMPLETED,
        description="Processing pipeline status",
    )
    processed_pages: List[ProcessedPage] = Field(
        default_factory=list,
        description="Cleaned and normalized pages preserving original sequence and page numbers",
    )
    total_pages: int = Field(ge=0, description="Total number of processed pages")
    total_characters: int = Field(ge=0, description="Total characters across all cleaned pages")
    processed_at: datetime = Field(
        default_factory=current_utc_time,
        description="UTC timestamp when processing completed",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error description if processing failed",
    )


class ProcessingSummaryResponse(BaseModel):
    """Summary response returned immediately upon processing completion."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    source_extraction_id: str
    total_pages: int
    processed_pages_count: int
    total_characters: int
    processing_status: ProcessingStageStatus
    processed_at: datetime

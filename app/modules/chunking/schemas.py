"""Pydantic schemas and contracts for the Document Chunking module."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class DocumentChunk(BaseModel):
    """Structured representation of a document chunk with rich hierarchical and scoping metadata."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(description="Deterministic unique ID: {document_id}_chunk_{index}")
    document_id: str = Field(description="Parent document UUID")
    chunk_index: int = Field(ge=0, description="0-indexed position within document chunks")
    content: str = Field(description="Cleaned, self-contained Markdown content of this chunk")
    token_count: int = Field(ge=0, description="Exact token count measured via tiktoken")
    heading: Optional[str] = Field(default=None, description="Immediate section heading for this chunk")
    heading_path: List[str] = Field(
        default_factory=list,
        description="Hierarchical section breadcrumbs (e.g. ['# 1. Leave Policy', '## 1.1 Casual Leave'])",
    )
    page_start: int = Field(ge=1, description="First source PDF page number spanned by this chunk")
    page_end: int = Field(ge=1, description="Last source PDF page number spanned by this chunk")
    source_pages: List[int] = Field(
        default_factory=list,
        description="List of all source PDF page numbers spanned by this chunk",
    )

    # Canonical school scope
    school_id: Optional[str] = Field(default=None, description="Specific school identifier")
    group_id: Optional[str] = Field(default=None, description="Educational group/trust identifier")
    branch_id: Optional[str] = Field(default=None, description="Campus/branch identifier")
    is_global: bool = Field(default=False, description="Global group-wide policy flag")
    state: Optional[str] = Field(default=None, description="Geographical jurisdiction")

    # Employee applicability
    staff_category: Optional[str] = Field(default=None, description="Target staff category")
    employment_type: Optional[str] = Field(default=None, description="Employment type applicability")
    department: Optional[str] = Field(default=None, description="Department applicability")

    # Document metadata
    document_name: str = Field(description="Sanitized document title or filename")
    document_type: str = Field(default="policy", description="Document type or classification")
    version: int = Field(default=1, ge=1, description="Document revision version")
    effective_from: Optional[datetime] = Field(default=None, description="Effective start date")
    effective_to: Optional[datetime] = Field(default=None, description="Effective end date")
    created_at: datetime = Field(default_factory=current_utc_time, description="UTC creation timestamp")


class ChunkingSummaryResponse(BaseModel):
    """Summary response returned immediately upon chunking execution."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    total_chunks: int
    average_tokens: float
    maximum_tokens: int
    processing_status: str


class PaginatedChunksResponse(BaseModel):
    """Paginated response containing document chunks."""

    model_config = ConfigDict(populate_by_name=True)

    items: List[DocumentChunk]
    total: int
    page: int
    page_size: int
    total_pages: int

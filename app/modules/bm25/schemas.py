"""Pydantic schemas and contracts for BM25 indexing and search."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BM25ChunkRecord(BaseModel):
    """Contract for a chunk record to be indexed into BM25."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(description="Unique deterministic chunk ID")
    document_id: str = Field(description="Parent document UUID")
    generation: int = Field(default=1, ge=1, description="Index generation identifier")
    chunk_index: int = Field(default=0, ge=0, description="Sequential position within document")
    content: str = Field(description="Chunk text content for full-text search")
    document_name: str = Field(description="Parent document title or filename")
    document_type: str = Field(default="policy", description="Document classification")
    version: int = Field(default=1, ge=1, description="Document revision version")

    # Scope and applicability metadata
    school_id: Optional[str] = Field(default=None, description="Specific school ID")
    group_id: Optional[str] = Field(default=None, description="School group / trust ID")
    branch_id: Optional[str] = Field(default=None, description="Campus branch ID")
    state: Optional[str] = Field(default=None, description="Geographic state filter")
    is_global: bool = Field(default=False, description="Whether policy applies globally")
    staff_category: Optional[str] = Field(default=None, description="Applicable staff category")
    employment_type: Optional[str] = Field(default=None, description="Applicable employment type")
    department: Optional[str] = Field(default=None, description="Applicable department")

    # Headings and pages
    heading: Optional[str] = Field(default=None, description="Immediate heading title")
    heading_path: List[str] = Field(default_factory=list, description="Hierarchical heading breadcrumbs")
    page_start: int = Field(default=1, ge=1, description="Starting page number")
    page_end: int = Field(default=1, ge=1, description="Ending page number")
    effective_from: Optional[str] = Field(default=None, description="ISO 8601 effective start date")
    effective_to: Optional[str] = Field(default=None, description="ISO 8601 effective end date")


class BM25SearchResult(BaseModel):
    """Result of a BM25 keyword query."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str
    document_id: str
    generation: int
    content: str
    document_name: str
    document_type: str = "policy"
    version: int = 1
    bm25_score: float = Field(description="BM25 relevance score (higher is more relevant)")
    bm25_rank: int = Field(ge=1, description="1-based rank order from BM25 retrieval")

    school_id: Optional[str] = None
    group_id: Optional[str] = None
    branch_id: Optional[str] = None
    state: Optional[str] = None
    is_global: bool = False
    staff_category: Optional[str] = None
    employment_type: Optional[str] = None
    department: Optional[str] = None
    heading: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None

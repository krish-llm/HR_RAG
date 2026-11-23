"""Pydantic schemas and contracts for Context Building & Citation Manifest (Phase 7)."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.schemas import EmployeeContext, QueryUnderstandingResult
from app.schemas.common import current_utc_time


class SourceCitation(BaseModel):
    """Structured citation manifest record linking context blocks back to authoritative source documents."""

    model_config = ConfigDict(populate_by_name=True)

    citation_index: int = Field(ge=1, description="1-based numerical index of the citation, e.g. 1 for Source 1")
    citation_label: str = Field(description="Short citation tag e.g. '[Source 1]'")
    citation_reference: str = Field(
        description="Full formatted source reference e.g. '[Source 1: Leave Policy, pages 2–3, Leave Policy > Sick Leave]'"
    )
    document_id: str = Field(description="Authoritative document UUID")
    document_name: str = Field(description="Policy document title or sanitized filename")
    document_type: str = Field(default="policy", description="Classification of source document")
    version: int = Field(default=1, ge=1, description="Document revision version")
    chunk_id: str = Field(description="Chunk identifier")
    heading: Optional[str] = Field(default=None, description="Immediate section heading")
    heading_path: List[str] = Field(default_factory=list, description="Section hierarchy breadcrumbs")
    page_start: int = Field(ge=1, description="First source PDF page number")
    page_end: int = Field(ge=1, description="Last source PDF page number")

    # Scope attributes
    school_id: Optional[str] = Field(default=None, description="Specific school identifier")
    group_id: Optional[str] = Field(default=None, description="Educational group/trust identifier")
    branch_id: Optional[str] = Field(default=None, description="Campus/branch identifier")

    # Effective dates
    effective_from: Optional[datetime] = Field(default=None, description="Effective start date")
    effective_to: Optional[datetime] = Field(default=None, description="Effective end date")


class ContextBuildResult(BaseModel):
    """Internal contract returned by the ContextBuilder engine."""

    model_config = ConfigDict(populate_by_name=True)

    assembled_context: str = Field(description="Formatted and grouped markdown context string")
    source_manifest: List[SourceCitation] = Field(
        default_factory=list,
        description="List of structured citations matching the order of sources in the context",
    )
    total_tokens: int = Field(ge=0, description="Total tiktoken token count of the assembled context")
    token_budget: int = Field(ge=1, description="Token budget enforced during assembly")
    selected_chunks_count: int = Field(ge=0, description="Number of complete chunks included in the context")
    skipped_chunks_count: int = Field(
        ge=0,
        description="Number of candidate chunks skipped (due to duplication or budget exhaustion)",
    )
    selected_chunks: List[RankedChunk] = Field(
        default_factory=list,
        description="List of complete RankedChunk instances included in the context",
    )


class ContextRequest(BaseModel):
    """Public API request schema for context building."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        min_length=1,
        description="Natural language question regarding HR policies",
    )
    employee_context: Optional[EmployeeContext] = Field(
        default=None,
        description="Optional employee categorization for personalized applicability filtering",
    )
    as_of_date: Optional[datetime] = Field(
        default=None,
        description="Optional evaluation date for policy effectiveness (defaults to current UTC time)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve before reranking and context assembly",
    )
    token_budget: Optional[int] = Field(
        default=None,
        ge=100,
        le=16000,
        description="Optional token budget override for context window assembly",
    )


class ContextResponse(BaseModel):
    """Public API response schema representing reranked and assembled policy context."""

    model_config = ConfigDict(populate_by_name=True)

    query_understanding: QueryUnderstandingResult = Field(
        description="Analysis and structured understanding of the query",
    )
    ranked_chunks: List[RankedChunk] = Field(
        default_factory=list,
        description="Complete list of candidate chunks evaluated and ranked by the reranker",
    )
    assembled_context: str = Field(
        description="Formatted, deduplicated, and budget-constrained context ready for LLM generation",
    )
    source_manifest: List[SourceCitation] = Field(
        default_factory=list,
        description="Structured source manifest mapping citation tags to source documents and pages",
    )
    total_tokens: int = Field(ge=0, description="Total tokens utilized in assembled context")
    token_budget: int = Field(ge=1, description="Effective token budget applied")
    selected_chunks_count: int = Field(ge=0, description="Count of complete chunks safely included")
    skipped_chunks_count: int = Field(
        ge=0,
        description="Count of candidate chunks omitted due to duplication or exceeding token budget",
    )
    status: str = Field(
        default="success",
        description="Context building status indicator: 'success' or 'empty'",
    )
    created_at: datetime = Field(
        default_factory=current_utc_time,
        description="UTC timestamp of context generation",
    )

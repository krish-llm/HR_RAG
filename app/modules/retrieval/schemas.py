"""Pydantic schemas and contracts for Query Understanding & Semantic Retrieval (Phase 6)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class EmployeeContext(BaseModel):
    """Employee categorization attributes for policy applicability filtering."""

    model_config = ConfigDict(populate_by_name=True)

    staff_category: Optional[str] = Field(
        default=None,
        description="Target staff category (e.g. teaching, non_teaching, administrative)",
    )
    employment_type: Optional[str] = Field(
        default=None,
        description="Employment arrangement (e.g. full_time, permanent, contract, part_time)",
    )
    department: Optional[str] = Field(
        default=None,
        description="Department applicability (e.g. science, mathematics, administration)",
    )


class QueryUnderstandingResult(BaseModel):
    """Deterministic output contract for query normalization, intent classification, and scoping."""

    model_config = ConfigDict(populate_by_name=True)

    original_query: str = Field(description="Verbatim user question as submitted")
    normalized_query: str = Field(description="Cleaned, whitespace-normalized query string")
    intent: str = Field(description="Identified HR policy intent (e.g. leave, compensation, conduct)")
    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="Deterministic list of key terms and domain nouns extracted from query",
    )
    employee_context: Optional[EmployeeContext] = Field(
        default=None,
        description="Validated employee applicability context (explicit or extracted)",
    )
    requested_scope: Dict[str, Any] = Field(
        default_factory=dict,
        description="Canonical school scoping parameters applied from request context",
    )
    as_of_date: datetime = Field(
        default_factory=current_utc_time,
        description="Effective date used for policy validity verification",
    )


class RetrievedChunk(BaseModel):
    """Structured representation of a retrieved document chunk with rich citation metadata."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(description="Deterministic chunk identifier")
    document_id: str = Field(description="Authoritative parent document UUID")
    document_name: str = Field(description="Policy document title or sanitized filename")
    document_type: str = Field(default="policy", description="Classification of source document")
    version: int = Field(default=1, ge=1, description="Document revision version")
    content: str = Field(description="Cleaned chunk text content")
    heading: Optional[str] = Field(default=None, description="Immediate section heading")
    heading_path: List[str] = Field(
        default_factory=list,
        description="Hierarchical section breadcrumbs for contextual navigation",
    )
    page_start: int = Field(ge=1, description="First source PDF page number spanned by this chunk")
    page_end: int = Field(ge=1, description="Last source PDF page number spanned by this chunk")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine/vector similarity score between query and chunk",
    )

    # Scoping attributes
    school_id: Optional[str] = Field(default=None, description="Specific school identifier")
    group_id: Optional[str] = Field(default=None, description="Educational group/trust identifier")
    branch_id: Optional[str] = Field(default=None, description="Campus/branch identifier")

    # Dates
    effective_from: Optional[datetime] = Field(default=None, description="Effective start date")
    effective_to: Optional[datetime] = Field(default=None, description="Effective end date")

    # Modality & RRF metadata (preserved for debugging and analysis)
    retrieval_mode: Optional[str] = Field(default=None, description="Retrieval modality: 'dense', 'bm25', or 'hybrid'")
    dense_score: Optional[float] = Field(default=None, description="Original dense vector similarity score")
    dense_rank: Optional[int] = Field(default=None, description="1-based candidate rank from dense search")
    bm25_score: Optional[float] = Field(default=None, description="Original BM25 keyword match score")
    bm25_rank: Optional[int] = Field(default=None, description="1-based candidate rank from BM25 search")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score (separate from similarity)")


class RetrievalRequest(BaseModel):
    """Public API request schema for semantic retrieval."""

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
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve",
    )


class RetrievalResponse(BaseModel):
    """Public API response schema representing semantic retrieval results."""

    model_config = ConfigDict(populate_by_name=True)

    query_understanding: QueryUnderstandingResult = Field(
        description="Analysis and structured understanding of the query",
    )
    chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Validated, scoped, and ranked candidate chunks",
    )
    total_candidates: int = Field(
        ge=0,
        description="Total raw candidate vectors evaluated from ChromaDB",
    )
    total_results: int = Field(
        ge=0,
        description="Total valid chunks returned after strict scoping and policy validation",
    )
    retrieval_status: str = Field(
        default="success",
        description="Retrieval status indicator: 'success' or 'empty'",
    )
    retrieval_mode: str = Field(
        default="dense",
        description="Retrieval modality applied: 'dense' or 'hybrid'",
    )
    retrieved_at: datetime = Field(
        default_factory=current_utc_time,
        description="UTC timestamp of retrieval execution",
    )

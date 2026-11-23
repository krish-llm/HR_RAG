"""Schemas and contracts for Phase 7 Reranking."""

from typing import Optional
from pydantic import ConfigDict, Field
from app.modules.retrieval.schemas import RetrievedChunk


class RankedChunk(RetrievedChunk):
    """Retrieved document chunk enhanced with composite reranking scores and ranks."""

    model_config = ConfigDict(populate_by_name=True)

    rerank_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite reranking score combining semantic, lexical, and heading relevance (heuristic, not a probability)",
    )
    lexical_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized lexical match score based on query keywords and chunk text content",
    )
    heading_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized heading relevance score based on query terms and section breadcrumbs",
    )
    original_rank: int = Field(
        ge=1,
        description="1-based original retrieval rank order from initial vector search",
    )
    rerank_rank: int = Field(
        ge=1,
        description="1-based final rank order after composite reranking",
    )
    score_explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of how the composite reranking score was computed",
    )

"""Pydantic schemas and contracts for the Embedding & Vector Indexing module."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class IndexingStageStatus(str, Enum):
    """Pipeline statuses for document embedding and vector indexing."""

    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkEmbeddingRecord(BaseModel):
    """MongoDB storage schema for tracking chunk-level embeddings and content hashes."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    document_id: str = Field(description="Parent document UUID")
    embedding_model: str = Field(description="Name of the embedding model used")
    embedding_version: str = Field(default="1", description="Version of the embedding generation")
    content_hash: str = Field(description="SHA-256 hash of chunk content to detect changes and avoid re-embedding")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")
    embedding_status: str = Field(default="completed", description="Status of chunk embedding: completed or failed")
    indexed_at: datetime = Field(default_factory=current_utc_time, description="Timestamp when embedding was generated")
    error_message: Optional[str] = Field(default=None, description="Error message if embedding generation failed")


class DocumentIndexingRecord(BaseModel):
    """MongoDB record tracking aggregate indexing status and progress for a document."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(description="Parent document UUID")
    total_chunks: int = Field(default=0, ge=0, description="Total expected chunks for this document")
    embedded_chunks: int = Field(default=0, ge=0, description="Number of chunks successfully embedded")
    indexed_chunks: int = Field(default=0, ge=0, description="Number of chunks successfully indexed in ChromaDB")
    failed_chunks: int = Field(default=0, ge=0, description="Number of chunks that failed embedding or indexing")
    active_generation: int = Field(default=0, ge=0, description="Currently active vector index generation")
    target_generation: Optional[int] = Field(default=None, ge=1, description="Generation currently being indexed")
    embedding_model: str = Field(description="Embedding model used for indexing")
    indexing_status: IndexingStageStatus = Field(
        default=IndexingStageStatus.CHUNKED,
        description="Overall document indexing pipeline status",
    )
    last_indexed_at: Optional[datetime] = Field(default=None, description="Timestamp of most recent successful indexing")
    error_message: Optional[str] = Field(default=None, description="Detailed error message if indexing failed")
    updated_at: datetime = Field(default_factory=current_utc_time, description="Timestamp of last status modification")


class IndexingStatusResponse(BaseModel):
    """Public API response schema representing document indexing progress and status."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(description="Unique document UUID")
    total_chunks: int = Field(description="Total chunks in the document")
    embedded_chunks: int = Field(description="Count of embedded chunks")
    indexed_chunks: int = Field(description="Count of chunks active in ChromaDB")
    failed_chunks: int = Field(description="Count of failed chunks")
    active_generation: int = Field(default=0, ge=0, description="Currently active vector index generation")
    embedding_model: str = Field(description="Embedding model name used")
    indexing_status: str = Field(description="Status: chunked, embedding, indexing, indexed, or failed")
    last_indexed_at: Optional[datetime] = Field(default=None, description="Timestamp when indexing completed")
    error_message: Optional[str] = Field(default=None, description="Error description if status is failed")

"""Pydantic schemas and contracts for the Document Ingestion module."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas.common import current_utc_time


class DocumentStatus(str, Enum):
    """Allowed document lifecycle statuses."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


class ProcessingStatus(str, Enum):
    """Status of background RAG processing/indexing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EmployeeApplicability(BaseModel):
    """Employee category applicability metadata."""

    model_config = ConfigDict(populate_by_name=True)

    staff_category: Optional[str] = Field(
        default=None,
        description="Target staff group (e.g. teaching, non-teaching, administrative, all)",
    )
    employment_type: Optional[str] = Field(
        default=None,
        description="Employment arrangement (e.g. full-time, part-time, contract, all)",
    )
    department: Optional[str] = Field(
        default=None,
        description="Departmental scope (e.g. science, mathematics, administrative, all)",
    )


class DocumentCreate(BaseModel):
    """Internal model for registering an ingested document in MongoDB."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(description="Unique UUID string for document")
    document_name: str = Field(description="Original sanitized document filename or title")
    stored_file_path: str = Field(description="Path to stored PDF on local filesystem")
    file_size_bytes: int = Field(gt=0, description="Size of file in bytes")
    file_hash: str = Field(description="SHA-256 hash of file content for integrity and deduplication")

    # Canonical School Scope
    school_id: Optional[str] = Field(default=None, description="Specific school identifier")
    group_id: Optional[str] = Field(default=None, description="Educational group/trust identifier")
    branch_id: Optional[str] = Field(default=None, description="Campus/branch identifier")
    is_global: bool = Field(default=False, description="Global group-wide policy flag")
    state: Optional[str] = Field(default=None, description="Geographical jurisdiction")

    # Document details
    document_type: str = Field(default="policy", description="Classification of document")
    version: int = Field(default=1, ge=1, description="Document revision version")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Document lifecycle status")

    # Dates & applicability
    effective_from: Optional[datetime] = Field(default=None, description="Date policy becomes effective")
    effective_to: Optional[datetime] = Field(default=None, description="Date policy expires or is superseded")
    employee_applicability: EmployeeApplicability = Field(
        default_factory=EmployeeApplicability,
        description="Target employee applicability metadata",
    )

    # Ingestion tracking
    uploaded_at: datetime = Field(default_factory=current_utc_time, description="UTC upload timestamp")
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Initial processing status",
    )
    error_message: Optional[str] = Field(default=None, description="Error explanation if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata")

    @model_validator(mode="after")
    def validate_scope(self) -> "DocumentCreate":
        """Verify at least one scoping dimension is defined."""
        if not self.is_global and not self.school_id and not self.group_id and not self.state:
            raise ValueError(
                "Document must specify at least one scoping dimension: "
                "school_id, group_id, state, or is_global=True."
            )
        return self


class DocumentResponse(BaseModel):
    """Output DTO for ingested document records."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    document_name: str
    stored_file_path: str
    file_size_bytes: int
    file_hash: str
    school_id: Optional[str] = None
    group_id: Optional[str] = None
    branch_id: Optional[str] = None
    is_global: bool = False
    state: Optional[str] = None
    document_type: str
    version: int
    status: DocumentStatus
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    employee_applicability: EmployeeApplicability
    uploaded_at: datetime
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentFilterParams(BaseModel):
    """Query parameters for document listing and filtering."""

    model_config = ConfigDict(populate_by_name=True)

    school_id: Optional[str] = None
    group_id: Optional[str] = None
    branch_id: Optional[str] = None
    is_global: Optional[bool] = None
    state: Optional[str] = None
    document_type: Optional[str] = None
    status: Optional[DocumentStatus] = None
    processing_status: Optional[ProcessingStatus] = None
    page: int = Field(default=1, ge=1, description="Page index (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

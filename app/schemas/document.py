"""Shared base schemas for documents and multi-tenant persistence.

Our canonical school scoping hierarchy uses school_id, group_id, branch_id,
and is_global, with state for geographical policy applicability.
Generic tenant_id is avoided in favor of explicit educational hierarchy scoping.
"""

from datetime import datetime
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas.common import current_utc_time


class SchoolScope(BaseModel):
    """Canonical school scope defining multi-tenant policy applicability.

    Hierarchical dimensions:
    - is_global: Applies across the entire organization/group network.
    - group_id: Identifies the educational society / trust / group.
    - school_id: Identifies the specific school institution.
    - branch_id: Identifies a specific campus/branch within a school.
    - state: Regional/state jurisdiction (e.g. Karnataka, Maharashtra) for localized laws.
    """

    model_config = ConfigDict(populate_by_name=True)

    group_id: Optional[str] = Field(
        default=None,
        description="Educational group/trust identifier (e.g. DPS Society, Ryan Group)",
    )
    school_id: Optional[str] = Field(
        default=None,
        description="Specific school identifier",
    )
    branch_id: Optional[str] = Field(
        default=None,
        description="Specific campus or branch identifier",
    )
    is_global: bool = Field(
        default=False,
        description="Flag indicating if policy applies across all schools globally",
    )
    state: Optional[str] = Field(
        default=None,
        description="Geographical state/region for localized policy compliance",
    )

    @model_validator(mode="after")
    def validate_scope_presence(self) -> "SchoolScope":
        """Verify that at least one scope dimension is defined."""
        if not self.is_global and not self.school_id and not self.group_id and not self.state:
            raise ValueError(
                "SchoolScope requires at least one scoping dimension: "
                "school_id, group_id, state, or is_global=True."
            )
        return self


class BaseDocument(BaseModel):
    """Base schema for multi-tenant documents stored in MongoDB & metadata stores."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier",
    )

    # Canonical school hierarchy scope fields
    school_id: Optional[str] = Field(
        default=None,
        description="Specific school identifier",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Educational group/trust identifier",
    )
    branch_id: Optional[str] = Field(
        default=None,
        description="Specific campus/branch identifier",
    )
    is_global: bool = Field(
        default=False,
        description="Indicates whether policy applies globally across all institutions",
    )
    state: Optional[str] = Field(
        default=None,
        description="Geographical state jurisdiction for policy compliance",
    )

    # Document versioning
    version: int = Field(
        default=1,
        ge=1,
        description="Monotonically increasing document version number",
    )

    # Timestamps & metadata
    created_at: datetime = Field(
        default_factory=current_utc_time,
        description="Creation timestamp in UTC",
    )
    updated_at: datetime = Field(
        default_factory=current_utc_time,
        description="Last updated timestamp in UTC",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary document metadata (e.g. tags, author, policy categories)",
    )

    @model_validator(mode="after")
    def validate_document_scope(self) -> "BaseDocument":
        """Ensure document specifies at least one valid scoping dimension."""
        if not self.is_global and not self.school_id and not self.group_id and not self.state:
            raise ValueError(
                "BaseDocument requires at least one scoping dimension: "
                "school_id, group_id, state, or is_global=True."
            )
        return self

    @property
    def scope(self) -> SchoolScope:
        """Helper to extract the SchoolScope object from document fields."""
        return SchoolScope(
            group_id=self.group_id,
            school_id=self.school_id,
            branch_id=self.branch_id,
            is_global=self.is_global,
            state=self.state,
        )


class BaseDocumentCreate(BaseModel):
    """Base schema for creating a new document with explicit school scoping."""

    school_id: Optional[str] = Field(default=None, description="Target school ID")
    group_id: Optional[str] = Field(default=None, description="Target group ID")
    branch_id: Optional[str] = Field(default=None, description="Target branch ID")
    is_global: bool = Field(default=False, description="Global policy flag")
    state: Optional[str] = Field(default=None, description="Geographical jurisdiction")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initial metadata")

    @model_validator(mode="after")
    def validate_create_scope(self) -> "BaseDocumentCreate":
        """Ensure create payload has valid scoping."""
        if not self.is_global and not self.school_id and not self.group_id and not self.state:
            raise ValueError(
                "BaseDocumentCreate requires at least one scoping dimension: "
                "school_id, group_id, state, or is_global=True."
            )
        return self


class BaseDocumentUpdate(BaseModel):
    """Base schema for updating existing document metadata or scoping."""

    branch_id: Optional[str] = Field(default=None, description="Updated branch ID")
    state: Optional[str] = Field(default=None, description="Updated geographical jurisdiction")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata dictionary")

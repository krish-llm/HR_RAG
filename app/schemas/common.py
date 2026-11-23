"""Common base schemas and API response wrappers."""

from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


def current_utc_time() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class ErrorDetail(BaseModel):
    """Detailed error item for form/field validation or internal issues."""

    field: Optional[str] = Field(default=None, description="Field name where error occurred, if applicable")
    message: str = Field(description="Human-readable error description")


class ApiResponse(BaseModel, Generic[T]):
    """Standardized API response envelope for successful requests."""

    success: bool = Field(default=True, description="Indicates if the request succeeded")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: Optional[T] = Field(default=None, description="Payload data")
    timestamp: datetime = Field(default_factory=current_utc_time, description="UTC timestamp of response")


class PaginatedData(BaseModel, Generic[T]):
    """Standardized wrapper for paginated collections."""

    items: List[T] = Field(description="Page items list")
    total: int = Field(description="Total count of matching records")
    page: int = Field(description="Current page index (1-based)")
    page_size: int = Field(description="Page size limit")
    total_pages: int = Field(description="Calculated total number of pages")


class ErrorResponse(BaseModel):
    """Standardized API error response envelope."""

    success: bool = Field(default=False, description="Always False for error responses")
    error_code: str = Field(default="INTERNAL_SERVER_ERROR", description="Machine-readable error identifier")
    message: str = Field(description="User-friendly error message")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="List of granular error details")
    timestamp: datetime = Field(default_factory=current_utc_time, description="UTC timestamp of error")


class ServiceHealth(BaseModel):
    """Health information for an individual dependency/service."""

    status: str = Field(description="Service status: healthy, degraded, or unhealthy")
    latency_ms: Optional[float] = Field(default=None, description="Latency in milliseconds if measured")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional diagnostic details")


class HealthResponse(BaseModel):
    """Comprehensive health check response."""

    status: str = Field(description="Overall system health status: healthy or degraded")
    app_name: str = Field(description="Name of the application")
    environment: str = Field(description="Current deployment environment")
    version: str = Field(description="Current application version")
    services: Dict[str, ServiceHealth] = Field(description="Health statuses of backing services")
    timestamp: datetime = Field(default_factory=current_utc_time, description="UTC timestamp of health check")

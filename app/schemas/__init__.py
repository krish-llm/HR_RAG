"""Shared schemas package."""

from app.schemas.common import (
    ApiResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PaginatedData,
    ServiceHealth,
    current_utc_time,
)
from app.schemas.document import (
    BaseDocument,
    BaseDocumentCreate,
    BaseDocumentUpdate,
    SchoolScope,
)

__all__ = [
    "ApiResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "PaginatedData",
    "ServiceHealth",
    "current_utc_time",
    "SchoolScope",
    "BaseDocument",
    "BaseDocumentCreate",
    "BaseDocumentUpdate",
]

"""Document Ingestion Module.

Responsible for handling PDF uploads, file integrity and magic-byte validation,
local storage management, and MongoDB policy registration.
"""

from app.modules.ingestion.repository import DocumentRepository
from app.modules.ingestion.router import router as documents_router
from app.modules.ingestion.schemas import (
    DocumentCreate,
    DocumentFilterParams,
    DocumentResponse,
    DocumentStatus,
    EmployeeApplicability,
    ProcessingStatus,
)
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import LocalStorageService

__all__ = [
    "documents_router",
    "DocumentCreate",
    "DocumentResponse",
    "DocumentFilterParams",
    "DocumentStatus",
    "ProcessingStatus",
    "EmployeeApplicability",
    "DocumentRepository",
    "LocalStorageService",
    "IngestionService",
]

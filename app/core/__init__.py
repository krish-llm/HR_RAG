"""Core application package containing exception definitions and handlers."""

from app.core.exceptions import (
    AppException,
    ChunkingException,
    ConfigurationException,
    DatabaseConnectionException,
    DuplicateDocumentException,
    EmbeddingException,
    ExtractionException,
    FileValidationException,
    IndexingException,
    NotFoundException,
    ProcessingException,
    StorageException,
    TenantNotFoundException,
)
from app.core.handlers import (
    app_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)

__all__ = [
    "AppException",
    "ChunkingException",
    "ConfigurationException",
    "DatabaseConnectionException",
    "DuplicateDocumentException",
    "EmbeddingException",
    "ExtractionException",
    "FileValidationException",
    "IndexingException",
    "NotFoundException",
    "ProcessingException",
    "StorageException",
    "TenantNotFoundException",
    "app_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",
    "validation_exception_handler",
]

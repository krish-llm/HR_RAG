"""Application-wide custom exception definitions."""

from typing import Any, Dict, List, Optional
from app.schemas.common import ErrorDetail


class AppException(Exception):
    """Base application exception for standardized error handling."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or []


class NotFoundException(AppException):
    """Exception raised when an entity or resource is not found."""

    def __init__(
        self,
        message: str = "Requested resource not found",
        error_code: str = "RESOURCE_NOT_FOUND",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=404,
            details=details,
        )


class TenantNotFoundException(AppException):
    """Exception raised when tenant context is missing or invalid."""

    def __init__(
        self,
        tenant_id: str,
        message: Optional[str] = None,
    ):
        msg = message or f"Tenant '{tenant_id}' was not found or is inactive"
        super().__init__(
            message=msg,
            error_code="TENANT_NOT_FOUND",
            status_code=404,
            details=[ErrorDetail(field="tenant_id", message=msg)],
        )


class DatabaseConnectionException(AppException):
    """Exception raised when database connectivity fails."""

    def __init__(
        self,
        message: str = "Database service is unavailable",
        error_code: str = "DATABASE_UNAVAILABLE",
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=503,
        )


class FileValidationException(AppException):
    """Exception raised when an uploaded file fails validation."""

    def __init__(
        self,
        message: str,
        error_code: str = "FILE_VALIDATION_ERROR",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=400,
            details=details,
        )


class DuplicateDocumentException(AppException):
    """Exception raised when an identical document is uploaded."""

    def __init__(
        self,
        message: str = "A document with identical content and scoping already exists",
        error_code: str = "DUPLICATE_DOCUMENT",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=409,
            details=details,
        )


class StorageException(AppException):
    """Exception raised when physical file storage operations fail."""

    def __init__(
        self,
        message: str = "File storage operation failed",
        error_code: str = "STORAGE_ERROR",
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=500,
        )


class ExtractionException(AppException):
    """Exception raised when PDF content extraction fails."""

    def __init__(
        self,
        message: str = "PDF extraction failed",
        error_code: str = "EXTRACTION_FAILED",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ProcessingException(AppException):
    """Exception raised when document processing, cleaning, or normalization fails."""

    def __init__(
        self,
        message: str = "Document processing failed",
        error_code: str = "PROCESSING_FAILED",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ChunkingException(AppException):
    """Exception raised when document chunking fails."""

    def __init__(
        self,
        message: str = "Document chunking failed",
        error_code: str = "CHUNKING_FAILED",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ConfigurationException(AppException):
    """Exception raised on critical configuration errors."""

    def __init__(
        self,
        message: str = "Application configuration error",
        error_code: str = "CONFIGURATION_ERROR",
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=500,
        )


class EmbeddingException(AppException):
    """Exception raised when vector embedding generation fails."""

    def __init__(
        self,
        message: str = "Embedding generation failed",
        error_code: str = "EMBEDDING_FAILED",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class IndexingException(AppException):
    """Exception raised when vector database indexing fails."""

    def __init__(
        self,
        message: str = "Vector indexing failed",
        error_code: str = "INDEXING_FAILED",
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class GenerationException(AppException):
    """Exception raised when LLM answer generation or parsing fails."""

    def __init__(
        self,
        message: str = "LLM answer generation failed",
        error_code: str = "GENERATION_FAILED",
        status_code: int = 502,
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )

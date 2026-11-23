"""Centralized FastAPI exception handlers."""

import logging
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.schemas.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    logger.warning(
        "Application exception on %s %s: [%s] %s",
        request.method,
        request.url.path,
        exc.error_code,
        exc.message,
    )
    error_payload = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details if exc.details else None,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump(mode="json"),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors from Pydantic."""
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    details = []
    for err in exc.errors():
        loc_str = " -> ".join(str(loc) for loc in err.get("loc", []))
        details.append(ErrorDetail(field=loc_str, message=err.get("msg", "Invalid input")))

    error_payload = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Request parameter validation failed",
        details=details,
    )
    return JSONResponse(
        status_code=422,
        content=error_payload.model_dump(mode="json"),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard Starlette/FastAPI HTTPExceptions (e.g. 404, 405)."""
    logger.info(
        "HTTP exception on %s %s: status=%s detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    error_payload = ErrorResponse(
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump(mode="json"),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any uncaught exceptions to prevent leakage of internal stack traces."""
    logger.exception(
        "Unhandled server error on %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
    )
    error_payload = ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal server error occurred",
    )
    return JSONResponse(
        status_code=500,
        content=error_payload.model_dump(mode="json"),
    )

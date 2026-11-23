"""Health check API endpoints."""

import time
from fastapi import APIRouter, Depends
from app import __version__
from app.api.dependencies import Settings, get_settings
from app.db.chroma import ChromaDBService, chroma_service
from app.db.mongo import MongoDatabaseManager, mongo_manager
from app.schemas.common import HealthResponse, ServiceHealth

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application Health and Readiness Check",
    description="Inspects availability of application core, MongoDB, and ChromaDB vector store.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return health status of the application and its connected dependencies."""
    services = {}
    is_fully_healthy = True

    # Check MongoDB
    mongo_start = time.perf_counter()
    mongo_ok = await mongo_manager.ping()
    mongo_latency = round((time.perf_counter() - mongo_start) * 1000, 2)

    if mongo_ok:
        services["mongodb"] = ServiceHealth(
            status="healthy",
            latency_ms=mongo_latency,
            details={"database": settings.MONGODB_DB_NAME},
        )
    else:
        services["mongodb"] = ServiceHealth(
            status="unhealthy",
            latency_ms=mongo_latency,
            details={"error": "Unable to ping MongoDB instance"},
        )
        is_fully_healthy = False

    # Check ChromaDB
    chroma_start = time.perf_counter()
    chroma_ok = chroma_service.is_healthy()
    chroma_latency = round((time.perf_counter() - chroma_start) * 1000, 2)

    if chroma_ok:
        services["chromadb"] = ServiceHealth(
            status="healthy",
            latency_ms=chroma_latency,
            details={"persist_directory": settings.CHROMA_PERSIST_DIRECTORY},
        )
    else:
        services["chromadb"] = ServiceHealth(
            status="unhealthy",
            latency_ms=chroma_latency,
            details={"error": "ChromaDB client is unresponsive or uninitialized"},
        )
        is_fully_healthy = False

    overall_status = "healthy" if is_fully_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        version=__version__,
        services=services,
    )

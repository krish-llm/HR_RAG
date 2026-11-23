"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.router import api_v1_router
from app.api.v1.health import router as health_router
from app.config import get_settings
from app.core.exceptions import AppException
from app.core.handlers import (
    app_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.db.chroma import chroma_service
from app.db.mongo import mongo_manager
from app.logging_config import clear_request_id, get_logger, set_request_id, setup_logging

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and graceful shutdown."""
    logger.info("Starting up %s (version: %s)...", settings.APP_NAME, __version__)

    # Initialize ChromaDB persistent storage
    try:
        chroma_service.initialize(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)
        logger.info("ChromaDB vector database service initialized.")
    except Exception as exc:
        logger.error("Failed to initialize ChromaDB service: %s", str(exc))

    # Initialize MongoDB async client connection pool
    try:
        await mongo_manager.connect(
            uri=settings.MONGODB_URL,
            db_name=settings.MONGODB_DB_NAME,
            min_pool_size=settings.MONGODB_MIN_POOL_SIZE,
            max_pool_size=settings.MONGODB_MAX_POOL_SIZE,
            timeout_ms=settings.MONGODB_TIMEOUT_MS,
        )
        if mongo_manager.db is not None:
            from app.modules.chunking.repository import ChunkRepository
            from app.modules.extraction.repository import ExtractionRepository
            from app.modules.indexing.repository import IndexingRepository
            from app.modules.ingestion.repository import DocumentRepository
            from app.modules.processing.repository import ProcessedRepository
            doc_repo = DocumentRepository(mongo_manager.db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
            await doc_repo.ensure_indexes()
            extract_repo = ExtractionRepository(mongo_manager.db, collection_name=settings.EXTRACTIONS_COLLECTION_NAME)
            await extract_repo.ensure_indexes()
            processed_repo = ProcessedRepository(mongo_manager.db, collection_name=settings.PROCESSED_COLLECTION_NAME)
            await processed_repo.ensure_indexes()
            chunk_repo = ChunkRepository(mongo_manager.db, collection_name=settings.CHUNKS_COLLECTION_NAME)
            await chunk_repo.ensure_indexes()
            indexing_repo = IndexingRepository(
                mongo_manager.db,
                embeddings_collection_name=settings.INDEXING_COLLECTION_NAME,
                tracking_collection_name=settings.INDEXING_TRACKING_COLLECTION_NAME,
            )
            await indexing_repo.ensure_indexes()
    except Exception as exc:
        logger.error("Failed to connect to MongoDB: %s", str(exc))

    logger.info("Application startup completed successfully.")

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)

    # Gracefully close database connections and BM25 index
    await mongo_manager.close()
    chroma_service.close()
    from app.api.dependencies import reset_bm25_provider

    reset_bm25_provider()

    logger.info("Application shutdown completed.")


def create_application() -> FastAPI:
    """Factory function for creating and configuring the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description="Multi-tenant HR Policy RAG API foundation",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and correlation middleware
    @application.middleware("http")
    async def request_id_middleware(request, call_next):
        req_id = request.headers.get("X-Request-ID")
        active_id = set_request_id(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = active_id
            return response
        finally:
            clear_request_id()

    # Centralized Exception Handlers
    application.add_exception_handler(AppException, app_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    # Root /health endpoint
    application.include_router(health_router)

    # API v1 routes
    application.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    @application.get(
        "/",
        tags=["Root"],
        summary="API Root Information",
        description="Returns basic application status and navigation links.",
    )
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": __version__,
            "environment": settings.APP_ENV,
            "status": "online",
            "docs_url": "/docs",
            "health_check": "/health",
        }

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

"""Central API router registry."""

from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.modules.chunking.router import router as chunking_router
from app.modules.extraction.router import router as extraction_router
from app.modules.context.router import router as context_router
from app.modules.generation.router import router as generation_router
from app.modules.indexing.router import router as indexing_router
from app.modules.ingestion.router import router as documents_router
from app.modules.processing.router import router as processing_router
from app.modules.retrieval.router import router as retrieval_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(extraction_router)
api_v1_router.include_router(processing_router)
api_v1_router.include_router(chunking_router)
api_v1_router.include_router(indexing_router)
api_v1_router.include_router(retrieval_router, prefix="/query")
api_v1_router.include_router(context_router, prefix="/query")
api_v1_router.include_router(generation_router, prefix="/query")

__all__ = ["api_v1_router"]

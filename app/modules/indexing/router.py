"""FastAPI routes for Document Embedding and Vector Indexing operations."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import (
    AsyncDatabase,
    BaseBM25IndexProvider,
    Settings,
    get_bm25_provider,
    get_mongo_db,
    get_settings,
)
from app.db.chroma import ChromaDBService, get_chroma_service
from app.modules.chunking.repository import ChunkRepository
from app.modules.indexing.provider import AzureOpenAIEmbeddingProvider
from app.modules.indexing.repository import IndexingRepository, VectorIndexRepository
from app.modules.indexing.schemas import IndexingStatusResponse
from app.modules.indexing.service import IndexingService
from app.modules.ingestion.repository import DocumentRepository
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/documents", tags=["Vector Indexing"])


def get_indexing_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    chroma: ChromaDBService = Depends(get_chroma_service),
    bm25: BaseBM25IndexProvider = Depends(get_bm25_provider),
    settings: Settings = Depends(get_settings),
) -> IndexingService:
    """Dependency provider for IndexingService."""
    document_repo = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
    chunk_repo = ChunkRepository(db, collection_name=settings.CHUNKS_COLLECTION_NAME)
    indexing_repo = IndexingRepository(
        db,
        embeddings_collection_name=settings.INDEXING_COLLECTION_NAME,
        tracking_collection_name=settings.INDEXING_TRACKING_COLLECTION_NAME,
    )
    vector_repo = VectorIndexRepository(
        chroma_service=chroma,
        collection_name=settings.CHROMA_CHUNKS_COLLECTION_NAME,
    )
    embedding_provider = AzureOpenAIEmbeddingProvider(
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    return IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=embedding_provider,
        bm25_provider=bm25,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@router.post(
    "/rebuild-bm25",
    response_model=ApiResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Rebuild SQLite BM25 Full-Text Index",
    description="Reconstructs the SQLite FTS5 BM25 index from authoritative MongoDB chunks for active generations.",
)
async def rebuild_bm25(
    document_id: Optional[str] = Query(default=None, description="Optional document ID to rebuild. If omitted, rebuilds all documents."),
    service: IndexingService = Depends(get_indexing_service),
) -> ApiResponse[Dict[str, Any]]:
    """Rebuild SQLite BM25 index from authoritative MongoDB chunks."""
    rebuilt_count = await service.rebuild_bm25_index(document_id=document_id)
    return ApiResponse[Dict[str, Any]](
        message=f"BM25 index rebuilt successfully ({rebuilt_count} chunks indexed)",
        data={"document_id": document_id, "rebuilt_chunks": rebuilt_count},
    )


@router.post(
    "/{document_id}/index",
    response_model=ApiResponse[IndexingStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="Index Document Chunks into Vector Database",
    description="Generates Azure OpenAI embeddings for document chunks and indexes them with retrieval metadata in ChromaDB.",
)
async def index_document(
    document_id: str,
    service: IndexingService = Depends(get_indexing_service),
) -> ApiResponse[IndexingStatusResponse]:
    """Trigger synchronous embedding generation and ChromaDB vector indexing."""
    result = await service.index_document(document_id)
    return ApiResponse[IndexingStatusResponse](
        message="Document content successfully embedded and indexed",
        data=result,
    )


@router.get(
    "/{document_id}/index-status",
    response_model=ApiResponse[IndexingStatusResponse],
    summary="Get Document Indexing Status",
    description="Retrieves current embedding generation and ChromaDB vector indexing progress and status.",
)
async def get_indexing_status(
    document_id: str,
    service: IndexingService = Depends(get_indexing_service),
) -> ApiResponse[IndexingStatusResponse]:
    """Retrieve embedding and vector indexing status for a document."""
    status_res = await service.get_indexing_status(document_id)
    return ApiResponse[IndexingStatusResponse](
        message="Indexing status retrieved successfully",
        data=status_res,
    )

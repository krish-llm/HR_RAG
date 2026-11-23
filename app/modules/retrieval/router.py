"""FastAPI routes for Query Understanding & Semantic Retrieval operations (Phase 6)."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import (
    AsyncDatabase,
    BaseBM25IndexProvider,
    SchoolContext,
    Settings,
    get_bm25_provider,
    get_chroma_service,
    get_mongo_db,
    get_school_context,
    get_settings,
)
from app.db.chroma import ChromaDBService
from app.modules.indexing.provider import AzureOpenAIEmbeddingProvider
from app.modules.indexing.repository import IndexingRepository, VectorIndexRepository
from app.modules.ingestion.repository import DocumentRepository
from app.modules.retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.modules.retrieval.service import RetrievalService
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Query & Retrieval"])


def get_retrieval_service(
    db: AsyncDatabase = Depends(get_mongo_db),
    chroma: ChromaDBService = Depends(get_chroma_service),
    bm25: BaseBM25IndexProvider = Depends(get_bm25_provider),
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    """Dependency provider for RetrievalService."""
    document_repo = DocumentRepository(db, collection_name=settings.DOCUMENTS_COLLECTION_NAME)
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
    return RetrievalService(
        document_repo=document_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=embedding_provider,
        bm25_provider=bm25,
        retrieval_mode=settings.RETRIEVAL_MODE,
        hybrid_dense_top_k=settings.HYBRID_DENSE_TOP_K,
        hybrid_bm25_top_k=settings.HYBRID_BM25_TOP_K,
        hybrid_rrf_k=settings.HYBRID_RRF_K,
        candidate_pool_size=settings.DEFAULT_CANDIDATE_POOL_SIZE,
    )


@router.post(
    "/retrieve",
    response_model=ApiResponse[RetrievalResponse],
    status_code=status.HTTP_200_OK,
    summary="Semantic Policy Retrieval",
    description="Performs deterministic query understanding, query embedding, and scoped vector retrieval with authoritative MongoDB validation.",
)
async def retrieve_policies(
    request: RetrievalRequest,
    school_context: SchoolContext = Depends(get_school_context),
    service: RetrievalService = Depends(get_retrieval_service),
) -> ApiResponse[RetrievalResponse]:
    """Retrieve relevant HR policy chunks scoped to the caller's school context."""
    results = await service.retrieve(request, school_context)
    return ApiResponse(
        success=True,
        data=results,
    )

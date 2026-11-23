"""Document Embedding and Vector Indexing Module."""

from app.modules.indexing.provider import (
    AzureOpenAIEmbeddingProvider,
    BaseEmbeddingProvider,
)
from app.modules.indexing.repository import (
    IndexingRepository,
    VectorIndexRepository,
    serialize_chroma_metadata,
)
from app.modules.indexing.router import router as indexing_router
from app.modules.indexing.schemas import (
    ChunkEmbeddingRecord,
    DocumentIndexingRecord,
    IndexingStageStatus,
    IndexingStatusResponse,
)
from app.modules.indexing.service import IndexingService

__all__ = [
    "AzureOpenAIEmbeddingProvider",
    "BaseEmbeddingProvider",
    "ChunkEmbeddingRecord",
    "DocumentIndexingRecord",
    "IndexingRepository",
    "IndexingService",
    "IndexingStageStatus",
    "IndexingStatusResponse",
    "VectorIndexRepository",
    "indexing_router",
    "serialize_chroma_metadata",
]

"""Base BM25 index provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.modules.bm25.schemas import BM25ChunkRecord, BM25SearchResult


class BaseBM25IndexProvider(ABC):
    """Abstract interface defining contracts for BM25 full-text indexing and retrieval."""

    @abstractmethod
    def index_chunks(self, chunks: List[BM25ChunkRecord]) -> None:
        """Idempotently index chunk records with full-text content and metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BM25SearchResult]:
        """Execute BM25 keyword search returning top candidates ordered by relevance."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove all indexed chunks belonging to a document."""
        pass

    @abstractmethod
    def delete_generation(self, document_id: str, generation: int) -> int:
        """Remove all chunks for a specific document and generation (e.g. aborting failed run)."""
        pass

    @abstractmethod
    def delete_previous_generations(self, document_id: str, active_generation: int) -> int:
        """Purge chunks from older generations than active_generation for a document."""
        pass

    @abstractmethod
    def count_generation_chunks(self, document_id: str, generation: int) -> int:
        """Return the number of chunks indexed for a specific document and generation."""
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Return total number of chunks currently in the index."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform index self-check and return diagnostic information."""
        pass

    @abstractmethod
    def rebuild_index(self, chunks: List[BM25ChunkRecord]) -> None:
        """Completely rebuild the index from scratch with the supplied chunk records."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying connections or resources."""
        pass

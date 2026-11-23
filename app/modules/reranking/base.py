"""Abstract base class interface for policy chunk rerankers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.schemas import RetrievedChunk


class BaseReranker(ABC):
    """Abstract interface for policy chunk reranking strategies."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        keywords: Optional[List[str]] = None,
    ) -> List[RankedChunk]:
        """Rerank retrieved chunks by relevance to query and extracted keywords.

        Args:
            query: The normalized user query string.
            chunks: Candidate chunks retrieved from semantic search.
            keywords: Optional list of domain keywords and key terms.

        Returns:
            List of RankedChunk instances sorted in descending order of relevance.
        """
        pass

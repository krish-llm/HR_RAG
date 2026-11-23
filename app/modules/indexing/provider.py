"""Embedding provider abstraction and Azure OpenAI implementation."""

from abc import ABC, abstractmethod
import logging
import math
from typing import List, Optional
from openai import AsyncAzureOpenAI

from app.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class defining the contract for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of input texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical model name or deployment ID."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the expected vector dimensionality."""
        pass


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        dimensions: int = 1536,
        client: Optional[AsyncAzureOpenAI] = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.dimensions = dimensions

        if client is not None:
            self.client = client
        else:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=api_key or "dummy-key",
                api_version=self.api_version,
            )

    def get_model_name(self) -> str:
        """Return the Azure OpenAI embedding deployment name."""
        return self.deployment_name

    def get_dimensions(self) -> int:
        """Return the expected embedding dimensionality (1536 for text-embedding-3-small)."""
        return self.dimensions

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings from Azure OpenAI for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Sanitize whitespace-only strings to avoid API validation errors
        sanitized_texts = [t if t.strip() else " " for t in texts]

        try:
            response = await self.client.embeddings.create(
                input=sanitized_texts,
                model=self.deployment_name,
            )
        except Exception as exc:
            logger.error("Azure OpenAI embedding generation failed: %s", str(exc))
            raise EmbeddingException(
                message=f"Azure OpenAI embedding call failed: {exc}",
                error_code="AZURE_OPENAI_ERROR",
                status_code=502,
            ) from exc

        embeddings: List[List[float]] = [item.embedding for item in response.data]

        # Validate vector dimensions and numerical sanity
        for idx, emb in enumerate(embeddings):
            if len(emb) != self.dimensions:
                raise EmbeddingException(
                    message=f"Embedding dimension mismatch at index {idx}: expected {self.dimensions}, got {len(emb)}",
                    error_code="INVALID_EMBEDDING_DIMENSION",
                    status_code=500,
                )
            for val in emb:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise EmbeddingException(
                        message=f"Invalid non-numerical or non-finite embedding value detected at index {idx}",
                        error_code="INVALID_EMBEDDING_VALUES",
                        status_code=500,
                    )

        return embeddings

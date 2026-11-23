"""Application Configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application settings
    APP_NAME: str = "HR Policy RAG API"
    APP_ENV: str = Field(default="development", description="Environment: development, production, test")
    DEBUG: bool = Field(default=False, description="Debug mode")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # CORS configuration
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string or list for CORS origins."""
        if isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                # Handle JSON-like array string e.g. ["http://localhost:3000"]
                import json
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # MongoDB Configuration (PyMongo Async API)
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string",
    )
    MONGODB_DB_NAME: str = Field(
        default="hr_policy_rag",
        description="Default MongoDB database name",
    )
    MONGODB_MIN_POOL_SIZE: int = Field(default=10, description="Minimum pool size")
    MONGODB_MAX_POOL_SIZE: int = Field(default=50, description="Maximum pool size")
    MONGODB_TIMEOUT_MS: int = Field(
        default=5000,
        description="Server selection and connection timeout in milliseconds",
    )

    # ChromaDB Configuration (Persistent Local Storage)
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/chroma",
        description="Path for local persistent ChromaDB storage",
    )

    # Document Upload & Ingestion Storage Settings
    UPLOAD_DIR: str = Field(
        default="./data/uploads",
        description="Directory for persistent storage of uploaded PDF files",
    )
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=25 * 1024 * 1024,
        description="Maximum allowed upload file size in bytes (default 25 MB)",
    )
    DOCUMENTS_COLLECTION_NAME: str = Field(
        default="documents",
        description="MongoDB collection name for document records",
    )
    EXTRACTIONS_COLLECTION_NAME: str = Field(
        default="document_extractions",
        description="MongoDB collection name for extracted document text",
    )
    PROCESSED_COLLECTION_NAME: str = Field(
        default="document_processed_content",
        description="MongoDB collection name for processed and normalized document content",
    )
    CHUNKS_COLLECTION_NAME: str = Field(
        default="document_chunks",
        description="MongoDB collection name for document text chunks",
    )
    CHUNK_TARGET_TOKENS: int = Field(
        default=500,
        description="Target chunk token length for splitting",
    )
    CHUNK_MAX_TOKENS: int = Field(
        default=700,
        description="Maximum allowed chunk token length",
    )
    CHUNK_OVERLAP_TOKENS: int = Field(
        default=50,
        description="Overlap token length between adjacent chunks of the same section",
    )
    TOKENIZER_ENCODING: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name for token counting",
    )

    # Azure OpenAI & Vector Indexing Configuration
    AZURE_OPENAI_ENDPOINT: str = Field(
        default="https://example.openai.azure.com/",
        description="Azure OpenAI endpoint URL",
    )
    AZURE_OPENAI_API_KEY: str = Field(
        default="",
        description="Azure OpenAI API key",
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version",
    )
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(
        default="gpt-4o",
        description="Deployment name for chat completions model",
    )
    AZURE_OPENAI_CHAT_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for deterministic grounded answer generation",
    )
    AZURE_OPENAI_CHAT_MAX_TOKENS: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum tokens for generated answer response",
    )
    AZURE_OPENAI_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Request timeout in seconds for Azure OpenAI chat completions",
    )
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(
        default="text-embedding-3-small",
        description="Deployment name for embeddings model",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="text-embedding-3-small",
        description="Canonical model name for embedding generation",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=1536,
        description="Expected vector embedding dimensionality",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=16,
        description="Batch size for Azure OpenAI embedding generation",
    )
    CHROMA_CHUNKS_COLLECTION_NAME: str = Field(
        default="hr_policy_chunks",
        description="Dedicated ChromaDB collection name for chunk vector embeddings",
    )
    INDEXING_COLLECTION_NAME: str = Field(
        default="document_chunk_embeddings",
        description="MongoDB collection name for chunk embedding records",
    )
    INDEXING_TRACKING_COLLECTION_NAME: str = Field(
        default="document_indexing_tracking",
        description="MongoDB collection name for document indexing progress and status",
    )

    # Hybrid Retrieval & BM25 Configuration
    RETRIEVAL_MODE: str = Field(
        default="dense",
        description="Retrieval mode: 'dense' or 'hybrid'",
    )
    BM25_INDEX_PATH: str = Field(
        default="./data/bm25/fts5_index.db",
        description="File path for persistent SQLite FTS5 BM25 index",
    )
    HYBRID_DENSE_TOP_K: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Candidate pool size for dense retrieval in hybrid mode",
    )
    HYBRID_BM25_TOP_K: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Candidate pool size for BM25 retrieval in hybrid mode",
    )
    HYBRID_RRF_K: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Constant k for Reciprocal Rank Fusion formula: 1 / (k + rank)",
    )

    # Retrieval Configuration
    DEFAULT_TOP_K: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of top chunks returned from retrieval",
    )
    DEFAULT_CANDIDATE_POOL_SIZE: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Default number of candidate vectors retrieved from ChromaDB prior to validation",
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for returned chunks",
    )

    # Reranking & Context Assembly Configuration (Phase 7)
    RERANK_TOP_K: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of candidate chunks retrieved for reranking",
    )
    CONTEXT_TOKEN_BUDGET: int = Field(
        default=2000,
        ge=100,
        le=16000,
        description="Default token budget for context window assembly",
    )


@lru_cache()
def get_settings() -> Settings:
    """Dependency provider for cached application settings."""
    return Settings()

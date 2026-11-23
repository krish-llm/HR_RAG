"""Tests for Pydantic Settings and environment configuration."""

import pytest
from app.config import Settings, get_settings


def test_settings_loaded_correctly():
    """Verify settings defaults and environment mapping."""
    settings = get_settings()

    assert settings.APP_NAME == "HR Policy RAG API"
    assert settings.APP_ENV in ["development", "production", "test"]
    assert isinstance(settings.DEBUG, bool)
    assert settings.MONGODB_URL.startswith("mongodb://")
    assert settings.MONGODB_DB_NAME != ""
    assert settings.CHROMA_PERSIST_DIRECTORY != ""
    assert settings.AZURE_OPENAI_ENDPOINT.startswith("https://")
    assert settings.AZURE_OPENAI_API_VERSION != ""
    assert settings.AZURE_OPENAI_CHAT_DEPLOYMENT in ["gpt-4o", "gpt-4.1-mini"]
    assert settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT == "text-embedding-3-small"
    assert settings.CHROMA_CHUNKS_COLLECTION_NAME == "hr_policy_chunks"
    assert settings.INDEXING_COLLECTION_NAME == "document_chunk_embeddings"
    assert settings.INDEXING_TRACKING_COLLECTION_NAME == "document_indexing_tracking"
    assert settings.EMBEDDING_BATCH_SIZE == 16
    assert settings.EMBEDDING_DIMENSIONS == 1536
    assert settings.DEFAULT_TOP_K == 5
    assert settings.DEFAULT_CANDIDATE_POOL_SIZE == 20
    assert settings.SIMILARITY_THRESHOLD == 0.0


def test_cors_origins_parsing():
    """Verify CORS origins field validator parsing."""
    # List format
    s1 = Settings(CORS_ORIGINS=["http://localhost:3000", "http://example.com"])
    assert s1.CORS_ORIGINS == ["http://localhost:3000", "http://example.com"]

    # Comma-separated string format
    s2 = Settings(CORS_ORIGINS="http://localhost:3000,http://example.com")
    assert s2.CORS_ORIGINS == ["http://localhost:3000", "http://example.com"]

    # Wildcard string format
    s3 = Settings(CORS_ORIGINS="*")
    assert s3.CORS_ORIGINS == ["*"]


def test_settings_singleton_caching():
    """Verify get_settings returns cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

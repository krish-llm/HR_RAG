"""Tests for MongoDB and ChromaDB database connection managers and dependencies."""

import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.exceptions import DatabaseConnectionException
from app.db.chroma import ChromaDBService, chroma_service, get_chroma_client, get_chroma_service
from app.db.mongo import MongoDatabaseManager, get_mongo_client, get_mongo_db, mongo_manager


# ---------------------------------------------------------
# ChromaDB Tests
# ---------------------------------------------------------

def test_chroma_service_lifecycle():
    """Verify ChromaDBService initializes persistent storage, creates collections, and checks health."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        service = ChromaDBService()
        assert not service.is_initialized()

        service.initialize(temp_dir)
        assert service.is_initialized()
        assert service.is_healthy()
        assert service.heartbeat() > 0

        # Collection operations
        collection = service.get_or_create_collection("test_hr_policies", metadata={"tenant_id": "test_tenant"})
        assert collection is not None
        assert collection.name == "test_hr_policies"

        collections = service.list_collections()
        assert any(c.name == "test_hr_policies" for c in collections)

        service.close()
        assert not service.is_initialized()


def test_chroma_dependency_injection():
    """Verify ChromaDB dependency injection providers."""
    # Active initialized service
    svc = get_chroma_service()
    assert isinstance(svc, ChromaDBService)

    client = get_chroma_client()
    assert client is not None


def test_chroma_uninitialized_exception():
    """Verify exception is raised when Chroma service is uninitialized."""
    uninit_service = ChromaDBService()
    with pytest.raises(DatabaseConnectionException) as exc_info:
        _ = uninit_service.client
    assert exc_info.value.error_code == "CHROMADB_NOT_INITIALIZED"


# ---------------------------------------------------------
# MongoDB Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_mongo_manager_lifecycle_mocked():
    """Verify MongoDatabaseManager async lifecycle methods."""
    manager = MongoDatabaseManager()
    assert not manager.is_connected()
    assert manager.client is None
    assert manager.db is None

    # Connect with mock
    with patch("app.db.mongo.AsyncMongoClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.admin.command = AsyncMock(return_value={"ok": 1})
        mock_instance.close = AsyncMock()
        mock_instance.__getitem__.return_value = MagicMock()
        mock_client_cls.return_value = mock_instance

        await manager.connect(
            uri="mongodb://localhost:27017",
            db_name="test_hr_rag",
            min_pool_size=5,
            max_pool_size=10,
            timeout_ms=1000,
        )

        assert manager.is_connected()
        assert manager.client is not None
        assert manager.db is not None

        # Ping
        ping_result = await manager.ping()
        assert ping_result is True

        # Close
        await manager.close()
        assert not manager.is_connected()
        assert manager.client is None
        assert manager.db is None
        mock_instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_mongo_dependencies_raise_when_uninitialized():
    """Verify get_mongo_db and get_mongo_client raise exception if not connected."""
    original_client = mongo_manager._client
    original_db = mongo_manager._db
    try:
        mongo_manager._client = None
        mongo_manager._db = None

        with pytest.raises(DatabaseConnectionException) as exc_db:
            await get_mongo_db()
        assert exc_db.value.error_code == "MONGODB_NOT_INITIALIZED"

        with pytest.raises(DatabaseConnectionException) as exc_client:
            await get_mongo_client()
        assert exc_client.value.error_code == "MONGODB_NOT_INITIALIZED"
    finally:
        mongo_manager._client = original_client
        mongo_manager._db = original_db

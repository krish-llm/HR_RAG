"""MongoDB connection manager using the official PyMongo Async API."""

import logging
from typing import Optional
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ConnectionFailure, PyMongoError

from app.core.exceptions import DatabaseConnectionException

logger = logging.getLogger(__name__)


class MongoDatabaseManager:
    """Manages asynchronous MongoDB client lifecycle and connection pool."""

    def __init__(self) -> None:
        self._client: Optional[AsyncMongoClient] = None
        self._db: Optional[AsyncDatabase] = None
        self._db_name: Optional[str] = None

    @property
    def client(self) -> Optional[AsyncMongoClient]:
        """Return the active AsyncMongoClient instance."""
        return self._client

    @property
    def db(self) -> Optional[AsyncDatabase]:
        """Return the active AsyncDatabase instance."""
        return self._db

    def is_connected(self) -> bool:
        """Check if the client instance is initialized."""
        return self._client is not None

    async def connect(
        self,
        uri: str,
        db_name: str,
        min_pool_size: int = 10,
        max_pool_size: int = 50,
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize MongoDB AsyncMongoClient and configure default database.

        Args:
            uri: MongoDB connection string
            db_name: Target database name
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            timeout_ms: Server selection timeout in milliseconds
        """
        if self._client is not None:
            logger.warning("MongoDB client is already connected.")
            return

        logger.info("Initializing MongoDB Async client for database: %s", db_name)
        try:
            self._client = AsyncMongoClient(
                uri,
                minPoolSize=min_pool_size,
                maxPoolSize=max_pool_size,
                serverSelectionTimeoutMS=timeout_ms,
            )
            self._db_name = db_name
            self._db = self._client[db_name]

            # Attempt a quick ping to verify connectivity on startup
            await self.ping()
            logger.info("Successfully connected to MongoDB (%s)", db_name)
        except (ConnectionFailure, PyMongoError) as exc:
            logger.warning(
                "MongoDB connection established with warning (server may be offline): %s",
                str(exc),
            )
            # We don't crash the entire app if MongoDB is temporarily down at startup;
            # the health check will report degraded status.

    async def ping(self) -> bool:
        """Verify live MongoDB connection using the admin 'ping' command."""
        if self._client is None:
            return False
        try:
            await self._client.admin.command("ping")
            return True
        except Exception as exc:
            logger.warning("MongoDB ping failed: %s", str(exc))
            return False

    async def close(self) -> None:
        """Gracefully close the MongoDB client and release connection pool resources."""
        if self._client is not None:
            logger.info("Closing MongoDB Async client connection.")
            try:
                await self._client.close()
            except Exception as exc:
                logger.error("Error closing MongoDB connection: %s", str(exc))
            finally:
                self._client = None
                self._db = None
                self._db_name = None
            logger.info("MongoDB client closed successfully.")


# Singleton instance for application lifecycle management
mongo_manager = MongoDatabaseManager()


async def get_mongo_db() -> AsyncDatabase:
    """FastAPI dependency for injecting the default AsyncDatabase."""
    if mongo_manager.db is None:
        raise DatabaseConnectionException(
            message="MongoDB database is not connected or initialized",
            error_code="MONGODB_NOT_INITIALIZED",
        )
    return mongo_manager.db


async def get_mongo_client() -> AsyncMongoClient:
    """FastAPI dependency for injecting the AsyncMongoClient."""
    if mongo_manager.client is None:
        raise DatabaseConnectionException(
            message="MongoDB client is not connected or initialized",
            error_code="MONGODB_NOT_INITIALIZED",
        )
    return mongo_manager.client

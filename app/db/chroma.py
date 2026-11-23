"""ChromaDB connection and vector database service using persistent local storage."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.core.exceptions import DatabaseConnectionException

logger = logging.getLogger(__name__)


class ChromaDBService:
    """Service wrapping ChromaDB persistent client operations for vector storage."""

    def __init__(self) -> None:
        self._client: Optional[ClientAPI] = None
        self._persist_directory: Optional[str] = None

    @property
    def client(self) -> ClientAPI:
        """Return the initialized ChromaDB client instance."""
        if self._client is None:
            raise DatabaseConnectionException(
                message="ChromaDB client has not been initialized",
                error_code="CHROMADB_NOT_INITIALIZED",
            )
        return self._client

    def is_initialized(self) -> bool:
        """Check if ChromaDB client has been initialized."""
        return self._client is not None

    def initialize(self, persist_directory: str) -> None:
        """Initialize ChromaDB PersistentClient with a local storage directory.

        Args:
            persist_directory: Local filesystem directory for persistent database storage
        """
        if self._client is not None:
            logger.info("ChromaDB PersistentClient already initialized.")
            return

        self._persist_directory = persist_directory
        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ChromaDB PersistentClient at directory: %s", path.resolve())
        try:
            self._client = chromadb.PersistentClient(path=str(path.resolve()))
            logger.info("ChromaDB PersistentClient initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB PersistentClient: %s", str(exc))
            raise DatabaseConnectionException(
                message=f"Failed to initialize ChromaDB: {exc}",
                error_code="CHROMADB_INIT_FAILED",
            ) from exc

    def heartbeat(self) -> int:
        """Perform a heartbeat check against ChromaDB.

        Returns:
            Timestamp in nanoseconds indicating client status.
        """
        return self.client.heartbeat()

    def is_healthy(self) -> bool:
        """Check if ChromaDB service is responsive."""
        if self._client is None:
            return False
        try:
            hb = self.client.heartbeat()
            return hb is not None and hb > 0
        except Exception as exc:
            logger.warning("ChromaDB health check failed: %s", str(exc))
            return False

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Retrieve or create a ChromaDB vector collection.

        Args:
            name: Collection name
            metadata: Optional collection metadata configuration

        Returns:
            Collection instance
        """
        return self.client.get_or_create_collection(name=name, metadata=metadata)

    def list_collections(self) -> List[Collection]:
        """List all existing vector collections."""
        return self.client.list_collections()

    def close(self) -> None:
        """Reset Chroma client reference and release SQLite handles on shutdown."""
        logger.info("Resetting ChromaDB client reference.")
        if self._client is not None:
            try:
                if hasattr(self._client, "_system") and hasattr(self._client._system, "stop"):
                    self._client._system.stop()
            except Exception as exc:
                logger.warning("Error stopping ChromaDB system: %s", str(exc))
        self._client = None
        self._persist_directory = None


# Singleton instance for application lifecycle management
chroma_service = ChromaDBService()


def get_chroma_service() -> ChromaDBService:
    """FastAPI dependency for injecting the ChromaDB service."""
    if not chroma_service.is_initialized():
        raise DatabaseConnectionException(
            message="ChromaDB service is not initialized",
            error_code="CHROMADB_NOT_INITIALIZED",
        )
    return chroma_service


def get_chroma_client() -> ClientAPI:
    """FastAPI dependency for injecting raw ChromaDB ClientAPI."""
    return get_chroma_service().client

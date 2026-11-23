"""MongoDB repository for storing and querying processed, cleaned document content."""

import logging
from typing import Any, Dict, Optional
from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.modules.processing.schemas import DocumentProcessedResult

logger = logging.getLogger(__name__)


class ProcessedRepository:
    """Manages persistence of cleaned, normalized document content in MongoDB."""

    def __init__(self, db: AsyncDatabase, collection_name: str = "document_processed_content") -> None:
        self.collection: AsyncCollection = db[collection_name]

    async def ensure_indexes(self) -> None:
        """Create unique index on document_id and auxiliary indexes."""
        indexes = [
            IndexModel([("document_id", ASCENDING)], unique=True, name="idx_processed_doc_id_unique"),
            IndexModel([("processed_at", ASCENDING)], name="idx_processed_at"),
            IndexModel([("source_extraction_id", ASCENDING)], name="idx_source_extraction_id"),
        ]
        try:
            await self.collection.create_indexes(indexes)
            logger.info("ProcessedRepository indexes verified.")
        except Exception as exc:
            logger.warning("Could not create ProcessedRepository indexes: %s", str(exc))

    async def save_processed(self, processed_data: DocumentProcessedResult) -> Dict[str, Any]:
        """Save processed result in MongoDB, replacing previous results upon reprocessing.

        Args:
            processed_data: Processed document content model

        Returns:
            Dictionary record stored in MongoDB
        """
        record = processed_data.model_dump(mode="python")
        await self.collection.replace_one(
            {"document_id": processed_data.document_id},
            record,
            upsert=True,
        )
        logger.info("Saved processed content for document_id=%s in MongoDB", processed_data.document_id)
        return record

    async def get_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processed document content for a given document_id."""
        return await self.collection.find_one({"document_id": document_id})

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete processed content record for a document_id."""
        res = await self.collection.delete_one({"document_id": document_id})
        return res.deleted_count > 0

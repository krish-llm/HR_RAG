"""MongoDB repository for storing and querying extracted document text content."""

import logging
from typing import Any, Dict, Optional
from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.modules.extraction.schemas import DocumentExtractionResult

logger = logging.getLogger(__name__)


class ExtractionRepository:
    """Manages persistence of page-level extracted Markdown content in MongoDB."""

    def __init__(self, db: AsyncDatabase, collection_name: str = "document_extractions") -> None:
        self.collection: AsyncCollection = db[collection_name]

    async def ensure_indexes(self) -> None:
        """Create unique index on document_id to guarantee one extraction per document."""
        indexes = [
            IndexModel([("document_id", ASCENDING)], unique=True, name="idx_extraction_doc_id_unique"),
            IndexModel([("extracted_at", ASCENDING)], name="idx_extracted_at"),
        ]
        try:
            await self.collection.create_indexes(indexes)
            logger.info("ExtractionRepository indexes verified.")
        except Exception as exc:
            logger.warning("Could not create ExtractionRepository indexes: %s", str(exc))

    async def save_extraction(self, extraction: DocumentExtractionResult) -> Dict[str, Any]:
        """Save extraction result in MongoDB, replacing previous extractions on re-extraction.

        Args:
            extraction: Extraction result data model

        Returns:
            Dictionary record stored in MongoDB
        """
        record = extraction.model_dump(mode="python")
        await self.collection.replace_one(
            {"document_id": extraction.document_id},
            record,
            upsert=True,
        )
        logger.info("Saved extraction for document_id=%s in MongoDB", extraction.document_id)
        return record

    async def get_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the extraction result for a given document_id."""
        return await self.collection.find_one({"document_id": document_id})

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete extraction record for a document_id."""
        res = await self.collection.delete_one({"document_id": document_id})
        return res.deleted_count > 0

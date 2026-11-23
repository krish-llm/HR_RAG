"""MongoDB repository layer for Document Ingestion operations."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.modules.ingestion.schemas import DocumentCreate, DocumentFilterParams

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Encapsulates all MongoDB data-access operations for ingested documents."""

    def __init__(self, db: AsyncDatabase, collection_name: str = "documents") -> None:
        self.collection: AsyncCollection = db[collection_name]

    async def ensure_indexes(self) -> None:
        """Create necessary database indexes for fast query resolution and unique integrity."""
        indexes = [
            IndexModel([("document_id", ASCENDING)], unique=True, name="idx_document_id_unique"),
            IndexModel(
                [
                    ("school_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("branch_id", ASCENDING),
                    ("is_global", ASCENDING),
                    ("status", ASCENDING),
                ],
                name="idx_school_scope_status",
            ),
            IndexModel([("file_hash", ASCENDING)], name="idx_file_hash"),
            IndexModel([("uploaded_at", DESCENDING)], name="idx_uploaded_at_desc"),
        ]
        try:
            await self.collection.create_indexes(indexes)
            logger.info("DocumentRepository indexes verified.")
        except Exception as exc:
            logger.warning("Could not create DocumentRepository indexes (server may be offline): %s", str(exc))

    async def find_duplicate(
        self,
        file_hash: str,
        school_id: Optional[str],
        group_id: Optional[str],
        branch_id: Optional[str],
        is_global: bool,
        state: Optional[str],
        version: int,
    ) -> Optional[Dict[str, Any]]:
        """Check if an identical document with matching file hash, scope, and version already exists."""
        query = {
            "file_hash": file_hash,
            "school_id": school_id,
            "group_id": group_id,
            "branch_id": branch_id,
            "is_global": is_global,
            "state": state,
            "version": version,
            "status": {"$ne": "failed"},
        }
        return await self.collection.find_one(query)

    async def insert_document(self, doc_data: DocumentCreate) -> Dict[str, Any]:
        """Insert a newly uploaded document record into MongoDB."""
        record = doc_data.model_dump(mode="python")
        await self.collection.insert_one(record)
        logger.info("Registered document in MongoDB: document_id=%s", doc_data.document_id)
        return record

    async def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its unique document_id."""
        return await self.collection.find_one({"document_id": document_id})

    async def list_documents(self, filters: DocumentFilterParams) -> Tuple[List[Dict[str, Any]], int]:
        """Query documents with filtering and pagination.

        Returns:
            Tuple of (list of document records, total count)
        """
        query: Dict[str, Any] = {}

        if filters.school_id is not None:
            query["school_id"] = filters.school_id
        if filters.group_id is not None:
            query["group_id"] = filters.group_id
        if filters.branch_id is not None:
            query["branch_id"] = filters.branch_id
        if filters.is_global is not None:
            query["is_global"] = filters.is_global
        if filters.state is not None:
            query["state"] = filters.state
        if filters.document_type is not None:
            query["document_type"] = filters.document_type
        if filters.status is not None:
            query["status"] = filters.status.value
        if filters.processing_status is not None:
            query["processing_status"] = filters.processing_status.value

        total = await self.collection.count_documents(query)

        skip = (filters.page - 1) * filters.page_size
        cursor = (
            self.collection.find(query)
            .sort("uploaded_at", DESCENDING)
            .skip(skip)
            .limit(filters.page_size)
        )

        records = await cursor.to_list(length=filters.page_size)
        return records, total

    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document record by its unique document_id.

        Returns:
            True if a record was deleted, False otherwise.
        """
        result = await self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0

    async def update_processing_status(
        self,
        document_id: str,
        processing_status: str,
        error_message: Optional[str] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update the processing status and optional error message of an ingested document."""
        update_doc: Dict[str, Any] = {
            "processing_status": processing_status,
            "error_message": error_message,
        }
        if status is not None:
            update_doc["status"] = status

        result = await self.collection.update_one(
            {"document_id": document_id},
            {"$set": update_doc},
        )
        return result.matched_count > 0

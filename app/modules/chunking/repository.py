"""MongoDB repository for persisting and querying document chunks."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ConfigurationError, InvalidOperation, OperationFailure, PyMongoError

from app.modules.chunking.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Manages persistence and retrieval of document chunks in MongoDB."""

    def __init__(self, db: AsyncDatabase, collection_name: str = "document_chunks") -> None:
        self.db: AsyncDatabase = db
        self.collection: AsyncCollection = db[collection_name]
        self.staging_collection: AsyncCollection = db[f"{collection_name}_staging"]
        self._supports_transactions: Optional[bool] = None

    async def ensure_indexes(self) -> None:
        """Create indexes on chunk_id, document_id, chunk_index, and scoping dimensions."""
        indexes = [
            IndexModel([("chunk_id", ASCENDING)], unique=True, name="idx_chunk_id_unique"),
            IndexModel([("document_id", ASCENDING)], name="idx_chunks_doc_id"),
            IndexModel(
                [("document_id", ASCENDING), ("chunk_index", ASCENDING)],
                name="idx_chunks_doc_id_index",
            ),
            IndexModel([("school_id", ASCENDING)], name="idx_chunks_school_id"),
            IndexModel([("group_id", ASCENDING)], name="idx_chunks_group_id"),
            IndexModel([("is_global", ASCENDING)], name="idx_chunks_is_global"),
            IndexModel([("state", ASCENDING)], name="idx_chunks_state"),
        ]
        try:
            await self.collection.create_indexes(indexes)
            logger.info("ChunkRepository indexes verified.")
        except Exception as exc:
            logger.warning("Could not create ChunkRepository indexes: %s", str(exc))

        staging_indexes = [
            IndexModel([("document_id", ASCENDING)], name="idx_staging_chunks_doc_id"),
            IndexModel([("chunk_id", ASCENDING)], name="idx_staging_chunk_id"),
        ]
        try:
            await self.staging_collection.create_indexes(staging_indexes)
            logger.info("ChunkRepository staging indexes verified.")
        except Exception as exc:
            logger.warning("Could not create ChunkRepository staging indexes: %s", str(exc))

    async def save_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> int:
        """Safely replace chunks for a document.

        Preserves existing valid chunks until replacement is fully successful.
        Uses a MongoDB transaction if the deployment supports it; otherwise uses
        a staging-collection strategy with automated rollback on failure to prevent
        partially written chunks from becoming active.

        Args:
            document_id: Unique document UUID
            chunks: List of DocumentChunk models

        Returns:
            Number of chunks saved
        """
        # Step 1: Pre-generate and serialize all records before modifying any storage
        records = [c.model_dump(mode="python") for c in chunks] if chunks else []

        # Step 2: Attempt transactional replacement if supported by deployment
        if self._supports_transactions is not False:
            client = getattr(self.db, "client", None)
            if client is not None and hasattr(client, "start_session"):
                session_ctx = None
                try:
                    session_ctx = client.start_session()
                except (OperationFailure, ConfigurationError, InvalidOperation, NotImplementedError, AttributeError) as exc:
                    logger.info("MongoDB sessions not supported (%s). Using staging replacement.", exc)
                    self._supports_transactions = False

                if session_ctx is not None:
                    try:
                        async with session_ctx as session:
                            async def _txn_callback(s):
                                await self.collection.delete_many({"document_id": document_id}, session=s)
                                if records:
                                    await self.collection.insert_many(records, session=s)
                                return len(records)

                            inserted_count = await session.with_transaction(_txn_callback)
                            self._supports_transactions = True
                            logger.info(
                                "Safely replaced %d chunks for document_id=%s via MongoDB transaction",
                                inserted_count,
                                document_id,
                            )
                            return inserted_count
                    except (OperationFailure, ConfigurationError, InvalidOperation) as exc:
                        err_msg = str(exc).lower()
                        is_unsupported = (
                            getattr(exc, "code", None) == 20
                            or "replica set" in err_msg
                            or "transaction numbers are only allowed" in err_msg
                            or "sessions are not supported" in err_msg
                            or "does not support transactions" in err_msg
                        )
                        if is_unsupported:
                            logger.info(
                                "MongoDB deployment does not support transactions (%s). Falling back to staging replacement.",
                                exc,
                            )
                            self._supports_transactions = False
                        else:
                            logger.error(
                                "Transaction aborted and rolled back for document_id=%s: %s",
                                document_id,
                                exc,
                            )
                            raise

        # Step 3: Staging-based replacement fallback
        # 3a. Clean up any stale staging chunks for this document
        await self.staging_collection.delete_many({"document_id": document_id})

        # 3b. Stage new chunks first (if staging insert fails, active collection is untouched)
        if records:
            staging_records = [dict(r) for r in records]
            await self.staging_collection.insert_many(staging_records)

        # 3c. Snapshot existing active valid chunks to preserve them in case of active write failure
        existing_chunks = await self.get_all_by_document_id(document_id)

        # 3d. Replace active chunks
        try:
            await self.collection.delete_many({"document_id": document_id})
            if records:
                active_records = []
                for r in records:
                    rec_copy = dict(r)
                    rec_copy.pop("_id", None)
                    active_records.append(rec_copy)
                await self.collection.insert_many(active_records)
        except Exception as active_exc:
            logger.error(
                "Active write failed for document_id=%s. Restoring %d previous valid chunks: %s",
                document_id,
                len(existing_chunks),
                active_exc,
            )
            # Rollback: Clean partial active records and restore original valid chunks
            try:
                await self.collection.delete_many({"document_id": document_id})
                if existing_chunks:
                    clean_backups = []
                    for item in existing_chunks:
                        item_copy = dict(item)
                        item_copy.pop("_id", None)
                        clean_backups.append(item_copy)
                    await self.collection.insert_many(clean_backups)
                    logger.info(
                        "Successfully restored %d previous valid chunks for document_id=%s",
                        len(clean_backups),
                        document_id,
                    )
            except Exception as rollback_exc:
                logger.critical(
                    "Critical failure during rollback restoration for document_id=%s: %s",
                    document_id,
                    rollback_exc,
                )
            raise active_exc
        finally:
            # 3e. Purge staging collection
            try:
                await self.staging_collection.delete_many({"document_id": document_id})
            except Exception as clean_exc:
                logger.warning(
                    "Failed to purge staging collection for document_id=%s: %s",
                    document_id,
                    clean_exc,
                )

        logger.info(
            "Saved %d chunks for document_id=%s via staging replacement in MongoDB",
            len(records),
            document_id,
        )
        return len(records)

    async def get_chunks_paginated(
        self,
        document_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve paginated chunks for a given document_id ordered by chunk_index."""
        query = {"document_id": document_id}
        total = await self.collection.count_documents(query)

        skip = (page - 1) * page_size
        cursor = self.collection.find(query).sort("chunk_index", ASCENDING).skip(skip).limit(page_size)
        items = await cursor.to_list(length=page_size)
        return items, total

    async def get_all_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document ordered by chunk_index."""
        cursor = self.collection.find({"document_id": document_id}).sort("chunk_index", ASCENDING)
        return await cursor.to_list(length=None)

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a document from active and staging collections."""
        try:
            await self.staging_collection.delete_many({"document_id": document_id})
        except Exception as exc:
            logger.warning("Could not purge staging chunks for document_id=%s: %s", document_id, exc)
        res = await self.collection.delete_many({"document_id": document_id})
        return res.deleted_count

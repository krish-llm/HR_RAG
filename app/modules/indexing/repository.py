"""Repository layer for MongoDB embedding tracking and ChromaDB vector persistence."""

import logging
from typing import Any, Dict, List, Optional, Union
from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.db.chroma import ChromaDBService
from app.modules.chunking.schemas import DocumentChunk
from app.modules.indexing.schemas import ChunkEmbeddingRecord, DocumentIndexingRecord

logger = logging.getLogger(__name__)


def _format_iso_date(value: Any) -> str:
    """Convert datetime or date-like value to consistent ISO 8601 string; return empty string if missing."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip()
    return str(value)


def serialize_chroma_metadata(
    chunk: DocumentChunk,
    embedding_model: str,
    generation: int = 1,
) -> Dict[str, Union[str, int, float, bool]]:
    """Convert chunk model attributes to strict ChromaDB-compatible scalar metadata.

    ChromaDB requires all metadata values to be str, int, float, or bool. None and
    nested dictionaries/lists are converted or omitted.
    """
    return {
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.chunk_id),
        "chunk_index": int(chunk.chunk_index),
        "generation": int(generation),
        "school_id": chunk.school_id or "",
        "group_id": chunk.group_id or "",
        "branch_id": chunk.branch_id or "",
        "is_global": bool(chunk.is_global),
        "state": chunk.state or "",
        "staff_category": chunk.staff_category or "",
        "employment_type": chunk.employment_type or "",
        "department": chunk.department or "",
        "document_name": str(chunk.document_name),
        "document_type": str(chunk.document_type),
        "version": int(chunk.version),
        "effective_from": _format_iso_date(chunk.effective_from),
        "effective_to": _format_iso_date(chunk.effective_to),
        "page_start": int(chunk.page_start),
        "page_end": int(chunk.page_end),
        "heading": chunk.heading or "",
        "embedding_model": str(embedding_model),
    }


class IndexingRepository:
    """Manages MongoDB storage for chunk embeddings and document indexing progress."""

    def __init__(
        self,
        db: AsyncDatabase,
        embeddings_collection_name: str = "document_chunk_embeddings",
        tracking_collection_name: str = "document_indexing_tracking",
    ) -> None:
        self.embeddings_collection: AsyncCollection = db[embeddings_collection_name]
        self.tracking_collection: AsyncCollection = db[tracking_collection_name]

    async def ensure_indexes(self) -> None:
        """Create indexes on chunk_id, document_id, content_hash, and indexing_status."""
        embedding_indexes = [
            IndexModel([("chunk_id", ASCENDING)], unique=True, name="idx_chunk_emb_id_unique"),
            IndexModel([("document_id", ASCENDING)], name="idx_chunk_emb_doc_id"),
            IndexModel(
                [
                    ("chunk_id", ASCENDING),
                    ("content_hash", ASCENDING),
                    ("embedding_model", ASCENDING),
                ],
                name="idx_chunk_hash_lookup",
            ),
        ]
        tracking_indexes = [
            IndexModel([("document_id", ASCENDING)], unique=True, name="idx_tracking_doc_id_unique"),
            IndexModel([("indexing_status", ASCENDING)], name="idx_tracking_status"),
        ]

        try:
            await self.embeddings_collection.create_indexes(embedding_indexes)
            await self.tracking_collection.create_indexes(tracking_indexes)
            logger.info("IndexingRepository MongoDB indexes verified.")
        except Exception as exc:
            logger.warning("Could not create IndexingRepository indexes: %s", str(exc))

    async def get_chunk_embedding(
        self,
        chunk_id: str,
        content_hash: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Find an existing embedding by chunk_id, content hash, and model name for caching."""
        return await self.embeddings_collection.find_one(
            {
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "embedding_model": model_name,
                "embedding_status": "completed",
            }
        )

    async def get_all_chunk_embeddings_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all cached chunk embeddings for a given document UUID."""
        cursor = self.embeddings_collection.find({"document_id": document_id})
        return await cursor.to_list(length=None)

    async def save_chunk_embedding(self, record: ChunkEmbeddingRecord) -> None:
        """Upsert a single chunk embedding record in MongoDB."""
        data = record.model_dump(mode="python")
        await self.embeddings_collection.replace_one(
            {"chunk_id": record.chunk_id},
            data,
            upsert=True,
        )

    async def save_chunk_embeddings_batch(self, records: List[ChunkEmbeddingRecord]) -> None:
        """Upsert multiple chunk embedding records into MongoDB."""
        if not records:
            return
        for rec in records:
            await self.save_chunk_embedding(rec)

    async def delete_chunk_embeddings_by_document_id(self, document_id: str) -> int:
        """Remove all chunk embedding records for a document."""
        res = await self.embeddings_collection.delete_many({"document_id": document_id})
        return res.deleted_count

    async def get_tracking_record(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the document-level indexing tracking record."""
        return await self.tracking_collection.find_one({"document_id": document_id})

    async def get_all_active_tracking_records(self) -> List[Dict[str, Any]]:
        """Fetch all tracking records for documents with an active generation."""
        cursor = self.tracking_collection.find(
            {"active_generation": {"$gt": 0}}
        )
        return await cursor.to_list(length=1000)

    async def save_tracking_record(self, record: DocumentIndexingRecord) -> None:
        """Upsert document-level indexing status and progress."""
        data = record.model_dump(mode="python")
        await self.tracking_collection.replace_one(
            {"document_id": record.document_id},
            data,
            upsert=True,
        )

    async def delete_tracking_by_document_id(self, document_id: str) -> bool:
        """Delete indexing tracking record for a document."""
        res = await self.tracking_collection.delete_one({"document_id": document_id})
        return res.deleted_count > 0


class VectorIndexRepository:
    """Manages vector operations, metadata storage, and stale vector pruning in ChromaDB."""

    def __init__(
        self,
        chroma_service: ChromaDBService,
        collection_name: str = "hr_policy_chunks",
    ) -> None:
        self.chroma_service = chroma_service
        self.collection_name = collection_name

    def get_collection(self):
        """Retrieve or initialize the target ChromaDB vector collection."""
        return self.chroma_service.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "HR policy chunks vector embeddings"},
        )

    def upsert_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert chunk vectors and retrieval metadata into ChromaDB.

        Using upsert guarantees that existing IDs are updated rather than duplicated.
        """
        if not ids:
            return
        collection = self.get_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d vector embeddings into ChromaDB collection '%s'", len(ids), self.collection_name)

    def count_generation_vectors(self, document_id: str, generation: int) -> int:
        """Return the number of vectors stored in ChromaDB for a specific document and generation."""
        collection = self.get_collection()
        try:
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            metadatas = results.get("metadatas", []) or []
            ids = results.get("ids", []) or []

            # If mock or client returned metadatas as a list of dicts
            count = 0
            gen_suffix = f"_g{generation}"
            for i, id_val in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else None
                if isinstance(meta, dict) and meta.get("generation") is not None:
                    if meta.get("generation") == generation:
                        count += 1
                elif str(id_val).endswith(gen_suffix):
                    count += 1
            return count
        except Exception as exc:
            logger.warning("Error counting generation %d vectors for document_id=%s: %s", generation, document_id, exc)
            return 0

    def get_indexed_ids_for_generation(self, document_id: str, generation: int) -> List[str]:
        """Fetch vector IDs for a specific document and generation."""
        collection = self.get_collection()
        try:
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids = results.get("ids", []) or []
            metadatas = results.get("metadatas", []) or []
            target_ids: List[str] = []
            gen_suffix = f"_g{generation}"
            for i, id_val in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else None
                if isinstance(meta, dict) and meta.get("generation") is not None:
                    if meta.get("generation") == generation:
                        target_ids.append(id_val)
                elif str(id_val).endswith(gen_suffix):
                    target_ids.append(id_val)
            return target_ids
        except Exception as exc:
            logger.warning("Could not query ChromaDB IDs for doc=%s, gen=%d: %s", document_id, generation, exc)
            return []

    def delete_generation(self, document_id: str, generation: int) -> int:
        """Delete all vectors for a specific document and generation (e.g. aborting failed run)."""
        target_ids = self.get_indexed_ids_for_generation(document_id, generation)
        if target_ids:
            self.delete_vectors(target_ids)
            logger.info("Purged %d unactivated vectors for document_id=%s gen=%d", len(target_ids), document_id, generation)
        return len(target_ids)

    def delete_previous_generations(self, document_id: str, active_generation: int) -> int:
        """Remove all vectors for document_id that belong to older generations than active_generation."""
        collection = self.get_collection()
        try:
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids = results.get("ids", []) or []
            metadatas = results.get("metadatas", []) or []
            old_ids: List[str] = []
            gen_suffix = f"_g{active_generation}"

            for i, id_val in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else None
                if isinstance(meta, dict) and meta.get("generation") is not None:
                    if meta.get("generation") != active_generation:
                        old_ids.append(id_val)
                elif not str(id_val).endswith(gen_suffix):
                    old_ids.append(id_val)

            if old_ids:
                self.delete_vectors(old_ids)
                logger.info(
                    "Purged %d older generation vector(s) for document_id=%s (active gen: %d)",
                    len(old_ids),
                    document_id,
                    active_generation,
                )
            return len(old_ids)
        except Exception as exc:
            logger.warning("Error deleting older generation vectors for document_id=%s: %s", document_id, exc)
            return 0

    def get_indexed_ids_for_document(self, document_id: str) -> List[str]:
        """Fetch all vector chunk IDs currently stored in ChromaDB for a given document_id."""
        collection = self.get_collection()
        try:
            results = collection.get(where={"document_id": document_id}, include=[])
            return results.get("ids", [])
        except Exception as exc:
            logger.warning("Could not query existing ChromaDB IDs for document_id=%s: %s", document_id, exc)
            return []

    def delete_vectors(self, ids: List[str]) -> None:
        """Delete specific vector records from ChromaDB by their chunk IDs."""
        if not ids:
            return
        collection = self.get_collection()
        collection.delete(ids=ids)
        logger.info("Deleted %d stale vectors from ChromaDB collection '%s'", len(ids), self.collection_name)

    def delete_vectors_by_document_id(self, document_id: str) -> None:
        """Purge all vector records belonging to a document from ChromaDB."""
        collection = self.get_collection()
        try:
            collection.delete(where={"document_id": document_id})
            logger.info("Purged all vectors for document_id=%s from ChromaDB", document_id)
        except Exception as exc:
            logger.warning("Error deleting ChromaDB vectors for document_id=%s: %s", document_id, exc)

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        collection = self.get_collection()
        return collection.count()

    def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query ChromaDB collection for nearest vector neighbors.

        Returns a dictionary with 'ids', 'documents', 'metadatas', and 'distances'.
        """
        collection = self.get_collection()
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            return collection.query(**kwargs)
        except Exception as exc:
            logger.error("Error querying ChromaDB collection '%s': %s", self.collection_name, exc)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

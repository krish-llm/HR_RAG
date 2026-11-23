"""Service orchestrating embedding generation, caching, and ChromaDB vector indexing."""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from app.core.exceptions import (
    AppException,
    EmbeddingException,
    IndexingException,
    NotFoundException,
    ProcessingException,
)
from app.core.logging_utils import TimedBlock
from app.modules.bm25 import BaseBM25IndexProvider, BM25ChunkRecord
from app.modules.chunking.repository import ChunkRepository
from app.modules.chunking.schemas import DocumentChunk
from app.modules.indexing.provider import BaseEmbeddingProvider
from app.modules.indexing.repository import (
    IndexingRepository,
    VectorIndexRepository,
    _format_iso_date,
    serialize_chroma_metadata,
)
from app.modules.indexing.schemas import (
    ChunkEmbeddingRecord,
    DocumentIndexingRecord,
    IndexingStageStatus,
    IndexingStatusResponse,
)
from app.modules.ingestion.repository import DocumentRepository
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)


class IndexingService:
    """Coordinates embedding generation, content-hash caching, and ChromaDB vector indexing."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        indexing_repo: IndexingRepository,
        vector_repo: VectorIndexRepository,
        embedding_provider: BaseEmbeddingProvider,
        bm25_provider: Optional[BaseBM25IndexProvider] = None,
        batch_size: int = 16,
    ) -> None:
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.indexing_repo = indexing_repo
        self.vector_repo = vector_repo
        self.embedding_provider = embedding_provider
        self.bm25_provider = bm25_provider
        self.batch_size = max(1, batch_size)

    async def index_document(self, document_id: str) -> IndexingStatusResponse:
        """Trigger synchronous embedding generation and ChromaDB indexing for a document.

        Steps:
        1. Verify registered document exists in MongoDB.
        2. Fetch all chunks from document_chunks. Validate chunks exist.
        3. Transition document status to 'embedding'.
        4. Calculate SHA-256 content hashes for each chunk.
           - Reuse existing valid embeddings from MongoDB (cache hit).
           - Queue missing/modified chunks for Azure OpenAI embedding generation (cache miss).
        5. Generate embeddings in configurable batches. Save new embeddings to MongoDB.
        6. Transition document status to 'indexing'.
        7. Upsert all chunk vectors and metadata into ChromaDB.
        8. Remove any stale vectors from prior re-chunking runs.
        9. Transition document status to 'indexed' once all chunks are verified.
        """
        # 1. Verify document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        # 2. Fetch chunks from MongoDB
        raw_chunks = await self.chunk_repo.get_all_by_document_id(document_id)
        if not raw_chunks:
            # Check if document was chunked
            doc_status = doc.get("processing_status")
            if doc_status not in ("completed", "chunked", "processing", "ready"):
                raise ProcessingException(
                    message=f"Document with ID '{document_id}' has not been chunked yet. Run chunking before indexing.",
                    error_code="DOCUMENT_NOT_CHUNKED",
                    status_code=400,
                )
            # Empty document case: mark indexed with 0 chunks
            now = current_utc_time()
            empty_record = DocumentIndexingRecord(
                document_id=document_id,
                total_chunks=0,
                embedded_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                active_generation=0,
                target_generation=None,
                embedding_model=self.embedding_provider.get_model_name(),
                indexing_status=IndexingStageStatus.INDEXED,
                last_indexed_at=now,
                error_message=None,
                updated_at=now,
            )
            await self.indexing_repo.save_tracking_record(empty_record)
            # Purge any old vectors in ChromaDB and BM25
            self.vector_repo.delete_vectors_by_document_id(document_id)
            if self.bm25_provider:
                self.bm25_provider.delete_document(document_id)
            return IndexingStatusResponse(
                document_id=document_id,
                total_chunks=0,
                embedded_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                active_generation=0,
                embedding_model=self.embedding_provider.get_model_name(),
                indexing_status="indexed",
                last_indexed_at=now,
            )

        parsed_chunks = [DocumentChunk.model_validate(c) for c in raw_chunks]
        total_chunks = len(parsed_chunks)
        model_name = self.embedding_provider.get_model_name()

        # 3. Determine active and target index generations
        existing_tracking = await self.indexing_repo.get_tracking_record(document_id)
        active_gen = existing_tracking.get("active_generation", 0) if existing_tracking else 0
        target_gen = active_gen + 1

        # Transition to 'embedding' stage while keeping active generation live
        tracking_record = DocumentIndexingRecord(
            document_id=document_id,
            total_chunks=total_chunks,
            embedded_chunks=0,
            indexed_chunks=existing_tracking.get("indexed_chunks", 0) if existing_tracking else 0,
            failed_chunks=0,
            active_generation=active_gen,
            target_generation=target_gen,
            embedding_model=model_name,
            indexing_status=IndexingStageStatus.EMBEDDING,
            last_indexed_at=existing_tracking.get("last_indexed_at") if existing_tracking else None,
            error_message=None,
        )
        await self.indexing_repo.save_tracking_record(tracking_record)
        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="indexing",
            error_message=None,
        )

        # 4. Check MongoDB embedding cache using content hashes and generate missing embeddings
        async with TimedBlock(logger, "embedding_generation") as timer:
            chunk_embeddings_map: Dict[str, List[float]] = {}
            missing_chunks: List[DocumentChunk] = []

            for chunk in parsed_chunks:
                # Handle empty content
                if not chunk.content.strip():
                    continue

                content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                cached = await self.indexing_repo.get_chunk_embedding(
                    chunk_id=chunk.chunk_id,
                    content_hash=content_hash,
                    model_name=model_name,
                )
                if cached and cached.get("embedding"):
                    chunk_embeddings_map[chunk.chunk_id] = cached["embedding"]
                else:
                    missing_chunks.append(chunk)

            # 5. Generate missing embeddings in batches with Azure OpenAI
            if missing_chunks:
                logger.info(
                    "Generating embeddings for %d/%d chunks for document_id=%s (cached=%d)",
                    len(missing_chunks),
                    total_chunks,
                    document_id,
                    len(chunk_embeddings_map),
                )
                for i in range(0, len(missing_chunks), self.batch_size):
                    batch = missing_chunks[i : i + self.batch_size]
                    batch_texts = [c.content for c in batch]

                    try:
                        vectors = await self.embedding_provider.embed_texts(batch_texts)
                    except Exception as exc:
                        error_msg = f"Failed to generate embeddings: {exc}"
                        logger.error("Embedding generation error for document_id=%s: %s", document_id, error_msg)
                        # Mark document tracking as failed
                        tracking_record.indexing_status = IndexingStageStatus.FAILED
                        tracking_record.failed_chunks = len(missing_chunks) - i
                        tracking_record.error_message = error_msg
                        tracking_record.updated_at = current_utc_time()
                        await self.indexing_repo.save_tracking_record(tracking_record)
                        await self.document_repo.update_processing_status(
                            document_id=document_id,
                            processing_status="failed",
                            error_message=error_msg,
                        )
                        if isinstance(exc, AppException):
                            raise
                        raise EmbeddingException(
                            message=error_msg,
                            error_code="EMBEDDING_FAILED",
                            status_code=500,
                        ) from exc

                    # Save generated embeddings to MongoDB cache
                    records_to_save: List[ChunkEmbeddingRecord] = []
                    for chunk, vec in zip(batch, vectors):
                        chunk_embeddings_map[chunk.chunk_id] = vec
                        c_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                        rec = ChunkEmbeddingRecord(
                            chunk_id=chunk.chunk_id,
                            document_id=document_id,
                            embedding_model=model_name,
                            embedding_version="1",
                            content_hash=c_hash,
                            embedding=vec,
                            embedding_status="completed",
                        )
                        records_to_save.append(rec)
                    await self.indexing_repo.save_chunk_embeddings_batch(records_to_save)

            timer.set_extra(
                chunks=total_chunks,
                missing=len(missing_chunks),
                cached=len(chunk_embeddings_map) - len(missing_chunks),
            )

        # 6. Transition to 'indexing' stage
        tracking_record.embedded_chunks = len(chunk_embeddings_map)
        tracking_record.indexing_status = IndexingStageStatus.INDEXING
        tracking_record.updated_at = current_utc_time()
        await self.indexing_repo.save_tracking_record(tracking_record)

        # 7. Prepare ChromaDB payloads and BM25 records under target generation identifier
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in parsed_chunks:
            if chunk.chunk_id in chunk_embeddings_map:
                # Generation-specific vector ID keeps previous generation available during write
                vector_id = f"{chunk.chunk_id}_g{target_gen}"
                ids.append(vector_id)
                embeddings.append(chunk_embeddings_map[chunk.chunk_id])
                documents.append(chunk.content)
                metadatas.append(serialize_chroma_metadata(chunk, model_name, generation=target_gen))

        bm25_records: List[BM25ChunkRecord] = []
        if self.bm25_provider:
            bm25_records = [
                BM25ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    generation=target_gen,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    document_name=chunk.document_name,
                    document_type=chunk.document_type,
                    version=chunk.version,
                    school_id=chunk.school_id,
                    group_id=chunk.group_id,
                    branch_id=chunk.branch_id,
                    state=chunk.state,
                    is_global=chunk.is_global,
                    staff_category=chunk.staff_category,
                    employment_type=chunk.employment_type,
                    department=chunk.department,
                    heading=chunk.heading,
                    heading_path=chunk.heading_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    effective_from=_format_iso_date(chunk.effective_from),
                    effective_to=_format_iso_date(chunk.effective_to),
                )
                for chunk in parsed_chunks
                if chunk.content.strip()
            ]

        # 8. Upsert new vectors into ChromaDB and BM25 under target generation
        try:
            with TimedBlock(logger, "vector_indexing") as timer:
                timer.set_extra(vectors=len(ids), bm25_chunks=len(bm25_records), generation=target_gen)
                self.vector_repo.upsert_vectors(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                if self.bm25_provider and bm25_records:
                    self.bm25_provider.index_chunks(bm25_records)
        except Exception as exc:
            error_msg = f"Failed to upsert vectors/BM25 chunks into storage: {exc}"
            logger.error("Storage indexing error for document_id=%s: %s", document_id, error_msg)
            # Rollback: prune target generation from both ChromaDB and BM25
            self.vector_repo.delete_generation(document_id, target_gen)
            if self.bm25_provider:
                self.bm25_provider.delete_generation(document_id, target_gen)
            tracking_record.indexing_status = IndexingStageStatus.FAILED
            tracking_record.failed_chunks = len(ids)
            tracking_record.target_generation = None
            tracking_record.error_message = error_msg
            tracking_record.updated_at = current_utc_time()
            await self.indexing_repo.save_tracking_record(tracking_record)
            await self.document_repo.update_processing_status(
                document_id=document_id,
                processing_status="failed",
                error_message=error_msg,
            )
            if isinstance(exc, AppException):
                raise
            raise IndexingException(
                message=error_msg,
                error_code="INDEXING_FAILED",
                status_code=500,
            ) from exc

        # 9. Verify all expected vectors and BM25 chunks were successfully written
        written_count = self.vector_repo.count_generation_vectors(document_id, target_gen)
        bm25_written_count = (
            self.bm25_provider.count_generation_chunks(document_id, target_gen)
            if self.bm25_provider
            else len(bm25_records)
        )

        if written_count != len(ids) or (self.bm25_provider and bm25_written_count != len(bm25_records)):
            self.vector_repo.delete_generation(document_id, target_gen)
            if self.bm25_provider:
                self.bm25_provider.delete_generation(document_id, target_gen)
            error_msg = (
                f"Generation verification failed for document '{document_id}': "
                f"expected {len(ids)} vectors and {len(bm25_records)} BM25 chunks for generation {target_gen}, "
                f"found {written_count} vectors and {bm25_written_count} BM25 chunks"
            )
            logger.error(error_msg)
            tracking_record.indexing_status = IndexingStageStatus.FAILED
            tracking_record.failed_chunks = len(ids) - written_count
            tracking_record.target_generation = None
            tracking_record.error_message = error_msg
            tracking_record.updated_at = current_utc_time()
            await self.indexing_repo.save_tracking_record(tracking_record)
            await self.document_repo.update_processing_status(
                document_id=document_id,
                processing_status="failed",
                error_message=error_msg,
            )
            raise IndexingException(
                message=error_msg,
                error_code="INDEXING_VERIFICATION_FAILED",
                status_code=500,
            )

        # 10. Activate new generation only after successful completion and verification
        now = current_utc_time()
        tracking_record.active_generation = target_gen
        tracking_record.target_generation = None
        tracking_record.indexing_status = IndexingStageStatus.INDEXED
        tracking_record.indexed_chunks = len(ids)
        tracking_record.last_indexed_at = now
        tracking_record.error_message = None
        tracking_record.updated_at = now
        await self.indexing_repo.save_tracking_record(tracking_record)

        await self.document_repo.update_processing_status(
            document_id=document_id,
            processing_status="completed",
            error_message=None,
        )

        # 11. Remove previous generation only after new generation is activated
        self.vector_repo.delete_previous_generations(
            document_id=document_id,
            active_generation=target_gen,
        )
        if self.bm25_provider:
            self.bm25_provider.delete_previous_generations(
                document_id=document_id,
                active_generation=target_gen,
            )

        logger.info(
            "Successfully indexed and activated generation %d for document_id=%s in ChromaDB (total=%d, embedded=%d, indexed=%d)",
            target_gen,
            document_id,
            total_chunks,
            len(chunk_embeddings_map),
            len(ids),
        )

        return IndexingStatusResponse(
            document_id=document_id,
            total_chunks=total_chunks,
            embedded_chunks=len(chunk_embeddings_map),
            indexed_chunks=len(ids),
            failed_chunks=0,
            active_generation=target_gen,
            embedding_model=model_name,
            indexing_status="indexed",
            last_indexed_at=now,
        )

    async def get_indexing_status(self, document_id: str) -> IndexingStatusResponse:
        """Retrieve the indexing status and progress for a document."""
        # Verify parent document exists
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException(
                message=f"Document with ID '{document_id}' not found",
                error_code="DOCUMENT_NOT_FOUND",
            )

        record = await self.indexing_repo.get_tracking_record(document_id)
        if not record:
            # Check chunks count
            chunks = await self.chunk_repo.get_all_by_document_id(document_id)
            total = len(chunks)
            status_str = "chunked" if total > 0 else "pending"
            return IndexingStatusResponse(
                document_id=document_id,
                total_chunks=total,
                embedded_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                active_generation=0,
                embedding_model=self.embedding_provider.get_model_name(),
                indexing_status=status_str,
                last_indexed_at=None,
            )

        return IndexingStatusResponse(
            document_id=record["document_id"],
            total_chunks=record.get("total_chunks", 0),
            embedded_chunks=record.get("embedded_chunks", 0),
            indexed_chunks=record.get("indexed_chunks", 0),
            failed_chunks=record.get("failed_chunks", 0),
            active_generation=record.get("active_generation", 0),
            embedding_model=record.get("embedding_model", self.embedding_provider.get_model_name()),
            indexing_status=record.get("indexing_status", "pending"),
            last_indexed_at=record.get("last_indexed_at"),
            error_message=record.get("error_message"),
        )

    async def rebuild_bm25_index(self, document_id: Optional[str] = None) -> int:
        """Rebuild SQLite BM25 index from authoritative MongoDB chunks.

        Args:
            document_id: Optional specific document to rebuild. If None, rebuilds for all
                         active indexed documents.

        Returns:
            Total number of chunks indexed into BM25.
        """
        if not self.bm25_provider:
            logger.warning("BM25 provider is not configured; cannot rebuild BM25 index.")
            return 0

        if document_id:
            tracking = await self.indexing_repo.get_tracking_record(document_id)
            if not tracking or tracking.get("active_generation", 0) <= 0:
                logger.warning("Document %s has no active index generation", document_id)
                return 0
            active_gen = tracking["active_generation"]
            raw_chunks = await self.chunk_repo.get_all_by_document_id(document_id)
            parsed_chunks = [DocumentChunk.model_validate(c) for c in raw_chunks]
            bm25_records = [
                BM25ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    generation=active_gen,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    document_name=chunk.document_name,
                    document_type=chunk.document_type,
                    version=chunk.version,
                    school_id=chunk.school_id,
                    group_id=chunk.group_id,
                    branch_id=chunk.branch_id,
                    state=chunk.state,
                    is_global=chunk.is_global,
                    staff_category=chunk.staff_category,
                    employment_type=chunk.employment_type,
                    department=chunk.department,
                    heading=chunk.heading,
                    heading_path=chunk.heading_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    effective_from=_format_iso_date(chunk.effective_from),
                    effective_to=_format_iso_date(chunk.effective_to),
                )
                for chunk in parsed_chunks
                if chunk.content.strip()
            ]
            self.bm25_provider.delete_document(document_id)
            self.bm25_provider.index_chunks(bm25_records)
            logger.info("Rebuilt BM25 index for document %s (%d chunks)", document_id, len(bm25_records))
            return len(bm25_records)
        else:
            records = await self.indexing_repo.get_all_active_tracking_records()
            all_bm25_records: List[BM25ChunkRecord] = []
            for track in records:
                doc_id = track["document_id"]
                active_gen = track.get("active_generation", 0)
                if active_gen <= 0:
                    continue
                raw_chunks = await self.chunk_repo.get_all_by_document_id(doc_id)
                for c in raw_chunks:
                    chunk = DocumentChunk.model_validate(c)
                    if not chunk.content.strip():
                        continue
                    all_bm25_records.append(
                        BM25ChunkRecord(
                            chunk_id=chunk.chunk_id,
                            document_id=doc_id,
                            generation=active_gen,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            document_name=chunk.document_name,
                            document_type=chunk.document_type,
                            version=chunk.version,
                            school_id=chunk.school_id,
                            group_id=chunk.group_id,
                            branch_id=chunk.branch_id,
                            state=chunk.state,
                            is_global=chunk.is_global,
                            staff_category=chunk.staff_category,
                            employment_type=chunk.employment_type,
                            department=chunk.department,
                            heading=chunk.heading,
                            heading_path=chunk.heading_path,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            effective_from=_format_iso_date(chunk.effective_from),
                            effective_to=_format_iso_date(chunk.effective_to),
                        )
                    )
            self.bm25_provider.rebuild_index(all_bm25_records)
            logger.info("Rebuilt BM25 index across all documents (%d total chunks)", len(all_bm25_records))
            return len(all_bm25_records)

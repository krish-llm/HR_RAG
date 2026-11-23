"""Retrieval Service orchestrating query understanding, embedding, dense & BM25 retrieval, and authoritative validation."""

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from app.api.dependencies import SchoolContext
from app.core.exceptions import AppException, EmbeddingException
from app.core.logging_utils import TimedBlock
from app.modules.bm25.base import BaseBM25IndexProvider
from app.modules.bm25.schemas import BM25SearchResult
from app.modules.indexing.provider import BaseEmbeddingProvider
from app.modules.indexing.repository import IndexingRepository, VectorIndexRepository
from app.modules.ingestion.repository import DocumentRepository
from app.modules.retrieval.filtering import PolicyScopeFilter, _parse_date
from app.modules.retrieval.fusion import reciprocal_rank_fusion
from app.modules.retrieval.schemas import (
    QueryUnderstandingResult,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.modules.retrieval.understanding import QueryUnderstandingService
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)


class RetrievalService:
    """Coordinates deterministic query understanding, dense & BM25 hybrid retrieval, and authoritative policy validation."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        indexing_repo: IndexingRepository,
        vector_repo: VectorIndexRepository,
        embedding_provider: BaseEmbeddingProvider,
        bm25_provider: Optional[BaseBM25IndexProvider] = None,
        retrieval_mode: str = "dense",
        hybrid_dense_top_k: int = 20,
        hybrid_bm25_top_k: int = 20,
        hybrid_rrf_k: int = 60,
        understanding_service: Optional[QueryUnderstandingService] = None,
        policy_filter: Optional[PolicyScopeFilter] = None,
        candidate_pool_size: int = 20,
    ) -> None:
        self.document_repo = document_repo
        self.indexing_repo = indexing_repo
        self.vector_repo = vector_repo
        self.embedding_provider = embedding_provider
        self.bm25_provider = bm25_provider
        self.retrieval_mode = (retrieval_mode or "dense").strip().lower()
        self.hybrid_dense_top_k = max(1, hybrid_dense_top_k)
        self.hybrid_bm25_top_k = max(1, hybrid_bm25_top_k)
        self.hybrid_rrf_k = max(1, hybrid_rrf_k)
        self.understanding_service = understanding_service or QueryUnderstandingService()
        self.policy_filter = policy_filter or PolicyScopeFilter()
        self.candidate_pool_size = max(5, candidate_pool_size)

    async def _embed_query(self, query: str) -> List[float]:
        """Generate vector embedding for the normalized query string."""
        async with TimedBlock(logger, "query_embedding"):
            try:
                vectors = await self.embedding_provider.embed_texts([query])
                if not vectors or not vectors[0]:
                    raise EmbeddingException(
                        message="Embedding provider returned empty vector representation for query",
                        error_code="QUERY_EMBEDDING_EMPTY",
                        status_code=502,
                    )
                return vectors[0]
            except Exception as exc:
                logger.error("Failed to generate query embedding: %s", exc)
                if isinstance(exc, AppException):
                    raise
                raise EmbeddingException(
                    message=f"Failed to generate query embedding: {exc}",
                    error_code="AZURE_OPENAI_ERROR",
                    status_code=502,
                ) from exc

    def _query_dense_vectors(
        self, query_embedding: List[float], pool_size: int
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """Synchronously query ChromaDB vector index."""
        with TimedBlock(logger, "dense_retrieval") as d_timer:
            with TimedBlock(logger, "vector_search") as v_timer:
                query_res = self.vector_repo.query_vectors(
                    query_embedding=query_embedding,
                    top_k=pool_size,
                )
                raw_ids = query_res.get("ids", [[]])[0]
                raw_docs = query_res.get("documents", [[]])[0]
                raw_metas = query_res.get("metadatas", [[]])[0]
                raw_dists = query_res.get("distances", [[]])[0]
                v_timer.set_extra(candidates=len(raw_ids))
                d_timer.set_extra(candidates=len(raw_ids))
                return raw_ids, raw_docs, raw_metas, raw_dists

    def _query_bm25_index(
        self,
        query: str,
        pool_size: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BM25SearchResult]:
        """Synchronously query SQLite FTS5 BM25 index with optional metadata scoping filters."""
        if not self.bm25_provider:
            return []
        with TimedBlock(logger, "bm25_retrieval") as timer:
            results = self.bm25_provider.search(query=query, top_k=pool_size, filters=filters)
            timer.set_extra(candidates=len(results))
            return results

    async def retrieve(
        self,
        request: RetrievalRequest,
        school_context: SchoolContext,
    ) -> RetrievalResponse:
        """Execute retrieval pipeline with strict scoping and authoritative MongoDB validation.

        Supports both 'dense' and 'hybrid' retrieval modes.
        """
        async with TimedBlock(logger, "total_retrieval") as total_timer:
            # 1. Deterministic Query Understanding
            with TimedBlock(logger, "query_understanding") as timer:
                understanding = self.understanding_service.understand_query(
                    query=request.question,
                    school_context=school_context,
                    employee_context=request.employee_context,
                    as_of_date=request.as_of_date,
                )
                timer.set_extra(
                    intent=understanding.intent,
                    keywords=len(understanding.extracted_keywords),
                )

            requested_top_k = min(max(1, request.top_k), 50)

            # 2. Branch: Hybrid Retrieval
            if self.retrieval_mode == "hybrid":
                return await self._retrieve_hybrid(request, understanding, school_context, requested_top_k)

            # 3. Branch: Pure Dense Retrieval (Default)
            return await self._retrieve_dense(request, understanding, school_context, requested_top_k)

    async def _retrieve_dense(
        self,
        request: RetrievalRequest,
        understanding: QueryUnderstandingResult,
        school_context: SchoolContext,
        requested_top_k: int,
    ) -> RetrievalResponse:
        """Execute traditional single-stream dense vector retrieval."""
        # 1. Query Embedding Generation
        query_embedding = await self._embed_query(understanding.normalized_query)

        # 2. Vector Candidate Search with Over-fetching
        pool_size = max(requested_top_k * 4, self.candidate_pool_size)
        raw_ids, raw_docs, raw_metas, raw_dists = self._query_dense_vectors(query_embedding, pool_size)
        total_candidates_evaluated = len(raw_ids)

        # 3. Authoritative Candidate Validation against MongoDB
        with TimedBlock(logger, "mongo_validation") as timer:
            doc_cache: Dict[str, Optional[Dict[str, Any]]] = {}
            tracking_cache: Dict[str, Optional[Dict[str, Any]]] = {}
            valid_chunks, seen_chunk_ids = await self._validate_dense_candidates(
                raw_ids=raw_ids,
                raw_docs=raw_docs,
                raw_metas=raw_metas,
                raw_dists=raw_dists,
                understanding=understanding,
                school_context=school_context,
                doc_cache=doc_cache,
                tracking_cache=tracking_cache,
            )

            # Secondary Over-fetch retry if candidate count fell below requested top_k
            if len(valid_chunks) < requested_top_k and self.vector_repo.count() > pool_size:
                secondary_pool_size = min(pool_size * 2, 100)
                logger.info(
                    "Over-fetching retry: expanding candidate pool from %d to %d (current valid: %d)",
                    pool_size,
                    secondary_pool_size,
                    len(valid_chunks),
                )
                retry_ids, retry_docs, retry_metas, retry_dists = self._query_dense_vectors(
                    query_embedding, secondary_pool_size
                )
                total_candidates_evaluated = max(total_candidates_evaluated, len(retry_ids))
                more_chunks, _ = await self._validate_dense_candidates(
                    raw_ids=retry_ids,
                    raw_docs=retry_docs,
                    raw_metas=retry_metas,
                    raw_dists=retry_dists,
                    understanding=understanding,
                    school_context=school_context,
                    doc_cache=doc_cache,
                    tracking_cache=tracking_cache,
                    seen_chunk_ids=seen_chunk_ids,
                )
                valid_chunks.extend(more_chunks)

            timer.set_extra(candidates=total_candidates_evaluated, valid_chunks=len(valid_chunks))

        # 4. Sort by similarity score descending and apply top_k limit
        valid_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        final_results = valid_chunks[:requested_top_k]
        status_str = "success" if final_results else "empty"

        logger.info(
            "Dense retrieval complete for query='%s': %d/%d valid chunks selected (status: %s)",
            understanding.normalized_query,
            len(final_results),
            total_candidates_evaluated,
            status_str,
        )

        return RetrievalResponse(
            query_understanding=understanding,
            chunks=final_results,
            total_candidates=total_candidates_evaluated,
            total_results=len(final_results),
            retrieval_status=status_str,
            retrieval_mode="dense",
            retrieved_at=current_utc_time(),
        )

    async def _retrieve_hybrid(
        self,
        request: RetrievalRequest,
        understanding: QueryUnderstandingResult,
        school_context: SchoolContext,
        requested_top_k: int,
    ) -> RetrievalResponse:
        """Execute concurrent dense + BM25 retrieval and Reciprocal Rank Fusion with starvation prevention."""
        if not self.bm25_provider:
            raise AppException(
                message="Hybrid retrieval requested but BM25 provider is not configured or unavailable",
                error_code="BM25_PROVIDER_UNAVAILABLE",
                status_code=500,
            )

        dense_pool_size = max(requested_top_k * 4, self.hybrid_dense_top_k, self.candidate_pool_size)
        bm25_pool_size = max(requested_top_k * 4, self.hybrid_bm25_top_k)

        # Build tenant scope filter for BM25 search to prevent other schools from starving candidates
        bm25_filters: Dict[str, Any] = {}
        if school_context.school_id:
            bm25_filters["school_id"] = school_context.school_id
        if school_context.group_id:
            bm25_filters["group_id"] = school_context.group_id
        if school_context.branch_id:
            bm25_filters["branch_id"] = school_context.branch_id
        if school_context.state:
            bm25_filters["state"] = school_context.state
        if school_context.is_global:
            bm25_filters["is_global"] = True

        async def _fetch_dense(pool: int):
            q_emb = await self._embed_query(understanding.normalized_query)
            return await asyncio.to_thread(self._query_dense_vectors, q_emb, pool)

        async def _fetch_bm25(pool: int):
            return await asyncio.to_thread(
                self._query_bm25_index,
                understanding.normalized_query,
                pool,
                bm25_filters,
            )

        # Run embedding+dense vector query and BM25 full-text search concurrently
        (raw_ids, raw_docs, raw_metas, raw_dists), bm25_results = await asyncio.gather(
            _fetch_dense(dense_pool_size),
            _fetch_bm25(bm25_pool_size),
        )

        total_candidates_evaluated = len(raw_ids) + len(bm25_results)

        # Validate candidates against MongoDB authoritative policy & scoping rules
        with TimedBlock(logger, "mongo_validation") as timer:
            doc_cache: Dict[str, Optional[Dict[str, Any]]] = {}
            tracking_cache: Dict[str, Optional[Dict[str, Any]]] = {}

            dense_valid, seen_dense_ids = await self._validate_dense_candidates(
                raw_ids=raw_ids,
                raw_docs=raw_docs,
                raw_metas=raw_metas,
                raw_dists=raw_dists,
                understanding=understanding,
                school_context=school_context,
                doc_cache=doc_cache,
                tracking_cache=tracking_cache,
            )

            bm25_valid, seen_bm25_ids = await self._validate_bm25_candidates(
                bm25_results=bm25_results,
                understanding=understanding,
                school_context=school_context,
                doc_cache=doc_cache,
                tracking_cache=tracking_cache,
            )

            # Controlled Over-fetching retry for BM25 if valid candidates fell below requested_top_k
            total_matching_bm25 = self.bm25_provider.count(filters=bm25_filters)
            if len(bm25_valid) < requested_top_k and total_matching_bm25 > bm25_pool_size:
                secondary_bm25_pool = min(bm25_pool_size * 3, 100)
                logger.info(
                    "BM25 over-fetching retry: expanding candidate pool from %d to %d (current valid: %d)",
                    bm25_pool_size,
                    secondary_bm25_pool,
                    len(bm25_valid),
                )
                retry_bm25_results = await asyncio.to_thread(
                    self._query_bm25_index,
                    understanding.normalized_query,
                    secondary_bm25_pool,
                    bm25_filters,
                )
                total_candidates_evaluated += len(retry_bm25_results)
                more_bm25_valid, _ = await self._validate_bm25_candidates(
                    bm25_results=retry_bm25_results,
                    understanding=understanding,
                    school_context=school_context,
                    doc_cache=doc_cache,
                    tracking_cache=tracking_cache,
                    seen_chunk_ids=seen_bm25_ids,
                )
                bm25_valid.extend(more_bm25_valid)

            # Controlled Over-fetching retry for Dense if valid candidates fell below requested_top_k
            if len(dense_valid) < requested_top_k and self.vector_repo.count() > dense_pool_size:
                secondary_dense_pool = min(dense_pool_size * 2, 100)
                logger.info(
                    "Dense over-fetching retry: expanding candidate pool from %d to %d (current valid: %d)",
                    dense_pool_size,
                    secondary_dense_pool,
                    len(dense_valid),
                )
                retry_q_emb = await self._embed_query(understanding.normalized_query)
                retry_ids, retry_docs, retry_metas, retry_dists = await asyncio.to_thread(
                    self._query_dense_vectors,
                    retry_q_emb,
                    secondary_dense_pool,
                )
                total_candidates_evaluated += len(retry_ids)
                more_dense_valid, _ = await self._validate_dense_candidates(
                    raw_ids=retry_ids,
                    raw_docs=retry_docs,
                    raw_metas=retry_metas,
                    raw_dists=retry_dists,
                    understanding=understanding,
                    school_context=school_context,
                    doc_cache=doc_cache,
                    tracking_cache=tracking_cache,
                    seen_chunk_ids=seen_dense_ids,
                )
                dense_valid.extend(more_dense_valid)

            timer.set_extra(
                candidates=total_candidates_evaluated,
                dense_valid=len(dense_valid),
                bm25_valid=len(bm25_valid),
            )

        # Merge candidate lists using Reciprocal Rank Fusion (RRF)
        with TimedBlock(logger, "rrf_fusion") as rrf_timer:
            fused_chunks = reciprocal_rank_fusion(
                dense_chunks=dense_valid,
                bm25_chunks=bm25_valid,
                k=self.hybrid_rrf_k,
                top_k=requested_top_k,
            )
            rrf_timer.set_extra(
                dense_count=len(dense_valid),
                bm25_count=len(bm25_valid),
                fused_count=len(fused_chunks),
            )

        status_str = "success" if fused_chunks else "empty"
        logger.info(
            "Hybrid retrieval complete for query='%s': %d fused chunks selected from %d candidates (dense=%d, bm25=%d)",
            understanding.normalized_query,
            len(fused_chunks),
            total_candidates_evaluated,
            len(dense_valid),
            len(bm25_valid),
        )

        return RetrievalResponse(
            query_understanding=understanding,
            chunks=fused_chunks,
            total_candidates=total_candidates_evaluated,
            total_results=len(fused_chunks),
            retrieval_status=status_str,
            retrieval_mode="hybrid",
            retrieved_at=current_utc_time(),
        )

    async def _validate_dense_candidates(
        self,
        raw_ids: List[str],
        raw_docs: List[str],
        raw_metas: List[Dict[str, Any]],
        raw_dists: List[float],
        understanding: QueryUnderstandingResult,
        school_context: SchoolContext,
        doc_cache: Dict[str, Optional[Dict[str, Any]]],
        tracking_cache: Dict[str, Optional[Dict[str, Any]]],
        seen_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[RetrievedChunk], Set[str]]:
        """Validate raw ChromaDB dense candidates against authoritative MongoDB rules."""
        valid_chunks: List[RetrievedChunk] = []
        seen = seen_chunk_ids if seen_chunk_ids is not None else set()

        for i, vector_id in enumerate(raw_ids):
            meta = raw_metas[i] if i < len(raw_metas) else {}
            content = raw_docs[i] if i < len(raw_docs) else ""
            dist = raw_dists[i] if i < len(raw_dists) else 0.5

            chunk_id = meta.get("chunk_id") or vector_id
            if chunk_id in seen:
                continue

            doc_id = meta.get("document_id")
            if not doc_id:
                continue

            # Fetch parent document from MongoDB (cached)
            if doc_id not in doc_cache:
                doc_cache[doc_id] = await self.document_repo.get_by_id(doc_id)
            doc_record = doc_cache[doc_id]
            if not doc_record:
                logger.debug("Rejected dense candidate %s: document %s not found", chunk_id, doc_id)
                continue

            # Policy Lifecycle & Dates
            is_active, status_reason = self.policy_filter.is_policy_active(
                doc_record, as_of_date=understanding.as_of_date
            )
            if not is_active:
                logger.debug("Rejected dense candidate %s: %s", chunk_id, status_reason)
                continue

            # Generation validation
            if doc_id not in tracking_cache:
                tracking_cache[doc_id] = await self.indexing_repo.get_tracking_record(doc_id)
            tracking_record = tracking_cache[doc_id]

            cand_gen = meta.get("generation", 1)
            is_gen_valid, gen_reason = self.policy_filter.is_generation_active(
                doc_id=doc_id,
                candidate_generation=cand_gen,
                tracking_record=tracking_record,
            )
            if not is_gen_valid:
                logger.debug("Rejected dense candidate %s: %s", chunk_id, gen_reason)
                continue

            # Tenant Scope
            if not self.policy_filter.is_tenant_scope_applicable(meta, school_context):
                logger.debug("Rejected dense candidate %s: tenant scope mismatch", chunk_id)
                continue

            # Employee Applicability
            if not self.policy_filter.is_employee_applicable(meta, understanding.employee_context):
                logger.debug("Rejected dense candidate %s: employee applicability mismatch", chunk_id)
                continue

            similarity_score = round(1.0 / (1.0 + max(0.0, float(dist))), 4)
            heading_path_raw = meta.get("heading_path")
            if isinstance(heading_path_raw, list):
                heading_path = [str(p) for p in heading_path_raw]
            elif isinstance(heading_path_raw, str) and heading_path_raw.strip():
                heading_path = [p.strip() for p in heading_path_raw.split(" > ") if p.strip()]
            else:
                heading_path = []

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                document_name=meta.get("document_name") or doc_record.get("document_name", "Unknown Policy"),
                document_type=meta.get("document_type") or doc_record.get("document_type", "policy"),
                version=int(meta.get("version") or doc_record.get("version", 1)),
                content=content,
                heading=meta.get("heading") or None,
                heading_path=heading_path,
                page_start=int(meta.get("page_start", 1)),
                page_end=int(meta.get("page_end", 1)),
                similarity_score=similarity_score,
                dense_score=similarity_score,
                dense_rank=len(valid_chunks) + 1,
                school_id=meta.get("school_id") or doc_record.get("school_id"),
                group_id=meta.get("group_id") or doc_record.get("group_id"),
                branch_id=meta.get("branch_id") or doc_record.get("branch_id"),
                effective_from=_parse_date(meta.get("effective_from") or doc_record.get("effective_from")),
                effective_to=_parse_date(meta.get("effective_to") or doc_record.get("effective_to")),
                retrieval_mode="dense",
            )
            seen.add(chunk_id)
            valid_chunks.append(chunk)

        return valid_chunks, seen

    async def _validate_bm25_candidates(
        self,
        bm25_results: List[BM25SearchResult],
        understanding: QueryUnderstandingResult,
        school_context: SchoolContext,
        doc_cache: Dict[str, Optional[Dict[str, Any]]],
        tracking_cache: Dict[str, Optional[Dict[str, Any]]],
        seen_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[RetrievedChunk], Set[str]]:
        """Validate raw SQLite BM25 candidates against authoritative MongoDB rules."""
        valid_chunks: List[RetrievedChunk] = []
        seen = seen_chunk_ids if seen_chunk_ids is not None else set()

        for res in bm25_results:
            chunk_id = res.chunk_id
            if chunk_id in seen:
                continue

            doc_id = res.document_id
            if not doc_id:
                continue

            # Fetch parent document from MongoDB (cached)
            if doc_id not in doc_cache:
                doc_cache[doc_id] = await self.document_repo.get_by_id(doc_id)
            doc_record = doc_cache[doc_id]
            if not doc_record:
                logger.debug("Rejected BM25 candidate %s: document %s not found", chunk_id, doc_id)
                continue

            # Policy Lifecycle & Dates
            is_active, status_reason = self.policy_filter.is_policy_active(
                doc_record, as_of_date=understanding.as_of_date
            )
            if not is_active:
                logger.debug("Rejected BM25 candidate %s: %s", chunk_id, status_reason)
                continue

            # Generation validation
            if doc_id not in tracking_cache:
                tracking_cache[doc_id] = await self.indexing_repo.get_tracking_record(doc_id)
            tracking_record = tracking_cache[doc_id]

            is_gen_valid, gen_reason = self.policy_filter.is_generation_active(
                doc_id=doc_id,
                candidate_generation=res.generation,
                tracking_record=tracking_record,
            )
            if not is_gen_valid:
                logger.debug("Rejected BM25 candidate %s: %s", chunk_id, gen_reason)
                continue

            # Tenant Scope
            cand_meta = {
                "school_id": res.school_id,
                "group_id": res.group_id,
                "branch_id": res.branch_id,
                "state": res.state,
                "is_global": res.is_global,
            }
            if not self.policy_filter.is_tenant_scope_applicable(cand_meta, school_context):
                logger.debug("Rejected BM25 candidate %s: tenant scope mismatch", chunk_id)
                continue

            # Employee Applicability
            emp_meta = {
                "staff_category": res.staff_category,
                "employment_type": res.employment_type,
                "department": res.department,
            }
            if not self.policy_filter.is_employee_applicable(emp_meta, understanding.employee_context):
                logger.debug("Rejected BM25 candidate %s: employee applicability mismatch", chunk_id)
                continue

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                document_name=res.document_name or doc_record.get("document_name", "Unknown Policy"),
                document_type=res.document_type or doc_record.get("document_type", "policy"),
                version=int(res.version or doc_record.get("version", 1)),
                content=res.content,
                heading=res.heading or None,
                heading_path=res.heading_path or [],
                page_start=int(res.page_start or 1),
                page_end=int(res.page_end or 1),
                similarity_score=0.0,
                dense_score=None,
                dense_rank=None,
                bm25_score=res.bm25_score,
                bm25_rank=res.bm25_rank,
                school_id=res.school_id or doc_record.get("school_id"),
                group_id=res.group_id or doc_record.get("group_id"),
                branch_id=res.branch_id or doc_record.get("branch_id"),
                effective_from=_parse_date(res.effective_from or doc_record.get("effective_from")),
                effective_to=_parse_date(res.effective_to or doc_record.get("effective_to")),
                retrieval_mode="bm25",
            )
            seen.add(chunk_id)
            valid_chunks.append(chunk)

        return valid_chunks, seen

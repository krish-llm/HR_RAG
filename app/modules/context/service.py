"""Context Service coordinating semantic retrieval, composite reranking, and context building."""

import logging
from typing import Optional

from app.api.dependencies import SchoolContext
from app.core.logging_utils import TimedBlock
from app.modules.context.builder import ContextBuilder
from app.modules.context.schemas import ContextRequest, ContextResponse
from app.modules.reranking.base import BaseReranker
from app.modules.reranking.reranker import DeterministicV1Reranker
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class ContextService:
    """Orchestrates query retrieval, deterministic reranking, and budget-constrained context construction."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        reranker: Optional[BaseReranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        default_top_k: int = 10,
        default_token_budget: int = 2000,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.reranker = reranker or DeterministicV1Reranker()
        self.context_builder = context_builder or ContextBuilder(default_token_budget=default_token_budget)
        self.default_top_k = default_top_k
        self.default_token_budget = default_token_budget

    async def build_context(
        self,
        request: ContextRequest,
        school_context: SchoolContext,
    ) -> ContextResponse:
        """Execute full context generation pipeline: Retrieval ➔ Reranking ➔ Context Assembly.

        Pipeline Steps:
        1. Query Retrieval: reuse Phase 6 RetrievalService to fetch validated candidate chunks.
        2. Reranking: apply deterministic composite reranking based on semantic, lexical, and heading alignment.
        3. Context Building: deduplicate, format structured citations, and enforce strict token budgets without truncating chunks.
        """
        # 1. Reuse existing Phase 6 semantic retrieval
        retrieval_req = RetrievalRequest(
            question=request.question,
            employee_context=request.employee_context,
            as_of_date=request.as_of_date,
            top_k=request.top_k or self.default_top_k,
        )

        retrieval_res = await self.retrieval_service.retrieve(
            request=retrieval_req,
            school_context=school_context,
        )

        budget = request.token_budget or self.default_token_budget

        if not retrieval_res.chunks:
            logger.info("Semantic retrieval yielded 0 valid chunks for question: '%s'", request.question)
            return ContextResponse(
                query_understanding=retrieval_res.query_understanding,
                ranked_chunks=[],
                assembled_context="",
                source_manifest=[],
                total_tokens=0,
                token_budget=budget,
                selected_chunks_count=0,
                skipped_chunks_count=0,
                status="empty",
            )

        # 2. Composite deterministic reranking
        with TimedBlock(logger, "reranking") as timer:
            ranked_chunks = self.reranker.rerank(
                query=retrieval_res.query_understanding.normalized_query,
                chunks=retrieval_res.chunks,
                keywords=retrieval_res.query_understanding.extracted_keywords,
            )
            timer.set_extra(candidates=len(retrieval_res.chunks), ranked=len(ranked_chunks))

        # 3. Context assembly and budget enforcement
        with TimedBlock(logger, "context_builder") as timer:
            build_result = self.context_builder.build_context(
                chunks=ranked_chunks,
                token_budget=budget,
            )
            timer.set_extra(
                selected=build_result.selected_chunks_count,
                skipped=build_result.skipped_chunks_count,
                tokens=build_result.total_tokens,
            )

        status_val = "success" if build_result.selected_chunks_count > 0 else "empty"

        return ContextResponse(
            query_understanding=retrieval_res.query_understanding,
            ranked_chunks=ranked_chunks,
            assembled_context=build_result.assembled_context,
            source_manifest=build_result.source_manifest,
            total_tokens=build_result.total_tokens,
            token_budget=build_result.token_budget,
            selected_chunks_count=build_result.selected_chunks_count,
            skipped_chunks_count=build_result.skipped_chunks_count,
            status=status_val,
        )

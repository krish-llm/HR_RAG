"""Retrieval evaluator calculating ranking quality metrics (Precision@K, Recall@K, MRR, nDCG@K)."""

import logging
from typing import Any, List, Set

from app.modules.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.modules.evaluation.schemas import CaseRetrievalMetrics, EvaluationTestCase

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates Phase 6/7 retrieval outputs against expected policy documents and chunks."""

    def __init__(self, k_values: List[int] = None) -> None:
        self.k_values = k_values or [1, 3, 5]

    def evaluate_retrieval(
        self,
        test_case: EvaluationTestCase,
        retrieved_chunks: List[Any],
    ) -> CaseRetrievalMetrics:
        """Evaluate retrieval performance for a single test case.

        Args:
            test_case: The ground truth evaluation test case.
            retrieved_chunks: List of RetrievedChunk or RankedChunk instances in returned rank order.

        Returns:
            CaseRetrievalMetrics containing Precision@K, Recall@K, ReciprocalRank, and nDCG@K.
        """
        retrieved_doc_ids: List[str] = []
        retrieved_chunk_ids: List[str] = []

        for chunk in retrieved_chunks:
            doc_id = getattr(chunk, "document_id", None)
            if doc_id is None and isinstance(chunk, dict):
                doc_id = chunk.get("document_id")
            if doc_id:
                retrieved_doc_ids.append(str(doc_id))

            chk_id = getattr(chunk, "chunk_id", None)
            if chk_id is None and isinstance(chunk, dict):
                chk_id = chunk.get("chunk_id")
            if chk_id:
                retrieved_chunk_ids.append(str(chk_id))

        # Relevant target set: prioritize document IDs, fallback to chunk IDs
        relevant_docs: Set[str] = set(test_case.expected_document_ids)

        precisions = {}
        recalls = {}
        ndcgs = {}

        for k in self.k_values:
            precisions[k] = precision_at_k(retrieved_doc_ids, relevant_docs, k)
            recalls[k] = recall_at_k(retrieved_doc_ids, relevant_docs, k)
            ndcgs[k] = ndcg_at_k(retrieved_doc_ids, relevant_docs, k)

        rr = reciprocal_rank(retrieved_doc_ids, relevant_docs)

        return CaseRetrievalMetrics(
            precision_at_k=precisions,
            recall_at_k=recalls,
            reciprocal_rank=rr,
            ndcg_at_k=ndcgs,
            retrieved_document_ids=retrieved_doc_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

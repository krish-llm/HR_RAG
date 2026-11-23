"""Evaluation module public exports."""

from app.modules.evaluation.answer_evaluator import AnswerEvaluator
from app.modules.evaluation.dataset import (
    load_evaluation_dataset,
    save_evaluation_dataset,
)
from app.modules.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_point_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.modules.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.modules.evaluation.runner import EvaluationRunner
from app.modules.evaluation.schemas import (
    AggregateAnswerMetrics,
    AggregateRetrievalMetrics,
    CaseAnswerMetrics,
    CaseEvaluationResult,
    CaseRetrievalMetrics,
    EvaluationDataset,
    EvaluationSummary,
    EvaluationTestCase,
)

__all__ = [
    "AggregateAnswerMetrics",
    "AggregateRetrievalMetrics",
    "AnswerEvaluator",
    "CaseAnswerMetrics",
    "CaseEvaluationResult",
    "CaseRetrievalMetrics",
    "EvaluationDataset",
    "EvaluationRunner",
    "EvaluationTestCase",
    "EvaluationSummary",
    "RetrievalEvaluator",
    "calculate_citation_metrics",
    "calculate_point_coverage",
    "load_evaluation_dataset",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "save_evaluation_dataset",
]

"""Pydantic v2 schemas and data contracts for RAG Evaluation & Quality Measurement (Phase 9)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.common import current_utc_time


class EvaluationTestCase(BaseModel):
    """Structured test case defining expected retrieval and answering criteria for an HR policy scenario."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Unique test case identifier (e.g. 'TC-HR-001')")
    name: str = Field(description="Short descriptive name of the test scenario")
    question: str = Field(description="Natural language user question regarding HR policies")
    school_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Canonical school context scoping filters (e.g. school_id, group_id, branch_id, state)",
    )
    employee_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Employee context attributes (staff_category, employment_type, department)",
    )
    as_of_date: Optional[datetime] = Field(
        default=None,
        description="Evaluation date for policy effectiveness verification",
    )
    expected_document_ids: List[str] = Field(
        default_factory=list,
        description="List of authoritative document IDs expected to be retrieved",
    )
    expected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of specific chunk IDs expected to be retrieved",
    )
    expected_answer_points: List[str] = Field(
        default_factory=list,
        description="Key factual assertions expected in the generated answer",
    )
    expected_citations: List[str] = Field(
        default_factory=list,
        description="Expected citation references or document names cited in the answer",
    )
    expect_insufficient_evidence: bool = Field(
        default=False,
        description="Whether the expected correct behavior is to abstain due to insufficient evidence",
    )
    expect_conflicting_evidence: bool = Field(
        default=False,
        description="Whether conflicting policy provisions across documents are expected",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Categorization tags (e.g. ['leave', 'medical', 'conflict', 'boundary_case'])",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Guidance for human reviewers or rationale for the test case",
    )


class EvaluationDataset(BaseModel):
    """Versioned collection of HR policy evaluation test cases."""

    model_config = ConfigDict(populate_by_name=True)

    version: str = Field(default="1.0.0", description="Semantic dataset schema/revision version")
    description: str = Field(description="Description of the dataset domain, coverage, and purpose")
    created_at: datetime = Field(default_factory=current_utc_time)
    test_cases: List[EvaluationTestCase] = Field(
        default_factory=list,
        description="List of evaluation test cases",
    )


class CaseRetrievalMetrics(BaseModel):
    """Retrieval quality metrics for a single test case across configured K cutoffs."""

    model_config = ConfigDict(populate_by_name=True)

    precision_at_k: Dict[int, float] = Field(
        default_factory=dict,
        description="Precision at rank cutoffs (e.g. {1: 1.0, 3: 0.67, 5: 0.4})",
    )
    recall_at_k: Dict[int, float] = Field(
        default_factory=dict,
        description="Recall at rank cutoffs (e.g. {1: 0.5, 3: 1.0, 5: 1.0})",
    )
    reciprocal_rank: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reciprocal rank of the first relevant document/chunk (1 / rank)",
    )
    ndcg_at_k: Dict[int, float] = Field(
        default_factory=dict,
        description="Normalized Discounted Cumulative Gain at rank cutoffs (e.g. {3: 0.92, 5: 0.95})",
    )
    retrieved_document_ids: List[str] = Field(
        default_factory=list,
        description="List of document IDs actually retrieved and ranked",
    )
    retrieved_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs actually retrieved and ranked",
    )


class CaseAnswerMetrics(BaseModel):
    """Answer generation and citation quality metrics for a single test case."""

    model_config = ConfigDict(populate_by_name=True)

    point_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of expected factual points covered in generated answer",
    )
    matched_points: List[str] = Field(
        default_factory=list,
        description="List of expected points verified in the answer",
    )
    missing_points: List[str] = Field(
        default_factory=list,
        description="List of expected points not found in the answer",
    )
    citation_validity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of cited sources that strictly match the authoritative context manifest",
    )
    citation_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of expected citations present in the actual citations",
    )
    unsupported_citations: List[str] = Field(
        default_factory=list,
        description="List of citation tags cited by the model that were ungrounded or pruned",
    )
    insufficient_evidence_correct: bool = Field(
        default=True,
        description="True if actual insufficient_evidence matches expected abstention behavior",
    )
    conflicting_evidence_correct: bool = Field(
        default=True,
        description="True if actual conflicting_evidence matches expected conflict detection",
    )
    human_review_recommended: bool = Field(
        default=False,
        description="Flag indicating that manual human review is advised (e.g. partial point coverage)",
    )


class CaseEvaluationResult(BaseModel):
    """Comprehensive evaluation record for an individual test case."""

    model_config = ConfigDict(populate_by_name=True)

    case_id: str = Field(description="Test case identifier")
    name: str = Field(description="Descriptive name")
    question: str = Field(description="Query question evaluated")
    passed: bool = Field(description="Whether the case satisfied all pass/fail evaluation criteria")
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for why the case failed",
    )
    retrieval_metrics: CaseRetrievalMetrics = Field(description="Computed retrieval metrics")
    answer_metrics: Optional[CaseAnswerMetrics] = Field(
        default=None,
        description="Computed answer generation metrics (if generation was evaluated)",
    )
    actual_answer: Optional[str] = Field(default=None, description="Actual answer text generated")
    actual_citations: List[str] = Field(default_factory=list, description="Actual citation references")
    actual_insufficient_evidence: bool = Field(default=False)
    actual_conflicting_evidence: bool = Field(default=False)
    actual_limitations: Optional[str] = Field(default=None)
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0, description="Retrieval latency in milliseconds")
    generation_latency_ms: float = Field(default=0.0, ge=0.0, description="Generation latency in milliseconds")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total execution latency in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Exception message if execution faulted")


class AggregateRetrievalMetrics(BaseModel):
    """Macro-averaged retrieval metrics across all evaluated test cases."""

    model_config = ConfigDict(populate_by_name=True)

    mean_precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank across all queries")
    mean_ndcg_at_k: Dict[int, float] = Field(default_factory=dict)


class AggregateAnswerMetrics(BaseModel):
    """Macro-averaged answer and citation metrics across evaluated test cases."""

    model_config = ConfigDict(populate_by_name=True)

    mean_point_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_citation_validity: float = Field(default=1.0, ge=0.0, le=1.0)
    mean_citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    insufficient_evidence_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    conflicting_evidence_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_claims_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationSummary(BaseModel):
    """Overall evaluation summary report aggregating performance across all test cases."""

    model_config = ConfigDict(populate_by_name=True)

    dataset_version: str = Field(description="Evaluation dataset version tested")
    timestamp: datetime = Field(default_factory=current_utc_time)
    total_cases: int = Field(ge=0, description="Total number of evaluated test cases")
    passed_cases: int = Field(ge=0, description="Number of test cases that passed")
    failed_cases: int = Field(ge=0, description="Number of test cases that failed")
    pass_rate: float = Field(ge=0.0, le=1.0, description="Proportion of passing test cases (0.0 to 1.0)")
    retrieval_summary: AggregateRetrievalMetrics = Field(description="Aggregated retrieval performance")
    answer_summary: Optional[AggregateAnswerMetrics] = Field(
        default=None,
        description="Aggregated generation performance (if answer evaluation was enabled)",
    )
    mean_latency_ms: float = Field(ge=0.0, description="Mean total latency per test case in ms")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile latency in ms")
    case_results: List[CaseEvaluationResult] = Field(
        default_factory=list,
        description="Granular result per test case",
    )

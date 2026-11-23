"""Unit and integration tests for Phase 9 RAG Evaluation & Quality Measurement."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.core.exceptions import AppException
from app.modules.context.schemas import ContextResponse, SourceCitation
from app.modules.context.service import ContextService
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
    EvaluationDataset,
    EvaluationTestCase,
)
from app.modules.generation.schemas import AnswerResponse, CitationItem
from app.modules.generation.service import AnswerService
from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.schemas import QueryUnderstandingResult, RetrievedChunk


# ==============================================================================
# 1. Dataset Loading and Schema Validation Tests
# ==============================================================================


def test_load_default_dataset_v1():
    """Verify that the bundled evaluation dataset v1 loads and validates successfully."""
    dataset_path = Path("data/evaluation/dataset_v1.json")
    dataset = load_evaluation_dataset(dataset_path)

    assert dataset.version == "1.0.0"
    assert len(dataset.test_cases) >= 8
    # Verify first test case has all required fields
    tc = dataset.test_cases[0]
    assert tc.id == "TC-HR-001"
    assert tc.question
    assert tc.expected_document_ids
    assert tc.expected_answer_points


def test_dataset_file_not_found_raises_app_exception():
    """Verify loading non-existent dataset raises 404 AppException."""
    with pytest.raises(AppException) as exc_info:
        load_evaluation_dataset("data/evaluation/non_existent.json")

    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "DATASET_NOT_FOUND"


def test_dataset_invalid_schema_raises_app_exception():
    """Verify loading malformed dataset raises 422 AppException."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write('{"version": "1.0.0", "test_cases": [{"id": 123}]}')  # id should be str
        temp_path = f.name

    try:
        with pytest.raises(AppException) as exc_info:
            load_evaluation_dataset(temp_path)
        assert exc_info.value.status_code == 422
        assert exc_info.value.error_code == "DATASET_VALIDATION_ERROR"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_save_and_reload_dataset():
    """Verify saving a dataset and re-reading it preserves full fidelity."""
    tc = EvaluationTestCase(
        id="TC-CUSTOM-001",
        name="Custom Test Case",
        question="What is the casual leave policy?",
        expected_document_ids=["doc-custom-1"],
        expected_answer_points=["12 days annual"],
        expected_citations=["[Source 1]"],
    )
    dataset = EvaluationDataset(
        version="2.0.0",
        description="Custom test dataset",
        test_cases=[tc],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "custom_dataset.json"
        save_evaluation_dataset(dataset, save_path)
        loaded = load_evaluation_dataset(save_path)

        assert loaded.version == "2.0.0"
        assert len(loaded.test_cases) == 1
        assert loaded.test_cases[0].id == "TC-CUSTOM-001"


# ==============================================================================
# 2. Metric Calculations Tests
# ==============================================================================


def test_precision_at_k_calculation():
    """Verify Precision@K calculation using known rankings."""
    relevant = {"doc-A", "doc-B"}

    # Case 1: First 2 are relevant in top-3 -> 2/3
    retrieved = ["doc-A", "doc-B", "doc-C", "doc-D"]
    assert precision_at_k(retrieved, relevant, k=1) == 1.0  # 1/1
    assert precision_at_k(retrieved, relevant, k=2) == 1.0  # 2/2
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3, 0.001)  # 0.6667
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # 2/4

    # Case 2: No relevant in top-k
    retrieved_irrelevant = ["doc-X", "doc-Y", "doc-Z"]
    assert precision_at_k(retrieved_irrelevant, relevant, k=3) == 0.0

    # Case 3: Empty relevant target (unaddressed query)
    assert precision_at_k([], set(), k=3) == 1.0
    assert precision_at_k(["doc-X"], set(), k=3) == 0.0


def test_recall_at_k_calculation():
    """Verify Recall@K calculation."""
    relevant = {"doc-A", "doc-B", "doc-C"}

    retrieved = ["doc-A", "doc-X", "doc-B"]
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3, 0.001)  # 0.3333
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3, 0.001)  # 0.3333
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3, 0.001)  # 0.6667

    # Empty target set
    assert recall_at_k(retrieved, set(), k=3) == 1.0


def test_reciprocal_rank_calculation():
    """Verify Reciprocal Rank calculation."""
    relevant = {"doc-target"}

    assert reciprocal_rank(["doc-target", "doc-other"], relevant) == 1.0  # 1/1
    assert reciprocal_rank(["doc-1", "doc-target"], relevant) == 0.5      # 1/2
    assert reciprocal_rank(["doc-1", "doc-2", "doc-target"], relevant) == pytest.approx(1 / 3, 0.001)
    assert reciprocal_rank(["doc-1", "doc-2"], relevant) == 0.0          # Not found


def test_ndcg_at_k_calculation():
    """Verify nDCG@K calculation with binary relevance."""
    relevant = {"doc-A", "doc-B"}

    # Perfect ranking: both relevant items at rank 1 and 2
    perfect = ["doc-A", "doc-B", "doc-C"]
    assert ndcg_at_k(perfect, relevant, k=3) == 1.0

    # Degraded ranking: relevant items at rank 2 and 3
    degraded = ["doc-C", "doc-A", "doc-B"]
    ndcg_val = ndcg_at_k(degraded, relevant, k=3)
    assert 0.0 < ndcg_val < 1.0

    # Zero relevant items retrieved
    none_found = ["doc-X", "doc-Y"]
    assert ndcg_at_k(none_found, relevant, k=2) == 0.0


def test_point_coverage_calculation():
    """Verify factual answer point coverage check."""
    expected_points = [
        "12 days casual leave",
        "credited at beginning of year",
        "requires prior supervisor notification",
    ]

    answer = "Employees are entitled to 12 days casual leave credited at the beginning of the year."
    cov, matched, missing = calculate_point_coverage(answer, expected_points)

    assert pytest.approx(cov, 0.01) == 0.67
    assert len(matched) == 2
    assert len(missing) == 1
    assert "requires prior supervisor notification" in missing


def test_citation_metrics_calculation():
    """Verify citation validity, coverage, and unsupported tag detection."""
    manifest_labels = {"[Source 1]", "[Source 2]"}
    expected_citations = ["[Source 1]", "[Source 2]"]

    # LLM cites valid [Source 1] plus fabricated [Source 99]
    actual = ["[Source 1]", "[Source 99]"]
    validity, coverage, unsupported = calculate_citation_metrics(actual, expected_citations, manifest_labels)

    assert validity == 0.5  # 1 valid out of 2 cited
    assert coverage == 0.5  # 1 expected out of 2 expected
    assert unsupported == ["[Source 99]"]


# ==============================================================================
# 3. Evaluator Modules Tests
# ==============================================================================


def test_retrieval_evaluator():
    """Verify RetrievalEvaluator scores rankings across cutoffs."""
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    tc = EvaluationTestCase(
        id="TC-1",
        name="Test",
        question="Query",
        expected_document_ids=["doc-1", "doc-2"],
    )

    retrieved = [
        RetrievedChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            document_name="Doc 1",
            content="content",
            page_start=1,
            page_end=1,
            similarity_score=0.9,
        ),
        RetrievedChunk(
            chunk_id="chk-2",
            document_id="doc-other",
            document_name="Doc Other",
            content="content",
            page_start=1,
            page_end=1,
            similarity_score=0.8,
        ),
        RetrievedChunk(
            chunk_id="chk-3",
            document_id="doc-2",
            document_name="Doc 2",
            content="content",
            page_start=1,
            page_end=1,
            similarity_score=0.7,
        ),
    ]

    metrics = evaluator.evaluate_retrieval(tc, retrieved)
    assert metrics.precision_at_k[1] == 1.0
    assert metrics.recall_at_k[1] == 0.5
    assert metrics.reciprocal_rank == 1.0
    assert metrics.recall_at_k[3] == 1.0
    assert metrics.ndcg_at_k[3] > 0.8


def test_answer_evaluator_conflicting_and_insufficient_evidence():
    """Verify AnswerEvaluator correctly scores abstention and conflict detection."""
    evaluator = AnswerEvaluator()

    # Case A: Insufficient evidence expected & correctly abstained
    tc_insufficient = EvaluationTestCase(
        id="TC-INSUFF",
        name="Insufficient test",
        question="Sabbatical query?",
        expect_insufficient_evidence=True,
    )
    ans_metrics = evaluator.evaluate_answer(
        test_case=tc_insufficient,
        actual_answer="Policies do not provide information.",
        actual_citations=[],
        actual_insufficient_evidence=True,
        actual_conflicting_evidence=False,
    )
    assert ans_metrics.insufficient_evidence_correct is True
    assert ans_metrics.point_coverage == 1.0

    # Case B: Conflicting evidence expected & correctly detected
    tc_conflict = EvaluationTestCase(
        id="TC-CONF",
        name="Conflict test",
        question="Medical certificate query?",
        expected_answer_points=["conflict between policies"],
        expect_conflicting_evidence=True,
    )
    ans_metrics_conf = evaluator.evaluate_answer(
        test_case=tc_conflict,
        actual_answer="There is a conflict between policies on medical certificates.",
        actual_citations=["[Source 1]"],
        actual_insufficient_evidence=False,
        actual_conflicting_evidence=True,
        manifest_labels={"[Source 1]"},
    )
    assert ans_metrics_conf.conflicting_evidence_correct is True
    assert ans_metrics_conf.point_coverage == 1.0


# ==============================================================================
# 4. Runner & Export Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_evaluation_runner_end_to_end_mock():
    """Verify EvaluationRunner executes dataset, computes aggregates, and exports reports."""
    tc1 = EvaluationTestCase(
        id="TC-E2E-001",
        name="Casual Leave",
        question="How many days casual leave?",
        expected_document_ids=["doc-leave-001"],
        expected_answer_points=["12 days casual leave"],
        expected_citations=["[Source 1]"],
    )
    tc2 = EvaluationTestCase(
        id="TC-E2E-002",
        name="Sabbatical Abstain",
        question="Sabbatical leave policy?",
        expect_insufficient_evidence=True,
    )
    dataset = EvaluationDataset(
        version="1.0-test",
        description="E2E test dataset",
        test_cases=[tc1, tc2],
    )

    mock_context_svc = MagicMock(spec=ContextService)
    mock_context_svc.build_context = AsyncMock(
        return_value=ContextResponse(
            query_understanding=QueryUnderstandingResult(
                original_query="q", normalized_query="q", intent="leave"
            ),
            ranked_chunks=[
                RankedChunk(
                    chunk_id="chk-1",
                    document_id="doc-leave-001",
                    document_name="Leave Policy",
                    document_type="policy",
                    version=1,
                    content="12 days casual leave",
                    page_start=1,
                    page_end=1,
                    similarity_score=0.9,
                    rerank_score=0.9,
                    lexical_score=0.9,
                    heading_score=0.9,
                    original_rank=1,
                    rerank_rank=1,
                )
            ],
            assembled_context="...",
            source_manifest=[
                SourceCitation(
                    citation_index=1,
                    citation_label="[Source 1]",
                    citation_reference="[Source 1: Policy]",
                    document_id="doc-leave-001",
                    document_name="Leave Policy",
                    version=1,
                    chunk_id="chk-1",
                    page_start=1,
                    page_end=1,
                )
            ],
            total_tokens=30,
            token_budget=2000,
            selected_chunks_count=1,
            skipped_chunks_count=0,
            status="success",
        )
    )

    mock_answer_svc = MagicMock(spec=AnswerService)

    async def mock_generate(req, ctx):
        if "sabbatical" in req.question.lower():
            return AnswerResponse(
                question=req.question,
                answer="No policy available.",
                citations=[],
                insufficient_evidence=True,
                conflicting_evidence=False,
                context_tokens=0,
                model_name="mock-gpt",
                status="insufficient_evidence",
            )
        return AnswerResponse(
            question=req.question,
            answer="Employees receive 12 days casual leave [Source 1].",
            citations=[
                CitationItem(
                    citation_label="[Source 1]",
                    citation_reference="[Source 1: Leave Policy]",
                    document_id="doc-leave-001",
                    document_name="Leave Policy",
                    version=1,
                    page_start=1,
                    page_end=1,
                )
            ],
            insufficient_evidence=False,
            conflicting_evidence=False,
            context_tokens=30,
            model_name="mock-gpt",
            status="success",
        )

    mock_answer_svc.generate_answer = mock_generate

    runner = EvaluationRunner(
        context_service=mock_context_svc,
        answer_service=mock_answer_svc,
    )

    summary = await runner.run_dataset(dataset, evaluate_answer=True)

    assert summary.total_cases == 2
    assert summary.passed_cases == 2
    assert summary.failed_cases == 0
    assert summary.pass_rate == 1.0
    assert summary.retrieval_summary.mrr > 0.0
    assert summary.answer_summary.insufficient_evidence_accuracy == 1.0
    assert summary.answer_summary.mean_point_coverage == 1.0

    # Test JSON and CSV export
    with tempfile.TemporaryDirectory() as tmpdir:
        json_out = Path(tmpdir) / "report.json"
        csv_out = Path(tmpdir) / "cases.csv"

        runner.export_json(summary, json_out)
        runner.export_csv(summary, csv_out)

        assert json_out.is_file()
        assert csv_out.is_file()

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_cases"] == 2
            assert data["pass_rate"] == 1.0

        with open(csv_out, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3  # Header + 2 cases


@pytest.mark.asyncio
async def test_evaluation_runner_failure_handling():
    """Verify that exceptions during evaluation are trapped and recorded as failures without crashing."""
    tc = EvaluationTestCase(
        id="TC-FAULT-001",
        name="Faulty Case",
        question="Question",
        expected_document_ids=["doc-expected"],
    )
    dataset = EvaluationDataset(version="1.0", description="Fault test", test_cases=[tc])

    mock_context_svc = MagicMock(spec=ContextService)
    mock_context_svc.build_context = AsyncMock(side_effect=RuntimeError("Database failure"))

    runner = EvaluationRunner(context_service=mock_context_svc)
    summary = await runner.run_dataset(dataset, evaluate_answer=False)

    assert summary.total_cases == 1
    assert summary.failed_cases == 1
    assert summary.passed_cases == 0
    assert summary.pass_rate == 0.0
    case_res = summary.case_results[0]
    assert case_res.passed is False
    assert any("Database failure" in r for r in case_res.failure_reasons)

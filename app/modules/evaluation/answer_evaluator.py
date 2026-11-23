"""Answer evaluator calculating factual point coverage, citation validity, and evidence detection."""

import logging
from typing import Any, List, Optional, Set

from app.modules.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_point_coverage,
)
from app.modules.evaluation.schemas import CaseAnswerMetrics, EvaluationTestCase

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """Evaluates Phase 8 generated answers against expected points, citations, and evidence conditions."""

    def evaluate_answer(
        self,
        test_case: EvaluationTestCase,
        actual_answer: str,
        actual_citations: List[Any],
        actual_insufficient_evidence: bool,
        actual_conflicting_evidence: bool,
        manifest_labels: Optional[Set[str]] = None,
    ) -> CaseAnswerMetrics:
        """Evaluate generation quality against ground-truth test case expectations.

        Args:
            test_case: The ground truth evaluation test case.
            actual_answer: The answer string produced by AnswerService.
            actual_citations: List of CitationItem objects or label strings.
            actual_insufficient_evidence: Boolean flag produced by AnswerService.
            actual_conflicting_evidence: Boolean flag produced by AnswerService.
            manifest_labels: Set of valid citation labels from Phase 7 context manifest.

        Returns:
            CaseAnswerMetrics with point coverage, citation validity, and detection accuracy.
        """
        # 1. Point Coverage
        # If test case expects insufficient evidence and system successfully abstained,
        # point coverage is considered complete (1.0)
        if test_case.expect_insufficient_evidence and actual_insufficient_evidence:
            point_cov = 1.0
            matched_pts = list(test_case.expected_answer_points)
            missing_pts = []
        else:
            point_cov, matched_pts, missing_pts = calculate_point_coverage(
                answer=actual_answer,
                expected_points=test_case.expected_answer_points,
            )

        # 2. Citation Metrics
        manifest = manifest_labels or set()
        cit_validity, cit_coverage, unsupported = calculate_citation_metrics(
            actual_citations=actual_citations,
            expected_citations=test_case.expected_citations,
            manifest_labels=manifest,
        )

        # 3. Evidence State Detection
        insufficient_correct = bool(actual_insufficient_evidence == test_case.expect_insufficient_evidence)
        conflicting_correct = bool(actual_conflicting_evidence == test_case.expect_conflicting_evidence)

        # 4. Human Review Recommendation
        # Automatic metrics distinguish complete matches from partial matches where expert review is needed
        human_review = (
            (0.0 < point_cov < 1.0)
            or bool(unsupported)
            or not insufficient_correct
            or not conflicting_correct
        )

        return CaseAnswerMetrics(
            point_coverage=point_cov,
            matched_points=matched_pts,
            missing_points=missing_pts,
            citation_validity=cit_validity,
            citation_coverage=cit_coverage,
            unsupported_citations=unsupported,
            insufficient_evidence_correct=insufficient_correct,
            conflicting_evidence_correct=conflicting_correct,
            human_review_recommended=human_review,
        )

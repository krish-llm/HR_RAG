"""Evaluation Runner orchestrating automated benchmark evaluation and report export."""

import csv
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Set
import numpy as np

from app.api.dependencies import SchoolContext
from app.modules.context.service import ContextService
from app.modules.evaluation.answer_evaluator import AnswerEvaluator
from app.modules.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.modules.evaluation.schemas import (
    AggregateAnswerMetrics,
    AggregateRetrievalMetrics,
    CaseEvaluationResult,
    EvaluationDataset,
    EvaluationSummary,
    EvaluationTestCase,
)
from app.modules.generation.schemas import AnswerRequest
from app.modules.generation.service import AnswerService
from app.modules.retrieval.schemas import EmployeeContext, RetrievalRequest
from app.modules.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates test case execution against retrieval and answering services."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        context_service: Optional[ContextService] = None,
        answer_service: Optional[AnswerService] = None,
        k_values: Optional[List[int]] = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.context_service = context_service
        self.answer_service = answer_service
        self.k_values = k_values or [1, 3, 5]
        self.retrieval_evaluator = RetrievalEvaluator(k_values=self.k_values)
        self.answer_evaluator = AnswerEvaluator()

    async def run_single_case(
        self,
        test_case: EvaluationTestCase,
        evaluate_answer: bool = True,
    ) -> CaseEvaluationResult:
        """Execute and score a single evaluation test case."""
        school_ctx_dict = test_case.school_context or {}
        school_ctx = SchoolContext(
            school_id=school_ctx_dict.get("school_id"),
            group_id=school_ctx_dict.get("group_id"),
            branch_id=school_ctx_dict.get("branch_id"),
            state=school_ctx_dict.get("state"),
        )

        emp_ctx_dict = test_case.employee_context
        emp_ctx = None
        if emp_ctx_dict:
            emp_ctx = EmployeeContext(
                staff_category=emp_ctx_dict.get("staff_category"),
                employment_type=emp_ctx_dict.get("employment_type"),
                department=emp_ctx_dict.get("department"),
            )

        retrieval_start = time.perf_counter()
        retrieval_latency = 0.0
        generation_latency = 0.0
        failure_reasons: List[str] = []

        retrieved_chunks = []
        manifest_labels: Set[str] = set()

        # Step 1: Retrieval Evaluation
        try:
            if self.context_service is not None:
                # Use context service to get reranked chunks and manifest
                from app.modules.context.schemas import ContextRequest
                ctx_req = ContextRequest(
                    question=test_case.question,
                    employee_context=emp_ctx,
                    as_of_date=test_case.as_of_date,
                    top_k=max(self.k_values),
                )
                ctx_res = await self.context_service.build_context(ctx_req, school_ctx)
                retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
                retrieved_chunks = ctx_res.ranked_chunks
                manifest_labels = {c.citation_label for c in ctx_res.source_manifest}
            elif self.retrieval_service is not None:
                ret_req = RetrievalRequest(
                    question=test_case.question,
                    employee_context=emp_ctx,
                    as_of_date=test_case.as_of_date,
                    top_k=max(self.k_values),
                )
                ret_res = await self.retrieval_service.retrieve(ret_req, school_ctx)
                retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
                retrieved_chunks = ret_res.chunks
            else:
                retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
        except Exception as exc:
            retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
            failure_reasons.append(f"Retrieval stage raised exception: {exc}")

        ret_metrics = self.retrieval_evaluator.evaluate_retrieval(test_case, retrieved_chunks)

        # Check retrieval pass criteria if documents were expected
        if test_case.expected_document_ids:
            # If documents were expected but none matched in top-5
            if ret_metrics.reciprocal_rank == 0.0:
                failure_reasons.append(f"Expected documents {test_case.expected_document_ids} not found in retrieved results")

        # Step 2: Answer Generation Evaluation (if enabled and answer_service available)
        ans_metrics = None
        actual_answer = None
        actual_citations = []
        actual_insufficient = False
        actual_conflicting = False
        actual_limitations = None

        if evaluate_answer and self.answer_service is not None:
            gen_start = time.perf_counter()
            try:
                ans_req = AnswerRequest(
                    question=test_case.question,
                    employee_context=emp_ctx,
                    as_of_date=test_case.as_of_date,
                    top_k=max(self.k_values),
                )
                ans_res = await self.answer_service.generate_answer(ans_req, school_ctx)
                generation_latency = (time.perf_counter() - gen_start) * 1000

                actual_answer = ans_res.answer
                actual_citations = [c.citation_label for c in ans_res.citations]
                actual_insufficient = ans_res.insufficient_evidence
                actual_conflicting = ans_res.conflicting_evidence
                actual_limitations = ans_res.limitations

                ans_metrics = self.answer_evaluator.evaluate_answer(
                    test_case=test_case,
                    actual_answer=actual_answer,
                    actual_citations=ans_res.citations,
                    actual_insufficient_evidence=actual_insufficient,
                    actual_conflicting_evidence=actual_conflicting,
                    manifest_labels=manifest_labels,
                )

                if not ans_metrics.insufficient_evidence_correct:
                    failure_reasons.append(
                        f"Insufficient evidence mismatch: expected {test_case.expect_insufficient_evidence}, got {actual_insufficient}"
                    )
                if not ans_metrics.conflicting_evidence_correct:
                    failure_reasons.append(
                        f"Conflicting evidence mismatch: expected {test_case.expect_conflicting_evidence}, got {actual_conflicting}"
                    )
                if not test_case.expect_insufficient_evidence and ans_metrics.point_coverage < 0.50:
                    failure_reasons.append(
                        f"Low expected points coverage: {ans_metrics.point_coverage * 100:.1f}% (missing: {ans_metrics.missing_points})"
                    )
                if ans_metrics.unsupported_citations:
                    failure_reasons.append(
                        f"Unsupported/fabricated citations detected: {ans_metrics.unsupported_citations}"
                    )

            except Exception as exc:
                generation_latency = (time.perf_counter() - gen_start) * 1000
                failure_reasons.append(f"Answer generation raised exception: {exc}")

        total_latency = retrieval_latency + generation_latency
        passed = (len(failure_reasons) == 0)

        return CaseEvaluationResult(
            case_id=test_case.id,
            name=test_case.name,
            question=test_case.question,
            passed=passed,
            failure_reasons=failure_reasons,
            retrieval_metrics=ret_metrics,
            answer_metrics=ans_metrics,
            actual_answer=actual_answer,
            actual_citations=actual_citations,
            actual_insufficient_evidence=actual_insufficient,
            actual_conflicting_evidence=actual_conflicting,
            actual_limitations=actual_limitations,
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            total_latency_ms=round(total_latency, 2),
            error_message="; ".join(failure_reasons) if failure_reasons else None,
        )

    async def run_dataset(
        self,
        dataset: EvaluationDataset,
        evaluate_answer: bool = True,
    ) -> EvaluationSummary:
        """Run all test cases in the dataset and compile aggregate performance report."""
        case_results: List[CaseEvaluationResult] = []

        for case in dataset.test_cases:
            res = await self.run_single_case(case, evaluate_answer=evaluate_answer)
            case_results.append(res)

        total_cases = len(case_results)
        passed_cases = sum(1 for r in case_results if r.passed)
        failed_cases = total_cases - passed_cases
        pass_rate = round(passed_cases / total_cases, 4) if total_cases > 0 else 0.0

        # Aggregate retrieval metrics
        mean_p_at_k: Dict[int, float] = {}
        mean_r_at_k: Dict[int, float] = {}
        mean_ndcg_at_k: Dict[int, float] = {}
        reciprocal_ranks = [r.retrieval_metrics.reciprocal_rank for r in case_results]
        mrr = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0

        for k in self.k_values:
            p_vals = [r.retrieval_metrics.precision_at_k.get(k, 0.0) for r in case_results]
            r_vals = [r.retrieval_metrics.recall_at_k.get(k, 0.0) for r in case_results]
            ndcg_vals = [r.retrieval_metrics.ndcg_at_k.get(k, 0.0) for r in case_results]
            mean_p_at_k[k] = round(float(np.mean(p_vals)), 4) if p_vals else 0.0
            mean_r_at_k[k] = round(float(np.mean(r_vals)), 4) if r_vals else 0.0
            mean_ndcg_at_k[k] = round(float(np.mean(ndcg_vals)), 4) if ndcg_vals else 0.0

        retrieval_summary = AggregateRetrievalMetrics(
            mean_precision_at_k=mean_p_at_k,
            mean_recall_at_k=mean_r_at_k,
            mrr=mrr,
            mean_ndcg_at_k=mean_ndcg_at_k,
        )

        # Aggregate answer metrics (if evaluated)
        answer_summary = None
        ans_results = [r.answer_metrics for r in case_results if r.answer_metrics is not None]
        if ans_results:
            point_covs = [m.point_coverage for m in ans_results]
            cit_valids = [m.citation_validity for m in ans_results]
            cit_covs = [m.citation_coverage for m in ans_results]
            ins_accs = [1.0 if m.insufficient_evidence_correct else 0.0 for m in ans_results]
            conf_accs = [1.0 if m.conflicting_evidence_correct else 0.0 for m in ans_results]
            unsupported_rates = [1.0 if m.unsupported_citations else 0.0 for m in ans_results]

            answer_summary = AggregateAnswerMetrics(
                mean_point_coverage=round(float(np.mean(point_covs)), 4) if point_covs else 0.0,
                mean_citation_validity=round(float(np.mean(cit_valids)), 4) if cit_valids else 1.0,
                mean_citation_coverage=round(float(np.mean(cit_covs)), 4) if cit_covs else 0.0,
                insufficient_evidence_accuracy=round(float(np.mean(ins_accs)), 4) if ins_accs else 1.0,
                conflicting_evidence_accuracy=round(float(np.mean(conf_accs)), 4) if conf_accs else 1.0,
                unsupported_claims_rate=round(float(np.mean(unsupported_rates)), 4) if unsupported_rates else 0.0,
            )

        # Latencies
        total_latencies = [r.total_latency_ms for r in case_results]
        mean_lat = round(float(np.mean(total_latencies)), 2) if total_latencies else 0.0
        p95_lat = round(float(np.percentile(total_latencies, 95)), 2) if total_latencies else 0.0

        return EvaluationSummary(
            dataset_version=dataset.version,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=pass_rate,
            retrieval_summary=retrieval_summary,
            answer_summary=answer_summary,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            case_results=case_results,
        )

    def export_json(self, summary: EvaluationSummary, output_path: Path) -> None:
        """Export evaluation summary to JSON format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(mode="json"), f, indent=2, default=str)
        logger.info("Exported JSON evaluation report to %s", output_path)

    def export_csv(self, summary: EvaluationSummary, output_path: Path) -> None:
        """Export case-level evaluation results to CSV format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "case_id",
            "name",
            "passed",
            "p@1",
            "p@3",
            "p@5",
            "r@1",
            "r@3",
            "r@5",
            "mrr",
            "ndcg@3",
            "ndcg@5",
            "point_coverage",
            "citation_validity",
            "citation_coverage",
            "insufficient_correct",
            "conflict_correct",
            "retrieval_ms",
            "generation_ms",
            "total_latency_ms",
            "failure_reasons",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for c in summary.case_results:
                ret = c.retrieval_metrics
                ans = c.answer_metrics

                p1 = ret.precision_at_k.get(1, 0.0)
                p3 = ret.precision_at_k.get(3, 0.0)
                p5 = ret.precision_at_k.get(5, 0.0)
                r1 = ret.recall_at_k.get(1, 0.0)
                r3 = ret.recall_at_k.get(3, 0.0)
                r5 = ret.recall_at_k.get(5, 0.0)
                rr = ret.reciprocal_rank
                ndcg3 = ret.ndcg_at_k.get(3, 0.0)
                ndcg5 = ret.ndcg_at_k.get(5, 0.0)

                pt_cov = ans.point_coverage if ans else 0.0
                cit_val = ans.citation_validity if ans else 1.0
                cit_cov = ans.citation_coverage if ans else 0.0
                ins_cor = ans.insufficient_evidence_correct if ans else True
                conf_cor = ans.conflicting_evidence_correct if ans else True

                writer.writerow([
                    c.case_id,
                    c.name,
                    c.passed,
                    p1,
                    p3,
                    p5,
                    r1,
                    r3,
                    r5,
                    rr,
                    ndcg3,
                    ndcg5,
                    pt_cov,
                    cit_val,
                    cit_cov,
                    ins_cor,
                    conf_cor,
                    c.retrieval_latency_ms,
                    c.generation_latency_ms,
                    c.total_latency_ms,
                    " | ".join(c.failure_reasons),
                ])

        logger.info("Exported CSV evaluation report to %s", output_path)

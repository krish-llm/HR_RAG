"""Command-line interface to execute the evaluation benchmark against the HR Policy RAG system."""

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

from app.modules.evaluation.dataset import load_evaluation_dataset
from app.modules.evaluation.runner import EvaluationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluation.cli")


def _print_summary_table(summary) -> None:
    """Print a clean CLI summary table of evaluation results."""
    print("\n" + "=" * 70)
    print("           HR POLICY RAG — EVALUATION BENCHMARK REPORT")
    print("=" * 70)
    print(f"Dataset Version   : {summary.dataset_version}")
    print(f"Timestamp         : {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Total Test Cases  : {summary.total_cases}")
    print(f"Passed Cases      : {summary.passed_cases} ({summary.pass_rate * 100:.1f}%)")
    print(f"Failed Cases      : {summary.failed_cases}")
    print(f"Latency (Mean/P95): {summary.mean_latency_ms:.1f} ms / {summary.p95_latency_ms:.1f} ms")

    print("\n--- RETRIEVAL METRICS ---")
    ret = summary.retrieval_summary
    print(f"MRR (Mean Reciprocal Rank) : {ret.mrr:.4f}")
    for k in sorted(ret.mean_precision_at_k.keys()):
        p = ret.mean_precision_at_k[k]
        r = ret.mean_recall_at_k[k]
        ndcg = ret.mean_ndcg_at_k.get(k, 0.0)
        print(f"  @K={k:02d} | Precision: {p:.4f} | Recall: {r:.4f} | nDCG: {ndcg:.4f}")

    if summary.answer_summary:
        ans = summary.answer_summary
        print("\n--- ANSWER & QUALITY METRICS ---")
        print(f"Mean Point Coverage         : {ans.mean_point_coverage * 100:.1f}%")
        print(f"Citation Validity           : {ans.mean_citation_validity * 100:.1f}%")
        print(f"Citation Coverage           : {ans.mean_citation_coverage * 100:.1f}%")
        print(f"Insufficient Evid. Accuracy : {ans.insufficient_evidence_accuracy * 100:.1f}%")
        print(f"Conflicting Evid. Accuracy  : {ans.conflicting_evidence_accuracy * 100:.1f}%")
        print(f"Unsupported Claims Rate     : {ans.unsupported_claims_rate * 100:.1f}%")

    if summary.failed_cases > 0:
        print("\n--- FAILED TEST CASES ---")
        for case in summary.case_results:
            if not case.passed:
                print(f"\n[FAIL] {case.case_id}: {case.name}")
                print(f"  Question: {case.question}")
                for reason in case.failure_reasons:
                    print(f"  Reason: {reason}")
                if case.actual_answer:
                    print(f"  Actual Answer: {case.actual_answer[:120]}...")

    print("=" * 70 + "\n")


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="HR Policy RAG Evaluation Runner CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/dataset_v1.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation/reports",
        help="Directory to save JSON and CSV reports",
    )
    parser.add_argument(
        "--no-answer-eval",
        action="store_true",
        help="Skip answer generation evaluation (retrieval-only evaluation)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock pipeline services (for dry-runs or test environments)",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    logger.info("Loading evaluation dataset from: %s", dataset_path)
    dataset = load_evaluation_dataset(dataset_path)

    # Initialize services
    retrieval_service = None
    context_service = None
    answer_service = None

    if args.mock:
        logger.info("Initializing mock pipeline services for evaluation dry-run...")
        from unittest.mock import AsyncMock, MagicMock
        from app.modules.context.schemas import ContextResponse, SourceCitation
        from app.modules.generation.schemas import AnswerResponse, CitationItem
        from app.modules.retrieval.schemas import QueryUnderstandingResult

        mock_context_svc = MagicMock()
        mock_answer_svc = MagicMock()

        from app.modules.reranking.schemas import RankedChunk

        async def mock_build_context(req, school_ctx):
            matched_case = next((tc for tc in dataset.test_cases if tc.question == req.question), None)
            has_docs = bool(matched_case and matched_case.expected_document_ids)
            doc_ids = matched_case.expected_document_ids if has_docs else []

            manifest = [
                SourceCitation(
                    citation_index=idx,
                    citation_label=f"[Source {idx}]",
                    citation_reference=f"[Source {idx}: Policy {did}]",
                    document_id=did,
                    document_name=f"Policy {did}",
                    version=1,
                    chunk_id=f"chk-{did}",
                    page_start=1,
                    page_end=2,
                )
                for idx, did in enumerate(doc_ids, start=1)
            ]

            ranked = [
                RankedChunk(
                    chunk_id=f"chk-{did}",
                    document_id=did,
                    document_name=f"Policy {did}",
                    document_type="policy",
                    version=1,
                    content=f"Policy content for {did}",
                    heading="Policy Section",
                    heading_path=["HR", "Policy Section"],
                    page_start=1,
                    page_end=2,
                    similarity_score=0.9,
                    rerank_score=0.9,
                    lexical_score=0.9,
                    heading_score=0.9,
                    original_rank=idx,
                    rerank_rank=idx,
                )
                for idx, did in enumerate(doc_ids, start=1)
            ]

            return ContextResponse(
                query_understanding=QueryUnderstandingResult(
                    original_query=req.question,
                    normalized_query=req.question,
                    intent="leave",
                ),
                ranked_chunks=ranked,
                assembled_context="Sample policy context" if has_docs else "",
                source_manifest=manifest,
                total_tokens=50 if has_docs else 0,
                token_budget=2000,
                selected_chunks_count=len(manifest),
                skipped_chunks_count=0,
                status="success" if has_docs else "empty",
            )

        async def mock_generate_answer(req, school_ctx):
            matched_case = next((tc for tc in dataset.test_cases if tc.question == req.question), None)
            is_insufficient = matched_case.expect_insufficient_evidence if matched_case else False
            is_conflict = matched_case.expect_conflicting_evidence if matched_case else False

            if is_insufficient:
                return AnswerResponse(
                    question=req.question,
                    answer="The available HR policies do not provide sufficient information.",
                    citations=[],
                    insufficient_evidence=True,
                    conflicting_evidence=False,
                    context_tokens=0,
                    model_name="mock-llm",
                    status="insufficient_evidence",
                )

            pts = " ".join(matched_case.expected_answer_points) if matched_case else ""
            cits = matched_case.expected_citations if (matched_case and matched_case.expected_citations) else ["[Source 1]"]

            return AnswerResponse(
                question=req.question,
                answer=f"Full policy answer covering: {pts} {cits[0]}.",
                citations=[
                    CitationItem(
                        citation_label=lbl,
                        citation_reference=f"{lbl}: Policy",
                        document_id=matched_case.expected_document_ids[0] if (matched_case and matched_case.expected_document_ids) else "doc-1",
                        document_name="HR Policy",
                        version=1,
                        page_start=1,
                        page_end=2,
                    )
                    for lbl in cits
                ],
                insufficient_evidence=False,
                conflicting_evidence=is_conflict,
                context_tokens=50,
                model_name="mock-llm",
                status="success",
            )

        mock_context_svc.build_context = mock_build_context
        mock_answer_svc.generate_answer = mock_generate_answer
        context_service = mock_context_svc
        answer_service = mock_answer_svc
    else:
        logger.info("Initializing live production services...")
        from app.config import get_settings
        from app.db.chroma import ChromaDBService
        from app.db.mongo import MongoManager
        from app.modules.context.router import get_context_service
        from app.modules.generation.router import get_answer_service
        from app.modules.retrieval.router import get_retrieval_service

        settings = get_settings()
        mongo_mgr = MongoManager(settings=settings)
        await mongo_mgr.connect()
        db = mongo_mgr.get_database()

        chroma_svc = ChromaDBService(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)
        chroma_svc.initialize()

        retrieval_service = get_retrieval_service(db=db, chroma=chroma_svc, settings=settings)
        context_service = get_context_service(retrieval_service=retrieval_service, settings=settings)
        answer_service = get_answer_service(context_service=context_service, settings=settings)

    runner = EvaluationRunner(
        retrieval_service=retrieval_service,
        context_service=context_service,
        answer_service=answer_service,
    )

    logger.info("Executing evaluation on %d test cases...", len(dataset.test_cases))
    summary = await runner.run_dataset(dataset, evaluate_answer=not args.no_answer_eval)

    # Print Report
    _print_summary_table(summary)

    # Export Reports
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"evaluation_report_{timestamp_str}.json"
    csv_path = output_dir / f"evaluation_cases_{timestamp_str}.csv"

    runner.export_json(summary, json_path)
    runner.export_csv(summary, csv_path)

    print(f"JSON Report exported to : {json_path}")
    print(f"CSV Report exported to  : {csv_path}\n")

    return 0 if summary.pass_rate >= 0.70 else 1


def main():
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()

"""Mathematical metrics calculator for retrieval ranking, answer completeness, and citation quality."""

import math
import re
from typing import Any, List, Set, Tuple

WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_-]{2,}\b")

# Evaluation stop words for keyword coverage check
EVAL_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "that", "this", "it", "as",
}


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Precision@K: fraction of retrieved items in the top-K that are relevant."""
    if k <= 0:
        return 0.0
    if not relevant:
        # If no relevant items exist (e.g. unaddressed policy query), any retrieval is a false positive
        return 1.0 if not retrieved else 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for item in top_k if item in relevant)
    return round(hits / k, 4)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@K: fraction of all relevant items captured within the top-K retrieved."""
    if not relevant:
        # If no items are relevant, recall is 1.0 (all relevant items were retrieved)
        return 1.0
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return round(hits / len(relevant), 4)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Reciprocal Rank (RR): 1 / rank of the first relevant item retrieved."""
    if not relevant:
        return 1.0 if not retrieved else 0.0

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K (nDCG@K) using binary relevance."""
    if k <= 0:
        return 0.0
    if not relevant:
        return 1.0 if not retrieved else 0.0

    top_k = retrieved[:k]

    # DCG calculation
    dcg = 0.0
    for idx, item in enumerate(top_k, start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(idx + 1)

    # Ideal DCG calculation
    idcg = 0.0
    max_hits = min(k, len(relevant))
    for idx in range(1, max_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def calculate_point_coverage(answer: str, expected_points: List[str]) -> Tuple[float, List[str], List[str]]:
    """Determine how many expected answer points are addressed in the generated answer text.

    Returns:
        Tuple of (coverage_ratio, matched_points_list, missing_points_list)
    """
    if not expected_points:
        return 1.0, [], []
    if not answer or not answer.strip():
        return 0.0, [], list(expected_points)

    answer_lower = answer.lower()
    answer_words = set(WORD_REGEX.findall(answer_lower))

    matched: List[str] = []
    missing: List[str] = []

    for point in expected_points:
        point_lower = point.lower().strip()

        # Check 1: Exact substring containment
        if point_lower in answer_lower:
            matched.append(point)
            continue

        # Check 2: Significant token overlap (>= 70% of non-stop-words present)
        point_words = [w for w in WORD_REGEX.findall(point_lower) if w not in EVAL_STOP_WORDS]
        if not point_words:
            # Short point without non-stop words
            if point_lower in answer_lower:
                matched.append(point)
            else:
                missing.append(point)
            continue

        present_words = sum(1 for w in point_words if w in answer_words)
        overlap_ratio = present_words / len(point_words)

        if overlap_ratio >= 0.70:
            matched.append(point)
        else:
            missing.append(point)

    coverage = len(matched) / len(expected_points)
    return round(coverage, 4), matched, missing


def calculate_citation_metrics(
    actual_citations: List[Any],
    expected_citations: List[str],
    manifest_labels: Set[str],
) -> Tuple[float, float, List[str]]:
    """Compute citation validity and citation coverage against expected references and source manifest.

    Returns:
        Tuple of (citation_validity, citation_coverage, unsupported_citations_list)
    """
    # Extract clean string labels from actual citations
    actual_labels: List[str] = []
    for item in actual_citations:
        if isinstance(item, str):
            actual_labels.append(item.strip())
        elif hasattr(item, "citation_label"):
            actual_labels.append(item.citation_label.strip())
        elif isinstance(item, dict) and "citation_label" in item:
            actual_labels.append(item["citation_label"].strip())

    if not actual_labels:
        validity = 1.0
        unsupported = []
    else:
        norm_manifest = {lbl.strip().lower() for lbl in manifest_labels}
        valid_count = 0
        unsupported = []
        for lbl in actual_labels:
            if lbl.lower() in norm_manifest:
                valid_count += 1
            else:
                unsupported.append(lbl)
        validity = round(valid_count / len(actual_labels), 4)

    # Citation coverage against expected citations
    if not expected_citations:
        coverage = 1.0
    else:
        # Check how many expected citations are represented
        actual_text = " ".join(actual_labels).lower()
        matched_expected = 0
        for exp in expected_citations:
            clean_exp = exp.strip().lower()
            if any(clean_exp in lbl.lower() for lbl in actual_labels) or clean_exp in actual_text:
                matched_expected += 1
        coverage = round(matched_expected / len(expected_citations), 4)

    return validity, coverage, unsupported

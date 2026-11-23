"""Lightweight deterministic V1 chunk reranker combining semantic, lexical, and heading relevance."""

import logging
import re
from typing import List, Optional, Set
from app.modules.reranking.base import BaseReranker
from app.modules.reranking.schemas import RankedChunk
from app.modules.retrieval.schemas import RetrievedChunk
from app.modules.retrieval.understanding import STOP_WORDS

logger = logging.getLogger(__name__)

WORD_SPLIT_REGEX = re.compile(r"\b[a-zA-Z0-9_-]{2,}\b")


class DeterministicV1Reranker(BaseReranker):
    """Deterministic, model-free reranker prioritizing semantic similarity, lexical coverage, and heading alignment.

    Scores are normalized ranking heuristics and must not be treated as confidence probabilities.
    """

    def __init__(
        self,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        heading_weight: float = 0.2,
    ) -> None:
        total_weight = semantic_weight + lexical_weight + heading_weight
        if total_weight <= 0:
            total_weight = 1.0
            semantic_weight = 0.5
            lexical_weight = 0.3
            heading_weight = 0.2

        self.w_sem = semantic_weight / total_weight
        self.w_lex = lexical_weight / total_weight
        self.w_head = heading_weight / total_weight

    def _extract_query_terms(self, query: str, keywords: Optional[List[str]] = None) -> Set[str]:
        """Extract significant query terms filtering out common stop words."""
        terms: Set[str] = set()

        # From raw query
        all_words = WORD_SPLIT_REGEX.findall(query.lower())
        filtered_words = {w for w in all_words if w not in STOP_WORDS}
        if filtered_words:
            terms.update(filtered_words)
        else:
            # Fallback if every word is considered a stop word
            terms.update(all_words)

        # From extracted keywords
        if keywords:
            for kw in keywords:
                for kw_word in WORD_SPLIT_REGEX.findall(kw.lower()):
                    if kw_word not in STOP_WORDS:
                        terms.add(kw_word)

        return terms

    def _compute_lexical_score(
        self,
        query: str,
        content: str,
        query_terms: Set[str],
        keywords: Optional[List[str]] = None,
    ) -> float:
        """Compute lexical match score in [0.0, 1.0] based on term coverage, frequency, and phrase matches."""
        if not query_terms or not content:
            return 0.0

        content_lower = content.lower()
        content_words = set(WORD_SPLIT_REGEX.findall(content_lower))

        # 1. Term coverage (fraction of unique query terms present in chunk)
        matched_terms = query_terms.intersection(content_words)
        coverage = len(matched_terms) / len(query_terms)

        # 2. Term frequency / hit count
        total_hits = sum(content_lower.count(t) for t in query_terms)
        freq_factor = min(1.0, total_hits / max(1, len(query_terms) * 2))

        # 3. Substring / phrase match bonus
        clean_query = query.strip().lower()
        phrase_bonus = 0.0
        if len(clean_query) > 3 and clean_query in content_lower:
            phrase_bonus = 0.25
        elif keywords:
            for kw in keywords:
                clean_kw = kw.strip().lower()
                if " " in clean_kw and clean_kw in content_lower:
                    phrase_bonus = max(phrase_bonus, 0.15)

        raw_score = (0.6 * coverage) + (0.2 * freq_factor) + phrase_bonus
        return min(1.0, max(0.0, raw_score))

    def _compute_heading_score(
        self,
        query: str,
        heading: Optional[str],
        heading_path: Optional[List[str]],
        query_terms: Set[str],
        keywords: Optional[List[str]] = None,
    ) -> float:
        """Compute heading alignment score in [0.0, 1.0] based on heading text and hierarchical breadcrumbs."""
        if not heading and not heading_path:
            return 0.0

        all_headings = []
        if heading:
            all_headings.append(heading)
        if heading_path:
            all_headings.extend(heading_path)

        heading_text = " ".join(all_headings).lower()
        clean_query = query.strip().lower()

        # Exact match bonus
        if clean_query and clean_query in heading_text:
            return 1.0

        heading_words = set(WORD_SPLIT_REGEX.findall(heading_text))
        if not heading_words or not query_terms:
            return 0.0

        # Term overlap in heading
        matched_terms = query_terms.intersection(heading_words)
        overlap = len(matched_terms) / len(query_terms)

        # Keyword match in heading
        keyword_bonus = 0.0
        if keywords:
            for kw in keywords:
                clean_kw = kw.strip().lower()
                if clean_kw in heading_text:
                    keyword_bonus = max(keyword_bonus, 0.3)

        raw_score = overlap + keyword_bonus
        return min(1.0, max(0.0, raw_score))

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        keywords: Optional[List[str]] = None,
    ) -> List[RankedChunk]:
        """Rerank candidate chunks deterministically combining semantic, lexical, and heading signals.

        Preserves original retrieval scores and returns ranked candidates in descending relevance order.
        """
        if not chunks:
            return []

        query_terms = self._extract_query_terms(query, keywords)
        scored_candidates: List[RankedChunk] = []

        for original_idx, chunk in enumerate(chunks, start=1):
            s_sem = min(1.0, max(0.0, chunk.similarity_score))
            s_lex = self._compute_lexical_score(query, chunk.content, query_terms, keywords)
            s_head = self._compute_heading_score(query, chunk.heading, chunk.heading_path, query_terms, keywords)

            composite = (self.w_sem * s_sem) + (self.w_lex * s_lex) + (self.w_head * s_head)
            composite_score = round(min(1.0, max(0.0, composite)), 4)

            explanation = (
                f"rerank={composite_score:.4f} "
                f"(sem={s_sem:.4f}*{self.w_sem:.2f} + "
                f"lex={s_lex:.4f}*{self.w_lex:.2f} + "
                f"head={s_head:.4f}*{self.w_head:.2f}) "
                f"[ranking heuristic, not confidence probability]"
            )

            ranked_chunk = RankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                document_type=chunk.document_type,
                version=chunk.version,
                content=chunk.content,
                heading=chunk.heading,
                heading_path=chunk.heading_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                similarity_score=chunk.similarity_score,
                school_id=chunk.school_id,
                group_id=chunk.group_id,
                branch_id=chunk.branch_id,
                effective_from=chunk.effective_from,
                effective_to=chunk.effective_to,
                retrieval_mode=chunk.retrieval_mode,
                dense_score=chunk.dense_score,
                dense_rank=chunk.dense_rank,
                bm25_score=chunk.bm25_score,
                bm25_rank=chunk.bm25_rank,
                rrf_score=chunk.rrf_score,
                rerank_score=composite_score,
                lexical_score=round(s_lex, 4),
                heading_score=round(s_head, 4),
                original_rank=original_idx,
                rerank_rank=original_idx,  # Updated after sorting
                score_explanation=explanation,
            )
            scored_candidates.append(ranked_chunk)

        # Sort descending by composite rerank score, breaking ties by semantic similarity, then original rank
        scored_candidates.sort(
            key=lambda c: (-c.rerank_score, -c.similarity_score, c.original_rank)
        )

        # Assign final rerank_rank
        for final_rank, candidate in enumerate(scored_candidates, start=1):
            candidate.rerank_rank = final_rank

        return scored_candidates

"""Context Builder engine enforcing deduplication, token budgets, and structured citation tracking."""

import hashlib
import logging
import re
from typing import List, Optional, Set, Tuple

import tiktoken

from app.modules.context.schemas import ContextBuildResult, SourceCitation
from app.modules.reranking.schemas import RankedChunk

logger = logging.getLogger(__name__)

WORD_TOKEN_REGEX = re.compile(r"\b\w+\b")


class ContextBuilder:
    """Builds prompt-ready context windows from reranked policy chunks under strict token budgets."""

    def __init__(
        self,
        default_token_budget: int = 2000,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.default_token_budget = default_token_budget
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def _normalize_content_for_hash(self, text: str) -> str:
        """Normalize whitespace and case for exact duplicate detection."""
        return " ".join(text.lower().split())

    def _compute_jaccard_similarity(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets of word tokens."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    def deduplicate_chunks(self, chunks: List[RankedChunk]) -> Tuple[List[RankedChunk], int]:
        """Prune exact and near-duplicate chunks while strictly preserving distinct policy clauses.

        Returns:
            Tuple of (deduplicated_chunks, duplicate_count)
        """
        if not chunks:
            return [], 0

        unique_chunks: List[RankedChunk] = []
        seen_exact_hashes: Set[str] = set()
        seen_clause_tokens: List[Tuple[Optional[str], Set[str]]] = []
        duplicates_removed = 0

        for chunk in chunks:
            norm_content = self._normalize_content_for_hash(chunk.content)
            content_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()

            # 1. Exact duplicate check
            if content_hash in seen_exact_hashes:
                duplicates_removed += 1
                continue

            # 2. Near-duplicate check: only consider duplicates if headings match
            # This ensures distinct clauses under different headings are never pruned
            chunk_tokens = set(WORD_TOKEN_REGEX.findall(norm_content))
            is_near_duplicate = False

            if chunk.heading:
                norm_heading = chunk.heading.strip().lower()
                for seen_heading, seen_tokens in seen_clause_tokens:
                    if seen_heading == norm_heading:
                        sim = self._compute_jaccard_similarity(chunk_tokens, seen_tokens)
                        if sim >= 0.88:
                            is_near_duplicate = True
                            break

            if is_near_duplicate:
                duplicates_removed += 1
                continue

            seen_exact_hashes.add(content_hash)
            if chunk.heading:
                seen_clause_tokens.append((chunk.heading.strip().lower(), chunk_tokens))
            unique_chunks.append(chunk)

        return unique_chunks, duplicates_removed

    def _format_source_header(self, index: int, chunk: RankedChunk) -> Tuple[str, str]:
        """Format human-readable citation label and header reference line."""
        pages_str = (
            f"page {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.page_start}–{chunk.page_end}"
        )

        heading_path_str = (
            " > ".join(chunk.heading_path)
            if chunk.heading_path
            else (chunk.heading or "General Policy")
        )

        citation_label = f"[Source {index}]"
        citation_reference = f"[Source {index}: {chunk.document_name}, {pages_str}, {heading_path_str}]"
        return citation_label, citation_reference

    def _format_chunk_block(
        self,
        index: int,
        chunk: RankedChunk,
        citation_label: str,
        citation_reference: str,
    ) -> str:
        """Format a single complete chunk block with full source attribution."""
        scope_parts = []
        if chunk.school_id:
            scope_parts.append(f"School: {chunk.school_id}")
        if chunk.group_id:
            scope_parts.append(f"Group: {chunk.group_id}")
        if chunk.branch_id:
            scope_parts.append(f"Branch: {chunk.branch_id}")
        scope_str = ", ".join(scope_parts) if scope_parts else "Global Policy"

        date_parts = []
        if chunk.effective_from:
            date_parts.append(f"From: {chunk.effective_from.strftime('%Y-%m-%d')}")
        if chunk.effective_to:
            date_parts.append(f"To: {chunk.effective_to.strftime('%Y-%m-%d')}")
        dates_str = " | ".join(date_parts) if date_parts else "Effective: Current"

        block = (
            f"### {citation_reference}\n"
            f"Document: {chunk.document_name} (v{chunk.version}) | "
            f"Doc ID: {chunk.document_id} | Chunk ID: {chunk.chunk_id}\n"
            f"Scope: {scope_str} | {dates_str}\n\n"
            f"{chunk.content.strip()}"
        )
        return block

    def build_context(
        self,
        chunks: List[RankedChunk],
        token_budget: Optional[int] = None,
    ) -> ContextBuildResult:
        """Select, deduplicate, and assemble complete policy chunks within the token budget.

        Preserves complete chunks. If a chunk cannot fit entirely within the remaining token
        budget, it is skipped rather than truncated mid-sentence or mid-table.
        """
        budget = token_budget if (token_budget is not None and token_budget > 0) else self.default_token_budget

        if not chunks:
            return ContextBuildResult(
                assembled_context="",
                source_manifest=[],
                total_tokens=0,
                token_budget=budget,
                selected_chunks_count=0,
                skipped_chunks_count=0,
                selected_chunks=[],
            )

        # 1. Deduplication
        deduped_chunks, duplicate_count = self.deduplicate_chunks(chunks)

        selected_blocks: List[str] = []
        source_manifest: List[SourceCitation] = []
        selected_chunks: List[RankedChunk] = []
        skipped_count = duplicate_count

        separator = "\n\n---\n\n"
        separator_tokens = self.count_tokens(separator)
        current_tokens = 0

        for chunk in deduped_chunks:
            citation_index = len(source_manifest) + 1
            citation_label, citation_reference = self._format_source_header(citation_index, chunk)
            block_text = self._format_chunk_block(citation_index, chunk, citation_label, citation_reference)

            block_tokens = self.count_tokens(block_text)
            projected_tokens = current_tokens + block_tokens + (separator_tokens if selected_blocks else 0)

            if projected_tokens > budget:
                # Chunk does not fit completely; do NOT truncate mid-content. Skip it.
                logger.debug(
                    "Skipping chunk %s (%d tokens) exceeding remaining budget (%d remaining)",
                    chunk.chunk_id,
                    block_tokens,
                    budget - current_tokens,
                )
                skipped_count += 1
                continue

            # Complete chunk fits comfortably
            selected_blocks.append(block_text)
            current_tokens = projected_tokens
            selected_chunks.append(chunk)

            manifest_entry = SourceCitation(
                citation_index=citation_index,
                citation_label=citation_label,
                citation_reference=citation_reference,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                document_type=chunk.document_type,
                version=chunk.version,
                chunk_id=chunk.chunk_id,
                heading=chunk.heading,
                heading_path=chunk.heading_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                school_id=chunk.school_id,
                group_id=chunk.group_id,
                branch_id=chunk.branch_id,
                effective_from=chunk.effective_from,
                effective_to=chunk.effective_to,
            )
            source_manifest.append(manifest_entry)

        assembled_text = separator.join(selected_blocks)
        final_token_count = self.count_tokens(assembled_text)

        return ContextBuildResult(
            assembled_context=assembled_text,
            source_manifest=source_manifest,
            total_tokens=final_token_count,
            token_budget=budget,
            selected_chunks_count=len(selected_chunks),
            skipped_chunks_count=skipped_count,
            selected_chunks=selected_chunks,
        )

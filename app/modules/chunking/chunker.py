"""Heading-aware document chunking engine with recursive splitting and structure preservation."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from app.modules.chunking.schemas import DocumentChunk
from app.modules.processing.schemas import DocumentProcessedResult, ProcessedPage
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)

# Pattern to detect markdown headings (# through ######)
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")

# Pattern to detect markdown table separator row
TABLE_SEP_REGEX = re.compile(r"^\s*\|(?:\s*[:-]+[-| :]*\|)+\s*$")

# Pattern to split list items while preserving indentation and bullet/number
LIST_SPLIT_REGEX = re.compile(r"\n(?=[-*\+]|\d+\.|\([a-zA-Z0-9]+\))")

# Sentence boundary regex for recursive paragraph splitting
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


class ContentBlock:
    """Internal representation of a discrete content unit tagged with page tracking."""

    def __init__(
        self,
        text: str,
        page_number: int,
        is_heading: bool = False,
        heading_level: int = 0,
        heading_text: str = "",
        is_table: bool = False,
        is_list: bool = False,
    ) -> None:
        self.text = text.strip()
        self.page_number = page_number
        self.is_heading = is_heading
        self.heading_level = heading_level
        self.heading_text = heading_text
        self.is_table = is_table
        self.is_list = is_list


class SectionGroup:
    """A collection of content blocks belonging to the same heading hierarchy."""

    def __init__(
        self,
        heading: Optional[str] = None,
        heading_path: Optional[List[str]] = None,
    ) -> None:
        self.heading = heading
        self.heading_path = list(heading_path or [])
        self.blocks: List[ContentBlock] = []

    def add_block(self, block: ContentBlock) -> None:
        self.blocks.append(block)

    @property
    def has_content(self) -> bool:
        return any(bool(b.text) for b in self.blocks)


class DocumentChunker:
    """Heading-aware document chunker with recursive splitting and table preservation."""

    def __init__(
        self,
        target_tokens: int = 500,
        max_tokens: int = 700,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured tiktoken encoding."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def chunk_document(
        self,
        processed_doc: DocumentProcessedResult,
        doc_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Convert a cleaned multi-page document into structured DocumentChunks.

        Splits primarily at heading/section boundaries, keeping related paragraphs
        together, while recursively splitting oversized sections and preserving tables.
        """
        document_id = processed_doc.document_id

        # 1. Parse pages into a stream of typed ContentBlocks
        blocks = self._parse_pages_to_blocks(processed_doc.processed_pages)
        if not blocks:
            logger.info("No content blocks found for document_id=%s. Returning empty chunks.", document_id)
            return []

        # 2. Group blocks into hierarchical Sections
        sections = self._group_blocks_into_sections(blocks)

        # 3. Chunk each section independently
        raw_chunks: List[Tuple[str, Optional[str], List[str], List[int]]] = []
        for section in sections:
            if not section.has_content:
                continue
            section_chunks = self._chunk_section(section)
            raw_chunks.extend(section_chunks)

        # 4. Assemble final DocumentChunk schemas with metadata
        emp_app = doc_metadata.get("employee_applicability") or {}
        chunks: List[DocumentChunk] = []

        for idx, (content, heading, heading_path, pages) in enumerate(raw_chunks):
            if not content.strip():
                continue

            sorted_pages = sorted(list(set(pages))) if pages else [1]
            token_count = self.count_tokens(content)

            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{idx}",
                document_id=document_id,
                chunk_index=idx,
                content=content.strip(),
                token_count=token_count,
                heading=heading,
                heading_path=heading_path,
                page_start=sorted_pages[0],
                page_end=sorted_pages[-1],
                source_pages=sorted_pages,
                # Canonical school scope
                school_id=doc_metadata.get("school_id"),
                group_id=doc_metadata.get("group_id"),
                branch_id=doc_metadata.get("branch_id"),
                is_global=doc_metadata.get("is_global", False),
                state=doc_metadata.get("state"),
                # Employee applicability
                staff_category=emp_app.get("staff_category"),
                employment_type=emp_app.get("employment_type"),
                department=emp_app.get("department"),
                # Document details
                document_name=doc_metadata.get("document_name", "document.pdf"),
                document_type=doc_metadata.get("document_type", "policy"),
                version=doc_metadata.get("version", 1),
                effective_from=doc_metadata.get("effective_from"),
                effective_to=doc_metadata.get("effective_to"),
                created_at=current_utc_time(),
            )
            chunks.append(chunk)

        logger.info(
            "Chunked document_id=%s into %d chunks (target=%d, max=%d, overlap=%d)",
            document_id,
            len(chunks),
            self.target_tokens,
            self.max_tokens,
            self.overlap_tokens,
        )
        return chunks

    def _parse_pages_to_blocks(self, pages: List[ProcessedPage]) -> List[ContentBlock]:
        """Deconstruct multi-page Markdown into ordered, page-tagged ContentBlocks."""
        blocks: List[ContentBlock] = []

        for page in pages:
            page_text = page.cleaned_content.strip()
            if not page_text:
                continue

            # Split page into paragraph/block units via double newline
            raw_units = re.split(r"\n\s*\n", page_text)
            for unit in raw_units:
                unit_str = unit.strip()
                if not unit_str:
                    continue

                # Check if Heading
                h_match = HEADING_REGEX.match(unit_str)
                if h_match and "\n" not in unit_str:
                    level = len(h_match.group(1))
                    heading_text = unit_str
                    blocks.append(
                        ContentBlock(
                            text=unit_str,
                            page_number=page.page_number,
                            is_heading=True,
                            heading_level=level,
                            heading_text=heading_text,
                        )
                    )
                    continue

                # Check if Markdown Table
                lines = [l.strip() for l in unit_str.splitlines() if l.strip()]
                is_table = (
                    len(lines) >= 3
                    and all(l.startswith("|") and l.endswith("|") for l in lines)
                    and any(TABLE_SEP_REGEX.match(l) for l in lines)
                )
                if is_table:
                    blocks.append(
                        ContentBlock(
                            text=unit_str,
                            page_number=page.page_number,
                            is_table=True,
                        )
                    )
                    continue

                # Check if List block
                is_list = all(
                    re.match(r"^\s*([-*+]|\d+\.|\([a-zA-Z0-9]+\))\s+", l)
                    or l.startswith(" ")
                    or l.startswith("\t")
                    for l in lines
                )
                if is_list:
                    blocks.append(
                        ContentBlock(
                            text=unit_str,
                            page_number=page.page_number,
                            is_list=True,
                        )
                    )
                    continue

                # Default: Paragraph block
                blocks.append(
                    ContentBlock(
                        text=unit_str,
                        page_number=page.page_number,
                    )
                )

        return blocks

    def _group_blocks_into_sections(self, blocks: List[ContentBlock]) -> List[SectionGroup]:
        """Assemble ContentBlocks into SectionGroups using hierarchical heading stacks."""
        sections: List[SectionGroup] = []
        heading_stack: List[Tuple[int, str]] = []  # [(level, heading_text)]

        current_section = SectionGroup(heading=None, heading_path=[])

        for block in blocks:
            if block.is_heading:
                # Flush existing section if it contains content
                if current_section.has_content:
                    sections.append(current_section)

                # Update hierarchical heading stack
                while heading_stack and heading_stack[-1][0] >= block.heading_level:
                    heading_stack.pop()
                heading_stack.append((block.heading_level, block.heading_text))

                current_path = [item[1] for item in heading_stack]
                current_section = SectionGroup(
                    heading=block.heading_text,
                    heading_path=current_path,
                )
            else:
                current_section.add_block(block)

        if current_section.has_content:
            sections.append(current_section)

        return sections

    def _chunk_section(
        self, section: SectionGroup
    ) -> List[Tuple[str, Optional[str], List[str], List[int]]]:
        """Chunk a section. If section tokens <= max_tokens, returns a single chunk.

        If oversized, recursively splits blocks, handles large tables, and adds overlap.
        Returns: List of tuples (chunk_text, heading, heading_path, source_pages)
        """
        combined_text = "\n\n".join(b.text for b in section.blocks)
        total_tokens = self.count_tokens(combined_text)
        all_pages = [b.page_number for b in section.blocks]

        # Case 1: Section fits cleanly within max_tokens
        if total_tokens <= self.max_tokens:
            content_with_heading = self._format_chunk_content(section.heading, combined_text)
            return [(content_with_heading, section.heading, section.heading_path, all_pages)]

        # Case 2: Section is oversized -> Split recursively
        return self._recursive_split_section(section)

    def _recursive_split_section(
        self, section: SectionGroup
    ) -> List[Tuple[str, Optional[str], List[str], List[int]]]:
        """Recursively split oversized section blocks with table and list preservation."""
        chunks: List[Tuple[str, Optional[str], List[str], List[int]]] = []

        accumulated_texts: List[str] = []
        accumulated_pages: Set[int] = set()
        accumulated_tokens = 0
        is_first_chunk = True

        def _flush_accumulated(apply_overlap: bool = True) -> None:
            nonlocal accumulated_texts, accumulated_pages, accumulated_tokens, is_first_chunk
            if not accumulated_texts:
                return

            text_payload = "\n\n".join(accumulated_texts)
            heading_to_use = section.heading if is_first_chunk else None
            formatted = self._format_chunk_content(heading_to_use, text_payload)

            chunks.append(
                (
                    formatted,
                    section.heading,
                    section.heading_path,
                    sorted(list(accumulated_pages)),
                )
            )

            # Extract overlap text from the tail of this chunk for the next chunk
            overlap_prefix = ""
            if apply_overlap and self.overlap_tokens > 0:
                overlap_prefix = self._get_overlap_text(text_payload, self.overlap_tokens)

            if overlap_prefix:
                accumulated_texts = [overlap_prefix]
                accumulated_tokens = self.count_tokens(overlap_prefix)
                # Keep last page in accumulated_pages for overlap continuity
                last_page = max(accumulated_pages) if accumulated_pages else 1
                accumulated_pages = {last_page}
            else:
                accumulated_texts = []
                accumulated_pages = set()
                accumulated_tokens = 0

            is_first_chunk = False

        for block in section.blocks:
            block_tokens = self.count_tokens(block.text)

            # A. Oversized Table: split by rows with repeated headers
            if block.is_table and block_tokens > self.max_tokens:
                _flush_accumulated(apply_overlap=False)
                table_slices = self._split_table(block.text, self.target_tokens)
                for tbl_slice in table_slices:
                    heading_to_use = section.heading if is_first_chunk else None
                    formatted = self._format_chunk_content(heading_to_use, tbl_slice)
                    chunks.append(
                        (
                            formatted,
                            section.heading,
                            section.heading_path,
                            [block.page_number],
                        )
                    )
                    is_first_chunk = False
                continue

            # B. Oversized List: split by list items
            if block.is_list and block_tokens > self.max_tokens:
                _flush_accumulated(apply_overlap=False)
                list_items = LIST_SPLIT_REGEX.split(block.text)
                for item in list_items:
                    item_str = item.strip()
                    if not item_str:
                        continue
                    item_tokens = self.count_tokens(item_str)
                    if accumulated_tokens + item_tokens > self.target_tokens and accumulated_texts:
                        _flush_accumulated(apply_overlap=True)
                    accumulated_texts.append(item_str)
                    accumulated_pages.add(block.page_number)
                    accumulated_tokens += item_tokens
                continue

            # C. Oversized Paragraph: split recursively by sentences / words
            if block_tokens > self.max_tokens:
                _flush_accumulated(apply_overlap=False)
                para_pieces = self._split_oversized_paragraph(block.text, self.target_tokens)
                for piece in para_pieces:
                    piece_tokens = self.count_tokens(piece)
                    if accumulated_tokens + piece_tokens > self.target_tokens and accumulated_texts:
                        _flush_accumulated(apply_overlap=True)
                    accumulated_texts.append(piece)
                    accumulated_pages.add(block.page_number)
                    accumulated_tokens += piece_tokens
                continue

            # D. Standard block: accumulate up to target_tokens
            if accumulated_tokens + block_tokens > self.target_tokens and accumulated_texts:
                _flush_accumulated(apply_overlap=True)

            accumulated_texts.append(block.text)
            accumulated_pages.add(block.page_number)
            accumulated_tokens += block_tokens

        # Flush any remaining content
        if accumulated_texts:
            text_payload = "\n\n".join(accumulated_texts)
            heading_to_use = section.heading if is_first_chunk else None
            formatted = self._format_chunk_content(heading_to_use, text_payload)
            chunks.append(
                (
                    formatted,
                    section.heading,
                    section.heading_path,
                    sorted(list(accumulated_pages)),
                )
            )

        return chunks

    def _split_table(self, table_text: str, target_tokens: int) -> List[str]:
        """Split a large Markdown table row-by-row, repeating the header and separator rows."""
        lines = [l.strip() for l in table_text.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            return [table_text]

        header_row = lines[0]
        separator_row = lines[1]
        data_rows = lines[2:]

        header_prefix = f"{header_row}\n{separator_row}"
        header_tokens = self.count_tokens(header_prefix)

        table_chunks: List[str] = []
        current_rows: List[str] = []
        current_tokens = header_tokens

        for row in data_rows:
            row_tokens = self.count_tokens("\n" + row)
            if current_rows and (current_tokens + row_tokens > target_tokens):
                chunk_table = header_prefix + "\n" + "\n".join(current_rows)
                table_chunks.append(chunk_table)
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        if current_rows:
            chunk_table = header_prefix + "\n" + "\n".join(current_rows)
            table_chunks.append(chunk_table)

        return table_chunks

    def _split_oversized_paragraph(self, text: str, target_tokens: int) -> List[str]:
        """Recursively split a single oversized paragraph by sentences, then words."""
        sentences = SENTENCE_SPLIT_REGEX.split(text)
        pieces: List[str] = []
        current_batch: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            s_tokens = self.count_tokens(sentence)

            # If an individual sentence exceeds target_tokens, split by words
            if s_tokens > target_tokens:
                if current_batch:
                    pieces.append(" ".join(current_batch))
                    current_batch = []
                    current_tokens = 0
                word_pieces = self._split_by_words(sentence, target_tokens)
                pieces.extend(word_pieces)
                continue

            if current_batch and (current_tokens + s_tokens > target_tokens):
                pieces.append(" ".join(current_batch))
                current_batch = [sentence]
                current_tokens = s_tokens
            else:
                current_batch.append(sentence)
                current_tokens += s_tokens

        if current_batch:
            pieces.append(" ".join(current_batch))

        return pieces

    def _split_by_words(self, text: str, target_tokens: int) -> List[str]:
        """Split text by words into chunks conforming to target_tokens."""
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        tokens = 0

        for word in words:
            w_tokens = self.count_tokens(" " + word)
            if current and (tokens + w_tokens > target_tokens):
                chunks.append(" ".join(current))
                current = [word]
                tokens = w_tokens
            else:
                current.append(word)
                tokens += w_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Extract a clean tail segment of approximately overlap_tokens aligned to a word boundary."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= overlap_tokens:
            return ""

        tail_tokens = tokens[-overlap_tokens:]
        decoded = self.encoder.decode(tail_tokens)
        first_space = decoded.find(" ")
        if first_space != -1 and first_space < 25:
            decoded = decoded[first_space + 1 :]

        return decoded.strip()

    @staticmethod
    def _format_chunk_content(heading: Optional[str], body_text: str) -> str:
        """Format chunk content, ensuring the section heading is at the top if present."""
        body = body_text.strip()
        if not heading:
            return body
        if body.startswith(heading):
            return body
        return f"{heading}\n\n{body}"

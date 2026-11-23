"""Conservative document cleaning and normalization service for extracted text."""

from collections import Counter
import logging
import re
from typing import List, Optional, Set, Tuple

from app.modules.extraction.schemas import DocumentExtractionResult, ExtractedPage
from app.modules.processing.schemas import (
    DetectedStructure,
    DocumentProcessedResult,
    ProcessedPage,
    ProcessingStageStatus,
)
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)

# Control and invisible characters to strip (leaving standard \n, \t)
CONTROL_AND_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad\ufffd\u200b\u200c\u200d\ufeff]"
)

# Standalone page number noise patterns:
# e.g. "Page 1", "Page 1 of 10", "Page 1 / 10", "- 1 -", "1 / 10", "1"
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(page\s+\d+(\s*(of|/)\s*\d+)?|-?\s*\d+\s*-?|\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)

# Markdown heading pattern
HEADING_PATTERN = re.compile(r"^(#{1,6}\s+.+)$")

# Markdown table row pattern
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|(?:\s*[:-]+[-| :]*\|)+\s*$")

# Markdown list item pattern
LIST_ITEM_PATTERN = re.compile(r"^\s*([-*+]|\d+\.|\([a-zA-Z0-9]+\))\s+")


class DocumentCleaner:
    """Performs conservative, rule-based text cleaning and structure detection."""

    def clean_document(self, extraction: DocumentExtractionResult) -> DocumentProcessedResult:
        """Clean and normalize extracted document content page by page.

        Preserves:
        - Original page order and page numbers
        - Headings and subheadings
        - Numbered and bulleted lists
        - Markdown tables and cell relationships
        - Currency, dates, policy references, and meaningful punctuation

        Removes:
        - Confidently identified repeated headers and footers across pages
        - Standalone page-number noise
        - Non-printable extraction artifacts and zero-width characters
        - Excessive blank lines and trailing whitespace
        """
        raw_pages: List[ExtractedPage] = extraction.extracted_pages

        # 1. Confidently identify repeated running headers and footers across pages
        raw_contents = [p.content for p in raw_pages]
        repeated_headers, repeated_footers = self.identify_repeated_headers_footers(raw_contents)

        # 2. Clean each page and detect structure
        processed_pages: List[ProcessedPage] = []
        total_chars = 0

        for page in raw_pages:
            cleaned_text = self.clean_page(
                raw_content=page.content,
                repeated_headers=repeated_headers,
                repeated_footers=repeated_footers,
            )
            structure = self.detect_structure(cleaned_text)
            char_count = len(cleaned_text)
            total_chars += char_count

            processed_pages.append(
                ProcessedPage(
                    page_number=page.page_number,
                    cleaned_content=cleaned_text,
                    detected_structure=structure,
                    character_count=char_count,
                )
            )

        source_ext_id = f"ext_{extraction.document_id}_{int(extraction.extracted_at.timestamp())}"

        return DocumentProcessedResult(
            document_id=extraction.document_id,
            source_extraction_id=source_ext_id,
            processing_status=ProcessingStageStatus.COMPLETED,
            processed_pages=processed_pages,
            total_pages=len(processed_pages),
            total_characters=total_chars,
            processed_at=current_utc_time(),
        )

    def identify_repeated_headers_footers(
        self, pages_content: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Identify repeated running headers and footers across multi-page documents.

        Only flags recurring lines when the document has >= 2 pages and a candidate line
        (or normalized digit-masked template) appears across at least 2 pages and >= 50%
        of total non-empty pages.
        """
        if len(pages_content) < 2:
            return set(), set()

        top_candidates: List[List[str]] = []
        bottom_candidates: List[List[str]] = []

        valid_pages_count = 0

        for content in pages_content:
            stripped = content.strip()
            if not stripped:
                continue
            valid_pages_count += 1
            lines = [l.strip() for l in stripped.splitlines() if l.strip()]
            if not lines:
                continue

            # Check up to first 2 non-empty lines for header candidates
            page_top = []
            for line in lines[:2]:
                if not PAGE_NUMBER_PATTERN.match(line):
                    norm = self._normalize_line(line)
                    page_top.append(norm)
                    page_top.append(self._mask_digits(norm))
            top_candidates.append(page_top)

            # Check up to last 2 non-empty lines for footer candidates
            page_bottom = []
            for line in lines[-2:]:
                if not PAGE_NUMBER_PATTERN.match(line):
                    norm = self._normalize_line(line)
                    page_bottom.append(norm)
                    page_bottom.append(self._mask_digits(norm))
            bottom_candidates.append(page_bottom)

        if valid_pages_count < 2:
            return set(), set()

        def _extract_frequent(candidates_list: List[List[str]]) -> Set[str]:
            counts: Counter[str] = Counter()
            for cand_set in candidates_list:
                for item in set(cand_set):
                    counts[item] += 1
            threshold = max(2, int(0.5 * valid_pages_count))
            return {item for item, c in counts.items() if c >= threshold}

        repeated_headers = _extract_frequent(top_candidates)
        repeated_footers = _extract_frequent(bottom_candidates)

        return repeated_headers, repeated_footers

    def clean_page(
        self,
        raw_content: str,
        repeated_headers: Optional[Set[str]] = None,
        repeated_footers: Optional[Set[str]] = None,
    ) -> str:
        """Perform conservative text normalization on a single page of Markdown text."""
        if not raw_content or not raw_content.strip():
            return ""

        repeated_headers = repeated_headers or set()
        repeated_footers = repeated_footers or set()

        # 1. Strip non-printable / control / zero-width characters
        text = CONTROL_AND_INVISIBLE_CHARS.sub("", raw_content)

        # 2. Split lines and strip trailing whitespace
        lines = [l.rstrip() for l in text.splitlines()]

        # 3. Strip initial empty lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        if not lines:
            return ""

        # 4. Remove repeated headers & standalone page number noise from top lines
        while lines:
            first = lines[0].strip()
            if not first:
                lines.pop(0)
                continue
            if PAGE_NUMBER_PATTERN.match(first):
                lines.pop(0)
                continue
            first_norm = self._normalize_line(first)
            first_masked = self._mask_digits(first_norm)
            if first_norm in repeated_headers or first_masked in repeated_headers:
                lines.pop(0)
                continue
            break

        # 5. Remove repeated footers & standalone page number noise from bottom lines
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if PAGE_NUMBER_PATTERN.match(last):
                lines.pop()
                continue
            last_norm = self._normalize_line(last)
            last_masked = self._mask_digits(last_norm)
            if last_norm in repeated_footers or last_masked in repeated_footers:
                lines.pop()
                continue
            break

        # 6. Normalize whitespace within lines without corrupting tables
        cleaned_lines: List[str] = []
        for line in lines:
            if not line:
                cleaned_lines.append("")
                continue

            # Markdown table row: preserve formatting and pipe alignment
            if line.strip().startswith("|") and line.strip().endswith("|") and line.strip().count("|") >= 2:
                cleaned_lines.append(line)
            else:
                # Regular line or list item: preserve up to 4 spaces of indentation
                leading_spaces = len(line) - len(line.lstrip(" "))
                indent = " " * min(leading_spaces, 4) if leading_spaces > 0 else ""
                norm_text = re.sub(r"[ \t]{2,}", " ", line.strip())
                cleaned_lines.append(f"{indent}{norm_text}")

        joined = "\n".join(cleaned_lines)

        # 7. Normalize paragraph spacing and heading separation
        # Collapse 3+ newlines to 2 newlines
        joined = re.sub(r"\n{3,}", "\n\n", joined)

        # Ensure headings have clean double newline before and after (unless start/end of document)
        joined = re.sub(r"([^\n])\n(#{1,6}\s+)", r"\1\n\n\2", joined)
        joined = re.sub(r"^(#{1,6}\s+[^\n]+)\n(?=[^\s#|])", r"\1\n\n", joined, flags=re.MULTILINE)
        joined = re.sub(r"\n{3,}", "\n\n", joined)

        return joined.strip()

    def detect_structure(self, content: str) -> DetectedStructure:
        """Reliably identify structural elements on a page without heuristic guessing."""
        if not content.strip():
            return DetectedStructure()

        lines = content.splitlines()

        # 1. Identify Headings
        headings: List[str] = []
        for line in lines:
            match = HEADING_PATTERN.match(line.strip())
            if match:
                headings.append(match.group(1).strip())

        # 2. Identify Tables
        tables_count = 0
        in_table = False
        has_separator = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                if TABLE_SEPARATOR_PATTERN.match(stripped):
                    has_separator = True
                in_table = True
            else:
                if in_table and has_separator:
                    tables_count += 1
                in_table = False
                has_separator = False
        if in_table and has_separator:
            tables_count += 1

        # 3. Identify Lists
        lists_count = 0
        in_list = False
        for line in lines:
            stripped = line.strip()
            if LIST_ITEM_PATTERN.match(stripped):
                if not in_list:
                    lists_count += 1
                    in_list = True
            elif not stripped:
                in_list = False

        # 4. Count Paragraphs (blocks separated by blank lines that are not headings, tables, or lists)
        blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
        paragraphs_count = 0
        for block in blocks:
            block_lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not block_lines:
                continue
            # Skip if single heading line
            if len(block_lines) == 1 and HEADING_PATTERN.match(block_lines[0]):
                continue
            # Skip if table block
            if all(l.startswith("|") and l.endswith("|") for l in block_lines) and any(
                TABLE_SEPARATOR_PATTERN.match(l) for l in block_lines
            ):
                continue
            # Skip if pure list block
            if all(LIST_ITEM_PATTERN.match(l) for l in block_lines):
                continue
            paragraphs_count += 1

        return DetectedStructure(
            has_headings=len(headings) > 0,
            has_tables=tables_count > 0,
            has_lists=lists_count > 0,
            headings=headings,
            tables_count=tables_count,
            lists_count=lists_count,
            paragraphs_count=paragraphs_count,
        )

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Strip whitespace and lowercase line for comparison."""
        return re.sub(r"\s+", " ", line.strip().lower())

    @staticmethod
    def _mask_digits(line: str) -> str:
        """Replace all numeric digit sequences with '#' to identify template headers/footers."""
        return re.sub(r"\d+", "#", line)

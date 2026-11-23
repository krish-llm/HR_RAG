"""Generation Service coordinating context retrieval, LLM prompt construction, and citation validation."""

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.api.dependencies import SchoolContext
from app.core.exceptions import GenerationException
from app.core.logging_utils import TimedBlock
from app.modules.context.schemas import ContextRequest, SourceCitation
from app.modules.context.service import ContextService
from app.modules.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.modules.generation.provider import BaseLLMProvider
from app.modules.generation.schemas import (
    AnswerRequest,
    AnswerResponse,
    CitationItem,
    LLMAnswerRaw,
)

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)


class AnswerService:
    """Orchestrates query answering: Context Retrieval ➔ Prompting ➔ Generation ➔ Citation Validation."""

    def __init__(
        self,
        context_service: ContextService,
        llm_provider: BaseLLMProvider,
    ) -> None:
        self.context_service = context_service
        self.llm_provider = llm_provider

    def _parse_llm_json(self, raw_text: str) -> LLMAnswerRaw:
        """Parse structured JSON from raw LLM output handling code fences and edge cases."""
        cleaned = raw_text.strip()

        # Strip markdown json code block fences if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from LLM output: %s\nRaw output: %s", exc, raw_text)
            raise GenerationException(
                message=f"LLM returned malformed JSON output: {exc}",
                error_code="MALFORMED_LLM_OUTPUT",
                status_code=502,
            ) from exc

        try:
            return LLMAnswerRaw.model_validate(payload)
        except Exception as exc:
            logger.error("LLM JSON payload failed schema validation: %s\nPayload: %s", exc, payload)
            raise GenerationException(
                message=f"LLM output does not match required answer schema: {exc}",
                error_code="MALFORMED_LLM_OUTPUT",
                status_code=502,
            ) from exc

    def _validate_and_map_citations(
        self,
        raw_answer: LLMAnswerRaw,
        source_manifest: List[SourceCitation],
    ) -> Tuple[List[CitationItem], Optional[str]]:
        """Validate citations against the Phase 7 manifest, filtering out fabricated or unknown tags."""
        manifest_by_label: Dict[str, SourceCitation] = {}
        manifest_by_clean: Dict[str, SourceCitation] = {}

        for item in source_manifest:
            manifest_by_label[item.citation_label.strip().lower()] = item
            clean_tag = re.sub(r"[\[\]]", "", item.citation_label).strip().lower()
            manifest_by_clean[clean_tag] = item

        # Collect candidate tags from both structured citations array and in-text references
        candidate_tags: List[str] = list(raw_answer.citations or [])
        in_text_tags = CITATION_REGEX.findall(raw_answer.answer)
        for tag in in_text_tags:
            if tag not in candidate_tags:
                candidate_tags.append(tag)

        validated_citations: List[CitationItem] = []
        unknown_citations: List[str] = []
        seen_matched_labels: Set[str] = set()

        for tag in candidate_tags:
            clean_tag_str = tag.strip()
            norm_key = clean_tag_str.lower()
            clean_key = re.sub(r"[\[\]]", "", clean_tag_str).strip().lower()

            matched_source: Optional[SourceCitation] = None
            if norm_key in manifest_by_label:
                matched_source = manifest_by_label[norm_key]
            elif clean_key in manifest_by_clean:
                matched_source = manifest_by_clean[clean_key]

            if matched_source is not None:
                if matched_source.citation_label not in seen_matched_labels:
                    seen_matched_labels.add(matched_source.citation_label)
                    validated_citations.append(
                        CitationItem(
                            citation_label=matched_source.citation_label,
                            citation_reference=matched_source.citation_reference,
                            document_id=matched_source.document_id,
                            document_name=matched_source.document_name,
                            version=matched_source.version,
                            page_start=matched_source.page_start,
                            page_end=matched_source.page_end,
                            heading=matched_source.heading,
                        )
                    )
            else:
                if clean_tag_str not in unknown_citations:
                    unknown_citations.append(clean_tag_str)

        citation_limitation: Optional[str] = None
        if unknown_citations:
            logger.warning("Pruned unsupported citation(s) from LLM answer: %s", unknown_citations)
            citation_limitation = (
                f"Removed unsupported citation(s) not present in authoritative context: {', '.join(unknown_citations)}."
            )

        return validated_citations, citation_limitation

    async def generate_answer(
        self,
        request: AnswerRequest,
        school_context: SchoolContext,
    ) -> AnswerResponse:
        """Execute end-to-end grounded question answering pipeline."""
        # 1. Reuse existing Phase 7 ContextService
        context_req = ContextRequest(
            question=request.question,
            employee_context=request.employee_context,
            as_of_date=request.as_of_date,
            top_k=request.top_k,
            token_budget=request.token_budget,
        )

        context_res = await self.context_service.build_context(
            request=context_req,
            school_context=school_context,
        )

        model_name = self.llm_provider.get_model_name()

        # 2. Insufficient Evidence Fast-Path: if no context is available, do not call LLM
        if (
            context_res.status == "empty"
            or context_res.selected_chunks_count == 0
            or not context_res.assembled_context.strip()
            or not context_res.source_manifest
        ):
            logger.info("No usable policy context found for question: '%s'", request.question)
            return AnswerResponse(
                question=request.question,
                answer="The available HR policies do not provide sufficient information to answer this question.",
                citations=[],
                insufficient_evidence=True,
                conflicting_evidence=False,
                limitations="No relevant active policy documents were found matching your school/branch context and query.",
                context_tokens=0,
                model_name=model_name,
                status="insufficient_evidence",
            )

        # 3. Grounded Prompt Construction
        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(
            question=request.question,
            assembled_context=context_res.assembled_context,
        )

        # 4. LLM Generation
        async with TimedBlock(logger, "llm_generation") as timer:
            raw_text = await self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            timer.set_extra(model=model_name)

        # 5. Parse and Validate Raw Output
        raw_answer = self._parse_llm_json(raw_text)

        # Handle empty answer text
        answer_text = raw_answer.answer.strip()
        if not answer_text:
            if raw_answer.insufficient_evidence:
                answer_text = "The available HR policies do not provide sufficient information to answer this question."
            else:
                logger.error("LLM returned an empty answer without insufficient_evidence flag")
                raise GenerationException(
                    message="LLM generated an empty answer",
                    error_code="EMPTY_GENERATION_ERROR",
                    status_code=502,
                )

        # 6. Validate and Map Citations against Source Manifest
        validated_citations, citation_limitation = self._validate_and_map_citations(
            raw_answer=raw_answer,
            source_manifest=context_res.source_manifest,
        )

        # Combine limitations
        all_limitations: List[str] = []
        if raw_answer.limitations and raw_answer.limitations.strip():
            all_limitations.append(raw_answer.limitations.strip())
        if citation_limitation:
            all_limitations.append(citation_limitation)

        final_limitations = " ".join(all_limitations) if all_limitations else None

        status_val = "insufficient_evidence" if raw_answer.insufficient_evidence else "success"

        return AnswerResponse(
            question=request.question,
            answer=answer_text,
            citations=validated_citations,
            insufficient_evidence=raw_answer.insufficient_evidence,
            conflicting_evidence=raw_answer.conflicting_evidence,
            limitations=final_limitations,
            context_tokens=context_res.total_tokens,
            model_name=model_name,
            status=status_val,
        )

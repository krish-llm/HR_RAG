"""FastAPI routes for Policy Context Building & Citation Tracking (Phase 7)."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import (
    SchoolContext,
    Settings,
    get_school_context,
    get_settings,
)
from app.modules.context.schemas import ContextRequest, ContextResponse
from app.modules.context.service import ContextService
from app.modules.retrieval.router import get_retrieval_service
from app.modules.retrieval.service import RetrievalService
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Query & Context"])


def get_context_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    settings: Settings = Depends(get_settings),
) -> ContextService:
    """Dependency provider for ContextService."""
    return ContextService(
        retrieval_service=retrieval_service,
        default_top_k=settings.RERANK_TOP_K,
        default_token_budget=settings.CONTEXT_TOKEN_BUDGET,
    )


@router.post(
    "/context",
    response_model=ApiResponse[ContextResponse],
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded Policy Context",
    description="Retrieves, reranks, deduplicates, and formats policy context with structured citations under strict token budgets.",
)
async def build_policy_context(
    request: ContextRequest,
    school_context: SchoolContext = Depends(get_school_context),
    service: ContextService = Depends(get_context_service),
) -> ApiResponse[ContextResponse]:
    """Assemble grounded context and source manifest for downstream LLM generation."""
    result = await service.build_context(request, school_context)
    return ApiResponse(
        success=True,
        data=result,
    )

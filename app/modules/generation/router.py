"""FastAPI routes for Grounded Policy Question Answering (Phase 8)."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import (
    SchoolContext,
    Settings,
    get_school_context,
    get_settings,
)
from app.core.logging_utils import TimedBlock
from app.logging_config import get_logger
from app.modules.context.router import get_context_service
from app.modules.context.service import ContextService
from app.modules.generation.provider import AzureOpenAIChatProvider
from app.modules.generation.schemas import AnswerRequest, AnswerResponse
from app.modules.generation.service import AnswerService
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Query & Answering"])
logger = get_logger(__name__)


def get_answer_service(
    context_service: ContextService = Depends(get_context_service),
    settings: Settings = Depends(get_settings),
) -> AnswerService:
    """Dependency provider for AnswerService with configured Azure OpenAI LLM provider."""
    llm_provider = AzureOpenAIChatProvider(
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=settings.AZURE_OPENAI_CHAT_TEMPERATURE,
        max_tokens=settings.AZURE_OPENAI_CHAT_MAX_TOKENS,
        timeout_seconds=settings.AZURE_OPENAI_TIMEOUT_SECONDS,
    )
    return AnswerService(
        context_service=context_service,
        llm_provider=llm_provider,
    )


@router.post(
    "/answer",
    response_model=ApiResponse[AnswerResponse],
    status_code=status.HTTP_200_OK,
    summary="Grounded Policy Question Answering",
    description="Executes the full RAG pipeline: Retrieval ➔ Reranking ➔ Context Assembly ➔ Grounded LLM Generation ➔ Citation Validation.",
)
async def generate_policy_answer(
    request: AnswerRequest,
    school_context: SchoolContext = Depends(get_school_context),
    service: AnswerService = Depends(get_answer_service),
) -> ApiResponse[AnswerResponse]:
    """Generate an authoritative, strictly-grounded HR policy answer with validated source citations."""
    async with TimedBlock(logger, "qa_pipeline") as timer:
        response = await service.generate_answer(request, school_context)
        timer.set_extra(status=response.status, tokens=response.context_tokens)
    return ApiResponse(
        success=True,
        data=response,
    )

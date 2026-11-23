"""LLM Provider abstraction and Azure OpenAI implementation for Phase 8."""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Optional
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    RateLimitError,
)

from app.core.exceptions import AppException, GenerationException

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract interface defining the contract for generative LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response text for the given system and user prompts.

        Args:
            system_prompt: High-priority system instructions.
            user_prompt: User query and formatted context.
            temperature: Optional sampling temperature override.
            max_tokens: Optional maximum completion tokens override.

        Returns:
            Raw string completion from the model.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical model name or deployment ID."""
        pass


class AzureOpenAIChatProvider(BaseLLMProvider):
    """Azure OpenAI Chat Completions provider implementation."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 30.0,
        client: Optional[AsyncAzureOpenAI] = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

        if client is not None:
            self.client = client
        else:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=api_key or "dummy-key",
                api_version=self.api_version,
                timeout=self.timeout_seconds,
            )

    def get_model_name(self) -> str:
        """Return the Azure OpenAI chat deployment name."""
        return self.deployment_name

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Azure OpenAI Chat Completions with JSON mode."""
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    response_format={"type": "json_object"},
                ),
                timeout=self.timeout_seconds,
            )
        except (APITimeoutError, asyncio.TimeoutError) as exc:
            logger.error("Azure OpenAI chat request timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise GenerationException(
                message=f"Azure OpenAI chat request timed out after {self.timeout_seconds}s",
                error_code="AZURE_OPENAI_TIMEOUT",
                status_code=504,
            ) from exc
        except RateLimitError as exc:
            logger.error("Azure OpenAI rate limit encountered: %s", exc)
            raise GenerationException(
                message="Azure OpenAI service rate limit exceeded. Please retry shortly.",
                error_code="AZURE_OPENAI_RATE_LIMIT",
                status_code=429,
            ) from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error("Azure OpenAI API status/connection error: %s", exc)
            raise GenerationException(
                message=f"Azure OpenAI service error: {exc}",
                error_code="AZURE_OPENAI_API_ERROR",
                status_code=502,
            ) from exc
        except AppException:
            raise
        except Exception as exc:
            logger.error("Unexpected error during Azure OpenAI chat generation: %s", exc)
            raise GenerationException(
                message=f"Azure OpenAI chat generation failed: {exc}",
                error_code="AZURE_OPENAI_ERROR",
                status_code=502,
            ) from exc

        if not response or not response.choices:
            raise GenerationException(
                message="Azure OpenAI returned an empty choices list",
                error_code="EMPTY_GENERATION_ERROR",
                status_code=502,
            )

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise GenerationException(
                message="Azure OpenAI returned empty response content",
                error_code="EMPTY_GENERATION_ERROR",
                status_code=502,
            )

        return content

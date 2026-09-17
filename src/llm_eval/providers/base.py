"""The provider abstraction every model backend implements.

Evaluation logic, the RAG pipeline, and the dashboard depend only on this
interface -- never on a concrete provider -- so swapping a mock for a
real model (or one real model for another) never requires touching
anything downstream of `generate()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """A single request to generate text from a model."""

    prompt: str = Field(..., min_length=1)
    model: str = Field(..., description="Provider-specific model identifier.")
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra parameters."
    )


class GenerationResponse(BaseModel):
    """The result of a generation call, with the metadata evaluation needs."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific raw response, for debugging."
    )


class ModelProvider(ABC):
    """Abstract interface every model backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider, e.g. 'mock', 'huggingface'."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the given request."""
        raise NotImplementedError

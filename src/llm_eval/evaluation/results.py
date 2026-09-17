"""What a model/system under test actually produced for one EvalExample.

Kept separate from EvalExample (Phase 1) deliberately: EvalExample is the
static ground truth, EvalResult is the runtime output of running something
against it. Retrieval fields are included now (empty by default) so this
schema doesn't need to change shape when the RAG system arrives in Phase 5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """The output of running one EvalExample through a system under test."""

    example_id: str
    question: str
    reference_answer: str
    generated_answer: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retrieved_document_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

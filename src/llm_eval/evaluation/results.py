"""What a model/system under test actually produced for one EvalExample.

Kept separate from EvalExample (Phase 1) deliberately: EvalExample is the
static ground truth, EvalResult is the runtime output of running something
against it. Retrieval fields are included now (empty by default) so this
schema doesn't need to change shape when the RAG system arrives in Phase 5.

prompt_id/prompt_version (Phase 4) are tracked as their own fields, separate
from model/model version, so a later regression check can isolate whether a
quality change came from the prompt or from something else.
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
    prompt_id: str = "qa_direct"
    prompt_version: str = "v1"
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retrieved_document_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

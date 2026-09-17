"""Core data model for evaluation datasets.

An EvalExample is the static, hand-curated input to an evaluation run: a
question plus its ground-truth answer. It intentionally does NOT contain
anything produced by running a model against it (a generated answer,
retrieved documents, latency) -- that belongs to a separate result schema
(added in Phase 3), which keeps "what we're testing against" cleanly
separate from "what the system under test produced."
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvalExample(BaseModel):
    """One row of an evaluation dataset."""

    id: str = Field(..., description="Stable unique identifier within its dataset.")
    question: str = Field(..., min_length=1)
    reference_answer: str = Field(
        ..., min_length=1, description="Ground-truth answer used for correctness scoring."
    )
    relevant_document_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of documents that should be retrieved for this question, used for "
            "RAG retrieval-quality metrics. Empty for non-RAG examples."
        ),
    )
    category: str | None = Field(
        default=None, description="Optional grouping, e.g. 'factual', 'reasoning', 'rag'."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "question", "reference_answer")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace-only")
        return v


class EvalDataset(BaseModel):
    """A named, versioned collection of EvalExamples."""

    name: str = Field(..., min_length=1)
    version: str = Field(
        ...,
        min_length=1,
        description=(
            "Bump when examples are added or changed, so experiment tracking "
            "can pin to an exact snapshot."
        ),
    )
    examples: list[EvalExample] = Field(..., min_length=1)

    @field_validator("examples")
    @classmethod
    def ids_must_be_unique(cls, examples: list[EvalExample]) -> list[EvalExample]:
        ids = [e.id for e in examples]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate example ids in dataset: {duplicates}")
        return examples

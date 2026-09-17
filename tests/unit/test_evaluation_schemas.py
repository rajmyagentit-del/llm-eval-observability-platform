"""Unit tests for the EvalExample / EvalDataset schema."""

import pytest
from pydantic import ValidationError

from llm_eval.evaluation.schemas import EvalDataset, EvalExample


def test_eval_example_valid() -> None:
    example = EvalExample(id="q1", question="What is 2+2?", reference_answer="4")
    assert example.id == "q1"
    assert example.relevant_document_ids == []
    assert example.metadata == {}


def test_eval_example_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        EvalExample(id="q1", question="   ", reference_answer="4")


def test_eval_dataset_rejects_duplicate_ids() -> None:
    examples = [
        EvalExample(id="q1", question="A?", reference_answer="a"),
        EvalExample(id="q1", question="B?", reference_answer="b"),
    ]
    with pytest.raises(ValidationError):
        EvalDataset(name="d", version="v1", examples=examples)


def test_eval_dataset_rejects_empty_examples() -> None:
    with pytest.raises(ValidationError):
        EvalDataset(name="d", version="v1", examples=[])

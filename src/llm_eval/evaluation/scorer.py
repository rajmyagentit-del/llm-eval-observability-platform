"""Combines an EvalResult with deterministic text metrics into a score record."""

from __future__ import annotations

from pydantic import BaseModel

from llm_eval.evaluation.metrics import exact_match, token_f1
from llm_eval.evaluation.results import EvalResult


class EvalScore(BaseModel):
    """Deterministic scores for one EvalResult."""

    example_id: str
    exact_match: bool
    token_f1: float


def score_result(result: EvalResult) -> EvalScore:
    return EvalScore(
        example_id=result.example_id,
        exact_match=exact_match(result.generated_answer, result.reference_answer),
        token_f1=token_f1(result.generated_answer, result.reference_answer),
    )


def score_results(results: list[EvalResult]) -> list[EvalScore]:
    return [score_result(r) for r in results]

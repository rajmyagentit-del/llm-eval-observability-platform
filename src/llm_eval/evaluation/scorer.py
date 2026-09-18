"""Combines an EvalResult with evaluation metrics into a score record.

exact_match/token_f1 are always computed (free, deterministic). Semantic
similarity and the LLM-judge verdict are optional -- None unless a
semantic_scorer or judge_provider is explicitly passed in -- because both
require extra dependencies or a model call, and neither should be a silent
default.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_eval.evaluation.llm_judge import llm_judge
from llm_eval.evaluation.metrics import exact_match, token_f1
from llm_eval.evaluation.results import EvalResult
from llm_eval.evaluation.semantic import SemanticSimilarityScorer
from llm_eval.providers.base import ModelProvider


class EvalScore(BaseModel):
    """Scores for one EvalResult."""

    example_id: str
    exact_match: bool
    token_f1: float
    semantic_similarity: float | None = None
    llm_judge_verdict: bool | None = None


def score_result(
    result: EvalResult,
    semantic_scorer: SemanticSimilarityScorer | None = None,
    judge_provider: ModelProvider | None = None,
    judge_model: str = "judge",
) -> EvalScore:
    semantic_similarity = None
    if semantic_scorer is not None:
        semantic_similarity = semantic_scorer.similarity(
            result.generated_answer, result.reference_answer
        )

    llm_judge_verdict = None
    if judge_provider is not None:
        llm_judge_verdict = llm_judge(
            judge_provider,
            question=result.question,
            reference_answer=result.reference_answer,
            candidate_answer=result.generated_answer,
            model=judge_model,
        )

    return EvalScore(
        example_id=result.example_id,
        exact_match=exact_match(result.generated_answer, result.reference_answer),
        token_f1=token_f1(result.generated_answer, result.reference_answer),
        semantic_similarity=semantic_similarity,
        llm_judge_verdict=llm_judge_verdict,
    )


def score_results(
    results: list[EvalResult],
    semantic_scorer: SemanticSimilarityScorer | None = None,
    judge_provider: ModelProvider | None = None,
    judge_model: str = "judge",
) -> list[EvalScore]:
    return [
        score_result(
            r,
            semantic_scorer=semantic_scorer,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )
        for r in results
    ]

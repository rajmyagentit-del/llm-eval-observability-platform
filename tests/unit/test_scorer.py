"""Unit tests for the scorer."""

from llm_eval.evaluation.llm_judge import _JUDGE_PROMPT_TEMPLATE
from llm_eval.evaluation.results import EvalResult
from llm_eval.evaluation.scorer import score_result
from llm_eval.providers.mock import MockProvider


def _make_result(generated: str, reference: str) -> EvalResult:
    return EvalResult(
        example_id="q1",
        question="Q?",
        reference_answer=reference,
        generated_answer=generated,
        model="mock-1",
        provider="mock",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
    )


def test_score_result_exact_match_and_f1_agree_on_perfect_answer() -> None:
    score = score_result(_make_result("Paris", "Paris"))
    assert score.exact_match is True
    assert score.token_f1 == 1.0


def test_score_result_partial_credit_when_not_exact() -> None:
    score = score_result(_make_result("The capital of France is Paris", "Paris"))
    assert score.exact_match is False
    assert 0.0 < score.token_f1 < 1.0


def test_score_result_optional_metrics_none_by_default() -> None:
    score = score_result(_make_result("4", "4"))
    assert score.semantic_similarity is None
    assert score.llm_judge_verdict is None


def test_score_result_includes_llm_judge_verdict_when_provider_given() -> None:
    result = _make_result("4", "4")
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=result.question,
        reference_answer=result.reference_answer,
        candidate_answer=result.generated_answer,
    )
    judge = MockProvider(responses={prompt: "CORRECT"})

    score = score_result(result, judge_provider=judge, judge_model="mock-judge")
    assert score.llm_judge_verdict is True

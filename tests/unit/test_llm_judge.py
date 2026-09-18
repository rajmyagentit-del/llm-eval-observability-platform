"""Unit tests for the LLM-as-judge function, using MockProvider (no real model)."""

from llm_eval.evaluation.llm_judge import _JUDGE_PROMPT_TEMPLATE, llm_judge
from llm_eval.providers.mock import MockProvider


def _prompt(question: str, reference: str, candidate: str) -> str:
    return _JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference_answer=reference, candidate_answer=candidate
    )


def test_llm_judge_true_for_correct_verdict() -> None:
    prompt = _prompt("What is 2+2?", "4", "4")
    provider = MockProvider(responses={prompt: "CORRECT"})
    assert llm_judge(provider, "What is 2+2?", "4", "4", model="mock-judge") is True


def test_llm_judge_false_for_incorrect_verdict() -> None:
    prompt = _prompt("What is 2+2?", "4", "5")
    provider = MockProvider(responses={prompt: "INCORRECT"})
    assert llm_judge(provider, "What is 2+2?", "4", "5", model="mock-judge") is False


def test_llm_judge_none_for_unparseable_verdict() -> None:
    prompt = _prompt("What is 2+2?", "4", "banana")
    provider = MockProvider(responses={prompt: "I am not sure what you mean"})
    assert llm_judge(provider, "What is 2+2?", "4", "banana", model="mock-judge") is None


def test_llm_judge_checks_incorrect_before_correct_substring() -> None:
    # "INCORRECT" contains "CORRECT" as a substring -- must not misparse.
    prompt = _prompt("Q", "R", "C")
    provider = MockProvider(responses={prompt: "INCORRECT"})
    assert llm_judge(provider, "Q", "R", "C", model="mock-judge") is False

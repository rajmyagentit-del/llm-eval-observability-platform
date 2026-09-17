"""Unit tests for the evaluation runner."""

from llm_eval.evaluation.runner import run_dataset, run_example
from llm_eval.evaluation.schemas import EvalDataset, EvalExample
from llm_eval.providers.mock import MockProvider


def test_run_example_produces_matching_eval_result() -> None:
    example = EvalExample(id="q1", question="What is 2+2?", reference_answer="4")
    provider = MockProvider(responses={"What is 2+2?": "4"})

    result = run_example(provider, example, model="mock-1")

    assert result.example_id == "q1"
    assert result.generated_answer == "4"
    assert result.reference_answer == "4"
    assert result.provider == "mock"
    assert result.prompt_tokens > 0
    assert result.latency_ms > 0


def test_run_dataset_runs_every_example() -> None:
    dataset = EvalDataset(
        name="d",
        version="v1",
        examples=[
            EvalExample(id="q1", question="A?", reference_answer="a"),
            EvalExample(id="q2", question="B?", reference_answer="b"),
        ],
    )
    provider = MockProvider(responses={"A?": "a", "B?": "b"})

    results = run_dataset(provider, dataset, model="mock-1")

    assert len(results) == 2
    assert {r.example_id for r in results} == {"q1", "q2"}
    assert all(r.generated_answer for r in results)

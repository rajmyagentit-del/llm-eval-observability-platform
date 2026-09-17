"""End-to-end test wiring dataset loading, running, and scoring together."""

from llm_eval.evaluation.runner import run_dataset
from llm_eval.evaluation.scorer import score_results
from llm_eval.providers.mock import MockProvider
from llm_eval.storage.dataset_loader import load_dataset


def test_end_to_end_mock_evaluation_pipeline() -> None:
    dataset = load_dataset("data/eval/sample_dataset.json")
    # Configure the mock to answer every question correctly, so a perfect
    # score end-to-end proves the full load -> run -> score pipeline wires
    # together correctly.
    responses = {ex.question: ex.reference_answer for ex in dataset.examples}
    provider = MockProvider(responses=responses)

    results = run_dataset(provider, dataset, model="mock-perfect")
    scores = score_results(results)

    assert len(scores) == len(dataset.examples)
    assert all(s.exact_match for s in scores)
    assert all(s.token_f1 == 1.0 for s in scores)

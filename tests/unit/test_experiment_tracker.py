"""Unit tests for experiment tracking."""

import time
from pathlib import Path

import pytest

from llm_eval.evaluation.results import EvalResult
from llm_eval.evaluation.scorer import EvalScore
from llm_eval.experiments.tracker import list_runs, load_run, record_run, summarize_metrics


def _sample_result(example_id: str = "q1") -> EvalResult:
    return EvalResult(
        example_id=example_id,
        question="What is 2+2?",
        reference_answer="4",
        generated_answer="4",
        model="mock-1",
        provider="mock",
        prompt_tokens=5,
        completion_tokens=1,
        latency_ms=12.5,
    )


def _sample_score(example_id: str = "q1") -> EvalScore:
    return EvalScore(example_id=example_id, exact_match=True, token_f1=1.0)


def test_summarize_metrics_computes_core_aggregates() -> None:
    metrics = summarize_metrics([_sample_score()], [_sample_result()])

    assert metrics["exact_match_rate"] == 1.0
    assert metrics["mean_token_f1"] == 1.0
    assert metrics["mean_latency_ms"] == 12.5
    assert metrics["total_prompt_tokens"] == 5.0
    assert metrics["total_completion_tokens"] == 1.0


def test_summarize_metrics_raises_on_empty_scores() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_metrics([], [])


def test_summarize_metrics_includes_optional_metrics_when_present() -> None:
    score = EvalScore(
        example_id="q1",
        exact_match=True,
        token_f1=1.0,
        semantic_similarity=0.9,
        llm_judge_verdict=True,
    )

    metrics = summarize_metrics([score], [_sample_result()])

    assert metrics["mean_semantic_similarity"] == 0.9
    assert metrics["llm_judge_pass_rate"] == 1.0


def test_record_run_persists_and_load_run_round_trips(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"

    run = record_run(
        experiment_name="baseline_check",
        model="mock-1",
        provider="mock",
        prompt_id="qa_direct",
        prompt_version="v1",
        dataset_name="sample_qa",
        dataset_version="1",
        results=[_sample_result()],
        scores=[_sample_score()],
        experiments_dir=experiments_dir,
    )

    loaded = load_run(run.run_id, experiments_dir=experiments_dir)

    assert loaded.run_id == run.run_id
    assert loaded.metrics["exact_match_rate"] == 1.0
    assert loaded.results[0].example_id == "q1"


def test_load_run_raises_for_unknown_run_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run("does-not-exist", experiments_dir=tmp_path)


def test_list_runs_returns_most_recent_first(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"

    first = record_run(
        experiment_name="run1",
        model="mock-1",
        provider="mock",
        prompt_id="qa_direct",
        prompt_version="v1",
        dataset_name="sample_qa",
        dataset_version="1",
        results=[_sample_result()],
        scores=[_sample_score()],
        experiments_dir=experiments_dir,
    )
    time.sleep(0.01)
    second = record_run(
        experiment_name="run2",
        model="mock-1",
        provider="mock",
        prompt_id="qa_direct",
        prompt_version="v1",
        dataset_name="sample_qa",
        dataset_version="1",
        results=[_sample_result()],
        scores=[_sample_score()],
        experiments_dir=experiments_dir,
    )

    run_ids = list_runs(experiments_dir=experiments_dir)

    assert run_ids[0] == second.run_id
    assert first.run_id in run_ids


def test_list_runs_returns_empty_list_when_directory_missing(tmp_path: Path) -> None:
    assert list_runs(experiments_dir=tmp_path / "does-not-exist") == []

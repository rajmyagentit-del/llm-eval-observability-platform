"""Lightweight experiment tracking.

WHY NOT MLFLOW (yet): MLflow is a legitimate, widely-used choice for this
job, but it's a real dependency with its own server/UI/storage-backend
concepts to learn and operate. Per this project's "don't overengineer" rule,
we start with the simplest thing that actually answers the questions we need
answered right now (what ran, with which model/prompt/dataset, and what were
the metrics): plain Pydantic models serialized to JSON files. If the
dashboard phase would clearly benefit from MLflow's UI for browsing runs, we
revisit that decision explicitly, with the tradeoff spelled out, rather than
defaulting to it because it's a well-known name.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_eval.evaluation.results import EvalResult
from llm_eval.evaluation.scorer import EvalScore

DEFAULT_EXPERIMENTS_DIR = Path("reports/experiments")


class ExperimentRun(BaseModel):
    """Everything needed to reproduce and compare one evaluation run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_name: str
    model: str
    provider: str
    prompt_id: str
    prompt_version: str
    dataset_name: str
    dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    results: list[EvalResult]
    scores: list[EvalScore]
    metrics: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


def summarize_metrics(scores: list[EvalScore], results: list[EvalResult]) -> dict[str, float]:
    """Aggregate per-example scores/results into run-level metrics.

    Kept as a plain function (not a method) so it can be unit-tested against
    hand-built lists without constructing a full ExperimentRun first.
    """
    if not scores:
        raise ValueError("Cannot summarize metrics for an empty list of scores.")

    n = len(scores)
    metrics: dict[str, float] = {
        "exact_match_rate": sum(1 for s in scores if s.exact_match) / n,
        "mean_token_f1": sum(s.token_f1 for s in scores) / n,
    }

    if results:
        metrics["mean_latency_ms"] = sum(r.latency_ms for r in results) / len(results)
        metrics["total_prompt_tokens"] = float(sum(r.prompt_tokens for r in results))
        metrics["total_completion_tokens"] = float(sum(r.completion_tokens for r in results))

    semantic_scores = [s.semantic_similarity for s in scores if s.semantic_similarity is not None]
    if semantic_scores:
        metrics["mean_semantic_similarity"] = sum(semantic_scores) / len(semantic_scores)

    judge_verdicts = [s.llm_judge_verdict for s in scores if s.llm_judge_verdict is not None]
    if judge_verdicts:
        metrics["llm_judge_pass_rate"] = sum(1 for v in judge_verdicts if v) / len(judge_verdicts)

    return metrics


def record_run(
    experiment_name: str,
    model: str,
    provider: str,
    prompt_id: str,
    prompt_version: str,
    dataset_name: str,
    dataset_version: str,
    results: list[EvalResult],
    scores: list[EvalScore],
    metadata: dict[str, Any] | None = None,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> ExperimentRun:
    """Build an ExperimentRun, persist it to disk, and return it."""
    metrics = summarize_metrics(scores, results)
    run = ExperimentRun(
        experiment_name=experiment_name,
        model=model,
        provider=provider,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        results=results,
        scores=scores,
        metrics=metrics,
        metadata=metadata or {},
    )

    experiments_dir.mkdir(parents=True, exist_ok=True)
    run_path = experiments_dir / f"{run.run_id}.json"
    run_path.write_text(run.model_dump_json(indent=2))

    return run


def load_run(run_id: str, experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR) -> ExperimentRun:
    """Load a previously recorded ExperimentRun by its run_id."""
    run_path = experiments_dir / f"{run_id}.json"
    if not run_path.exists():
        raise FileNotFoundError(
            f"No experiment run found with run_id={run_id!r} in {experiments_dir}"
        )
    return ExperimentRun.model_validate_json(run_path.read_text())


def list_runs(experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR) -> list[str]:
    """List run_ids under experiments_dir, most recently modified first."""
    if not experiments_dir.exists():
        return []
    run_files = sorted(
        experiments_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [p.stem for p in run_files]

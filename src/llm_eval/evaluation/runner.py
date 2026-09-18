"""Runs a ModelProvider against an evaluation dataset."""

from __future__ import annotations

from typing import Any

from llm_eval.evaluation.prompts import PROMPT_V1, PromptTemplate
from llm_eval.evaluation.results import EvalResult
from llm_eval.evaluation.schemas import EvalDataset, EvalExample
from llm_eval.providers.base import GenerationRequest, ModelProvider


def run_example(
    provider: ModelProvider,
    example: EvalExample,
    model: str,
    prompt: PromptTemplate = PROMPT_V1,
    **generation_kwargs: Any,
) -> EvalResult:
    """Run a single EvalExample through a ModelProvider and package the result."""
    rendered_prompt = prompt.render(example.question)
    request = GenerationRequest(prompt=rendered_prompt, model=model, **generation_kwargs)
    response = provider.generate(request)
    return EvalResult(
        example_id=example.id,
        question=example.question,
        reference_answer=example.reference_answer,
        generated_answer=response.text,
        model=response.model,
        provider=provider.name,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
    )


def run_dataset(
    provider: ModelProvider,
    dataset: EvalDataset,
    model: str,
    prompt: PromptTemplate = PROMPT_V1,
    **generation_kwargs: Any,
) -> list[EvalResult]:
    """Run every example in a dataset through a ModelProvider."""
    return [
        run_example(provider, example, model, prompt=prompt, **generation_kwargs)
        for example in dataset.examples
    ]

"""A deterministic, fully offline model provider.

Exists so evaluation logic, RAG pipelines, and CI tests never depend on
network access, an API key, or a real model's non-determinism. Given the
same prompt, it always returns the same answer -- which is exactly what
lets tests inject a known-good, known-bad, or known-hallucinated response
and assert the evaluation engine scores it correctly.
"""

from __future__ import annotations

import hashlib

from llm_eval.providers.base import GenerationRequest, GenerationResponse, ModelProvider


class MockProvider(ModelProvider):
    """Returns a caller-configured answer if given one, else a deterministic
    placeholder derived from the prompt.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        text = self._responses.get(request.prompt) or self._fallback_text(request.prompt)
        return GenerationResponse(
            text=text,
            model=request.model,
            prompt_tokens=self._approx_token_count(request.prompt),
            completion_tokens=self._approx_token_count(text),
            latency_ms=self._pseudo_latency_ms(request.prompt),
            raw={"provider": "mock"},
        )

    @staticmethod
    def _fallback_text(prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"[mock-{digest}] This is a deterministic placeholder answer for: {prompt}"

    @staticmethod
    def _approx_token_count(text: str) -> int:
        # Whitespace-split word count as a cheap, dependency-free proxy.
        # Replaced by a real tokenizer's count once a real provider is added.
        return len(text.split())

    @staticmethod
    def _pseudo_latency_ms(prompt: str) -> float:
        # Deterministic pseudo-latency derived from the prompt: stable across
        # repeated calls, but varies by prompt -- useful for later testing of
        # latency-percentile logic without introducing real randomness.
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return 20.0 + (int(digest[:8], 16) % 200)

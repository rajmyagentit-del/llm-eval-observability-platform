"""Unit tests for the ModelProvider interface and MockProvider."""

import pytest

from llm_eval.providers.base import GenerationRequest, ModelProvider
from llm_eval.providers.mock import MockProvider


def test_model_provider_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ModelProvider()  # type: ignore[abstract]


def test_mock_provider_name() -> None:
    assert MockProvider().name == "mock"


def test_mock_provider_is_deterministic() -> None:
    provider = MockProvider()
    request = GenerationRequest(prompt="What is 2+2?", model="mock-1")
    first = provider.generate(request)
    second = provider.generate(request)
    assert first.text == second.text
    assert first.latency_ms == second.latency_ms
    assert first.prompt_tokens == second.prompt_tokens


def test_mock_provider_uses_configured_response() -> None:
    provider = MockProvider(responses={"What is 2+2?": "4"})
    request = GenerationRequest(prompt="What is 2+2?", model="mock-1")
    response = provider.generate(request)
    assert response.text == "4"
    assert response.completion_tokens == 1


def test_mock_provider_fallback_is_clearly_synthetic() -> None:
    provider = MockProvider()
    request = GenerationRequest(prompt="Unconfigured prompt", model="mock-1")
    response = provider.generate(request)
    assert response.text.startswith("[mock-")


def test_mock_provider_different_prompts_differ() -> None:
    provider = MockProvider()
    r1 = provider.generate(GenerationRequest(prompt="Prompt A", model="mock-1"))
    r2 = provider.generate(GenerationRequest(prompt="Prompt B", model="mock-1"))
    assert r1.text != r2.text

"""Lightweight tests that must pass with or without the 'local-model' extra installed."""

from llm_eval.providers.local_transformers import LocalTransformersProvider


def test_local_transformers_provider_name() -> None:
    assert LocalTransformersProvider().name == "local-transformers"


def test_local_transformers_provider_constructs_without_loading_model() -> None:
    # Constructing the provider must not trigger a model download/load --
    # that only happens lazily inside generate().
    provider = LocalTransformersProvider()
    assert provider._model is None
    assert provider._tokenizer is None

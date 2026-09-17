"""End-to-end test of the real local model. Requires the 'local-model' extra
and downloads model weights on first run. Excluded from the default test run;
invoke explicitly with: pytest -m integration -v
"""

import pytest

pytest.importorskip("transformers", reason="requires the 'local-model' extra")

from llm_eval.providers.base import GenerationRequest
from llm_eval.providers.local_transformers import LocalTransformersProvider


@pytest.mark.integration
def test_local_transformers_generates_real_text() -> None:
    provider = LocalTransformersProvider()
    request = GenerationRequest(
        prompt="What is the capital of France? Answer in one word.",
        model="google/flan-t5-small",
        max_tokens=16,
    )
    response = provider.generate(request)
    assert response.text
    assert response.completion_tokens > 0
    assert response.latency_ms > 0

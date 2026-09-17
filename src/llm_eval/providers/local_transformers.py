"""A locally-run, open-source model provider using Hugging Face transformers.

Runs entirely offline once the model weights are cached -- no API key, no
network dependency at generation time, and no third-party free-tier policy
that could break the public demo out from under us.

Requires the optional 'local-model' dependency group:
    pip install -e ".[local-model]"

Loads the tokenizer/model directly (AutoTokenizer + AutoModelForSeq2SeqLM)
rather than using the high-level `pipeline()` convenience wrapper: pipeline
task names have shifted across transformers versions, while the tokenizer
and model classes are the stable, version-independent building blocks. This
also gives real tokenizer-based token counts instead of a word-count guess.

Imports are deferred to first use, so importing this module (and even
constructing LocalTransformersProvider) costs nothing if the 'local-model'
extra isn't installed -- only calling .generate() requires it.
"""

from __future__ import annotations

import time
from typing import Any

from llm_eval.providers.base import GenerationRequest, GenerationResponse, ModelProvider

DEFAULT_MODEL = "google/flan-t5-small"


class LocalTransformersProvider(ModelProvider):
    """Runs a small, CPU-friendly, instruction-tuned open model locally."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def name(self) -> str:
        return "local-transformers"

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "LocalTransformersProvider requires the 'local-model' extra. "
                    "Install it with: pip install -e '.[local-model]'"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        return self._model, self._tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        model, tokenizer = self._load_model()
        prompt = (
            f"{request.system_prompt}\n\n{request.prompt}"
            if request.system_prompt
            else request.prompt
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        generation_kwargs: dict[str, Any] = {"max_new_tokens": request.max_tokens}
        if request.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = request.temperature
        else:
            generation_kwargs["do_sample"] = False

        start = time.perf_counter()
        output_ids = model.generate(**inputs, **generation_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return GenerationResponse(
            text=text,
            model=self._model_name,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(output_ids.shape[-1]),
            latency_ms=latency_ms,
            raw={"provider": "local-transformers"},
        )

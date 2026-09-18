"""Embedding-based semantic similarity, for catching "correct but reworded" answers.

Deterministic metrics (metrics.py) reward word overlap: a correct
paraphrase scores low on token_f1 even though its meaning is right.
Semantic similarity fixes that specific blind spot by comparing sentence
embeddings instead of surface tokens -- but introduces its own blind spot
(a confidently wrong answer that's topically similar can still score high),
so it is used ALONGSIDE deterministic metrics, never as a replacement.

Requires the optional 'semantic' dependency group:
    pip install -e ".[semantic]"

The sentence_transformers import is deferred to first use, matching the
lazy-import pattern used by LocalTransformersProvider.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSimilarityScorer:
    """Cosine similarity between sentence embeddings of two texts."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "SemanticSimilarityScorer requires the 'semantic' extra. "
                    "Install it with: pip install -e '.[semantic]'"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def similarity(self, prediction: str, reference: str) -> float:
        model = self._load_model()
        from sentence_transformers import util

        embeddings = model.encode([prediction, reference])
        score = util.cos_sim(embeddings[0], embeddings[1])
        return float(score.item())

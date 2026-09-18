"""Real embedding-model tests. Requires the 'semantic' extra; excluded from
the default run. Invoke explicitly with: pytest -m integration -v
"""

import pytest

pytest.importorskip("sentence_transformers", reason="requires the 'semantic' extra")

from llm_eval.evaluation.semantic import SemanticSimilarityScorer


@pytest.mark.integration
def test_semantic_similarity_high_for_paraphrase() -> None:
    scorer = SemanticSimilarityScorer()
    score = scorer.similarity("The capital of France is Paris.", "Paris is France's capital.")
    assert score > 0.7


@pytest.mark.integration
def test_semantic_similarity_low_for_unrelated_answers() -> None:
    scorer = SemanticSimilarityScorer()
    score = scorer.similarity("Paris is the capital of France.", "Bananas are yellow fruit.")
    assert score < 0.5

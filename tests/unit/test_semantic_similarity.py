"""Lightweight test that must pass without the 'semantic' extra installed."""

from llm_eval.evaluation.semantic import SemanticSimilarityScorer


def test_semantic_similarity_scorer_constructs_without_loading_model() -> None:
    scorer = SemanticSimilarityScorer()
    assert scorer._model is None

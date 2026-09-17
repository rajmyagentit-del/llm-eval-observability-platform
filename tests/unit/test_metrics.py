"""Unit tests for deterministic text metrics."""

from llm_eval.evaluation.metrics import exact_match, normalize_text, token_f1


def test_normalize_text_strips_punctuation_articles_and_case() -> None:
    assert normalize_text("The Capital is Paris.") == "capital is paris"


def test_exact_match_true_for_equivalent_phrasing() -> None:
    assert exact_match("The capital is Paris.", "capital is paris") is True


def test_exact_match_false_for_different_answer() -> None:
    assert exact_match("The capital is Berlin.", "Paris") is False


def test_token_f1_perfect_match() -> None:
    assert token_f1("Paris", "Paris") == 1.0


def test_token_f1_partial_overlap() -> None:
    score = token_f1("The capital of France is Paris", "Paris")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap_is_zero() -> None:
    assert token_f1("Berlin", "Paris") == 0.0


def test_token_f1_both_empty_is_one() -> None:
    assert token_f1("", "") == 1.0

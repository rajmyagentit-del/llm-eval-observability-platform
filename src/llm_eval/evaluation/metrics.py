"""Deterministic, dependency-free text evaluation metrics.

These run instantly with no model call and no network access, so they
never introduce non-determinism into scoring. Semantic (embedding-based)
similarity -- which understands paraphrase and meaning rather than surface
wording -- is a separate, optional metric added in Part B; the two are
complementary: exact/F1 metrics catch obviously wrong answers cheaply,
semantic similarity catches "correct but reworded" answers that
word-overlap metrics would unfairly penalize.
"""

from __future__ import annotations

import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace.

    Standard SQuAD-style normalization: "The capital is Paris." and
    "capital is paris" are treated as equivalent, without claiming to
    understand meaning the way semantic similarity does.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, reference: str) -> bool:
    """True if the normalized prediction exactly equals the normalized reference."""
    return normalize_text(prediction) == normalize_text(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between prediction and reference (SQuAD-style).

    More forgiving than exact_match: a prediction containing the right
    answer plus extra words still scores partial credit.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

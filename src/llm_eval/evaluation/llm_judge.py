"""LLM-as-judge evaluation: using a ModelProvider to judge answer correctness.

LIMITATIONS -- read before trusting this metric:

- Position/verbosity bias: judges have been shown to favor longer or
  earlier-presented answers regardless of actual correctness.
- Self-preference bias: a model judging its own (or a very similar model's)
  outputs tends to score them higher than an independent judge would.
- Prompt sensitivity: small wording changes in the judge prompt can shift
  verdicts non-trivially -- results are less stable than deterministic
  metrics.
- The judge's own capability matters: a small or weak model may fail to
  follow the requested output format entirely, so callers must handle an
  unparseable verdict rather than assuming success.

Because of this, LLM-as-judge is deliberately opt-in and never the sole
basis for correctness -- it supplements exact_match/token_f1/semantic
similarity rather than replacing them.
"""

from __future__ import annotations

from llm_eval.providers.base import GenerationRequest, ModelProvider

_JUDGE_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Reference answer: {reference_answer}\n"
    "Candidate answer: {candidate_answer}\n\n"
    "Does the candidate answer correctly and completely answer the question, "
    "consistent with the reference answer? Respond with exactly one word: "
    "CORRECT or INCORRECT."
)


def llm_judge(
    provider: ModelProvider,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    model: str,
) -> bool | None:
    """Ask a ModelProvider to judge whether candidate_answer is correct.

    Returns True/False, or None if the judge's output could not be parsed
    as a clear verdict -- never silently assumes a default, since that
    would hide a real failure mode of LLM judges.
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )
    response = provider.generate(GenerationRequest(prompt=prompt, model=model, max_tokens=8))
    verdict = response.text.strip().upper()

    # Check INCORRECT first: it contains "CORRECT" as a substring.
    if "INCORRECT" in verdict:
        return False
    if "CORRECT" in verdict:
        return True
    return None

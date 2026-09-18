"""Prompt versioning.

A prompt is a first-class, versioned artifact, separate from the model and
the dataset. If we only tracked "model + dataset -> metrics" we could never
answer "did THIS prompt change cause the quality delta", because a metric
swing could just as easily be explained by a model swap or a dataset update
that happened to land at the same time. Giving prompts their own identity
(prompt_id, version) lets the regression/comparison tooling in later phases
hold model and dataset constant and isolate the effect of a prompt edit.
"""

from __future__ import annotations

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """A versioned prompt template.

    `template` must contain a `{question}` placeholder that `.render()` fills
    in. A plain `str.format`-style template (rather than a templating engine)
    is a deliberate "don't overengineer" choice: nothing in this project yet
    needs conditionals, loops, or partials in a prompt.
    """

    prompt_id: str
    version: str
    template: str

    def render(self, question: str) -> str:
        """Fill the template's {question} placeholder."""
        return self.template.format(question=question)


PROMPT_V1 = PromptTemplate(
    prompt_id="qa_direct",
    version="v1",
    template="{question}",
)
"""Baseline prompt: the raw question, unmodified.

Byte-for-byte equal to passing example.question straight through, so
introducing prompt versioning does not change behavior for anything that
doesn't opt into a different prompt.
"""

PROMPT_V2 = PromptTemplate(
    prompt_id="qa_instructed",
    version="v2",
    template=(
        "Answer the following question accurately and concisely. "
        "If you are unsure, say so rather than guessing.\n\nQuestion: {question}"
    ),
)
"""Candidate prompt: wraps the question in an explicit instruction.

This is exactly the kind of change the regression/comparison tooling exists
to evaluate: does the instruction measurably improve correctness/groundedness,
or does it just burn extra tokens?
"""

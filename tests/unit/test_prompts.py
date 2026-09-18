"""Unit tests for prompt templates."""

from llm_eval.evaluation.prompts import PROMPT_V1, PROMPT_V2, PromptTemplate


def test_prompt_v1_renders_question_unchanged() -> None:
    assert PROMPT_V1.render("What is 2+2?") == "What is 2+2?"


def test_prompt_v2_wraps_question_with_instruction() -> None:
    rendered = PROMPT_V2.render("What is 2+2?")

    assert "What is 2+2?" in rendered
    assert rendered != "What is 2+2?"
    assert len(rendered) > len("What is 2+2?")


def test_prompt_v1_and_v2_have_distinct_identity() -> None:
    assert PROMPT_V1.prompt_id != PROMPT_V2.prompt_id or PROMPT_V1.version != PROMPT_V2.version


def test_custom_prompt_template_renders() -> None:
    custom = PromptTemplate(prompt_id="custom", version="v1", template="Q: {question}")

    assert custom.render("Why?") == "Q: Why?"

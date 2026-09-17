"""Sanity tests for the llm_eval package itself.

These exist to prove the packaging and test-discovery setup works
before any real evaluation logic is added on top of it.
"""

import re

import llm_eval


def test_package_is_importable() -> None:
    """The package should import cleanly with no side effects."""
    assert llm_eval is not None


def test_version_is_semver() -> None:
    """__version__ should follow basic MAJOR.MINOR.PATCH semantic versioning."""
    assert hasattr(llm_eval, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+$", llm_eval.__version__), (
        f"Expected semver like '0.1.0', got {llm_eval.__version__!r}"
    )

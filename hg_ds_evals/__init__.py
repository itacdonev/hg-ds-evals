"""
hg-ds-evals: LLM-as-Judge Evaluation Framework
===============================================

A modular library for evaluating production Hey George responses 
using rubric-based judging. It is based on the LLM as a judge paradigm.

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hg-ds-evals")
except PackageNotFoundError:
    __version__ = "0.1.0"


def get_version() -> str:
    """Return the version of the hg-ds-evals package."""
    return __version__


from . import preprocessing


__all__ = [
    "__version__",
    "get_version",
    "preprocessing",
]
"""urlish - a leakage-free benchmark for client-side malicious URL detection."""

from importlib.metadata import version
from pathlib import Path

from .core import (
    deletes,
    entropy,
    fold,
    host_of,
    levenshtein,
    registrable_domain,
    split_url,
    tokenize,
)
from .features import FEATURE_NAMES, BrandIndex, analyze, featurize

# Installed editable from src/: the checkout root, where data/ and artifacts/ live.
ROOT = Path(__file__).resolve().parents[2]

# The package version is declared once, in pyproject.toml. Never copy it here.
__version__ = version("urlish")

__all__ = [
    "ROOT",
    "__version__",
    "BrandIndex",
    "FEATURE_NAMES",
    "analyze",
    "deletes",
    "entropy",
    "featurize",
    "fold",
    "host_of",
    "levenshtein",
    "registrable_domain",
    "split_url",
    "tokenize",
]

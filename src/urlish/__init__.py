"""urlish - a leakage-free benchmark for client-side malicious URL detection."""

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
from .features import FEATURE_NAMES, BrandIndex, analyze, explain, featurize

# Installed editable from src/: the checkout root, where data/ and artifacts/ live.
ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "ROOT",
    "BrandIndex",
    "FEATURE_NAMES",
    "analyze",
    "deletes",
    "entropy",
    "explain",
    "featurize",
    "fold",
    "host_of",
    "levenshtein",
    "registrable_domain",
    "split_url",
    "tokenize",
]

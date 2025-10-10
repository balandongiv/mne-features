"""Shared helpers that are safe to import from tests or tutorials.

This module deliberately exposes only the small collection of utilities that
both the regression unit tests and the tutorial scripts rely on.  Dataset
paths, frequency-band configuration, and other test-specific details now live
in :mod:`unitest.utils_test_helpers` so the public helpers here remain
lightweight and generally applicable.
"""

from __future__ import annotations

from contextlib import contextmanager
from sklearn.pipeline import FeatureUnion

__all__ = [
    "patched_feature_union",
]


@contextmanager
def patched_feature_union():
    """Guard :class:`~sklearn.pipeline.FeatureUnion` against 1D outputs."""

    original_hstack = FeatureUnion._hstack

    def _safe_hstack(self, matrices):
        reshaped = [
            matrix.reshape(1, -1)
            if getattr(matrix, "ndim", 0) == 1
            else matrix
            for matrix in matrices
        ]
        return original_hstack(self, reshaped)

    FeatureUnion._hstack = _safe_hstack
    try:
        yield
    finally:
        FeatureUnion._hstack = original_hstack


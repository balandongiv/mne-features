"""Shared helpers that are safe to import from tests or tutorials.

This module deliberately exposes only the small collection of utilities that
both the regression unit tests and the tutorial scripts rely on.  Dataset
paths, frequency-band configuration, and other test-specific details now live
in :mod:`unitest.test_utils` so the public helpers here remain lightweight and
generally applicable.
"""

from __future__ import annotations

from ast import literal_eval
from contextlib import contextmanager
import pandas as pd
from sklearn.pipeline import FeatureUnion

__all__ = [
    "ensure_multiindex",
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


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a simple two-level column :class:`~pandas.MultiIndex`.

    The regression tests and tutorials both need feature tables whose columns
    behave like the objects that :mod:`mne_features` ordinarily emits.  Some
    extraction paths flatten the MultiIndex into plain strings (for example
    when results are saved to disk and reloaded), so this helper restores the
    structured column layout in a way that is safe for any caller that expects
    the canonical two-level shape.  Keeping the utility here ensures the
    behaviour is shared consistently across tests and user-facing examples.
    """

    if isinstance(df.columns, pd.MultiIndex):
        return df

    def _normalise(column):
        if isinstance(column, tuple):
            return column
        if isinstance(column, str):
            try:
                parsed = literal_eval(column)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, tuple):
                return parsed
        return (column, "")

    result = df.copy()
    result.columns = pd.MultiIndex.from_tuples([_normalise(col) for col in df.columns])
    return result


"""Shared configuration and helpers for regression data access.

This module purposefully keeps a very small surface area so it can be reused
by both the automated tests and the accompanying tutorials.  Everything in
here is safe for general consumption; utilities that are intended strictly for
unit tests now live in :mod:`unitest.test_utils` instead.
"""

from __future__ import annotations

from ast import literal_eval
from contextlib import contextmanager
from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import FeatureUnion

# ---------------------------------------------------------------------------
# Ensure the local package is importable when running from the repo root.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Shared dataset locations and extraction parameters.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
EPOCHS_PATH = DATA_DIR / "eeg_clean_epo.fif"
GROUND_TRUTH_PATH = DATA_DIR / "features_output" / "ground_truth_features.parquet"

__all__ = [
    "DATA_DIR",
    "EPOCHS_PATH",
    "GROUND_TRUTH_PATH",
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


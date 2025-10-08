"""Helper functions that exist exclusively to support the regression tests.

This module intentionally exposes only the minimal set of helpers the tests
need.  They centralise the mechanics for loading data, running
``extract_features`` and normalising the resulting tables so that the test
files themselves can remain compact and assertion focused.  If more helpers are
required for unit tests in the future they should be added here, but utilities
for tutorials or examples belong alongside those scripts instead.
"""

from __future__ import annotations

from ast import literal_eval
from contextlib import contextmanager
from pathlib import Path
import sys

import mne
import pandas as pd
from sklearn.pipeline import FeatureUnion

# ---------------------------------------------------------------------------
# Ensure the local package is importable when the tests run from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_features.feature_extraction import extract_features  # noqa: E402

__all__ = [
    "extract_feature_dataframe",
    "load_ground_truth_df",
    "load_epochs",
    "to_epoch_indexed",
]

# ---------------------------------------------------------------------------
# Paths and parameters shared across the tests
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent
EPOCHS_PATH = _DATA_DIR / "eeg_clean_epo.fif"
GROUND_TRUTH_PATH = _DATA_DIR / "features_output" / "ground_truth_features.parquet"

FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30.0],
}

FUNCS_PARAMS = {
    "pow_freq_bands__normalize": False,
    "pow_freq_bands__ratios": "all",
    "pow_freq_bands__psd_method": "fft",
    "pow_freq_bands__freq_bands": FREQ_BANDS,
}


@contextmanager
def _patched_feature_union():
    """Guard :class:`~sklearn.pipeline.FeatureUnion` against 1D outputs.

    Older ``FeatureUnion`` implementations expect each transformer to return a
    two-dimensional array.  Some of the feature extractors return 1D vectors,
    which would normally trigger a ``ValueError``.  The tests rely on the
    public API of :func:`mne_features.feature_extraction.extract_features`, so
    we temporarily patch the private stacking helper to coerce 1D arrays into
    row vectors.  The original implementation is restored immediately after the
    extraction finishes.
    """

    original_hstack = FeatureUnion._hstack

    def _safe_hstack(self, matrices):
        return original_hstack(
            self,
            tuple(
                matrix.reshape(1, -1)
                if getattr(matrix, "ndim", 0) == 1
                else matrix
                for matrix in matrices
            ),
        )

    FeatureUnion._hstack = _safe_hstack
    try:
        yield
    finally:
        FeatureUnion._hstack = original_hstack


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a simple two-level column :class:`~pandas.MultiIndex`.

    The parquet ground-truth file and the freshly extracted DataFrame both
    store feature names using a two-level structure.  This helper normalises
    the columns should either input be loaded with a plain index, while keeping
    the original object untouched whenever possible.
    """

    if isinstance(df.columns, pd.MultiIndex):
        return df

    tuples = []
    for column in df.columns:
        if isinstance(column, tuple):
            tuples.append(column)
            continue
        if isinstance(column, str):
            try:
                parsed = literal_eval(column)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, tuple):
                tuples.append(parsed)
                continue
        tuples.append((column, ""))

    result = df.copy()
    result.columns = pd.MultiIndex.from_tuples(tuples)
    return result


def extract_feature_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """Extract power-band features and prepend the ``epoch_id`` column.

    Parameters
    ----------
    epochs
        The epochs to process.  Their :attr:`selection` attribute already
        reflects any dropped entries, which lets the helper preserve the
        original epoch identifiers in the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        A copy of the extracted features whose first column is ``epoch_id``.
        The column is stored using the same two-level layout as the other
        features so that downstream alignment logic can treat it uniformly.
    """

    epoch_ids = epochs.selection.copy()
    with _patched_feature_union():
        df = extract_features(
            epochs.get_data(),
            epochs.info["sfreq"],
            selected_funcs=["pow_freq_bands"],
            return_as_df=True,
            funcs_params=FUNCS_PARAMS,
        )

    df = _ensure_multiindex(df)
    df = df.copy()
    df.insert(0, ("epoch_id", ""), epoch_ids.astype(int))
    return df


def load_ground_truth_df() -> pd.DataFrame:
    """Load the regression baseline produced by the original pipeline."""

    return _ensure_multiindex(
        pd.read_parquet(GROUND_TRUTH_PATH, engine="pyarrow")
    )


def load_epochs() -> mne.Epochs:
    """Load the EEG recording used throughout the regression tests."""

    return mne.read_epochs(EPOCHS_PATH, preload=True, verbose="ERROR")


def to_epoch_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` indexed by the ``epoch_id`` column.

    The returned DataFrame is a copy so that callers are free to mutate it in
    assertions without affecting the original object owned by the tests.
    """

    normalised = _ensure_multiindex(df)
    result = normalised.copy()
    epoch_column = result.columns[0]
    result.set_index(result[epoch_column], inplace=True)
    result.index.name = "epoch_id"
    result.drop(columns=[epoch_column], inplace=True)
    return result

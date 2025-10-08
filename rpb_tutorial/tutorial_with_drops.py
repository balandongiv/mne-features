"""Tutorial: Extract features after dropping specific epochs.

This script mirrors the regression test scenario where a few epochs are
removed before running :func:`mne_features.feature_extraction.extract_features`.
It guides you through the following steps:

1. Load the bundled EEG epochs and drop a handful of indices.
2. Extract the frequency-band features and keep a front-facing ``epoch_id``
   column that records the *original* epoch numbers (not a reindexed range).
3. Confirm that the resulting values stay perfectly aligned with the official
   ground-truth DataFrame for a representative subset of epochs.

Run the file directly with ``python tutorial_with_drops.py`` to reproduce the
entire process.  The code favours clarity over clever tricks so new users can
follow each step comfortably.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
import sys
from contextlib import contextmanager
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module="mne_features.univariate",
)

import mne
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.pipeline import FeatureUnion

from mne_features.feature_extraction import extract_features

DATA_DIR = REPO_ROOT / "unitest"
EPOCHS_PATH = DATA_DIR / "eeg_clean_epo.fif"
GROUND_TRUTH_PATH = DATA_DIR / "features_output" / "ground_truth_features.parquet"

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
DROPPED_EPOCHS = [2, 4, 17, 40]


@contextmanager
def _patched_feature_union():
    """Convert 1D outputs from ``FeatureUnion`` transformers into row vectors."""

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


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a simple two-level column :class:`pandas.MultiIndex`."""

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


def main() -> None:
    """Execute the drop workflow and print a quick validation summary."""

    # ------------------------------------------------------------------
    # 1. Load the epochs, drop a few entries, and capture the remaining indices.
    # ------------------------------------------------------------------
    epochs = mne.read_epochs(EPOCHS_PATH, preload=True, verbose="ERROR")
    epochs.drop(DROPPED_EPOCHS)
    epoch_ids = epochs.selection.astype(int)

    # ------------------------------------------------------------------
    # 2. Extract the features and append the preserved ``epoch_id`` column.
    # ------------------------------------------------------------------
    with _patched_feature_union():
        features_df = extract_features(
            epochs.get_data(),
            epochs.info["sfreq"],
            selected_funcs=["pow_freq_bands"],
            return_as_df=True,
            funcs_params=FUNCS_PARAMS,
        ).copy()

    features_df = _ensure_multiindex(features_df)

    features_df.insert(0, ("epoch_id", ""), epoch_ids)

    # ------------------------------------------------------------------
    # 3. Align the freshly extracted features with the stored ground truth and
    #    verify that the remaining epochs still match perfectly.
    # ------------------------------------------------------------------
    ground_truth_df = _ensure_multiindex(
        pd.read_parquet(GROUND_TRUTH_PATH, engine="pyarrow")
    )

    extracted_indexed = features_df.set_index(("epoch_id", ""))
    ground_truth_indexed = ground_truth_df.set_index(("epoch_id", ""))
    extracted_indexed.index.name = "epoch_id"
    ground_truth_indexed.index.name = "epoch_id"

    expected_epochs = [0, 5, 10, 30, 41, 50]
    shared_columns = [
        col for col in extracted_indexed.columns if col in ground_truth_indexed.columns
    ]
    comparison_columns = shared_columns[:4]

    extracted_slice = extracted_indexed.loc[expected_epochs, comparison_columns]
    baseline_slice = ground_truth_indexed.loc[expected_epochs, comparison_columns]

    try:
        assert_frame_equal(
            extracted_slice,
            baseline_slice,
            check_exact=False,
            rtol=1e-12,
            atol=0.0,
        )
    except AssertionError as error:
        print("❌ Differences detected! Check the tables for details.")
        print(error)
        print("Extracted slice:\n", extracted_slice)
        print("Baseline slice:\n", baseline_slice)
    else:
        print("✅ Dropped-epoch features still match the ground truth selection.")
        print("Extracted slice:\n", extracted_slice)
        print("Ground-truth slice:\n", baseline_slice)

    missing_epochs = set(DROPPED_EPOCHS).intersection(extracted_indexed.index)
    if missing_epochs:
        print("⚠️ These dropped epochs unexpectedly remain in the output:", missing_epochs)
    else:
        print("🎯 Dropped epochs are absent from the results, as expected.")


if __name__ == "__main__":
    main()

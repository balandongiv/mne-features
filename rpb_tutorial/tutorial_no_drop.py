"""Tutorial: Extract power-band features without dropping epochs.

This short, beginner-friendly walkthrough demonstrates how to:

1. Load the example EEG epochs file that ships with the regression tests.
2. Run :func:`mne_features.feature_extraction.extract_features` with
   ``return_as_df=True`` so that we receive a :class:`pandas.DataFrame`.
3. Preserve the original epoch numbers in a new ``epoch_id`` column placed at
   the front of the table.
4. Validate a small slice of the resulting DataFrame against the ground-truth
   parquet file used by the automated tests.

Every step is implemented with straightforward, readable Python so you can use
this script as a starting point for your own experiments.
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

# ---------------------------------------------------------------------------
# Locate the shared dataset and regression baseline files that live alongside
# the automated unit tests.  They are bundled with the repository so this
# script works out of the box.
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "unitest"
EPOCHS_PATH = DATA_DIR / "eeg_clean_epo.fif"
GROUND_TRUTH_PATH = DATA_DIR / "features_output" / "ground_truth_features.parquet"

# ``extract_features`` expects a frequency-band configuration.  We reuse the
# same values as the regression tests so the outputs match the stored baseline.
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
    """Run the end-to-end extraction pipeline and compare it to the baseline."""

    # ------------------------------------------------------------------
    # 1. Load the epochs and remember their original indices.  The selection
    #    array reflects the epoch numbers after any rejects; because we do not
    #    drop anything here it is simply ``range(len(epochs))``.
    # ------------------------------------------------------------------
    epochs = mne.read_epochs(EPOCHS_PATH, preload=True, verbose="ERROR")
    epoch_ids = epochs.selection.astype(int)

    # ------------------------------------------------------------------
    # 2. Extract the frequency-band features as a DataFrame.  We copy the
    #    result so that inserting ``epoch_id`` does not mutate the object that
    #    ``extract_features`` returned.
    # ------------------------------------------------------------------
    with _patched_feature_union():
        features_df = extract_features(
            epochs.get_data(),
            epochs.info["sfreq"],
            selected_funcs=["pow_freq_bands"],
            return_as_df=True,
            funcs_params=FUNCS_PARAMS,
        ).copy()

    # Ensure the feature names use a two-level MultiIndex so we can insert the
    # identifier column while keeping a consistent layout.
    features_df = _ensure_multiindex(features_df)

    features_df.insert(0, ("epoch_id", ""), epoch_ids)

    # ------------------------------------------------------------------
    # 3. Load the ground-truth DataFrame and align both tables by ``epoch_id``
    #    so we can perform an easy equality check on a few rows.
    # ------------------------------------------------------------------
    ground_truth_df = _ensure_multiindex(
        pd.read_parquet(GROUND_TRUTH_PATH, engine="pyarrow")
    )

    extracted_indexed = features_df.set_index(("epoch_id", ""))
    ground_truth_indexed = ground_truth_df.set_index(("epoch_id", ""))
    extracted_indexed.index.name = "epoch_id"
    ground_truth_indexed.index.name = "epoch_id"

    # ------------------------------------------------------------------
    # 4. Compare a representative subset of epochs and feature columns to
    #    confirm the freshly extracted values agree with the regression
    #    baseline.  If the comparison succeeds we print a friendly message for
    #    the reader.
    # ------------------------------------------------------------------
    epoch_subset = [0, 5, 10, 30, 41, 50]
    shared_columns = [
        col for col in extracted_indexed.columns if col in ground_truth_indexed.columns
    ]
    comparison_columns = shared_columns[:4]  # keep the output compact for the tutorial

    extracted_slice = extracted_indexed.loc[epoch_subset, comparison_columns]
    baseline_slice = ground_truth_indexed.loc[epoch_subset, comparison_columns]

    try:
        assert_frame_equal(
            extracted_slice,
            baseline_slice,
            check_exact=False,
            rtol=1e-12,
            atol=0.0,
        )
    except AssertionError as error:
        print("❌ Differences detected! Inspect the following tables to investigate.")
        print(error)
        print("Extracted slice:\n", extracted_slice)
        print("Baseline slice:\n", baseline_slice)
    else:
        print("✅ Extracted features match the ground truth for the selected epochs.")


if __name__ == "__main__":
    main()

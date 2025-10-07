"""Integration test for extracting frequency band power features.

This test relies on the local source tree, therefore we make sure that the
``mne_features`` package can be imported without being installed in the
environment by temporarily adding the repository root to ``sys.path``.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

from sklearn.pipeline import FeatureUnion

import mne

# Ensure the repository root (containing ``mne_features``) is importable when
# the package is not installed in the environment running the tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Compatibility patch for scikit-learn >= 1.5 where ``FeatureUnion`` expects
# transformers to return 2D arrays. The feature extractors in ``mne-features``
# still return 1D arrays (one feature vector per epoch), so we coerce them to
# 2D before stacking.
_ORIGINAL_HSTACK = FeatureUnion._hstack


def _hstack_with_1d_support(self, Xs):
    safe_Xs = tuple(X[None, :] if getattr(X, "ndim", 0) == 1 else X for X in Xs)
    return _ORIGINAL_HSTACK(self, safe_Xs)


FeatureUnion._hstack = _hstack_with_1d_support

from mne_features.feature_extraction import extract_features


@pytest.mark.filterwarnings("ignore:Default separator `_` in feature names")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_pow_freq_band_features_can_be_saved(tmp_path):
    """End-to-end test ensuring pow_freq_bands features are extracted."""

    data_path = Path(__file__).resolve().parent / "eeg_clean_epo.fif"
    epochs = mne.read_epochs(data_path, preload=True, verbose="ERROR")

    freq_bands = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30.0],
    }

    funcs_params = {
        "pow_freq_bands__normalize": False,
        "pow_freq_bands__ratios": "all",
        "pow_freq_bands__psd_method": "fft",
        "pow_freq_bands__freq_bands": freq_bands,
    }

    features_all = extract_features(
        epochs.get_data(),
        epochs.info["sfreq"],
        selected_funcs=["pow_freq_bands"],
        return_as_df=True,
        funcs_params=funcs_params,
    )

    # Ensure the DataFrame contains one row per epoch and prepend the epoch ID.
    assert len(features_all) == len(epochs)
    features_all.insert(0, "epoch_id", np.arange(len(epochs)))
    assert features_all.columns[0][0] == "epoch_id"

    # Persist to Parquet if the optional dependency is available, otherwise skip.
    output_path = tmp_path / "ground_truth_features.parquet"
    try:
        features_all.to_parquet(output_path, index=False)
    except ImportError:
        pytest.skip("Parquet engine (pyarrow/fastparquet) not available")
    else:
        assert output_path.exists()

"""Tests for feature extraction when epochs are removed prior to processing."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import mne
from sklearn.pipeline import FeatureUnion

# Ensure the local source tree is importable without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_features.feature_extraction import extract_features


@pytest.fixture(scope="module", autouse=True)
def _patch_feature_union():
    """Patch FeatureUnion to support 1D outputs in scikit-learn >= 1.5."""

    if not hasattr(FeatureUnion, "_original_hstack"):
        original_hstack = FeatureUnion._hstack

        def _hstack(self, Xs):
            return original_hstack(
                self,
                tuple(
                    X[None, :] if getattr(X, "ndim", 0) == 1 else X for X in Xs
                ),
            )

        FeatureUnion._original_hstack = original_hstack
        FeatureUnion._hstack = _hstack

    yield

    if hasattr(FeatureUnion, "_original_hstack"):
        FeatureUnion._hstack = FeatureUnion._original_hstack
        del FeatureUnion._original_hstack


@pytest.fixture(scope="module")
def epochs():
    """Load epochs used for the feature extraction tests."""

    data_path = Path(__file__).resolve().parent / "eeg_clean_epo.fif"
    return mne.read_epochs(data_path, preload=True, verbose="ERROR")


@pytest.fixture(scope="module")
def ground_truth():
    """Load the reference Parquet file with pre-computed features."""

    path = Path(__file__).resolve().parent / "features_output" / "ground_truth_features.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def extraction_kwargs(epochs):
    """Shared keyword arguments for ``extract_features`` calls."""

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

    return {
        "sfreq": epochs.info["sfreq"],
        "selected_funcs": ["pow_freq_bands"],
        "funcs_params": funcs_params,
        "return_as_df": True,
    }


def test_features_with_drops(epochs, extraction_kwargs, ground_truth):
    """Feature extraction remains consistent after removing epochs."""

    drop_indices = np.array([2, 4, 17, 40])
    keep_indices = np.array([idx for idx in range(len(epochs)) if idx not in drop_indices])

    new_epochs = epochs.copy().drop(drop_indices.tolist())
    features_reduced = extract_features(new_epochs.get_data(), **extraction_kwargs)

    assert isinstance(features_reduced.columns, pd.MultiIndex)
    epoch_column = ("epoch_id", "")
    assert features_reduced.columns[0] == epoch_column
    np.testing.assert_array_equal(
        features_reduced[epoch_column].to_numpy(),
        np.arange(len(keep_indices), dtype=int),
    )

    features_str = features_reduced.copy()
    features_str.columns = features_str.columns.map(str)

    feature_columns = [
        col for col in ground_truth.columns if not col.startswith("('epoch_id'")
    ]
    sample_columns = feature_columns[:4]

    expected = ground_truth.loc[keep_indices, sample_columns].to_numpy()
    actual = features_str.loc[:, sample_columns].to_numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    target_epoch_ids = np.array([0, 5, 10, 30, 41, 50])
    for epoch_id in target_epoch_ids:
        assert epoch_id not in drop_indices, "Target epoch unexpectedly dropped."

        matched_pos = np.flatnonzero(keep_indices == epoch_id)
        assert matched_pos.size == 1, "Epoch index mapping should be unique."

        actual_row = features_str.loc[matched_pos[0], sample_columns].to_numpy()
        expected_row = ground_truth.loc[epoch_id, sample_columns].to_numpy()

        np.testing.assert_allclose(actual_row, expected_row, rtol=1e-9, atol=1e-9)

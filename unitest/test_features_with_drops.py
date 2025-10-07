"""Tests for feature extraction when epochs are removed prior to processing."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

import mne
from sklearn.pipeline import FeatureUnion

# Ensure the local source tree is importable without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_features.feature_extraction import extract_features


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


def _unpatch_feature_union():
    """Restore the original FeatureUnion._hstack implementation."""

    if hasattr(FeatureUnion, "_original_hstack"):
        FeatureUnion._hstack = FeatureUnion._original_hstack
        del FeatureUnion._original_hstack


def setUpModule():  # noqa: N802 - unittest hook
    _patch_feature_union()


def tearDownModule():  # noqa: N802 - unittest hook
    _unpatch_feature_union()


class FeatureExtractionWithDropsTest(unittest.TestCase):
    """Verify feature extraction parity after removing multiple epochs."""

    drop_indices = np.array([2, 4, 17, 40])
    target_epoch_ids = np.array([0, 5, 10, 30, 41, 50])

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        data_path = Path(__file__).resolve().parent / "eeg_clean_epo.fif"
        cls.epochs = mne.read_epochs(data_path, preload=True, verbose="ERROR")

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

        cls.extraction_kwargs = {
            "sfreq": cls.epochs.info["sfreq"],
            "selected_funcs": ["pow_freq_bands"],
            "funcs_params": funcs_params,
            "return_as_df": True,
        }

        gt_path = Path(__file__).resolve().parent / "features_output" / "ground_truth_features.parquet"
        cls.ground_truth = pd.read_parquet(gt_path)

    def test_features_with_drops(self):
        """Feature extraction remains consistent after removing epochs."""

        new_epochs = self.epochs.copy().drop(self.drop_indices.tolist())
        features_reduced = extract_features(
            new_epochs.get_data(),
            epoch_ids=new_epochs.selection,
            **self.extraction_kwargs,
        )

        self.assertIsInstance(features_reduced.columns, pd.MultiIndex)
        epoch_column = ("epoch_id", "")
        self.assertEqual(features_reduced.columns[0], epoch_column)

        reduced_epoch_ids = features_reduced[epoch_column].to_numpy()
        self.assertTrue(
            np.array_equal(reduced_epoch_ids, new_epochs.selection),
            "Epoch identifiers should align with the retained epoch selection.",
        )

        overlap = np.intersect1d(reduced_epoch_ids, self.drop_indices)
        self.assertEqual(
            overlap.size,
            0,
            "Dropped epoch identifiers must not appear in the reduced output.",
        )

        features_str = features_reduced.copy()
        features_str.columns = features_str.columns.map(str)

        feature_columns = [
            col for col in self.ground_truth.columns if not col.startswith("('epoch_id'")
        ]
        sample_columns = feature_columns[:4]

        expected = self.ground_truth.loc[new_epochs.selection, sample_columns].to_numpy()
        actual = features_str.loc[:, sample_columns].to_numpy()

        all_match = np.allclose(actual, expected, rtol=1e-9, atol=1e-9)
        self.assertTrue(all_match, "Features should match ground truth for retained epochs.")

        mismatched_epochs = []
        for epoch_id in self.target_epoch_ids:
            if epoch_id in self.drop_indices:
                continue

            mask = reduced_epoch_ids == epoch_id
            self.assertTrue(mask.any(), f"Epoch {epoch_id} missing after drops.")

            actual_row = features_str.loc[mask, sample_columns].to_numpy()
            expected_row = self.ground_truth.loc[[epoch_id], sample_columns].to_numpy()

            if not np.allclose(actual_row, expected_row, rtol=1e-9, atol=1e-9):
                mismatched_epochs.append(epoch_id)

        self.assertFalse(
            mismatched_epochs,
            f"Feature mismatch for epochs: {mismatched_epochs}",
        )


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()

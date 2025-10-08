import unittest

import pyarrow  # noqa: F401
from pandas.testing import assert_frame_equal

from unitest import test_utils


class TestFeatureExtractionNoDrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.epochs = test_utils.load_epochs()
        cls.features = test_utils.extract_feature_dataframe(cls.epochs)
        cls.ground_truth = test_utils.load_ground_truth_df()
        cls.features_indexed = test_utils.to_epoch_indexed(cls.features)
        cls.ground_truth_indexed = test_utils.to_epoch_indexed(cls.ground_truth)
        cls.expected_epoch_ids = [0, 5, 10, 30, 41, 50]

    def test_dataframe_integrity_against_ground_truth(self):
        epoch_column = self.features.columns[0]
        if isinstance(epoch_column, tuple):
            self.assertEqual(epoch_column[0], "epoch_id")
        else:  # pragma: no cover - defensive fallback
            self.assertEqual(epoch_column, "epoch_id")

        epoch_ids = {int(val) for val in self.features.iloc[:, 0].to_numpy()}
        for idx in self.expected_epoch_ids:
            self.assertIn(idx, epoch_ids)
            self.assertIn(idx, self.features_indexed.index)
            self.assertIn(idx, self.ground_truth_indexed.index)

        common_columns = [
            col for col in self.features_indexed.columns if col in self.ground_truth_indexed.columns
        ]
        self.assertGreaterEqual(
            len(common_columns),
            4,
            "Expected at least four shared feature columns between extracted data and ground truth.",
        )
        ordered_common_columns = sorted(common_columns)

        features_slice = self.features_indexed.loc[
            self.expected_epoch_ids, ordered_common_columns
        ].round(9)
        ground_truth_slice = self.ground_truth_indexed.loc[
            self.expected_epoch_ids, ordered_common_columns
        ].round(9)
        assert_frame_equal(
            features_slice,
            ground_truth_slice,
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()

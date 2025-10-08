"""Helper functions dedicated to the regression unit tests."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from mne_features.feature_extraction import extract_features

from .utils import ensure_multiindex, patched_feature_union

DATA_DIR = Path(__file__).resolve().parent
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

__all__ = [
    "DATA_DIR",
    "EPOCHS_PATH",
    "GROUND_TRUTH_PATH",
    "extract_feature_dataframe",
    "load_ground_truth_df",
    "load_epochs",
    "to_epoch_indexed",
]


def extract_feature_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """Extract power-band features and prepend the ``epoch_id`` column."""

    epoch_ids = epochs.selection.copy()
    with patched_feature_union():
        df = extract_features(
            epochs.get_data(),
            epochs.info["sfreq"],
            selected_funcs=["pow_freq_bands"],
            return_as_df=True,
            funcs_params=FUNCS_PARAMS,
        )

    df = ensure_multiindex(df)
    df = df.copy()
    df.insert(0, ("epoch_id", ""), epoch_ids.astype(int))
    return df


def load_ground_truth_df() -> pd.DataFrame:
    """Load the regression baseline produced by the original pipeline."""

    return ensure_multiindex(pd.read_parquet(GROUND_TRUTH_PATH, engine="pyarrow"))


def load_epochs() -> mne.Epochs:
    """Load the EEG recording used throughout the regression tests."""

    return mne.read_epochs(EPOCHS_PATH, preload=True, verbose="ERROR")


def to_epoch_indexed(df: pd.DataFrame, /, *, drop_epoch_column: bool = True) -> pd.DataFrame:
    """Return ``df`` indexed by the ``epoch_id`` column."""

    normalised = ensure_multiindex(df)
    result = normalised.copy()
    epoch_column = result.columns[0]
    result.set_index(result[epoch_column], inplace=True)
    result.index.name = "epoch_id"
    if drop_epoch_column:
        result.drop(columns=[epoch_column], inplace=True)
    return result


"""
Integration test for extracting frequency band power features.

This script relies on the local source tree, therefore we make sure that the
``mne_features`` package can be imported without being installed in the
environment by temporarily adding the repository root to ``sys.path``.
"""

from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import FeatureUnion
import mne
from numba import jit, config
# config.DISABLE_JIT = True
config.DISABLE_JIT = False
# --------------------------------------------------------------------------
# Make local `mne_features` importable if not installed
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# Patch scikit-learn >= 1.5 FeatureUnion to handle 1D feature vectors
# --------------------------------------------------------------------------
_ORIGINAL_HSTACK = FeatureUnion._hstack
FeatureUnion._hstack = lambda self, Xs: _ORIGINAL_HSTACK(
    self, tuple(X[None, :] if getattr(X, "ndim", 0) == 1 else X for X in Xs)
)

# --------------------------------------------------------------------------
# Import after patching
# --------------------------------------------------------------------------
from mne_features.feature_extraction import extract_features

# --------------------------------------------------------------------------
# Parameters and file paths
# --------------------------------------------------------------------------
TMP_PATH = Path.cwd() / "tmp_features_output"
TMP_PATH.mkdir(exist_ok=True)

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

# --------------------------------------------------------------------------
# Extract features
# --------------------------------------------------------------------------
features_all = extract_features(
    epochs.get_data(),
    epochs.info["sfreq"],
    selected_funcs=["pow_freq_bands"],
    return_as_df=True,
    funcs_params=funcs_params,
)

# Ensure output integrity
assert len(features_all) == len(epochs)
features_all.insert(0, "epoch_id", np.arange(len(epochs)))
assert features_all.columns[0][0] == "epoch_id"

# --------------------------------------------------------------------------
# Save features as Parquet
# --------------------------------------------------------------------------
output_path = TMP_PATH / "ground_truth_features.parquet"
features_all.to_parquet(output_path, index=False)

assert output_path.exists()
print(f"✅ Feature extraction successful. Output saved to: {output_path}")

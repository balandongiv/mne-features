import mne
import numpy as np

from mne_features.feature_extraction import extract_features

# --- Load your epochs ---
epochs = mne.read_epochs("eeg_clean_epo.fif", preload=True)

# --- Define frequency bands ---
FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta":  [15.5, 30.0],
}

selected_features = ['pow_freq_bands']
freq_bands=np.unique(np.concatenate(list(map(list, (FREQ_BANDS.values())))))
# --- Parameters for feature extraction ---
funcs_params = dict ( pow_freq_bands__normalize=False,pow_freq_bands__ratios='all',pow_freq_bands__psd_method='fft',pow_freq_bands__freq_bands=freq_bands)

sfreq = epochs.info["sfreq"]

# --- Extract features ---
features_all = extract_features(
    epochs.get_data(),   # (n_epochs, n_channels, n_times)
    sfreq,
    selected_funcs=selected_features,
    return_as_df=True,
    funcs_params=funcs_params,
)

# --- Add epoch_id as first column ---
features_all.insert(0, "epoch_id", np.arange(len(epochs)))

# --- Save as Parquet (ground truth) ---
output_path = "ground_truth_features.parquet"
features_all.to_parquet(output_path, index=False)

print(f"✅ Saved features with 'epoch_id' as first column to: {output_path}")
print(features_all.head())

import mne
import numpy as np
from scipy.io import savemat
'''
This script is used to export edf file from raw candidate_signal.
There are two type of file that can be exported:
1. Export all channels to edf file
2. Export only selected channel to edf file

Both of the exported edf already undergone resampling to 100 Hz.

For the single channel edf file, the channel name is 'EEG 002'. This single channel edf file is used when migrating from EEGLAB to Python.
Basiclly, all of the output assertion in the unit test is based on the single channel edf file.

Wheras,the all channels edf file is used when we migrating or understand
how the BLINKER algortihm select the best channel for blink detection.
'''
# Load the sample candidate_signal
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Pick the 'EEG 002' channel
# raw.pick_channels(['EEG 002'])
srate = 100
# Resample to 100 Hz
raw.resample(srate, npad="auto")

# In EEGLAB, CAN USE fILEio to import
# raw.save('resampled_raw.fif', overwrite=True)
mne.export.export_raw('resampled_raw_all_channels.edf', raw,overwrite=True)
# mne.export.export_raw('resampled_raw.edf', raw,overwrite=True)
# Get the candidate_signal
data = raw.get_data()

# Extract the candidate_signal into a 1D array
blinkComp = data[0, :]  # Shape: (N,)

# Define sampling rate and standard deviation threshold
srate = 20
stdThreshold = 1.5

# Calculate the standard deviation of the candidate_signal
std_dev = np.std(blinkComp)

# Identify indices where the signal exceeds the threshold
threshold_value = stdThreshold * std_dev
artifact_indices = np.where(np.abs(blinkComp) > threshold_value)[0]

# Create a cleaned version of blink_component where artifacts are set to zero
blinkComp_cleaned = np.copy(blinkComp)
blinkComp_cleaned[artifact_indices] = 0

# Convert blinkComp_cleaned to single precision (float32)
blinkComp_cleaned = blinkComp_cleaned.astype(np.float32)

# Save the cleaned candidate_signal to a .mat file with single precision
savemat('blinkComp_cleaned.mat', {
    'blink_component': blinkComp_cleaned,
    'srate': srate,
    'stdThreshold': stdThreshold
})

print("Cleaned candidate_signal saved to 'blinkComp_cleaned.mat' successfully as type single.")

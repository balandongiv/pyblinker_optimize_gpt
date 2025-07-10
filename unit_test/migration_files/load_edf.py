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
raw.pick_channels(['EEG 002'])
srate = 100
# Resample to 100 Hz
raw.resample(srate, npad="auto")


mne.export.export_raw('resampled_raw_all_channels.edf', raw,overwrite=True)

# After save as edf, in typical gui approach
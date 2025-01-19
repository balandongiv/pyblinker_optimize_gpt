import logging
import os
from hello import hello

import matplotlib
import mne
from pyblinkers.pyblinker import BlinkDetector

logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')

def plot_blinks(raw_file):
    """Plot eye close events based on EEG signals.

    Args:
    raw_file (str): Path to the raw EEG data file in .fif format.

    Returns:
    None
    """
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design='firwin')
    raw.resample(100)
    # srate = 100
    # Resample to 100 Hz
    # raw.resample(srate, npad="auto")

    drange=[f'EEG 00{X}' for X in range (10)]
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    raw = raw.drop_channels(to_drop_ch)
    # raw.save('temp_raw.fif')
    # Pick the 'EEG 002' channel
    # raw.pick_channels(['EEG 002'])
    annot, ch, number_good_blinks, df, fig_data, ch_selected = BlinkDetector(raw, visualize=False, annot_label=None, filter_low=0.5, filter_high=20.5, resample_rate=100, n_jobs=2).get_blink_stat()
    raw.set_annotations(annot)
    raw.plot(block=True, title=f'Eye close based on channel {ch}')

if __name__ == '__main__':
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    plot_blinks(sample_data_raw_file)

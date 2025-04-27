import logging
import os


import matplotlib
import mne
from pyblinkers.pyblinker import BlinkDetector
from pyblinkers.pyblinker import run_blink_detection_pipeline
logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')

def plot_blinks(raw_file):
    """Plot eye close events based on EEG signals.

    Args:
    raw_file (str): Path to the raw EEG candidate_signal file in .fif format.

    Returns:
    None
    """
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design='firwin')
    raw.resample(100)
    # drange=[f'EEG 00{X}' for X in range (10)]
    # raw.pick_channels(drange)
    # raw.pick_channels([f'EEG {i:03}' for i in range(1, 4)])
    # raw.pick_channels(['EEG 001','EEG 002'])
    raw.pick_channels([
        # 'EEG 001',
        'EEG 002',
        'EEG 003',
        # 'EEG 004', 'EEG 005',
        # 'EEG 006',
        # 'EEG 007',
        # 'EEG 008',
        # 'EEG 009',
        # 'EEG 010'
    ])
    # drange=[f'EEG 00{X}' for X in range (10)]
    # to_drop_ch = list(set(raw.ch_names) - set(drange))
    # raw = raw.drop_channels(to_drop_ch)
    epoch_length = 3.0  # in seconds

    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epoch_length, baseline=None, preload=True)
    # drop_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    # drop_indices =
#     drop_indices = list(range(90))
#     exclude = {33, 38, 42, 45, 47, 51, 64, 72, 73}  # <<< make it a set (no duplicates)
#     #
#     exclude = {33, 38, 42, 45, 47, 51}  # <<< make it a set (no duplicates)
# #
#     for num in exclude:
#         drop_indices.remove(num)
#     epochs.drop(indices=drop_indices)
    # epochs.plot(block=True,n_epochs=4)
    # raw.plot(block=True)

    config = {
        'visualize': False,
        'annot_label': 'my_blink_label',
        'filter_low': 0.5,
        'filter_high': 20.5,
        'resample_rate': 100,
        'n_jobs': 1,
        'use_multiprocessing': False,
        'pick_types_options': {'eeg': True,
                               # 'eog': True
                               }
    }
    # results = run_blink_detection_pipeline(raw, config=config)
    selected_rows = BlinkDetector(epochs, visualize=False, annot_label=None,
                                                                             filter_low=0.5, filter_high=20.5, resample_rate=100,
                                                                             n_jobs=2,use_multiprocessing=True).get_blink()


if __name__ == '__main__':
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    plot_blinks(sample_data_raw_file)

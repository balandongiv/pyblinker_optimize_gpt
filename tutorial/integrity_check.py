import pandas as pd

import logging
import os


import mne
from pyblinkers.pyblinker import BlinkDetector
from pandas.testing import assert_frame_equal
logging.basicConfig(level=logging.INFO)



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
    drange=[f'EEG 00{X}' for X in range (10)]
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    raw = raw.drop_channels(to_drop_ch)

    _, _, _,df_opt = BlinkDetector(raw,visualize=False, annot_label=None).get_blink_stat()
    df_gt = pd.read_pickle("unit_test_1.pkl")
    # print(df.equals(df_opt))
    # df_opt = df_opt.astype({"maxFrames": int})
    df_gt = df_gt.astype({"maxFrames": int,
                          'startBlinks': int,
                          'endBlinks': int,
                          'outerStarts': int,
                          'outerEnds': int,
                          'leftZero': int,
                            'rightZero': int,
                          'rightBase': int,
                          'blinkBottomPoint_l_X': int,
                          'blinkTopPoint_l_X': int,
                          'blinkBottomPoint_r_X': int,
                          'blinkTopPoint_r_X': int,
                          'leftXIntercept_int': int,
                          'rightXIntercept_int': int,
                          })
    # print(df.dtypes)
    # print('xxx')
    # print(df_opt.dtypes)
    assert_frame_equal(df_gt, df_opt)

if __name__ == '__main__':
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    plot_blinks(sample_data_raw_file)


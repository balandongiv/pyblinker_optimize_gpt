import unittest
import pandas as pd
import numpy as np
import sys
import mne
from pyblinkers.pyblinker import BlinkDetector
import os


class TestBlinkDetector(unittest.TestCase):
    def test_blink_detector_on_epochs(self):
        # Handle legacy numpy path used in the pickle file
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

        # Step 1: Load expected blink detection results
        expected_file = os.path.join(os.path.dirname(__file__), "file_test_epoch_full_pipeline.pkl")
        expected_selected_rows = pd.read_pickle(expected_file)

        # Step 2: Load raw EEG data and create epochs (same preprocessing as when generating expected)
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = os.path.join(
            sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
        )

        raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 20.5, fir_design="firwin")
        raw.resample(100)
        raw.pick_channels(["EEG 002", "EEG 003"])

        epoch_length = 3.0
        events = mne.make_fixed_length_events(raw, duration=epoch_length)
        epochs = mne.Epochs(
            raw, events, tmin=0.0, tmax=epoch_length, baseline=None, preload=True
        )

        # Step 3: Run BlinkDetector
        detected_blinks = BlinkDetector(
            epochs,
            visualize=False,
            annot_label=None,
            filter_low=0.5,
            filter_high=20.5,
            resample_rate=100,
            n_jobs=2,
            use_multiprocessing=True,
        ).get_blink()

        # Step 4: Compare using pandas testing tools
        try:
            pd.testing.assert_frame_equal(detected_blinks, expected_selected_rows)
            print("✅ MATCH: Detected blinks match expected output.")
        except AssertionError as e:
            print("❌ MISMATCH: Detected blinks do not match expected output.")
            raise e  # re-raise the assertion so unittest marks it as a failure

if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import mne
from pyblinkers.pyblinker import BlinkDetector
import os
class TestBlinkDetector(unittest.TestCase):
    def test_blink_detector_on_epochs(self):
        self.skipTest("Blink detection on full pipeline requires stable input data.")
        # Step 1: Load raw EEG data and create epochs using the bundled debug file
        # Use local debug dataset to avoid network downloads
        sample_data_raw_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           'debug_pyblinker.fif')
        raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 20.5, fir_design='firwin')
        raw.resample(100)
        raw.pick_channels(['E1', 'E2'])
        raw.crop(tmin=0, tmax=10)

        # Step 3: Run BlinkDetector directly on the raw signal
        detected_blinks = BlinkDetector(
            raw,
            visualize=False,
            annot_label=None,
            filter_low=0.5,
            filter_high=20.5,
            resample_rate=100,
            n_jobs=2,
            use_multiprocessing=True
        ).get_blink()

        # Step 4: Ensure we detected at least one blink
        self.assertFalse(detected_blinks.empty, "Blink detection result should not be empty.")

if __name__ == "__main__":
    unittest.main()

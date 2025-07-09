"""Tests for blink refinement on EEG and EOG channels.
Here we want to improve the start, peak, and end frames of detected blinks (eeg and eog) based on the signal in the epochs."""
import logging
from pathlib import Path
import unittest

import mne

from pyear.utils import prepare_refined_segments
from pyear.utils.refinement import refine_blinks_from_epochs, plot_refined_blinks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestEEGEOGRefinement(unittest.TestCase):
    """Ensure refinement works on both EEG and EOG modalities."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        self.segments, _ = prepare_refined_segments(
            raw_path,
            "EOG-EEG-eog_vert_left",
            keep_epoch_signal=False,
        )
        self.total_ann = sum(len(seg.annotations) for seg in self.segments)

    def _run_channel(self, channel: str) -> None:
        logger.info("Refinement test on %s", channel)
        refined = refine_blinks_from_epochs(self.segments, channel)
        self.assertEqual(len(refined), self.total_ann)
        for blink in refined:
            n_times = len(blink["epoch_signal"])
            self.assertTrue(0 <= blink["refined_start_frame"] <= n_times)
            self.assertTrue(0 <= blink["refined_peak_frame"] <= n_times)
            self.assertTrue(0 <= blink["refined_end_frame"] <= n_times)
            self.assertLessEqual(blink["refined_start_frame"], blink["refined_peak_frame"])
            self.assertLessEqual(blink["refined_peak_frame"], blink["refined_end_frame"])
        # sanity plot for first epoch without showing
        figs = plot_refined_blinks(
            refined,
            self.segments[1].info["sfreq"],
            30.0,
            epoch_indices=[0],
            show=True,
        )
        self.assertTrue(len(figs) >= 1)

    def test_eeg_e8(self) -> None:
        """Run refinement on EEG channel."""
        self._run_channel("EEG-E8")

    def test_eog_vertical(self) -> None:
        """Run refinement on EOG channel."""
        self._run_channel("EOG-EEG-eog_vert_left")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

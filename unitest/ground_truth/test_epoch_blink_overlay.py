import unittest
import logging
from pathlib import Path
import pandas as pd
import mne
from ground_truth.epoch_blink_overlay import summarize_blink_counts

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestEpochBlinkOverlay(unittest.TestCase):
    """Tests for blink count summarization."""

    def test_blink_count_summary(self) -> None:
        """Blink counts for first epochs match reference CSV."""
        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        df, _ = summarize_blink_counts(raw, epoch_len=30.0, blink_label=None)
        expected = pd.read_csv(PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv")
        pd.testing.assert_frame_equal(df.iloc[: len(expected)], expected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

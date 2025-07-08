"""Blink detection pipeline tutorial.

This example demonstrates how to detect eye-blink events from an EEG
recording using :class:`BlinkDetector`. The script loads the
``sample_audvis_filt-0-40_raw.fif`` dataset shipped with MNE, runs the
blink detection pipeline, and visualizes the resulting annotations.
"""

from __future__ import annotations

import logging
import os

import matplotlib
import mne

from pyblinkers.pyblinker import BlinkDetector, run_blink_detection_pipeline

logging.basicConfig(level=logging.INFO)
matplotlib.use("TkAgg")


def plot_blinks(raw_file: str) -> None:
    """Load ``raw_file`` and display detected blinks.

    Parameters
    ----------
    raw_file : str
        Path to the raw EEG file in FIF format.
    """
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design="firwin")
    raw.resample(100)

    channel_range = [f"EEG 00{x}" for x in range(10)]
    drop_channels = list(set(raw.ch_names) - set(channel_range))
    raw.drop_channels(drop_channels)

    config = {
        "visualize": False,
        "annot_label": "my_blink_label",
        "filter_low": 0.5,
        "filter_high": 20.5,
        "resample_rate": 100,
        "n_jobs": 1,
        "use_multiprocessing": False,
        "pick_types_options": {"eeg": True},
    }

    results = run_blink_detection_pipeline(raw, config=config)
    raw.set_annotations(results["annotations"])
    raw.plot(block=True, title=f"Eye close based on channel {results['channel']}")


if __name__ == "__main__":
    sample_data_folder = mne.datasets.sample.data_path()
    raw_file = os.path.join(
        sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )
    plot_blinks(raw_file)
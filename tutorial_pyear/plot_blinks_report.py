"""Utility to visualize blinks across all segments.

Run this script with a raw FIF recording to produce an HTML report that shows
every detected blink. When invoked without arguments, the bundled test file
``unitest/ear_eog.fif`` is used and the report is written to ``blink_report.html``.
All annotation labels are interpreted as blinks (``blink_label=None``).

The generated report title explicitly states which channel was used when
detecting blinks so that readers can easily interpret the figures.
This script lives inside the ``tutorial`` folder so that users can easily run
it as an example when exploring the project.
"""
from __future__ import annotations

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinkers.features.blink_events import generate_blink_dataframe
from pyblinkers.utils.epochs import slice_raw_into_epochs

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _add_segment_figures(
    report: mne.Report,
    seg: mne.io.BaseRaw,
    seg_df: pd.DataFrame,
    seg_id: int,
    channel: str,
) -> None:
    """Add blink figures for a single segment to ``report``."""
    signal = seg.get_data(picks=channel)[0]
    sfreq = seg.info["sfreq"]
    for _, row in seg_df.iterrows():
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        left_zero = int(row["left_zero_idx"])
        right_zero = (
            int(row["right_zero_idx"])
            if pd.notna(row["right_zero_idx"])
            else None
        )

        base_start = left_zero if left_zero < start_idx else start_idx
        base_end = (
            right_zero if right_zero is not None and right_zero > end_idx else end_idx
        )

        win_start = max(0, base_start - 10)
        win_end = min(signal.size - 1, base_end + 10)
        times = np.arange(win_start, win_end + 1) / sfreq
        fig, ax = plt.subplots(figsize=(6, 2))
        window_data = signal[win_start : win_end + 1]
        y_top = window_data.max()
        ax.plot(times, window_data, color="black", alpha=0.3, label="signal")

        label = "start"
        if start_idx == int(row["left_zero_idx"]):
            label = "start/left_zero"
            draw_left_zero = False
        else:
            draw_left_zero = True

        ax.axvline(start_idx / sfreq, color="green")
        ax.text(
            start_idx / sfreq,
            y_top,
            label,
            rotation=45,
            va="bottom",
            fontsize="small",
        )

        ax.axvline(row["peak_idx"] / sfreq, color="red")
        ax.text(
            row["peak_idx"] / sfreq,
            y_top,
            "peak",
            rotation=45,
            va="bottom",
            fontsize="small",
        )

        ax.axvline(end_idx / sfreq, color="blue")
        ax.text(
            end_idx / sfreq,
            y_top,
            "end",
            rotation=45,
            va="bottom",
            fontsize="small",
        )

        if draw_left_zero:
            ax.axvline(row["left_zero_idx"] / sfreq, color="magenta")
            ax.text(
                row["left_zero_idx"] / sfreq,
                y_top,
                "left_zero",
                rotation=45,
                va="bottom",
                fontsize="small",
            )

        if pd.notna(row["right_zero_idx"]):
            if end_idx == int(row["right_zero_idx"]):
                label_r = "end/right_zero"
                draw_right_zero = False
            else:
                label_r = "right_zero"
                draw_right_zero = True
            if draw_right_zero:
                ax.axvline(row["right_zero_idx"] / sfreq, color="cyan")
                ax.text(
                    row["right_zero_idx"] / sfreq,
                    y_top,
                    label_r,
                    rotation=45,
                    va="bottom",
                    fontsize="small",
                )
            else:
                ax.text(
                    end_idx / sfreq,
                    y_top,
                    label_r,
                    rotation=45,
                    va="bottom",
                    fontsize="small",
                )

        ax.set_title(f"Seg {seg_id} Blink {row['blink_id']}")
        if window_data.min() < 0 < window_data.max():
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize="x-small", loc="upper right", framealpha=0.5)
        report.add_figure(fig, title=f"Seg {seg_id} Blink {row['blink_id']}", section="blinks")
        plt.close(fig)


def create_report(segments: list[mne.io.BaseRaw], df: pd.DataFrame, channel: str) -> mne.Report:
    """Return an :class:`mne.Report` visualizing blink events."""
    report = mne.Report(title=f"Blink Report - {channel}")
    for seg_id, seg in enumerate(tqdm(segments, desc="Plot segments")):
        seg_df = df[df["seg_id"] == seg_id]
        _add_segment_figures(report, seg, seg_df, seg_id, channel)
    return report


def main(raw_path: Path, out_html: Path, channel: str = "EEG-E8") -> None:
    """Generate and save a blink visualization report."""
    logging.basicConfig(level=logging.INFO)

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    segments, _, _, _ = slice_raw_into_epochs(
        raw, epoch_len=30.0, blink_label=None
    )
    df = generate_blink_dataframe(
        segments, channel=channel, blink_label=None
    )

    report = create_report(segments, df, channel)
    report.save(out_html, overwrite=True, open_browser=False)
    logger.info("Saved report to %s", out_html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate blink visualization report")
    parser.add_argument(
        "raw",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif",
        help="Path to raw FIF file (default: unit_test/features/ear_eog_raw.fif)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "blink_report.html",
        help="Destination HTML file",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="EEG-E8",
        help="Channel to use for blink detection",
    )
    args = parser.parse_args()

    main(args.raw, args.output, channel=args.channel)

import os
import pickle
import zipfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from pyblinkers.extractBlinkProperties import BlinkProperties, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks


def generate_blink_reports(
        raw,
        blink_df,
        picks='avg_ear',
        sfreq=100.0,
        output_dir='reports',
        base_filename='blink_report',
        max_events_per_report=20
):
    """
    Generate one or more MNE reports with annotated blink plots, capped by max_events_per_report.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing the data.
    blink_df : pd.DataFrame
        DataFrame containing 'startBlinks', 'endBlinks', 'blink_min'.
    picks : str
        Channel to plot.
    sfreq : float
        Sampling frequency of the times series  in Hz.
    output_dir : str
        Directory where report files will be saved.
    base_filename : str
        Base name for the report files.
    max_events_per_report : int
        Maximum number of figures per report.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = []

    # Generate all figures first
    for idx, row in blink_df.iterrows():
        fig = plot_with_annotation_lines(
            raw=raw,
            start_frame=row['startBlinks'],
            end_frame=row['endBlinks'],
            mid_frame=row['blink_min'],
            picks=picks,
            sfreq=sfreq,
            show=False
        )
        if fig:
            caption = f"Blink {idx}: frames {row['startBlinks']}–{row['endBlinks']}, min at {row['blink_min']}"
            figures.append((fig, f"Blink {idx}", caption))

    # Group figures into batches
    for i in range(0, len(figures), max_events_per_report):
        batch = figures[i:i + max_events_per_report]
        report_index = i // max_events_per_report + 1
        report = mne.Report(title=f"Blink Report {report_index}")

        for fig, title, caption in batch:
            report.add_figure(fig=fig, title=title, caption=caption, section='Blinks')
            plt.close(fig)  # release memory

        report_path = os.path.join(output_dir, f"{base_filename}_{report_index}.html")
        report.save(report_path, overwrite=True)
        print(f"✅ Saved MNE report: {report_path}")









def plot_with_annotation_lines(
        raw,
        start_frame=0,
        end_frame=10,
        mid_frame=5,
        picks='avg_ear',
        sfreq=100.0,
        show=True
):
    if picks not in raw.ch_names:
        print(f"Channel '{picks}' not found in raw data.")
        return None
    # sfreq=100
    start_sec = start_frame / sfreq
    end_sec = end_frame / sfreq
    buffer_time = 0.5

    crop_start_sec = start_sec - buffer_time
    crop_end_sec = end_sec + buffer_time

    raw_segment = raw.copy().crop(tmin=crop_start_sec, tmax=crop_end_sec)
    data, times = raw_segment.get_data(picks=picks, return_times=True)
    times += crop_start_sec

    signal_orig = -data[0]
    signal = signal_orig - np.mean(signal_orig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(times, signal, s=10, color='black', label=picks)

    for adjusted_frame in [start_frame, mid_frame, end_frame]:
        frame_sec = adjusted_frame / sfreq
        ax.axvline(x=frame_sec, color='orange', linestyle='--')
        ax.annotate(
            f"Frame {adjusted_frame}",
            xy=(frame_sec, signal.min()),
            xytext=(frame_sec, signal.max()),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=8,
            rotation=90
        )

    ax.set_title(f"{picks} scatter plot with blink annotations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig
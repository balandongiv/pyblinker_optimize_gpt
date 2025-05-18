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
        video_sfreq=30.0,
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
    video_sfreq : float
        Sampling frequency of the video in Hz.
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
            video_sfreq=video_sfreq,
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



def load_fif_and_annotations(
        fif_path: str,
        zip_path: str,
        debug_dir: str = "./_debug_cache",
        overwrite_cache: bool = False
) -> tuple:
    """
    Load FIF signal and annotation CSV from ZIP with optional debug caching.

    Parameters
    ----------
    fif_path : str
        Full path to the .fif EEG file.
    zip_path : str
        Full path to the ZIP file containing the annotation CSV.
    debug_dir : str
        Path to store/reuse cached debug files.
    overwrite_cache : bool
        If True, overwrite any existing cached files.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing EEG/EOG/EAR signals.
    annotation_df : pd.DataFrame
        DataFrame of blink annotations.
    """
    os.makedirs(debug_dir, exist_ok=True)
    fif_cache = os.path.join(debug_dir, "cached_raw.pkl")
    ann_cache = os.path.join(debug_dir, "cached_annotations.pkl")

    if os.path.exists(fif_cache) and not overwrite_cache:
        print(f"Loading cached FIF from: {fif_cache}")
        with open(fif_cache, "rb") as f:
            raw = pickle.load(f)
    else:
        print(f"Loading FIF from: {fif_path}")
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        with open(fif_cache, "wb") as f:
            pickle.dump(raw, f)
        print(f"Cached FIF to: {fif_cache}")

    if os.path.exists(ann_cache) and not overwrite_cache:
        print(f"Loading cached annotations from: {ann_cache}")
        with open(ann_cache, "rb") as f:
            annotation_df = pickle.load(f)
    else:
        print(f"Reading annotations from: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            ann_files = [f for f in zip_ref.namelist() if f.endswith("default-annotations-human-imagelabels.csv")]
            if not ann_files:
                raise FileNotFoundError("No annotation CSV found in ZIP.")
            with zip_ref.open(ann_files[0]) as csv_file:
                annotation_df = pd.read_csv(csv_file)
        with open(ann_cache, "wb") as f:
            pickle.dump(annotation_df, f)
        print(f"Cached annotations to: {ann_cache}")

    return raw, annotation_df


def extract_blink_durations(annotation_df,frame_offset):
    """
    Extract blink durations based on triplets in the annotation DataFrame.

    Parameters
    ----------
    annotation_df : pd.DataFrame
        DataFrame containing CVAT blink annotations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: blink_start, blink_end, blink_type.
    """
    blink_data = []
    for i in range(0, len(annotation_df) - 2, 3):
        start_label = annotation_df.iloc[i]['LabelName']
        mid_label = annotation_df.iloc[i+1]['LabelName']
        end_label = annotation_df.iloc[i+2]['LabelName']

        if start_label.endswith('_start') and end_label.endswith('_end'):
            blink_type = start_label.rsplit('_', 1)[0]
            blink_start = int(annotation_df.iloc[i]['ImageID'].replace('frame_', ''))
            blink_min = int(annotation_df.iloc[i+1]['ImageID'].replace('frame_', ''))
            blink_end = int(annotation_df.iloc[i+2]['ImageID'].replace('frame_', ''))
            blink_data.append({
                'startFrame': blink_start,
                'endFrame': blink_end,
                'minFrame': blink_min,
                'blink_type': blink_type
            })
    df=pd.DataFrame(blink_data)

    df[['startBlinks', 'endBlinks', 'blink_min']] = df[['startFrame', 'endFrame', 'minFrame']] - frame_offset

    return df


def process_blinks(candidate_signal, df, params):
    """
    Process blink detection and extract blink properties.

    Parameters
    ----------
    candidate_signal : mne.io.Raw
        The raw signal data.
    df : pd.DataFrame
        DataFrame with blink intervals.
    params : dict
        Configuration including 'sfreq' and 'z_thresholds'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame enriched with blink properties.
    blink_stats : dict
        Dictionary of blink statistics.
    """
    fitblinks = FitBlinks(candidate_signal, df, params)
    fitblinks.process_blink_candidate()
    df = fitblinks.frame_blinks

    blink_stats = get_blink_statistic(df, params['z_thresholds'], candidate_signal)

    # Optional filtering step (commented out by default)
    # good_blink_mask, df = get_good_blink_mask(df, blink_stats['bestMedian'], blink_stats['bestRobustStd'], params['z_thresholds'])

    df = BlinkProperties(candidate_signal, df, params['sfreq'], params).df


    return df, blink_stats



def plot_with_annotation_lines(
        raw,
        start_frame=0,
        end_frame=10,
        mid_frame=5,
        picks='avg_ear',
        video_sfreq=30.0,
        show=True
):
    if picks not in raw.ch_names:
        print(f"Channel '{picks}' not found in raw data.")
        return None

    start_sec = start_frame / video_sfreq
    end_sec = end_frame / video_sfreq
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
        frame_sec = adjusted_frame / video_sfreq
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


if __name__ == "__main__":
    fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
    zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"

    # Optional parameters


    # Load data
    raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)



    # raw.plot(
    #     picks=['avg_ear','E8'],
    #     block=True,
    #     show_scrollbars=False,
    #     title='avg_ear Blink Signal'
    # )
    # Extract blink intervals
    frame_offset=5
    blink_df = extract_blink_durations(annotation_df,frame_offset)
    generate_blink_reports(
        raw=raw,
        blink_df=blink_df,
        picks='avg_ear',
        video_sfreq=30.0,
        output_dir='blink_reports',
        base_filename='blink_report',
        max_events_per_report=40
    )

    for _, row in blink_df.iterrows():
        plot_with_annotation_lines(
            raw=raw,
            start_frame=row['startBlinks'],
            end_frame=row['endBlinks'],
            mid_frame=row['blink_min'],
            picks='avg_ear',
            video_sfreq=30.0,
        )


    # with open("fitblinks_debug.pkl", "rb") as f:
    #         debug_data = pickle.load(f)
    # params = debug_data["params"]

    # #
    #
    # # Process and extract properties
    # processed_df, stats = process_blinks( raw.get_data(picks=1)[0] , blink_df, params)
    #
    # print("\nProcessed Blink DataFrame:")
    # # print(processed_df.head())
    #
    # print("\nBlink Statistics:")
    # # print(stats)

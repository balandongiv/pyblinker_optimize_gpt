import os
import os
import pickle
import zipfile

import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne import Report

from pyblinkers.extractBlinkProperties import BlinkProperties, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks


def generate_blink_mne_reports_matplotlib(
        raw,
        blink_df,
        sfreq,
        channels,
        video_sfreq: float = 30.0,
        t_pre_frames: int = 10,
        t_post_frames: int = 10,
        offset_frames: int = 0,
        max_events_per_report: int = 200,
        report_dir: str = "./reports"
):
    """
    Generate MNE HTML reports using custom matplotlib plots of blink events.

    Parameters
    ----------
    raw : mne.io.Raw
        Loaded EEG/EOG/EAR signal.
    blink_df : pd.DataFrame
        DataFrame with ['startBlinks','endBlinks','blink_type'].
    sfreq : float
        Sampling frequency of raw signal.
    channels : list of str
        List of channels to plot (e.g., ['avg_ear']).
    video_sfreq : float
        Frame rate of annotated video (default 30.0).
    t_pre_frames : int
        Frames before startBlinks.
    t_post_frames : int
        Frames after endBlinks.
    offset_frames : int
        Optional correction to shift blink frame indices.
    max_events_per_report : int
        Max number of blinks per HTML file.
    report_dir : str
        Where to save the report HTMLs and temporary images.
    """
    os.makedirs(report_dir, exist_ok=True)
    image_dir = os.path.join(report_dir, "_img")
    os.makedirs(image_dir, exist_ok=True)

    picks = mne.pick_channels(raw.info["ch_names"], include=channels)
    ch_name = channels[0]  # only one channel assumed per plot

    fig_paths = []

    for idx, row in blink_df.iterrows():
        # Convert video frame to signal time
        start_f = (row["startBlinks"] + offset_frames)
        end_f = (row["endBlinks"] + offset_frames)

        # Convert to seconds
        start_s = start_f / video_sfreq
        end_s = end_f / video_sfreq
        t_pre = t_pre_frames / video_sfreq
        t_post = t_post_frames / video_sfreq

        tmin = max(0.0, start_s - t_pre)
        tmax = min(raw.times[-1], end_s + t_post)

        data, times = raw.copy().pick(picks).crop(tmin=tmin, tmax=tmax).get_data(return_times=True)
        signal = data[0]

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(times, signal, label=ch_name, color='black')

        # Blink markers
        blink_start_time = start_s
        blink_end_time = end_s
        y_min, y_max = signal.min(), signal.max()

        ax.axvline(blink_start_time, color='r', linestyle='--', label='blink_start')
        ax.axvline(blink_end_time, color='g', linestyle='--', label='blink_end')

        ax.annotate('start', xy=(blink_start_time, y_min), xytext=(blink_start_time, y_max),
                    arrowprops=dict(arrowstyle='->', color='r'), fontsize=8)
        ax.annotate('end', xy=(blink_end_time, y_min), xytext=(blink_end_time, y_max),
                    arrowprops=dict(arrowstyle='->', color='g'), fontsize=8)

        ax.set_title(f"Blink: {row['blink_type']} [{row['startBlinks']}→{row['endBlinks']}]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        fname = os.path.join(image_dir, f"blink_{idx}_{ch_name}.png")
        fig.savefig(fname, dpi=150)
        fig_paths.append(fname)
        plt.close(fig)

    # Group images into MNE reports
    for i in range(0, len(fig_paths), max_events_per_report):
        batch = fig_paths[i:i + max_events_per_report]
        report = Report(title=f"Blink Report {i//max_events_per_report + 1}")
        for fig_path in batch:
            report.add_image(fig_path, title=os.path.basename(fig_path))
        html_path = os.path.join(report_dir, f"report_{i//max_events_per_report + 1}.html")
        report.save(html_path, overwrite=True)
        print(f"✅ Saved MNE report: {html_path}")

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


def extract_blink_durations(annotation_df):
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
                'startBlinks': blink_start,
                'endBlinks': blink_end,
                'blink_min': blink_min,
                'blink_type': blink_type
            })

    return pd.DataFrame(blink_data)


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
        start_sec=10.0,
        end_sec=13.0,
        picks='avg_ear',
        annotation_frames=None,
        video_sfreq=30.0,
        frame_offset: int = 0
):
    """
    Plot a segment of signal from raw using scatter plot and overlay annotation frame markers.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    start_sec : float
        Start of time window (seconds).
    end_sec : float
        End of time window (seconds).
    picks : str
        Channel to plot (e.g., 'avg_ear').
    annotation_frames : list of str
        List of frame labels like ['frame_000358'].
    video_sfreq : float
        Frame rate of the annotated video (Hz).
    frame_offset : int
        Subtracted from each frame index before conversion.
    """
    if picks not in raw.ch_names:
        print(f"Channel '{picks}' not found in raw data.")
        return

    # Crop and extract data
    raw_segment = raw.copy().crop(tmin=start_sec, tmax=end_sec)
    data, times = raw_segment.get_data(picks=picks, return_times=True)
    times += start_sec  # adjust to global time

    signal = data[0]

    # Create scatter plot
    plt.figure(figsize=(10, 4))
    plt.scatter(times, signal, s=10, color='black', label=picks)

    # Overlay annotation lines
    if annotation_frames:
        for frame_str in annotation_frames:
            frame_num = int(frame_str.replace("frame_", ""))
            adjusted_frame = frame_num - frame_offset
            frame_sec = adjusted_frame / video_sfreq

            if start_sec <= frame_sec <= end_sec:
                plt.axvline(x=frame_sec, color='orange', linestyle='--')
                plt.annotate(
                    f"{frame_str} -{frame_offset}",
                    xy=(frame_sec, signal.min()),
                    xytext=(frame_sec, signal.max()),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=8,
                    rotation=90
                )

    plt.title(f"{picks} scatter plot with annotation markers")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
    zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"

    # Optional parameters


    # Load data
    raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)



    # raw.plot(
    #     picks='avg_ear',
    #     block=True,
    #     show_scrollbars=False,
    #     title='avg_ear Blink Signal'
    # )
    # Extract blink intervals
    blink_df = extract_blink_durations(annotation_df)
    annotation_frames = ['frame_000358', 'frame_000361', 'frame_000366']
    ['E8', 'E10', 'eog_vert_right', 'avg_ear']
    plot_with_annotation_lines(
        raw=raw,
        start_sec=10,
        end_sec=13,
        picks='E8',
        annotation_frames=annotation_frames,
        video_sfreq=30.0,
        frame_offset=6 # shifts 358 to 353, etc.
    )

    with open("fitblinks_debug.pkl", "rb") as f:
            debug_data = pickle.load(f)
    params = debug_data["params"]
    # now generate reports
    generate_blink_mne_reports_matplotlib(
        raw=raw,
        blink_df=blink_df,
        sfreq=params['sfreq'],
        channels=['avg_ear'],
        video_sfreq=30.0,
        t_pre_frames=10,
        t_post_frames=10,
        offset_frames=0,
        max_events_per_report=200,
        report_dir=r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\direct_blink_properties\blink_reports"
    )
    #

    # Process and extract properties
    processed_df, stats = process_blinks( raw.get_data(picks=1)[0] , blink_df, params)

    print("\nProcessed Blink DataFrame:")
    # print(processed_df.head())

    print("\nBlink Statistics:")
    # print(stats)

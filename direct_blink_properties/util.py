import os
import pickle
import zipfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from pyblinkers.extractBlinkProperties import BlinkProperties, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks



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



def extract_blink_durations(annotation_df, frame_offset, sfreq, video_fps):
    """
    Extract blink durations from a CVAT-annotated DataFrame and align them with time series data.

    This function processes blink annotations that appear in sequential triplets: a blink start,
    a midpoint (typically the minimum eye aperture), and a blink end. It converts the frame-based
    annotations from video (e.g., annotated at 30 Hz) into sample indices compatible with a
    higher-resolution time series (e.g., EEG sampled at 100 Hz, 1000 Hz, etc.).

    Parameters
    ----------
    annotation_df : pd.DataFrame
        DataFrame containing blink annotations exported from CVAT, with each row corresponding
        to a labeled video frame. Expected labels include triplets like 'blink_start', 'blink_min',
        and 'blink_end'.
    frame_offset : int
        The number of frames to subtract from all annotated frame numbers to align them with
        the actual video frame indexing used during processing (e.g., if video frames were cropped).
    sfreq : float
        Sampling frequency (Hz) of the time series data (e.g., EEG). This value is typically
        obtained from `raw.info['sfreq']` in MNE.
    video_fps : float
        Frame rate of the video from which the annotations were made (e.g., 30 for 30 Hz video).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'startFrame', 'endFrame', 'minFrame': original frame indices from CVAT
        - 'startBlinks_cvat', 'endBlinks_cvat', 'blink_min_cvat': adjusted CVAT frame indices after subtracting frame_offset
        - 'startBlinks', 'endBlinks', 'blink_min': corresponding sample indices aligned to the time series, computed by scaling
          with `sfreq / video_fps` and rounding to nearest integer
        - 'blink_type': type/category of the blink (e.g., 'blink', 'long_blink', etc.)

    Notes
    -----
    This function ensures compatibility between frame-based annotations (from CVAT) and sample-based
    time series data by converting the annotated video frame indices to time series sample indices.
    This is critical when migrating blink labels into physiological data streams (e.g., EEG) for
    further analysis.
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

    df[['startBlinks_cvat', 'endBlinks_cvat', 'blink_min_cvat']] = df[['startFrame', 'endFrame', 'minFrame']] - frame_offset
    df[['startBlinks', 'endBlinks', 'blinkmin']] = (df[['startBlinks_cvat', 'endBlinks_cvat', 'blink_min_cvat']] * (sfreq / video_fps)).round().astype(int)


    return df
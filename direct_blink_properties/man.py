import os
import zipfile
import pickle
import pandas as pd
import mne

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
            blink_end = int(annotation_df.iloc[i+2]['ImageID'].replace('frame_', ''))
            blink_data.append({
                'startBlinks': blink_start,
                'endBlinks': blink_end,
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


if __name__ == "__main__":
    fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
    zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"

    # Optional parameters


    # Load data
    raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)

    # Extract blink intervals
    blink_df = extract_blink_durations(annotation_df)
    with open("fitblinks_debug.pkl", "rb") as f:
        debug_data = pickle.load(f)


    params = debug_data["params"]

    # Process and extract properties
    processed_df, stats = process_blinks( raw.get_data(picks=0)[0] , blink_df, params)

    print("\nProcessed Blink DataFrame:")
    print(processed_df.head())

    print("\nBlink Statistics:")
    print(stats)

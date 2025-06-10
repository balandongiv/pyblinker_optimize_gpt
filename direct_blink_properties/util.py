import hashlib
import os
import pickle
import zipfile
from dataclasses import dataclass
from dataclasses import field
from typing import Callable, Dict

import mne
import pandas as pd

from direct_blink_properties.signal_refinement import *


import os
import pickle
import zipfile
import mne
import pandas as pd


import mne
import pandas as pd

def load_and_pick_channels(fif_path, channel_names):
    """
    Load the raw FIF file and pick selected channels.
    Also extract annotations as a DataFrame.

    Parameters:
    - fif_path (str): Path to the .fif file.
    - channel_names (list of str): List of channel names to pick.

    Returns:
    - raw_picked (mne.io.Raw): Raw object with only the selected channels.
    - annot_df (pd.DataFrame): DataFrame with annotations (onset, duration, description).
    """
    # Load full raw file
    raw = mne.io.read_raw_fif(fif_path, preload=True)

    # Pick specified channels
    raw_picked = raw.copy().pick(channel_names)

    # Convert annotations to DataFrame
    if raw.annotations is not None and len(raw.annotations) > 0:
        annot_df = pd.DataFrame({
            'onset': raw.annotations.onset,
            'duration': raw.annotations.duration,
            'description': raw.annotations.description
        })
    else:
        annot_df = pd.DataFrame(columns=['onset', 'duration', 'description'])

    return raw_picked, annot_df


def load_fif_and_annotations(
        fif_path: str,
        zip_path: str,
        debug_dir: str = "./_debug_cache",
        use_cache: bool = True,
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
    use_cache : bool
        If True, use cached files if available.
    overwrite_cache : bool
        If True, overwrite any existing cached files.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing EEG/EOG/EAR signals.
    annotation_df : pd.DataFrame
        DataFrame of blink annotations.
    """
    if use_cache:
        os.makedirs(debug_dir, exist_ok=True)
        fif_cache = os.path.join(debug_dir, "cached_raw.pkl")
        ann_cache = os.path.join(debug_dir, "cached_annotations.pkl")
    else:
        fif_cache = ann_cache = None  # disables cache

    # --- Load or reload FIF file ---
    raw = None
    if use_cache and fif_cache and os.path.exists(fif_cache) and not overwrite_cache:
        print(f"Loading cached FIF from: {fif_cache}")
        with open(fif_cache, "rb") as f:
            raw = pickle.load(f)
    else:
        print(f"Loading FIF from: {fif_path}")
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        if use_cache and fif_cache:
            with open(fif_cache, "wb") as f:
                pickle.dump(raw, f)
            print(f"Cached FIF to: {fif_cache}")

    # --- Load or reload annotation CSV ---
    annotation_df = None
    if use_cache and ann_cache and os.path.exists(ann_cache) and not overwrite_cache:
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
        if use_cache and ann_cache:
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


def get_refiners() -> Dict[str, Callable]:
    return {
        # Not yet run
        # "loess": refine_loess,
        # "emd": refine_emd,
        # "zscore": refine_zscore,
        # "tophat": refine_tophat,
        # "highpass_iir": refine_highpass_iir,
        # "firhp": refine_firhp,
        # "ceemdan": refine_ceemdan,
        # "vmd": refine_vmd,
        # "epoch_poly": refine_epoch_poly,
        # "quantile_hp": refine_quantile_hp,
        # "quantile_hp_flip": refine_quantile_hp_epoch_flip,
        #
        # # Run, sorted by kept descending
        "quantile": refine_quantile,           # kept=1.000
        "highpass": refine_highpass,           # kept=0.959
        # "medfilt": refine_medfilt,             # kept=0.685
        # "als": refine_als,                     # kept=0.877
        # "wavelet": refine_wavelet,             # kept=0.863
        # "local_mean": refine_local_mean,       # kept=0.849
        # "airpls": refine_airpls,               # kept=0.603
        # "detrend_poly": refine_detrend_poly,   # kept=0.534
        # "detrend_linear": refine_detrend_linear,  # kept=0.479
        # "savgol": refine_savgol,               # kept=0.384
        # "global_mean": refine_global_mean,     # kept=0.397
        # "robust_scale": refine_robust,         # kept=0.288
        # "morph": refine_morph,                 # kept=0.247
        # "bandpass": refine_bandpass,           # kept=0.000
        # "whittaker": refine_whittaker          # kept=0.000
    }

def get_grids(df: pd.DataFrame, sfreq: float) -> Dict[str, List[Dict]]:
    epochs = list(zip(df.startBlinks, df.endBlinks))
    return {
        # Basic methods
        "global_mean": [{}],
        "local_mean": [{"epochs": epochs}],
        "detrend_linear": [{}],
        "detrend_poly": [{"order": o} for o in (2, 3, 4)],
        "savgol": [{"window": w, "polyorder": 3} for w in (151, 301, 601)],
        "robust_scale": [{}],

        # Filters
        "highpass": [{"fs": sfreq, "cutoff": c} for c in (0.1, 0.2, 0.3)],
        "highpass_iir": [{"fs": sfreq, "fc": fc} for fc in (0.15, 0.2, 0.25)],
        "bandpass": [{"fs": sfreq, "low": 0.1, "high": h} for h in (10, 15)],
        "firhp": [{"fs": sfreq, "fc": 0.25, "width": 0.1}],

        # Smoothing / trend removal
        "als": [{"lam": l, "p": p} for l in (1e4, 1e5) for p in (0.01, 0.05)],
        "airpls": [{"lam": l} for l in (1e4, 1e5, 1e6)],
        "whittaker": [{"lam": l} for l in (800, 1600, 3200)],
        "medfilt": [{"win": w} for w in (301, 501, 801)],
        "quantile": [{"win": w, "q": 0.05} for w in (401, 601)],
        "loess": [{"frac": f} for f in (0.01, 0.02, 0.03)],
        "zscore": [{"win": w} for w in (201, 301, 401)],
        "tophat": [{"size": s} for s in (301, 401, 501)],
        "morph": [{"size": s} for s in (101, 201, 301)],

        # Decomposition-based
        "wavelet": [{"wave": "db4", "level": lv} for lv in (4, 5, 6)],
        "emd": [{"keep_imfs": k} for k in (2, 3, 4)],
        "ceemdan": [{"keep_imfs": k} for k in (3, 4, 5)],
        "vmd": [{"K": k, "alpha": a} for k in (3, 4) for a in (1500, 2000)],

        # Epoch-aware
        "epoch_poly": [{"epochs": epochs, "order": o} for o in (1, 2)],
        "quantile_hp": [
            {"fs": sfreq, "win": w, "cutoff": c}
            for w in (401, 601, 801)
            for c in (0.10, 0.15, 0.20)
        ],
        "quantile_hp_flip": [
            {"fs": sfreq, "epochs": epochs, "win": w, "q": 0.05, "cutoff": c}
            for w in (401, 601, 801)
            for c in (0.1, 0.15, 0.2)
        ]
    }

# -----------------------------------------------------------------------------
# Helper – hashing utils for caching
# -----------------------------------------------------------------------------

def _hash(obj) -> str:
    """Stable md5 hash of any picklable object."""
    return hashlib.md5(pickle.dumps(obj, protocol=4)).hexdigest()



# -----------------------------------------------------------------------------
# Dataclass containers
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
    kept_ratio: float
    positive_peak_fraction: float
    median_peak_amp: float
    df: pd.DataFrame = field(repr=False, compare=False)

    def to_dict(self):
        # only include the numeric fields
        return {
            "kept_ratio": self.kept_ratio,
            "positive_peak_fraction": self.positive_peak_fraction,
            "median_peak_amp": self.median_peak_amp,
        }

@dataclass
class ExperimentResult:
    method: str
    params: Dict
    metrics: Metrics

    def to_row(self):
        d = self.metrics.to_dict()
        d.update(method=self.method)
        return d
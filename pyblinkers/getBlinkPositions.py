import numpy as np
import pandas as pd

from pyblinkers.default_setting import SCALING_FACTOR
from pyblinkers.matlab_forking import mad_matlab


def compute_global_stats(good_data, ch_idx, params):
    """
    Compute global statistics for blink detection.

    Parameters:
        good_data (numpy.ndarray): Valid epoch data.
        ch_idx (int): Index of the channel to process.
        params (dict): Parameters for blink detection.

    Returns:
        tuple: (mu, robust_std, threshold, min_blink_frames)
    """
    if good_data.ndim == 1:
        blink_component=good_data
        min_blink_frames = params['minEventLen'] * params['sfreq'] # for continous, we use what has been used in the original implementation
    else:

        blink_component = good_data[:, ch_idx, :].reshape(-1) # This is a 1D array of all the epochs now being flattened into 1D
        min_blink_frames=1 # For epoch, we need to set it to 1
    mu = np.mean(blink_component, dtype=np.float64)
    mad_val = mad_matlab(blink_component)
    robust_std = SCALING_FACTOR * mad_val


    threshold = mu + params['stdThreshold'] * robust_std
    return dict(
        mu=mu,
        robust_std=robust_std,
        threshold=threshold,
        min_blink_frames=min_blink_frames,
        mad_val=mad_val
    )
def get_blink_position(
        params: dict,
        blink_component: np.ndarray | None = None,
        ch: str | None = None,
        threshold: float | None = None,
        min_blink_frames: int | None = None,
) -> pd.DataFrame:
    """
    Detect blink start- and end-frame indices in a 1-D blink component.

    Parameters
    ----------
    params : dict
        Must contain:
        • 'sfreq' (float) – sampling rate in Hz
        • 'minEventLen' (float) – minimum gap allowed between two blinks (s)
    blink_component : 1-D ndarray
        Independent component or channel dominated by eye blinks.
    ch : str, optional
        Channel name (kept for compatibility; no longer used internally).
    threshold : float
        Amplitude threshold that defines “inside a blink”.
    min_blink_frames : int
        Minimum number of consecutive frames over threshold to count as a blink.

    Returns
    -------
    pd.DataFrame with columns ['startBlinks', 'endBlinks']
    """
    if blink_component is None:
        raise ValueError("blink_component must be provided")
    if threshold is None or min_blink_frames is None:
        raise ValueError("threshold and min_blink_frames must be provided")

    # --- 1. binary mask of “blink” frames -----------------------------------
    mask = blink_component > threshold

    # --- 2. locate rising & falling edges -----------------------------------
    diff = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(diff == 1) + 1          # +1 because diff is shifted
    ends   = np.flatnonzero(diff == -1) + 1

    # handle blink starting at index 0 or ending at last sample
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, mask.size - 1)

    if starts.size == 0 or ends.size == 0:
        # No complete blink found
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []})

    # --- 3. enforce minimum blink duration ----------------------------------
    durations = ends - starts
    valid = durations > min_blink_frames
    starts, ends = starts[valid], ends[valid]

    if starts.size == 0:
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []})

    # --- 4. remove blinks that are too close together -----------------------
    sfreq = params["sfreq"]
    min_gap_sec = params["minEventLen"]

    # gap between current start and previous end (in seconds)
    gaps = (starts[1:] - ends[:-1]) / sfreq
    too_close = gaps <= min_gap_sec

    # mark both members of each too-close pair for deletion
    keep = np.ones_like(starts, dtype=bool)
    idx = np.where(too_close)[0]
    keep[idx] = False          # previous blink
    keep[idx + 1] = False      # next blink

    starts, ends = starts[keep], ends[keep]

    return pd.DataFrame({'startBlinks': starts, 'endBlinks': ends})

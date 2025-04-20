from pyblinkers.utils._logging import logger

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinkers.default_setting import SCALING_FACTOR

from pyblinkers.matlab_forking import mad_matlab


def get_blink_position(params, blink_component=None, ch=None,threshold=None,min_blink_frames=None):
    """Detects blink positions (start and end frames) in a blink component.
    
    Parameters
    ----------
    params : dict
        A dictionary containing processing parameters, which must include:
        - 'sfreq' (float): Sampling frequency of the candidate_signal in Hz.
        - 'minEventLen' (float): Minimum blink length in seconds.
        - 'stdThreshold' (float): Standard deviation threshold for blink detection.
    blink_component : numpy.ndarray
        A 1D array representing the blink component (e.g., an independent component related to eye blinks).
    ch : str, optional
        The name of the channel for logging purposes. Default is None.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing two columns:
        - 'startBlinks' (numpy.ndarray): Indices of the start frames of detected blinks.
        - 'endBlinks' (numpy.ndarray): Indices of the end frames of detected blinks.
        If no blinks are detected, an empty DataFrame with the same column names is returned.
    """
    if threshold is None:
        # Ensure 1D array
        # assert blink_component.ndim == 1, "blink_component must be a 1D array"

        # Compute basic statistics
        mu = np.mean(blink_component, dtype=np.float64)
        mad_val = mad_matlab(blink_component)
        robust_std= SCALING_FACTOR * mad_val

        # Minimum blink length in frames
        min_blink_frames = params['minEventLen'] * params['sfreq']
        threshold = mu + params['stdThreshold'] * robust_std
    else:
        threshold=threshold
        min_blink_frames=min_blink_frames
    in_blink = False
    start_blinks = []
    end_blinks = []

    for idx in tqdm(range(blink_component.size), desc=f"Get blink start and end for channel {ch}"):
        val = blink_component[idx]

        # Start condition
        if (not in_blink) and (val > threshold):
            start = idx
            in_blink = True

        # End condition
        elif in_blink and (val < threshold):
            if (idx - start) > min_blink_frames:
                start_blinks.append(start)
                end_blinks.append(idx)
            in_blink = False

    # Convert lists to arrays
    arr_start = np.array(start_blinks)
    arr_end = np.array(end_blinks)

    if arr_end.size == 0:
        # No blinks found, return empty DataFrame
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []})

    # Remove blinks that are too close together (< minEventLen apart)
    pos_mask = np.ones(arr_end.size, dtype=bool)
    # Differences between consecutive end and subsequent start
    blink_durations = (arr_start[1:] - arr_end[:-1]) / params['sfreq']
    close_indices = np.argwhere(blink_durations <= params['minEventLen'])

    # Invalidate both the earlier and later blink intervals
    pos_mask[close_indices] = False
    pos_mask[close_indices + 1] = False

    blink_position = {
        'startBlinks': arr_start[pos_mask],
        'endBlinks': arr_end[pos_mask]
    }
    return pd.DataFrame(blink_position)


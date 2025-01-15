# LLMed on 15 January 2025

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinkers.misc import mad_matlab
from pyblinkers.default_setting import scalingFactor

logging.getLogger().setLevel(logging.INFO)


def getBlinkPosition(params, blinkComp=None, ch='No_channel'):
    """
    in this refactoring process, I remove the srate into the params
    % Parameters:
    %    blinkComp       independent component (IC) of eye-related
    %                    activations derived from EEG.  This component should
    %                    be "upright"
    %    srate:         sample rate at which the EEG data was taken
    %    stdTreshold    number of standard deviations above threshold for blink
    %    blinkPositions (output) 2 x n array with start and end frames of blinks
        :param params:
        :param sfreq:
        :param signal_eeg:
        :param ch:
        :return:
    """

    # Ensure 1D array
    assert blinkComp.ndim == 1, "blinkComp must be a 1D array"

    # Compute basic statistics
    mu = np.mean(blinkComp, dtype=np.float64)
    mad_val = mad_matlab(blinkComp)
    robust_std= scalingFactor * mad_val

    # Minimum blink length in frames
    min_blink_frames = params['minEventLen'] * params['sfreq']
    threshold = mu + params['stdThreshold'] * robust_std

    in_blink = False
    start_blinks = []
    end_blinks = []

    # Detect blink start/end
    for idx in tqdm(range(blinkComp.size), desc=f"Get blink start and end for channel {ch}"):
        val = blinkComp[idx]

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




def getBlinkPosition_pythonic(params, blinkComp=None, ch='No_channel'):
    """
    we can do experiment with the vectorized approach, to see if it is faster than the previous approach


    Vectorized approach to find blink start/end positions from a blink component.

    Parameters
    ----------
    params : dict
        Must contain keys:
            'minEventLen'  : minimal blink length in seconds
            'sfreq'        : sampling frequency (Hz)
            'stdThreshold' : number of robust standard deviations above mean
    blinkComp : np.ndarray
        1D array containing the blink component data
    ch : str
        Channel name, used only for logging

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with 'startBlinks' and 'endBlinks'
    """

    # Ensure 1D array
    assert blinkComp.ndim == 1, "blinkComp must be a 1D array"

    # Compute basic statistics
    mu = np.mean(blinkComp, dtype=np.float64)
    mad_val = mad_matlab(blinkComp)
    robust_std = scalingFactor * mad_val

    # Minimum blink length in frames
    min_blink_frames = int(params['minEventLen'] * params['sfreq'])
    threshold = mu + params['stdThreshold'] * robust_std

    # Boolean mask: True where data exceeds threshold
    over_mask = blinkComp > threshold

    # Find transitions: +1 -> blink starts, -1 -> blink ends
    # np.diff(...) yields array of length N-1, so add 1 to positions
    transitions = np.diff(over_mask.astype(int))
    blink_starts = np.where(transitions == 1)[0] + 1
    blink_ends   = np.where(transitions == -1)[0] + 1

    # Edge cases:
    # If the very first sample is above threshold, we "miss" a start at idx=0
    # but only if there's at least one blink_end. Conversely for the last sample.
    if over_mask[0] and len(blink_ends) > 0 and (blink_ends[0] > blink_starts[0]):
        # We can treat index 0 as a start
        blink_starts = np.insert(blink_starts, 0, 0)
    if over_mask[-1] and len(blink_starts) > 0 and (blink_starts[-1] < blink_ends[-1]):
        # We can treat last index as an end
        blink_ends = np.append(blink_ends, len(blinkComp) - 1)

    # If no blink starts or ends found, return empty
    if len(blink_starts) == 0 or len(blink_ends) == 0:
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []})

    # Sometimes we might find an "end" that occurs before the first "start"
    # or a "start" that occurs after the last "end". We'll drop those extra elements.
    # e.g., if blink_ends[0] < blink_starts[0], remove that blink end
    if blink_ends[0] < blink_starts[0]:
        blink_ends = blink_ends[1:]
    # if blink_starts[-1] > blink_ends[-1], remove that blink start
    if blink_starts[-1] > blink_ends[-1]:
        blink_starts = blink_starts[:-1]

    # Now pair up starts and ends
    # We'll assume they match in order, i.e. start[i] < end[i]
    pair_len = min(len(blink_starts), len(blink_ends))
    arr_start = blink_starts[:pair_len]
    arr_end   = blink_ends[:pair_len]

    # Filter out blinks that are too short
    blink_lengths = arr_end - arr_start
    valid_mask = blink_lengths >= min_blink_frames
    arr_start = arr_start[valid_mask]
    arr_end   = arr_end[valid_mask]

    # If everything got filtered out, return empty
    if len(arr_start) == 0:
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []})

    # Remove consecutive blinks that are too close:
    # i.e., if the next blink's start is within `minEventLen` seconds
    # after the current blink's end, we merge or remove them.
    # We'll remove whichever is considered "too close" to keep them distinct.
    gap_frames = np.diff(arr_start)
    gap_seconds = gap_frames / params['sfreq']
    too_close = gap_seconds <= params['minEventLen']

    # We'll keep a simple approach:
    # if blink i and blink i+1 are too close, remove both.
    # You can also choose to merge or keep the first blink only, etc.
    keep_mask = np.ones(len(arr_start), dtype=bool)
    close_indices = np.where(too_close)[0]
    # Invalidate both: the earlier and the later blink intervals
    for ci in close_indices:
        keep_mask[ci] = False     # current blink
        if ci + 1 < len(keep_mask):
            keep_mask[ci + 1] = False

    arr_start = arr_start[keep_mask]
    arr_end   = arr_end[keep_mask]

    # Build the final DataFrame
    blink_position = {
        'startBlinks': arr_start,
        'endBlinks': arr_end
    }

    return pd.DataFrame(blink_position)

'''

This is how or module should looks like once we got the best approach for finding the zero crossing and ensure the
blink max is positive value
'''

import mne
import numpy as np
import pandas as pd

from pyblinkers.extractBlinkProperties import BlinkProperties, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks


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
    candidate_signal = -candidate_signal
    candidate_signal = candidate_signal - np.mean(candidate_signal)

    fitblinks = FitBlinks(candidate_signal, df, params)
    fitblinks.process_blink_candidate()
    df = fitblinks.frame_blinks

    blink_stats = get_blink_statistic(df, params['z_thresholds'], candidate_signal)

    # Optional filtering step (commented out by default)
    # good_blink_mask, df = get_good_blink_mask(df, blink_stats['bestMedian'], blink_stats['bestRobustStd'], params['z_thresholds'])

    df = BlinkProperties(candidate_signal, df, params['sfreq'], params).df


    return df, blink_stats
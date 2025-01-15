import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinkers.misc import mad_matlab

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
    assert blinkComp.ndim == 1
    # minEventLen = 0.05 # events less than 50 ms are discarded
    # blinkComp = signal_eeg
    mu = np.mean(blinkComp, dtype=np.float64)

    mad_val = mad_matlab(blinkComp)
    robustStdDev = 1.4826 * mad_val

    minBlink = params['minEventLen'] * params['sfreq']  # minimum blink frames
    threshold = mu + params['stdThreshold'] * robustStdDev  # actual threshold

    '''
    % The return structure.  Initially there is room for an event at every time
    % tick, to be trimmed later
    '''

    inBlink = False
    startBlinks = []
    endBlinks = []
    # kk=blinkComp.size
    for index in tqdm(range(blinkComp.size), desc=f"Get blink start and end for channel {ch}"):
        Drule = ~inBlink and (blinkComp[index] > threshold)
        # If amplitude exceeds threshold and not currently detecting a blink
        if Drule:
            start = index
            inBlink = np.ones((1), dtype=bool)

        # if previous point was in a blink and signal retreats below threshold and duration greater than discard
        # threshold
        krule = (inBlink == True) and (blinkComp[index] < threshold)
        if krule:
            if (index - start) > minBlink:
                startBlinks.append(start)  # t_up
                endBlinks.append(index)  # t_dn

            inBlink = False

    arr_startBlinks = np.array(startBlinks)
    arr_endBlinks = np.array(endBlinks)

    # Now remove blinks that aren't separated
    positionMask = np.ones(arr_endBlinks.size, dtype=bool)

    x = (arr_startBlinks[1:] - arr_endBlinks[:-1]) / params['sfreq']  # Calculate the blink duration
    y = np.argwhere(x <= params['minEventLen'])  # Index where blink duration is less than 0.05 sec
    positionMask[y] = np.zeros((1), dtype=bool)
    positionMask[y + 1] = np.zeros((1), dtype=bool)
    # v=arr_startBlinks[positionMask]
    blink_position = {'startBlinks': arr_startBlinks[positionMask],
                      'endBlinks': arr_endBlinks[positionMask]}
    # df=pd.DataFrame(blink_position)
    return pd.DataFrame(blink_position)



def getBlinkPositionPython(params, blinkComp=None, ch='No_channel'):
    """

    This still not give the desire output. to many error


    Detects blinks in EEG data by finding segments where the signal exceeds a certain threshold.
    Utilizes a vectorized approach for efficiency.

    Parameters:
        params (dict): Parameters including 'minEventLen', 'sfreq', and 'stdThreshold'.
        blinkComp (1D array): Independent component (IC) of eye-related activations derived from EEG.
        ch (str): Channel name.

    Returns:
        dict: Contains 'start_blink', 'end_blink', and 'ch' keys with corresponding values.
    """

    # Ensure 1D array
    assert blinkComp.ndim == 1

    # Compute mean and robust standard deviation
    mu = np.mean(blinkComp, dtype=np.float64)
    mad_val = mad_matlab(blinkComp)
    robustStdDev = 1.4826 * mad_val

    # Calculate minimum blink frames and threshold
    minBlink = params['minEventLen'] * params['sfreq']  # minimum blink frames
    threshold = mu + params['stdThreshold'] * robustStdDev  # actual threshold

    # Vectorized detection of threshold crossings
    above_threshold = blinkComp > threshold
    # Pad the above_threshold array to detect events that start at the first sample or end at the last sample
    padded = np.r_[False, above_threshold, False]
    diff_above_threshold = np.diff(padded.astype(int))

    # Find start (rising edge) and end (falling edge) indices of blinks
    starts = np.where(diff_above_threshold == 1)[0]
    ends = np.where(diff_above_threshold == -1)[0]

    # Compute durations and filter out short blinks
    event_durations = ends - starts
    valid_events = event_durations >= minBlink
    startBlinks = starts[valid_events]
    endBlinks = ends[valid_events]

    # Remove blinks that aren't sufficiently separated
    if len(startBlinks) > 1:
        inter_blink_intervals = (startBlinks[1:] - endBlinks[:-1]) / params['sfreq']
        close_blinks = np.where(inter_blink_intervals <= params['minEventLen'])[0]
        positionMask = np.ones(len(startBlinks), dtype=bool)
        # Ensure we don't go out of bounds
        positionMask[close_blinks] = False
        positionMask[close_blinks + 1] = False
        startBlinks = startBlinks[positionMask]
        endBlinks = endBlinks[positionMask]

    # Prepare the output dictionary
    blink_position = {
        'start_blink': startBlinks,
        'end_blink': endBlinks,
        'ch': ch
    }

    return blink_position

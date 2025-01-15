import warnings

import numpy as np
import pandas as pd
# from scipy import signal
from scipy.signal import find_peaks


def left_zero_crossing(candidateSignal, maxFrame, outerStarts):
    """

    For actual implementation, better use the function to find the zero crossing seprate. For two reason
    1: Speed
        1.4880754947662354 s >> time combine to run left_zero_crossing and right_zero_crossing
        1.5236587524414062 s  >> time to execute left_right_zero_crossing
    2: Debug process
        Easy to debug due to the modular architecture

    3: We can experiment with diffrent approach to get the zero crossing
    4: Btw, need to be carefull for a case whereby, there is no negative value!
    :param candidateSignal:
    :param maxFrame:
    :param outerStarts:
    :return:
    """
    theRange = np.arange(outerStarts, maxFrame)
    sInd_leftZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_leftZero.size != 0):
        leftZero = theRange[sInd_leftZero[-1]]

    else:
        ## TODO: We need to consider also taking the positive value instead should the -ve index frame is to far away!
        extreme_outerStartss = np.arange(0, maxFrame)
        sInd_rightZero_ex = np.flatnonzero(candidateSignal[extreme_outerStartss] < 0)[-1]
        leftZero = extreme_outerStartss[sInd_rightZero_ex]

    if leftZero > maxFrame:
        warnings.warn('something is not right')
        return np.nan

    return leftZero.astype(int)
    # return leftZero, rightZero


def right_zero_crossing(candidateSignal, maxFrame, outerEnds):
    theRange = np.arange(maxFrame, outerEnds)
    sInd_rightZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_rightZero.size != 0):
        rightZero = theRange[sInd_rightZero[0]]
    else:
        """
        We take extreme remedy by extending the outerEnds to the maximum
        ## TODO: We need to consider also taking the positive value instead should the -ve index frame is to far away!
        """
        extreme_outerEns = np.arange(maxFrame, candidateSignal.shape)
        sInd_rightZero_ex_s = np.flatnonzero(candidateSignal[extreme_outerEns] < 0)

        if (sInd_rightZero_ex_s.size != 0):
            # This usually happen for end of signal
            sInd_rightZero_ex = sInd_rightZero_ex_s[0]
            rightZero = extreme_outerEns[sInd_rightZero_ex]
        else:
            return None

    if maxFrame > rightZero:
        warnings.warn('something is not right')
        return np.nan
        # raise ValueError('something is not right')

    return rightZero.astype(int)


def left_right_zero_crossing(candidateSignal, maxFrame, outerStarts, outerEnds):
    """
    df[['size_mb','tt']] = df.apply(lambda x: left_right_zero_crossing(data, x['peaks_point'], x['leftp'],x['rightp']), axis=1, result_type="expand")

    Get the index of last negative value, before the signal cross over the 0
    :param candidateSignal:
    :param maxFrame:
    :param outerStarts:
    :param outerEnds:
    :return:
    """
    theRange = np.arange(outerStarts, maxFrame)
    sInd_leftZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_leftZero.size != 0):
        leftZero = theRange[sInd_leftZero[-1]]

    else:
        extreme_outerStartss = np.arange(0, maxFrame)
        sInd_rightZero_ex = np.flatnonzero(candidateSignal[extreme_outerStartss] < 0)[-1]
        leftZero = extreme_outerStartss[sInd_rightZero_ex]

    theRange = np.arange(maxFrame, outerEnds)
    sInd_rightZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_rightZero.size != 0):
        rightZero = theRange[sInd_rightZero[0]]
    else:
        """
        We take extreme remedy by extending the outerEnds to the maximum
        """
        extreme_outerEns = np.arange(maxFrame, candidateSignal.shape)
        sInd_rightZero_ex_s = np.flatnonzero(candidateSignal[extreme_outerEns] < 0)

        if (sInd_rightZero_ex_s.size != 0):
            # This usually happen for end of signal
            sInd_rightZero_ex = sInd_rightZero_ex_s[0]
            rightZero = extreme_outerEns[sInd_rightZero_ex]
        else:
            return leftZero, None

    if leftZero > maxFrame:
        raise ValueError('something is not right')

    if maxFrame > rightZero:
        raise ValueError('something is not right')

    # return leftZero
    return leftZero, rightZero


def _adjacent_valley(valley_indexes, peaks):
    i = np.searchsorted(valley_indexes, peaks)
    m = ~np.isin(i, [0, len(valley_indexes)])

    df = pd.DataFrame({'peaks_point': peaks})
    df.loc[m, 'base_left'], df.loc[m, 'base_right'] = valley_indexes[i[m] - 1], valley_indexes[i[m]]
    df=df.dropna().reset_index(drop=True)
    df = df.astype(int)
    return df


def _valley_argrelextrema(data):
    """
    A bit slow compare to inverting approach
    0.0021774768829345703  : argrelextrema
    0.0015132427215576172   : inverting
    :param data:
    :return:

    require # from scipy import signal
    """
    return signal.argrelextrema(data, np.less)[0]


def _valley_flip(data):
    # data=data*-1
    peaks, _ = find_peaks(data * -1)
    return peaks


def get_peak_base_zero_crossing(data, height=0):
    """

    WIP:

    Consider supplying the height using

    mu = np.mean(blinkComp, dtype=np.float64)

    mad_val = mad_matlab(blinkComp)
    robustStdDev = 1.4826 * mad_val

    minBlink = params['minEventLen'] * sfreq  # minimum blink frames
    threshold = mu +  robustStdDev  # actual threshold

    :param data:
    :param height:
    :return:
    """
    # data = electrocardiogram()
    peaks, _ = find_peaks(data, height=height)


    vpeak2 = _valley_flip(data)
    df = _adjacent_valley(vpeak2, peaks)

    df['zero_left'] = df.apply(lambda x: left_zero_crossing(data, x['peaks_point'], x['base_left']), axis=1)

    df['zero_right'] = df.apply(lambda x: right_zero_crossing(data, x['peaks_point'], x['base_right']), axis=1)

    return df


def how_to():
    """
    Tutotrial on how to find the peak, zero_Crossing, and base

    :return:
    """
    from scipy.misc import electrocardiogram
    from rblinks.viz_rblinks import viz_peak_zero_crossing

    data = electrocardiogram()[2000:2500]
    # data = electrocardiogram()
    df = get_peak_base_zero_crossing(data)
    viz_peak_zero_crossing(data, df)


how_to()


import mne
import pandas as pd



def create_annotation(sblink, sfreq, label):


    if not isinstance(sblink, pd.DataFrame):
        raise ValueError('No appropriate channel. sorry. Try to use large channel selection')


    onset = (sblink['startBlinks'] / sfreq).tolist()
    duration= (sblink['endBlinks'] - sblink['startBlinks']) / sfreq
    des_s = [label] * len(onset)

    annot = mne.Annotations(onset=onset,  # in seconds
                            duration=duration,  # in seconds, too
                            description=des_s)

    return annot
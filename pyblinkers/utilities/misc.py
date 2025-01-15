
import numpy as np
import mne
import os
import re
import shutil
import pandas as pd
def mad_matlab(arr, axis=None, keepdims=True):
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=axis, keepdims=keepdims)[0]
    return mad

def check_make_folder(path, remove=False):
    if not remove:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    else:
        '''
        Some time I want the folder to be emptied
        '''
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)


def create_annotation(sblink, sfreq, label):
    # st_blink = 'leftZero'
    # blink_en = 'rightZero'

    if not isinstance(sblink, pd.DataFrame):
        raise ValueError('No appropriate channel. sorry. Try to use large channel selection')


    onset_s = (sblink['maxFrame'] / sfreq).tolist()
    des_s = [label] * len(onset_s)
    d_s=[0]* len(onset_s)

    annot = mne.Annotations(onset=onset_s,  # in seconds
                            duration=d_s,  # in seconds, too
                            description=des_s)

    return annot
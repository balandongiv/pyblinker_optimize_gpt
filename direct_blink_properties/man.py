import pickle

import mne
import numpy as np
import pandas as pd

from direct_blink_properties.util import load_fif_and_annotations, extract_blink_durations


def process_blinks(candidate_signal, df, params):
    """
    Processes raw EEG data to analyze and extract properties of eye blinks.

    Parameters
    ----------
    candidate_signal : mne.io.Raw
        Raw EEG signal from which blink characteristics are to be extracted.

    df : pd.DataFrame
        DataFrame containing ground truth blink intervals, typically verified by a human expert.
        In unit tests, this DataFrame contains 73 rows. The most relevant columns include:
        - 'startBlinks': Frame index marking the start of a blink.
        - 'endBlinks': Frame index marking the end of a blink.
        - 'blink_min': Frame index representing the blink minimum (usually peak amplitude).

        All time-related values are in frame units and correspond to the candidate_signal's time base.

    params : dict
        Dictionary of configuration parameters. Expected keys include:
        - 'sfreq': Sampling frequency of the signal.
        - 'z_thresholds': Thresholds used for blink detection or validation.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame augmented with additional columns describing extracted blink properties.
    """

    candidate_signal = -candidate_signal
    candidate_signal = candidate_signal - np.mean(candidate_signal)

    # find the left right zero crossing, and drop rows with NaN values or the different between the zero crossing of left and right is less than 3 frames
    ## there should be some function here

    # then we test whether the total number of rows of df is similar to the output of total rows of the left right zero crossing
    # by right it should be the same



    return df

if __name__ == "__main__":
    fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
    zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"

    # Optional parameters


    # Load data
    raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)

    # get the sampling rate
    sfreq = raw.info['sfreq']
    video_fps=30

    # Extract blink intervals
    frame_offset=5
    blink_df = extract_blink_durations(annotation_df,frame_offset,sfreq,video_fps)


    with open("fitblinks_debug.pkl", "rb") as f:
            debug_data = pickle.load(f)
    params = debug_data["params"]


    # # Process and extract properties
    processed_df, stats = process_blinks( raw.get_data(picks=1)[0] , blink_df, params)
    #
    print("\nProcessed Blink DataFrame:")
    # print(processed_df.head())

    print("\nBlink Statistics:")
    # print(stats)

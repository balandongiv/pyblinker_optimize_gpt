import numpy as np
import pandas as pd
from pyblinkers.zero_crossing import (_max_pos_vel_frame, _get_left_base, _get_right_base)

def create_left_right_base(candidate_signal, df):
    """
    Computes the left and right base values for each row in the DataFrame df,
    using the blink velocity derived from the input signal candidate_signal. The function
    also adds columns for the maximum positive and negative velocity frames to df.

    Parameters
    ----------
    candidate_signal : numpy.ndarray
        A 1D array of signal candidate_signal representing the blink component. The length
        of this array must match the number of rows in the DataFrame df, as it
        is used to compute the blink velocity.

    df : pandas.DataFrame
        A DataFrame containing the following required columns:
        - 'max_blink' (int): The maximum frame index for the blink event.
        - 'left_zero' (int): The index of the left zero crossing.
        - 'right_zero' (int): The index of the right zero crossing.
        - 'outer_start' (int): The starting index of the outer blink event.
        - 'outer_end' (int): The ending index of the outer blink event.
        Additional columns may be present but are not utilized in this function.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the following additional columns:
        - 'max_pos_vel_frame' (int): The frame index of the maximum positive velocity
          calculated from the blink velocity.
        - 'max_neg_vel_frame' (int): The frame index of the maximum negative velocity
          calculated from the blink velocity.
        - 'left_base' (float): The calculated left base value for the blink event,
          derived from the blink velocity and the specified outer start index.
        - 'right_base' (float): The calculated right base value for the blink event,
          derived from the blink velocity and the specified outer end index.
        Rows with NaN values in any of these new columns are dropped from the DataFrame.
    """

    # Ensure df is a fresh copy to prevent SettingWithCopyWarning
    df = df.copy()

    # Compute blink velocity by differencing the candidate_signal
    blinkVelocity = np.diff(candidate_signal, axis=0)

    # Remove rows with NaNs so we don't pass invalid candidate_signal to our calculations
    df.dropna(inplace=True)

    # Calculate max_pos_vel_frame and max_neg_vel_frame safely
    df[['max_pos_vel_frame', 'max_neg_vel_frame']] = df.apply(
        lambda row: _max_pos_vel_frame(
            blink_velocity=blinkVelocity,
            max_blink=row['max_blink'],
            left_zero=row['left_zero'],
            right_zero=row['right_zero']
        ),
        axis=1,
        result_type='expand'
    )

    # Ensure df is a new variable after filtering
    df = df[df['outer_start'] < df['max_pos_vel_frame']].copy()

    # Compute left_base safely using .assign()
    df = df.assign(left_base=df.apply(
        lambda row: _get_left_base(
            blink_velocity=blinkVelocity,
            left_outer=row['outer_start'],
            max_pos_vel_frame=row['max_pos_vel_frame']
        ),
        axis=1
    ))

    # Drop rows with NaNs again if any were introduced
    df.dropna(inplace=True)

    # Compute right_base safely using .assign()
    df = df.assign(right_base=df.apply(
        lambda row: _get_right_base(
            candidate_signal=candidate_signal,
            blink_velocity=blinkVelocity,
            right_outer=row['outer_end'],
            max_neg_vel_frame=row['max_neg_vel_frame']
        ),
        axis=1
    ))

    return df

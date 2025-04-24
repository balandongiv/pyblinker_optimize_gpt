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
        - 'maxFrame' (int): The maximum frame index for the blink event.
        - 'leftZero' (int): The index of the left zero crossing.
        - 'rightZero' (int): The index of the right zero crossing.
        - 'outerStarts' (int): The starting index of the outer blink event.
        - 'outerEnds' (int): The ending index of the outer blink event.
        Additional columns may be present but are not utilized in this function.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the following additional columns:
        - 'maxPosVelFrame' (int): The frame index of the maximum positive velocity
          calculated from the blink velocity.
        - 'maxNegVelFrame' (int): The frame index of the maximum negative velocity
          calculated from the blink velocity.
        - 'leftBase' (float): The calculated left base value for the blink event,
          derived from the blink velocity and the specified outer start index.
        - 'rightBase' (float): The calculated right base value for the blink event,
          derived from the blink velocity and the specified outer end index.
        Rows with NaN values in any of these new columns are dropped from the DataFrame.
    """

    # Ensure df is a fresh copy to prevent SettingWithCopyWarning
    df = df.copy()

    # Compute blink velocity by differencing the candidate_signal
    blinkVelocity = np.diff(candidate_signal, axis=0)

    # Remove rows with NaNs so we don't pass invalid candidate_signal to our calculations
    df.dropna(inplace=True)

    # # Calculate maxPosVelFrame and maxNegVelFrame safely
    df[['maxPosVelFrame', 'maxNegVelFrame']] = df.apply(
        lambda row: _max_pos_vel_frame(
            blink_velocity=blinkVelocity,
            max_frame=row['maxFrame'],
            left_zero=row['leftZero'],
            right_zero=row['rightZero']
        ),
        axis=1,
        result_type='expand'
    )
    max_pos_vel_frames = []
    # max_neg_vel_frames = []

    # Safely compute maxPosVelFrame and maxNegVelFrame
    # for _, row in df.iterrows():
    #     try:
    #         max_pos, max_neg = _max_pos_vel_frame(
    #             blink_velocity=blinkVelocity,
    #             max_frame=row['maxFrame'],
    #             left_zero=row['leftZero'],
    #             right_zero=row['rightZero']
    #         )
    #         max_pos_vel_frames.append(max_pos)
    #         max_neg_vel_frames.append(max_neg)
    #     except Exception as e:
    #         # Log warning and fill with NaN if any error
    #         print(f"Warning: Could not compute max velocities for row {row.name}: {e}")
    #         max_pos_vel_frames.append(np.nan)
    #         max_neg_vel_frames.append(np.nan)

    # df['maxPosVelFrame'] = max_pos_vel_frames
    # df['maxNegVelFrame'] = max_neg_vel_frames
    # Ensure df is a new variable after filtering
    df = df[df['outerStarts'] < df['maxPosVelFrame']].copy()

    left_bases = []
    for idx, row in df.iterrows():
        try:
            base = _get_left_base(
                blink_velocity=blinkVelocity,
                left_outer=row['outerStarts'],
                max_pos_vel_frame=row['maxPosVelFrame']
            )
            left_bases.append(base)
        except Exception as e:
            print(f"Warning: Failed left base at row {idx}: {e}")
            left_bases.append(np.nan)

    df['leftBase'] = left_bases
    df.dropna(subset=['leftBase'], inplace=True)

    # Drop rows with NaNs again if any were introduced
    df.dropna(inplace=True)

    # Compute rightBase safely using .assign()
    right_bases = []
    for idx, row in df.iterrows():
        try:
            base = _get_right_base(
                candidate_signal=candidate_signal,
                blink_velocity=blinkVelocity,
                right_outer=row['outerEnds'],
                max_neg_vel_frame=row['maxNegVelFrame']
            )
            right_bases.append(base)
        except Exception as e:
            print(f"Warning: Failed right base at row {idx}: {e}")
            right_bases.append(np.nan)

    df['rightBase'] = right_bases
    df.dropna(subset=['rightBase'], inplace=True)

    return df

import numpy as np
from pyblinkers.zero_crossing import (_maxPosVelFrame, _get_left_base, _get_right_base)

def create_left_right_base(data, df):
    """

    Computes the left and right base values for each row in the DataFrame df,
    using the blink velocity derived from the input signal data. The function
    also adds columns for the maximum positive and negative velocity frames to df.

    Parameters
    ----------
    data : numpy.ndarray
        A 1D array of signal data representing the blink component. The length
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

    # Compute blink velocity by differencing the data
    blinkVelocity = np.diff(data, axis=0)

    # Remove rows with NaNs so we don't pass invalid data to our calculations
    df.dropna(inplace=True)

    # Calculate maxPosVelFrame and maxNegVelFrame
    df[['maxPosVelFrame', 'maxNegVelFrame']] = df.apply(
        lambda row: _maxPosVelFrame(
            blinkVelocity=blinkVelocity,
            maxFrame=row['maxFrame'],
            leftZero=row['leftZero'],
            rightZero=row['rightZero']
        ),
        axis=1,
        result_type='expand'
    )

    # Filter out anomalous rows where outerStarts >= maxPosVelFrame
    df = df[df['outerStarts'] < df['maxPosVelFrame']]

    # Calculate leftBase
    df['leftBase'] = df.apply(
        lambda row: _get_left_base(
            blinkVelocity=blinkVelocity,
            leftOuter=row['outerStarts'],
            maxPosVelFrame=row['maxPosVelFrame']
        ),
        axis=1
    )

    # Drop rows with NaNs again if any were introduced
    df.dropna(inplace=True)

    # Calculate rightBase
    df['rightBase'] = df.apply(
        lambda row: _get_right_base(
            candidateSignal=data,
            blinkVelocity=blinkVelocity,
            rightOuter=row['outerEnds'],
            maxNegVelFrame=row['maxNegVelFrame']
        ),
        axis=1
    )

    return df

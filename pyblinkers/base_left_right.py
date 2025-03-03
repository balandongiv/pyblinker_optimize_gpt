# LLMed on 15 January 2025

import numpy as np
from pyblinkers.zero_crossing import (_maxPosVelFrame, _get_left_base, _get_right_base)

def create_left_right_base_vislab(data, df):
    """
    Computes leftBase and rightBase for each row in df, using blinkVelocity
    derived from data. Additional columns maxPosVelFrame and maxNegVelFrame
    are also added to df.

    Parameters
    ----------
    data : numpy.ndarray
        A 1D array of signal data representing the blink component.
        The length of this array should match the number of rows in df.
    
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'maxFrame' (int): The maximum frame index for the blink event.
        - 'leftZero' (int): The index of the left zero crossing.
        - 'rightZero' (int): The index of the right zero crossing.
        - 'outerStarts' (int): The starting index of the outer blink event.
        - 'outerEnds' (int): The ending index of the outer blink event.
        Additional columns may be present but are not used in this function.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the following additional columns:
        - 'maxPosVelFrame' (int): The frame index of the maximum positive velocity.
        - 'maxNegVelFrame' (int): The frame index of the maximum negative velocity.
        - 'leftBase' (float): The calculated left base value for the blink event.
        - 'rightBase' (float): The calculated right base value for the blink event.
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

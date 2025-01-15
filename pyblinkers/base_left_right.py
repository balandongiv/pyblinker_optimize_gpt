# LLMed on 15 January 2025

import numpy as np
from pyblinkers.zero_crossing import (_maxPosVelFrame, _get_left_base, _get_right_base)

def create_left_right_base_vislab(data, df):
    """
    Computes leftBase and rightBase for each row in df, using blinkVelocity
    derived from data. Additional columns maxPosVelFrame and maxNegVelFrame
    are also added to df.

    :param data: 1D numpy array of signal data
    :param df: Pandas DataFrame containing columns:
        ['maxFrame', 'leftZero', 'rightZero', 'outerStarts', 'outerEnds', ...]
    :return: Updated df with columns:
        ['maxPosVelFrame', 'maxNegVelFrame', 'leftBase', 'rightBase']
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

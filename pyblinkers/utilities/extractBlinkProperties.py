import logging

import numpy as np
import pandas as pd

from pyblinkers.utilities.misc import mad_matlab

logging.getLogger().setLevel(logging.INFO)


def _goodblink_based_corr_median_std(df, correlationThreshold):
    R2 = df['leftR2'] >= correlationThreshold
    R3 = df['rightR2'] >= correlationThreshold

    # Now calculate the cutoff ratios -- use default for the values
    good_data = df.loc[R2.values & R3.values, :]
    bestValues = good_data['maxValue'].array

    specifiedMedian = np.nanmedian(bestValues)
    specifiedStd = 1.4826 * mad_matlab(bestValues)

    return R2, R3, specifiedMedian, specifiedStd

#
# def get_mask_optimise(df, indicesNaN, correlationThreshold, zScoreThreshold):
#     """
#     "used Feb 02 2023"
#     The calculation of bestmedian,worst median, worrst rbobustst
#     is from https://github.com/VisLab/EEG-Blinks/blob/16b6ea04101ecfa74fb1c9cbceb037324572687e/blinker/utilities/extractBlinks.m#L97
#
#     :param df:
#     :param indicesNaN:
#     :param correlationThreshold:
#     :param zScoreThreshold:
#     :return:
#     """
#     R1 = ~indicesNaN
#     R2, R3, specifiedMedian, specifiedStd = _goodblink_based_corr_median_std(df, correlationThreshold)
#
#     R4 = df['maxValue'] >= max(0, specifiedMedian - zScoreThreshold * specifiedStd)
#     R5 = df['maxValue'] <= specifiedMedian + zScoreThreshold * specifiedStd
#     bool_test = R1.values & R2.values & R3.values & R4.values & R5.values
#
#     return bool_test, specifiedMedian, specifiedStd


def calculate_within_range(all_values, best_median, best_robust_std):
    """
    This function calculates the count of values within the specified range.

    Parameters:
        all_values (list or numpy array): Array of values to check.
        best_median (float): Median value.
        best_robust_std (float): Robust standard deviation.

    Returns:
        int: Count of values within the range.
    """
    # Define the range boundaries
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std

    # Apply logical condition to find values within the range
    within_range = (all_values <= upper_bound) & (all_values >= lower_bound)

    # Sum the logical array to count the values within the range
    result = np.sum(within_range)

    return result


def calculateGoodRatio(all_values, best_median, best_robust_std,all_x):
    """
    This function calculates the count of values within the specified range.

    Parameters:
        all_values (list or numpy array): Array of values to check.
        best_median (float): Median value.
        best_robust_std (float): Robust standard deviation.

    Returns:
        int: Count of values within the range.
    """
    # Define the range boundaries
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std

    # Apply logical condition to find values within the range
    within_range = (all_values <= upper_bound) & (all_values >= lower_bound)

    # Sum the logical array to count the values within the range
    result = np.sum(within_range)/all_x

    return result


def get_blink_statistic(df, zThresholds, signal=None):
    "used Feb 02 2023"
    # Now calculate the cutoff ratios -- use default for the values
    ## These is the default value
    # signal_length=len(signal)
    indices = np.arange(len(signal))
    dfx=df.copy()
    dfx[['leftZero', 'rightZero']] = dfx[['leftZero', 'rightZero']]-1
    # dfx=dfx.head(1)
    # Create the mask using NumPy broadcasting and logical operations
    blinkMask = np.any(
        [(indices >= left) & (indices <= right) for left, right in zip(dfx["leftZero"], dfx["rightZero"])],
        axis=0
    ).astype(int)

    # Create masks for inside and outside the blink intervals
    outsideBlink = (signal > 0) & ~blinkMask

    # Count the number of True and False values
    # count_true_outsideBlink  = np.sum(outsideBlink)
    # count_false_outsideBlink  = len(outsideBlink ) - count_true_outsideBlink

    insideBlink = (signal > 0) & blinkMask
    # count_true_insideBlink  = np.sum(insideBlink)

    # Calculate the blink amplitude ratio
    blinkAmpRatio = np.mean(signal[insideBlink.astype(bool)]) / np.mean(signal[outsideBlink.astype(bool)])

    correlationThresholdBottom, correlationThresholdTop = zThresholds[0]
    zScoreThreshold_s1, zScoreThreshold_s2 = zThresholds[1]
    # correlationThresholdBottom, correlationThresholdTop, zScoreThreshold_s1, zScoreThreshold_s2 = zThresholds


    df_data = df[['leftR2', 'rightR2', 'maxValue']]


    # Compute goodMaskTop
    goodMaskTop = (df_data['leftR2'] >= correlationThresholdTop) & (df_data['rightR2'] >= correlationThresholdTop)

    # Compute goodMaskBottom
    goodMaskBottom = (df_data['leftR2'] >= correlationThresholdBottom) & (df_data['rightR2'] >= correlationThresholdBottom)

    # Extract the values
    bestValues = df_data.loc[goodMaskTop, 'maxValue'].values  # Values from goodMaskTop
    worstValues = df_data.loc[~goodMaskBottom, 'maxValue'].tolist()  # Values from NOT goodMaskBottom
    goodValues = df_data.loc[goodMaskBottom, 'maxValue'].tolist()  # Values from goodMaskBottom

    bestMedian = np.nanmedian(bestValues)
    bestRobustStd = 1.4826 * mad_matlab(bestValues)
    worstMedian = np.nanmedian(worstValues);
    worstRobustStd = 1.4826*mad_matlab(worstValues)


    cutoff = (bestMedian*worstRobustStd + worstMedian*bestRobustStd)/(bestRobustStd + worstRobustStd)


    all_x=calculate_within_range(df_data['maxValue'].values, bestMedian, bestRobustStd )

    if all_x>0:
        goodRatio=calculateGoodRatio(goodValues, bestMedian, bestRobustStd,all_x)
    else:
        goodRatio=np.nan
    numberGoodBlinks=np.sum(goodMaskBottom)

    final_output=dict(numberBlinks=len(df_data),
                      numberGoodBlinks=numberGoodBlinks,
                      blinkAmpRatio=blinkAmpRatio,
                      cutoff=cutoff, bestMedian=bestMedian, bestRobustStd=bestRobustStd, goodRatio=goodRatio)

    return final_output


def getGoodBlinkMask(blinkFits, specifiedMedian, specifiedStd, zThresholds):
    """
    Compute a mask for good blinks based on correlation thresholds and z-score thresholds.

    Parameters:
    - blinkFits: pandas DataFrame with columns 'leftR2', 'rightR2', 'maxValue'.
    - specifiedMedian: median value for 'maxValue'.
    - specifiedStd: standard deviation for 'maxValue'.
    - zThresholds: list or array of [correlationThreshold, zScoreThreshold] pairs.

    Returns:
    - goodBlinkMask: NumPy array of booleans indicating good blinks.
    """
    # ff=zThresholds.ndim
    # Drop rows with NaN values in 'leftR2', 'rightR2', and 'maxValue'
    blinkFits = blinkFits.dropna(subset=['leftR2', 'rightR2', 'maxValue'])

    # Extract relevant columns as NumPy arrays
    leftR2 = blinkFits['leftR2'].to_numpy()
    rightR2 = blinkFits['rightR2'].to_numpy()
    maxValue = blinkFits['maxValue'].to_numpy()

    #  correlationThresholds =[0.9, 0.98] zScoreThresholds [2,5]
    # Convert zThresholds to a NumPy array
    # zThresholds = np.array(zThresholds)  # Shape: (n_thresholds, 2)
    correlationThresholds = zThresholds[0]  # Shape: (n_thresholds,)
    zScoreThresholds = zThresholds[1]       # Shape: (n_thresholds,)

    # Compute lower and upper bounds for 'maxValue' based on zScoreThresholds
    lowerBounds = np.maximum(0, specifiedMedian - zScoreThresholds * specifiedStd)
    upperBounds = specifiedMedian + zScoreThresholds * specifiedStd

    # Expand dimensions to enable broadcasting
    leftR2 = leftR2[:, np.newaxis]              # Shape: (n_rows, 1)
    rightR2 = rightR2[:, np.newaxis]            # Shape: (n_rows, 1)
    maxValue = maxValue[:, np.newaxis]          # Shape: (n_rows, 1)
    correlationThresholds = correlationThresholds[np.newaxis, :]  # Shape: (1, n_thresholds)
    lowerBounds = lowerBounds[np.newaxis, :]    # Shape: (1, n_thresholds)
    upperBounds = upperBounds[np.newaxis, :]    # Shape: (1, n_thresholds)

    # Compute the mask for all thresholds simultaneously using broadcasting
    masks = (
            (leftR2 >= correlationThresholds) &
            (rightR2 >= correlationThresholds) &
            (maxValue >= lowerBounds) &
            (maxValue <= upperBounds)
    )  # Shape: (n_rows, n_thresholds)

    # Combine masks across all thresholds using logical OR
    goodBlinkMask = np.any(masks, axis=1)  # Shape: (n_rows,)
    selected_rows = blinkFits[goodBlinkMask]

    return goodBlinkMask,selected_rows




class BlinkProperties:
    '''
    Return a structure with blink shapes and properties for individual blinks
    '''
    def __init__(self, data, df, srate,params):
        self.data = data
        self.df = df
        self.srate = srate
        self.shutAmpFraction = params['shutAmpFraction']
        self.pAVRThreshold = params['shutAmpFraction']
        self.zThresholds =  params['zThresholds']


        self.df_res=[]
        self.reset_index()
        logging.warning("The error will start from here")
        # self.blinkStatProperties = get_blink_statistic(self.df, self.zThresholds, signal=self.data)



        # self.get_good_blink_mask()
        self.set_blink_velocity()

        # Blink durations
        self.set_blink_duration()

        # Blink amplitude-velocity ratio from zero to max
        self.set_blink_amp_velocity_ratio_zero_to_max()
        self.amplitude_velocity_ratio_base()
        self.amplitude_velocity_ratio_tent()
        self.time_zero_shut()
        self.time_base_shut()
        self.extract_other_times()



    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)



    def set_blink_velocity(self):
        self.signal_l = self.data.shape[0]
        self.blinkVelocity = np.diff(self.data)

    def set_blink_duration(self):

        constant=  1 # The constant 1 is added as what is in the matlab code
        self.df['durationBase'] = (self.df['rightBase'] - self.df['leftBase']) / self.srate
        self.df['durationZero'] = (self.df['rightZero'] - self.df['leftZero']) / self.srate
        self.df['durationTent'] = (self.df['rightXIntercept'] - self.df['leftXIntercept']) / self.srate
        self.df['durationHalfBase'] = (self.df['rightBaseHalfHeight'] - self.df['leftBaseHalfHeight'] + constant) / self.srate
        self.df['durationHalfZero'] = (self.df['rightZeroHalfHeight'] - self.df['leftZeroHalfHeight'] + constant) / self.srate


    def compute_negAmpVelRatioZero(self):
        """
        Compute and assign the 'negAmpVelRatioZero' for all rows in self.df.
        Implementaton to calculate the first row

        # nrow=0 # The first row
        # downstroke = np.arange(self.df['maxFrame'][nrow],self.df['rightZero'][nrow] + 1)
        # velFrame=np.argmin(self.blinkVelocity[downstroke])
        # velFrame = velFrame +downstroke[0]    # downstroke[0] is the first value of the downstroke
        # negAmpVelRatioZero = 100 * abs(self.data[self.df['maxFrame'][nrow]]  /self.blinkVelocity[velFrame]) / self.srate
        """

        multiplication_constant=100
        # Calculate the lengths of downstroke ranges for all rows
        lengths = self.df['rightZero'] - self.df['maxFrame'] + 1
        max_length = lengths.max()

        # Create an array of offsets for the maximum possible downstroke length
        offsets = np.arange(max_length)

        # Expand offsets to a 2D array and mask values beyond actual lengths
        downstroke_offsets = offsets[np.newaxis, :]
        mask = downstroke_offsets < lengths.to_numpy()[:, np.newaxis]

        # Calculate downstroke indices for all rows
        downstroke_all = self.df['maxFrame'].to_numpy()[:, np.newaxis] + downstroke_offsets
        downstroke_all = downstroke_all[mask]

        # Create an array of row indices corresponding to downstroke indices
        lengths=lengths.astype(int)
        row_idx_all = np.repeat(np.arange(len(lengths)), lengths.to_numpy())

        # Extract velocities at downstroke indices
        downstroke_all=downstroke_all.astype(int)
        velocities = self.blinkVelocity[downstroke_all]

        # Create a DataFrame to find the minimum velocities and their indices per row
        df_downstroke = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': downstroke_all
        })

        # Find the indices of the minimum velocities for each row
        idx_min = df_downstroke.groupby('row_idx')['velocity'].idxmin()
        df_min = df_downstroke.loc[idx_min].sort_values('row_idx')

        # Compute negAmpVelRatioZero for all rows
        negAmpVelRatioZero = multiplication_constant * abs(
            self.data[self.df['maxFrame'].to_numpy()] / df_min['velocity'].to_numpy()
        ) / self.srate

        # Assign the computed values to the DataFrame
        self.df['negAmpVelRatioZero'] = negAmpVelRatioZero

    def compute_posAmpVelRatioZero(self, multiplication_constant=100):
        """
        Compute and assign the 'posAmpVelRatioZero' for all rows in self.df.
        Implementaton to calculate the first row

        # nrow=0 # The first row
        # upStroke = np.arange(self.df['leftZero'][nrow],self.df['maxFrame'][nrow] + 1)
        # velFrame=np.argmax(self.blinkVelocity[upStroke])
        # velFrame = velFrame +upStroke[0]    # downstroke[0] is the first value of the downstroke
        # posAmpVelRatioZero  = 100 * abs(self.data[self.df['maxFrame'][nrow]]  /self.blinkVelocity[velFrame]) / self.srate
        """

        # Calculate the lengths of upstroke ranges for all rows

        lengths = self.df['maxFrame'] - self.df['leftZero'] + 1
        max_length = lengths.max()

        # Create an array of offsets for the maximum possible upstroke length
        offsets = np.arange(max_length)

        # Expand offsets to a 2D array and mask values beyond actual lengths
        upstroke_offsets = offsets[np.newaxis, :]
        mask = upstroke_offsets < lengths.to_numpy()[:, np.newaxis]

        # Calculate upstroke indices for all rows
        upstroke_all = self.df['leftZero'].to_numpy()[:, np.newaxis] + upstroke_offsets
        upstroke_all = upstroke_all[mask]
        lengths=lengths.astype(int)
        # Create an array of row indices corresponding to upstroke indices
        row_idx_all = np.repeat(np.arange(len(lengths)), lengths.to_numpy())

        # Extract velocities at upstroke indices
        upstroke_all=upstroke_all.astype(int)
        velocities = self.blinkVelocity[upstroke_all]

        # Create a DataFrame to find the maximum velocities and their indices per row
        df_upstroke = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': upstroke_all
        })

        # Find the indices of the maximum velocities for each row
        idx_max = df_upstroke.groupby('row_idx')['velocity'].idxmax()
        df_max = df_upstroke.loc[idx_max].sort_values('row_idx')

        # Assign velFrame to peaksPosVelBase
        self.df['peaksPosVelZero'] = df_max['index'].to_numpy()

        # Compute posAmpVelRatioZero for all rows
        posAmpVelRatioZero = multiplication_constant * abs(
            self.data[self.df['maxFrame'].to_numpy()] / df_max['velocity'].to_numpy()
        ) / self.srate

        # Assign the computed values to the DataFrame
        self.df['posAmpVelRatioZero'] = posAmpVelRatioZero


    def set_blink_amp_velocity_ratio_zero_to_max(self):
        '''
        Blink amplitude-velocity ratio from zero to max
        :return:
        '''

        self.compute_posAmpVelRatioZero()
        self.compute_negAmpVelRatioZero()


    def compute_posAmpVelRatioBase(self, multiplication_constant=100):
        """
        Compute and assign 'posAmpVelRatioBase' for all rows in self.df,
        and assign velFrame as 'peaksPosVelBase'.
        """
        # import numpy as np
        # import pandas as pd

        # Calculate the lengths of upstroke ranges for all rows
        lengths = self.df['maxFrame'] - self.df['leftBase'] + 1
        max_length = lengths.max()

        # Create an array of offsets for the maximum possible upstroke length
        offsets = np.arange(max_length)

        # Expand offsets to a 2D array and mask values beyond actual lengths
        upstroke_offsets = offsets[np.newaxis, :]
        mask = upstroke_offsets < lengths.to_numpy()[:, np.newaxis]

        # Calculate upstroke indices for all rows
        upstroke_all = self.df['leftBase'].to_numpy()[:, np.newaxis] + upstroke_offsets
        upstroke_all = upstroke_all[mask]

        # Create an array of row indices corresponding to upstroke indices
        row_idx_all = np.repeat(np.arange(len(lengths)), lengths.to_numpy())

        # Extract velocities at upstroke indices
        velocities = self.blinkVelocity[upstroke_all]

        # Create a DataFrame to find the maximum velocities and their indices per row
        df_upstroke = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': upstroke_all
        })

        # Find the indices of the maximum velocities for each row
        idx_max = df_upstroke.groupby('row_idx')['velocity'].idxmax()
        df_max = df_upstroke.loc[idx_max].sort_values('row_idx')

        # Assign velFrame to peaksPosVelBase
        self.df['peaksPosVelBase'] = df_max['index'].to_numpy()

        # Compute posAmpVelRatioBase for all rows
        posAmpVelRatioBase = multiplication_constant * abs(
            self.data[self.df['maxFrame'].to_numpy()] / df_max['velocity'].to_numpy()
        ) / self.srate

        # Assign the computed values to the DataFrame
        self.df['posAmpVelRatioBase'] = posAmpVelRatioBase

    def compute_posAmpVelRatioBase_X(self, multiplication_constant=100):
        """
        Compute and assign the 'posAmpVelRatioBase' for all rows in self.df.

        multiplication_constant=100
        nrow=0 # The first row
        upStroke = np.arange(self.df['leftBase'][nrow],self.df['maxFrame'][nrow] + 1)
        velFrame=np.argmax(self.blinkVelocity[upStroke])
        velFrame = velFrame +upStroke[0]    # downstroke[0] is the first value of the downstroke
        posAmpVelRatioBase  = multiplication_constant * abs(self.data[self.df['maxFrame'][nrow]]  /self.blinkVelocity[velFrame]) / self.srate
        """


        # Calculate the lengths of upstroke ranges for all rows
        lengths = self.df['maxFrame'] - self.df['leftBase'] + 1
        max_length = lengths.max()

        # Create an array of offsets for the maximum possible upstroke length
        offsets = np.arange(max_length)

        # Expand offsets to a 2D array and mask values beyond actual lengths
        upstroke_offsets = offsets[np.newaxis, :]
        mask = upstroke_offsets < lengths.to_numpy()[:, np.newaxis]

        # Calculate upstroke indices for all rows
        upstroke_all = self.df['leftBase'].to_numpy()[:, np.newaxis] + upstroke_offsets
        upstroke_all = upstroke_all[mask]

        # Create an array of row indices corresponding to upstroke indices
        row_idx_all = np.repeat(np.arange(len(lengths)), lengths.to_numpy())

        # Extract velocities at upstroke indices
        velocities = self.blinkVelocity[upstroke_all]

        # Create a DataFrame to find the maximum velocities and their indices per row
        df_upstroke = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': upstroke_all
        })

        # Find the indices of the maximum velocities for each row
        idx_max = df_upstroke.groupby('row_idx')['velocity'].idxmax()
        df_max = df_upstroke.loc[idx_max].sort_values('row_idx')

        # Compute posAmpVelRatioBase for all rows
        posAmpVelRatioBase = multiplication_constant * abs(
            self.data[self.df['maxFrame'].to_numpy()] / df_max['velocity'].to_numpy()
        ) / self.srate

        # Assign the computed values to the DataFrame
        self.df['posAmpVelRatioBase'] = posAmpVelRatioBase


    def compute_negAmpVelRatioBase(self, multiplication_constant=100):
        """
        Compute and assign the 'negAmpVelRatioBase' for all rows in self.df.
        """


        # Calculate the lengths of downstroke ranges for all rows
        lengths = self.df['rightBase'] - self.df['maxFrame'] + 1
        max_length = lengths.max()

        # Create an array of offsets for the maximum possible downstroke length
        offsets = np.arange(max_length)

        # Expand offsets to a 2D array and mask values beyond actual lengths
        downstroke_offsets = offsets[np.newaxis, :]
        mask = downstroke_offsets < lengths.to_numpy()[:, np.newaxis]

        # Calculate downstroke indices for all rows
        downstroke_all = self.df['maxFrame'].to_numpy()[:, np.newaxis] + downstroke_offsets
        downstroke_all = downstroke_all[mask]

        # Convert lengths to integer before using it in np.repeat
        lengths_int = lengths.astype(int)


        # Create an array of row indices corresponding to downstroke indices
        row_idx_all = np.repeat(np.arange(len(lengths_int)), lengths_int.to_numpy())

        # Extract velocities at downstroke indices
        downstroke_all = downstroke_all.astype(int)
        velocities = self.blinkVelocity[downstroke_all]

        # Create a DataFrame to find the minimum velocities and their indices per row
        df_downstroke = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': downstroke_all
        })

        # Find the indices of the minimum velocities for each row
        idx_min = df_downstroke.groupby('row_idx')['velocity'].idxmin()
        df_min = df_downstroke.loc[idx_min].sort_values('row_idx')

        # Compute negAmpVelRatioBase for all rows
        negAmpVelRatioBase = multiplication_constant * abs(
            self.data[self.df['maxFrame'].to_numpy()] / df_min['velocity'].to_numpy()
        ) / self.srate

        # Assign the computed values to the DataFrame
        self.df['negAmpVelRatioBase'] = negAmpVelRatioBase


    def amplitude_velocity_ratio_base(self):

        '''
        Blink amplitude-velocity ratio from base to max

        '''
        # Here we will extract also the peaksPosVelBase(k) = velFrame from compute_posAmpVelRatioBase()
        self.compute_posAmpVelRatioBase()
        self.compute_negAmpVelRatioBase()


    def amplitude_velocity_ratio_tent(self):
        '''
        Blink amplitude-velocity ratio estimated from tent slope
        :return:
        '''


        # negAmpVelRatioTent Included
        self.df['negAmpVelRatioTent'] = (100 * abs(self.data[self.df['maxFrame']] / self.df['averRightVelocity'])) / self.srate

        # posAmpVelRatioTent Included
        self.df['posAmpVelRatioTent'] = (100 * abs(self.data[self.df['maxFrame']] / self.df['averLeftVelocity'])) / self.srate

    @staticmethod
    def compute_time_shut_zero(row, data, srate, shut_amp_fraction):
        matlab_index_offset = 1 #The offset is added to match the value as in the matlab code
        left_zero = int(row['leftZero'])
        right_zero = int(row['rightZero'])
        max_value = row['maxValue']
        amp_threshold = shut_amp_fraction * max_value
        data_slice = data[left_zero:right_zero + 1]

        # Find start_shut_tzs
        condition_start = data_slice >= amp_threshold
        if condition_start.any():
            start_shut = np.argmax(condition_start)
        else:
            return np.nan  # or np.nan if you prefer

        # Find end_shut_tzs
        data_slice_after_start = data_slice[start_shut + 1:]
        condition_end = data_slice_after_start < amp_threshold
        if condition_end.any():
            end_shut = np.argmax(condition_end) +matlab_index_offset
        else:
            end_shut = np.nan

        # Calculate timeShutZero
        time_shut_zero = end_shut / srate
        return time_shut_zero

    @staticmethod
    def compute_time_shut_base(row, data, srate, shut_amp_fraction):
        matlab_index_offset = 1 #The offset is added to match the value as in the matlab code
        left_base = int(row['leftBase'])
        right_base = int(row['rightBase'])
        max_value = row['maxValue']
        amp_threshold = shut_amp_fraction * max_value
        data_slice = data[left_base:right_base + 1]

        # Find start_shut_tbs
        condition_start = data_slice >= amp_threshold
        if condition_start.any():
            start_shut_tbs = np.argmax(condition_start)
        else:
            return 0  # or np.nan if you prefer

        # Find endShut_tbs
        data_slice_after_start = data_slice[start_shut_tbs+1:]
        condition_end = data_slice_after_start < amp_threshold
        if condition_end.any():
            end_shut_tbs =  np.argmax(condition_end)+matlab_index_offset
        else:
            end_shut_tbs = np.nan

        # Calculate timeShutBase
        time_shut_base = end_shut_tbs / srate
        return time_shut_base

    def time_zero_shut(self):
        '''
        Time zero shut
        :return:
        '''

        # closingTimeZero and reopeningTimeZero included
        self.df['closingTimeZero'] = (self.df['maxFrame'] - self.df['leftZero']) / self.srate
        self.df['reopeningTimeZero'] = (self.df['rightZero'] - self.df['maxFrame']) / self.srate


        # Calculation of time shut zero by applying the function to each row in the DataFrame
        self.df['timeShutZero'] = self.df.apply(
            lambda row: self.compute_time_shut_zero(row, self.data, self.srate, self.shutAmpFraction), axis=1)

    @staticmethod
    def compute_time_shut_tent(row, data, srate, shut_amp_fraction):
        left = int(row['leftXIntercept'])
        right = int(row['rightXIntercept']) + 1  # Include the right endpoint
        # if left >= right:
        #     return np.nan  # Handle invalid ranges

        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val
        data_slice = data[left:right]

        # Find start_shut_tst
        condition_start = data_slice >= amp_threshold
        if condition_start.any():
            start_shut_tst = np.argmax(condition_start)
        else:
            return 0  # or np.nan if you prefer

        # Find endShut_tst
        subset = data_slice[start_shut_tst:-1]
        condition_end = subset < amp_threshold
        if condition_end.any():
            end_shut_tst =  np.argmax(condition_end)
        else:
            end_shut_tst = np.nan

        # Calculate timeShutTent
        time_shut_tent = end_shut_tst / srate
        return time_shut_tent

    def time_base_shut(self):

        self.df['timeShutBase'] = self.df.apply(
            lambda row: self.compute_time_shut_base(row, self.data, self.srate, self.shutAmpFraction), axis=1)




        ## Time shut tent

        # closingTimeTent and reopeningTimeTent included
        self.df['closingTimeTent'] = (self.df['xIntersect'] - self.df['leftXIntercept']) / self.srate
        self.df['reopeningTimeTent'] = (self.df['rightXIntercept'] - self.df['xIntersect']) / self.srate

        # Calculation for timeShutTent
        # Apply the function to compute 'timeShutTent'
        self.df['timeShutTent'] = self.df.apply(
            lambda row: self.compute_time_shut_tent(row, self.data, self.srate, self.shutAmpFraction), axis=1)

        # self.df['ampThreshhold_tst'] = self.shutAmpFraction * self.df['maxValue']
        #
        # self.df[['leftXIntercept_int', 'rightXIntercept_int']] = self.df[['leftXIntercept', 'rightXIntercept']].astype(
        #     int)
        #
        #
        # self.df=self.df[self.df.leftXIntercept_int<self.df.rightXIntercept_int]
        # self.df.reset_index(drop=True,inplace=True)
        # self.df['start_shut_tst'] = self.df.apply(
        #     lambda x: np.argmax(self.data[x['leftXIntercept_int']:x['rightXIntercept_int'] + 1] >= x['ampThreshhold']), axis=1)
        #
        #
        # self.df['endShut_tst'] = self.df.apply(self.get_argmax_val,axis=1)
        #
        # # timeShutTent included
        # self.df['timeShutTent'] = self.df.apply(
        #     lambda x: 0 if x['endShut_tst'] == np.isnan else (x['endShut_tst'] / self.srate), axis=1)

    def get_argmax_val(self,row):
        left = row['leftXIntercept_int']
        right = row['rightXIntercept_int'] + 1
        start = row['start_shut_tst']
        max_val = row['maxValue']
        shut_amp_frac = self.shutAmpFraction

        subset = self.data[left:right][start:-1]
        dconstant=shut_amp_frac * max_val

        try:
            return np.argmax(subset<dconstant)
        except ValueError:
            return np.nan
    def extract_other_times(self):
        ## Other times

        # peakMaxBlink, peakMaxTent, peakTimeTent, peakTimeBlink included
        self.df['peakMaxBlink'] = self.df['maxValue']
        self.df['peakMaxTent'] = self.df['yIntersect']
        self.df['peakTimeTent'] = self.df['xIntersect'] / self.srate
        self.df['peakTimeBlink'] = self.df['maxFrame'] / self.srate



        # Append the length of self.data to the peaks array
        peaks_with_len = np.append(self.df['maxFrame'].to_numpy(), len(self.data))
        self.df['interBlinkMaxAmp'] = np.diff(peaks_with_len) / self.srate   # Normalize by sampling rate


        # Here we multiply by -1 to get the same -ve value as in the matlab code
        self.df['interBlinkMaxVelBase']=(self.df['peaksPosVelBase']*-1)/self.srate

        self.df['interBlinkMaxVelZero']=(self.df['peaksPosVelZero']*-1)/self.srate
        # existing_array = self.df['maxFrame'].to_numpy()
        # range_to_append = np.arange(len(self.data))
        #
        # # Append the range to the existing array
        # combined_array = np.append(existing_array, range_to_append)
        #
        # peaks=self.df['maxFrame'].to_numpy(),len(self.data)
        # # Calculation for interBlinkMaxVelBase and interBlinkMaxVelZero
        # dfcal = self.df[['maxFrame', 'peaksPosVelBase', 'peaksPosVelZero']]
        #
        # df_t = pd.DataFrame.from_records([[self.signal_l] * 3], columns=['maxFrame', 'peaksPosVelBase', 'peaksPosVelZero'])
        #
        # dfcal = pd.concat([dfcal, df_t]).reset_index(drop=True)
        #
        # dfcal['ibmx'] = dfcal.maxFrame.diff().shift(-1)
        #
        # # -->>interBlinkMaxAmp included
        # dfcal['interBlinkMaxAmp'] = dfcal['ibmx'] / self.srate
        #
        # dfcal['ibmvb'] = 1 - dfcal['peaksPosVelBase']
        #
        # # -->> interBlinkMaxVelBase included
        # dfcal['interBlinkMaxVelBase'] = dfcal['ibmvb'] / self.srate  # peaksPosVelBase == velFrame
        #
        # dfcal['ibmvz'] = 1 - dfcal['peaksPosVelZero']
        #
        # # interBlinkMaxVelZero included
        # dfcal['interBlinkMaxVelZero'] = dfcal['ibmvz'] / self.srate
        #
    # def applyPAVRRestriction(self):
    #     # Define the conditions
    #     condition_1 = self.df['posAmpVelRatioZero'] < self.pAVRThreshold
    #     condition_2 = self.df['maxValue'] < (self.blinkStatProperties['bestMedian'] - self.blinkStatProperties['bestRobustStd'])
    #
    #     # filters the DataFrame to keep only rows that do not meet both conditions.
    #     self.df= self.df[~(condition_1 & condition_2)]




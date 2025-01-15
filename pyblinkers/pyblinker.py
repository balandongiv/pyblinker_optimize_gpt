
# LLMed on 15 Jan 2025
import logging
import numpy as np
import pandas as pd
from default_setting import scalingFactor
from pyblinkers.misc import mad_matlab

logging.getLogger().setLevel(logging.INFO)

def _goodblink_based_corr_median_std(df, correlationThreshold):
    """
    Internal function.
    Computes which rows exceed the correlationThreshold in both leftR2 and rightR2.
    Then calculates median and std from 'maxValue' of those rows.
    Returns a tuple (R2, R3, specifiedMedian, specifiedStd).
    """
    left_r2 = df['leftR2']
    right_r2 = df['rightR2']

    R2 = (left_r2 >= correlationThreshold)
    R3 = (right_r2 >= correlationThreshold)

    good_data = df.loc[R2 & R3, :]
    best_values = good_data['maxValue'].to_numpy()

    specifiedMedian = np.nanmedian(best_values)
    specifiedStd = 1.4826 * mad_matlab(best_values)
    return R2, R3, specifiedMedian, specifiedStd


def calculate_within_range(all_values, best_median, best_robust_std):
    """
    This function calculates the count of values within the specified range.
    """
    arr = np.asarray(all_values)
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    mask = (arr >= lower_bound) & (arr <= upper_bound)
    return np.sum(mask)


def calculateGoodRatio(all_values, best_median, best_robust_std, all_x):
    """
    This function calculates the count of values within the specified range.
    """
    arr = np.asarray(all_values)
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    mask = (arr >= lower_bound) & (arr <= upper_bound)
    return np.sum(mask) / all_x


def get_blink_statistic(df, zThresholds, signal=None):
    "used Feb 02 2023"
    dfx = df.copy()
    dfx[['leftZero', 'rightZero']] = dfx[['leftZero', 'rightZero']] - 1

    indices = np.arange(len(signal))
    blinkMask = np.any(
        [(indices >= lz) & (indices <= rz) for lz, rz in zip(dfx["leftZero"], dfx["rightZero"])],
        axis=0
    ).astype(bool)

    outsideBlink = (signal > 0) & ~blinkMask
    insideBlink = (signal > 0) & blinkMask
    blinkAmpRatio = np.mean(signal[insideBlink]) / np.mean(signal[outsideBlink])

    correlationThresholdBottom, correlationThresholdTop = zThresholds[0]
    zScoreThreshold_s1, zScoreThreshold_s2 = zThresholds[1]

    df_data = df[['leftR2', 'rightR2', 'maxValue']]
    goodMaskTop = (df_data['leftR2'] >= correlationThresholdTop) & (df_data['rightR2'] >= correlationThresholdTop)
    goodMaskBottom = (df_data['leftR2'] >= correlationThresholdBottom) & (df_data['rightR2'] >= correlationThresholdBottom)

    bestValues = df_data.loc[goodMaskTop, 'maxValue'].to_numpy()
    worstValues = df_data.loc[~goodMaskBottom, 'maxValue'].to_numpy()
    goodValues = df_data.loc[goodMaskBottom, 'maxValue'].to_numpy()

    bestMedian = np.nanmedian(bestValues)

    bestRobustStd = scalingFactor * mad_matlab(bestValues)
    worstMedian = np.nanmedian(worstValues)
    worstRobustStd = scalingFactor * mad_matlab(worstValues)

    cutoff = (bestMedian * worstRobustStd + worstMedian * bestRobustStd) / (bestRobustStd + worstRobustStd)
    all_x = calculate_within_range(df_data['maxValue'].to_numpy(), bestMedian, bestRobustStd)

    if all_x > 0:
        goodRatio = calculateGoodRatio(goodValues, bestMedian, bestRobustStd, all_x)
    else:
        goodRatio = np.nan

    numberGoodBlinks = np.sum(goodMaskBottom)

    final_output = dict(
        numberBlinks=len(df_data),
        numberGoodBlinks=numberGoodBlinks,
        blinkAmpRatio=blinkAmpRatio,
        cutoff=cutoff,
        bestMedian=bestMedian,
        bestRobustStd=bestRobustStd,
        goodRatio=goodRatio
    )
    return final_output


def getGoodBlinkMask(blinkFits, specifiedMedian, specifiedStd, zThresholds):
    """
    Compute a mask for good blinks based on correlation thresholds and z-score thresholds.
    """
    blinkFits = blinkFits.dropna(subset=['leftR2', 'rightR2', 'maxValue'])

    leftR2 = blinkFits['leftR2'].to_numpy()
    rightR2 = blinkFits['rightR2'].to_numpy()
    maxValue = blinkFits['maxValue'].to_numpy()

    correlationThresholds = zThresholds[0]
    zScoreThresholds = zThresholds[1]

    lowerBounds = specifiedMedian - zScoreThresholds * specifiedStd
    upperBounds = specifiedMedian + zScoreThresholds * specifiedStd

    leftR2 = leftR2[:, np.newaxis]
    rightR2 = rightR2[:, np.newaxis]
    maxValue = maxValue[:, np.newaxis]
    correlationThresholds = correlationThresholds[np.newaxis, :]
    lowerBounds = lowerBounds[np.newaxis, :]
    upperBounds = upperBounds[np.newaxis, :]

    masks = (
            (leftR2 >= correlationThresholds) &
            (rightR2 >= correlationThresholds) &
            (maxValue >= lowerBounds) &
            (maxValue <= upperBounds)
    )
    goodBlinkMask = np.any(masks, axis=1)
    selected_rows = blinkFits[goodBlinkMask]
    return goodBlinkMask, selected_rows


class BlinkProperties:
    '''
    Return a structure with blink shapes and properties for individual blinks
    '''
    def __init__(self, data, df, srate, params):
        self.data = data
        self.df = df
        self.srate = srate
        self.shutAmpFraction = params['shutAmpFraction']
        self.pAVRThreshold = params['shutAmpFraction']
        self.zThresholds = params['zThresholds']

        self.df_res = []
        self.reset_index()
        logging.warning("The error will start from here")

        self.set_blink_velocity()
        self.set_blink_duration()
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
        self.blinkVelocity = np.diff(self.data, axis=0)

    def set_blink_duration(self):
        constant = 1
        self.df['durationBase'] = (self.df['rightBase'] - self.df['leftBase']) / self.srate
        self.df['durationZero'] = (self.df['rightZero'] - self.df['leftZero']) / self.srate
        self.df['durationTent'] = (self.df['rightXIntercept'] - self.df['leftXIntercept']) / self.srate
        self.df['durationHalfBase'] = ((self.df['rightBaseHalfHeight'] - self.df['leftBaseHalfHeight']) + constant) / self.srate
        self.df['durationHalfZero'] = ((self.df['rightZeroHalfHeight'] - self.df['leftZeroHalfHeight']) + constant) / self.srate

    # ------------------------------------------------------------------------
    # Internal helper to unify pos/neg computations for amplitude-velocity ratios
    # ------------------------------------------------------------------------
    def _compute_amp_vel_ratio(self, start_col, end_col, ratio_col, aggregator='max',
                               velocity_idx_col=None, multiplier=100):
        """
        Internal helper to compute amplitude-velocity ratio between start_col and end_col.
        aggregator = 'max' or 'min' to pick the velocity extremes.
        velocity_idx_col is an optional column name to store the indices of those extremes.
        """
        start_vals = self.df[start_col].to_numpy().astype(int)
        end_vals = self.df[end_col].to_numpy().astype(int)
        blinkVel = self.blinkVelocity  # local ref for clarity

        lengths = (end_vals - start_vals + 1)
        max_len = lengths.max()
        offsets = np.arange(max_len)[np.newaxis, :]
        mask = offsets < lengths[:, np.newaxis]

        all_indices = start_vals[:, np.newaxis] + offsets
        all_indices = all_indices[mask].astype(int)

        row_idx_all = np.repeat(np.arange(len(lengths)), lengths)
        velocities = blinkVel[all_indices]

        tmp_df = pd.DataFrame({
            'row_idx': row_idx_all,
            'velocity': velocities,
            'index': all_indices
        })

        if aggregator == 'max':
            idx_extreme = tmp_df.groupby('row_idx')['velocity'].idxmax()
        else:
            idx_extreme = tmp_df.groupby('row_idx')['velocity'].idxmin()

        df_extreme = tmp_df.loc[idx_extreme].sort_values('row_idx')
        ratio_values = (
                               multiplier *
                               abs(self.data[self.df['maxFrame'].to_numpy()] / df_extreme['velocity'].to_numpy())
                       ) / self.srate

        self.df[ratio_col] = ratio_values
        if velocity_idx_col:
            self.df[velocity_idx_col] = df_extreme['index'].to_numpy()

    # ---------------------------------------------------------------
    # The four ratio methods below now become small wrappers
    # ---------------------------------------------------------------
    def compute_negAmpVelRatioZero(self):
        """
        Compute and assign the 'negAmpVelRatioZero' for all rows in self.df.
        """
        self._compute_amp_vel_ratio(
            start_col='maxFrame',
            end_col='rightZero',
            ratio_col='negAmpVelRatioZero',
            aggregator='min'
        )

    def compute_posAmpVelRatioZero(self, multiplication_constant=100):
        """
        Compute and assign the 'posAmpVelRatioZero' for all rows in self.df.
        """
        self._compute_amp_vel_ratio(
            start_col='leftZero',
            end_col='maxFrame',
            ratio_col='posAmpVelRatioZero',
            aggregator='max',
            velocity_idx_col='peaksPosVelZero',
            multiplier=multiplication_constant
        )

    def set_blink_amp_velocity_ratio_zero_to_max(self):
        """
        Blink amplitude-velocity ratio from zero to max.
        """
        self.compute_posAmpVelRatioZero()
        self.compute_negAmpVelRatioZero()

    def compute_posAmpVelRatioBase(self, multiplication_constant=100):
        """
        Compute and assign 'posAmpVelRatioBase' for all rows in self.df,
        and assign velFrame as 'peaksPosVelBase'.
        """
        self._compute_amp_vel_ratio(
            start_col='leftBase',
            end_col='maxFrame',
            ratio_col='posAmpVelRatioBase',
            aggregator='max',
            velocity_idx_col='peaksPosVelBase',
            multiplier=multiplication_constant
        )

    def compute_posAmpVelRatioBase_X(self, multiplication_constant=100):
        """
        (Alternate version) Compute and assign the 'posAmpVelRatioBase' for all rows in self.df.
        """
        self._compute_amp_vel_ratio(
            start_col='leftBase',
            end_col='maxFrame',
            ratio_col='posAmpVelRatioBase',
            aggregator='max',
            multiplier=multiplication_constant
        )

    def compute_negAmpVelRatioBase(self, multiplication_constant=100):
        """
        Compute and assign the 'negAmpVelRatioBase' for all rows in self.df.
        """
        self._compute_amp_vel_ratio(
            start_col='maxFrame',
            end_col='rightBase',
            ratio_col='negAmpVelRatioBase',
            aggregator='min',
            multiplier=multiplication_constant
        )

    def amplitude_velocity_ratio_base(self):
        """
        Blink amplitude-velocity ratio from base to max.
        """
        self.compute_posAmpVelRatioBase()
        self.compute_negAmpVelRatioBase()

    def amplitude_velocity_ratio_tent(self):
        """
        Blink amplitude-velocity ratio estimated from tent slope.
        """
        self.df['negAmpVelRatioTent'] = (
                                                100 * abs(self.data[self.df['maxFrame']] / self.df['averRightVelocity'])
                                        ) / self.srate
        self.df['posAmpVelRatioTent'] = (
                                                100 * abs(self.data[self.df['maxFrame']] / self.df['averLeftVelocity'])
                                        ) / self.srate

    @staticmethod
    def compute_time_shut_zero(row, data, srate, shut_amp_fraction):
        """
        Compute the time (in seconds) for shutting from zero crossing.
        """
        left_zero = int(row['leftZero'])
        right_zero = int(row['rightZero'])
        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val

        data_slice = data[left_zero : right_zero + 1]
        cond_start = (data_slice >= amp_threshold)

        if cond_start.any():
            start_idx = np.argmax(cond_start)
        else:
            return np.nan

        cond_end = (data_slice[start_idx + 1:] < amp_threshold)
        if cond_end.any():
            end_shut = np.argmax(cond_end) + 1
        else:
            end_shut = np.nan

        return end_shut / srate

    @staticmethod
    def compute_time_shut_base(row, data, srate, shut_amp_fraction):
        """
        Compute the time (in seconds) for shutting from base.
        """
        left_base = int(row['leftBase'])
        right_base = int(row['rightBase'])
        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val

        data_slice = data[left_base : right_base + 1]
        cond_start = (data_slice >= amp_threshold)

        if cond_start.any():
            start_idx = np.argmax(cond_start)
        else:
            return 0

        cond_end = (data_slice[start_idx + 1:] < amp_threshold)
        if cond_end.any():
            end_shut = np.argmax(cond_end) + 1
        else:
            end_shut = np.nan

        return end_shut / srate

    def time_zero_shut(self):
        """
        Time zero shut
        """
        self.df['closingTimeZero'] = (self.df['maxFrame'] - self.df['leftZero']) / self.srate
        self.df['reopeningTimeZero'] = (self.df['rightZero'] - self.df['maxFrame']) / self.srate

        self.df['timeShutZero'] = self.df.apply(
            lambda row: self.compute_time_shut_zero(row, self.data, self.srate, self.shutAmpFraction), axis=1
        )

    @staticmethod
    def compute_time_shut_tent(row, data, srate, shut_amp_fraction):
        """
        Compute the time (in seconds) for shutting based on tent intercepts.
        """
        left = int(row['leftXIntercept'])
        right = int(row['rightXIntercept']) + 1
        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val

        data_slice = data[left : right]
        cond_start = (data_slice >= amp_threshold)
        if cond_start.any():
            start_idx = np.argmax(cond_start)
        else:
            return 0

        cond_end = (data_slice[start_idx:-1] < amp_threshold)
        if cond_end.any():
            end_shut = np.argmax(cond_end)
        else:
            end_shut = np.nan

        return end_shut / srate

    def time_base_shut(self):
        """
        Compute time shut based on base intercepts, as well as tent intercepts.
        """
        self.df['timeShutBase'] = self.df.apply(
            lambda row: self.compute_time_shut_base(row, self.data, self.srate, self.shutAmpFraction),
            axis=1
        )
        self.df['closingTimeTent'] = (self.df['xIntersect'] - self.df['leftXIntercept']) / self.srate
        self.df['reopeningTimeTent'] = (self.df['rightXIntercept'] - self.df['xIntersect']) / self.srate

        self.df['timeShutTent'] = self.df.apply(
            lambda row: self.compute_time_shut_tent(row, self.data, self.srate, self.shutAmpFraction),
            axis=1
        )

    def _get_argmax_val(self, row):
        """
        Internal helper to compute the argmax for a partial blink range.
        """
        left = row['leftXIntercept_int']
        right = row['rightXIntercept_int'] + 1
        start = row['start_shut_tst']
        max_val = row['maxValue']
        dconstant = self.shutAmpFraction * max_val

        subset = self.data[left:right][start:-1]
        try:
            return np.argmax(subset < dconstant)
        except ValueError:
            return np.nan

    def extract_other_times(self):
        """
        Extract additional time metrics such as:
          - peakMaxBlink
          - peakMaxTent
          - peakTimeTent
          - peakTimeBlink
          - interBlinkMaxAmp
          - interBlinkMaxVelBase
          - interBlinkMaxVelZero
        """
        self.df['peakMaxBlink'] = self.df['maxValue']
        self.df['peakMaxTent'] = self.df['yIntersect']
        self.df['peakTimeTent'] = self.df['xIntersect'] / self.srate
        self.df['peakTimeBlink'] = self.df['maxFrame'] / self.srate

        # interBlinkMaxAmp
        peaks_with_len = np.append(self.df['maxFrame'].to_numpy(), len(self.data))
        self.df['interBlinkMaxAmp'] = np.diff(peaks_with_len) / self.srate

        self.df['interBlinkMaxVelBase'] = (self.df['peaksPosVelBase'] * -1) / self.srate
        self.df['interBlinkMaxVelZero'] = (self.df['peaksPosVelZero'] * -1) / self.srate

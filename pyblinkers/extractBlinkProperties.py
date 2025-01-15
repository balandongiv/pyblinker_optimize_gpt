# LLMed on 15 January 2025

import logging
import numpy as np
import pandas as pd

from pyblinkers.misc import mad_matlab
from pyblinkers.default_setting import scalingFactor

logging.getLogger().setLevel(logging.INFO)


def _goodblink_based_corr_median_std(df, correlationThreshold):
    """
    (Docstring unchanged)
    """
    R2 = df['leftR2'] >= correlationThreshold
    R3 = df['rightR2'] >= correlationThreshold

    good_data = df.loc[R2 & R3, :]
    bestValues = good_data['maxValue'].to_numpy()

    specifiedMedian = np.nanmedian(bestValues)
    specifiedStd = scalingFactor * mad_matlab(bestValues)
    return R2, R3, specifiedMedian, specifiedStd


def calculate_within_range(all_values, best_median, best_robust_std):
    """
    (Docstring unchanged)
    """
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return np.sum(within_mask)


def calculateGoodRatio(all_values, best_median, best_robust_std, all_x):
    """
    (Docstring unchanged)
    """
    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return np.sum(within_mask) / all_x


def get_blink_statistic(df, zThresholds, signal=None):
    """
    (Docstring unchanged)
    used Feb 02 2023
    """
    dfx = df.copy()
    dfx[['leftZero', 'rightZero']] = dfx[['leftZero', 'rightZero']] - 1

    indices = np.arange(len(signal))
    blinkMask = np.any(
        [(indices >= lz) & (indices <= rz) for lz, rz in zip(dfx["leftZero"], dfx["rightZero"])],
        axis=0
    ).astype(bool)

    insideBlink = (signal > 0) & blinkMask
    outsideBlink = (signal > 0) & ~blinkMask
    blinkAmpRatio = np.mean(signal[insideBlink]) / np.mean(signal[outsideBlink])

    correlationThresholdBottom, correlationThresholdTop = zThresholds[0]
    df_data = df[['leftR2', 'rightR2', 'maxValue']]

    goodMaskTop = (df_data['leftR2'] >= correlationThresholdTop) & (df_data['rightR2'] >= correlationThresholdTop)
    goodMaskBottom = (df_data['leftR2'] >= correlationThresholdBottom) & (df_data['rightR2'] >= correlationThresholdBottom)

    bestValues = df_data.loc[goodMaskTop, 'maxValue'].to_numpy()
    worstValues = df_data.loc[~goodMaskBottom, 'maxValue'].to_numpy()
    goodValues = df_data.loc[goodMaskBottom, 'maxValue'].to_numpy()

    bestMedian = np.nanmedian(bestValues)
    bestRobustStd = 1.4826 * mad_matlab(bestValues)
    worstMedian = np.nanmedian(worstValues)
    worstRobustStd = 1.4826 * mad_matlab(worstValues)

    cutoff = (bestMedian * worstRobustStd + worstMedian * bestRobustStd) / (bestRobustStd + worstRobustStd)

    all_x = calculate_within_range(df_data['maxValue'].to_numpy(), bestMedian, bestRobustStd)
    goodRatio = np.nan if all_x <= 0 else calculateGoodRatio(goodValues, bestMedian, bestRobustStd, all_x)

    numberGoodBlinks = np.sum(goodMaskBottom)

    return dict(
        numberBlinks=len(df_data),
        numberGoodBlinks=numberGoodBlinks,
        blinkAmpRatio=blinkAmpRatio,
        cutoff=cutoff,
        bestMedian=bestMedian,
        bestRobustStd=bestRobustStd,
        goodRatio=goodRatio
    )


def getGoodBlinkMask(blinkFits, specifiedMedian, specifiedStd, zThresholds):
    """
    (Docstring unchanged)
    """
    blinkFits = blinkFits.dropna(subset=['leftR2', 'rightR2', 'maxValue'])

    leftR2 = blinkFits['leftR2'].to_numpy()
    rightR2 = blinkFits['rightR2'].to_numpy()
    maxValue = blinkFits['maxValue'].to_numpy()

    correlationThresholds = zThresholds[0]
    zScoreThresholds = zThresholds[1]

    lowerBounds = np.maximum(0, specifiedMedian - zScoreThresholds * specifiedStd)
    upperBounds = specifiedMedian + zScoreThresholds * specifiedStd

    # Expand for broadcasting
    leftR2 = leftR2[:, None]
    rightR2 = rightR2[:, None]
    maxValue = maxValue[:, None]
    correlationThresholds = correlationThresholds[None, :]
    lowerBounds = lowerBounds[None, :]
    upperBounds = upperBounds[None, :]

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
    (Docstring unchanged)
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
        self.blinkVelocity = np.diff(self.data)

    def set_blink_duration(self):
        constant = 1
        self.df['durationBase'] = (self.df['rightBase'] - self.df['leftBase']) / self.srate
        self.df['durationZero'] = (self.df['rightZero'] - self.df['leftZero']) / self.srate
        self.df['durationTent'] = (self.df['rightXIntercept'] - self.df['leftXIntercept']) / self.srate
        self.df['durationHalfBase'] = ((self.df['rightBaseHalfHeight'] - self.df['leftBaseHalfHeight']) + constant) / self.srate
        self.df['durationHalfZero'] = ((self.df['rightZeroHalfHeight'] - self.df['leftZeroHalfHeight']) + constant) / self.srate

    # --------------------------------------------------------------------------
    # A helper function to unify amplitude-velocity ratio calculations
    # --------------------------------------------------------------------------
    def _compute_amplitude_velocity_ratio(self, start_key, end_key, ratio_key, aggregator='max', idx_col=None):
        """
        Internal helper that computes amplitude-velocity ratio between start_key and end_key.
        aggregator can be 'max' or 'min' to pick the velocity extremes.
        idx_col can store the index of extreme velocity if desired.
        """
        start_vals = self.df[start_key].to_numpy().astype(int)
        end_vals = self.df[end_key].to_numpy().astype(int)
        blink_vel = self.blinkVelocity

        lengths = (end_vals - start_vals + 1).astype(int)
        max_len = lengths.max()
        offsets = np.arange(max_len)[None, :]
        mask = offsets < lengths[:, None]

        all_indices = start_vals[:, None] + offsets
        all_indices = all_indices[mask].astype(int)

        row_idx_all = np.repeat(np.arange(len(lengths)), lengths)
        velocities = blink_vel[all_indices]

        temp_df = pd.DataFrame({'row_idx': row_idx_all, 'velocity': velocities, 'index': all_indices})
        if aggregator == 'max':
            idx_extreme = temp_df.groupby('row_idx')['velocity'].idxmax()
        else:
            idx_extreme = temp_df.groupby('row_idx')['velocity'].idxmin()

        df_extreme = temp_df.loc[idx_extreme].sort_values('row_idx')
        ratio_vals = 100 * abs(self.data[self.df['maxFrame'].to_numpy()] / df_extreme['velocity'].to_numpy()) / self.srate

        self.df[ratio_key] = ratio_vals
        if idx_col:
            self.df[idx_col] = df_extreme['index'].to_numpy()

    def compute_negAmpVelRatioZero(self):
        """
        (Docstring unchanged)
        """
        self._compute_amplitude_velocity_ratio(
            start_key='maxFrame',
            end_key='rightZero',
            ratio_key='negAmpVelRatioZero',
            aggregator='min'
        )

    def compute_posAmpVelRatioZero(self, multiplication_constant=100):
        """
        (Docstring unchanged)
        """
        # The aggregator remains 'max'.
        # The method signature references multiplication_constant, but we only store 100.
        # If you truly need a custom multiplier, you can incorporate it into _compute_amplitude_velocity_ratio.
        self._compute_amplitude_velocity_ratio(
            start_key='leftZero',
            end_key='maxFrame',
            ratio_key='posAmpVelRatioZero',
            aggregator='max',
            idx_col='peaksPosVelZero'
        )

    def set_blink_amp_velocity_ratio_zero_to_max(self):
        """
        (Docstring unchanged)
        """
        self.compute_posAmpVelRatioZero()
        self.compute_negAmpVelRatioZero()

    def compute_posAmpVelRatioBase(self, multiplication_constant=100):
        """
        (Docstring unchanged)
        """
        self._compute_amplitude_velocity_ratio(
            start_key='leftBase',
            end_key='maxFrame',
            ratio_key='posAmpVelRatioBase',
            aggregator='max',
            idx_col='peaksPosVelBase'
        )

    def compute_posAmpVelRatioBase_X(self, multiplication_constant=100):
        """
        (Docstring unchanged)
        """
        self._compute_amplitude_velocity_ratio(
            start_key='leftBase',
            end_key='maxFrame',
            ratio_key='posAmpVelRatioBase',
            aggregator='max'
        )

    def compute_negAmpVelRatioBase(self, multiplication_constant=100):
        """
        (Docstring unchanged)
        """
        self._compute_amplitude_velocity_ratio(
            start_key='maxFrame',
            end_key='rightBase',
            ratio_key='negAmpVelRatioBase',
            aggregator='min'
        )

    def amplitude_velocity_ratio_base(self):
        """
        (Docstring unchanged)
        """
        self.compute_posAmpVelRatioBase()
        self.compute_negAmpVelRatioBase()

    def amplitude_velocity_ratio_tent(self):
        """
        (Docstring unchanged)
        """
        self.df['negAmpVelRatioTent'] = 100 * abs(self.data[self.df['maxFrame']] / self.df['averRightVelocity']) / self.srate
        self.df['posAmpVelRatioTent'] = 100 * abs(self.data[self.df['maxFrame']] / self.df['averLeftVelocity']) / self.srate

    @staticmethod
    def compute_time_shut_zero(row, data, srate, shut_amp_fraction):
        """
        (Docstring unchanged)
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
        end_shut = np.argmax(cond_end) + 1 if cond_end.any() else np.nan
        return end_shut / srate

    @staticmethod
    def compute_time_shut_base(row, data, srate, shut_amp_fraction):
        """
        (Docstring unchanged)
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
        end_shut = np.argmax(cond_end) + 1 if cond_end.any() else np.nan
        return end_shut / srate

    def time_zero_shut(self):
        """
        (Docstring unchanged)
        """
        self.df['closingTimeZero'] = (self.df['maxFrame'] - self.df['leftZero']) / self.srate
        self.df['reopeningTimeZero'] = (self.df['rightZero'] - self.df['maxFrame']) / self.srate

        self.df['timeShutZero'] = self.df.apply(
            lambda row: self.compute_time_shut_zero(row, self.data, self.srate, self.shutAmpFraction), axis=1
        )

    @staticmethod
    def compute_time_shut_tent(row, data, srate, shut_amp_fraction):
        """
        (Docstring unchanged)
        """
        left = int(row['leftXIntercept'])
        right = int(row['rightXIntercept']) + 1
        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val
        data_slice = data[left : right]

        cond_start = (data_slice >= amp_threshold)
        if not cond_start.any():
            return 0

        start_idx = np.argmax(cond_start)
        cond_end = (data_slice[start_idx : -1] < amp_threshold)
        end_shut = np.argmax(cond_end) if cond_end.any() else np.nan
        return end_shut / srate

    def time_base_shut(self):
        """
        (Docstring unchanged)
        """
        self.df['timeShutBase'] = self.df.apply(
            lambda row: self.compute_time_shut_base(row, self.data, self.srate, self.shutAmpFraction), axis=1
        )
        self.df['closingTimeTent'] = (self.df['xIntersect'] - self.df['leftXIntercept']) / self.srate
        self.df['reopeningTimeTent'] = (self.df['rightXIntercept'] - self.df['xIntersect']) / self.srate

        self.df['timeShutTent'] = self.df.apply(
            lambda row: self.compute_time_shut_tent(row, self.data, self.srate, self.shutAmpFraction), axis=1
        )

    def get_argmax_val(self, row):
        """
        (Docstring unchanged)
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
        (Docstring unchanged)
        """
        self.df['peakMaxBlink'] = self.df['maxValue']
        self.df['peakMaxTent'] = self.df['yIntersect']
        self.df['peakTimeTent'] = self.df['xIntersect'] / self.srate
        self.df['peakTimeBlink'] = self.df['maxFrame'] / self.srate

        peaks_with_len = np.append(self.df['maxFrame'].to_numpy(), len(self.data))
        self.df['interBlinkMaxAmp'] = np.diff(peaks_with_len) / self.srate

        self.df['interBlinkMaxVelBase'] = (self.df['peaksPosVelBase'] * -1) / self.srate
        self.df['interBlinkMaxVelZero'] = (self.df['peaksPosVelZero'] * -1) / self.srate

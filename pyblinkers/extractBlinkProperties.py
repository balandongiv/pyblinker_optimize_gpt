
import logging

import numpy as np
import pandas as pd

from pyblinkers.matlab_forking import mad_matlab

logging.getLogger().setLevel(logging.INFO)

from pyblinkers.default_setting import SCALING_FACTOR


def calculate_within_range(all_values, best_median, best_robust_std):

    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return np.sum(within_mask)


def calculate_good_ratio(all_values, best_median, best_robust_std, all_x):

    lower_bound = best_median - 2 * best_robust_std
    upper_bound = best_median + 2 * best_robust_std
    within_mask = (all_values >= lower_bound) & (all_values <= upper_bound)
    return np.sum(within_mask) / all_x


def get_blink_statistic(df, zThresholds, signal=None):

    dfx = df.copy()
    dfx[['leftZero', 'rightZero']] = dfx[['leftZero', 'rightZero']] - 1

    indices = np.arange(len(signal))
    blink_mask = np.any(
        [(indices >= lz) & (indices <= rz) for lz, rz in zip(dfx["leftZero"], dfx["rightZero"])],
        axis=0
    ).astype(bool)

    inside_blink = (signal > 0) & blink_mask
    outside_blink = (signal > 0) & ~blink_mask
    blink_amp_ratio = np.mean(signal[inside_blink]) / np.mean(signal[outside_blink])

    correlation_threshold_bottom, correlation_threshold_top = zThresholds[0]
    df_data = df[['leftR2', 'rightR2', 'maxValue']]

    good_mask_top = (df_data['leftR2'] >= correlation_threshold_top) & (df_data['rightR2'] >= correlation_threshold_top)
    good_mask_bottom = (df_data['leftR2'] >= correlation_threshold_bottom) & (df_data['rightR2'] >= correlation_threshold_bottom)

    best_values = df_data.loc[good_mask_top, 'maxValue'].to_numpy()
    worst_values = df_data.loc[~good_mask_bottom, 'maxValue'].to_numpy()
    good_values = df_data.loc[good_mask_bottom, 'maxValue'].to_numpy()

    best_median = np.nanmedian(best_values)
    best_robust_std = SCALING_FACTOR * mad_matlab(best_values)
    worst_median = np.nanmedian(worst_values)
    worst_robust_std = SCALING_FACTOR * mad_matlab(worst_values)

    cutoff = (best_median * worst_robust_std + worst_median * best_robust_std) / (best_robust_std + worst_robust_std)

    all_x = calculate_within_range(df_data['maxValue'].to_numpy(), best_median, best_robust_std)
    good_ratio = np.nan if all_x <= 0 else calculate_good_ratio(good_values, best_median, best_robust_std, all_x)

    number_good_blinks = np.sum(good_mask_bottom)

    return dict(
        numberBlinks=len(df_data),
        numberGoodBlinks=number_good_blinks,
        blinkAmpRatio=blink_amp_ratio,
        cutoff=cutoff,
        bestMedian=best_median,
        bestRobustStd=best_robust_std,
        goodRatio=good_ratio
    )


def get_good_blink_mask(blink_fits, specified_median, specified_std, z_thresholds):
    """Generates a boolean mask for 'good' blinks based on blink fit parameters and thresholds.

    This function determines which blinks are considered 'good' based on correlation coefficients
    (leftR2, rightR2) and maximum blink amplitude (maxValue) from a DataFrame of blink fits.
    It applies both correlation thresholds and z-score based amplitude thresholds to classify blinks.

    Parameters
    ----------
    blink_fits : pandas.DataFrame
        DataFrame containing blink fit parameters, expected to have columns
        'leftR2', 'rightR2', and 'maxValue'. Rows with NaN values in these columns will be dropped.
    specified_median : float
        Median value of 'good' blinks' maxValue, used as the center for z-score thresholding.
    specified_std : float
        Robust standard deviation of 'good' blinks' maxValue, used for z-score thresholding.
    z_thresholds : list
        A list containing two lists of thresholds.
        z_thresholds[0] is a list of correlation thresholds for leftR2 and rightR2.
        z_thresholds[1] is a list of z-score thresholds for maxValue.

    Returns
    -------
    tuple
        A tuple containing:
            - good_blink_mask : numpy.ndarray (bool)
                Boolean mask indicating 'good' blinks (True) and 'bad' blinks (False).
            - selected_rows : pandas.DataFrame
                DataFrame containing only the rows from `blink_fits` that correspond to 'good' blinks.
    """
    blink_fits = blink_fits.dropna(subset=['leftR2', 'rightR2', 'maxValue'])

    left_r2 = blink_fits['leftR2'].to_numpy()
    right_r2 = blink_fits['rightR2'].to_numpy()
    max_value = blink_fits['maxValue'].to_numpy()

    correlation_thresholds = z_thresholds[0]
    z_score_thresholds = z_thresholds[1]

    lower_bounds = np.maximum(0, specified_median - z_score_thresholds * specified_std)
    upper_bounds = specified_median + z_score_thresholds * specified_std

    # Expand for broadcasting
    left_r2 = left_r2[:, None]
    right_r2 = right_r2[:, None]
    max_value = max_value[:, None]
    correlation_thresholds = correlation_thresholds[None, :]
    lower_bounds = lower_bounds[None, :]
    upper_bounds = upper_bounds[None, :]

    masks = (
            (left_r2 >= correlation_thresholds) &
            (right_r2 >= correlation_thresholds) &
            (max_value >= lower_bounds) &
            (max_value <= upper_bounds)
    )
    good_blink_mask = np.any(masks, axis=1)
    selected_rows = blink_fits[good_blink_mask]
    return good_blink_mask, selected_rows


class BlinkProperties:
    '''
    Return a structure with blink shapes and properties for individual blinks
    % Return a structure with blink shapes and properties for individual blinks
    %
    % Parameters:
    %     signalData    signalData structure
    %     params        params structure with parameters
    %     blinkProps    (output) structure with the blink properties
    %     blinkFits     (output) structure with the blink landmarks
    '''
    def __init__(self, candidate_signal, df, srate, params):
        """Initializes BlinkProperties object to calculate blink features.

       This class calculates various properties of detected blinks based on
       the input candidate_signal, DataFrame of blink fits, sampling rate, and parameters.
       It initializes blink velocity, durations, amplitude-velocity ratios,
       and time-related features.

       Parameters
       ----------
       candidate_signal : numpy.ndarray
           The raw signal candidate_signal from which blinks were detected.
       df : pandas.DataFrame
           DataFrame containing blink fit parameters, expected to have columns like
           'leftBase', 'rightBase', 'leftZero', 'rightZero', 'rightXIntercept', 'leftXIntercept',
           'leftBaseHalfHeight', 'rightBaseHalfHeight', 'leftZeroHalfHeight', 'rightZeroHalfHeight',
           'maxFrame', 'maxValue', 'averRightVelocity', 'averLeftVelocity', 'xIntersect', 'yIntersect',
           'leftXIntercept_int', 'rightXIntercept_int', 'start_shut_tst', 'peaksPosVelBase', 'peaksPosVelZero'.
       srate : float
           Sampling rate of the signal candidate_signal in Hz.
       params : dict
           Dictionary of parameters, expected to contain keys:
               - 'shutAmpFraction': Fraction of maximum amplitude for shut time calculation.
               - 'z_thresholds': Z-score thresholds (structure of thresholds is assumed to be handled internally by methods using it).
       """
        self.signal_l = None
        self.blinkVelocity = None
        self.candidate_signal = candidate_signal
        self.df = df
        self.srate = srate
        self.shutAmpFraction = params['shutAmpFraction']
        self.pAVRThreshold = params['shutAmpFraction']
        self.zThresholds = params['z_thresholds']

        self.df_res = []
        self.reset_index()
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
        self.signal_l = self.candidate_signal.shape[0]
        self.blinkVelocity = np.diff(self.candidate_signal)

    def set_blink_duration(self):

        '''
         Calculates 'durationBase', 'durationZero', 'durationTent', 'durationHalfBase', and 'durationHalfZero'
        and adds them as new columns to the `self.df` DataFrame.
        Durations are calculated based on different blink landmarks and sampling rate.
        '''
        constant = 1 # Constant is for matching Matlab implementation output
        self.df['durationBase'] = (self.df['rightBase'] - self.df['leftBase']) / self.srate
        self.df['durationZero'] = (self.df['rightZero'] - self.df['leftZero']) / self.srate
        self.df['durationTent'] = (self.df['rightXIntercept'] - self.df['leftXIntercept']) / self.srate
        self.df['durationHalfBase'] = ((self.df['rightBaseHalfHeight'] - self.df['leftBaseHalfHeight']) + constant) / self.srate
        self.df['durationHalfZero'] = ((self.df['rightZeroHalfHeight'] - self.df['leftZeroHalfHeight']) + constant) / self.srate


    def compute_amplitude_velocity_ratio(self, start_key, end_key, ratio_key, aggregator='max', idx_col=None):
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
        ratio_vals = 100 * abs(self.candidate_signal[self.df['maxFrame'].to_numpy()] / df_extreme['velocity'].to_numpy()) / self.srate

        self.df[ratio_key] = ratio_vals
        if idx_col:
            self.df[idx_col] = df_extreme['index'].to_numpy()

    def compute_neg_amp_vel_ratio_zero(self):
        """
        Computes and sets negative amplitude-velocity ratio from maxFrame to rightZero in DataFrame.
        Computes and sets positive amplitude-velocity ratio from leftZero to maxFrame in DataFrame.

        """
        self.compute_amplitude_velocity_ratio(
            start_key='maxFrame',
            end_key='rightZero',
            ratio_key='negAmpVelRatioZero',
            aggregator='min'
        )

    def compute_pos_amp_vel_ratio_zero(self):
        """
        Computes and sets positive amplitude-velocity ratio from leftZero to maxFrame in DataFrame.
        """

        self.compute_amplitude_velocity_ratio(
            start_key='leftZero',
            end_key='maxFrame',
            ratio_key='posAmpVelRatioZero',
            aggregator='max',
            idx_col='peaksPosVelZero'
        )

    def set_blink_amp_velocity_ratio_zero_to_max(self):
        """"Computes and sets both positive and negative amplitude-velocity ratios (zero-to-max)."""
        self.compute_pos_amp_vel_ratio_zero()
        self.compute_neg_amp_vel_ratio_zero()

    def compute_pos_amp_vel_ratio_base(self):
        """Computes and sets positive amplitude-velocity ratio from leftBase to maxFrame in DataFrame."""
        self.compute_amplitude_velocity_ratio(
            start_key='leftBase',
            end_key='maxFrame',
            ratio_key='posAmpVelRatioBase',
            aggregator='max',
            idx_col='peaksPosVelBase'
        )

    def compute_neg_amp_vel_ratio_base(self):
        """Computes and sets negative amplitude-velocity ratio from maxFrame to rightBase in DataFrame."""
        self.compute_amplitude_velocity_ratio(
            start_key='maxFrame',
            end_key='rightBase',
            ratio_key='negAmpVelRatioBase',
            aggregator='min'
        )

    def amplitude_velocity_ratio_base(self):
        '''
        Blink amplitude-velocity ratio from base to max
        :return:
        '''
        self.compute_pos_amp_vel_ratio_base()
        self.compute_neg_amp_vel_ratio_base()

    def amplitude_velocity_ratio_tent(self):
        '''
         Blink amplitude-velocity ratio estimated from tent slope
        :return:
        '''
        self.df['negAmpVelRatioTent'] = 100 * abs(self.candidate_signal[self.df['maxFrame']] / self.df['averRightVelocity']) / self.srate
        self.df['posAmpVelRatioTent'] = 100 * abs(self.candidate_signal[self.df['maxFrame']] / self.df['averLeftVelocity']) / self.srate
    @staticmethod
    def compute_time_shut(row, data, srate, shut_amp_fraction, key_prefix, default_no_thresh):
        """
        Compute shut duration using specified landmarks for a blink.
        Basiclly, we combine what before having seperate function but mostly overlap syntax

        This code will be use to calculate the  timeShutBase and timeShutZero
        Parameters:
          row : pandas.Series
              A row containing the blink candidate_signal.
          data : array-like
              Signal candidate_signal from which to compute the duration.
          srate : float
              Sampling rate.
          shut_amp_fraction : float
              Fraction of the max amplitude used to determine the threshold.
          key_prefix : str
              The suffix used in the column names to determine the left/right boundaries.
              Use 'Zero' for zero-crossing landmarks or 'Base' for base landmarks.
          default_no_thresh : scalar
              The value to return if the signal never meets the threshold.
              For example, np.nan for zero-crossing or 0 for base landmarks.

        Returns:
          float
              The computed shut duration in seconds.
        """
        left = int(row[f'left{key_prefix}'])    # This approach is sensetive should we change the pandas column name in the future
        right = int(row[f'right{key_prefix}'])  # This approach is sensetive should we change the pandas column name in the future
        threshold = shut_amp_fraction * row['maxValue']
        data_slice = data[left:right+1]

        # Find the start index where the signal first reaches/exceeds the threshold.
        cond = data_slice >= threshold
        if not cond.any():
            return default_no_thresh
        start_idx = cond.argmax()

        # Find the first index after start_idx where the signal drops below the threshold.
        cond = data_slice[start_idx+1:] < threshold
        end_idx = cond.argmax() + 1 if cond.any() else np.nan
        return end_idx / srate


    def time_zero_shut(self):
        '''
        Time zero shut
        :return:
        '''
        self.df['closingTimeZero'] = (self.df['maxFrame'] - self.df['leftZero']) / self.srate
        self.df['reopeningTimeZero'] = (self.df['rightZero'] - self.df['maxFrame']) / self.srate


        self.df['timeShutBase'] = self.df.apply(
            lambda row: self.compute_time_shut(row, self.candidate_signal, self.srate, self.shutAmpFraction,
                                               key_prefix='Base', default_no_thresh=0),
            axis=1
        )

    @staticmethod
    def compute_time_shut_tent(row, candidate_signal, srate, shut_amp_fraction):

        left = int(row['leftXIntercept'])
        right = int(row['rightXIntercept']) + 1
        max_val = row['maxValue']
        amp_threshold = shut_amp_fraction * max_val
        data_slice = candidate_signal[left: right]

        cond_start = (data_slice >= amp_threshold)
        if not cond_start.any():
            return 0

        start_idx = np.argmax(cond_start)
        cond_end = (data_slice[start_idx : -1] < amp_threshold)
        end_shut = np.argmax(cond_end) if cond_end.any() else np.nan
        return end_shut / srate

    def time_base_shut(self):
        '''
        Time base shut
        :return:
        '''


        self.df['timeShutBase'] = self.df.apply(
            lambda row: self.compute_time_shut(row, self.candidate_signal, self.srate, self.shutAmpFraction,
                                               key_prefix='Base', default_no_thresh=0),
            axis=1
        )
        # Below is to calculate the Time shut tent
        self.df['closingTimeTent'] = (self.df['xIntersect'] - self.df['leftXIntercept']) / self.srate
        self.df['reopeningTimeTent'] = (self.df['rightXIntercept'] - self.df['xIntersect']) / self.srate

        self.df['timeShutTent'] = self.df.apply(
            lambda row: self.compute_time_shut_tent(row, self.candidate_signal, self.srate, self.shutAmpFraction), axis=1
        )

    def get_argmax_val(self, row):

        left = row['leftXIntercept_int']
        right = row['rightXIntercept_int'] + 1
        start = row['start_shut_tst']
        max_val = row['maxValue']
        subset = self.candidate_signal[left:right][start:-1]
        try:
            return np.argmax(subset < self.shutAmpFraction * max_val)
        except ValueError:
            return np.nan

    def extract_other_times(self):

        self.df['peakMaxBlink'] = self.df['maxValue']
        self.df['peakMaxTent'] = self.df['yIntersect']
        self.df['peakTimeTent'] = self.df['xIntersect'] / self.srate
        self.df['peakTimeBlink'] = self.df['maxFrame'] / self.srate

        peaks_with_len = np.append(self.df['maxFrame'].to_numpy(), len(self.candidate_signal))
        self.df['interBlinkMaxAmp'] = np.diff(peaks_with_len) / self.srate

        self.df['interBlinkMaxVelBase'] = (self.df['peaksPosVelBase'] * -1) / self.srate
        self.df['interBlinkMaxVelZero'] = (self.df['peaksPosVelZero'] * -1) / self.srate

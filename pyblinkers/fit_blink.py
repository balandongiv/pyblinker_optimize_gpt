# LLMed on 15 January 2025
from pyblinkers.utils._logging import logger
import numpy as np

from pyblinkers.zero_crossing import (
    _get_half_height,
    compute_fit_range,
    left_right_zero_crossing
)
from pyblinkers.base_left_right import create_left_right_base
from pyblinkers.line_intersection_matlab import lines_intersection_matlabx


class FitBlinks:

    def __init__(self, candidate_signal=None, df=None, params=None):
        # candidateSignal    IC or channel time course of blinks to be fitted

        # saved as numpy array the candidate_signal


        # np.save('S1_candidate_signal.npy', candidate_signal)
        self.candidate_signal = candidate_signal
        self.df = df
        self.frame_blinks = []
        self.base_fraction = params['base_fraction']

        # Column lists (these names remain unchanged)
        self.cols_half_height = [
            'left_zero_half_height', 'right_zero_half_height',
            'left_base_half_height', 'right_base_half_height'
        ]
        self.cols_fit_range = [
            'xLeft', 'xRight', 'leftRange', 'rightRange',
            'blinkBottomPoint_l_Y', 'blinkBottomPoint_l_X',
            'blinkTopPoint_l_Y', 'blinkTopPoint_l_X',
            'blinkBottomPoint_r_X', 'blinkBottomPoint_r_Y',
            'blinkTopPoint_r_X', 'blinkTopPoint_r_Y'
        ]
        self.cols_lines_intesection = [
            'leftSlope', 'rightSlope', 'aver_left_velocity', 'aver_right_velocity',
            'rightR2', 'leftR2', 'x_intersect', 'y_intersect',
            'left_x_intercept', 'right_x_intercept',
            'xLineCross_l', 'yLineCross_l', 'xLineCross_r', 'yLineCross_r'
        ]

    def get_max_frame(self, start_idx, end_idx):
        """
        Find the maxFrames
        Compute the maximum value in self.candidate_signal between start_idx and end_idx
        and return (max_value, frame_index_at_max).
        """
        blink_range = np.arange(start_idx, end_idx + 1)
        blink_frame = self.candidate_signal[start_idx:end_idx + 1]
        # One-pass for max value and index
        max_idx = np.argmax(blink_frame)
        max_val = blink_frame[max_idx]
        max_fr = blink_range[max_idx]
        return max_val, max_fr

    def dprocess(self):
        data_size = self.candidate_signal.size  # store locally to avoid repeated lookups

        # Find the max_blink index and max_value at that max_blink index
        self.df[['max_value', 'max_blink']] = self.df.apply(
            lambda row: self.get_max_frame(row['start_blink'], row['end_blink']),
            axis=1,
            result_type='expand'
        )
        # Ensure the max_blink is integer
        self.df['max_blink'] = self.df['max_blink'].astype(int)

        # Shifts for outer starts/ends
        self.df['outer_start'] = self.df['max_blink'].shift(1, fill_value=0)
        self.df['outer_end'] = self.df['max_blink'].shift(-1, fill_value=data_size)

        # Add columns for left_zero/right_zero
        self.df[['left_zero', 'right_zero']] = self.df.apply(
            lambda row: left_right_zero_crossing(
                self.candidate_signal,
                row['max_blink'],
                row['outer_start'],
                row['outer_end']
            ),
            axis=1,
            result_type='expand'
        )

        # Perform fitting calculations
        self.fit()

    def fit(self):
        """
        Main method to create base line fits, compute half-height, fit ranges,
        and line intersections.
        """
        # candidate_signal = self.candidate_signal  # Local reference for efficiency

        # Create left and right base lines
        self.frame_blinks = create_left_right_base(self.candidate_signal, self.df)

        # Get half height
        self.frame_blinks[self.cols_half_height] = self.frame_blinks.apply(
            lambda row: _get_half_height(
                self.candidate_signal,
                row['max_blink'],
                row['left_zero'],
                row['right_zero'],
                row['left_base'],
                row['outer_end']
            ),
            axis=1,
            result_type='expand'
        )

        # Compute fit ranges
        self.frame_blinks[self.cols_fit_range] = self.frame_blinks.apply(
            lambda row: compute_fit_range(
                self.candidate_signal,
                row['max_blink'],
                row['left_zero'],
                row['right_zero'],
                self.base_fraction,
                top_bottom=True
            ),
            axis=1,
            result_type='expand'
        )

        # Drop rows with NaN values
        self.frame_blinks.dropna(inplace=True)
        self.frame_blinks['nsize_xLeft'] = self.frame_blinks['xLeft'].apply(len)
        self.frame_blinks['nsize_xRight'] = self.frame_blinks['xRight'].apply(len)

        # Keep only rows with nsize_xLeft > 1 and nsize_xRight > 1
        self.frame_blinks = self.frame_blinks[
            (self.frame_blinks['nsize_xLeft'] > 1) &
            (self.frame_blinks['nsize_xRight'] > 1)
            ].reset_index(drop=True)

        # Calculate line intersections

        self.frame_blinks[self.cols_lines_intesection] = self.frame_blinks.apply(
            lambda row: lines_intersection_matlabx(
                signal=self.candidate_signal,
                xRight=row['xRight'],
                xLeft=row['xLeft'],
            ),
            axis=1,
            result_type='expand'
        )



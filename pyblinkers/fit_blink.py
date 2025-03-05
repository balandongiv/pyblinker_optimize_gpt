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
        self.candidate_signal = candidate_signal
        self.df = df
        self.frame_blinks = []
        self.baseFraction = params['baseFraction']

        # Column lists (these names remain unchanged)
        self.cols_half_height = [
            'leftZeroHalfHeight', 'rightZeroHalfHeight',
            'leftBaseHalfHeight', 'rightBaseHalfHeight'
        ]
        self.cols_fit_range = [
            'xLeft', 'xRight', 'leftRange', 'rightRange',
            'blinkBottomPoint_l_Y', 'blinkBottomPoint_l_X',
            'blinkTopPoint_l_Y', 'blinkTopPoint_l_X',
            'blinkBottomPoint_r_X', 'blinkBottomPoint_r_Y',
            'blinkTopPoint_r_X', 'blinkTopPoint_r_Y'
        ]
        self.cols_lines_intesection = [
            'leftSlope', 'rightSlope', 'averLeftVelocity', 'averRightVelocity',
            'rightR2', 'leftR2', 'xIntersect', 'yIntersect',
            'leftXIntercept', 'rightXIntercept',
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

        # Find the maxFrame index and maxValue at that maxFrame index
        self.df[['maxValue', 'maxFrame']] = self.df.apply(
            lambda row: self.get_max_frame(row['startBlinks'], row['endBlinks']),
            axis=1,
            result_type='expand'
        )
        # Ensure the maxFrame is integer
        self.df['maxFrame'] = self.df['maxFrame'].astype(int)

        # Shifts for outer starts/ends
        self.df['outerStarts'] = self.df['maxFrame'].shift(1, fill_value=0)
        self.df['outerEnds'] = self.df['maxFrame'].shift(-1, fill_value=data_size)

        # Add columns for leftZero/rightZero
        self.df[['leftZero', 'rightZero']] = self.df.apply(
            lambda row: left_right_zero_crossing(
                self.candidate_signal,
                row['maxFrame'],
                row['outerStarts'],
                row['outerEnds']
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
                row['maxFrame'],
                row['leftZero'],
                row['rightZero'],
                row['leftBase'],
                row['outerEnds']
            ),
            axis=1,
            result_type='expand'
        )

        # Compute fit ranges
        self.frame_blinks[self.cols_fit_range] = self.frame_blinks.apply(
            lambda row: compute_fit_range(
                self.candidate_signal,
                row['maxFrame'],
                row['leftZero'],
                row['rightZero'],
                self.baseFraction,
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

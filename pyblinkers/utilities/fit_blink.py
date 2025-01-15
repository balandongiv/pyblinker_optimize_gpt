
import logging

import numpy as np

from pyblinkers.utilities.zero_crossing import (_get_half_height,
                                                compute_fit_range)
from pyblinkers.vislab.base_left_right import create_left_right_base_vislab
from pyblinkers.line_intersection_matlab import lines_intersection_matlabx
from pyblinkers.utilities.zero_crossing import left_right_zero_crossing
logging.getLogger().setLevel(logging.INFO)

class FitBlinks:

    def __init__(self, data=None, df=None,params=None):
        self.data = data
        self.df = df
        self.frame_blinks=[]
        self.baseFraction = params['baseFraction']
        # self.baseFraction = 0.1
        self.cols_half_height = ['leftZeroHalfHeight', 'rightZeroHalfHeight', 'leftBaseHalfHeight', 'rightBaseHalfHeight']
        self.cols_fit_range = ['xLeft', 'xRight', 'leftRange', 'rightRange',
                               'blinkBottomPoint_l_Y', 'blinkBottomPoint_l_X', 'blinkTopPoint_l_Y', 'blinkTopPoint_l_X',
                               'blinkBottomPoint_r_X', 'blinkBottomPoint_r_Y', 'blinkTopPoint_r_X', 'blinkTopPoint_r_Y']
        self.cols_lines_intesection = ['leftSlope', 'rightSlope', 'averLeftVelocity', 'averRightVelocity',
                                       'rightR2', 'leftR2', 'xIntersect', 'yIntersect', 'leftXIntercept',
                                       'rightXIntercept', 'xLineCross_l', 'yLineCross_l', 'xLineCross_r', 'yLineCross_r']



    def _get_max_frame(self, startBlinks, endBlinks):
        blinkRange = np.arange(startBlinks, endBlinks + 1)
        blink_frame = self.data[startBlinks:endBlinks + 1]
        maxValues = np.amax(blink_frame)
        maxFrames = blinkRange[np.argmax(blink_frame)]
        return maxValues, maxFrames

    def dprocess(self):

        # Find the maxFrames index and maxValues at that maxFrames index
        self.df[['maxValue', 'maxFrame']] = self.df.apply(
            lambda row: self._get_max_frame(row['startBlinks'], row['endBlinks']),
            axis=1, result_type='expand'
        )

        self.df['maxFrame'] = self.df['maxFrame'].astype(int)


        self.df['outerStarts'] = self.df['maxFrame'].shift(1, fill_value=0)
        self.df['outerEnds'] = self.df['maxFrame'].shift(-1, fill_value=self.data.size)
        self.df[['leftZero', 'rightZero']] = self.df.apply(lambda x: left_right_zero_crossing(self.data,x['maxFrame'], x['outerStarts'],
                                                                                              x['outerEnds']), axis=1,
                                                           result_type="expand")


        # Calculate the fits
        self.fit()


    def fit(self):
        # 11/11/2024 I have cross check with graphical output for the create_left_right_base_vislab function
        self.frame_blinks = create_left_right_base_vislab(self.data, self.df)

        # 11/11/2024 I have cross check with graphical output for the _get_half_height function
        self.frame_blinks[self.cols_half_height] = self.frame_blinks.apply(lambda x: _get_half_height(self.data, x['maxFrame'], x['leftZero'], x['rightZero'],
                                                                                       x['leftBase'], x['outerEnds']), axis=1,
                                                       result_type="expand")

        # Compute fit ranges
        self.frame_blinks[self.cols_fit_range] = self.frame_blinks.apply(lambda x: compute_fit_range(self.data, x['maxFrame'], x['leftZero'], x['rightZero'],
                                                                                      self.baseFraction, top_bottom=True), axis=1,
                                                     result_type="expand")
        self.frame_blinks = self.frame_blinks.dropna()

        # These three line to check nsize_xLeft and nsize_xRight is equivalent to checking if length(xLeft) > 1 && length(xRight) > 1
        self.frame_blinks['nsize_xLeft'] = self.frame_blinks.apply(lambda x: x['xLeft'].size, axis=1)
        self.frame_blinks['nsize_xRight'] = self.frame_blinks.apply(lambda x: x['xRight'].size, axis=1)
        self.frame_blinks = self.frame_blinks[~(self.frame_blinks['nsize_xLeft'] <= 1) & ~(self.frame_blinks['nsize_xRight'] <= 1)]
        self.frame_blinks.reset_index(drop=True, inplace=True)


        self.frame_blinks[self.cols_lines_intesection] = self.frame_blinks.apply(lambda x: lines_intersection_matlabx(signal=self.data,xRight=x['xRight'], xLeft=x['xLeft'],
                                                                                                              yRight=self.data[x['xRight']], yLeft=self.data[x['xLeft']],
                                                                                                              dic_type=False), axis=1, result_type="expand")

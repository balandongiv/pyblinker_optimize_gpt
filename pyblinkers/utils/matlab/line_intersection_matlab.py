
from pyblinkers.utils.matlab.matlab_forking import (
    corr_matlab,
    polyval_matlab,
    polyfit_matlab,
    get_intersection,
)
from pyblinkers.zero_crossing import get_line_intersection_slope
import numpy as np
def lines_intersection_matlabx(signal=None,xRight=None, xLeft=None):

    yRight = signal[xRight]
    yLeft = signal[xLeft]
    n=1
    pLeft, SLeft, muLeft = polyfit_matlab(xLeft, yLeft, n)
    yPred, delta = polyval_matlab(pLeft, xLeft, S=SLeft, mu=muLeft)
    leftR2, _ = corr_matlab(yLeft, yPred)



    pRight, SRight, muRight = polyfit_matlab(xRight, yRight, 1)
    yPredRight, delta = polyval_matlab(pRight, xRight, S=SRight, mu=muRight)
    rightR2, _ = corr_matlab(yRight, yPredRight)

    x_intersect, y_intersect, left_x_intercept, right_x_intercept = get_intersection(pLeft, pRight, muLeft, muRight)



    ### leftSlope,rightSlope
    leftSlope,rightSlope=get_line_intersection_slope(x_intersect,y_intersect,left_x_intercept,right_x_intercept)

    ### aver_left_velocity,aver_right_velocity
    aver_left_velocity=pLeft[0]/muLeft[1]
    aver_right_velocity=pRight[0]/muRight[1]

    # I am not sure about the following lines, and whether it will be use or not
    xLineCross_l, yLineCross_l, xLineCross_r, yLineCross_r=np.nan, np.nan, np.nan, np.nan
    return leftSlope, rightSlope, aver_left_velocity, aver_right_velocity, \
        rightR2[0][0], leftR2[0][0], x_intersect, y_intersect, left_x_intercept, right_x_intercept, \
        xLineCross_l, yLineCross_l, xLineCross_r, yLineCross_r

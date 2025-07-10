
import numpy as np

from pyblinkers.fitutils.forking import (
    corr,
    polyval,
    polyfit,
    get_intersection,
)
from pyblinkers.blinkers.zero_crossing import get_line_intersection_slope


def lines_intersection(signal=None, xRight=None, xLeft=None):

    yRight = signal[xRight]
    yLeft = signal[xLeft]
    n=1
    pLeft, SLeft, muLeft = polyfit(xLeft, yLeft, n)
    yPred, _ = polyval(pLeft, xLeft, S=SLeft, mu=muLeft)
    leftR2, _ = corr(yLeft, yPred)



    pRight, SRight, muRight = polyfit(xRight, yRight, 1)
    yPredRight, _ = polyval(pRight, xRight, S=SRight, mu=muRight)
    rightR2, _ = corr(yRight, yPredRight)

    xIntersect, yIntersect, leftXIntercept, rightXIntercept = get_intersection(pLeft, pRight, muLeft, muRight)



    ### leftSlope,rightSlope
    leftSlope,rightSlope=get_line_intersection_slope(xIntersect,yIntersect,leftXIntercept,rightXIntercept)

    ### averLeftVelocity,averRightVelocity
    averLeftVelocity=pLeft[0]/muLeft[1]
    averRightVelocity=pRight[0]/muRight[1]

    # I am not sure about the following lines, and whether it will be use or not
    xLineCross_l, yLineCross_l, xLineCross_r, yLineCross_r=np.nan, np.nan, np.nan, np.nan
    return leftSlope, rightSlope, averLeftVelocity, averRightVelocity, \
        rightR2[0][0], leftR2[0][0], xIntersect, yIntersect, leftXIntercept, rightXIntercept, \
        xLineCross_l, yLineCross_l, xLineCross_r, yLineCross_r

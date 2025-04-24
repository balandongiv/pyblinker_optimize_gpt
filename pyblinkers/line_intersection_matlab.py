
from pyblinkers.matlab_forking import corrMatlab, polyvalMatlab, polyfitMatlab, get_intersection
from pyblinkers.zero_crossing import get_line_intersection_slope
import numpy as np
def lines_intersection_matlabx(signal=None,xRight=None, xLeft=None):

    yRight = signal[xRight]
    yLeft = signal[xLeft]

    # The value n equals 1 means that we are fitting a line (degree 1 polynomial)
    poly_degree=1
    pLeft, SLeft, muLeft = polyfitMatlab(xLeft, yLeft, poly_degree)
    yPred, delta = polyvalMatlab(pLeft, xLeft, S=SLeft, mu= muLeft)
    leftR2, _ = corrMatlab(yLeft , yPred)



    pRight, SRight, muRight = polyfitMatlab(xRight, yRight, 1)
    yPredRight, delta = polyvalMatlab(pRight, xRight, S=SRight, mu= muRight)
    rightR2, _ = corrMatlab(yRight , yPredRight)

    xIntersect, yIntersect, leftXIntercept, rightXIntercept = get_intersection(pLeft, pRight, muLeft, muRight)



    ### leftSlope,rightSlope
    leftSlope,rightSlope=get_line_intersection_slope(xIntersect,yIntersect,leftXIntercept,rightXIntercept)

    ### averLeftVelocity,averRightVelocity
    averLeftVelocity=pLeft[0]/muLeft[1]
    averRightVelocity=pRight[0]/muRight[1]


    return leftSlope, rightSlope, averLeftVelocity, averRightVelocity, \
        rightR2[0][0], leftR2[0][0], xIntersect, yIntersect, leftXIntercept, rightXIntercept, \


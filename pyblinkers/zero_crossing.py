import warnings

import numpy as np


def get_line_intersection_slope(xIntersect,yIntersect,leftXIntercept,rightXIntercept):
    # YES
    leftSlope = yIntersect / (xIntersect - leftXIntercept)
    rightSlope = yIntersect / (xIntersect - rightXIntercept)
    return leftSlope,rightSlope

def get_average_velocity(pLeft,pRight,xLeft,xRight):
    # YES
    averLeftVelocity = pLeft.coef[1] / np.std(xLeft)  # 0.36513701
    averRightVelocity = pRight.coef[1] / np.std(xRight)  # -0.068895057

    return averLeftVelocity,averRightVelocity




def left_right_zero_crossing(candidateSignal, maxFrame, outerStarts, outerEnds):
    ### YES Latest as of 29 April 2022 which is more efficient
    theRange = np.arange(int(outerStarts), int(maxFrame))
    sInd_leftZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_leftZero.size != 0):
        leftZero = theRange[sInd_leftZero[-1]]

    else:
        extreme_outerStartss = np.arange(0, maxFrame)
        extreme_outerStartss = extreme_outerStartss.astype(int)
        sInd_rightZero_ex = np.flatnonzero(candidateSignal[extreme_outerStartss] < 0)[-1]
        leftZero = extreme_outerStartss[sInd_rightZero_ex]

    theRange = np.arange(int(maxFrame), int(outerEnds))
    sInd_rightZero = np.flatnonzero(candidateSignal[theRange] < 0)

    if (sInd_rightZero.size != 0):
        rightZero = theRange[sInd_rightZero[0]]
    else:
        """
        We take extreme remedy by extending the outerEnds to the maximum
        """
        try:
            extreme_outerEns = np.arange(maxFrame, candidateSignal.shape)
        except TypeError:
            print('Error')
        extreme_outerEns = extreme_outerEns.astype(int)
        sInd_rightZero_ex_s = np.flatnonzero(candidateSignal[extreme_outerEns] < 0)

        if (sInd_rightZero_ex_s.size != 0):
            # This usually happen for end of signal
            sInd_rightZero_ex = sInd_rightZero_ex_s[0]
            rightZero = extreme_outerEns[sInd_rightZero_ex]
        else:
            return leftZero, None

    if leftZero > maxFrame:
        raise ValueError('something is not right')

    if maxFrame > rightZero:
        raise ValueError('something is not right')


    return leftZero, rightZero


def get_up_down_stroke(maxFrame, leftZero, rightZero):
    # YES Compute the place of maximum positive and negative velocities.
    # upStroke is the interval between leftZero and maxFrame, downStroke is the interval between maxFrame and rightZero.
    # For time being, lets us apppend +1 to ensure we get the same nsize as what in the original MATLAB code
    upStroke = np.arange(leftZero, maxFrame+1)
    downStroke = np.arange(maxFrame, rightZero+1)
    return upStroke, downStroke


def _maxPosVelFrame(blinkVelocity, maxFrame, leftZero, rightZero):
    '''
    YES


    In the context of *blinkVelocity* time series, the `maxPosVelFrame` and `maxNegVelFrame` represent the indices where the *blinkVelocity* reaches its maximum positive value and maximum negative value, respectively. These values are determined within the boundaries defined by `LeftZero` and `RightZero`.
    :param blinkVelocity:
    :param maxFrame:
    :param leftZero:
    :param rightZero:
    :return:
    '''
    maxFrame, leftZero, rightZero = int(maxFrame), int(leftZero), int(rightZero)
    upStroke, downStroke = get_up_down_stroke(maxFrame, leftZero, rightZero)
    maxPosVelFrame = np.argmax(blinkVelocity[upStroke])
    maxPosVelFrame = maxPosVelFrame + upStroke[0]


    if len(blinkVelocity[downStroke])>0:
        maxNegVelFrame = np.argmin(blinkVelocity[downStroke])
        maxNegVelFrame = maxNegVelFrame + downStroke[0]
    else:
        warnings.warn('Force nan but require further investigation why happen like this')
        maxNegVelFrame=np.nan


    return maxPosVelFrame, maxNegVelFrame


def _get_left_base(blinkVelocity, leftOuter, maxPosVelFrame):
    # YES

    leftOuter, maxPosVelFrame = int(leftOuter), int(maxPosVelFrame)

    # We need to append +1 to ensure we get the same nsize as what in the original MATLAB code
    leftBase = np.arange(leftOuter, maxPosVelFrame+1)
    xx = blinkVelocity[leftBase]
    leftBaseVelocity = np.flip(xx)

    leftBaseIndex = np.argmax(leftBaseVelocity <= 0)

    # we need to -1 to ensure the leftBase position is tally what we define in the manual graph obvservation


    leftBase = maxPosVelFrame - leftBaseIndex-1

    return leftBase


def _get_right_base(candidateSignal, blinkVelocity, rightOuter, maxNegVelFrame):
    # YES Start Line 102 Matlab
    rightOuter, maxNegVelFrame = int(rightOuter), int(maxNegVelFrame)
    a_tend = np.minimum(rightOuter, candidateSignal.size)

    if maxNegVelFrame > a_tend:
        # warnings.warn(
        #     'Failed to fit blink %s but due to MaxNegVelFrame: %s larger than a_tend: %s .For now I will skip this file'
        #     % (number, maxNegVelFrame, a_tend))
        return None


    rightBase = np.arange(maxNegVelFrame, a_tend)  # Line 102 matlab

    # hh=blinkVelocity.size
    # nn=np.max(rightBase)

    if rightBase.size == 0:
        return None

    if np.max(rightBase) >= blinkVelocity.size:
        # For some reason, the original rightBase has index value greate than blinkVelocity which cause index error.
        # To address this issue, we remove some value
        rightBase = rightBase[:-1]
        if np.max(rightBase) >= blinkVelocity.size:
            raise ValueError('Please strategise how to address this')

    rightBaseVelocity = blinkVelocity[rightBase]  #

    '''
    if rightBaseIndex.size == 0:  # Line 108 Matlab
        rightBaseIndex = 0
    '''
    rightBaseIndex = np.argmax(rightBaseVelocity >= 0)

    # we need to -1 to ensure the leftBase position is tally what we define in the manual graph obvservation
    rightBase = maxNegVelFrame + rightBaseIndex+1


    return rightBase


def _get_half_height(candidateSignal, maxFrame, leftZero, rightZero, leftBase, rightOuter):
    #### YES
    """
    leftBaseHalfHeight
    The coordinate of the signal halfway (in height) between the blink maximum and the left base value. [A positive numeric value.]

    rightBaseHalfHeight
    The coordinate of the signal halfway (in height) between the blink maximum and the right base value.
    [A positive numeric value.]
    """

    maxFrame, leftZero, rightZero, leftBase, rightOuter = int(maxFrame), int(leftZero), int(rightZero), int(
        leftBase), int(rightOuter)

    blinkHalfHeight = candidateSignal[maxFrame] - (0.5 * (candidateSignal[maxFrame] - candidateSignal[leftBase]))

    # We need to append +1 to ensure we get the same nsize as what in the original MATLAB code
    leftHalfBase = np.arange(leftBase, maxFrame+1)
    dd=candidateSignal[leftHalfBase]
    cc= np.argmax(dd >= blinkHalfHeight)
    leftBaseHalfHeight = leftBase +cc+1

    # warnings.warn(
    #     'Need to double check this line:To confirm whether it is correct to used rightOuter instead of rightBase?')
    ## WIP : To confirm whether it is correct to used rightOuter instead of rightBase?
    rightHalfBase = np.arange(maxFrame, rightOuter+1)

    try:
        rightBaseHalfHeight = np.minimum(rightOuter,
                                         np.argmax(candidateSignal[rightHalfBase] <= blinkHalfHeight) + maxFrame)
    except IndexError:
        rightHalfBase = np.arange(maxFrame, rightOuter)
        rightBaseHalfHeight = np.minimum(rightOuter,
                                         np.argmax(candidateSignal[rightHalfBase] <= blinkHalfHeight) + maxFrame)
    # Consider to move this to its own designated function
    # Compute the left and right half-height frames from zero
    leftHalfBase = np.arange(leftZero, maxFrame+1)
    blinkHalfHeight = 0.5 * candidateSignal[maxFrame]  # with_val 4.3747134

    """
    leftZeroHalfHeight
    The coordinate of the signal halfway (in height) between the blink maximum and the left zero value.
    """
    leftZeroHalfHeight = np.argmax(candidateSignal[leftHalfBase] >= blinkHalfHeight) + leftZero+1

    rightHalfBase = np.arange(maxFrame, rightZero+1)

    """
    rightZeroHalfHeight
    The coordinate of the signal halfway (in height) between the blink maximum and the right zero value.
    """
    rightZeroHalfHeight = np.minimum(rightOuter, maxFrame +
                                     np.argmax(candidateSignal[rightHalfBase] <= blinkHalfHeight))

    return leftZeroHalfHeight, rightZeroHalfHeight, leftBaseHalfHeight, rightBaseHalfHeight

def get_left_range(leftZero, maxFrame, candidateSignal, blinkTop, blinkBottom):
    # YES
    blinkRange = np.arange(leftZero, maxFrame + 1, dtype=int)

    blinkTopPoint = np.where(candidateSignal[blinkRange] < blinkTop)[0][-1]#np.argmin(dd) ## return 7
    blinkBottomPoint = np.argmax(candidateSignal[blinkRange] > blinkBottom) # return 2


    leftRange = [blinkRange[blinkBottomPoint], blinkRange[blinkTopPoint]] #[42,48]
    blinkTopPoint_l_X = blinkRange[blinkTopPoint]
    blinkTopPoint_l_Y = candidateSignal[blinkTopPoint_l_X]
    blinkBottomPoint_l_X = blinkRange[blinkBottomPoint]
    blinkBottomPoint_l_Y = candidateSignal[blinkBottomPoint_l_X]

    return leftRange,blinkTopPoint_l_X,blinkTopPoint_l_Y,blinkBottomPoint_l_X,blinkBottomPoint_l_Y

def get_right_range(maxFrame, rightZero, candidateSignal, blinkTop, blinkBottom):
    # YES
    blinkRange = np.arange(maxFrame, rightZero + 1, dtype=int)
    bxbz=candidateSignal[blinkRange]
    axaz= bxbz< blinkTop
    blinkTopPoint_r = np.argmax(axaz)

    qqq=candidateSignal[blinkRange]
    wwwe=qqq > blinkBottom
    blinkBottomPoint_r=np.where(wwwe)[0][-1]
    # blinkBottomPoint_r = np.argmin(wwwe)

    blinkTopPoint_r_X = blinkRange[blinkTopPoint_r]
    blinkTopPoint_r_Y = candidateSignal[blinkTopPoint_r_X]
    blinkBottomPoint_r_X = blinkRange[blinkBottomPoint_r]
    blinkBottomPoint_r_Y = candidateSignal[blinkBottomPoint_r_X]

    rightRange = [blinkRange[blinkTopPoint_r], blinkRange[blinkBottomPoint_r]]
    return rightRange,blinkTopPoint_r_X,blinkTopPoint_r_Y,blinkBottomPoint_r_X,blinkBottomPoint_r_Y


def compute_fit_range(candidateSignal, maxFrame, leftZero, rightZero, baseFraction, top_bottom=None):
    # YES
    maxFrame, leftZero, rightZero = int(maxFrame), int(leftZero), int(rightZero)
    blinkHeight = candidateSignal[maxFrame] - candidateSignal[leftZero]  # ?? 8.8286028
    blinkTop = candidateSignal[maxFrame] - baseFraction * blinkHeight  # ?? 7.8665667
    blinkBottom = candidateSignal[leftZero] + baseFraction * blinkHeight  # ?? 0.80368418
    leftRange,blinkTopPoint_l_X,blinkTopPoint_l_Y,blinkBottomPoint_l_X,blinkBottomPoint_l_Y=get_left_range(leftZero, maxFrame, candidateSignal, blinkTop, blinkBottom)

    rightRange,blinkTopPoint_r_X,blinkTopPoint_r_Y,blinkBottomPoint_r_X,blinkBottomPoint_r_Y=get_right_range(maxFrame, rightZero, candidateSignal, blinkTop, blinkBottom)


    # use this to visualise
    # from eeg_blinks.viz.viz_sanity import viz_blink_top_buttom_point
    # viz_blink_top_buttom_point(candidateSignal,blinkRange,blinkTop,blinkBottom,maxFrame,rightZero,leftRange,rightRange)

    xLeft = np.arange(leftRange[0], leftRange[1] + 1, dtype=int)  # THe +1 to ensure we include the last frame
    xRight = np.arange(rightRange[0], rightRange[1] + 1, dtype=int)




    if blinkBottomPoint_l_X == blinkTopPoint_l_X:
        warnings.warn('same value for left top_blink and left bottom_blink')

    if blinkBottomPoint_r_X == blinkTopPoint_r_X:
        warnings.warn('same value for right top_blink and right bottom_blink')

    if xLeft.size == 0:
        xLeft = np.nan

    if xRight.size == 0:
        xRight = np.nan

    if top_bottom is None:
        warnings.warn('To modify this so that all function return the top_bottom point')
        return xLeft, xRight, leftRange, rightRange
    else:
        return xLeft, xRight, leftRange, rightRange, \
               blinkBottomPoint_l_Y,blinkBottomPoint_l_X,blinkTopPoint_l_Y,blinkTopPoint_l_X,\
               blinkBottomPoint_r_X,blinkBottomPoint_r_Y,blinkTopPoint_r_X,blinkTopPoint_r_Y


